#!/usr/bin/env python
"""Quantified restore verification for BALROG's Crafter, with and without the
``crafter_balance_chunk_sorted`` determinism patch.

Protocol (per arm):
  1. Play ONE base episode (seed 0) of ``--base-steps`` steps with a fixed,
     seeded action script through the *same* adapter the PAE loop uses, recording
     per-step observation / stats / state digests.  Snapshot the env at steps
     10, 50 and 150 with ``CrafterAdapter.env_snapshot()``.
  2. For each snapshot at step S: restore it, then replay the SAME actions
     A[S..S+49].  The restored trajectory must equal base steps S+1..S+50
     digest-for-digest.  The first index where it does not is the divergence.

Arm ``patched`` applies ``crafter_ckpt.apply_determinism_patch()``; arm
``unpatched`` does not.  Everything else is identical, so the difference between
the two arms *is* the justification for the patch.

Run one arm per process (the patch is a module-level monkeypatch):

    .venv-balrog/bin/python -m tools.balrog_pae.games.crafter.verify_restore \
        --arm patched --out /tmp/patched.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import random
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

CHECKPOINT_STEPS = (10, 50, 150)
FOLLOW_STEPS = 50


def fingerprint(x) -> str:
    """Stable content hash (PIL images by pixel bytes, not by id())."""
    h = hashlib.sha256()

    def feed(v):
        if v is None:
            h.update(b"None")
        elif isinstance(v, (str, bytes)):
            h.update(v if isinstance(v, bytes) else v.encode())
        elif isinstance(v, (bool, int, float, np.integer, np.floating)):
            h.update(repr(v).encode())
        elif isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode() + str(v.shape).encode() + np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                feed(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                feed(item)
        elif isinstance(v, (set, frozenset)):
            for item in sorted(v, key=repr):
                feed(item)
        elif hasattr(v, "tobytes") and hasattr(v, "size"):  # PIL.Image
            h.update(v.tobytes())
        else:
            h.update(repr(v).encode())

    feed(x)
    return h.hexdigest()[:16]


def balrog_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.load(str(importlib.resources.files("balrog") / "config" / "config.yaml"))


def observe(adapter, obs, reward, done) -> dict:
    """Everything we assert equality on for a single step."""
    stats = adapter.env.get_stats()
    return {
        "text": fingerprint(obs["text"]),
        "obs_full": fingerprint(obs),          # text + rendered image + raw array
        "state": adapter.state_digest(),        # canonical crafter.Env game state
        "score": float(stats["score"]),
        "progression": float(stats["progression"]),
        "achievements": fingerprint(sorted((stats["achievements"] or {}).items())),
        "reward": float(reward),
        "done": bool(done),
    }


def first_divergence(a: list[dict], b: list[dict]) -> dict | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {
                "index": i,
                "fields": sorted(k for k in x if x.get(k) != y.get(k)),
                "original": {k: x[k] for k in x if x.get(k) != y.get(k)},
                "restored": {k: y[k] for k in y if x.get(k) != y.get(k)},
            }
    return None


def divergence_by_field(a: list[dict], b: list[dict]) -> dict:
    """First follow-step index at which each compared field differs (None = never).

    ``state`` is the canonical world state; ``text`` is what the LLM actually
    reads.  A divergence that starts in ``state`` and only later reaches
    ``text`` is still a divergence: the two trajectories are different games.
    """
    out = {}
    for k in a[0] if a else []:
        out[k] = next((i for i, (x, y) in enumerate(zip(a, b)) if x.get(k) != y.get(k)), None)
    return out


def run_arm(arm: str, seed: int, base_steps: int, action_seed: int) -> dict:
    from tools.balrog_pae.adapters import CrafterAdapter

    cfg = balrog_cfg()

    if arm == "patched":
        adapter = CrafterAdapter("default", cfg)           # __init__ applies the patch
    else:
        import crafter

        stock = crafter.Env._balance_chunk
        adapter = CrafterAdapter("default", cfg)
        crafter.Env._balance_chunk = stock                 # undo it: this arm runs stock Crafter
        adapter._ck._PATCHED = False

    import crafter as _crafter

    # stock is ``Env._balance_chunk``; the patch installs
    # ``apply_determinism_patch.<locals>._balance_chunk``
    patch_active = _crafter.Env._balance_chunk.__qualname__ != "Env._balance_chunk"

    adapter.make()
    from balrog.environments.crafter import ACTIONS

    rng = random.Random(action_seed)
    actions = [rng.choice(ACTIONS) for _ in range(base_steps)]

    obs, _ = adapter.reset(seed)
    initial_state = adapter.state_digest()   # same in every process iff the seed is pinned
    base: list[dict] = [observe(adapter, obs, 0.0, False)]   # index 0 == env step 0
    snaps: dict[int, dict] = {}
    t0 = time.time()
    died_at = None
    for i, act in enumerate(actions):
        if i in CHECKPOINT_STEPS:
            snaps[i] = {"snap": adapter.env_snapshot(), "state": adapter.state_digest()}
        obs, reward, term, trunc, _ = adapter.step(act)
        base.append(observe(adapter, obs, reward, term or trunc))
        if term or trunc:
            died_at = i + 1
            break
    base_wall = time.time() - t0

    results = []
    for S in CHECKPOINT_STEPS:
        if S not in snaps or len(base) < S + 1 + FOLLOW_STEPS:
            results.append({"checkpoint_step": S, "status": "unavailable",
                            "reason": f"base episode only reached step {len(base) - 1}"})
            continue
        t = time.time()
        adapter.env_restore(snaps[S]["snap"])
        restore_ms = (time.time() - t) * 1e3
        state_after = adapter.state_digest()
        restored = []
        for act in actions[S:S + FOLLOW_STEPS]:
            obs, reward, term, trunc, _ = adapter.step(act)
            restored.append(observe(adapter, obs, reward, term or trunc))
        original = base[S + 1:S + 1 + FOLLOW_STEPS]
        div = first_divergence(original, restored)
        by_field = divergence_by_field(original, restored)
        results.append({
            "checkpoint_step": S,
            "status": "ok",
            "restore_ms": round(restore_ms, 3),
            "state_digest_at_snapshot": snaps[S]["state"],
            "state_digest_after_restore": state_after,
            "state_digest_identical_on_restore": state_after == snaps[S]["state"],
            "follow_steps": FOLLOW_STEPS,
            "identical_steps": FOLLOW_STEPS if div is None else div["index"],
            "all_identical": div is None,
            "first_divergence": div,
            "first_divergence_by_field": by_field,
        })

    adapter.close()
    return {
        "arm": arm,
        "determinism_patch": arm == "patched",
        "balance_chunk_is_patched": bool(patch_active),
        "seed": seed,
        "action_seed": action_seed,
        "env_patches": list(adapter.env_patches),
        "initial_state_digest": initial_state,
        "base_steps_requested": base_steps,
        "base_steps_played": len(base) - 1,
        "base_episode_ended_at": died_at,
        "base_wall_s": round(base_wall, 1),
        "final_score": base[-1]["score"],
        "final_progression": base[-1]["progression"],
        "checkpoints": results,
        "all_checkpoints_identical": all(r.get("all_identical") for r in results),
    }


def run_both(args) -> dict:
    """Run both arms, each in its own process (the patch is a monkeypatch), and
    merge them into the single artifact games/crafter/restore_verification.json."""
    import subprocess
    import tempfile

    def one(arm: str, td: str, tag: str) -> dict:
        out = os.path.join(td, f"{tag}.json")
        subprocess.run(
            [sys.executable, "-m", "tools.balrog_pae.games.crafter.verify_restore",
             "--arm", arm, "--seed", str(args.seed), "--action-seed", str(args.action_seed),
             "--base-steps", str(args.base_steps), "--out", out],
            check=True, capture_output=True, text=True,
        )
        return json.load(open(out))

    def offsets(res: dict) -> dict:
        return {str(c["checkpoint_step"]): (c.get("first_divergence") or {}).get("index")
                for c in res["checkpoints"]}

    arms = {}
    repeats = []
    with tempfile.TemporaryDirectory() as td:
        for arm in ("patched", "unpatched"):
            arms[arm] = one(arm, td, arm)
        # The unpatched failure mode depends on id()-based set ordering, so it is
        # not even reproducible across processes: repeat it to show the
        # divergence point moves. Each repeat is a fresh process.
        for r in range(args.unpatched_repeats):
            res = one("unpatched", td, f"un{r}")
            repeats.append({"repeat": r, "initial_state_digest": res["initial_state_digest"],
                            "all_identical": res["all_checkpoints_identical"],
                            "identical_steps": sum(c.get("identical_steps") or 0 for c in res["checkpoints"]),
                            "first_divergence_offsets": offsets(res)})
        for r in range(args.patched_repeats):
            res = one("patched", td, f"pa{r}")
            repeats.append({"repeat": r, "arm": "patched",
                            "initial_state_digest": res["initial_state_digest"],
                            "all_identical": res["all_checkpoints_identical"],
                            "identical_steps": sum(c.get("identical_steps") or 0 for c in res["checkpoints"]),
                            "first_divergence_offsets": offsets(res)})

    pa, un = arms["patched"], arms["unpatched"]
    return {
        "what": "Crafter checkpoint/restore fidelity, with and without the "
                "crafter_balance_chunk_sorted determinism patch",
        "protocol": (
            f"one base episode (seed {args.seed}) of {args.base_steps} scripted actions; "
            f"env_snapshot() at steps {list(CHECKPOINT_STEPS)}; each snapshot restored and "
            f"replayed with the SAME {FOLLOW_STEPS} actions; each restored step compared to the "
            "original continuation on observation text, full observation (text+rendered "
            "image+raw array), canonical world state, score, progression, achievements, reward, done"
        ),
        "same_world_in_both_arms": pa["initial_state_digest"] == un["initial_state_digest"],
        "initial_state_digest": pa["initial_state_digest"],
        "verdict": {
            "patched_all_identical": pa["all_checkpoints_identical"],
            "unpatched_all_identical": un["all_checkpoints_identical"],
            "patched_identical_steps": sum(c.get("identical_steps") or 0 for c in pa["checkpoints"]),
            "unpatched_identical_steps": sum(c.get("identical_steps") or 0 for c in un["checkpoints"]),
            "total_compared_steps": len(CHECKPOINT_STEPS) * FOLLOW_STEPS,
            "unpatched_first_divergence_offsets": {
                c["checkpoint_step"]: (c.get("first_divergence") or {}).get("index")
                for c in un["checkpoints"]
            },
        },
        "decision": (
            "the patch is applied in BOTH arms of the experiment (--base-only included) and "
            "disclosed in summary.json under env_patches"
        ),
        "repeats": {
            "why": "the unpatched failure depends on id()-based set iteration order, so the "
                   "divergence point moves from process to process; the patched arm must not",
            "unpatched": [r for r in repeats if r.get("arm") != "patched"],
            "patched": [r for r in repeats if r.get("arm") == "patched"],
            "patched_always_identical": all(r["all_identical"] for r in repeats if r.get("arm") == "patched")
                                        and pa["all_checkpoints_identical"],
            "unpatched_ever_identical": any(r["all_identical"] for r in repeats if r.get("arm") != "patched")
                                        or un["all_checkpoints_identical"],
        },
        "arms": arms,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["patched", "unpatched", "both"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-seed", type=int, default=26)  # survives 210 steps on the pinned seed-0 world
    p.add_argument("--base-steps", type=int, default=210)
    p.add_argument("--unpatched-repeats", type=int, default=4, help="extra unpatched processes (--arm both)")
    p.add_argument("--patched-repeats", type=int, default=4, help="extra patched processes (--arm both)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    res = run_both(a) if a.arm == "both" else run_arm(a.arm, a.seed, a.base_steps, a.action_seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
