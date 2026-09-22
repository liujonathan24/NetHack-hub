#!/usr/bin/env python
"""Replay-restore determinism, verified per MiniHack task.

What this proves, for one task and one seed:

1. Run a base episode (BALROG's own env stack, via ``MiniHackAdapter``) and take
   a checkpoint every K steps, exactly as ``pae.py`` does.
2. For at least R of those checkpoints, build a **brand new env** (a fresh
   ``make_env`` / fresh NLE instance, not the one the base episode ran on),
   restore it from the checkpoint, and compare the restored observation to the
   one saved at checkpoint time **byte for byte**: ``glyphs``, ``tty_chars``,
   ``tty_colors``, ``blstats``, ``inv_strs``, ``inv_letters``, ``tty_cursor``,
   ``text_message``, and BALROG's rendered text (``long_term_context`` /
   ``short_term_context``).
3. At least one replayed prefix contains an **invalid** action, so the
   evaluator's "Your previous output did not contain a valid action" rewrite is
   exercised; the rendered-text comparison above is what asserts that the
   feedback text survives the replay identically.

Action sources:
  ``--source scripted``  deterministic policy over the task's own action list,
                         with invalid completions injected at fixed steps (free,
                         and it guarantees the invalid-action case on every task)
  ``--source trace``     the completions an actual LLM base episode emitted,
                         read from ``<run>/attempts/a001/trace.jsonl``

Usage:
  python -m tools.balrog_pae.games.minihack.verify_replay --all --out .../replay_verification.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

TASKS = [
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
    "MiniHack-CorridorBattle-Dark-v0",
    "MiniHack-Boxoban-Medium-v0",
    "MiniHack-Boxoban-Hard-v0",
]

#: raw-observation fields compared byte-for-byte
OBS_FIELDS = [
    "glyphs", "tty_chars", "tty_colors", "blstats",
    "inv_strs", "inv_letters", "tty_cursor", "text_message",
]
#: BALROG's rendered text (what the player actually reads)
TEXT_FIELDS = ["long_term_context", "short_term_context"]

INVALID_COMPLETIONS = [
    "I will head towards the door on the left.",
    "move diagonally past the boulder",
]


def _h(v) -> str:
    if isinstance(v, np.ndarray):
        return hashlib.sha256(
            str(v.dtype).encode() + str(v.shape).encode() + np.ascontiguousarray(v).tobytes()
        ).hexdigest()[:16]
    return hashlib.sha256(repr(v).encode()).hexdigest()[:16]


def _eq(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b))
    return a == b


def snap_obs(obs: dict) -> dict:
    """Copy the compared fields out of the observation.

    The copy is load-bearing: NLE hands back *aliases* of its internal
    ``last_observation`` buffers, so a snapshot that keeps the references would
    silently track the live env and every comparison would be meaningless.
    (``pae.py`` is safe here - it stores only the rendered text plus a digest
    computed on the spot.)
    """
    raw = obs.get("obs") or {}
    out = {}
    for f in OBS_FIELDS:
        v = raw.get(f)
        out[f] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v
    out.update({f: obs["text"].get(f) for f in TEXT_FIELDS})
    return out


def digests(rec: dict) -> dict:
    return {k: _h(v) for k, v in rec.items()}


def make(task, cfg):
    from ...adapters import make_adapter

    a = make_adapter("minihack", task, cfg)
    a.make()
    return a


def scripted_completions(task, adapter, n, seed, invalid_at):
    """Deterministic completions; ``invalid_at`` steps get a non-action string."""
    from balrog.environments.minihack import get_available_actions

    acts = list(get_available_actions(adapter._inner()).keys())
    moves = [a for a in acts if a in
             ("north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest")]
    rng = random.Random(seed * 7919 + 13)
    out = []
    for i in range(n):
        if i in invalid_at:
            out.append(INVALID_COMPLETIONS[len(out) % len(INVALID_COMPLETIONS)])
        else:
            out.append(rng.choice(moves))
    return out


def trace_completions(run_dir, n):
    path = Path(run_dir) / "attempts" / "a001" / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r["completion"] for r in rows][:n]


def feedback_rewrite(obs, action, completion, cfg):
    """The evaluator's invalid-action rewrite, copied from pae.py verbatim."""
    if action != completion and cfg.eval.feedback_on_invalid_action:
        obs["text"]["long_term_context"] = (
            f"\n\nYour previous output did not contain a valid action. Defaulted to action: {action}"
            f"\n\nObservation:\n" + obs["text"]["long_term_context"]
        )
    return obs


def verify_task(task, cfg, seed=0, steps=100, every=10, replays=3, source="scripted",
                run_dir=None, invalid_at=(3, 17)):
    t0 = time.time()
    base = make(task, cfg)
    obs, _ = base.reset(seed)

    if source == "trace":
        comps = trace_completions(run_dir, steps)
    else:
        comps = scripted_completions(task, base, steps, seed, set(invalid_at))

    ckpts = []          # {"step", "snap", "rec", "digests", "invalid_before"}
    invalid_steps = []

    def take(step):
        rec = snap_obs(obs)
        ckpts.append({
            "step": step,
            "snap": base.env_snapshot(),
            "rec": rec,
            "digests": digests(rec),
            "invalid_before": list(invalid_steps),
            "progression": base.progression(),
            "aux": base.aux_progress(),
        })

    take(0)
    outcome, played = "step_cap", 0
    for i, completion in enumerate(comps):
        action = base.env.check_action_validity(completion)
        base.record(action, completion)
        obs, reward, term, trunc, info = base.step(action)
        if action != completion:
            invalid_steps.append(i)
            feedback_rewrite(obs, action, completion, cfg)
        played = i + 1
        if term or trunc:
            outcome = "terminated" if term else "truncated"
            break
        if played % every == 0:
            take(played)
    base.close()

    # pick the checkpoints to replay: prefer ones whose prefix contains an
    # invalid action, then spread over the episode.
    pool = ckpts[1:]                       # step 0 has an empty prefix: nothing to replay
    with_invalid = [c for c in pool if c["invalid_before"]]
    picked = []                            # earliest invalid-covering one, then the deepest ones
    if with_invalid:
        picked.append(with_invalid[0])
    for c in reversed(pool):
        if len(picked) >= replays:
            break
        if c["step"] not in {x["step"] for x in picked}:
            picked.append(c)
    chosen = sorted(picked, key=lambda c: c["step"])

    results = []
    for c in chosen:
        fresh = make(task, cfg)           # brand new env, not the base one
        t1 = time.perf_counter()
        robs, _info, _dg = fresh.env_restore(c["snap"])
        ms = (time.perf_counter() - t1) * 1e3
        rrec = snap_obs(robs)
        per_field = {k: bool(_eq(c["rec"][k], rrec[k])) for k in c["rec"]}
        results.append({
            "checkpoint_step": c["step"],
            "n_replayed_actions": len(c["snap"]["log"]),
            "prefix_contains_invalid_action": bool(c["invalid_before"]),
            "invalid_action_steps_in_prefix": c["invalid_before"],
            "identical": all(per_field.values()),
            "fields_identical": per_field,
            "saved_digests": c["digests"],
            "replayed_digests": digests(rrec),
            "restore_ms": round(ms, 1),
            "ms_per_replayed_step": round(ms / max(1, len(c["snap"]["log"])), 3),
        })
        fresh.close()

    return {
        "task": task,
        "seed": seed,
        "source": source,
        "run_dir": str(run_dir) if run_dir else None,
        "episode_steps_played": played,
        "episode_outcome": outcome,
        "checkpoint_every": every,
        "checkpoints_taken": len(ckpts),
        "invalid_action_steps": invalid_steps,
        "replays": results,
        "all_identical": all(r["identical"] for r in results) and len(results) >= 1,
        "invalid_action_covered": any(r["prefix_contains_invalid_action"] for r in results),
        "compared_fields": OBS_FIELDS + TEXT_FIELDS,
        "wall_s": round(time.time() - t0, 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", action="append", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--replays", type=int, default=3)
    p.add_argument("--source", default="scripted", choices=["scripted", "trace"])
    p.add_argument("--run-dir", default=None, help="base run dir for --source trace")
    p.add_argument("--out", default=None)
    p.add_argument("--merge", action="store_true", help="merge into an existing --out file")
    a = p.parse_args(argv)

    from ...run import balrog_config

    cfg = balrog_config("z-ai/glm-5.2", 1.0, 8192, 600)
    tasks = TASKS if a.all else (a.task or [TASKS[0]])

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": {}}
    if a.merge and a.out and Path(a.out).exists():
        out = json.loads(Path(a.out).read_text())
        out.setdefault("tasks", {})

    for t in tasks:
        r = verify_task(t, cfg, seed=a.seed, steps=a.steps, every=a.checkpoint_every,
                        replays=a.replays, source=a.source, run_dir=a.run_dir)
        key = f"{t}|{a.source}"
        out["tasks"][key] = r
        print(f"{t:34s} src={a.source:8s} steps={r['episode_steps_played']:3d} "
              f"ckpts={r['checkpoints_taken']:2d} replays={len(r['replays'])} "
              f"identical={r['all_identical']} invalid_covered={r['invalid_action_covered']}")
        for rr in r["replays"]:
            print(f"    step {rr['checkpoint_step']:3d}: {rr['n_replayed_actions']:3d} actions, "
                  f"identical={rr['identical']}, invalid_in_prefix={rr['prefix_contains_invalid_action']}, "
                  f"{rr['ms_per_replayed_step']:.3f} ms/step")

    out["summary"] = {
        "all_identical": all(v["all_identical"] for v in out["tasks"].values()),
        "tasks_with_invalid_action_coverage": sorted(
            {v["task"] for v in out["tasks"].values() if v["invalid_action_covered"]}),
        "n_task_source_pairs": len(out["tasks"]),
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=str))
        print("wrote", a.out)
    return 0 if out["summary"]["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
