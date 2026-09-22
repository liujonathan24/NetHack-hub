#!/usr/bin/env python
"""Verify snapshot/restore for BALROG's Crafter env (task "default").

Creates the env exactly like balrog.evaluator (make_env + reset(seed=...)),
plays K prefix actions, snapshots, then:
  (a) restores 3x and plays the same 50 fixed actions, asserting identical
      observation/reward/done/info traces (the live env is mutated between
      restores);
  (b) checks the snapshot is not aliased to the live env: the saved blob is
      byte-identical after the branch mutated the live env, and the inner env
      state right after a restore equals the state at snapshot time.

Approaches (see crafter_ckpt.py):
  A0. copy.deepcopy(env) of the whole BALROG stack  -> expected to FAIL (gym's
      `_np_random` Generator cannot be deep-copied under numpy 2)
  A.  deepcopy of the inner crafter.Env with `_np_random` stripped
  B.  pickle of the inner crafter.Env, restored in place (__dict__ swap)
  C.  explicit copy of env._world (objects+player+RandomState), counters
Each approach is run with and without the Crafter determinism patch
(crafter_ckpt.apply_determinism_patch): without it, Crafter's set-ordered
despawn choice makes restored copies diverge.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Timer, banner, compare_traces, fingerprint, fixed_actions, load_balrog_config, write_result  # noqa: E402
import crafter_ckpt as C  # noqa: E402


def obs_record(obs, reward, term, trunc, info) -> dict:
    return {
        "text": obs["text"]["long_term_context"] + "\n" + obs["text"]["short_term_context"],
        "image": fingerprint(np.asarray(obs["image"])),
        "obs": fingerprint(np.asarray(obs["obs"])),
        "reward": float(reward),
        "done": bool(term or trunc),
        "inventory": fingerprint(info.get("inventory")),
        "achievements": fingerprint(info.get("achievements")),
        "player_pos": fingerprint(np.asarray(info.get("player_pos"))),
        "semantic": fingerprint(info.get("semantic")),
    }


def play(env, actions, step_timer=None) -> list:
    trace = []
    for a in actions:
        if step_timer is not None:
            with step_timer:
                out = env.step(a)
        else:
            out = env.step(a)
        trace.append(obs_record(*out))
        if trace[-1]["done"]:
            break
    return trace


def core_fp(env) -> str:
    _, core = C.inner_crafter(env)
    return C.state_digest(core)


def run_approach(name, env, snap_fn, restore_fn, branch_actions, mutate_actions):
    res = {"approach": name}
    try:
        snap_t, rest_t, step_t = Timer(), Timer(), Timer()
        prefix_state_fp = core_fp(env)
        with snap_t:
            snap = snap_fn(env)
        blob_fp = lambda s: fingerprint(s) if isinstance(s, bytes) else fingerprint(pickle.dumps(s))  # noqa: E731
        saved_fp = blob_fp(snap)

        with rest_t:
            restore_fn(env, snap)
        b1 = play(env, branch_actions, step_t)
        play(env, mutate_actions)  # keep mutating the live env
        saved_fp_after = blob_fp(snap)

        with rest_t:
            restore_fn(env, snap)
        fp_after_restore = core_fp(env)
        b2 = play(env, branch_actions, step_t)
        play(env, mutate_actions)

        with rest_t:
            restore_fn(env, snap)
        b3 = play(env, branch_actions, step_t)

        res["identical_branches"] = compare_traces(b1, b2, f"{name} b1-vs-b2") and compare_traces(b2, b3, f"{name} b2-vs-b3")
        res["saved_state_unchanged_after_branch_mutation"] = saved_fp == saved_fp_after
        res["restore_reproduces_snapshot_state"] = fp_after_restore == prefix_state_fp
        res["no_alias"] = res["saved_state_unchanged_after_branch_mutation"] and res["restore_reproduces_snapshot_state"]
        res["snapshot_ms"] = snap_t.mean_ms
        res["restore_ms"] = rest_t.mean_ms
        res["step_ms"] = step_t.mean_ms
        res["snapshot_bytes"] = len(snap) if isinstance(snap, bytes) else len(pickle.dumps(snap))
        res["branch_len"] = len(b1)
        res["ok"] = bool(res["identical_branches"] and res["no_alias"])
        print(f"[{name}] identical={res['identical_branches']} no_alias={res['no_alias']} "
              f"(blob_unchanged={res['saved_state_unchanged_after_branch_mutation']} restore_reproduces={res['restore_reproduces_snapshot_state']}) "
              f"snap={res['snapshot_ms']:.2f}ms restore={res['restore_ms']:.2f}ms step={res['step_ms']:.2f}ms bytes={res['snapshot_bytes']}")
        # leave the env at the snapshot state for the next approach
        restore_fn(env, snap)
    except Exception as e:  # noqa: BLE001
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"
        print(f"[{name}] FAILED: {res['error']}")
        traceback.print_exc(limit=2)
    return res


def make_prefixed_env(cfg, seed, prefix_n):
    from balrog.environments import make_env
    from balrog.environments.crafter.env import ACTIONS

    env = make_env("crafter", "default", cfg)
    obs, info = env.reset(seed=seed)
    prefix_actions = fixed_actions(ACTIONS, prefix_n, seed=seed * 7919)
    play(env, prefix_actions)
    return env, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--prefix", type=int, default=30)
    ap.add_argument("--branch", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "crafter.json"))
    args = ap.parse_args()

    from balrog.environments.crafter.env import ACTIONS

    cfg = load_balrog_config()
    seeds = [int(x) for x in args.seeds.split(",")]
    banner(f"Crafter / task=default seeds={seeds} prefix={args.prefix} branch={args.branch}")
    results = {"env": "crafter", "task": "default", "seeds": seeds, "runs": {"unpatched": {}, "with_determinism_patch": {}}}

    # ---- A0: deepcopy of the whole BALROG stack (expected failure, recorded verbatim)
    env, obs = make_prefixed_env(cfg, seeds[0], args.prefix)
    print("obs keys:", list(obs.keys()), "| text keys:", list(obs["text"].keys()))
    try:
        copy.deepcopy(env)
        results["A0_deepcopy_full_stack"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        results["A0_deepcopy_full_stack"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print("[A0_deepcopy_full_stack] FAILED:", results["A0_deepcopy_full_stack"]["error"])
    env.close()

    # ---- unpatched vs patched, several seeds (the unpatched divergence is probabilistic)
    for patched in (False, True):
        key = "with_determinism_patch" if patched else "unpatched"
        if patched:
            C.apply_determinism_patch()
        for seed in seeds:
            banner(f"Crafter approaches, {key}, seed={seed}")
            branch_actions = fixed_actions(ACTIONS, args.branch, seed=seed * 104729)
            mutate_actions = fixed_actions(ACTIONS, 20, seed=seed * 15485863)
            env, _ = make_prefixed_env(cfg, seed, args.prefix)
            _, core = C.inner_crafter(env)
            print(f"prefix done: step={core._step} pos={core._player.pos.tolist()} health={core._player.health}")
            runs = []
            for name, (sf, rf) in C.APPROACHES.items():
                runs.append(run_approach(name, env, sf, rf, branch_actions, mutate_actions))
            results["runs"][key][seed] = runs
            env.close()

    summary = {}
    for key, per_seed in results["runs"].items():
        summary[key] = {}
        for name in C.APPROACHES:
            oks = [r["ok"] for runs in per_seed.values() for r in runs if r["approach"] == name]
            summary[key][name] = f"{sum(oks)}/{len(oks)} seeds ok"
    results["summary"] = summary
    results["working_approaches_with_patch"] = [
        n for n in C.APPROACHES if all(r["ok"] for runs in results["runs"]["with_determinism_patch"].values() for r in runs if r["approach"] == n)
    ]
    banner("Crafter summary: " + str(summary))
    write_result(args.out, results)
    return 0 if results["working_approaches_with_patch"] else 1


if __name__ == "__main__":
    sys.exit(main())
