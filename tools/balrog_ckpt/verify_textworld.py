#!/usr/bin/env python
"""Verify snapshot/restore for BALROG's TextWorld tasks.

BALROG's stack: EnvWrapper -> GymV21CompatibilityV0 -> TextWorldWrapper ->
textworld.gym TextworldGymEnv -> SyncBatchEnv -> Filter -> Limit ->
GenericEnvironment -> TWInform7 -> StateTracking -> Inform7Data -> GameData ->
  * GitGlulxEnv  (.ulx games: treasure_hunter, coin_collector) -- a git-glulx-ml
    subprocess talking over a UNIX socket; Environment.copy() is
    NotImplementedError, no state export. In-game "save" kills the interpreter
    (it prompts for a filename), "undo" is single-level only.
  * JerichoEnv   (.z8 games: the_cooking_game) -- jericho.FrotzEnv, which has
    get_state()/set_state()/copy().

What is verified here, per task:
  1. replay determinism: 3 x [reset(seed) -> prefix -> 50 fixed commands]
     produce identical BALROG observations / rewards / infos.
  2. native in-place snapshot (z8 only): jericho get_state() + a deepcopy of the
     mutable attributes of every Python wrapper in the chain; restore =
     set_state() + attribute swap. Identical-branches + no-alias check.
     For .ulx we record that no native mechanism exists (and show the probe
     results for env.copy()).
  3. timings: reset, per-step, snapshot, restore, replay cost per step.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Timer, banner, compare_traces, fingerprint, fixed_actions, load_balrog_config, write_result  # noqa: E402

COMMANDS = [
    "look", "inventory", "go north", "go south", "go east", "go west",
    "open door", "open chest", "take key", "take coin", "examine chest",
    "unlock chest with key", "open fridge", "take knife", "cook egg", "help",
]


def chain(env):
    """Return the list of textworld Environment/Wrapper objects, outermost first."""
    gymenv = env.env.gym_env.env  # TextworldGymEnv
    e = gymenv.batch_env.envs[0]
    out = []
    while e is not None:
        out.append(e)
        e = getattr(e, "_wrapped_env", None)
    return gymenv, out


# attributes that are immutable / process handles: never copied
_SKIP = {"_wrapped_env", "_game", "_inform7", "_jericho", "_gamefile", "gamefile", "request_infos",
         "_tracked_infos", "_process", "_names_struct", "_last_backend", "max_episode_steps", "_seed"}


def snapshot_native(env):
    gymenv, objs = chain(env)
    base = objs[-1]
    if not hasattr(base, "_jericho"):
        raise NotImplementedError(f"{type(base).__name__} has no state export (Environment.copy() -> NotImplementedError)")
    wrappers = [{k: v for k, v in vars(o).items() if k not in _SKIP} for o in objs]
    return {
        "jericho": base._jericho.get_state(),
        "wrappers": copy.deepcopy(wrappers),
        # SyncBatchEnv latches the last (obs, score, done, infos) and short-circuits step() once done
        "gym": copy.deepcopy({"last_commands": gymenv.last_commands, "obs": gymenv.obs, "batch_last": gymenv.batch_env.last}),
        "tw": {"progression": env.env.gym_env.progression},
        "wrapper": {"failed_candidates": list(env.failed_candidates)},
    }


def restore_native(env, snap):
    gymenv, objs = chain(env)
    base = objs[-1]
    base._jericho.set_state(snap["jericho"])
    for o, saved in zip(objs, copy.deepcopy(snap["wrappers"])):
        for k, v in saved.items():
            setattr(o, k, v)
    g = copy.deepcopy(snap["gym"])
    gymenv.last_commands = g["last_commands"]
    gymenv.obs = g["obs"]
    gymenv.batch_env.last = g["batch_last"]
    env.env.gym_env.progression = snap["tw"]["progression"]
    env.failed_candidates = list(snap["wrapper"]["failed_candidates"])


def obs_record(obs, reward, term, trunc, info) -> dict:
    return {
        "text": obs["text"]["long_term_context"],
        "reward": float(reward),
        "done": bool(term or trunc),
        "score": info.get("score"),
        "won": info.get("won"),
        "max_score": info.get("max_score"),
        "description": info.get("description"),
    }


def play(env, actions, timer=None):
    trace = []
    for a in actions:
        if timer is not None:
            with timer:
                out = env.step(a)
        else:
            out = env.step(a)
        trace.append(obs_record(*out))
        if trace[-1]["done"]:
            break
    return trace


def run_task(task, cfg, seed, n_prefix, n_branch, n_replays):
    from balrog.environments import make_env

    banner(f"TextWorld task={task} seed={seed} prefix={n_prefix} branch={n_branch}")
    res = {"task": task, "seed": seed}
    prefix = fixed_actions(COMMANDS, n_prefix, seed=seed * 7919)
    branch = fixed_actions(COMMANDS, n_branch, seed=seed * 104729)
    mutate = fixed_actions(COMMANDS, 10, seed=seed * 15485863)

    env = make_env("textworld", task, cfg)

    # ---- replay determinism ------------------------------------------------
    traces, step_ms, reset_ms = [], [], []
    for k in range(n_replays):
        t = time.perf_counter()
        obs, info = env.reset(seed=seed)
        reset_ms.append((time.perf_counter() - t) * 1e3)
        if k == 0:
            _, objs = chain(env)  # the game backend only exists after reset()
            res["backend"] = type(objs[-1]).__name__
            res["chain"] = [type(o).__name__ for o in objs]
            print("chain:", " > ".join(res["chain"]))
            res["observation_keys"] = list(obs.keys())
            res["gamefile"] = objs[-1].gamefile if hasattr(objs[-1], "gamefile") else getattr(objs[-1], "_gamefile", None)
            print("obs keys:", res["observation_keys"], "| gamefile:", os.path.basename(str(res["gamefile"])))
        st = Timer()
        first = obs_record(obs, 0.0, False, False, info)
        p = play(env, prefix, st)
        b = play(env, branch, st) if not (p and p[-1]["done"]) else []
        traces.append(([first] + p, b))
        step_ms.append(st.mean_ms)
    ok = True
    for k in range(1, n_replays):
        ok = compare_traces(traces[0][0], traces[k][0], f"{task} replay{k} prefix") and ok
        ok = compare_traces(traces[0][1], traces[k][1], f"{task} replay{k} branch") and ok
    res.update(replay_identical=ok, n_replays=n_replays, prefix_len=len(traces[0][0]) - 1, branch_len=len(traces[0][1]),
               reset_ms=float(np.mean(reset_ms)), step_ms=float(np.mean(step_ms)),
               restore_by_replay_ms=float(np.mean(reset_ms)) + float(np.mean(step_ms)) * (len(traces[0][0]) - 1))
    print(f"replay identical={ok} reset={np.mean(reset_ms):.1f}ms step={np.mean(step_ms):.2f}ms")

    # ---- probe: does the textworld chain expose copy()? ---------------------
    try:
        objs[0].copy()
        res["chain_copy"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["chain_copy"] = f"{type(e).__name__}: {e}"
    try:
        objs[-1].copy()
        res["base_copy"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["base_copy"] = f"{type(e).__name__}: {e}"
    print("copy() probes: chain ->", res["chain_copy"], "| base ->", res["base_copy"])

    # ---- native in-place snapshot (jericho only) ----------------------------
    env.reset(seed=seed)
    play(env, prefix)
    try:
        snap_t, rest_t = Timer(), Timer()
        with snap_t:
            snap = snapshot_native(env)
        fp0 = fingerprint(pickle.dumps(snap))
        with rest_t:
            restore_native(env, snap)
        b1 = play(env, branch)
        play(env, mutate)
        fp1 = fingerprint(pickle.dumps(snap))
        with rest_t:
            restore_native(env, snap)
        b2 = play(env, branch)
        play(env, mutate)
        with rest_t:
            restore_native(env, snap)
        b3 = play(env, branch)
        ident = compare_traces(b1, b2, f"{task} native b1-vs-b2") and compare_traces(b2, b3, f"{task} native b2-vs-b3")
        # the native branch must also equal the replay-based branch from the same prefix
        same_as_replay = compare_traces(traces[0][1], b1, f"{task} native-vs-replay branch")
        res["native"] = {
            "ok": bool(ident and fp0 == fp1 and same_as_replay),
            "identical_branches": ident,
            "saved_state_unchanged_after_branch_mutation": fp0 == fp1,
            "matches_replay_branch": same_as_replay,
            "snapshot_ms": snap_t.mean_ms,
            "restore_ms": rest_t.mean_ms,
            "snapshot_bytes": len(pickle.dumps(snap)),
        }
        print(f"native: identical={ident} no_alias={fp0 == fp1} snapshot={snap_t.mean_ms:.2f}ms restore={rest_t.mean_ms:.2f}ms")
    except NotImplementedError as e:
        res["native"] = {"ok": False, "error": f"NotImplementedError: {e}"}
        print("native snapshot: not available ->", e)
    except Exception as e:  # noqa: BLE001
        res["native"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print("native snapshot FAILED:", res["native"]["error"])
        traceback.print_exc(limit=3)
    env.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="treasure_hunter,the_cooking_game,coin_collector")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--prefix", type=int, default=20)
    ap.add_argument("--branch", type=int, default=50)
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "textworld.json"))
    args = ap.parse_args()
    cfg = load_balrog_config()
    # BALROG registers games under <balrog pkg parent>/tw_games; make sure they are there
    results = {"env": "textworld", "tasks": {}}
    for task in args.tasks.split(","):
        try:
            results["tasks"][task] = run_task(task, cfg, args.seed, args.prefix, args.branch, args.replays)
        except Exception as e:  # noqa: BLE001
            results["tasks"][task] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
    write_result(args.out, results)
    return 0 if all(t.get("replay_identical") for t in results["tasks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
