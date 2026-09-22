#!/usr/bin/env python
"""MiniHack path (2): replay-from-seed determinism on balrog-nle's MiniHack.

balrog-nle exposes only get_seeds()/set_seeds() (no state snapshot), so the
only "restore" is: reset with the same (core, disp, reseed) seeds and replay
the action prefix. This script verifies, through BALROG's own make_env stack
(EnvWrapper -> GymV21CompatibilityV0 -> NLETimeLimit -> NLELanguageWrapper ->
AutoMore -> MiniHack), that

  same seeds + same actions => identical BALROG observations (3 replays),

and that a Go-Explore branch (replay prefix, then 50 fixed actions) matches
the original continuation. It also records BALROG's MiniHack observation keys
for a later parity check against the fork engine, and the replay cost/step.

NOTE on seeding: BALROG's evaluator calls env.reset(seed=s), which reaches
NLE.seed(core=s, disp=None) -> disp is drawn from random.SystemRandom, i.e.
BALROG episodes are NOT bit-reproducible from `s` alone. We therefore read the
effective seeds with get_seeds() after the first reset and re-apply them with
seed(core, disp, reseed) on every replay (this is what a Go-Explore layer must
do too).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Timer, banner, compare_traces, fingerprint, fixed_actions, load_balrog_config, write_result  # noqa: E402


def nle_env(env):
    """Reach the MiniHack (NLE) env under BALROG's wrapper stack."""
    return env.env.gym_env.unwrapped


def balrog_lang_wrapper(env):
    """BALROG's NLELanguageWrapper subclass (balrog/environments/nle/base.py)."""
    e = env.env.gym_env
    # gym.Wrapper.__getattr__ forwards reads (so hasattr() lies); look for the owner
    while e is not None and not ("done" in vars(e) and "progress" in vars(e)):
        e = vars(e).get("env")
    return e


def full_reset(env, seed=None):
    """env.reset(...) plus clearing BALROG's `done` latch.

    balrog.environments.nle.base.NLELanguageWrapper.step() does
    `self.done = done if not self.done else self.done` and reset() never clears
    it, so once an episode ended every later step reports done=True. BALROG's
    evaluator never notices (fresh env per episode); a replay/Go-Explore layer
    reusing the env must clear it.
    """
    out = env.reset(seed=seed) if seed is not None else env.reset()
    lw = balrog_lang_wrapper(env)
    if lw is not None:
        lw.done = False
    return out


def obs_record(obs, reward, term, trunc, info) -> dict:
    rec = {}
    for k, v in obs.items():
        rec[k] = fingerprint(v) if isinstance(v, np.ndarray) else v
    rec["reward"] = float(reward)
    rec["done"] = bool(term or trunc)
    rec["end_status"] = str(info.get("end_status"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="MiniHack-Quest-Easy-v0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prefix", type=int, default=30)
    ap.add_argument("--branch", type=int, default=50)
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--actions", default=None, help="comma-separated BALROG action names to sample from (default: all movement-ish)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                   f"minihack_replay_{args.task}{'_safe' if args.actions else ''}.json")

    from balrog.environments import make_env

    cfg = load_balrog_config()
    banner(f"MiniHack replay determinism (balrog-nle): task={args.task} seed={args.seed} prefix={args.prefix} branch={args.branch}")
    result = {"env": "minihack", "task": args.task, "seed": args.seed}

    env = make_env("minihack", args.task, cfg)
    inner = nle_env(env)
    print("inner env:", type(inner).__name__, "| pynethack API:",
          [n for n in dir(inner.nethack._pynethack) if not n.startswith("_")])
    lang_actions = list(env.env.gym_env.env.action_str_enum_map.keys()) if hasattr(env.env.gym_env, "env") else None
    # BALROG's language action space for this task
    from balrog.environments.minihack import get_available_actions

    avail = list(get_available_actions(inner).keys())
    print("available BALROG actions:", avail)
    # movement-heavy fixed action set (as the naive agent would emit)
    wanted = args.actions.split(",") if args.actions else ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "search", "wait", "kick", "pickup"]
    moves = [a for a in avail if a in wanted]
    prefix = fixed_actions(moves, args.prefix, seed=args.seed * 7919)
    branch = fixed_actions(moves, args.branch, seed=args.seed * 104729)

    # --- first run, BALROG-style seeding -----------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    t = time.perf_counter()
    obs0, info0 = env.reset(seed=args.seed)
    reset_ms = (time.perf_counter() - t) * 1e3
    seeds = inner.get_seeds()
    print("effective seeds after reset(seed=%d): core=%d disp=%d reseed=%s" % (args.seed, seeds[0], seeds[1], seeds[2]))
    result["observation_keys"] = list(obs0.keys())
    result["observation_keys_nested"] = {
        k: (sorted(v.keys()) if isinstance(v, dict) else type(v).__name__) for k, v in obs0.items()
    }
    print("BALROG MiniHack nested observation keys:", result["observation_keys_nested"])
    result["effective_seeds"] = list(map(int, seeds[:2])) + [bool(seeds[2])]
    print("BALROG MiniHack observation keys:", result["observation_keys"])
    print("first long_term_context:\n" + str(obs0.get("text", {}).get("long_term_context", ""))[:400])

    def run_episode(reseed_from_effective: bool):
        """reset + prefix + branch; returns (prefix_trace, branch_trace, per-step ms)."""
        random.seed(args.seed)
        np.random.seed(args.seed)
        if reseed_from_effective:
            inner.seed(int(seeds[0]), int(seeds[1]), bool(seeds[2]))
            o, i = full_reset(env)
        else:
            o, i = full_reset(env, seed=args.seed)
        first = obs_record(o, 0.0, False, False, {})
        st = Timer()
        ptrace, btrace = [first], []
        done = False
        for a in prefix:
            with st:
                r = env.step(a)
            ptrace.append(obs_record(*r))
            if ptrace[-1]["done"]:
                done = True
                break
        for a in branch if not done else []:
            with st:
                r = env.step(a)
            btrace.append(obs_record(*r))
            if btrace[-1]["done"]:
                break
        return ptrace, btrace, st

    # run 0 = continuation of the reset above (BALROG-style seeding)
    st0 = Timer()
    p0 = [obs_record(obs0, 0.0, False, False, {})]
    done0 = False
    for a in prefix:
        with st0:
            r = env.step(a)
        p0.append(obs_record(*r))
        if p0[-1]["done"]:
            done0 = True
            break
    b0 = []
    for a in branch if not done0 else []:
        with st0:
            r = env.step(a)
        b0.append(obs_record(*r))
        if b0[-1]["done"]:
            break
    print(f"run0: prefix steps={len(p0)-1} (episode ended in prefix: {done0}), branch steps={len(b0)}")

    # replays with the effective seeds re-applied
    all_ok = True
    step_ms = [st0.mean_ms]
    for k in range(args.replays):
        t = time.perf_counter()
        pk, bk, st = run_episode(reseed_from_effective=True)
        total = (time.perf_counter() - t) * 1e3
        okp = compare_traces(p0, pk, f"replay{k} prefix")
        okb = compare_traces(b0, bk, f"replay{k} branch")
        all_ok = all_ok and okp and okb
        step_ms.append(st.mean_ms)
        print(f"  replay {k}: prefix+branch wall {total:.1f} ms, {st.mean_ms:.2f} ms/step")

    # what happens with BALROG's own reset(seed=) (random disp)?
    p_bal, b_bal, _ = run_episode(reseed_from_effective=False)
    balrog_seed_only_identical = compare_traces(p0, p_bal, "reset(seed) only, prefix") and compare_traces(b0, b_bal, "reset(seed) only, branch")
    print("seeds after reset(seed=) again:", inner.get_seeds())

    # replay cost to reach the prefix state (the "restore" cost of this path)
    t = time.perf_counter()
    random.seed(args.seed); np.random.seed(args.seed)
    inner.seed(int(seeds[0]), int(seeds[1]), bool(seeds[2]))
    full_reset(env)
    for a in prefix[: len(p0) - 1]:
        env.step(a)
    restore_by_replay_ms = (time.perf_counter() - t) * 1e3

    result.update(
        ok=bool(all_ok),
        identical_replays=all_ok,
        n_replays=args.replays,
        prefix_len=len(p0) - 1,
        branch_len=len(b0),
        episode_ended_in_prefix=bool(done0),
        episode_ended_in_branch=bool(b0 and b0[-1]["done"]),
        reset_ms=reset_ms,
        step_ms_mean=float(np.mean(step_ms)),
        restore_by_replay_ms=restore_by_replay_ms,
        restore_by_replay_ms_per_step=restore_by_replay_ms / max(1, len(p0) - 1),
        balrog_reset_seed_only_identical=bool(balrog_seed_only_identical),
        actions_used=moves,
    )
    banner(f"MiniHack replay identical={all_ok}  step={np.mean(step_ms):.2f}ms  restore-by-replay({len(p0)-1} steps)={restore_by_replay_ms:.1f}ms  "
           f"reset(seed)-only reproducible={balrog_seed_only_identical}")
    env.close()
    write_result(out, result)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
