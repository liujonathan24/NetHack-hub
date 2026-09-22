#!/usr/bin/env python
"""Restore determinism for BALROG's three TextWorld games.

Protocol (one run per task, base episode seed 0):

  * play a fixed 40-command prefix that contains an **unrecognised verb**
    (``xyzzy plugh`` -> "That's not a verb I recognise.") and an **unknown
    noun** (``take flurb`` -> "You can't see any such thing."), so the parser's
    error feedback is part of the state that has to come back identically;
  * checkpoint every 10 commands (steps 10, 20, 30, 40);
  * for every checkpoint: deliberately **mutate** the live env with 10 extra
    commands, then restore it and compare against a *reference* env that was
    freshly reset and replayed to the same step:
      - identical observation text, score, done flag, won, max_score at the
        restore point (via the reference trace record and, for the native .z8
        path, the Jericho state digest);
      - then play the **same 20 continuation commands** on the restored env and
        on the reference env and require every one of the 20 records to match.
    The continuation itself starts with an unrecognised command, so parser
    feedback parity is checked after the restore too.

Restore method is whatever ``TextWorldAdapter`` uses in production:
Jericho ``get_state``/``set_state`` for ``the_cooking_game`` (.z8) and
reset+replay for ``treasure_hunter`` / ``coin_collector`` (.ulx).

  python -m tools.balrog_pae.games.textworld.verify_restore \
      --out tools/balrog_pae/games/textworld/restore_verification.json
"""
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

TASKS = ["treasure_hunter", "the_cooking_game", "coin_collector"]

# A fixed, game-agnostic command script.  Most commands are legal TextWorld
# verbs; two are deliberately bad.  Index 3 = unrecognised verb, index 7 =
# unknown noun.
PREFIX = [
    "look", "go north", "inventory",
    "xyzzy plugh",                      # unrecognised verb
    "go east", "look", "open door",
    "take flurb",                       # unknown noun
    "go south", "inventory",
    "go west", "look", "take coin", "examine cookbook", "go north",
    "open fridge", "inventory", "go east", "look", "take knife",
    "go south", "examine table", "go west", "look", "inventory",
    "go north", "open door", "go east", "take apple", "look",
    "go south", "inventory", "go west", "examine door", "look",
    "go north", "take key", "go east", "inventory", "look",
]
# played on the live env between snapshot and restore, to prove the snapshot
# is not aliasing live state
MUTATE = ["go south", "go south", "take all", "drop all", "go west",
          "open door", "go north", "look", "inventory", "go east"]
# replayed identically on both arms after the restore
CONTINUE = [
    "frobnicate the widget",            # unrecognised verb, post-restore
    "look", "inventory", "go north", "go east", "open door", "take coin",
    "examine bed", "go south", "look", "go west", "take flurb",
    "inventory", "go north", "look", "open fridge", "go east", "go south",
    "look", "inventory",
]


def balrog_config():
    from omegaconf import OmegaConf

    return OmegaConf.load(str(importlib.resources.files("balrog") / "config" / "config.yaml"))


def record(obs, reward, term, trunc, info) -> dict:
    return {
        "text": obs["text"]["long_term_context"],
        "short": obs["text"].get("short_term_context", ""),
        "reward": float(reward),
        "done": bool(term or trunc),
        "score": info.get("score"),
        "max_score": info.get("max_score"),
        "won": info.get("won"),
        "description": info.get("description"),
    }


def diff(a: list[dict], b: list[dict], label: str) -> str | None:
    if len(a) != len(b):
        return f"{label}: length {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        for k in x:
            if x[k] != y[k]:
                return f"{label}: step {i} key {k!r}: {x[k]!r} != {y[k]!r}"
    return None


def play(ad, cmds, trace=None, stop_on_done=True):
    for c in cmds:
        ad.record(c, c)
        out = ad.step(c)
        r = record(*out)
        if trace is not None:
            trace.append(r)
        if stop_on_done and r["done"]:
            break
    return trace


def fresh_reference(cfg, task, seed, n_prefix):
    """A second adapter, reset and replayed to step ``n_prefix``."""
    from tools.balrog_pae.adapters import TextWorldAdapter

    ad = TextWorldAdapter(task, cfg.copy())
    ad.make()
    obs, info = ad.reset(seed)
    trace = [record(obs, 0.0, False, False, info)]
    play(ad, PREFIX[:n_prefix], trace)
    return ad, trace


def run_task(task, seed, every, n_prefix, n_continue, out):
    from tools.balrog_pae.adapters import TextWorldAdapter

    cfg = balrog_config()
    res = {"task": task, "seed": seed, "checkpoint_every": every,
           "prefix_len": n_prefix, "continue_len": n_continue,
           "invalid_commands_in_prefix": [PREFIX[3], PREFIX[7]],
           "invalid_command_in_continuation": CONTINUE[0],
           "checkpoints": [], "ok": False}

    live = TextWorldAdapter(task, cfg.copy())
    live.make()
    t0 = time.perf_counter()
    obs, info = live.reset(seed)
    res["reset_ms"] = (time.perf_counter() - t0) * 1e3
    res["gamefile"] = os.path.basename(live.gamefile())
    res["method"] = "jericho" if live._native() else "replay"
    _, objs = live._chain()
    res["chain"] = [type(o).__name__ for o in objs]
    res["max_steps"] = live.max_steps
    res["max_score"] = None

    # -- the base trajectory, checkpointing every `every` steps -------------
    base = [record(obs, 0.0, False, False, info)]
    snaps = {}
    feedback = {}
    snap_ms, snap_bytes = [], []
    for i, c in enumerate(PREFIX[:n_prefix], start=1):
        live.record(c, c)
        r = record(*live.step(c))
        base.append(r)
        if c in (PREFIX[3], PREFIX[7]):
            feedback[c] = r["text"]
        if r["done"]:
            break
        if i % every == 0:
            t0 = time.perf_counter()
            s = live.env_snapshot()
            snap_ms.append((time.perf_counter() - t0) * 1e3)
            import pickle

            snap_bytes.append(len(pickle.dumps(s, protocol=4)))
            snaps[i] = s
    res["max_score"] = base[-1]["max_score"]
    res["parser_feedback"] = feedback
    res["snapshot_ms"] = sum(snap_ms) / max(len(snap_ms), 1)
    res["snapshot_bytes"] = max(snap_bytes) if snap_bytes else 0
    res["base_done_at"] = next((i for i, r in enumerate(base) if r["done"]), None)

    all_ok = True
    for step in sorted(snaps):
        snap = snaps[step]
        row = {"step": step, "method": snap["kind"]}
        try:
            # reference arm: a *fresh* env replayed to the same step
            ref, ref_trace = fresh_reference(cfg, task, seed, step)
            row["prefix_identical"] = diff(base[: step + 1], ref_trace, f"{task}@{step} prefix") is None
            row["prefix_error"] = diff(base[: step + 1], ref_trace, f"{task}@{step} prefix")

            # mutate the live env so the restore has something to undo
            play(live, MUTATE)
            live_after_mutation = live.summary()

            t0 = time.perf_counter()
            robs, _, digests = live.env_restore(snap)
            row["restore_ms"] = (time.perf_counter() - t0) * 1e3
            row["live_state_before_restore"] = live_after_mutation

            # restore-point identity
            if snap["kind"] == "jericho":
                row["restored_digest"] = live.state_digest()
                row["reference_digest"] = ref.state_digest()
                row["restore_point_identical"] = row["restored_digest"] == row["reference_digest"]
                row["restore_point_check"] = "jericho state digest (restored vs freshly replayed)"
            else:
                got = record(robs, 0.0, base[step]["done"], False, {})
                row["restore_point_identical"] = robs["text"]["long_term_context"] == base[step]["text"]
                row["restore_point_check"] = "replayed observation text vs the recorded one"
                row["replayed_steps"] = len(digests) - 1
            # observation text / score / done at the restore point, both arms
            row["restore_point_fields"] = {
                "text_equal": base[step]["text"] == ref_trace[step]["text"],
                "score_equal": base[step]["score"] == ref_trace[step]["score"],
                "done_equal": base[step]["done"] == ref_trace[step]["done"],
                "won_equal": base[step]["won"] == ref_trace[step]["won"],
                "score": base[step]["score"],
                "max_score": base[step]["max_score"],
                "done": base[step]["done"],
            }

            # 20 identical continuation commands on both arms
            ta, tb = [], []
            play(live, CONTINUE[:n_continue], ta)
            play(ref, CONTINUE[:n_continue], tb)
            err = diff(ta, tb, f"{task}@{step} continuation")
            row["continuation_len"] = len(ta)
            row["continuation_identical"] = err is None
            row["continuation_error"] = err
            row["continuation_feedback_first"] = ta[0]["text"] if ta else None
            row["continuation_feedback_identical"] = bool(ta) and ta[0]["text"] == tb[0]["text"]
            ref.close()

            row["ok"] = bool(
                row["restore_point_identical"]
                and row["continuation_identical"]
                and row["continuation_feedback_identical"]
                and all(v for k, v in row["restore_point_fields"].items() if k.endswith("_equal"))
            )
            # put the live env back on the snapshot for the next round
            live.env_restore(snap)
        except Exception as e:  # noqa: BLE001
            row["ok"] = False
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc(limit=4)
        all_ok = all_ok and row["ok"]
        res["checkpoints"].append(row)
        print(f"  [{task}] ckpt@{step:>3} {'OK ' if row['ok'] else 'FAIL'} "
              f"{row.get('continuation_error') or ''}")

    res["ok"] = all_ok and len(res["checkpoints"]) >= 3
    res["n_checkpoints_verified"] = len(res["checkpoints"])
    live.close()
    out["tasks"][task] = res
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--prefix", type=int, default=40)
    ap.add_argument("--continue-len", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "restore_verification.json"))
    a = ap.parse_args(argv)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": a.seed,
           "checkpoint_every": a.every, "prefix_len": a.prefix,
           "continuation_len": a.continue_len, "tasks": {}}
    for task in a.tasks.split(","):
        print(f"== {task} ==")
        try:
            run_task(task, a.seed, a.every, a.prefix, a.continue_len, out)
        except Exception as e:  # noqa: BLE001
            out["tasks"][task] = {"task": task, "ok": False, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
    out["all_ok"] = all(t.get("ok") for t in out["tasks"].values())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: {"ok": v.get("ok"), "method": v.get("method"),
                          "n": v.get("n_checkpoints_verified"),
                          "gamefile": v.get("gamefile"),
                          "max_score": v.get("max_score")}
                      for k, v in out["tasks"].items()}, indent=1))
    print("all_ok:", out["all_ok"], "->", a.out)
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
