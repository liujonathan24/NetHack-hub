#!/usr/bin/env python
"""`TextWorldAdapter.progression()` is live, and equals BALROG's own at `done`.

`TextWorldWrapper` computes `max(score/max_score, 1.0 if won else 0.0)` only
inside `if done:`, so `get_stats()["progression"]` is 0.0 for the whole episode.
The adapter evaluates the same formula on the same numbers at every step so
`pae.py`'s checkpoint trigger and plateau guard can see a score gain.

This replays a *winning* command sequence for each game (`winning_traces.json`,
lifted from the seed-0 calibration runs) and asserts, per step:

  * `adapter.progression() == max(info["score"]/info["max_score"], won)`
  * before `done`: BALROG's own `get_stats()["progression"]` is 0.0 while the
    adapter's is live — i.e. the live value is doing real work;
  * at `done`: **the two agree exactly**;
  * on the dense game (`the_cooking_game`, `max_score` 17) the live value
    strictly increases at least twice *before* `done`, so a "new achievement"
    checkpoint really can fire mid-episode; on the two binary games
    (`max_score` 1) the single possible gain necessarily coincides with `done`,
    which the test requires instead;
  * `env_patches` discloses the deviation and carries BALROG's own number.

No LLM calls, no money.

  python -m tools.balrog_pae.games.textworld.test_progression_parity
"""
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "winning_traces.json")


def balrog_config():
    from omegaconf import OmegaConf

    return OmegaConf.load(str(importlib.resources.files("balrog") / "config" / "config.yaml"))


def run_task(task: str, spec: dict) -> dict:
    from tools.balrog_pae.adapters import TextWorldAdapter

    ad = TextWorldAdapter(task, balrog_config())
    ad.make()
    ad.reset(spec["seed"])

    failures: list[str] = []
    gains, last, done = 0, ad.progression(), False
    stock_before_done: list[float] = []
    final = {}

    for i, cmd in enumerate(spec["commands"]):
        ad.record(cmd, cmd)
        obs, reward, term, trunc, info = ad.step(cmd)
        done = bool(term or trunc)
        score, mx = info.get("score"), info.get("max_score") or 1
        expect = max(float(score or 0) / float(mx), 1.0 if info.get("won") else 0.0)
        live, stock = ad.progression(), ad.balrog_progression()

        if abs(live - expect) > 1e-12:
            failures.append(f"step {i}: live progression {live} != formula {expect}")
        if live > last + 1e-12:
            gains += 1
        last = live
        if not done:
            stock_before_done.append(stock)
        else:
            final = {"step": i, "live": live, "stock": stock, "score": score,
                     "max_score": mx, "won": info.get("won"),
                     "end_status": info.get("end_status")}
            if abs(live - stock) > 1e-12:
                failures.append(f"at done: live {live} != BALROG get_stats {stock}")
            break

    if not done:
        failures.append("trace never reached done - fixture is stale")
    dense = float(final.get("max_score") or 1) > 1
    if dense and gains < 2:
        failures.append(f"dense game but only {gains} live progression increase(s); "
                        "a mid-episode checkpoint trigger would not fire")
    if not dense and gains != 1:
        failures.append(f"binary game: expected exactly 1 live gain, got {gains}")
    if stock_before_done and max(stock_before_done) != 0.0:
        failures.append("BALROG's own progression was non-zero before done - "
                        "TextWorldWrapper changed, re-check this adapter")

    patches = list(ad.env_patches)
    if "textworld_live_progression" not in patches:
        failures.append("env_patches does not disclose the live-progression change")
    if not any(p.startswith("balrog_get_stats_progression=") for p in patches):
        failures.append("env_patches does not carry BALROG's own progression")
    ad.close()

    return {"task": task, "ok": not failures, "failures": failures,
            "live_progression_gains": gains, "steps": len(spec["commands"]),
            "final": final, "env_patches": patches}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="treasure_hunter,the_cooking_game,coin_collector")
    ap.add_argument("--traces", default=TRACES)
    a = ap.parse_args(argv)

    specs = json.load(open(a.traces))
    ok = True
    for task in a.tasks.split(","):
        r = run_task(task, specs[task])
        ok = ok and r["ok"]
        f = r["final"]
        print(f"{'PASS' if r['ok'] else 'FAIL'} {task:<18} "
              f"max_score={f.get('max_score'):<3} gains={r['live_progression_gains']:<3} "
              f"at done: live={f.get('live')} balrog={f.get('stock')} "
              f"score={f.get('score')}/{f.get('max_score')} won={f.get('won')}")
        for msg in r["failures"]:
            print("   !", msg)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
