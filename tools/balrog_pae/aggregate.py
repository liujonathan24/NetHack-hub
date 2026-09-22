#!/usr/bin/env python
"""Aggregate run summaries: base mean vs PAE mean-of-best, per game/task.

Base arm  -> mean over seeds of `attempt1_progression` (a stock BALROG episode).
PAE arm   -> mean over seeds of `pae_best_progression` (the run-level best).
The PAE arm's own `attempt1_progression` is also averaged: it is a second,
free sample of the base protocol and should track the base column.

  python -m tools.balrog_pae.aggregate runs/           # walks for summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(roots) -> list[dict]:
    out = []
    for root in roots:
        p = Path(root)
        files = [p] if p.name == "summary.json" else sorted(p.rglob("summary.json"))
        for f in files:
            try:
                out.append(json.loads(f.read_text()))
            except Exception as e:  # noqa: BLE001
                print(f"skip {f}: {e}")
    return out


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    runs = load(a.roots)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r.get("game"), r.get("task"), r.get("arm"))].append(r)

    hdr = (f"{'game':<10}{'task':<30}{'arm':<38}{'n':>3}{'seeds':>7}"
           f"{'a1 prog':>9}{'best prog':>11}{'steps':>8}{'llm':>7}{'list $':>9}{'billed $':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for (game, task, arm), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))):
        a1 = mean([r.get("attempt1_progression", 0.0) for r in rs])
        best = mean([r.get("pae_best_progression", r.get("max_progression", 0.0)) for r in rs])
        steps = mean([r.get("committed_steps_max", 0) for r in rs])
        llm = mean([r.get("llm_steps", 0) for r in rs])
        lst = sum(r["tokens"]["total"]["cost_usd"] for r in rs)
        billed = sum(r["tokens"].get("billed_by_provider_usd", 0.0) for r in rs)
        seeds = len({r.get("seed") for r in rs})
        print(f"{game:<10}{task:<30}{arm:<38}{len(rs):>3}{seeds:>7}"
              f"{a1:>9.3f}{best:>11.3f}{steps:>8.1f}{llm:>7.1f}{lst:>9.3f}{billed:>10.3f}")
        rows.append(dict(game=game, task=task, arm=arm, n=len(rs), seeds=seeds,
                         attempt1_progression_mean=a1, pae_best_progression_mean=best,
                         committed_steps_mean=steps, llm_steps_mean=llm,
                         list_usd=lst, billed_usd=billed))

    # base vs PAE side by side where both exist
    print()
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["game"], r["task"])]["base" if r["arm"] == "base" else r["arm"]] = r
    print(f"{'game/task':<42}{'base mean':>11}{'PAE mean-of-best':>19}{'delta':>9}")
    print("-" * 81)
    for (game, task), d in sorted(pairs.items()):
        base = d.get("base")
        for arm, r in d.items():
            if arm == "base":
                continue
            b = base["attempt1_progression_mean"] if base else float("nan")
            print(f"{game + '/' + task:<42}{b:>11.3f}{r['pae_best_progression_mean']:>19.3f}"
                  f"{r['pae_best_progression_mean'] - b:>9.3f}   [{arm}]")

    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
