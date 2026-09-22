#!/usr/bin/env python
"""Aggregate the MiniHack panel: per-run rows plus a per-task base-vs-PAE table.

Reported numbers, all of them BALROG's own ``get_stats()["progression"]``:

  base arm : ``progression`` of the one episode (== ``attempt1_progression``)
  PAE arm  : ``pae_best_progression`` (run-level best over attempts) AND
             ``attempt1_progression`` (the run's first attempt, which is itself
             a stock BALROG episode - a free second sample of the base protocol)

Cost is reported twice: ``list $`` from the price table and ``billed $`` from
Prime's own ``usage.cost``. The wallet delta from ``wallet.json`` is carried
alongside as an upper bound only - the wallet is shared with other jobs.

  python -m tools.balrog_pae.games.minihack.aggregate_minihack /root/nld/gen_runs/minihack \
      --json out.json --csv out.csv
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

TASK_ORDER = [
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
    "MiniHack-CorridorBattle-Dark-v0",
    "MiniHack-Boxoban-Medium-v0",
    "MiniHack-Boxoban-Hard-v0",
]


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def rows_for(roots) -> list[dict]:
    out = []
    for root in roots:
        for f in sorted(Path(root).rglob("summary.json")):
            s = json.loads(f.read_text())
            if s.get("game") != "minihack":
                continue
            d = f.parent
            wallet = {}
            if (d / "wallet.json").exists():
                wallet = json.loads((d / "wallet.json").read_text())
            tok = s["tokens"]["total"]
            arm = "base" if s.get("arm") == "base" else "pae"
            out.append({
                "task": s["task"], "seed": s["seed"], "arm": arm, "arm_full": s.get("arm"),
                "run_dir": str(d),
                "attempt1_progression": s.get("attempt1_progression", 0.0),
                "pae_best_progression": s.get("pae_best_progression", s.get("max_progression", 0.0)),
                "attempts": s.get("attempts", 0),
                "checkpoints": s.get("checkpoints", 0),
                "committed_steps_max": s.get("committed_steps_max", 0),
                "total_env_steps": s.get("total_env_steps", 0),
                "llm_steps": s.get("llm_steps", 0),
                "input_tokens": tok["input_tokens"], "output_tokens": tok["output_tokens"],
                "in_per_step": s.get("tokens_per_llm_step", {}).get("input"),
                "out_per_step": s.get("tokens_per_llm_step", {}).get("output"),
                "list_usd": tok["cost_usd"],
                "billed_usd": s["tokens"].get("billed_by_provider_usd"),
                "wallet_delta_usd": wallet.get("wallet_delta_usd"),
                "aux_label": s.get("aux_label"), "aux_max_measured": s.get("aux_max_measured"),
                "stop_reason": s.get("stop_reason"),
                "directive_rejections": (s.get("orchestrator") or {}).get("directive_rejections", 0),
                "root_picks": (s.get("orchestrator") or {}).get("root_picks", 0),
                "env_patches": s.get("env_patches", []),
            })
    key = lambda r: (TASK_ORDER.index(r["task"]) if r["task"] in TASK_ORDER else 99, r["seed"], r["arm"])  # noqa: E731
    return sorted(out, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = rows_for(a.roots)
    if not rows:
        print("no minihack summary.json under", a.roots)
        return

    hdr = (f"{'task':<32}{'seed':>5}{'arm':>6}{'att':>5}{'a1prog':>8}{'best':>7}"
           f"{'steps':>7}{'llm':>6}{'in/st':>7}{'out/st':>7}{'list$':>8}{'billed$':>9}{'aux':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['task']:<32}{r['seed']:>5}{r['arm']:>6}{r['attempts']:>5}"
              f"{r['attempt1_progression']:>8.3f}{r['pae_best_progression']:>7.3f}"
              f"{r['committed_steps_max']:>7}{r['llm_steps']:>6}"
              f"{(r['in_per_step'] or 0):>7}{(r['out_per_step'] or 0):>7}"
              f"{r['list_usd']:>8.3f}{(r['billed_usd'] or 0.0):>9.3f}"
              f"{(r['aux_max_measured'] or 0):>6.0f}")

    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task"]][r["arm"]].append(r)

    print()
    hdr2 = (f"{'task':<32}{'n_base':>7}{'base mean':>11}{'n_pae':>6}{'PAE a1 mean':>13}"
            f"{'PAE best mean':>15}{'delta':>8}{'list $':>9}{'billed $':>10}")
    print(hdr2)
    print("-" * len(hdr2))
    table = []
    for task in sorted(by_task, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
        b, p = by_task[task]["base"], by_task[task]["pae"]
        bm = mean([r["attempt1_progression"] for r in b])
        pa1 = mean([r["attempt1_progression"] for r in p])
        pb = mean([r["pae_best_progression"] for r in p])
        lst = sum(r["list_usd"] for r in b + p)
        bil = sum(r["billed_usd"] or 0.0 for r in b + p)
        print(f"{task:<32}{len(b):>7}{bm:>11.3f}{len(p):>6}{pa1:>13.3f}{pb:>15.3f}"
              f"{(pb - bm):>8.3f}{lst:>9.3f}{bil:>10.3f}")
        table.append(dict(task=task, n_base=len(b), base_progression_mean=bm, n_pae=len(p),
                          pae_attempt1_progression_mean=pa1, pae_best_progression_mean=pb,
                          delta=pb - bm, list_usd=lst, billed_usd=bil))

    tot_l = sum(r["list_usd"] for r in rows)
    tot_b = sum(r["billed_usd"] or 0.0 for r in rows)
    print(f"\ntotal: {len(rows)} runs, list ${tot_l:.2f}, billed ${tot_b:.2f}"
          f" (ratio {tot_b / tot_l:.3f})" if tot_l else "")

    out = {"runs": rows, "per_task": table,
           "totals": {"runs": len(rows), "list_usd": tot_l, "billed_usd": tot_b,
                      "billed_ratio": (tot_b / tot_l) if tot_l else None}}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
