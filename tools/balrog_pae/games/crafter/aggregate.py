#!/usr/bin/env python
"""Per-seed aggregate for the Crafter batch (seeds 0-9 x {base, PAE N=10}).

Base arm  : BALROG progression for that episode  (`attempt1_progression`,
            which for `--base-only` IS the single stock BALROG episode).
PAE arm   : `attempt1_progression` (its attempt 1 is also a stock BALROG
            episode, so it is a second free sample of the base protocol) and
            `pae_best_progression` (the run-level best over all attempts).

Also reports achievements = 22 x progression, steps, tokens and both costs
(list price and the provider's own `usage.cost`).

  python -m tools.balrog_pae.games.crafter.aggregate /root/nld/gen_runs/crafter \
      --json .../aggregate.json --csv .../aggregate.csv
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def stderr(xs):
    return st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


def load(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.rglob("summary.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"skip {f}: {e}")
            continue
        if d.get("game") != "crafter":
            continue
        t = d["tokens"]["total"]
        out.append({
            "run_dir": str(f.parent),
            "seed": d.get("seed"),
            "arm": "base" if d.get("arm") == "base" else "pae",
            "arm_full": d.get("arm"),
            "attempts": d.get("attempts", 0),
            "checkpoints": d.get("checkpoints", 0),
            "attempt1_progression": d.get("attempt1_progression", 0.0),
            "pae_best_progression": d.get("pae_best_progression", d.get("max_progression", 0.0)),
            "attempt1_achievements": round(22 * d.get("attempt1_progression", 0.0)),
            "best_achievements": round(22 * d.get("pae_best_progression", d.get("max_progression", 0.0))),
            "committed_steps_max": d.get("committed_steps_max", 0),
            "total_env_steps": d.get("total_env_steps", 0),
            "llm_steps": d.get("llm_steps", 0),
            "input_tokens": t["input_tokens"],
            "output_tokens": t["output_tokens"],
            "list_usd": t["cost_usd"],
            "billed_usd": d["tokens"].get("billed_by_provider_usd", 0.0),
            "env_patches": d.get("env_patches", []),
            "stop_reason": d.get("stop_reason"),
        })
    return sorted(out, key=lambda r: (r["arm"], r["seed"] if r["seed"] is not None else -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = load(Path(a.root))
    if not rows:
        print(f"no crafter summary.json under {a.root}")
        return 1

    hdr = (f"{'arm':<6}{'seed':>5}{'att':>5}{'ckpt':>6}{'a1 prog':>9}{'a1 ach':>8}"
           f"{'best prog':>11}{'best ach':>10}{'steps':>8}{'llm':>6}{'in tok':>10}{'out tok':>9}"
           f"{'list $':>9}{'billed $':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<6}{r['seed']:>5}{r['attempts']:>5}{r['checkpoints']:>6}"
              f"{r['attempt1_progression']:>9.4f}{r['attempt1_achievements']:>8}"
              f"{r['pae_best_progression']:>11.4f}{r['best_achievements']:>10}"
              f"{r['committed_steps_max']:>8}{r['llm_steps']:>6}{r['input_tokens']:>10}"
              f"{r['output_tokens']:>9}{r['list_usd']:>9.3f}{r['billed_usd']:>10.3f}")

    base = [r for r in rows if r["arm"] == "base"]
    pae = [r for r in rows if r["arm"] == "pae"]
    summary = {
        "root": str(a.root),
        "n_base": len(base),
        "n_pae": len(pae),
        "base_progression_per_episode": {str(r["seed"]): r["attempt1_progression"] for r in base},
        "pae_attempt1_progression": {str(r["seed"]): r["attempt1_progression"] for r in pae},
        "pae_best_progression": {str(r["seed"]): r["pae_best_progression"] for r in pae},
        "base_mean_progression": mean([r["attempt1_progression"] for r in base]),
        "base_sem": stderr([r["attempt1_progression"] for r in base]),
        "pae_attempt1_mean_progression": mean([r["attempt1_progression"] for r in pae]),
        "pae_best_mean_progression": mean([r["pae_best_progression"] for r in pae]),
        "pae_best_sem": stderr([r["pae_best_progression"] for r in pae]),
        "delta_pae_best_minus_base": mean([r["pae_best_progression"] for r in pae])
                                     - mean([r["attempt1_progression"] for r in base]),
        "base_mean_achievements": 22 * mean([r["attempt1_progression"] for r in base]),
        "pae_best_mean_achievements": 22 * mean([r["pae_best_progression"] for r in pae]),
        "base_mean_committed_steps": mean([r["committed_steps_max"] for r in base]),
        "pae_mean_committed_steps": mean([r["committed_steps_max"] for r in pae]),
        "total_list_usd": sum(r["list_usd"] for r in rows),
        "total_billed_usd": sum(r["billed_usd"] for r in rows),
        "billed_over_list": (sum(r["billed_usd"] for r in rows) / sum(r["list_usd"] for r in rows))
                            if sum(r["list_usd"] for r in rows) else None,
        "env_patches": sorted({p for r in rows for p in r["env_patches"]}),
        "rows": rows,
    }
    print()
    print(f"base      mean progression {summary['base_mean_progression']:.4f} "
          f"(+/- {summary['base_sem']:.4f} sem)  = {summary['base_mean_achievements']:.2f}/22 achievements, n={len(base)}")
    print(f"PAE a1    mean progression {summary['pae_attempt1_mean_progression']:.4f}  (free second base sample)")
    print(f"PAE best  mean progression {summary['pae_best_mean_progression']:.4f} "
          f"(+/- {summary['pae_best_sem']:.4f} sem) = {summary['pae_best_mean_achievements']:.2f}/22, n={len(pae)}")
    print(f"delta (PAE best - base)    {summary['delta_pae_best_minus_base']:+.4f}")
    print(f"total ${summary['total_list_usd']:.2f} list / ${summary['total_billed_usd']:.2f} billed"
          + (f" (ratio {summary['billed_over_list']:.2f}x)" if summary["billed_over_list"] else ""))
    print(f"env_patches (disclosed): {summary['env_patches']}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {a.json}")
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "env_patches"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
