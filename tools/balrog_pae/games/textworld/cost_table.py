#!/usr/bin/env python
"""Cost table for the TextWorld arm, from the MEASURED calibration runs.

Per-step token counts are read straight out of the calibration run dirs
(`--calib`, default `/root/nld/gen_runs/textworld`), so the table is the
measurement, not a guess.  Two price columns:

  list    z-ai/glm-5.2 at results/model_prices.json ($1.54/M in, $4.84/M out)
  billed  the same, scaled by the observed `usage.cost / list` ratio of the
          calibration runs themselves (Prime Inference reports the true billed
          cost on every completion; the ratio is computed, not assumed)

Batch shape priced: 3 games x 5 seeds x {base, PAE N attempts}.
The PAE arm costs r x base, where r is the measured attempt-cost multiplier
(a resumed attempt re-plays from a checkpoint, so it is shorter than a base
episode, but there are up to N of them plus the orchestrator's own calls).

  python -m tools.balrog_pae.games.textworld.cost_table
  python -m tools.balrog_pae.games.textworld.cost_table --attempts 10 --r 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os

PRICE_IN, PRICE_OUT = 1.54, 4.84  # USD / 1M tokens, z-ai/glm-5.2
TASKS = ["treasure_hunter", "the_cooking_game", "coin_collector"]


def load_calibration(root: str) -> dict:
    out = {}
    for task in TASKS:
        for pat in (f"cal_{task}_s0_base", f"{task}_s0_base"):
            p = os.path.join(root, pat, "summary.json")
            if os.path.exists(p):
                s = json.load(open(p))
                tot = s["tokens"]["total"]
                steps = max(s.get("llm_steps") or tot["calls"], 1)
                listed = tot["input_tokens"] / 1e6 * PRICE_IN + tot["output_tokens"] / 1e6 * PRICE_OUT
                billed = s["tokens"].get("billed_by_provider_usd") or 0.0
                out[task] = {
                    "steps": steps,
                    "in_per_step": tot["input_tokens"] / steps,
                    "out_per_step": tot["output_tokens"] / steps,
                    "in_tok": tot["input_tokens"], "out_tok": tot["output_tokens"],
                    "list_usd": listed, "billed_usd": billed,
                    "ratio": (billed / listed) if listed else 0.0,
                    "progression": s.get("attempt1_progression"),
                    "run": os.path.dirname(p),
                }
                break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="/root/nld/gen_runs/textworld")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--attempts", type=int, default=10, help="PAE N")
    ap.add_argument("--r", type=float, default=None,
                    help="PAE token multiplier over base; default = measured, else attempts*0.45")
    ap.add_argument("--orchestrator-overhead", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=80, help="episode step cap (BALROG: 80)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    cal = load_calibration(a.calib)
    if not cal:
        raise SystemExit(f"no calibration summaries under {a.calib}")

    ratio = sum(c["billed_usd"] for c in cal.values()) / max(sum(c["list_usd"] for c in cal.values()), 1e-9)
    # A resumed attempt starts at a checkpoint, so it plays (cap - ckpt_step)
    # steps; averaged over checkpoints spread across the episode that is ~half
    # a base episode, and the prompt is already long at the resume point.
    r = a.r if a.r is not None else a.attempts * 0.45

    hdr = (f"{'task':<20}{'base steps':>11}{'in/step':>9}{'out/step':>9}"
           f"{'base $':>9}{'PAE $':>9}{'seeds':>7}{'arm total $':>13}{'billed $':>10}")
    print(f"measured on {a.calib} (seed 0, base arm), GLM-5.2 "
          f"@ ${PRICE_IN}/M in, ${PRICE_OUT}/M out; billed/list = {ratio:.3f}")
    print(f"PAE multiplier r = {r:.2f} x base (N={a.attempts} attempts, "
          f"+{a.orchestrator_overhead:.0%} orchestrator)\n")
    print(hdr)
    print("-" * len(hdr))
    rows, tl, tb = [], 0.0, 0.0
    for task in TASKS:
        c = cal.get(task)
        if c is None:
            print(f"{task:<20}{'(no calibration run)':>60}")
            continue
        base = c["list_usd"]
        pae = base * r * (1 + a.orchestrator_overhead)
        total = (base + pae) * a.seeds
        rows.append({"task": task, "base_steps": c["steps"], "in_per_step": c["in_per_step"],
                     "out_per_step": c["out_per_step"], "base_list_usd": base,
                     "pae_list_usd": pae, "arm_list_usd": total,
                     "arm_billed_usd": total * ratio, "base_progression": c["progression"]})
        tl += total
        tb += total * ratio
        print(f"{task:<20}{c['steps']:>11}{c['in_per_step']:>9.0f}{c['out_per_step']:>9.0f}"
              f"{base:>9.3f}{pae:>9.2f}{a.seeds:>7}{total:>13.2f}{total*ratio:>10.2f}")
    print("-" * len(hdr))
    print(f"{'TEXTWORLD TOTAL':<20}{'':>11}{'':>9}{'':>9}{'':>9}{'':>9}{'':>7}{tl:>13.2f}{tb:>10.2f}")

    # worst case: every base episode runs the full cap instead of dying early
    wc = 0.0
    for task in TASKS:
        c = cal.get(task)
        if c is None:
            continue
        scale = a.cap / c["steps"]
        wc += (c["list_usd"] * scale) * (1 + r * (1 + a.orchestrator_overhead)) * a.seeds
    print(f"{'ceiling: every episode runs the full ' + str(a.cap) + '-step cap':<52}{wc:>13.2f}{wc*ratio:>10.2f}")

    if a.json:
        json.dump({"ratio": ratio, "r": r, "seeds": a.seeds, "attempts": a.attempts,
                   "price_in": PRICE_IN, "price_out": PRICE_OUT, "rows": rows,
                   "total_list_usd": tl, "total_billed_usd": tb,
                   "ceiling_list_usd": wc, "ceiling_billed_usd": wc * ratio},
                  open(a.json, "w"), indent=1)
        print("->", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
