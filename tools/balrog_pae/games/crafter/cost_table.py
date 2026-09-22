#!/usr/bin/env python
"""Cost table for the Crafter panel: 10 seeds x {base, PAE N=10}.

All per-step token figures are MEASURED on this stack (GLM-5.2 over Prime
Inference); see games/crafter/calibration.json. Two columns are given for every
row: the list price from results/model_prices.json, and the price at the
observed billed ratio (Prime's own per-call ``usage.cost`` summed over the
calibration runs, divided by their list price).

  python -m tools.balrog_pae.games.crafter.cost_table
  python -m tools.balrog_pae.games.crafter.cost_table --calibration <file.json>
"""
from __future__ import annotations

import argparse
import json

PRICE_IN, PRICE_OUT = 1.54, 4.84      # z-ai/glm-5.2, USD per 1M tokens

# --- MEASURED defaults (overridden by --calibration) -------------------------
DEFAULTS = {
    "in_tok_per_step": 1930,
    "out_tok_per_step": 150,
    "base_steps": 215,
    "billed_ratio": 0.39,
    "orch_overhead": 0.05,   # orchestrator turns as a fraction of player tokens
}


def usd(steps: int, tin: float, tout: float) -> float:
    return steps * (tin * PRICE_IN + tout * PRICE_OUT) / 1e6


def table(seeds: int, attempts: int, cap: int, m: dict) -> list[dict]:
    tin, tout, base_steps = m["in_tok_per_step"], m["out_tok_per_step"], m["base_steps"]
    rows = []

    base_calls = base_steps
    rows.append({
        "arm": "base (unchanged BALROG)",
        "runs": seeds,
        "llm_calls_per_run": base_calls,
        "note": f"1 episode/seed, dies at ~{base_steps} steps (BALROG cap 2000 never binds)",
        "list_usd": seeds * usd(base_calls, tin, tout),
    })

    # PAE: attempt 1 is a base episode; each later attempt resumes ~K steps before
    # the previous ending and plays to the explicit cap, so its LLM calls are
    # bounded above by `cap` and, empirically, are close to the base length.
    for lab, per_attempt in (("low  (resume plays ~base length)", min(cap, base_steps)),
                             ("high (every resume runs to the cap)", cap)):
        calls = base_steps + (attempts - 1) * per_attempt
        rows.append({
            "arm": f"PAE N={attempts}, cap {cap} - {lab}",
            "runs": seeds,
            "llm_calls_per_run": calls,
            "note": f"attempt 1 = base episode; {attempts - 1} resumes x {per_attempt} steps, +{int(m['orch_overhead']*100)}% orchestrator",
            "list_usd": seeds * usd(calls, tin, tout) * (1 + m["orch_overhead"]),
        })

    # the uncapped PAE arm, for reference: this is the budget risk tasks.md flags
    unc = base_steps + (attempts - 1) * 2000
    rows.append({
        "arm": f"PAE N={attempts}, UNCAPPED (BALROG 2000)",
        "runs": seeds,
        "llm_calls_per_run": unc,
        "note": "what happens if --max-steps is not passed: the budget risk",
        "list_usd": seeds * usd(unc, tin, tout) * (1 + m["orch_overhead"]),
    })
    for r in rows:
        r["billed_usd"] = r["list_usd"] * m["billed_ratio"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--cap", type=int, default=400, help="PAE --max-steps")
    ap.add_argument("--calibration", default=None, help="games/crafter/calibration.json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    m = dict(DEFAULTS)
    if a.calibration:
        c = json.load(open(a.calibration))
        mm = c.get("measured", {})
        for k in ("in_tok_per_step", "out_tok_per_step", "base_steps", "billed_ratio"):
            if mm.get(k) is not None:
                m[k] = mm[k]

    rows = table(a.seeds, a.attempts, a.cap, m)
    print(f"Crafter panel cost — GLM-5.2 @ ${PRICE_IN}/M in, ${PRICE_OUT}/M out (list); "
          f"billed ratio {m['billed_ratio']:.2f}x")
    print(f"measured: {m['in_tok_per_step']} in / {m['out_tok_per_step']} out tokens per step; "
          f"base episode {m['base_steps']} steps")
    print()
    hdr = f"{'arm':<44}{'runs':>5}{'calls/run':>11}{'list $':>10}{'billed $':>10}"
    print(hdr)
    print("-" * len(hdr))
    tl = tb = 0.0
    for r in rows:
        if "UNCAPPED" in r["arm"]:
            print("-" * len(hdr))
        print(f"{r['arm']:<44}{r['runs']:>5}{r['llm_calls_per_run']:>11}{r['list_usd']:>10.2f}{r['billed_usd']:>10.2f}")
        if "UNCAPPED" not in r["arm"] and "high" not in r["arm"]:
            tl += r["list_usd"]
            tb += r["billed_usd"]
    print("-" * len(hdr))
    print(f"{'PANEL TOTAL (base + PAE low estimate)':<44}{'':>5}{'':>11}{tl:>10.2f}{tb:>10.2f}")
    hi = rows[0]["list_usd"] + rows[2]["list_usd"]
    print(f"{'PANEL TOTAL (base + PAE high estimate)':<44}{'':>5}{'':>11}{hi:>10.2f}{hi*m['billed_ratio']:>10.2f}")
    print()
    for r in rows:
        print(f"  {r['arm']}: {r['note']}")
    if a.json:
        json.dump({"params": {**m, "seeds": a.seeds, "attempts": a.attempts, "cap": a.cap,
                              "price_in": PRICE_IN, "price_out": PRICE_OUT},
                   "rows": rows,
                   "panel_total_list_usd": tl, "panel_total_billed_usd": tb},
                  open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
