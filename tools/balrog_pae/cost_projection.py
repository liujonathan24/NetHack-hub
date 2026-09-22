#!/usr/bin/env python
"""Per-task cost projection for the game-generalization study.

Per-step token figures are MEASURED on this stack with GLM-5.2 (see the smoke
runs under runs/); env-step counts per task are BALROG's published-trace
averages over the four frontier reasoning models (tasks.md section 3.4), which
are an upper bound for GLM-5.2 (it dies earlier).

Prices: results/model_prices.json, z-ai/glm-5.2 = $1.54/M in, $4.84/M out.
Prime Inference's own per-call `usage.cost` came in at 0.29-0.39x of that list
price across our smokes, so the list-price column is a conservative ceiling and
the second column is what the wallet is likely to see.
"""
from __future__ import annotations

import argparse

PRICE_IN, PRICE_OUT = 1.54, 4.84
BILLED_RATIO = 0.36  # observed usage.cost / list price over the three smokes

# task -> (episodes, env steps for that many episodes, input tok/step, output tok/step)
TASKS = {
    "MiniHack Quest-Easy":          (5, 470, 4500, 300),
    "MiniHack Quest-Medium":        (5, 240, 4500, 300),
    "MiniHack CorridorBattle-Dark": (5, 250, 4500, 300),
    "MiniHack Boxoban-Medium":      (5, 450, 4500, 300),
    "MiniHack Boxoban-Hard":        (5, 440, 4500, 300),
    "Crafter default":              (10, 2700, 1490, 170),
    "TextWorld treasure_hunter":    (5, 400, 1500, 300),
    "TextWorld the_cooking_game":   (5, 400, 1500, 300),
    "TextWorld coin_collector":     (5, 400, 1500, 300),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=3.0, help="PAE token multiplier over base")
    ap.add_argument("--controls", type=int, default=1, help="token-matched sampling controls (each costs r x base)")
    ap.add_argument("--orchestrator-overhead", type=float, default=0.05)
    a = ap.parse_args()

    hdr = f"{'task':<30}{'eps':>4}{'steps':>7}{'in (M)':>9}{'out (M)':>9}{'base $':>9}{'PAE $':>9}{'all arms $':>12}"
    print(hdr)
    print("-" * len(hdr))
    tot = [0.0] * 5
    arms = 1 + a.r * (1 + a.controls)
    for name, (eps, steps, tin, tout) in TASKS.items():
        mi, mo = steps * tin / 1e6, steps * tout / 1e6
        base = mi * PRICE_IN + mo * PRICE_OUT
        pae = base * a.r * (1 + a.orchestrator_overhead)
        allarms = base * arms * (1 + a.orchestrator_overhead)
        print(f"{name:<30}{eps:>4}{steps:>7}{mi:>9.2f}{mo:>9.2f}{base:>9.2f}{pae:>9.2f}{allarms:>12.2f}")
        tot = [t + v for t, v in zip(tot, (mi, mo, base, pae, allarms))]
    print("-" * len(hdr))
    print(f"{'TOTAL (list price)':<30}{'':>4}{'':>7}{tot[0]:>9.2f}{tot[1]:>9.2f}{tot[2]:>9.2f}{tot[3]:>9.2f}{tot[4]:>12.2f}")
    print(f"{'TOTAL (observed billed ~0.36x)':<30}{'':>4}{'':>7}{'':>9}{'':>9}"
          f"{tot[2]*BILLED_RATIO:>9.2f}{tot[3]*BILLED_RATIO:>9.2f}{tot[4]*BILLED_RATIO:>12.2f}")
    print(f"\narms priced in 'all arms': base + PAE (r={a.r}) + {a.controls} token-matched control(s) at r x base")


if __name__ == "__main__":
    main()
