#!/usr/bin/env python
"""Per-task cost projection for the game-generalization panel.

Every number carries its provenance: `measured:<run>` means it came out of a
smoke in runs/, `assumed:<why>` means it did not. Nothing here is labelled
"measured" unless a run produced it.

Two PAE multipliers are kept separate, because they answer different questions:

  step_mult   = PAE LLM calls / base LLM calls. With N attempts that resume
                mid-episode, PAE pays for the tail of each attempt, not a whole
                extra episode.
  prompt_mult = PAE mean prompt length / base mean prompt length. Resumed
                attempts start with a full 16-observation window, whereas a base
                episode spends its first 16 steps on short prompts, so PAE's
                average input per call is HIGHER than base's. Applied to input
                tokens only; output per call is unchanged.

The orchestrator overhead applies to the PAE arm only - the base arm has no
orchestrator.

  python -m tools.balrog_pae.cost_projection
  python -m tools.balrog_pae.cost_projection --step-mult 4.2 --measure runs/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRICE_IN, PRICE_OUT = 1.54, 4.84  # results/model_prices.json, z-ai/glm-5.2

# task -> episodes, env steps for that many episodes, in tok/step, out tok/step, provenance
TASKS = [
    # (game, task, episodes, steps, tok_in, tok_out, steps_src, tok_src)
    ("minihack", "Quest-Easy", 5, 470, 4500, 320, "tasks.md 3.4 (frontier traces)", "measured: mh_quest_easy_s0_* (saturated window)"),
    ("minihack", "Quest-Medium", 5, 240, 4500, 320, "tasks.md 3.4 (frontier traces)", "assumed: same prompt shape as Quest-Easy"),
    ("minihack", "CorridorBattle-Dark", 5, 250, 4500, 320, "tasks.md 3.4 (frontier traces)", "assumed: same prompt shape as Quest-Easy"),
    ("minihack", "Boxoban-Medium", 5, 450, 4500, 320, "tasks.md 3.4 (frontier traces)", "assumed: same prompt shape as Quest-Easy"),
    ("minihack", "Boxoban-Hard", 5, 440, 4500, 320, "tasks.md 3.4 (frontier traces)", "assumed: same prompt shape as Quest-Easy"),
    ("crafter", "default", 10, 2700, 1490, 170, "tasks.md 3.4 (frontier traces)", "measured: crafter_s0_pae"),
    ("textworld", "treasure_hunter", 5, 400, 1500, 300, "assumed: 80-step cap x 5 episodes", "assumed: scaled from the_cooking_game"),
    ("textworld", "the_cooking_game", 5, 400, 1500, 300, "assumed: 80-step cap x 5 episodes", "measured: tw_cooking_s0_pae"),
    ("textworld", "coin_collector", 5, 400, 1500, 300, "assumed: 80-step cap x 5 episodes", "assumed: scaled from the_cooking_game"),
]

DEFAULTS = dict(step_mult=4.0, prompt_mult=1.25, billed_ratio=0.36, orch=0.05)


def measure(run_root: str, attempts: int = 10) -> dict:
    """Derive the multipliers and the billed ratio from actual runs.

    step_mult is NORMALISED to the panel's N and measured WITHIN each PAE run:

        step_mult(N) = (a1_calls + (N-1) * mean_resume_calls) / a1_calls

    The denominator is the run's OWN attempt 1, which is a stock BALROG episode
    under the same protocol as the base arm. Dividing by a separate base run
    instead makes the number swing wildly on small samples - a base episode that
    happens to die at step 7 against a PAE attempt 1 that survives to step 13 is
    sampling noise on one seed, not a real 2x. The base-run ratio is reported
    alongside as a cross-check, never used.

    On a one-seed smoke this is a planning estimate, not a measurement.
    """
    base, pae = [], []
    for f in sorted(Path(run_root).rglob("summary.json")):
        s = json.loads(f.read_text())
        s["_dir"] = f.parent
        (base if s.get("arm") == "base" else pae).append(s)
    out = {}
    if base and pae:
        base_calls = sum(s["llm_steps"] for s in base) / len(base)
        projected = []
        for s in pae:
            try:
                rows = [json.loads(l) for l in (s["_dir"] / "attempts.jsonl").open()]
            except Exception:  # noqa: BLE001
                continue
            if len(rows) < 2:
                continue
            a1 = rows[0]["calls"]
            resume = sum(r["calls"] for r in rows[1:]) / (len(rows) - 1)
            projected.append((a1 + (attempts - 1) * resume, a1))
        if projected:
            out["step_mult"] = sum(p / a for p, a in projected) / len(projected)
            if base_calls:
                out["_step_mult_vs_base_run"] = (
                    sum(p for p, _ in projected) / len(projected) / base_calls)
        bp = sum(s["tokens_per_llm_step"]["input"] for s in base) / len(base)
        pp = sum(s["tokens_per_llm_step"]["input"] for s in pae) / len(pae)
        out["prompt_mult"] = pp / bp if bp else None
    lst = sum(s["tokens"]["total"]["cost_usd"] for s in base + pae)
    bil = sum(s["tokens"].get("billed_by_provider_usd", 0.0) for s in base + pae)
    if lst and bil:
        out["billed_ratio"] = bil / lst
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=10, help="PAE attempts per episode (N)")
    ap.add_argument("--step-mult", type=float, default=None)
    ap.add_argument("--prompt-mult", type=float, default=None)
    ap.add_argument("--billed-ratio", type=float, default=None)
    ap.add_argument("--orchestrator-overhead", type=float, default=None)
    ap.add_argument("--measure", default=None, help="derive multipliers from this runs/ root")
    a = ap.parse_args()

    d = dict(DEFAULTS)
    src = {k: "assumed: default" for k in d}
    xcheck = None
    if a.measure:
        m = measure(a.measure, attempts=a.attempts)
        xcheck = m.pop("_step_mult_vs_base_run", None)
        for k, v in m.items():
            if v:
                d[k], src[k] = v, f"measured: {a.measure}"
    for k, v in (("step_mult", a.step_mult), ("prompt_mult", a.prompt_mult),
                 ("billed_ratio", a.billed_ratio), ("orch", a.orchestrator_overhead)):
        if v is not None:
            d[k], src[k] = v, "given on the command line"

    print("PANEL: 5 MiniHack tasks x 5 seeds, Crafter x 10 seeds, TextWorld x 3 games x 5 seeds")
    print(f"ARMS : base (stock BALROG) and PAE (orchestrator + directive, N={a.attempts} attempts)")
    print(f"       step_mult   = {d['step_mult']:.2f}   [{src['step_mult']}]")
    print(f"       prompt_mult = {d['prompt_mult']:.2f}   [{src['prompt_mult']}]  (input only)")
    print(f"       orchestrator overhead {d['orch']:.0%} on the PAE arm only")
    print(f"       billed ratio = {d['billed_ratio']:.2f} of list [{src['billed_ratio']}]")
    if a.measure and xcheck:
        print(f"       (cross-check: dividing by the separate base run instead gives "
              f"step_mult {xcheck:.1f} - noisy on one seed, not used)")
    print()
    hdr = (f"{'game':<10}{'task':<22}{'eps':>4}{'steps':>7}{'in/step':>9}{'out/step':>9}"
           f"{'base $':>9}{'PAE $':>9}{'both $':>9}")
    print(hdr)
    print("-" * len(hdr))
    tb = tp = 0.0
    for game, task, eps, steps, tin, tout, _ssrc, _tsrc in TASKS:
        base = steps * tin / 1e6 * PRICE_IN + steps * tout / 1e6 * PRICE_OUT
        pae_in = steps * d["step_mult"] * tin * d["prompt_mult"] / 1e6 * PRICE_IN
        pae_out = steps * d["step_mult"] * tout / 1e6 * PRICE_OUT
        pae = (pae_in + pae_out) * (1 + d["orch"])
        print(f"{game:<10}{task:<22}{eps:>4}{steps:>7}{tin:>9}{tout:>9}{base:>9.2f}{pae:>9.2f}{base + pae:>9.2f}")
        tb += base
        tp += pae
    print("-" * len(hdr))
    print(f"{'TOTAL list price':<32}{'':>7}{'':>9}{'':>9}{tb:>9.2f}{tp:>9.2f}{tb + tp:>9.2f}")
    r = d["billed_ratio"]
    print(f"{'TOTAL at billed ratio':<32}{'':>7}{'':>9}{'':>9}{tb * r:>9.2f}{tp * r:>9.2f}{(tb + tp) * r:>9.2f}")
    print()
    print("per-row provenance")
    for game, task, _e, steps, tin, tout, ssrc, tsrc in TASKS:
        print(f"  {game}/{task:<22} steps={steps:<5} [{ssrc}]")
        print(f"  {'':<{len(game) + 1}}{'':<22} tokens={tin}/{tout} [{tsrc}]")


if __name__ == "__main__":
    main()
