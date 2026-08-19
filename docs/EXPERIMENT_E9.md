# E9 — Enforced constraints (does forcing the model's own stated plan help?)

E8 established that the model **reads guidance, reasons about it correctly, and
then does not change its policy** (E8a descent norm shown 20×, followed 0×;
prayer already well-timed without the hint). And across 35 games the death mode
is overwhelmingly singular: **34/35 games end with Dlvl > XL (mean gap 3.83);
32/33 deaths were underleveled.** The model descends faster than it levels, and
NetHack scales monster difficulty with depth, so it arrives underprepared and
dies.

E9 turns advice into a **constraint** to test whether that gap is *causal*.

## E9a — hard descent enforcement (PLANNING axis)

**Knob:** `descent_gate="enforce"` (`nethack.py`). Every descent (`np_down`, or
`np_press_key('>')`) is **refused with no override** while `XL < norm(Dlvl)`,
where `norm` is the human-ascended-run arrival norm (`human_norms.py`, the same
table E8a used). Ascent is never gated. The stairs "unlock" once the model
reaches the norm; it must gain XP on the current level first.

**Config:** identical to the NPCORE_v3 control — GLM-5.2, `full_nle`, BBOX_MIN,
Valkyrie, 200 turns, `skill_set="np_core,request_map,search"`, frames on — plus
`descent_gate=enforce`. Seeds 0–4.

**The two outcomes, both informative:**
- **Survival/BALROG improves** → underleveling is *causal*; the reasoning→policy
  gap is the binding constraint, and scaffold-enforced pacing is a real lever.
- **It dies anyway** (attrition at shallow depth, or stalls unable to level) →
  depth is not the killer; the gap is deeper than pacing, and forcing the plan
  the model already endorses is not enough.

**Watch for:** stall/deadlock (can't descend, can't kill enough to level) — the
200-turn budget bounds it, and a cell that spends its budget stuck at low Dlvl
is itself a finding (the model cannot farm XP efficiently).

## Control replicates

To read E9a (and E8) against real within-seed variance, the NPCORE_v3 control is
extended from 1 to **3 runs per seed** (same 5 seeds, same BBOX_MIN config; the
game seed is fixed so the variance is the model's sampling). Median/spread from
these replicates is the null band every E-cell is judged against.

## Deferred

- **E9b (reward/objective reshaping):** not run — the model's depth-greed may be
  metric-induced (BALROG rewards depth; descending is one keystroke, leveling is
  many risky kills), but we test the enforcement lever first.
- **Cross-model replication** (Claude / GPT on the same harness) — is
  "reads-and-ignores" GLM-specific or general? Load-bearing for the blog thesis.
- **Continual life** — does the model learn to pace itself across deaths?
