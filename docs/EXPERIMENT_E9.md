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

## E9b — awareness probe (does prompted metacognition change behavior?)

Where E9a removes agency, E9b keeps it and instead makes the model *reflect*.
E8 found the reasoning is accurate but **inert** — the model narrates the game
well and then re-issues moves that just failed (E7 seed 0: `np_move_to(57,13)`
nine times, three at zero progress). The hypothesis: the model has the reasoning
ability but never *triggers* self-assessment, so forcing it each turn — "is this
working? what's the new plan?" — may close the reasoning→policy gap without a
hard constraint.

**Knob:** `reflect` (`nethack.py`, `prompt/reflection.py`). When on, three
reflection questions are appended to EVERY turn's observation — nothing else. No
machine analysis of the model's behavior, no stall detection, no map forcing:
this is deliberately *just a prompt change* asking the model to reflect, so the
only thing under test is whether prompted reflection changes behavior. The
published tool schemas stay byte-identical to the control. The block:

```
[Reflect on the previous steps before acting:
 1. What have you been working on in the past steps?
 2. What is going wrong, if anything?
 3. What is the updated plan?]
```

**Config:** the NPCORE_v3 control + `reflect=on`. Seeds 0–4.

**What it tests / reads:** whether *prompted* reflection moves behavior off the
~2.12 baseline. E8 showed the model reasons well but never triggers
self-assessment; this makes it reflect every turn and asks whether that alone is
enough to change the policy it executes.

**Contrast with E9a:** E9a *forces* the right pacing; E9b *asks* the model to
notice and self-correct. Together they bracket the reasoning→policy gap — E9a
answers "would enforcing the plan help?", E9b answers "would the model fix
itself if made to look?"

## Deferred

- **Reward/objective reshaping:** the model's depth-greed may be metric-induced
  (BALROG rewards depth; descending is one keystroke, leveling many risky
  kills). Test the enforcement (E9a) and awareness (E9b) levers first.
- **Cross-model replication** (Claude / GPT on the same harness) — is
  "reads-and-ignores" GLM-specific or general? Load-bearing for the blog thesis.
- **Continual life** — does the model learn to pace itself across deaths?
