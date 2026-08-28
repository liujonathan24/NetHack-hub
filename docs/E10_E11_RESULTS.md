# E10/E11 — the honest-harness re-baseline and the gate reruns

## Why these experiments exist

A node-by-node read of a raw E7 transcript (NPCORE_v3 seed 2) found the harness
was systematically misinforming the model: SKILL.md examples called tools that
don't exist (`explore_and_descend`, unprefixed `move_to`), every tool's JSON
schema was empty while the docs said to inspect it, `np_press_key`'s
description denied the descend key `>` works, `help(nethack)` leaked eval
plumbing, the system prompt was delivered twice, and the wiki commanded
phantom tools. Separately, the E9/E10-era launch scripts dropped
`tune.reveal_map=1.0` and `auto_dismiss="false"` from cells that claimed to
replicate the E7 control (caught by reading the traces: partial maps).

Fixes: the harness honesty pass (`fd8aa13`, `9b8d5a4`) + launcher restoration
(`0458275`). Every cell below was verified pre-launch: a scripted/Haiku mock
play confirmed full vision (first `request_map` shows the whole level),
auto-dismiss off (prompts stay open), the exact 10-tool np_core surface, and
correct resolved configs read back from disk.

## E10 — clean re-baseline (NPCORE_v3, 3 runs x 5 seeds, GLM-5.2, 200 calls)

| run | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| r1 | 4.85 (dl7/xl5) | 2.65 (dl5/xl4) | 2.65 (dl5/xl4) | **16.13 (dl11/xl4)** | 4.85 (dl7/xl2) |
| r2 | 2.08 (dl3/xl3) | **9.77 (dl9/xl3)** | 1.75 (dl3/xl1) | 2.65 (dl5/xl1) | **9.77 (dl9/xl4)** |
| r3 | 2.12 (dl4/xl1, survived) | 4.85 (dl7/xl1) | 3.54 (dl6/xl4) | **9.77 (dl9/xl4)** | 2.65 (dl5/xl1) |

**median 3.54 · mean 5.34 · 14/15 died · ceiling dl11** (BALROG x100)
vs the old-docs E7 pooled baseline: median 2.12 · mean 2.44 · 13/15 died ·
ceiling dl7.

**Harness honesty roughly doubled measured performance** on identical seeds,
model, and budget (mean +119%), and moved the depth ceiling from dl7 to dl11
with four games at dl9+. Survival did not move. The distribution is bimodal:
half the games still die shallow exactly as before; honest docs unlocked the
ceiling, not the floor. Opening discovery cost fell from ~8 probing turns to 1
(read SKILL.md, play).

Two flagged games were rerun as fresh single-seed rollouts (`run_e10_patch.sh`)
and folded in; one wedge-truncated attempt is archived under
`INVALID_wedge_stubs/`, and the fog-configured first attempt under
`INVALID_nofog_r1__prime_agent/`.

## E11 — descent gates on the honest harness (5 seeds each)

Gate semantics were live-verified pre-launch (fire-once + pass-through for
norm; block-without-override at XL<norm and unlock at XL>=norm for enforce;
`np_move_to` proven NOT to bypass the gate).

**E11a `descent_gate=norm` (ignorable):** median 3.54 · mean 9.78 · 4/5 died ·
31 gate messages. Includes the **project-record game: Dlvl 15, bal 30.88**
(seed 4) and a surviving dl9/XL6 game (seed 2).

**E11b `descent_gate=enforce` (forced):** median 2.65 · mean 6.13 · 4/5 died ·
12 blocks fired. Includes a surviving dl11/XL6 game (seed 4). Seed 1 died on
Dlvl 1 with zero blocks fired — the shallow-death mode is gate-proof.

### Leveling metrics (the real effect of the gates)

| metric | E10 | E11a | E11b |
|---|---|---|---|
| mean XL | 2.80 | 3.80 | 3.60 |
| XL/Dlvl | 0.47 | 0.61 | **0.74** |
| Dlvl-XL gap | +3.53 | +3.80 | **+2.20** |
| XL vs human norm at final depth | -1.73 | -1.60 | **-0.80** |
| XL per 1000 game turns | 5.27 | 5.53 | **7.57** |

The forced gate measurably changes the leveling *policy* (halves the pacing
deficit vs human norms, +44% leveling rate — the model grinds when blocked)
without moving the median outcome. Across all 25 games, the only two that were
deep AND alive at the end are the only two XL-6 games. Deaths that remain are
(a) combat losses at now-appropriate depth and (b) pre-descent shallow deaths
no descent mechanic can touch.

## Tool-repetition audit (why budgets bleed)

67% of all 2,898 calls sit inside consecutive same-tool clusters. Dominant
causes, all in the vendored NetPlay layer:

- `np_move_to` interrupted en route with no progress accounting (468 calls);
  zero-step failures are now 100% explained by the `[why: ...]` clause added
  in the honesty pass (220 calls).
- `np_explore_level` interrupts on trivia — item drops, pet noise (291).
- `np_rest(count=N)` delivers ~3 turns regardless of N: ambient "You hear..."
  messages count as "something happened" (74).
- `np_melee_attack` fails "Unable to reach" on 24% of calls without reporting
  where the target moved (79).
- **`RawKeyPress` has no `-` key** (nor `'`): the "write with what? [- bch]"
  prompt cannot be answered, so dust-engraving Elbereth — the game's canonical
  survival technique — is structurally impossible on this surface. Observed
  live: the model attempted `E` then `-`, got "Unable to press the given
  key -", and abandoned the plan.

Conclusion: NetPlay's skill layer under-reports (interruption and failure
telemetry) and under-equips (missing keys/affordances) — the model's policy
under this starvation is consistently sensible; the tools are what flail.
Fixes tracked separately.
