# E8 results — planning injection, mechanic guidance, doors, density

Date: 2026-08-19. Model z-ai/glm-5.2, Prime Agent scaffold, np_core surface,
full vision, 200 calls, 5 seeds/cell. Control: NPCORE_v3 (median bal_max 2.12,
5/5 died, corpses XL1-3 @ Dl3-7). Full per-move telemetry (frames + glyph ids
from E8c onward) on the data/e8-gameplay branch.

## Verdict table

| Cell | Knob | Median bal_max | Deaths | Cell-specific readout |
|---|---|---|---|---|
| control (NPCORE_v3) | — | 2.12 | 5/5 | mv_fails 49 |
| E8a NORM | descent_gate=norm | **2.12** | 5/5 | 20 gates; **7 engaged (reasoned about XP, chose to descend anyway), 13 repeated without engaging**; 0 behavior change |
| E8a NORM_v1_ungated | (gate bug; control replicate) | 2.12* | 5/5 | gate never fired; fresh-seed control |
| E8b PRAYER | mechanic_hints=prayer | 6.96 | 5/5 | **3/3 prayers correctly at critical HP** (control already 3/4); hint had ~nothing to fix; median = variance |
| E8c DOORS | locked_door=0 (seed-matched) | 3.54 | 4/5 | kicks 18→3, freeze 43→32; per-seed 3↑ 2↓ |
| E8d D025 | room_density=0.25 | 3.54 | 5/5 | mv_fails 30 |
| E8d D010 | room_density=0.10 | 3.54 | 5/5 | mv_fails 19 |
| E8d D0025 | room_density=0.025 (~1 room) | 2.65 | 5/5 | mv_fails 6 |

\* aggregate of the ungated run's completed seeds.

## Findings

1. **Verbal guidance is ENGAGED WITH but rarely changes behavior (E8a + E8b).**
   The planning injection was read and reasoned about: on 7 of 20 gates the
   model explicitly weighed the XP tradeoff and chose to descend anyway with a
   defensible argument ("grid bugs give ~1 XP; I need 10 for XL 2; better to
   dive and fight stronger monsters"). It is not blind ignoring — it is
   *considered and overruled* (and the model's counter-argument is partly
   correct: early monsters are poor XP; human winners arrive better-leveled via
   routing, not by grinding grid bugs). Behavior didn't change; median == control.
   For prayer, the picture is the opposite of what I first reported (see the
   CORRECTION below): prayer timing was **already good in the control** (3 of 4
   prayers at critical HP) and the hint made it 3 of 3 — a 1-prayer delta across
   7 total, far too small to explain the 6.96 vs 2.12 median gap, which is seed
   variance (spread 1.54-20.61). So the hint changed a behavior that was barely
   broken. Net across both: telling the model things does not reliably move
   policy — sometimes because it reasons to a different choice, sometimes
   because the behavior was already correct. (NetPlay's guidance-*hurt* result
   is a verbal, point-in-time comparison only.)

   > **CORRECTION (found by re-reading the traces).** My first-pass aggregation
   > read the pray-turn record's `hp`, which is the **post-heal** value, and so
   > reported "all prayers at full HP, hint ignored." That was a measurement
   > bug, not a model behavior: the true pre-prayer HP (prior turn) was 2/16,
   > 3/43, 5/16 — all critical, all correctly timed, each healing to full, with
   > explicit reasoning ("43/7 ≈ 6.1, I'm at 3, pray!"). The model read the
   > mechanic (16 heal-band references in reasoning) and applied it right. The
   > corrected finding is above; `tools/agent_capability_audit.py` now extracts
   > pre-prayer HP so this cannot recur.
2. **Removing execution obstacles helps a little, and honestly (E8c).**
   Unlocking doors (seed-matched, RNG-preserved layouts) removed the kick tax
   (18→3) and produced the round's only budget survivor; median 3.54 vs 2.12,
   but per-seed movement is mixed (3 up, 2 down) — a small real control win
   inside seed noise.
3. **Navigation was never the binding constraint (E8d).** The density
   dose-response is the cleanest curve of the round: route failures fall
   monotonically 49→30→19→6 as rooms disappear, and bal_max does NOT rise with
   them (3.54, 3.54, 2.65). With one big room and essentially zero pathfinding
   friction, the agent still dies at the same depths. Perception/control
   simplification does not unlock depth.
4. **Deaths: 24 of 25 rollouts died** (the sole survivor: DOORS seed 1,
   budget-exhausted). Survival remains the wall, and neither telling the model
   how to survive (E8b) nor when to consolidate (E8a) changes its behavior.

## Implication for the next round

The lever is not information (E8a/E8b null) and not mechanical friction
(E8c small, E8d flat). What remains untested on the survival wall:
**hard constraints** (actually blocking descent under-XL rather than advising),
**in-loop control changes** (single-swing melee vs pursue-until-dead), and the
deferred **E8e infinite-life** planning probe. Advice doesn't transfer to
policy; scaffold-enforced policy might.

---

## POST-HOC VALIDITY CAVEAT — RETRACTED (2026-08-21, same day)

An earlier addendum here claimed every E8 cell ran without `tune.reveal_map`
and `auto_dismiss=false`. That was a false alarm: the check grepped config.toml
files that do not exist in the E8 output dirs (that launcher era wrote none).
The authoritative resolved configs in each cell's eval.log carry BOTH knobs,
and the first request_map observation in the transcripts shows the full level.
**E8 ran with full vision, matching its control; its comparative findings
stand.** The remaining (true) caveat: E8 predates the harness-honesty pass
(fd8aa13), so its ABSOLUTE numbers sit on the old-docs harness -- consistent
with the E7 control it was measured against, but ~2x below the honest-harness
E10 baseline. Compare E8 cells to E7 controls only, never to E10+ cells.
