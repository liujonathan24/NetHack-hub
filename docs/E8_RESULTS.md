# E8 results — planning injection, mechanic guidance, doors, density

Date: 2026-08-19. Model z-ai/glm-5.2, Prime Agent scaffold, np_core surface,
full vision, 200 calls, 5 seeds/cell. Control: NPCORE_v3 (median bal_max 2.12,
5/5 died, corpses XL1-3 @ Dl3-7). Full per-move telemetry (frames + glyph ids
from E8c onward) on the data/e8-gameplay branch.

## Verdict table

| Cell | Knob | Median bal_max | Deaths | Cell-specific readout |
|---|---|---|---|---|
| control (NPCORE_v3) | — | 2.12 | 5/5 | mv_fails 49 |
| E8a NORM | descent_gate=norm | **2.12** | 5/5 | 20 gates shown, 15 immediate re-descends, 5 non-grinding waits, **0 XL responses** |
| E8a NORM_v1_ungated | (gate bug; control replicate) | 2.12* | 5/5 | gate never fired; fresh-seed control |
| E8b PRAYER | mechanic_hints=prayer | 6.96 | 5/5 | 3 prayers, **all at full HP, 0 in heal band** |
| E8c DOORS | locked_door=0 (seed-matched) | 3.54 | 4/5 | kicks 18→3, freeze 43→32; per-seed 3↑ 2↓ |
| E8d D025 | room_density=0.25 | 3.54 | 5/5 | mv_fails 30 |
| E8d D010 | room_density=0.10 | 3.54 | 5/5 | mv_fails 19 |
| E8d D0025 | room_density=0.025 (~1 room) | 2.65 | 5/5 | mv_fails 6 |

\* aggregate of the ungated run's completed seeds.

## Findings

1. **Verbal guidance is read and ignored (E8a + E8b).** The planning injection
   put the human norm in front of the model at every descent — 20 times — and
   the model re-descended immediately 15/20 times and NEVER once raised XL in
   response; median identical to control (2.12). The prayer mechanics hint
   changed nothing about prayer behavior: every prayer in the cell was cast at
   FULL HP, zero in the heal band. PRAYER's higher median (6.96) rides on seed
   variance (spread 1.54-20.61), not on the hinted behavior — the targeted
   behavior did not move. Relative to NetPlay's observation that guidance
   *hurt* (a verbal, point-in-time comparison only): here guidance neither
   helps nor hurts — it is simply not incorporated into policy.
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
