# A strong model, NetPlay's own tools, full vision — and it still doesn't beat the game

*Draft. Prose here; the live gameplay archive is `e7_viewer.html`.*

GLM-5.2 driving NetPlay's published action surface, Prime Agent scaffold, the
whole level revealed. Across three tool-surface variants and 15 games it never
clears the early dungeon...

## How they die
<!-- taxonomy: underleveled dive → swarmed → mistimed prayer; reward is pure depth -->

## The games
See `e7_viewer.html` (open in a browser; no server needed).

## E8 — does telling the model help? (2026-08-19)

Four experiments against the NPCORE_v3 baseline (median 2.12, 5/5 died).
Viewers: `e8a_viewer.html` (descent gate), `e8b_viewer.html` (prayer hint),
`e8c_viewer.html` (unlocked doors, seed-matched), `e8d_viewer.html` (density
sweep). All E8 games replay **move-by-move** (per-step frames).

| Cell | Median | Deaths | Readout |
|---|---|---|---|
| E8a norm gate | 2.12 (=control) | 5/5 | 20 gates shown, 0 followed |
| E8b prayer hint | 6.96 (variance) | 5/5 | all prayers at full HP |
| E8c doors unlocked | 3.54 | 4/5 | kicks 18→3; only survivor of the round |
| E8d density 0.25/0.10/0.025 | 3.54/3.54/2.65 | 5/5 | route-fails 49→30→19→6, bal flat |

**The story:** verbal guidance is read and ignored — the model re-descends
straight through the human-norm line and prays at full HP despite being told
the heal band. Removing friction (doors, rooms) removes errors but not deaths.
24/25 rollouts died. Advice doesn't transfer to policy; the next lever is
scaffold-enforced constraints.
