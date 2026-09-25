# BALROG transfer results, cost and Pareto aggregates (2026-09-25)

Everything here is derived from existing runs; no new runs were launched for this bundle.
Model: GLM-5.2 (Prime Inference). PAE = checkpointed exploration over BALROG's unchanged
naive agent, 10 attempts; base = attempt 1 of the same run; control = best of 10 independent
fresh episodes with no checkpoint, resume, or directive.

## Suite means (equal weight per game)

| suite | runs | base | PAE | delta | one-sided sign test |
|---|---|---|---|---|---|
| Crafter | 10 | 0.545 | 0.659 | +0.114 | 8/8, p=0.004 |
| TextWorld (3 games) | 30 | 0.420 | 0.933 | +0.514 | 19/19, p=1.9e-6 |
| MiniHack (8 tasks, Boxoban entered at 0) | 25 | 0.150 | 0.325 | +0.175 | 7/7, p=0.008 |

MiniHack seeds per task: CorridorBattle-Dark 5, Corridor-R3 5, MazeWalk-15x15 5, MazeWalk-9x9 5,
Quest-Easy 4, Quest-Medium 1. Boxoban-Medium excluded (27/77 steps defaulted after exhausted
client retries); Boxoban-Hard not launched. See `minihack/RESULTS_NOTES.md`.

## Matched control (best of 10 fresh episodes) vs PAE best of 10

| game | control | PAE | control $/run | PAE $/run |
|---|---|---|---|---|
| treasure_hunter | 0.900 | 0.900 | 0.130 | 0.078 |
| coin_collector | 1.000 | 1.000 | 0.176 | 0.138 |
| the_cooking_game | 0.806 | 0.900 | 0.433 | 0.233 |
| Crafter | 0.723 | 0.659 | 3.94 | 1.00 |
| CorridorBattle-Dark | 0.6 | 0.8 | 0.61 | 0.40 |
| Corridor-R3 | 0.6 | 0.6 | 2.70 | 1.88 |

Costs are provider-billed (`tokens.billed_by_provider_usd`).

## NetHack cost and BAL_min / BAL_max

NetHack runs carry no billing field. They are priced at GLM-5.2 $1.54/M fresh input,
$0.154/M cached input, $4.84/M output, plus the orchestrator's `cost_usd_reported`. That rate
card reproduces the three wallet-measured 30-attempt PAE runs (treesmoke11, _r2, _r3) within
6% ($293.50 predicted vs $292.54 wallet).

| arm | cost | BAL_max | BAL_min |
|---|---|---|---|
| PAE, fog of war (e16_fog_s0-4_r1, 137 attempts) | $252.71 | 30.93 | 6.03 |
| Claude Code single lives (15) | $21.92 | 11.05 | 1.57 |

Leaderboard BAL_max recomputed from per-episode dlvl/xp lists equals the published
progression for all nine models checked (`aggregate/nh2.py`, `aggregate/nh_minmax.json`).

## Pareto aggregates

Mean progression = equal-weight mean of suite means (NetHack BAL / 100). Cost = sum of
suite costs, each normalised to the leaderboard's episode count (MiniHack 40, TextWorld 30,
Crafter 10, NetHack 5). Leaderboard cost = list price on published token totals.

- `aggregate/aggregate.json`: all four / held-out three, top-10 leaderboard models incl. the
  Sept 2026 batch (GPT-6 Astra, GPT-5.6 Sol/Luna/Terra, Opus 5).
- `aggregate/aggregate_old.json` (four suites) and `aggregate/aggregate_old_heldout.json`
  (MiniHack, TextWorld, Crafter): leaderboard submissions dated before 2026-09-17 with all
  four suites and a list price. Gemini-3 Pro is priced at Gemini 3.1 Pro preview's $2/$12
  per M (not in the price table).

Held-out-three frontier (pre-2026-09-17 models), cheapest first: Llama-3.2-1B, Llama-3.2-3B,
Mistral-Nemo, Gem-2.5-Flash, GLM-5.2 base ($7.44, 0.372), Gem-3-Flash, Gem-3.1-Pro,
Gem-3-Pro, Gem-3.1-Pro-T ($40.16, 0.527), GLM-5.2 PAE ($49.23, 0.639).

## Figures

`figures/` holds the paper-style PDFs; generators are on branch `paper/figures`
(`paper/figs/plots/fig_balrog_pareto*.py`, `fig_balrog_aggregate.py`).
