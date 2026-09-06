# E16 fog-of-war sweep: Go-Explore at the observability floor, five seeds

Five runs of the 30-attempt Go-Explore protocol on tier `[e16_gewiki_norb_fog]`
— `[e16_gewiki_norb]` with `tune.reveal_map = 0.0` and nothing else, verified as
exactly one differing key against the registry and one experimental knob against
the served `config.toml` of every run. One replica per game seed 0–4. The
matched control for each seed is the unfogged run of the same tier on the same
seed: `e16_s{0,2,3,4}_r1` and the three `treesmoke11*` replicas (seed 1).

Launchers are vendored here verbatim: `fogrun30.sh` ran seed 1; `fogsweep30.sh`
is its seed-parameterised form and ran 0/2/3/4. Every environment variable is
identical across the five runs (`E16_MAX_ATTEMPTS=30`, `E16_STALL=12`,
milestones off); the tier registry hashed to `1f65e8a46431cd0b` for all five,
the same `tier_registry_sha256_16` each run recorded.

## Headline

Fog **halves BALROG-min (0.53×) while barely touching BALROG-max (0.89×)**.
A blind agent still finds the stairs; what it cannot do is build a character on
the way down — fog runs top out at XL 5–8 where their sighted controls reach
8–10, on every seed.

## Per seed, matched attempt cuts

Each seed is compared at the fog run's own attempt count, cutting the control's
archive to the same budget with `tools/cli_harness_eval/e16_lineage.py --cut N`.

| seed | cut | FOG min / max | Dlvl, XL | fog stop | CTRL min / max | Dlvl, XL |
|---|---|---|---|---|---|---|
| 0 | 30 | 5.08 / **32.49** | 16, 7 | cap | 11.70 / 16.13 | 10, 9 |
| 1 | 26 | 7.45 / **40.83** | 22, 8 | stall | 12.35* / 37.55* | 22, 10 |
| 2 | 21 | 5.08 / 12.56 | 10, 7 | stall | 11.70 / 34.01 | 17, 8 |
| 3 | 30 | 5.08 / 37.90 | 20, 5 | cap | 7.45 / 37.90 | 20, 8 |
| 4 | 30 | 7.45 / 30.88 | 15, 8 | cap | 11.70 / 39.29 | 21, 8 |

\* seed-1 control is the mean of the three `treesmoke11*` replicas at cut 26.

## Aggregates

|  | BALROG-min | BALROG-max |
|---|---|---|
| Fog (n=5) | **6.03 ± 0.58** (median 5.08) | **30.93 ± 4.76** (median 32.49) |
| Control (n=7) | 11.38 ± 1.33 (median 11.70) | 34.66 ± 3.19 (median 37.90) |
| ratio of means | **0.53×** | **0.89×** |

## Three observations

1. **Depth survives blindness; experience does not.** On two seeds fog matches
   or beats control depth (seed 1 ties at Dlvl 22; seed 0 out-descends its
   control 16 vs 10). Seed 3 is the cleanest single-seed statement: identical
   Dlvl 20 and identical BALROG-max 37.90 to its control, at XL 5 instead of 8.
2. **Fog BALROG-min sits on a two-value lattice.** Every seed lands on exactly
   5.08 or 7.45 — the XL axis binding at 7 or 8. The gap to control is real but
   quantised; do not quote it to two decimals without saying so.
3. **Fog searches die younger.** Two stalls (21, 26 attempts) and three cap-outs
   against controls that ran 30–100 attempts. Seed 2 is the collapse case:
   stalled at 21 having reached Dlvl 10 against its control's 17.

## Caveats

- One replica per seed; the control pool (n=7) triple-weights seed 1.
- **The measured gap is a lower bound.** The unfogged controls are themselves
  partially fogged by the restore-drops-tune bug (#52): 18–28% of their
  attempts resumed with the reveal overlay silently off. Fog arms have no
  overlay to lose and are unaffected, so the bug degrades the control toward
  the treatment. A clean control (post PR #53) can only widen the gap.
- BALROG-min values here are best-of-archive per run (Go-Explore lineage
  semantics), not single-life scores; compare against other Go-Explore runs,
  not against single-life arms.

## Files

- `wrap_fog_full.json` — fog and control lineages, seeds 0/3/4, cut 30
- `wrap_s1.json` — seed 1, fog + three control replicas, cut 26
- `wrap_s2.json` — seed 2, fog + control, cut 21
- `fog5.json` — all five fog lineages at cut 30 (uniform-cut view)
- `run_summaries.json` — per-run stop reason, attempts, deaths, checkpoints,
  calls, wall clock, best state
- `fogrun30.sh`, `fogsweep30.sh` — the launchers, verbatim

Raw archives: `/root/nld/e16_runs/e16_fog_s{0..4}_r1` (~1 GB total, not in
git). Live explorer over all arms:
https://claude.ai/code/artifact/9941775c-36f3-4a33-990e-f5f8f4e65b34
