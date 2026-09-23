# PAE on BALROG: generalization panel

25 runs, GLM-5.2, PAE layered over BALROG's unchanged naive agent.
Crafter seeds 0-9 (400-step cap); three TextWorld games seeds 0-4 (BALROG's 80-step cap).
N=10 attempts, plateau guard 4. Total billed $12.06.

## The statistic

A PAE run's attempt 1 is a fresh episode with no checkpoint, resume or directive, so it is
configuration-identical to a base-arm episode and serves as the paired base sample. This
matters because fresh-episode sampling variance at temperature 1 spans the whole metric
range: on TextWorld treasure_hunter seed 0 both arms served 443 input tokens at step 0 and
diverged only because the model sampled a different first action, one ending at 0.0 and the
other solving. Cross-arm comparison at these sample sizes is therefore uninformative. The
reported quantity is the within-run paired delta, `pae_best_progression - attempt1_progression`.

## Results

| game | n | base (attempt 1) | paired delta | SD | test |
|---|--:|--:|--:|--:|---|
| treasure_hunter | 5 | 0.000 | +1.000 | 0.000 | 5/5 sign test, p=0.031 (one-sided) |
| the_cooking_game | 5 | 0.482 | +0.341 | 0.210 | t(4)=3.64, p=0.022 |
| crafter | 10 | 0.545 | +0.114 | 0.143 | t(9)=2.52, p=0.033 |
| coin_collector | 5 | 0.800 | +0.200 | 0.447 | p=0.37, null |

Pooled across the three games with headroom, all 17 nonzero deltas are positive
(sign test p<1e-5, one-sided). `coin_collector` is a null and reported as one: four of its
five runs already solved on attempt 1, so there was no headroom, consistent with the
published frontier already scoring 100% on that game.

## Per run

| run | attempts | attempt 1 | best | delta | stop | client_retry_exhausted | billed |
|---|--:|--:|--:|--:|---|--:|--:|
| tw_treasure_hunter_s0 | 2 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.08 |
| tw_treasure_hunter_s1 | 2 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.12 |
| tw_treasure_hunter_s2 | 3 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.04 |
| tw_treasure_hunter_s3 | 5 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.11 |
| tw_treasure_hunter_s4 | 4 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.07 |
| tw_the_cooking_game_s0 | 2 | 0.529 | 1.000 | +0.471 | solved | 0 | $0.15 |
| tw_the_cooking_game_s1 | 9 | 0.118 | 0.647 | +0.529 | plateau_4 | 0 | $0.26 |
| tw_the_cooking_game_s2 | 6 | 0.235 | 0.647 | +0.412 | plateau_4 | 0 | $0.40 |
| tw_the_cooking_game_s3 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.07 |
| tw_the_cooking_game_s4 | 10 | 0.529 | 0.824 | +0.294 | attempts_exhausted | 0 | $0.34 |
| crafter_s0 | 7 | 0.545 | 0.591 | +0.045 | plateau_4 | 0 | $0.78 |
| crafter_s1 | 9 | 0.773 | 0.818 | +0.045 | plateau_4 | 0 | $2.14 |
| crafter_s2 | 8 | 0.455 | 0.545 | +0.091 | plateau_4 | 0 | $1.18 |
| crafter_s3 | 10 | 0.682 | 0.773 | +0.091 | plateau_4 | 0 | $0.69 |
| crafter_s4 | 5 | 0.682 | 0.682 | +0.000 | plateau_4 | 0 | $1.22 |
| crafter_s5 | 10 | 0.182 | 0.636 | +0.455 | attempts_exhausted | 0 | $0.90 |
| crafter_s6 | 10 | 0.591 | 0.864 | +0.273 | attempts_exhausted | 0 | $1.59 |
| crafter_s7 | 8 | 0.409 | 0.500 | +0.091 | plateau_4 | 0 | $0.40 |
| crafter_s8 | 9 | 0.591 | 0.636 | +0.045 | plateau_4 | 0 | $0.81 |
| crafter_s9 | 5 | 0.545 | 0.545 | +0.000 | plateau_4 | 0 | $0.32 |
| tw_coin_collector_s0 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.09 |
| tw_coin_collector_s1 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.06 |
| tw_coin_collector_s2 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.05 |
| tw_coin_collector_s3 | 2 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.14 |
| tw_coin_collector_s4 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.05 |

## Disclosures

Applied to every arm and recorded in each run's `env_patches`:

- **Crafter determinism.** The world-generation despawn draws from an unordered set; it is
  sorted. Without the patch, 252 of 750 post-restore continuation steps match; with it, 750 of 750.
- **Crafter seeding.** BALROG never seeded Crafter: the world seed came from the global numpy
  RNG at construction and `reset(seed=k)` never reached it, so one nominal seed gave a
  different world per process. The adapter now pins it to the episode seed.
- **TextWorld seeding.** BALROG's null seed selected the game file from a per-process counter,
  so seeds 0 and 1 were the same game. Games are now bound to the episode seed.
- **Client retry exhaustion.** Three of BALROG's four client wrappers raised after exhausted
  retries, killing an episode at a random point for a harness reason and biasing whichever arm
  plays more steps, which is PAE. Exhausted retries now return an empty completion counted as
  an invalid action and recorded separately. The counter is 0 across all 25 runs.

Restore fidelity and resumed-prompt parity were checked on every resume; see
`restore_fidelity.jsonl` and `resume_prompt_check.jsonl` per run.

## Reproducing

```
.venv-balrog/bin/python -m tools.balrog_pae.run --game crafter --task default \
    --seed 0 --attempts 10 --max-steps 400 --plateau 4 --select orchestrator --directive on
```

`tools/balrog_pae/aggregate.py` rebuilds the tables above; `runs/panel/analyze.py` reproduces
the per-game statistics. Bulky per-step traces and checkpoint blobs are untracked per `runs/.gitignore`.
