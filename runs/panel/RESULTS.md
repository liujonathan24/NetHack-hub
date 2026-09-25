# PAE on BALROG: generalization panel

40 runs, GLM-5.2, PAE layered over BALROG's unchanged naive agent.
Crafter seeds 0-9 (400-step cap); three TextWorld games seeds 0-9 (BALROG's 80-step cap,
matching BALROG's own default of 10 episodes per TextWorld task).
N=10 attempts, plateau guard 4. Total billed $14.51 ($12.06 for the first 25 runs,
$2.45 for TextWorld seeds 5-9).

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
| treasure_hunter | 10 | 0.100 | +0.800 | 0.422 | 8+/0- sign test, p=0.0039 (one-sided) |
| the_cooking_game | 10 | 0.559 | +0.341 | 0.293 | t(9)=3.68, p=0.0025 |
| crafter | 10 | 0.545 | +0.114 | 0.143 | t(9)=2.52, p=0.033 |
| coin_collector | 10 | 0.600 | +0.400 | 0.516 | 4+/0- sign test, p=0.0625 — **not separated from 0** |

The two binary games (`treasure_hunter`, `coin_collector` are won/not-won) are tested with an
exact one-sided sign test over the discordant pairs; `the_cooking_game` is graded out of 17, so
a paired t-test on the deltas is used there. Pooled across all four games, all 27 nonzero
deltas are positive.

`coin_collector` is the one task not separated from zero. Widening it from 5 to 10 seeds did
change its character: at n=5 four of five runs solved on attempt 1 and the task looked
saturated, but at n=10 four of ten attempt-1 episodes fail, PAE recovers all four, and the
delta doubles to +0.400. It remains short of significance only because an exact sign test on
4 discordant pairs cannot go below p=0.0625. The earlier "no headroom" reading is retired;
the honest statement is a positive but underpowered result.

### Seed-to-game binding

BALROG leaves `envs.env_kwargs.seed` null and selects the game file from a per-process
counter, so its episodes are not seed-addressable. The adapter sets the seed, which routes
through `TextWorldFactory.get_textworld_env` to `env_ids[task][seed % 25]`. BALROG ships 25
games per task, so seeds 0-9 give 10 distinct games per task — confirmed by constructing each
env and reading its bound game file:

| seed | treasure_hunter | the_cooking_game | coin_collector |
|--:|---|---|---|
| 0 | seed_10033.ulx | cooking_item5_seed_10980.z8 | level_220_seed_100.ulx |
| 1 | seed_10915.ulx | cooking_item5_seed_11996.z8 | level_220_seed_1171.ulx |
| 2 | seed_14115.ulx | cooking_item5_seed_12274.z8 | level_220_seed_12089.ulx |
| 3 | seed_16404.ulx | cooking_item5_seed_12896.z8 | level_220_seed_15858.ulx |
| 4 | seed_18762.ulx | cooking_item5_seed_13009.z8 | level_220_seed_16706.ulx |
| 5 | seed_20085.ulx | cooking_item5_seed_15394.z8 | level_220_seed_20174.ulx |
| 6 | seed_21726.ulx | cooking_item5_seed_16632.z8 | level_220_seed_23258.ulx |
| 7 | seed_24649.ulx | cooking_item5_seed_16877.z8 | level_220_seed_24972.ulx |
| 8 | seed_27903.ulx | cooking_item5_seed_17067.z8 | level_220_seed_34290.ulx |
| 9 | seed_30233.ulx | cooking_item5_seed_19196.z8 | level_220_seed_38603.ulx |

30 of 30 distinct within task, no collisions.

### Against the archived leaderboard

Attempt 1 is an unmodified naive-agent episode, so its mean is comparable to the archived
naive-agent submissions in `balrog-experiments/submissions/LLM/*/textworld/`, which also
report 10 episodes per task.

| task | our attempt-1 base | luna-max | sol-max | astra-max |
|---|--:|--:|--:|--:|
| treasure_hunter | 10.0% | 20.0% | 40.0% | 40.0% |
| the_cooking_game | 55.9% | 34.7% | 50.6% | 23.5% |
| coin_collector | 60.0% | 0.0% | 100.0% | 100.0% |

GLM-5.2's naive baseline is below all three on treasure_hunter, above all three on
the_cooking_game, and inside the (very wide) range on coin_collector.

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
| tw_treasure_hunter_s5 | 4 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.10 |
| tw_treasure_hunter_s6 | 2 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.07 |
| tw_treasure_hunter_s7 | 4 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.12 |
| tw_treasure_hunter_s8 | 5 | 0.000 | 0.000 | +0.000 | plateau_4 | 0 | $0.07 |
| tw_treasure_hunter_s9 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.02 |
| tw_the_cooking_game_s5 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.06 |
| tw_the_cooking_game_s6 | 4 | 0.118 | 1.000 | +0.882 | solved | 0 | $0.25 |
| tw_the_cooking_game_s7 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.11 |
| tw_the_cooking_game_s8 | 10 | 0.294 | 0.882 | +0.588 | plateau_4 | 0 | $0.47 |
| tw_the_cooking_game_s9 | 5 | 0.765 | 1.000 | +0.235 | solved | 0 | $0.20 |
| tw_coin_collector_s5 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.09 |
| tw_coin_collector_s6 | 1 | 1.000 | 1.000 | +0.000 | solved | 0 | $0.08 |
| tw_coin_collector_s7 | 2 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.18 |
| tw_coin_collector_s8 | 5 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.46 |
| tw_coin_collector_s9 | 2 | 0.000 | 1.000 | +1.000 | solved | 0 | $0.17 |

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
  an invalid action and recorded separately. The counter is 0 across all 40 runs.

Restore fidelity and resumed-prompt parity were checked on every resume; see
`restore_fidelity.jsonl` and `resume_prompt_check.jsonl` per run.

## Reproducing

```
.venv-balrog/bin/python -m tools.balrog_pae.run --game crafter --task default \
    --seed 0 --attempts 10 --max-steps 400 --plateau 4 --select orchestrator --directive on

.venv-balrog/bin/python -m tools.balrog_pae.run --game textworld --task treasure_hunter \
    --seed 5 --attempts 10 --plateau 4 --select orchestrator --directive on
```

`runs/panel/launch_tw10.sh` is the batch launcher for TextWorld seeds 5-9;
`runs/panel/tw10_report.py` prints the 10-seed TextWorld tables and the leaderboard
comparison above.

`tools/balrog_pae/aggregate.py` rebuilds the tables above; `runs/panel/analyze.py` reproduces
the per-game statistics. Bulky per-step traces and checkpoint blobs are untracked per `runs/.gitignore`.
