# `balrog_pae` — PAE (checkpointed exploration) on BALROG's original naive agent

Goal (tasks.md §3): show PAE generalizes by layering it on an agent we did not
design, on games other than NetHack, **adding nothing else**.

## What this is

One run = N attempts on **one** episode (one game, one task, one seed).
Attempt 1 is a plain BALROG episode. Every later attempt:

1. restores the env to a saved checkpoint,
2. restores the player's conversation history to exactly what it was there,
3. inserts **one** extra user message (the orchestrator's directive) immediately
   before the current-observation message, and
4. plays on under the same episode horizon.

Checkpoints are written every `K=10` steps and on any progress increase.

The player is `balrog.agents.naive.NaiveAgent`. `player.PAEAgent.act` is a
copy of `NaiveAgent.act` with three added lines that insert the directive; with
no directive the message list it sends is byte-identical to stock BALROG's, and
`resume_prompt_check.jsonl` proves it on every resume. The LLM client is
BALROG's own `OpenAIWrapper` (prompt formatting and answer parsing untouched);
only `_initialize_client` is overridden so the `X-Prime-Team-ID` header Prime
Inference requires gets sent.

## Files

| file | what |
|---|---|
| `prime_client.py` | GLM-5.2 over Prime Inference; token + provider-billed-cost ledger |
| `adapters.py` | per-game env + checkpoint adapters (MiniHack / Crafter / TextWorld) |
| `player.py` | `PAEAgent`, prompt-state save/load, `simulate_next_prompt` (the resume proof) |
| `orchestrator.py` | the separate GLM-5.2 chat: ledger in, `{checkpoint, directive}` out |
| `pae.py` | the loop, the archive, the ablation switches, the metrics |
| `run.py` | CLI |
| `selftest_offline.py` | full loop with a scripted client — no LLM calls, no money |
| `cost_projection.py` | per-task projection for the full study |
| `../balrog_ckpt/` | the earlier checkpoint-feasibility work this reuses (see its REPORT.md) |

## Ablation switches (the three used on NetHack)

| switch | values | meaning |
|---|---|---|
| `--select` | `fixed` \| `orchestrator` | `fixed` = latest resumable checkpoint of the previous attempt (the "pre-death" rule) |
| `--directive` | `on` \| `off` | `off` = no extra message at all |
| `--blind` | flag | the extra message is the neutral `"Continue playing."` |

`--base-only` runs a single unchanged BALROG episode (the leaderboard protocol).
Stop rules: `--attempts N` (default 10) and `--plateau P` attempts with no
frontier advance.

## Outputs (run dir)

`attempts.jsonl`, `archive/cN/{meta.json,checkpoint.pkl}`, `summary.json`,
`selection.jsonl`, `orchestrator_rounds.jsonl`, `attempts/aNNN/trace.jsonl`,
plus two audit streams that are the point of the whole thing:

* `restore_fidelity.jsonl` — did the restore reproduce the exact observation?
* `resume_prompt_check.jsonl` — did the resumed player's message list equal the
  checkpoint's stored next-prompt, plus exactly the directive turn?

Both are asserted at runtime (`--no-verify-restore` to downgrade to logging).

## Restore method per game

| game | method | verified |
|---|---|---|
| MiniHack | replay of the action prefix after re-seeding with the episode's **effective** `(core, disp, reseed)` seeds; BALROG's sticky `done` latch cleared; the evaluator's invalid-action feedback rewrite re-applied during replay | byte-identical observations (full obs digest incl. `glyphs`/`tty_chars`/`blstats`) on every resume in every smoke |
| Crafter | pickle of the inner `crafter.Env` (minus gym's `_np_random`) restored in place, plus the `_balance_chunk` determinism patch from `balrog_ckpt/crafter_ckpt.py` | canonical `state_digest` identical on restore |
| TextWorld | `.z8` (`the_cooking_game`): Jericho `get_state`/`set_state` + deepcopy of every wrapper's mutable state. `.ulx` (`treasure_hunter`, `coin_collector`): replay (no state export exists) | digest identical on restore (`.z8`) |

## Engine patches (disclosed)

`summary.json` carries `env_patches`. MiniHack and TextWorld run stock
(`env_patches: []`). **Crafter runs with one patch in every arm, `--base-only`
included**: `crafter_balance_chunk_sorted` sorts the per-chunk object set before
Crafter's despawn logic indexes into it. Stock Crafter is not reproducible
across processes because that set is iterated in `id()`-hash order, and a
pickle-restored copy diverges from the original within ~30 steps (1/5 seeds pass
unpatched, 5/5 patched — `balrog_ckpt/REPORT.md`). The patch reorders an
already-arbitrary choice; it leaves the RNG stream, rewards and achievements
alone. Both arms get it so base and PAE play the same game; the paper states it.

## Progress accounting

The reported metric is always **BALROG's own** `env.get_stats()["progression"]`.
On MiniHack that is binary (1.0 only if the task is solved), which is far too
sparse to choose a checkpoint from, so each adapter also exposes an
`aux_progress` used **only** to trigger checkpoints and to fill the
orchestrator's ledger — never reported:

* MiniHack: number of distinct `(dlvl, x, y)` cells seen
* Crafter: achievements unlocked (`score_tracker`; BALROG's progression is this / 22)
* TextWorld: `score / max_score` (same as progression)

Step accounting under the caps (tasks.md §3.5 open question) is reported both
ways: `committed_steps` (steps of the committed trajectory, i.e. the BALROG
horizon — a checkpoint at step 60 leaves 40) and `total_env_steps` (every env
step including replays). A checkpoint at the cap is not resumable and is
filtered out of both the fixed rule and the orchestrator's ledger.

## Measured (GLM-5.2, 2026-09-22)

| run | arm | attempts | ckpts | committed steps | LLM calls | in tok | out tok | list $ | billed $ | progression |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `runs/mh_quest_easy_s0_base` | base (unchanged BALROG) | 1 | 6 | 7 | 7 | 9,668 | 1,385 | 0.0216 | 0.0062 | 0.000 |
| `runs/mh_quest_easy_s0_pae` | PAE fixed + directive | 3 | 11 | 17 | 22 | 63,322 | 5,495 | 0.1241 | n/a | 0.000 |
| `runs/mh_quest_easy_s0_pae_orch` | PAE orchestrator + directive | 4 | 20 | 13 | 31 | 53,301 | 13,734 | 0.1486 | 0.0572 | 0.000 |
| `runs/abl_directive_off` | PAE fixed, no directive | 2 | 5 | 6 | 7 | 9,399 | 1,706 | 0.0227 | 0.0089 | 0.000 |
| `runs/abl_blind` | PAE fixed, neutral turn | 2 | 7 | 10 | 11 | 19,622 | 3,046 | 0.0450 | 0.0132 | 0.000 |
| `runs/crafter_s0_pae` | PAE fixed + directive | 2 | 10 | 60 (capped) | 70 | 104,046 | 11,630 | 0.2165 | 0.0834 | **0.091** (2/22) |

Per-step: MiniHack input grows ~235 tok/step until the 16-observation window
saturates (≈4.5K/step steady state), output 200-440. Crafter 1.49K in / 170 out
per step. Prime's own `usage.cost` came in at 0.29-0.39x the list price from
`model_prices.json`.

## Cost projection

`python tools/balrog_pae/cost_projection.py --r 3` (env-step counts from
tasks.md §3.4, token/step measured here):

```
task                           eps  steps   in (M)  out (M)   base $    PAE $  all arms $
MiniHack Quest-Easy              5    470     2.12     0.14     3.94    12.41       28.96
MiniHack Quest-Medium            5    240     1.08     0.07     2.01     6.34       14.79
MiniHack CorridorBattle-Dark     5    250     1.12     0.07     2.10     6.60       15.40
MiniHack Boxoban-Medium          5    450     2.02     0.14     3.77    11.88       27.72
MiniHack Boxoban-Hard            5    440     1.98     0.13     3.69    11.62       27.11
Crafter default                 10   2700     4.02     0.46     8.42    26.51       61.86
TextWorld treasure_hunter        5    400     0.60     0.12     1.50     4.74       11.06
TextWorld the_cooking_game       5    400     0.60     0.12     1.50     4.74       11.06
TextWorld coin_collector         5    400     0.60     0.12     1.50     4.74       11.06
TOTAL (list price)                           14.15     1.37    28.44    89.58      209.02
TOTAL (observed billed ~0.36x)                                 10.24    32.25       75.25
```

`all arms` = base + PAE + one token-matched sampling control, +5% orchestrator
overhead. At r=5 the same table totals $328 list / $118 billed. Both fit the
~$400 remaining budget; the MiniHack-only subset is $114 list / $41 billed at r=3.

## Feasibility notes

**MiniHack.** Works today; restore is exact. Two things to decide before the
real runs: (i) the fixed "pre-death" rule picks the checkpoint 1-2 steps before
death, and on Quest-Easy the agent simply walks into the same lava again — a
lookback (e.g. the checkpoint ≥5 steps before the end) is needed, which is
exactly tasks.md §1's open question; (ii) replay cost is ~0.25 ms/step, so
restore is free relative to the LLM call. The engine-fork path (10x faster) is
still blocked by the `load_mirror` heap corruption on Quest levels documented in
`balrog_ckpt/REPORT.md`; the replay path avoids it entirely.

**Crafter.** Works today. "Progress" is achievements/22, which is dense enough
to drive both the checkpoint trigger and the orchestrator. Budget risk from
tasks.md §3.4 is real: base Crafter episodes end on death at 150-350 steps, and
PAE resuming before death will push toward the 2,000-step cap — the projection
above assumes base-length episodes, so cap the per-attempt horizon explicitly
(`--max-steps`) if the PAE arm starts running long.

**TextWorld.** Only `the_cooking_game` (.z8) has real save/restore; the two
Glulx games must use replay, which is verified-deterministic but costs a full
`reset` + prefix (~0.6 ms/step, 17-50 ms reset) — still negligible. Progression
is `score/max_score`, the best-shaped signal of the three. The 80-step cap means
resumes have little room late in an episode.

## Reproduce

```bash
cd /root/nld/gen-pae
.venv-balrog/bin/python -m tools.balrog_pae.selftest_offline --game minihack --run-dir /tmp/st   # free
.venv-balrog/bin/python -m tools.balrog_pae.run --game minihack --task MiniHack-Quest-Easy-v0 \
    --seed 0 --attempts 3 --select fixed --directive on --run-dir runs/x
.venv-balrog/bin/python -m tools.balrog_pae.run --game minihack --task MiniHack-Quest-Easy-v0 \
    --seed 0 --base-only --run-dir runs/x_base
```

The venv is `/root/nld/gen-pae/.venv-balrog` (BALROG `b7afe79` installed editable
from `/root/nld/gen-pae/BALROG`, Boxoban levels and `tw_games` in place). Neither
is tracked by git (see `.git/worktrees/gen-pae/info/exclude`).
