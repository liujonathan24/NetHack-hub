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
| `cost_projection.py` | per-task projection for the full panel, with per-row provenance |
| `aggregate.py` | base mean vs PAE mean-of-best across seeds, per game/task |
| `archive_tree.py` | rebuild + validate the checkpoint tree from `archive/*/meta.json` |
| `test_replay_invalid_action.py` | replay parity across an invalid action (free) |
| `test_client_retry_exhausted.py` | a client that gives up must not truncate the episode (free) |
| `../balrog_ckpt/` | the earlier checkpoint-feasibility work this reuses (see its REPORT.md) |

## Ablation switches (the three used on NetHack)

| switch | values | meaning |
|---|---|---|
| `--select` | `fixed` \| `orchestrator` | `fixed` = latest resumable checkpoint of the previous attempt (the "pre-death" rule) |
| `--directive` | `on` \| `off` | `off` = no extra message at all |
| `--blind` | flag | the extra message is the neutral `"Continue playing."` |

`--checkpoint-every K` (default 10) and `--max-steps` set the checkpoint
density and the episode horizon. Both matter more than they look — see
"Checkpoint starvation on MiniHack" below.

`--base-only` runs a single unchanged BALROG episode (the leaderboard protocol).
Stop rules: `--attempts N` (default 10) and `--plateau P` attempts with no
frontier advance.

## Directive discipline

Directives are strategy, never keystrokes — the rule the NetHack orchestrator
runs under. The system prompt says so, and `orchestrator.validate_directive`
enforces it: at most 2 sentences, at most 300 characters, and no literal
key/action tokens (`press`, `key`, `action:`, `ctrl-x`, single-quoted
characters, `<esc>`, "type the letter", …). A violation buys exactly one re-ask
with the reasons quoted back; a second failure keeps the checkpoint choice and
drops the directive. Every rejection is logged in `orchestrator_rounds.jsonl`
under `directive_rejections`, and counted in `summary.json`.

The ledger names each checkpoint's parent and states that the parentless one is
a full restart, to be chosen only with a reason. `chosen_is_root` is logged per
round and per selection, and `summary.json` counts `root_picks`.

## Outputs (run dir)

`attempts.jsonl`, `archive/cN/{meta.json,checkpoint.pkl}`, `summary.json`,
`selection.jsonl`, `orchestrator_rounds.jsonl`, `attempts/aNNN/trace.jsonl`,
plus two audit streams that are the point of the whole thing:

* `restore_fidelity.jsonl` — did the restore reproduce the exact observation?
* `resume_prompt_check.jsonl` — did the resumed player's message list equal the
  checkpoint's stored next-prompt, plus exactly the directive turn?

Both are asserted at runtime (`--no-verify-restore` to downgrade to logging).

Every `summary.json` carries `provenance` (this tool's git commit, branch and
dirty flag, and the BALROG checkout's commit), `env_patches`, and an
`orchestrator` block (`rounds`, `directive_rejections`, `root_picks`). Each
`archive/cN/meta.json` carries `parent`, so the archive is a tree
reconstructible without opening a single pickle — `archive_tree.py` does that
and flags multiple roots or orphans.

Run outputs are tracked in git except the per-step traces, the checkpoint
pickles and the driver logs (`runs/.gitignore`).

## Restore method per game

| game | method | verified |
|---|---|---|
| MiniHack | replay of the action prefix after re-seeding with the episode's **effective** `(core, disp, reseed)` seeds; BALROG's sticky `done` latch cleared; the evaluator's invalid-action feedback rewrite re-applied during replay | byte-identical observations (full obs digest incl. `glyphs`/`tty_chars`/`blstats`) on every resume in every smoke |
| Crafter | pickle of the inner `crafter.Env` (minus gym's `_np_random`) restored in place, plus the `_balance_chunk` determinism patch from `balrog_ckpt/crafter_ckpt.py` | canonical `state_digest` identical on restore |
| TextWorld | `.z8` (`the_cooking_game`): Jericho `get_state`/`set_state` + deepcopy of every wrapper's mutable state. `.ulx` (`treasure_hunter`, `coin_collector`): replay (no state export exists) | digest identical on restore (`.z8`) |

## Engine patches (disclosed)

`summary.json` carries `env_patches`. MiniHack and TextWorld run stock
(`env_patches: []`). **Crafter runs with TWO patches in every arm, `--base-only` included:**

1. `crafter_balance_chunk_sorted` — sorts the per-chunk object set before
   Crafter's despawn logic indexes into it. Stock Crafter is not reproducible
   across processes because that set is iterated in `id()`-hash order, so a
   pickle-restored copy diverges from the original within ~30 steps. It reorders
   an already-arbitrary choice; the RNG stream, rewards and achievements are
   untouched.
2. `crafter_seed_pinning` — Crafter drew its world seed from the **global** numpy
   RNG, so `reset(seed=)` never reached it and the same nominal seed produced
   different worlds. Owned by the Crafter agent's branch.

Restore verification with both patches: **750/750 branch steps identical**,
against **252/750 unpatched**. Both arms get the patches so base and PAE play
the same game; `summary.json` lists them in `env_patches` and the paper states
them.

## Progress accounting

The reported metric is always **BALROG's own** `env.get_stats()["progression"]`.
Checkpoints are written every K steps and on any increase of **that** metric;
the plateau guard reads **that** metric only (with MiniHack's binary
progression it therefore never resets, and the attempt cap N is what bounds
compute — intended). `summary.json` reports both `attempt1_progression` (the
base-comparable single-episode number, since attempt 1 is a stock BALROG
episode) and `pae_best_progression` (the run-level best over attempts).

Each adapter additionally exposes a dense **`aux`** signal. It drives nothing:
not checkpointing, not stopping, not the reported metric. It appears as a
labelled measured field in the orchestrator's ledger (an LLM cannot choose a
branch point from an all-zero column) and in `summary.json` only under
`aux_label` / `aux_max_measured`:

* MiniHack: number of distinct `(dlvl, x, y)` cells seen
* Crafter: achievements unlocked (`score_tracker`; BALROG's progression is this / 22)
* TextWorld: `score / max_score` (identical to progression)

Step accounting under the caps (tasks.md §3.5 open question) is reported both
ways: `committed_steps` (steps of the committed trajectory, i.e. the BALROG
horizon — a checkpoint at step 60 leaves 40) and `total_env_steps` (every env
step including replays). A checkpoint at the cap is not resumable and is
filtered out of both the fixed rule and the orchestrator's ledger.

## Measured (GLM-5.2, 2026-09-22, seed 0, all at commit `6a26ee8`'s parent tree)

| run | arm | att | ckpts | committed steps | LLM calls | in tok | out tok | list $ | billed $ | a1 prog | PAE best | restore parity | prompt parity |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|
| `runs/mh_quest_easy_s0_base` | base (unchanged BALROG) | 1 | 1 | 5 | 5 | 5,706 | 1,607 | 0.0166 | 0.0052 | 0.000 | **0.000** | n/a | n/a |
| `runs/mh_quest_easy_s0_pae` | PAE fixed + directive | 3 | 2 | 14 | 20 | 52,169 | 6,860 | 0.1135 | 0.0293 | 0.000 | **0.000** | 2/2 | 2/2 |
| `runs/mh_quest_easy_s0_pae_orch` | PAE orchestrator + directive | 4 | 1 | 10 | 30 | 46,117 | 11,323 | 0.1258 | 0.0445 | 0.000 | **0.000** | 3/3 | 3/3 |
| `runs/mh_quest_easy_s0_pae_orch_k3` | PAE orchestrator + directive, K=3 | 4 | 6 | 15 | 26 | 62,812 | 9,121 | 0.1409 | 0.0436 | 0.000 | **0.000** | 3/3 | 3/3 |
| `runs/abl_directive_off` | PAE fixed, no directive | 2 | 1 | 8 | 15 | 21,589 | 2,575 | 0.0457 | 0.0107 | 0.000 | **0.000** | 1/1 | 1/1 |
| `runs/abl_blind` | PAE fixed, neutral turn | 2 | 2 | 17 | 18 | 45,959 | 4,728 | 0.0937 | 0.0228 | 0.000 | **0.000** | 1/1 | 1/1 |
| `runs/crafter_s0_pae` | PAE fixed + directive | 2 | 13 | 60 | 70 | 125,448 | 10,536 | 0.2442 | 0.0913 | 0.273 | **0.273** | 1/1 | 1/1 |
| `runs/tw_cooking_s0_pae` | PAE fixed + directive | 2 | 8 | 80 | 90 | 152,900 | 6,349 | 0.2662 | 0.0909 | 0.235 | **0.235** | 1/1 | 1/1 |

totals: list $1.0466  billed $0.3383  llm calls 274; billed/list ratio: 0.323

"restore parity" = restores whose replayed/restored state digest and observation
text matched the checkpoint exactly. "prompt parity" = resumes whose message list
equalled the checkpoint's stored next-prompt plus exactly the directive turn.
**Both are 100% across all three games.**

Per-step: MiniHack input grows ~235 tok/step until the 16-observation window
saturates (~4.5K/step steady state), output 200-440. Crafter 1.5K in / 170 out.
TextWorld 1.9K in / 130 out. Prime's own `usage.cost` totalled $0.338 against a
$1.047 list-price estimate — **a 0.32 ratio**, so `model_prices.json` overstates
wallet cost ~3x for this model.

Variance warning: these are one seed each, at temperature 1.0. An earlier
TextWorld run of the identical configuration reached progression **1.0** (solved)
where the run tabulated above reached 0.235. Nothing here is a measurement of
PAE's effect; they are plumbing checks with cost attached.

## Cost projection

`python -m tools.balrog_pae.cost_projection --measure runs/` (multipliers and the
billed ratio derived from the runs above; env-step counts from tasks.md 3.4):

```
PANEL: 5 MiniHack tasks x 5 seeds, Crafter x 10 seeds, TextWorld x 3 games x 5 seeds
ARMS : base (stock BALROG) and PAE (orchestrator + directive, N=10 attempts)
       step_mult   = 6.16   [measured: runs/]
       prompt_mult = 1.76   [measured: runs/]  (input only)
       orchestrator overhead 5% on the PAE arm only
       billed ratio = 0.32 of list [measured: runs/]
       (cross-check: dividing by the separate base run instead gives step_mult 18.6 - noisy on one seed, not used)

game      task                   eps  steps  in/step out/step   base $    PAE $   both $
----------------------------------------------------------------------------------------
minihack  Quest-Easy               5    470     4500      320     3.99    41.72    45.71
minihack  Quest-Medium             5    240     4500      320     2.03    21.31    23.34
minihack  CorridorBattle-Dark      5    250     4500      320     2.12    22.19    24.31
minihack  Boxoban-Medium           5    450     4500      320     3.82    39.95    43.76
minihack  Boxoban-Hard             5    440     4500      320     3.73    39.06    42.79
crafter   default                 10   2700     1490      170     8.42    84.77    93.19
textworld treasure_hunter          5    400     1500      300     1.50    14.26    15.76
textworld the_cooking_game         5    400     1500      300     1.50    14.26    15.76
textworld coin_collector           5    400     1500      300     1.50    14.26    15.76
----------------------------------------------------------------------------------------
TOTAL list price                                             28.62   291.77   320.38
TOTAL at billed ratio                                         9.25    94.31   103.56
```

Every row states whether its step count and token rates were measured (and in
which run) or assumed — `cost_projection.py` prints that provenance block after
the table. The panel fits the ~$400 remaining budget at list price and has
roughly 4x headroom at the observed billed ratio. `step_mult` is the weakest
number in it: 6.2 is extrapolated from 2-4-attempt smokes to N=10 on one seed.

## Feasibility notes

**MiniHack.** Works today; restore is exact. Replay cost is ~0.25 ms/step, so
restore is free next to the LLM call, and the engine-fork path (10x faster, but
blocked by the `load_mirror` heap corruption on Quest levels in
`balrog_ckpt/REPORT.md`) is not needed. Two open items:

*Checkpoint starvation.* Checkpoints fire every K steps or on an increase of
BALROG's progression. MiniHack's progression is binary, and GLM-5.2 dies on
Quest-Easy at step 7-9, so at the default K=10 an episode produces **exactly one
checkpoint — the root**. PAE then degenerates into "restart the episode", and
the orchestrator's 3/3 root picks in `runs/mh_quest_easy_s0_pae_orch` are that,
not restart bias: `selection.jsonl` records `n_candidates: 1`. `--checkpoint-every
3` fixes it without touching any metric (`runs/mh_quest_easy_s0_pae_orch_k3`).
**K must be set per game against the observed episode length before the real
runs.**

*The fixed rule branches too late.* It picks the last checkpoint of the previous
attempt, 1-2 steps before death, and the agent walks into the same lava again.
A lookback (the checkpoint ≥5 steps before the end) is tasks.md §1's open
question and should be decided with the same evidence.

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

Free checks first — no API calls, no money:

```bash
cd /root/nld/gen-pae
# the whole loop with a scripted client, on each of the three games
for g in "minihack:MiniHack-Quest-Easy-v0" "crafter:default" "textworld:the_cooking_game"; do
  IFS=':' read -r game task <<< "$g"
  .venv-balrog/bin/python -m tools.balrog_pae.selftest_offline \
      --game "$game" --task "$task" --attempts 3 --max-steps 25 \
      --checkpoint-every 5 --run-dir "/tmp/st_$game"
done
# replay parity across an invalid action
.venv-balrog/bin/python -m tools.balrog_pae.test_replay_invalid_action
.venv-balrog/bin/python -m tools.balrog_pae.test_client_retry_exhausted
# rebuild the checkpoint tree from meta.json alone
.venv-balrog/bin/python tools/balrog_pae/archive_tree.py runs/crafter_s0_pae
```

Paid runs:

```bash
cd /root/nld/gen-pae
.venv-balrog/bin/python -m tools.balrog_pae.run --game minihack --task MiniHack-Quest-Easy-v0 \
    --seed 0 --attempts 3 --select fixed --directive on --run-dir runs/x
.venv-balrog/bin/python -m tools.balrog_pae.run --game minihack --task MiniHack-Quest-Easy-v0 \
    --seed 0 --base-only --run-dir runs/x_base
.venv-balrog/bin/python -m tools.balrog_pae.aggregate runs/
.venv-balrog/bin/python -m tools.balrog_pae.cost_projection --measure runs/
```

The venv is `/root/nld/gen-pae/.venv-balrog` (BALROG `b7afe79` installed editable
from `/root/nld/gen-pae/BALROG`, Boxoban levels and `tw_games` in place). Neither
is tracked by git (see `.git/worktrees/gen-pae/info/exclude`).

## Client failures are recorded, not fatal

BALROG's `execute_with_retries` raises after `max_retries` (5). `OpenAIWrapper`,
`ClaudeWrapper` and `AWSBedrockWrapper` let that exception escape `generate()`;
only `GoogleGenerativeAIWrapper` caught it and returned an empty completion.
Since the panel runs `PrimeOpenAIWrapper(OpenAIWrapper)`, a provider blip - or a
GLM completion that burns all of `max_tokens` on reasoning and returns
`finish_reason="length"` with `content=None` - killed the whole episode at a
random step. That truncation is not arm-neutral: it costs whichever arm plays
more steps, i.e. PAE, so it would bias the very comparison the panel makes.

All four wrappers now return `LLMResponse(completion="",
stop_reason="error_max_retries")` instead (`balrog.client.RETRY_EXHAUSTED_STOP_REASON`).
The loop treats such a step as an invalid action - the env advances on the
default action and the player gets the usual feedback banner, so replay parity
is unaffected - and counts it apart:

* `summary.json` → `client_retry_exhausted` (harness fault) and `invalid_actions`
  (genuine parse failures by the agent), never mixed;
* `trace.jsonl` → per-step `client_retry_exhausted` flag next to `valid`;
* `client_failures.jsonl` → one row per occurrence (attempt, step, reason).

Report `client_retry_exhausted` alongside any result: a run with a non-zero
count had steps the model did not really get to play.
