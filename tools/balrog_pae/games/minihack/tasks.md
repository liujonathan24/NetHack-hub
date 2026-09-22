# MiniHack panel — what BALROG actually measures, per task

Scope: the five non-plateaued MiniHack tasks from `/root/tasks.md` §3.2.
Everything below was read out of the installed BALROG / MiniHack / NLE code in
`/root/nld/gen-pae/.venv-balrog` and confirmed against live envs, not from docs.

| | Quest-Easy | Quest-Medium | CorridorBattle-Dark | Boxoban-Medium | Boxoban-Hard |
|---|---|---|---|---|---|
| gym id | `MiniHack-Quest-Easy-v0` | `MiniHack-Quest-Medium-v0` | `MiniHack-CorridorBattle-Dark-v0` | `MiniHack-Boxoban-Medium-v0` | `MiniHack-Boxoban-Hard-v0` |
| base class | `MiniHackSkill` | `MiniHackSkill` | `MiniHackNavigation` | `BoxoHack` | `BoxoHack` |
| env's own `max_episode_steps` | 500 | 1000 | 350 | 400 | 400 |
| **BALROG's step cap** | **100** | **100** | **100** | **100** | **100** |
| language actions | 32 | 32 | 8 (compass) | 4 (cardinal) | 4 (cardinal) |
| character | `@` (random role) | `kni-hum-law-fem` | `kni-hum-law-fem` | `@` (random role) | `@` (random role) |
| success condition | reach `>` | reach `>` | reach `>` past the rats | every boulder on a fountain | every boulder on a fountain |
| `progression` | binary | binary | binary | binary | binary |
| dense `aux` (measured, drives nothing) | distinct `(dlvl,x,y)` cells | distinct cells | distinct cells | **boulders on fountains** | **boulders on fountains** |
| level sampling | fixed `.des` | fixed `.des` | fixed `.des` | random level per `reset()` | random level per `reset()` |

## 1. `get_stats()["progression"]` is binary on all five

`balrog/environments/nle/progress.py::get_progress_system` gives every
`MiniHack*` env `BaseProgress`, whose whole update rule is:

```python
self.episode_return += reward
self.progression = 1.0 if reward >= 1.0 else 0.0
```

Two consequences the loop has to respect:

* **Binary, not graded.** There is no partial credit on any of the five. The
  leaderboard's "0–20% / 20–60% / 0–100%" figures are *means over the 5
  episodes* BALROG runs per task, not within-episode progress.
* **Not monotone and not sticky.** `progression` is recomputed from *this*
  step's reward, so it drops back to 0 on the very next step. It only ends at
  1.0 because the rewarding step is also the terminating one. Anything that
  reads it as a high-water mark must take the max itself — `pae.py` does
  (`pae_best_progression`, `attempt1_progression`), and its checkpoint trigger
  `prog > last_prog` therefore effectively never fires on MiniHack: **every
  MiniHack checkpoint is a periodic one.** That is intended; `N` (attempts), not
  the plateau guard, is what bounds PAE compute here.

Where the `reward >= 1.0` comes from, per task:

* **Quest-Easy / Quest-Medium / CorridorBattle-Dark** — no reward manager, so
  `minihack/base.py::_reward_fn` returns `reward_win = 1` on
  `StepStatus.TASK_SUCCESSFUL` plus `_get_time_penalty`. BALROG sets
  `penalty_mode="constant"`, `penalty_step=-0.01`, and in that mode
  (`nle/env/tasks.py::_get_time_penalty`) the penalty is charged **only when the
  in-game clock did not advance**. A winning move always advances it, so the
  winning step pays exactly `1.0` and `progression` reaches 1.0. (Had the
  penalty been unconditional, the win would have scored 0.99 and *no* MiniHack
  episode could ever register progression — worth knowing before changing
  `penalty_mode`.)
* **Boxoban-Medium / Hard** — `BoxoHack._reward_fn` short-circuits: it returns a
  bare `1` on `TASK_SUCCESSFUL` (no penalty term at all), `0` on any other
  terminal status, and otherwise
  `penalty_time + 0.1 * Δ(boulders on fountains)`. Note `penalty_time` is forced
  to `-0.001` by `MiniHackBoxoban{Medium,Hard}` and BALROG's `penalty_time: 0.0`
  is overwritten. So the *shaping* reward for a single box landing on a goal is
  `0.099` — well under 1.0, i.e. **it does not show up in `progression`**.
  It shows up only in `episode_return`, which BALROG does not report.

## 2. Step cap: 100 on every task

`config.yaml -> envs.minihack_kwargs.max_episode_steps: 100` is passed straight
into `gym.make(task, **minihack_kwargs)`, and every MiniHack class takes it with
`kwargs.pop("max_episode_steps", <its own default>)`, so BALROG's 100 wins on
all five (the "env's own" row above is what you would get outside BALROG).
`eval.max_steps_per_episode` is `null`, so nothing overrides it later.
`adapter.max_steps` reads back 100 on all five, live.

Under PAE this cap is the *committed-trajectory* horizon: resuming a checkpoint
taken at step 60 leaves 40 steps, and a checkpoint at step 100 is not resumable
at all (`pae.Run.resumable` filters those out). `summary.json` reports
`committed_steps_max` and `total_env_steps` separately, which is the two-way
accounting `/root/tasks.md` §3.5 asks for.

## 3. The `aux` measured field

`aux` is a labelled measured column in the orchestrator's ledger and a field in
`summary.json` (`aux_label`, `aux_max_measured`). It drives **nothing**: not the
checkpoint trigger, not the stopping rule, not the reported metric. Its only job
is to stop the ledger being an all-zero column that no orchestrator could branch
on, given §1.

* **Quest-Easy, Quest-Medium, CorridorBattle-Dark** — `distinct (dlvl, x, y)
  cells seen`, accumulated from `blstats` over the attempt. These are
  exploration tasks on sizeable maps (CorridorBattle-Dark is unlit, so the count
  also tracks how much the agent has actually revealed).
* **Boxoban-Medium, Boxoban-Hard** — `boulders on fountains (boxes on goals)`.
  Cells seen is meaningless there: one small, `premapped`, `lit` room. The count
  mirrors MiniHack's own `BoxoHack._count_boulders_on_fountains`: the goal cells
  are the fountains present in the **first frame** of the episode (a boulder
  standing on a fountain hides the fountain glyph, so they must be latched at
  reset), and the score is how many of those cells now show the boulder glyph.
  Glyph ids come from `nle.nethack`'s own tables — `GLYPH_OBJ_OFF + <"boulder"
  in OBJ_NAME/objclass>` = 2353 and `GLYPH_CMAP_OFF + S_fountain(31)` = 2390.
  This is exactly the quantity Boxoban's own shaping reward differences, so it
  is the game's notion of partial progress, just not BALROG's.

## 4. Level sampling and seeds

`BoxoHack.reset()` calls `random.choice(self._levels)` **before** the NLE reset,
so a Boxoban episode's *level* is drawn from Python's global `random`, not from
the NLE seed. The adapter therefore re-seeds `random` and `numpy.random` from
the episode seed on both `reset()` and `env_restore()`; without that a replay
would restore into a different level. (Quest and CorridorBattle have a fixed
`.des` and are immune.)

Independently, `env.reset(seed=s)` reaches `NLE.seed(core=s, disp=None)` and
`disp` is drawn from `random.SystemRandom`, so `s` alone does not pin an
episode. The adapter reads the effective `(core, disp, reseed)` triple back with
`get_seeds()` after the first reset and re-applies it before every replay.

Also note the `character: "@"` row: Quest-Easy and both Boxobans get a *random*
role each episode (the Quest-Easy smoke drew an elven Priest, the Boxoban-Medium
probe a chaotic elven Priest). Quest-Medium and CorridorBattle-Dark pin a human
knight in their own constructors and ignore BALROG's `@`. Seed-to-seed variance
on the three `@` tasks therefore includes role variance — relevant when reading
5-seed means.

## 5. Restore: replay, verified per task

`replay_verification.json` (produced by `verify_replay.py`) is the evidence.
Per task: a 100-step base episode, a checkpoint every 10 steps, and then ≥3 of
those checkpoints restored into a **freshly constructed env** and compared
field-by-field against what was saved:

`glyphs`, `tty_chars`, `tty_colors`, `blstats`, `inv_strs`, `inv_letters`,
`tty_cursor`, `text_message`, plus BALROG's rendered `long_term_context` and
`short_term_context`.

Every replayed prefix contains at least one **invalid** action, so the
evaluator's `"Your previous output did not contain a valid action. Defaulted to
action: …"` rewrite is inside the compared text. Result: 15/15 replays across
the five tasks, all 10 fields byte-identical, at 0.26–0.95 ms per replayed step.

The five GLM-5.2 base episodes emitted **zero** invalid actions between them,
so the invalid-action case had to be forced: `--source scripted` injects
non-action completions at fixed steps. `--source trace` re-runs the same
verification over the action sequence an actual base episode produced
(Quest-Easy 2 replays over 9 steps, Quest-Medium 3 over 79, CorridorBattle-Dark
3 over 46 — all identical), which is the weaker but more realistic check.

One trap found while writing the verifier and worth repeating: **NLE hands back
aliases of its internal `last_observation` buffers.** A checkpoint that keeps
`obs["obs"]` by reference silently tracks the live env and every later
comparison is vacuous. `pae.py` is safe (it stores rendered text plus a digest
computed on the spot); anything new must copy.

## 6. Calibration runs on this branch (what was actually run, and what it cost)

Runs live under `/root/nld/gen_runs/minihack/<task>_s0_{base,pae}`; the numbers
are in `summary.json` and are reproduced in the branch report. Protocol:

* **base** — one unchanged BALROG episode (`--base-only`), seed 0.
* **pae** — one PAE run, `--attempts 2 --select fixed --directive on`, seed 0.
  Two attempts is the minimum that exercises a resume; the point of these runs
  is `restore_fidelity.jsonl` + `resume_prompt_check.jsonl`, not progression.

Deliberate deviations, all for budget/wall-clock, all disclosed:

| run | deviation | why |
|---|---|---|
| Boxoban-Medium/Hard base | `--max-steps 40` instead of the 100-step cap | see the `max_tokens` finding below: GLM-5.2 spends 4–8K output tokens per Boxoban step and a full 100-step episode did not fit the smoke budget |
| Boxoban-Medium/Hard pae | `--max-steps 20` | the resume only needs one resumable checkpoint (step 10) |
| CorridorBattle-Dark pae | `--max-steps 30` | the uncapped base episode *solved* at step 46, and `pae.Run.go` stops on `progression >= 1.0`, so an uncapped PAE run would never have reached attempt 2 |
| Quest-Easy pae | `--checkpoint-every 5` | the base episode died at step 9, so with `K=10` the only checkpoint would have been step 0 and the "replay" would have been empty |

`--max-steps` sets both the horizon and the resumability cap in `pae.py`, so
these runs are internally consistent; they are **not** leaderboard-comparable
and the cost table extrapolates their per-call tokens to the real 100-step cap
rather than using their episode lengths (`cost_table.py --resume-model`).

### Finding: BALROG's default `max_tokens: 8192` truncates GLM-5.2 on Boxoban

On Boxoban (and only there) GLM-5.2 regularly returns `finish_reason="length"`
with ~5.5K reasoning tokens and **no content**, which BALROG's client treats as
a retryable error (`balrog/client.py`, `max_retries: 5`). Each retry is a full
8192-token completion that is thrown away — ~$0.017 billed each — and after
five of them the client returns an empty completion with
`stop_reason="error_max_retries"`, which the evaluator then scores as an
invalid action and defaults to `north`.

Two consequences:

1. **The token ledger under-reports.** `pae.py` accumulates the tokens of the
   *successful* call only, because that is all BALROG's `LLMResponse` carries.
   Retries are invisible to `tokens.total`. They are **not** invisible to
   `tokens.billed_by_provider_usd`, which wraps `chat.completions.create` and
   sums Prime's own `usage.cost` on every attempt. On Boxoban the gap between
   the two is the retry waste; use the billed number.
2. **Boxoban is the expensive task**, by a wide margin, and the reason is
   output tokens, not input. Anything that raises `max_tokens` would fix the
   retries but is a change to BALROG's published config, so it was not made.
