# Experiment 2 — CLI-harness scaffold comparison: results and reproduction

**Status:** partial. The `netplay_true` matrix and the `balrog80` matrix are complete;
the `v3_bbox` CLI cells are the current run. Every number below is reproducible from
the commands in §4 at the pinned commits in §3.

**Question.** Hold BALROG's task, action set, observation and model fixed; swap their
fixed 16-step observation window for a CLI agent that carries the whole history. Does
the scaffold change the outcome, and by how much?

Arms: `control` (v0 in-process loop — the baseline), `claude_code`, `prime_agent`.

---

## 1. Headline results

### 1a. `netplay_true` surface (31 vendored upstream skills)

| model | arm | n | max dlvl ± SE | BALROG % ± SE | died | calls | degenerate |
|---|---|---|:-:|:-:|:-:|:-:|:-:|
| Gemini 3 Flash | claude_code | 4 | 2.50 ± 0.50 | 1.32 ± 0.44 | 50% | 145 | 1 |
| Gemini 3 Flash | prime_agent | 3 | 3.33 ± 0.88 | 1.98 ± 0.34 | 100% | 108 | 2 |
| GLM 5.2 | claude_code | 5 | 4.00 ± 0.84 | 2.38 ± 0.38 | 100% | 176 | 0 |
| **GLM 5.2** | **prime_agent** | 5 | **5.00 ± 1.14** | **3.75 ± 1.52** | 80% | 285 | 0 |

Replication at n=10 on a later build (`m3`, arms symmetric, `max_relaunches=0`):

| arm | n | max dlvl ± SE | BALROG % ± SE | died | calls |
|---|---|:-:|:-:|:-:|:-:|
| claude_code | 9 | 3.33 ± 0.71 | 2.04 ± 0.68 | 67% | 262 |
| **prime_agent** | 9 | **5.33 ± 0.83** | **3.83 ± 1.20** | 100% | 224 |

**Prime Agent > Claude Code in both runs, but not significantly.** Exact permutation
test on `m3`: depth p=0.107, BALROG p=0.227. The strongest evidence is the *paired*
view (same seed, both arms), which removes seed difficulty: prime_agent deeper on
**7 of 9** seeds, shallower on 1, tied 1 — two-sided sign test **p=0.070**. Consistent
direction across two runs and two action surfaces, but nothing clears p<0.05. Report as
suggestive-and-replicating, not established.

### 1b. `balrog80` surface (BALROG's 80 commands + `bal_a`–`bal_z` = 106 tools)

| model | arm | n | max dlvl ± SE | BALROG % ± SE | died | calls |
|---|---|:-:|:-:|:-:|:-:|:-:|
| GLM 5.2 | claude_code | 5 | 1.20 ± 0.20 | 0.31 ± 0.31 | 40% | 672 |
| **GLM 5.2** | **prime_agent** | 5 | **2.20 ± 0.58** | **1.08 ± 0.45** | 40% | 634 |
| Gemini 3 Flash | both | 0 | — | — | — | — |

Same arm ordering as §1a on a completely different action surface — the most useful
thing this matrix produced, since it argues the scaffold gap is not an artifact of our
skill layer.

**Do not read the low absolute scores as "BALROG's method is worse."** It is a budget
artifact. A NetPlay skill batches many game turns into one LLM call; a raw keystroke
buys about one:

| surface | calls | game turns / call | total game turns | dlvl |
|---|:-:|:-:|:-:|:-:|
| netplay_true | 176 | 7.0 – 10.5 | 1,497 | 4.00 |
| balrog80 | 672 | 1.1 – 1.4 | 783 | 1.20 |

The b80 arm spent **3.8× the LLM calls to play half as much NetHack**. BALROG's own
episodes used 1,057–2,231 calls; two of five b80 seeds died at `harness_timeout`
(2 h wall clock) before the 1,200-call budget ever bound. This configuration could not
have replicated the sourced 6.77% regardless of model.

### 1c. Control arm, before/after the observation repairs

| build | n | BALROG % ± SE | max dlvl ± SE | died | calls |
|---|:-:|:-:|:-:|:-:|:-:|
| exp1 baseline `B0` (pre-repair) | — | 3.49 ± 1.66 | — | — | 150 |
| `best_fixed` `B0` (post-repair) | 5 | **4.13 ± 1.41** | 4.20 ± 1.36 | 80% | 226 |

Means overlap well within error; the trustworthy signal is the **floor lifting** — the
worst seed went from near-zero to 2.42%, which is what removing wasted-call paths should
do. Not yet a result at n=5.

### Reference points

- Sourced BALROG SOTA: `gemini-3-pro-preview` **6.77 ± 3.21%**. Published field clusters 1.3–3.0%.
- The **12.56% / DL10 figure is NOT BALROG's** — it came from a third-party blog. Do not
  anchor on it. Note that a single lucky rollout reaches exactly 12.56% (dlvl 10 at XL 1);
  `m3` prime_agent seed 9 did.

---

## 2. Harness defects found (all measured, all fixed unless noted)

Ordered by how much outcome they moved. Each was verified against a live game.

| defect | measured impact | fix |
|---|---|---|
| **Coordinate frame mismatch** — `VISIBLE FEATURES` emitted tty rows while `Pos:`/skills used map rows | **86 of 106** coordinates wrong in one rollout | `prompt/features.py` reads `chars` |
| **Menu text bleeding into the map** — map rendered from `tty_chars` | **10.6%** of all map blocks (723/6,802) | `prompt/ascii_map.py` renders from the glyph plane (`1785d25`) |
| **Open menus/prompts invisible** — caused by the fix above; `chars` cannot represent an overlay | 3 of 5 `b80_b0` seeds burned **2,499 calls each at game time 1**; 97.2% zero-progress | `prompt/interactive_state.py`, keyed on the engine's own `misc` UI flags (`98f77ba`) |
| **Blocked prompt named only `esc`** | agent called `bal_e` **90× in 103 turns**, clock frozen at 15, 91% zero-progress | warning now names each legal reply as a callable tool, expanding `[d-f]` ranges (`8e8ed61`) |
| **Gemini emits tool calls as text** — `call:default_api:mcp__nethack__bal_south{}` | **12/12** Gemini `claude_code` rollouts ending `agent_completed` contain it; **0/25** GLM rollouts do | normalized in the *served* wire body at `_completion_response` (`d5ab2ed`) |
| **`^M` swallowed input** | — | fixed at source in `engrave_elbereth` + 3 chokepoints (`55d6017`) |
| **`launch_cell.sh` lacked `tools/pycompat` on PYTHONPATH** | neither the `service_tier` nor the Gemini fix ever reached the CLI arms | added (`bd10122`) |

### Two defects that are NOT fixed

- **`bwrap` sandbox disabled** (`2921753`). The host set `max_user_namespaces = 0`, so
  `bwrap --unshare-user` fails with ENOSPC (the namespace cap, *not* disk). Prime Agent's
  only builtin is `ipython`, which `disabled_tools` cannot constrain; a prior run globbed
  `/scratch` and printed a live MCP bearer token into its own trace. **Restore `sandbox = true`
  in `configs/prime_agent.toml` the moment the host permits namespaces.**
- **Provider transients still kill rollouts.** `Provider finish_reason: error` surfaces as a
  subprocess exit, so the rollout-level retry never engages. Cost rollouts in four runs.

---

## 3. Pinned state

```
branch          exp/cli-harness-eval
results below   HEAD d5ab2ed .. 2921753
engine          /scratch/gpfs/ZHUANGL/jl0796/NetHackHarness  (nethack_core, the fork)
venv            .venv-cli-eval  (python 3.12, verifiers 0.2.1 stock)
harness repairs PR #19 — in-tree as d407d12 (NOT 69d7bce, its pre-rebase twin)
```

**The tree has diverged from `origin/exp/cli-harness-eval`** (a rebase produced duplicate
SHAs for the same work). The four commits unique to local were pushed to
`exp/cli-harness-eval-fixes` rather than force-pushed, to avoid dropping `dc78f47` (the
gitignore commit that keeps 70 MB traces out of git). Resolve before adding more.


### Code state and hyperparameters at run time

Traces are disposable; the config and the commit are not. Both are committed:

| path | what |
|---|---|
| `results/configs/<cell>__<arm>.toml` | the **resolved** config the eval CLI actually ran — every hyperparameter after CLI/env overrides, not the template. 272 files. |
| `results/run_provenance.json` | `run -> HEAD` for every driver-launched sweep, plus a finish timestamp per cell. Entries written from 2026-07-31 also carry an `engine` block (see below); earlier entries do **not**, and the engine behind them cannot be recovered. |
| `<cell outdir>/engine_provenance.json` | the engine fingerprint for that cell — engine repo HEAD, `third_party/NetHack` SHA + the SHA the superproject pins, dirty flags, and the resolved `libnethack.so` path/mtime/size against the newest tracked build input. |
| `results/cli_harness_rollouts.json` | per-seed outcomes, 121 rollouts. |

**The HEAD above pins the harness only, not the engine.** Two measured incidents made
that gap expensive:

* `third_party/NetHack` was checked out to `66c84e6` while the engine repo pinned
  `fefd557`. About 30 engine-dependent tests failed with what read as engine *logic*
  bugs — `test_engine_env::test_snapshot_restore_branching_via_env` failed
  `assert not np.array_equal(glyphs_a, glyphs_b)`, i.e. two *branched* games returning
  byte-identical maps. No traceback mentioned the submodule.
* With the pointer restored, the compiled artifact was still from the other source
  line: `src/src/nle.c` at 2026-07-31 18:52 vs `libnethack.so` at 2026-07-22 04:26, a
  230 h gap. Every rollout `ctypes`-loads that `.so`; a source checkout does not
  rebuild it and nothing warns.

`launch_cell.sh` now runs `tools/cli_harness_eval/engine_provenance.py --check` before
every cell: it logs a one-line `[engine] …` fingerprint, writes the JSON above into the
cell's output dir, and **refuses to launch** (exit 4) when the `.so` predates the newest
tracked build input, when the submodule is dirty, or when it sits off the pinned commit.
`ALLOW_STALE_ENGINE=1` bypasses it, loudly, on stderr — a run launched that way is not
reproducible from any commit and must say so in its notes.

Commit pinned per sweep:

| sweep | started | HEAD | what it was |
|---|---|---|---|
| `matrix` (m2, netplay_true) | 2026-07-27 04:18 | `b8089e3` | before the relaunch hook and the play-until-death prompt — both arms symmetric by construction |
| `b80` (balrog80) | 2026-07-27 06:32 | `6f97f47` | |
| `m3` (10-seed replication) | 2026-07-27 12:05 | `eeabd86` | `max_relaunches=0` pinned for arm symmetry |
| `p1` (b80 GLM) | 2026-07-28 12:52 | `8e8ed61` | prompt answer-list fix |
| `p1` (b80 Flash) | 2026-07-28 13:32 | `bd10122` | Gemini shim, wrong layer — trace only |
| `v3 claude_code` | 2026-07-28 23:27 | `d920394` | Gemini shim at the served-bytes layer |
| `v3 prime_agent` | 2026-07-28 23:33 | `2921753` | bwrap sandbox disabled |

`git show <HEAD>` reconstructs the exact tree for any run. Note the `m2` HEAD matters for
interpretation: it predates `11d773b`, so neither arm had the auto-relaunch hook — which is
why that matrix is not confounded by the arm asymmetry that hook would otherwise introduce.

To regenerate the two JSON records after new runs, re-run the extraction blocks in this
document's history (they read only `traces.jsonl` + `config.toml` and write to `results/`).

---

## 4. Reproduction

Raw traces are **1.3 GB and gitignored** (`traces.jsonl`, `turns/`, `*.log`). The committed
record is `results/cli_harness_rollouts.json` — 121 rollouts, per-seed, 56 KB: stop
condition, skill calls, max dlvl, XP level, BALROG %, died, LLM calls, prompt/completion
tokens, model, and the top-8 skill histogram.

```bash
ENG=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness
export PYTHONPATH="$PWD/tools/pycompat:$ENG:$PWD:$PWD/environments/nethack"

# CLI arms. MODEL / VARIANT / SKILL_SET / ROLLOUT_TIMEOUT are FIXED FACTORS --
# set once for a sweep, never per-arm, or the comparison measures the wrong thing.
MODEL="z-ai/glm-5.2" tools/cli_harness_eval/launch_cell.sh claude_code  <outdir> 400 5
MODEL="z-ai/glm-5.2" tools/cli_harness_eval/launch_cell.sh prime_agent  <outdir> 400 5

# balrog80 surface
MODEL="z-ai/glm-5.2" tools/cli_harness_eval/launch_cell.sh claude_code_b80 <outdir> 1200 5

# v3_bbox reproduction through a CLI arm
MODEL="google/gemini-3-flash-preview" VARIANT=BBOX \
  SKILL_SET="netplay_true,reveal,rollback" ROLLOUT_TIMEOUT=7200 \
  tools/cli_harness_eval/launch_cell.sh claude_code <outdir> 400 5

# control arm (v0) — skill_set goes inside EXTRA_ARGS, not SKILL_SET
MODEL="google/gemini-3-flash-preview" \
  EXTRA_ARGS='{"variant":"BBOX","skill_set":"netplay_true,reveal,rollback",
               "belief_state_interval":25,"max_turns":400,
               "explicit_seeds":[0],"n_examples":1}' \
  tools/encoding_eval/launch_encoding_cell.sh BBOX <outdir> 400 1
```

### Traps that have cost real measurements here

1. **Never glob `turns/*.ndjson` to aggregate.** A rollout that retried writes one file per
   attempt, and a stale directory silently inflates `n`. `best_fixed` held files from two
   earlier attempts — globbing would have reported n=15 for a 5-seed cell. Join on the PID
   in the filename, or read `traces.jsonl`.
2. **`traces.jsonl` metrics can be zero for a rollout that played.** A retried rollout records
   only the failed final attempt: `b0_flash/prime_agent` shows `skill_calls=0` but its turn
   files hold 2,956 calls to dlvl 2.
3. **`tool_calls` is empty in CLI-arm turn files.** Those arms dispatch over MCP and the
   env-side record never populates it. Read the trace `nodes` instead — but **only the
   nodes with `sampled == true`**. `nodes` is a CUMULATIVE PREFIX REPLAY of the
   conversation, not a list of calls: after `k` turns it holds every prefix `1..k`, so an
   assistant message that issued one call on turn 3 reappears in the replay of turns
   4, 5, … . Counting every assistant node over-counts by ~15× and grows quadratically
   with rollout length (measured on `outputs/trace_probe`: 911 nodes → "455 skills" for a
   30-call rollout; `np_move_to` reported 292 against an actual 22). The `sampled` nodes
   reproduce `metrics`' own per-skill referee counters to the call. Prime Agent nests
   real skills inside `ipython` (`await nethack.np_move_to(...)`), so counting the outer
   tool name reports 100% `ipython` and 0% everything else — both wrong.
   `tools/eval_metrics.py :: executed_call_histogram` is the shared implementation.
4. **Always use absolute paths in launcher env vars.** A relative `trace_dir` resolves inside
   a rollout's ephemeral workdir and is discarded at teardown. This class of bug appeared
   four separate times.
5. **`agent_completed` is not a completion signal.** It is verifiers' fallback for a clean
   process exit (`v1/harness.py:118`) — under `--print`, Claude Code exits the instant a turn
   arrives with no tool call.

### Degeneracy rule used throughout

A rollout is degenerate if `stop_condition == "error"`, or `skill_calls < 40`, or
(`stop_condition == "agent_completed"` and `skill_calls < 80`). Degenerate rollouts are
excluded from means and reported separately — never silently dropped.

**`skill_calls` does not exist on the v0-legacy path.** When `eval.log` says
`running Nx1 v0 rollouts … (legacy: nethack)`, v1's `NetHackTask.finalize` never runs, so
`metrics` carries no `skill_calls` / `max_dlvl_reached` / `died` / `terminated`. A literal
`metrics['skill_calls']` raises `KeyError`; `metrics.get('skill_calls', 999)` marks every
rollout non-degenerate. Use `tools/eval_metrics.py :: skill_call_count`, whose fallback
chain is `metrics.skill_calls` → `metrics.total_tool_calls` (the v0 `ToolEnv` stock
metric) → summed `metrics.<skill>_calls` → the executed-call histogram → `len(turns)`, and
which returns `(None, "unavailable")` rather than a sentinel. `degeneracy()` then returns
`None` — **unknown**, a third answer distinct from "fine".

### Both BALROG numbers, and the pace

Every table emitted by `tools/cli_harness_eval/aggregate.py`,
`tools/cli_harness_eval/progress.py` and `tools/encoding_eval/aggregate.py` carries
BALROG's published `max` over the (Dlvl, Xp) achievement axes **and** the `min` over the
same table, plus an `xp-carried` count (`max > 0 and min == 0`) — a rollout that only
levelled up otherwise keeps its full headline score (pilot: `reveal` seed 1 is 2.12% max /
0.00% min at Dlvl 4, XP 1). Never the deprecated `progression_score` proxy. Each table also
carries the progression SLOPE — depth gained and BALROG-% per **game** turn (`status.time`)
and per LLM call — because the research question is how fast the agent progresses relative
to a human, and one LLM call runs a whole pathfinding macro (pilot: `reveal` seed 0 reached
Dlvl 5 in 388 game turns vs `fog` seed 0's Dlvl 3 in 936, at an identical 100-call cap).

---

## 5. Artifacts

- `results/cli_harness_rollouts.json` — per-seed record, all 121 rollouts (committed).
- `outputs/results.html` — consolidated results page (published artifact).
- `outputs/trace_player.html` — two-pane observation/action replayer, Claude Code vs
  Prime Agent, shared model/seed selectors, 5 steps/sec playback. Regenerate with
  `tmp/gen_player.py` against the newest cells.
