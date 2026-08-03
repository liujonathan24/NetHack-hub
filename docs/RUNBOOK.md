# Runbook: running NetHack CLI-harness experiments

Instructions to hand to whoever (or whatever) runs the next sweep. Every rule
here exists because breaking it cost a real measurement — the incident is named
so you can judge whether it still applies rather than following it blindly.

Companion docs: `docs/HARNESS_DEFECTS.md` (what is broken and what is fixed),
`docs/experiments/exp2-cli-harness-results.md` (results and pinned state).

---

## 0. One-time setup on a new machine

```bash
git clone git@github.com:liujonathan24/NetHack-hub.git
cd NetHack-hub && git checkout exp/cli-harness-eval
# The engine is a SEPARATE repo, not a submodule of this one:
#   ENG=/path/to/NetHackHarness   (clone git@github.com:liujonathan24/NetHack.git inside it)
git -C "$ENG" submodule update --init --recursive
```

**Rebuild the engine before the first run, and after every submodule move.** The
compiled `libnethack.so` and the submodule pointer drift independently and
nothing used to notice: on 2026-07-31 the `.so` was 230 hours older than
`nle.c`, so every rollout was linking stale code while ~30 tests failed with
errors that read like engine logic bugs (`test_engine_env` reporting two
branched games with identical glyph planes). `launch_cell.sh` now refuses to
start in that state — that refusal is the feature, not an obstacle.

Sanity gate before spending anything:

```bash
ENG=/path/to/NetHackHarness
export PYTHONPATH="$PWD/tools/pycompat:$ENG:$PWD:$PWD/environments/nethack"
.venv-cli-eval/bin/python -m pytest environments/nethack/tests/ tests/ -q
```

`tools/pycompat` **must be first**. It carries two vendor-quirk fixes the CLI
arms cannot run correctly without (Prime's `service_tier: "provisioned"`, which
no released OpenAI SDK accepts; and Gemini emitting tool calls as text). It was
missing from `launch_cell.sh` for weeks and neither fix reached the CLI arms.

Record the baseline failure count *before* you change anything, so you can prove
you did not add to it. Pre-existing failures come from host `bwrap`
(`max_user_namespaces = 0` disables the Prime Agent sandbox) and whatever the
engine is mid-change on.

---

## 1. Launching

Everything goes through `tools/cli_harness_eval/launch_cell.sh`. Do not call the
`eval` CLI directly — the launcher is what keeps the fixed factors from drifting
between arms.

```bash
MODEL="z-ai/glm-5.2" \
VARIANT=BBOX \
SKILL_SET="netplay_true,reveal,rollback" \
MAX_CONCURRENT=4 \
ROLLOUT_TIMEOUT=7200 \
tools/cli_harness_eval/launch_cell.sh claude_code outputs/cli_harness_eval/<cell>/claude_code 400 5
#                                     ^arm         ^outdir                                    ^calls ^seeds
```

| knob | what it does | why you care |
|---|---|---|
| `MODEL` | overrides the model pinned in the arm TOML | **Fixed factor.** Set once per sweep, never per-arm. |
| `VARIANT` | observation encoding (`B0`, `BBOX`, `SPARSE`, `SPARSE_ONDEMAND`, `NLE_LANG`, …) | the usual independent variable |
| `SKILL_SET` | action surface | `netplay_true` = 31 tools; `+reveal,rollback` = 33; `balrog80` = 106 |
| `MAX_CONCURRENT` | seeds in flight **per cell** | default 128 = all 5 at once. See §2. |
| `MAX_PARALLEL` | skills executed per assistant turn | `1` matches the control arm. See §3. |
| `ALLOW_BATCHING` | strips the no-batch rule from Prime's `SKILL.md` | prime_agent only. See §3. |
| `ROLLOUT_TIMEOUT` | wall-clock cap per rollout | **The cap that actually bites.** See §2. |
| `ALLOW_STALE_ENGINE` | bypasses the engine preflight | only when you know why |

Arms: `control` (v0 in-process baseline), `claude_code`, `prime_agent`, and the
`*_b80` variants for BALROG's keystroke surface. The control arm takes its knobs
in `EXTRA_ARGS` JSON, not these env vars — the launcher refuses `SKILL_SET` there
rather than guessing.

---

## 2. Concurrency and timeouts — pick these deliberately

**Concurrency.** Running three cells at once (16 rollouts) produced **19 of 20
degenerate rollouts**: MCP tool servers dropped and every agent reported losing
the game. At 2 rollouts total it was **2 disconnects in 40**. The failure was
load, not the encodings — two whole cells were nearly discarded before that was
established.

But 2 is almost certainly over-corrected. The disconnects appeared at 16-way;
5-way ran fine for days beforehand. **Start around 4–5 and watch the disconnect
rate**, rather than inheriting 2. Wall-clock scales directly: a 5-seed cell took
~43 min at 5-way and ~61–75 min at 2-way.

**Timeouts.** `ROLLOUT_TIMEOUT` defaults to 7200 s and *that* is what ends live
games, not the call budget — in the balrog80 cells no rollout ever reached the
1,200-call cap, but two of five died on `harness_timeout`. If you raise
`max_calls`, raise this with it: ~5.3 s/call median means 10,000 calls needs
~15 h, so budget ~30 h.

**Arm a stall watchdog.** Rollouts hang with every process alive, the MCP server
up, and nothing advancing — `no_progress_timeout` cannot see it because it keys
on *game state*, not on whether the harness is making calls at all. Three hangs
in one sweep; each cost ~20 min and a seed until a watchdog killed cells whose
turn file had not grown in 20 min. Also worth a credit guard if the wallet
matters: poll the balance and abort below a floor.

---

## 3. Keeping the arms comparable

The two CLI arms were **never on equal terms** and it went unnoticed for days.

Claude Code batches tool calls; Prime Agent does not, because our own
`SKILL.md` tells it *"Do not batch blind sequences of calls."* Measured on the
same budget: Claude Code 2.31 tool calls per assistant turn (34 batches of six,
five of ten on one seed), Prime 0.80–0.95. Since the budget counts **calls, not
decisions**, Claude Code bought ~2.3× the game actions per decision.

Two flags fix it from either direction — `MAX_PARALLEL=1` constrains Claude
Code, `ALLOW_BATCHING=1` frees Prime. **Run both or neither.** One alone just
flips the asymmetry.

**And verify the flag actually bound.** `parallel_refusals` is published in
trace metrics for exactly this reason. A run where it reads `0` did not test
anything: Claude Code's batching turned out to be *stochastic* — 1.24 calls/turn
in one run, 1.00 in the next with identical config — so a "batching disabled"
cell can be indistinguishable from a cell where batching simply did not occur.
That vacated two experiments.

Other factors that must be pinned, not left to defaults: `max_relaunches = 0`
(Prime has an auto-resume loop; Claude Code has none), and the same `skill_set`
across arms unless the action surface *is* the variable.

---

## 4. Aggregating — use the tool, never a one-liner

```bash
tools/cli_harness_eval/progress.py <cell-dir> [<cell-dir> ...]
```

It encodes four traps that each cost a real measurement:

1. **One row per seed.** A retried rollout writes a *new* turn file per attempt.
   Globbing `turns/*.ndjson` double-counted a 5-seed cell as n=6.
2. **Turn files beat `traces.jsonl` metrics for game state.** A retried rollout
   records only its failed final attempt, so `skill_calls` can read `0` for a
   rollout that played 2,956 calls to dlvl 2.
3. **`tool_calls` is empty in CLI-arm turn files.** Those arms dispatch over MCP;
   skill usage must come from the trace `nodes`.
4. **Prime nests skills inside `ipython`.** Counting the outer tool name reports
   100% `ipython` and 0% of everything else — both wrong.

**Degeneracy rule** (applied automatically): a rollout is degenerate if it
errored, ran under 40 calls, or exited cleanly under 80. Report those separately;
never drop them silently. Whole cells have come back 0/5 usable.

**Report both metrics.** BALROG's published score is `max(Dlvl, Xp)`, which pays
full price for diving without fighting. `min(Dlvl, Xp)` is the balanced-play
view and it reorders the table: our best cell fell 4.92 → 1.79 because it reached
dlvl 5.6 on mean XP 3.0. If you quote `min`, quote it for the baselines too.

---

## 5. Before you believe a result

- **Never compare a partial cell against a finished one.** A mid-flight
  aggregate read 1.73%; the same cell finished at 2.60% because one seed was
  still descending. That sent a whole investigation down the wrong path.
- **Check `n` per cell.** Arms routinely finish at n=4 vs n=5 through hangs and
  provider errors, so one side rests on 20% less data.
- **Check the interval before the mean.** A cell at 8.14 ± 5.84 is one lucky
  seed. n=5 cannot separate configurations that differ by less than ~1 point;
  plan 16+ seeds for anything you intend to publish.
- **Confirm what actually differs.** Two runs described as "the same cell" had
  different `skill_set` *and* 70 commits of harness drift between them.
  `results/configs/<cell>__<arm>.toml` is the resolved config as executed —
  prefer it over the template, which has been edited since.

---

## 6. Provenance — record it or the result is not reproducible

Committed under `results/`: per-seed outcomes, resolved configs, run→commit
provenance, and (since PR #20) the engine fingerprint — engine HEAD, submodule
SHA, and the `.so` path/mtime it actually linked.

Raw traces are ~1.3 GB and gitignored. They are the only copy; if they matter,
snapshot them off `/scratch`, which is not backed up.

Known-fatal and still open: `Provider finish_reason: error` and
`API Error: stream closed before completion` surface as subprocess exits, so the
rollout-level retry never engages. One took an entire cell (5/5 unusable). If a
cell comes back empty, read the final assistant message before blaming the
config — the agents say plainly when they have lost the server.
