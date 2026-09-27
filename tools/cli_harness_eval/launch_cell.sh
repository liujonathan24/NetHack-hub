#!/usr/bin/env bash
# Launch ONE arm of the CLI-harness comparison (control / claude_code / prime_agent).
# Every arm must run through THIS script so the fixed factors (model, seeds,
# character, task_spec, skill_set -- see tests/test_arm_configs.py) cannot drift
# between arms. Mirrors tools/encoding_eval/launch_cell.sh's role for the
# encoding sweep, adapted for the verifiers v1 `eval` CLI and the two config
# key shapes (`[args]` for the control arm's v0 legacy bridge, `[taskset]` for
# the two MCP-driven CLI arms).
#
# Usage:
#   tools/cli_harness_eval/launch_cell.sh <ARM> <OUTDIR> [MAX_CALLS] [N]
# e.g.
#   tools/cli_harness_eval/launch_cell.sh claude_code outputs/cli_harness_eval/smoke/claude_code 20 1
#   tools/cli_harness_eval/launch_cell.sh control outputs/cli_harness_eval/run1/control 150 16
#
# MAX_CALLS binds a DIFFERENT knob per arm, on purpose -- see
# configs/control.toml's `max_turns` note and configs/README.md Sec 8. The
# control arm's `args.max_turns` caps LM TURNS: a turn can pass with zero
# skills executed (a no-tool-call nudge), or drop every parallel tool call
# past the first, so the control arm gets *at most* MAX_CALLS executed
# skills. The CLI arms' `taskset.max_skill_calls` is a toolset-side referee
# that grants exactly MAX_CALLS EXECUTED skills. They are pinned to the same
# nominal number by convention, not because they measure the same thing.
# tools/cli_harness_eval/aggregate.py normalizes on the MEASURED
# total_tool_calls / skill_calls per rollout, never on this nominal value --
# do not "fix" the asymmetry by hand-tuning MAX_CALLS per arm; report it in
# the aggregated table's notes instead.
#
# N selects how many of the 16 pinned seeds run: the eval CLI's `--num_tasks`
# takes the first N task indices (`shuffle = false` in every config), which
# are seeds 0..N-1 of the pinned `explicit_seeds` list -- so N=1 always means
# "seed 0 only", matching the committed acceptance artifacts.
set -euo pipefail

# The shared engine repo (NetHackHarness) must be importable for all three
# arms -- the control arm loads it in-process; the two CLI arms' tool server
# is launched as `python -m nethack_v1` with PYTHONPATH inherited from this
# process (v1/mcp/launch.py). Override if this checkout lives elsewhere.
ENG="${ENG:-/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness}"

KNOWN_ARMS="control claude_code prime_agent claude_code_b80 prime_agent_b80"
# The *_b80 arms are the same two CLI harnesses on BALROG's 80-keystroke
# surface (skill_set=balrog80) instead of netplay_true. Separate configs
# rather than a SKILL_SET env override, because the action surface is the
# thing under test in those cells and must not be settable from a shell
# variable that a future sweep could forget to pass.

usage() {
  echo "usage: launch_cell.sh <ARM> <OUTDIR> [MAX_CALLS] [N]" >&2
  echo "  ARM one of: ${KNOWN_ARMS}" >&2
}

if [ "$#" -lt 2 ]; then
  usage
  exit 2
fi

ARM="$1"
OUTDIR="$2"
MAX_CALLS="${3:-150}"
N="${4:-16}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

case " ${KNOWN_ARMS} " in
  *" ${ARM} "*) ;;
  *)
    echo "launch_cell: unknown arm '${ARM}' (expected one of: ${KNOWN_ARMS})" >&2
    exit 2
    ;;
esac

CFG="tools/cli_harness_eval/configs/${ARM}.toml"
[ -f "$CFG" ] || {
  echo "launch_cell: no such config: ${CFG}" >&2
  exit 2
}

EVAL_BIN="${EVAL_BIN:-${REPO}/.venv-cli-eval/bin/eval}"
[ -x "$EVAL_BIN" ] || {
  echo "launch_cell: eval binary not found or not executable: ${EVAL_BIN}" >&2
  exit 2
}

mkdir -p "${OUTDIR}"
# Absolute, or a rollout's runtime workdir teardown silently discards it --
# see configs/control.toml / configs/prime_agent.toml's `trace_dir` notes.
TRACE_DIR="$(cd "${OUTDIR}" && mkdir -p turns && cd turns && pwd)"
OUT_ABS="$(cd "${OUTDIR}" && pwd)"

# tools/pycompat FIRST: its sitecustomize.py is imported at interpreter start in
# this process and every worker/tool-server subprocess, and carries two vendor
# quirk fixes the CLI arms cannot run correctly without --
#   * `service_tier: "provisioned"` from Prime, which no released OpenAI SDK
#     accepts, killing rollouts mid-run with a pydantic ValidationError;
#   * Gemini emitting a tool call as TEXT (`call:default_api:...{}`), which
#     leaves the turn with no structured tool call. Claude Code's --print mode
#     exits on such a turn, so ONE bad emission ends the rollout as
#     `agent_completed` (measured: 12/12 Gemini claude_code rollouts that ended
#     that way carry the string; 0/25 GLM ones do).
# This line previously omitted pycompat, so neither fix reached the CLI arms --
# `launch_encoding_cell.sh` has always had it, which is why only the encoding
# sweep was protected.
export PYTHONPATH="${REPO}/tools/pycompat:${ENG}:${REPO}:${REPO}/environments/nethack${PYTHONPATH:+:${PYTHONPATH}}"

# --- engine preflight --------------------------------------------------------
# The one factor this script did NOT pin was the ENGINE. Two measured incidents,
# both of which produced results that looked fine:
#
#   1. A session checked `third_party/NetHack` out to 66c84e6 while the engine
#      repo pinned fefd557. ~30 engine-dependent tests then failed with what
#      read as engine LOGIC bugs -- test_engine_env's
#      test_snapshot_restore_branching_via_env failed
#      `assert not np.array_equal(glyphs_a, glyphs_b)`, i.e. two BRANCHED games
#      returning byte-identical maps. No traceback mentioned the submodule.
#
#   2. With the pointer restored, the COMPILED artifact was still from the other
#      source line: src/src/nle.c at 2026-07-31 18:52 vs libnethack.so at
#      2026-07-22 04:26, a 230h gap. Every rollout ctypes-loads that .so; a
#      source checkout does not rebuild it and nothing warns. Under the
#      concurrency this sweep runs at, that silently corrupts a whole sweep.
#
# `results/run_provenance.json` recorded the HARNESS commit only, so neither
# incident was recoverable after the fact. The preflight now (a) prints a
# greppable engine fingerprint into every cell's log, (b) drops that fingerprint
# into the cell's output dir next to the resolved config, and (c) REFUSES to
# launch when the .so is older than the newest tracked build input, when the
# submodule is dirty, or when it sits off the pinned commit.
#
# ALLOW_STALE_ENGINE=1 bypasses the refusal, loudly. Legitimate uses exist -- a
# box where the engine cannot be rebuilt, or deliberately reproducing an old
# run's binary -- but the run is then not reproducible from any commit, so the
# bypass shouts it on stderr rather than passing quietly.
#
# The venv interpreter is used when present because the real .so resolution goes
# through `nethack_core._engine.library_path()` (there are at least two
# libnethack.so on disk and only the build/ one is loaded); a bare python3 that
# cannot import nethack_core degrades to "unknown", which never blocks.
PY_BIN="${PY_BIN:-$(dirname "${EVAL_BIN}")/python}"
[ -x "$PY_BIN" ] || PY_BIN=python3
"$PY_BIN" "${REPO}/tools/cli_harness_eval/engine_provenance.py" \
  --check --json "${OUT_ABS}/engine_provenance.json" || {
  echo "launch_cell: refusing to launch ${ARM} -> ${OUT_ABS} (see above)." >&2
  exit 4
}

if [ "$ARM" = "control" ]; then
  # `[args]` is an untyped free-form dict (verifiers v1 EnvConfig.args: dict
  # = {}); a scalar override like `--args.max_turns 150` lands as the STRING
  # "150" (pydantic_config only JSON-decodes a sub-field value that starts
  # with `{`/`[`), and nethack.py's `self.max_turns > 0` then raises
  # TypeError against a real GLM run. A single `--args '{...}'` JSON blob
  # deep-merges over the TOML's `[args]` table (keeping task_spec,
  # character, explicit_seeds, skill_set, ... untouched) while giving
  # `max_turns` and `trace_dir` their real JSON types.
  ARGS_JSON=$(printf '{"max_turns": %s, "trace_dir": %s}' \
    "${MAX_CALLS}" "$(printf '%s' "${TRACE_DIR}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")
  OVERRIDES=(--args "${ARGS_JSON}")
else
  # `[taskset]` is a typed sub-model, so plain dotted overrides coerce
  # correctly (max_skill_calls lands as a real int).
  # MAX_CALLS=0 means PLAY TO COMPLETION -- the rollout ends when the character
  # dies, ascends, or NLE truncates, not when a budget runs out.
  #
  # Disabling the referee alone is NOT enough. `max_turns` is a second, quieter
  # cap: nethack.py:2026 ORs `max_turns_reached` into `is_completed`, and the
  # arm TOML pins it at 200. Overriding only `max_skill_calls` leaves that in
  # force, so the rollout still stops at 200 -- with stop_condition
  # `max_turns_reached` instead of `call_budget_exhausted`, which looks like a
  # different phenomenon rather than the same cap wearing a hat.
  OVERRIDES=(--taskset.max_skill_calls "${MAX_CALLS}" --taskset.trace_dir "${TRACE_DIR}")
  if [ "${MAX_CALLS}" -le 0 ] 2>/dev/null; then
    OVERRIDES+=(--taskset.max_turns 0)
    echo "[launch_cell] MAX_CALLS=0 -> uncapped: no call budget, no LM-turn cap." >&2
    echo "[launch_cell]   backstops remain: max_episode_steps=100k NLE steps," >&2
    echo "[launch_cell]   [timeout] rollout=7200s wall clock." >&2
  fi
fi

# MODEL overrides the model pinned in the arm's TOML. The arms MUST agree on it
# -- that is the whole point of the comparison -- so set it once for the sweep
# (run_sweep.sh passes it through), never per-arm.
if [ -n "${MODEL:-}" ]; then
  OVERRIDES+=(--model "${MODEL}")
fi

# VARIANT overrides the observation encoding. Same contract as MODEL: it is a
# FIXED FACTOR across arms within one cell, so set it once for the sweep. The
# `[args]`/`[taskset]` split applies here too -- the control arm's variant is a
# `load_environment` kwarg, the CLI arms' is a taskset field.
if [ -n "${VARIANT:-}" ]; then
  if [ "${ARM}" = "control" ]; then
    ARGS_JSON=$(printf '{"max_turns": %s, "trace_dir": %s, "variant": %s}' \
      "${MAX_CALLS}" \
      "$(printf '%s' "${TRACE_DIR}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
      "$(printf '%s' "${VARIANT}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")
    OVERRIDES=(--args "${ARGS_JSON}")
  else
    OVERRIDES+=(--taskset.variant "${VARIANT}")
  fi
fi

# SKILL_SET overrides the action surface. Normally the surface is pinned in the
# arm's TOML precisely because it is the thing under test -- but a cell that
# reproduces an existing baseline arm must be able to match that baseline's
# surface exactly, and `v3_bbox` uses `netplay_true,reveal,rollback` (33 tools:
# `reveal` returns an ASCII crop without consuming an NLE step; `rollback`
# becomes usable after death). Like MODEL and VARIANT this is a FIXED FACTOR
# across arms in one cell -- set it once for the sweep, never per-arm, or the
# comparison is measuring the surface instead of the scaffold.
if [ -n "${SKILL_SET:-}" ]; then
  if [ "${ARM}" = "control" ]; then
    echo "launch_cell: SKILL_SET override is for the CLI arms; the control arm" >&2
    echo "  takes skill_set inside EXTRA_ARGS/[args]. Refusing to guess." >&2
    exit 2
  fi
  OVERRIDES+=(--taskset.env_args.skill_set "${SKILL_SET}")
fi

# ENV_ARGS: a JSON object merged over `[taskset.env_args]`, for cells that vary a
# `load_environment` kwarg the CLI arms have no dotted override for -- notably the
# engine difficulty/generation knobs:
#     ENV_ARGS='{"tune":{"reveal_map":1.0}}'
#
# A whole-object override REPLACES the table rather than merging into it, so this
# folds in the config's own env_args first and refuses to run alongside SKILL_SET
# (which writes the same table through a dotted path and would be silently
# clobbered). One config, one differing factor -- forking the toml instead would
# drift from the base and reintroduce exactly the confounds the single-base design
# exists to prevent.
if [ -n "${ENV_ARGS:-}" ]; then
  if [ "${ARM}" = "control" ]; then
    echo "launch_cell: ENV_ARGS is for the CLI arms; the control arm takes these" >&2
    echo "  inside EXTRA_ARGS/[args]. Refusing to guess." >&2
    exit 2
  fi
  if [ -n "${SKILL_SET:-}" ]; then
    echo "launch_cell: set SKILL_SET *inside* ENV_ARGS, not alongside it -- a whole-" >&2
    echo "  object env_args override replaces the table and would drop it." >&2
    exit 2
  fi
  # The venv interpreter, NOT `python3`: tomllib is 3.11+ and the system python
  # here is 3.10, where this failed with a bare ModuleNotFoundError that the
  # `|| exit 2` below reported as "not valid JSON".
  # Emitted as DOTTED SCALARS, one per leaf, not as one JSON object: the eval CLI
  # validates `--taskset.env_args` as a dict and rejects a JSON *string* for it
  # ("Input should be a valid dictionary"). Dotted paths are also what the
  # existing SKILL_SET override uses, so both go through the same mechanism and
  # merge into the config's table instead of replacing it.
  mapfile -t _ENV_ARG_FLAGS < <(ENV_ARGS="${ENV_ARGS}" "$PY_BIN" - <<'PYFLAT'
import json, os
def walk(prefix, node):
    for k, v in node.items():
        path = f"{prefix}.{k}"
        if isinstance(v, dict):
            walk(path, v)
        else:
            print(f"--taskset.env_args{path}")
            # Scalars reach the config as STRINGS through the CLI, and a knob
            # like tune.reveal_map must be a float -- json.dumps keeps 1.0 as
            # `1.0` and true as `true`, which the loader coerces correctly,
            # whereas bare str(v) would hand it "1.0"/"True".
            print(v if isinstance(v, str) else json.dumps(v))
walk("", json.loads(os.environ["ENV_ARGS"]))
PYFLAT
) || { echo "launch_cell: ENV_ARGS is not valid JSON" >&2; exit 2; }
  echo "[launch_cell] env_args overrides: ${_ENV_ARG_FLAGS[*]}"
  OVERRIDES+=("${_ENV_ARG_FLAGS[@]}")
fi

# MAX_CONCURRENT caps how many of this cell's seeds run at once. The config
# default is 128, i.e. every seed of a cell starts simultaneously -- so "one cell
# at a time" is still 5-way concurrency, and running three cells together was
# 15-16 rollouts sharing one box.
#
# That load is not free: the FIRST b0_cli attempt lost ALL FIVE claude_code seeds
# to MCP tool-server disconnects at 24-55 calls ("Unable to connect", "I have
# lost connection to the NetHack game environment"), and sparse_cli returned 0/10
# usable in the same window. Both were launched under 16-way load. The same
# failure took m3 seed 1 earlier. Lower this when a run must not lose seeds to
# infrastructure -- the rollouts are long, so the wall-clock cost is real but the
# alternative is discarding whole cells.
if [ -n "${MAX_CONCURRENT:-}" ]; then
  OVERRIDES+=(--max_concurrent "${MAX_CONCURRENT}")
fi

# MAX_PARALLEL caps skills executed per ASSISTANT TURN (0 = unlimited, 1 = one
# skill per turn, matching the v0 control arm which has always dropped parallel
# tool calls past the first).
#
# The call budget counts CALLS, not decisions, so a batching client gets fewer
# decisions for the same budget. Measured on identical GLM-5.2/B0/seed-2 runs:
# the 2026-07-27 cell emitted exactly 1.00 calls per turn and reached the
# down-stair around decision 250; the 2026-07-30 cell batched (404 calls in 175
# turns) and exhausted the same 400-call budget after 175 decisions, never
# leaving dlvl 1. Set 1 to compare against the control arm or any pre-batching
# run; leave unset to reproduce the batching behaviour as-run.
if [ -n "${MAX_PARALLEL:-}" ]; then
  OVERRIDES+=(--taskset.max_parallel_skill_calls "${MAX_PARALLEL}")
fi

# ALLOW_BATCHING=1 strips the "Do not batch blind sequences of calls" rule from
# the SKILL.md shipped to Prime Agent, letting it issue as many skills per turn
# as it likes. prime_agent only. The counterpart is MAX_PARALLEL=1, which
# constrains Claude Code instead -- run both or neither, or the asymmetry just
# flips direction.
if [ -n "${ALLOW_BATCHING:-}" ]; then
  OVERRIDES+=(--harness.allow_batching "${ALLOW_BATCHING}")
fi

# ROLLOUT_TIMEOUT raises the per-rollout wall-clock cap. The default 7200s (2h)
# is what actually ended live games in the b80 cells -- two of five seeds
# stopped at `harness_timeout` while the 1200-call budget never bound -- so any
# long-horizon cell MUST raise it alongside MAX_CALLS or it silently measures
# the clock instead of the agent. At the measured ~5.3s/call median, 10k calls
# needs ~15h; 108000 (30h) leaves headroom for the latency tail.
if [ -n "${ROLLOUT_TIMEOUT:-}" ]; then
  OVERRIDES+=(--timeout.rollout "${ROLLOUT_TIMEOUT}")
fi

# --- stall watchdog (opt-in: STALL_WATCHDOG=1) --------------------------------
# Two unbounded hangs are still open and NEITHER reaches `env.step`, so the
# engine's `no_progress_timeout` can never fire (HARNESS_DEFECTS §3.1):
# `np_explore_level` spinning in vendored pathfinding (no return in 600s), and
# post-death `rollback` + any NetPlay skill deadlocking the adapter (3/3).
# The only surviving signal is that `turns/<seed>_<pid>_<ts>.ndjson` stops
# growing, which tools/stall_watchdog.py watches from OUTSIDE the process.
#
# Armed HERE rather than once per sweep on purpose: RUNBOOK's documented failure
# mode is a watchdog that exits when the first batch's queue drains, leaving
# later batches unguarded. One `launch_cell.sh` invocation == one batch == one
# watchdog, and `--parent-pid $$` ties its life to the eval process below (the
# `exec` keeps this PID), so it cannot outlive or under-live the batch.
#
# OPT-IN, not mandatory: a foreground/interactive run where someone is watching
# does not want a background process SIGKILLing it, and an unset env var must
# leave the launcher behaving exactly as it did before. Set STALL_WATCHDOG=1 for
# anything unattended.
if [ -n "${STALL_WATCHDOG:-}" ]; then
  "$PY_BIN" "${REPO}/tools/stall_watchdog.py" \
    --turns-dir "${TRACE_DIR}" \
    --timeout "${STALL_TIMEOUT:-300}" \
    --poll "${STALL_POLL:-15}" \
    --parent-pid "$$" \
    ${STALL_EXTRA_ARGS:-} >>"${OUT_ABS}/stall_watchdog.log" 2>&1 &
  echo "[launch_cell:watchdog] armed: pid=$! timeout=${STALL_TIMEOUT:-300}s" \
       "log=${OUT_ABS}/stall_watchdog.log quarantine=${TRACE_DIR}.stalled"
fi

# NEW DEFAULT (2026-08-19): every launched experiment logs the FULL baseline
# telemetry -- every game step's screen + glyph ids (record_step_frames), all
# observations and all LLM responses (already in turns/ + traces.jsonl). A cell
# that must opt out sets record_step_frames explicitly in ENV_ARGS; we only
# inject the default when the caller did not speak to it, so pinned arms that
# predate this default replay byte-identically from their committed configs.
case "${ENV_ARGS:-}" in
  *record_step_frames*) : ;;  # caller decided; respect it
  *) OVERRIDES+=(--taskset.env_args.record_step_frames true) ;;  # recorded in the resolved config.toml
esac
# describe_args: spell each tool's arguments into its description so the model
# does not burn ~10 calls probing at session start (MCP inputSchema does not
# survive transport to Prime Agent's client). Default on for new runs; opt out
# by naming it in ENV_ARGS. See docs/EXPERIMENT_E8.md / helpers._args_clause.
case "${ENV_ARGS:-}" in
  *describe_args*) : ;;
  *) OVERRIDES+=(--taskset.env_args.describe_args true) ;;
esac

# TOOL_TIER selects a row of configs/tool_tiers.toml. The mapping is NOT
# duplicated here any more: tool_tiers.py reads the registry and emits the
# override flags, so launcher and registry cannot drift. (They did: the previous
# bash `case` block was pinned by a substring test that still passed when the
# entire block was deleted.)
#
# TOOL_TIER now expands the CELL CONTRACT as well as the fix flags. It used to
# expand to nothing for `base`, so a cell declaring the E10 baseline silently
# inherited configs/prime_agent.toml -- netplay_true instead of np_core, B0
# instead of BBOX_MIN, 150 calls instead of 200, 16 seeds instead of 5.
if [ -n "${TOOL_TIER:-}" ]; then
  case " ${ARM} " in
    *" prime_agent "*|*" prime_agent_b80 "*) ;;
    *)
      # `HarnessConfig` is extra="forbid" and only the prime_agent harness
      # declares `skill_doc_coords`, so a tier carrying a harness-side flag is a
      # hard ValidationError on the other arms. Refuse here, where the message
      # can say why, rather than deep in pydantic.
      echo "launch_cell: TOOL_TIER is a prime_agent concept (the tier sets" >&2
      echo "  --harness.skill_doc_coords, which only that harness declares)." >&2
      echo "  Arm '${ARM}' cannot take it. Set the env_args explicitly instead." >&2
      exit 2
      ;;
  esac
  if [ -n "${ENV_ARGS:-}" ] || [ -n "${SKILL_SET:-}" ]; then
    # Both write the same dotted paths and the CLI resolves duplicates
    # last-wins, silently. Measured: ENV_ARGS='{"netplay_telemetry":true}' with
    # TOOL_TIER=base produced a cell labelled base running a human-tier fix.
    # Same class of foot-gun the SKILL_SET/ENV_ARGS guard already refuses.
    echo "launch_cell: TOOL_TIER cannot be combined with ENV_ARGS or SKILL_SET --" >&2
    echo "  they set the same dotted paths and the last one silently wins, so the" >&2
    echo "  cell would not run the tier it claims. Put the difference in" >&2
    echo "  configs/tool_tiers.toml as its own tier." >&2
    exit 2
  fi
  # The tier resolver needs tomllib (3.11+). PY_BIN falls back to the system
  # `python3` when EVAL_BIN has no sibling interpreter, and on this box that is
  # 3.10 -- the same trap the ENV_ARGS flattener documents. Fail with the reason
  # rather than a bare ModuleNotFoundError from inside a subshell.
  if ! "$PY_BIN" -c 'import tomllib' 2>/dev/null; then
    echo "launch_cell: TOOL_TIER needs a Python with tomllib (3.11+); ${PY_BIN} lacks it." >&2
    echo "  Point EVAL_BIN at the venv binary (its sibling ./python is used), or" >&2
    echo "  set PY_BIN to a 3.11+ interpreter." >&2
    exit 2
  fi
  # Tier-aware: an experiment tier's [<tier>.contract] overrides (variant,
  # skill_set, tune knobs) are part of ITS contract -- the checks below must
  # enforce the tier as declared, not the global baseline row.
  _TIER_CONTRACT="$("$PY_BIN" "${REPO}/tools/cli_harness_eval/tool_tiers.py" contract "${TOOL_TIER}")" || exit 2
  _TIER_VARIANT="$(printf '%s' "$_TIER_CONTRACT" | "$PY_BIN" -c 'import json,sys; print(json.load(sys.stdin)["variant"])')"
  _TIER_CALLS="$(printf '%s' "$_TIER_CONTRACT" | "$PY_BIN" -c 'import json,sys; print(json.load(sys.stdin)["max_calls"])')"
  _TIER_SEEDS="$(printf '%s' "$_TIER_CONTRACT" | "$PY_BIN" -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["seeds"]))')"

  # The contract owns the encoding and the budget. A caller who passes a
  # different one is running a different experiment and must say so.
  if [ -n "${VARIANT:-}" ] && [ "${VARIANT}" != "${_TIER_VARIANT}" ]; then
    echo "launch_cell: VARIANT=${VARIANT} contradicts the tier contract (${_TIER_VARIANT})." >&2
    exit 2
  fi
  # TIER_SHORT_BUDGET=1 is the preflight's escape: a mock play needs a handful of
  # calls, not the contract's 200, and it is checking PLUMBING (does the resolved
  # config match the tier, does the surface come up) rather than producing a
  # measurable cell. It records the fact in the artifact so a short cell can
  # never be mistaken for a real one.
  # ALLOW_BATCHING is not a free knob under a tier: it strips the no-batch rule
  # from the served document AFTER the frozen-doc hash check, so a [base] cell
  # would serve a document the E10 baseline never served, and nothing would
  # fire. The contract pins it; a contradicting env var is refused.
  if [ -n "${ALLOW_BATCHING:-}" ]; then
    echo "launch_cell: ALLOW_BATCHING contradicts the tier contract" >&2
    echo "  (allow_batching is pinned by configs/tool_tiers.toml). A batching" >&2
    echo "  cell is a different experiment -- give it its own tier." >&2
    exit 2
  fi
  # The seed list is the contract's; running fewer rows than it names is a
  # SUBSET, which is fine for a mock play and misleading for anything else.
  _TIER_NSEEDS="$(printf '%s' "$_TIER_SEEDS" | "$PY_BIN" -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  if [ -z "${SEEDS:-}" ] && [ "${TIER_SHORT_BUDGET:-}" != "1" ] && [ "${N}" != "${_TIER_NSEEDS}" ]; then
    echo "launch_cell: N=${N} but the tier contract names ${_TIER_NSEEDS} seeds." >&2
    echo "  Pass SEEDS to run a different set deliberately, or TIER_SHORT_BUDGET=1" >&2
    echo "  for a preflight mock play." >&2
    exit 2
  fi
  if [ "${TIER_SHORT_BUDGET:-}" = "1" ]; then
    OVERRIDES+=(--taskset.env_args.tier_short_budget "true")
  elif [ "${MAX_CALLS}" != "${_TIER_CALLS}" ]; then
    echo "launch_cell: MAX_CALLS=${MAX_CALLS} contradicts the tier contract (${_TIER_CALLS})." >&2
    echo "  The reference numbers in tool_tiers.toml describe ${_TIER_CALLS} calls" >&2
    echo "  (0 = uncapped: play until death, ascension or NLE truncation)." >&2
    exit 2
  fi
  VARIANT="${_TIER_VARIANT}"
  OVERRIDES+=(--taskset.variant "${VARIANT}")
  # Pin ROW SELECTION too: `--num_tasks N` only ever takes the first N of the
  # config's own seed list, so without this the tier's seeds are advisory.
  # SEEDS overrides the contract's row selection for cells that deliberately
  # run a different half -- E13's reflection corpus is seeds 5-9 while its
  # evaluation stays on the contract's 0-4. It cannot go through ENV_ARGS (the
  # guard above refuses that alongside TOOL_TIER, for good reason), so it is its
  # own knob, and the artifact records that the contract's seeds were replaced.
  if [ -n "${SEEDS:-}" ]; then
    OVERRIDES+=(--taskset.env_args.explicit_seeds "${SEEDS}")
    OVERRIDES+=(--taskset.env_args.seeds_overridden "true")
    echo "[launch_cell] seeds overridden: ${SEEDS} (contract: ${_TIER_SEEDS})"
  else
    OVERRIDES+=(--taskset.env_args.explicit_seeds "${_TIER_SEEDS}")
  fi
  # NOT `mapfile -t X < <(cmd) || exit`: process substitution does not set the
  # pipeline status, so mapfile succeeds even when the resolver died and the
  # cell launches with NO tier flags at all. Capture, check, then split.
  _TIER_FLAGS_RAW="$("$PY_BIN" "${REPO}/tools/cli_harness_eval/tool_tiers.py" \
    flags "${TOOL_TIER}" --arm "${ARM}")" || exit 2
  mapfile -t _TIER_FLAGS <<< "$_TIER_FLAGS_RAW"
  OVERRIDES+=("${_TIER_FLAGS[@]}")
  echo "[launch_cell] tool_tier=${TOOL_TIER} -> ${_TIER_FLAGS[*]}"
fi

# CONTINUAL_HARNESS mounts a shared Prime Agent continual-harness store into
# every rollout of this cell, so lessons an earlier game persisted are in a
# later game's system prompt (docs/EXPERIMENT_E13.md). prime_agent only.
#
# Per EXPERIMENT, not global. Several continual experiments run side by side --
# different reflection prompts, same base surface -- so the store path, the run
# id and the reflection-prompt hash all vary per run and are pinned into this
# cell's config.toml next to tool_tier. Without the id in the artifact, two
# experiments' outputs are indistinguishable after the fact.
# INSTALL_DIR isolates one experiment's skill package and kernel venv from
# another's. It is fixed per experiment on purpose -- Prime Agent keys the kernel
# venv on the set of Python-skill paths -- but the DEFAULT is global, so two
# worktrees running concurrently would overwrite each other's SKILL.md mid-run.
if [ -n "${INSTALL_DIR:-}" ]; then
  OVERRIDES+=(--harness.install_dir "${INSTALL_DIR}")
fi

if [ -n "${CONTINUAL_HARNESS:-}" ]; then
  case " ${ARM} " in
    *" prime_agent "*|*" prime_agent_b80 "*) ;;
    *)
      echo "launch_cell: CONTINUAL_HARNESS is a prime_agent knob (the store is" >&2
      echo "  mounted into that harness's per-rollout agent dir). Arm '${ARM}'" >&2
      echo "  has no such store. Refusing." >&2
      exit 2
      ;;
  esac
  if [ -z "${CONTINUAL_RUN_ID:-}" ]; then
    echo "launch_cell: CONTINUAL_HARNESS requires CONTINUAL_RUN_ID -- an" >&2
    echo "  unlabelled continual cell cannot be told apart from another" >&2
    echo "  experiment's once it is on disk." >&2
    exit 2
  fi
  OVERRIDES+=(--harness.continual_harness_dir "${CONTINUAL_HARNESS}")
  if [ -n "${CONTINUAL_HARNESS_MODE:-}" ]; then
    # shared-ro = one store, symlinked, read-only for players (single writer).
    # copy-merge = a private copy per rollout, merged afterwards -- the arm where
    # players write. Measured: five concurrent writers on ONE store kept 12 of
    # 30 entries and one of the five contributed nothing, because rlm.harness
    # persists with a non-atomic whole-file rewrite and no lock.
    OVERRIDES+=(--harness.continual_harness_mode "${CONTINUAL_HARNESS_MODE}")
  fi
  if [ -n "${CONTINUAL_SELF_EDIT:-}" ]; then
    # Provisioning a writable store is not instruction: without this block the
    # player is never told the store exists, and E9 measured exactly one
    # spontaneous write in 15 rollouts.
    OVERRIDES+=(--taskset.env_args.continual_self_edit "${CONTINUAL_SELF_EDIT}")
  fi
  if [ -n "${CONTINUAL_SPEC_SHA:-}" ]; then
    OVERRIDES+=(--taskset.env_args.continual_spec_sha "${CONTINUAL_SPEC_SHA}")
  fi
  OVERRIDES+=(--taskset.env_args.continual_run_id "${CONTINUAL_RUN_ID}")
  if [ -n "${CONTINUAL_PROMPT_SHA:-}" ]; then
    # Which reflection instructions produced this store. Two runs that differ
    # only in the orchestrator's prompt are otherwise identical on disk.
    OVERRIDES+=(--taskset.env_args.continual_prompt_sha "${CONTINUAL_PROMPT_SHA}")
  fi
  if [ -n "${CONTINUAL_HARNESS_WRITABLE:-}" ]; then
    OVERRIDES+=(--harness.continual_harness_writable "${CONTINUAL_HARNESS_WRITABLE}")
  fi
  echo "[launch_cell] continual: run_id=${CONTINUAL_RUN_ID} store=${CONTINUAL_HARNESS}"
fi

# E16_ARGS: a JSON object of E16 Go-Explore env_args, allowed ALONGSIDE
# TOOL_TIER. That is the difference from ENV_ARGS, and it is a whitelist, not a
# relaxation.
#
# ENV_ARGS is refused next to TOOL_TIER because both write arbitrary
# `--taskset.env_args.*` paths and the eval CLI resolves duplicates last-wins,
# SILENTLY -- measured: ENV_ARGS='{"netplay_telemetry":true}' with TOOL_TIER=base
# produced a cell labelled base running a human-tier fix. That reasoning is about
# COLLISION, not about the mechanism: keys no tier can ever write cannot collide.
#
# So this knob accepts exactly five keys, all of which name E16 run RESOURCES
# (which archive, which pages, which checkpoint, where to log) rather than
# experiment factors. Every experiment factor -- doc, model, surface, encoding,
# budget, fix flags -- still comes only from the tier. The whitelist is enforced
# below and an unknown key is a hard error, so this cannot become a second,
# quieter ENV_ARGS.
#
#   resume_checkpoint   archive/<run>/c<id> this rollout resumes from
#   checkpoint_archive  archive root the published `save` skill writes into
#   wiki_dir            the run's own COPY of the curated wiki pages
#   ledger_text         orchestrator-rendered ledger for the first observation
#   fidelity_log        JSONL the restore-fidelity audit appends to
#   directive           the orchestrator's instruction for THIS attempt, served
#                       verbatim in the player's first observation
#   reseed              "[core, disp]" to reseed the RNG with after the restore,
#                       or absent for the deterministic default. It belongs
#                       here and not in a tier for the same reason
#                       resume_checkpoint does: it is derived PER ATTEMPT (from
#                       the run's rng_seed, the checkpoint id and the attempt
#                       index), so no fixed tier could carry it. WHETHER a run
#                       reseeds at all is the experiment factor, and that lives
#                       in the orchestrator's --reseed flag and in
#                       provenance.json, where a reader can find it.
if [ -n "${E16_ARGS:-}" ]; then
  case " ${ARM} " in
    *" prime_agent "*|*" prime_agent_b80 "*|*" claude_code "*) ;;
    *)
      echo "launch_cell: E16_ARGS is for the CLI arms (it writes taskset.env_args)." >&2
      exit 2
      ;;
  esac
  # NUL-SEPARATED, not newline-separated, and this is load-bearing rather than
  # tidy. `ledger_text` is the rendered archive table plus the checkpoint's
  # lessons plus its quoted conversation prefix: it is MULTI-LINE by
  # construction, always. A `mapfile -t` over newline-separated records splits
  # one such value into one array element per LINE, and every line after the
  # first then reaches the eval CLI as a bare positional argument -- which it
  # rejects with "Unrecognized arguments: id Dlvl XL HP turn score ...".
  #
  # Measured: that is not an edge case, it is EVERY E16 attempt. The 4-attempt
  # dry run never saw it because a stub player never goes through this script,
  # and the model-in-the-loop sims never saw it because they passed only a
  # single-line `directive`. The first real resumed rollout hit it immediately.
  # VIA A FILE, NOT A PROCESS SUBSTITUTION, and that is the second bug this
  # block had. `mapfile ... < <(cmd) || { exit 2; }` binds the `||` to MAPFILE,
  # whose status has nothing to do with `cmd`'s -- so the whitelist rejection
  # below printed its error, returned 2, and the launch CONTINUED with an empty
  # flag array. A cell that silently lost its resume_checkpoint, its ledger and
  # its directive is the exact silent-substitution class that invalidated E15,
  # and it exited 0 while doing it.
  _E16_OUT="$(mktemp "${TMPDIR:-/tmp}/e16args.XXXXXX")"
  if ! E16_ARGS="${E16_ARGS}" "$PY_BIN" - > "$_E16_OUT" <<'PYE16'
import json, os, sys
ALLOWED = {"resume_checkpoint", "checkpoint_archive", "wiki_dir",
           "ledger_text", "fidelity_log", "directive", "reseed"}
try:
    obj = json.loads(os.environ["E16_ARGS"])
except Exception as exc:
    print(f"launch_cell: E16_ARGS is not valid JSON: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not isinstance(obj, dict):
    print("launch_cell: E16_ARGS must be a JSON object", file=sys.stderr)
    raise SystemExit(2)
bad = sorted(set(obj) - ALLOWED)
if bad:
    print(f"launch_cell: E16_ARGS keys not allowed: {bad}. Allowed: "
          f"{sorted(ALLOWED)}. An experiment FACTOR belongs in "
          f"configs/tool_tiers.toml as its own tier, never here.", file=sys.stderr)
    raise SystemExit(2)
for k in sorted(obj):
    v = obj[k]
    if isinstance(v, (dict, list)):
        print(f"launch_cell: E16_ARGS.{k} must be a scalar", file=sys.stderr)
        raise SystemExit(2)
    val = v if isinstance(v, str) else json.dumps(v)
    if "\0" in val:
        print(f"launch_cell: E16_ARGS.{k} contains a NUL byte", file=sys.stderr)
        raise SystemExit(2)
    sys.stdout.write(f"--taskset.env_args.{k}\0")
    sys.stdout.write(val + "\0")
PYE16
  then
    rm -f "$_E16_OUT"
    echo "launch_cell: E16_ARGS rejected (see above)." >&2
    exit 2
  fi
  mapfile -d '' -t _E16_FLAGS < "$_E16_OUT"
  rm -f "$_E16_OUT"
  # A value that is multi-line must arrive as ONE argv element. If the count is
  # odd, the flag/value pairing broke and the next thing that happens is a
  # rollout launched with a truncated ledger -- served bytes that no config
  # would show as wrong. Refuse instead.
  if [ $(( ${#_E16_FLAGS[@]} % 2 )) -ne 0 ]; then
    echo "launch_cell: E16_ARGS produced ${#_E16_FLAGS[@]} argv items (odd);" >&2
    echo "  flag/value pairing is broken and the served bytes would be wrong." >&2
    exit 2
  fi
  OVERRIDES+=("${_E16_FLAGS[@]}")
  # The ledger text can be long; log the KEYS only, and let the resolved
  # config.toml carry the values (which is where an audit should read them).
  echo "[launch_cell] E16 env_args: $(printf '%s\n' "${_E16_FLAGS[@]}" | grep '^--' | tr '\n' ' ')"
fi

echo "[launch_cell] arm=${ARM} config=${CFG} model=${MODEL:-<from config>} variant=${VARIANT:-<from config>} max_calls=${MAX_CALLS} n=${N} timeout=${ROLLOUT_TIMEOUT:-<from config>} out=${OUT_ABS} trace_dir=${TRACE_DIR}"

# ROLLOUT_MAX_RETRIES: the WHOLE-ROLLOUT retry bound (verifiers'
# `[retries.rollout] max_retries`). This is a COST MULTIPLIER, not a reliability
# knob: `verifiers/v1/retries.py:run_with_retry` replays the entire trajectory
# from turn 1, so an attempt with max_retries=2 can pay for THREE full rollouts.
# Measured (E16 method test): a provider `429 Rate limit reached` storm drove
# `retry 1/2` then `retry 2/2`, billing $31.43 for one attempt that then
# recorded `censored:harness_error`.
#
# WHY THE ARMS DEFAULT TO 1 AND NOT 2. The knob was added to survive Prime's
# intermittent `finish_reason: "error"` (see the arm configs) -- a FIRST-CALL
# schema rejection that costs nothing to replay and is independent per attempt,
# so a single retry already recovers essentially all of it. The second retry
# only ever pays off on a failure that is CORRELATED in time, and the headline
# correlated failure is a rate-limit storm -- which `run_with_retry` re-enters
# immediately, because unlike the framework's `retrying()` policy it passes no
# `wait=` at all, so there is no backoff between attempts. Retry 2 therefore
# buys a third full rollout at the exact moment retrying cannot work. Capping
# at 1 halves the worst-case multiplier (3x -> 2x); on a $385 run that is the
# difference between $1,155 and $770 of exposure.
#
# Set explicitly to override; hard-capped, because a typo here is precisely the
# unbounded-cost failure this exists to bound.
if [ -n "${ROLLOUT_MAX_RETRIES:-}" ]; then
  case "${ROLLOUT_MAX_RETRIES}" in
    ''|*[!0-9]*)
      echo "launch_cell: ROLLOUT_MAX_RETRIES must be a non-negative integer" \
           "(got '${ROLLOUT_MAX_RETRIES}')" >&2
      exit 2 ;;
  esac
  if [ "${ROLLOUT_MAX_RETRIES}" -gt "${ROLLOUT_MAX_RETRIES_CAP:-2}" ]; then
    echo "launch_cell: ROLLOUT_MAX_RETRIES=${ROLLOUT_MAX_RETRIES} exceeds the cap" \
         "of ${ROLLOUT_MAX_RETRIES_CAP:-2}. Each retry replays a WHOLE rollout;" >&2
    echo "  N retries means an attempt can bill (N+1)x. Raise" \
         "ROLLOUT_MAX_RETRIES_CAP deliberately if that is really intended." >&2
    exit 2
  fi
  OVERRIDES+=("--retries.rollout.max_retries" "${ROLLOUT_MAX_RETRIES}")
  echo "[launch_cell] rollout retries: max_retries=${ROLLOUT_MAX_RETRIES}" \
       "(an attempt can bill up to $((ROLLOUT_MAX_RETRIES + 1)) full rollouts)"
fi

# EXTRA_EVAL_FLAGS: whitespace-separated eval-CLI flags appended LAST (they win
# on duplicate dotted paths). Added for the localhost interception override
# (--interception.tunnel.type custom ...): the default prime tunnel counts
# against a 32-tunnel team quota, and quota exhaustion killed whole cells with
# HarnessError at boot (E15 r1 groups 1-2, 2026-08-26). Infra-transport only —
# never put experiment factors here; those belong in the tier registry.
if [ -n "${EXTRA_EVAL_FLAGS:-}" ]; then
  read -r -a _EXTRA_EVAL <<< "${EXTRA_EVAL_FLAGS}"
  OVERRIDES+=("${_EXTRA_EVAL[@]}")
  echo "[launch_cell] extra eval flags: ${EXTRA_EVAL_FLAGS}"
fi

exec "${EVAL_BIN}" @ "${CFG}" \
  --num_tasks "${N}" \
  --output_dir "${OUT_ABS}" \
  "${OVERRIDES[@]}"
