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

export PYTHONPATH="${ENG}:${REPO}:${REPO}/environments/nethack${PYTHONPATH:+:${PYTHONPATH}}"

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
  OVERRIDES=(--taskset.max_skill_calls "${MAX_CALLS}" --taskset.trace_dir "${TRACE_DIR}")
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

# ROLLOUT_TIMEOUT raises the per-rollout wall-clock cap. The default 7200s (2h)
# is what actually ended live games in the b80 cells -- two of five seeds
# stopped at `harness_timeout` while the 1200-call budget never bound -- so any
# long-horizon cell MUST raise it alongside MAX_CALLS or it silently measures
# the clock instead of the agent. At the measured ~5.3s/call median, 10k calls
# needs ~15h; 108000 (30h) leaves headroom for the latency tail.
if [ -n "${ROLLOUT_TIMEOUT:-}" ]; then
  OVERRIDES+=(--timeout.rollout "${ROLLOUT_TIMEOUT}")
fi

echo "[launch_cell] arm=${ARM} config=${CFG} model=${MODEL:-<from config>} variant=${VARIANT:-<from config>} max_calls=${MAX_CALLS} n=${N} timeout=${ROLLOUT_TIMEOUT:-<from config>} out=${OUT_ABS} trace_dir=${TRACE_DIR}"

exec "${EVAL_BIN}" @ "${CFG}" \
  --num_tasks "${N}" \
  --output_dir "${OUT_ABS}" \
  "${OVERRIDES[@]}"
