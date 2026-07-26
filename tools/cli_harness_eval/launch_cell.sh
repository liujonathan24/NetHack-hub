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

KNOWN_ARMS="control claude_code prime_agent"

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

echo "[launch_cell] arm=${ARM} config=${CFG} max_calls=${MAX_CALLS} n=${N} out=${OUT_ABS} trace_dir=${TRACE_DIR}"

exec "${EVAL_BIN}" @ "${CFG}" \
  --num_tasks "${N}" \
  --output_dir "${OUT_ABS}" \
  "${OVERRIDES[@]}"
