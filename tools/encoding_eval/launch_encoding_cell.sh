#!/usr/bin/env bash
# Launch ONE cell of the encoding sweep.
#
#   tools/encoding_eval/launch_encoding_cell.sh <VARIANT> <OUTDIR> [MAX_TURNS] [N]
#   tools/encoding_eval/launch_encoding_cell.sh JSON outputs/encoding_eval/run1/JSON 400 5
#
# Every cell goes through THIS script and shares configs/encoding_base.toml, so
# the fixed factors (model, seeds, character, task_spec, skill_set, compaction)
# cannot drift between cells. Only `variant` and `trace_dir` differ.
#
# Mirrors tools/cli_harness_eval/launch_cell.sh's role for the harness arms,
# adapted for the encoding axis.
set -euo pipefail

# Mirrors nethack_harness.prompt.prompt_spec.VARIANT_REGISTRY. DM / DM_JSON /
# BBOX are sub-experiment 1d's observation-timing cells (delayed map, and the
# bounding-box reveal) — 1d is driven by the VARIANT, not by a kwarg: there is
# no `obs_mode` parameter on `load_environment` (that field lives on the v1
# taskset config), so passing one would land in **kwargs and be silently
# ignored, producing a cell identical to its baseline.
KNOWN="SPARSE_MEM BBOX_MEM SPARSE SPARSE_ONDEMAND BBOX_GUIDE_LEAD BBOX_GUIDE_LAG B0_GUIDE_LEAD B0_GUIDE_LAG B0 B1 JSON TOON IMG_TTY IMG B B_ASCII B_JSON DM_B_ASCII BBOX_JSON N R DM DM_JSON BBOX GLYPHBOX NETPLAY E1 E2 ND FD CH P G"

usage() {
  echo "usage: launch_encoding_cell.sh <VARIANT> <OUTDIR> [MAX_TURNS] [N]" >&2
  echo "  VARIANT one of: ${KNOWN}" >&2
}
[ "$#" -ge 2 ] || { usage; exit 2; }

VARIANT="$1"
OUTDIR="$2"
MAX_TURNS="${3:-400}"
N="${4:-5}"

case " ${KNOWN} " in
  *" ${VARIANT} "*) ;;
  *) echo "unknown variant: ${VARIANT}" >&2; usage; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ENG="${ENG:-/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness}"
EVAL_BIN="${EVAL_BIN:-${REPO}/.venv-cli-eval/bin/eval}"
[ -x "$EVAL_BIN" ] || { echo "missing eval CLI: $EVAL_BIN" >&2; exit 2; }

# tools/pycompat FIRST: its sitecustomize.py is imported at interpreter start in
# this process and every worker subprocess, widening ChatCompletion.service_tier
# which Prime intermittently returns as "provisioned" — a value no released
# OpenAI SDK accepts. Without it, rollouts die mid-run with a pydantic
# ValidationError, and only SOME seeds are hit, so the failure looks random.
export PYTHONPATH="${REPO}/tools/pycompat:${ENG}:${REPO}:${REPO}/environments/nethack${PYTHONPATH:+:${PYTHONPATH}}"
export PI_API_KEY="${PI_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")}"

mkdir -p "$OUTDIR"
OUT_ABS="$(cd "$OUTDIR" && pwd)"
mkdir -p "${OUT_ABS}/turns"

# One JSON blob, not scalar `--args.x` overrides: a scalar override arrives as a
# STRING and reaches load_environment as e.g. max_turns="400", which fails only
# once a real rollout is underway (see launch_cell.sh's note on the same trap).
# EXTRA_ARGS is a JSON object merged on top, for ablation cells that vary a
# knob other than `variant` — e.g. the JSON cell-content arms
#   EXTRA_ARGS='{"cell_schema":["seen","visited"]}'
# or the memory arms
#   EXTRA_ARGS='{"belief_state_interval":0,"pin_objective_on_setup":false}'
# Everything else stays pinned by encoding_base.toml, so an ablation cell
# differs from its baseline in exactly the keys named here.
ARGS_JSON="$(python3 -c "
import json,sys
d = {
  'variant': sys.argv[1],
  'max_turns': int(sys.argv[2]),
  'trace_dir': sys.argv[3] + '/turns',
}
extra = sys.argv[4]
if extra:
    d.update(json.loads(extra))
print(json.dumps(d))" "$VARIANT" "$MAX_TURNS" "${OUT_ABS}" "${EXTRA_ARGS:-}")"

echo "[cell] variant=${VARIANT} max_turns=${MAX_TURNS} n=${N}"
echo "[cell] out=${OUT_ABS}"
echo "[cell] args=${ARGS_JSON}"

# Optional per-cell model override. The base config pins
# google/gemini-3-flash-preview as a FIXED factor for the encoding sweep, so the
# model-strength arms need it settable without forking the config -- a second
# toml would drift from the base and silently reintroduce exactly the confounds
# the single-base design exists to prevent.
MODEL_ARGS=()
if [ -n "${MODEL:-}" ]; then
  MODEL_ARGS=(--model "${MODEL}")
  echo "[cell] model=${MODEL} (override)"
fi

# --- stall watchdog (opt-in: STALL_WATCHDOG=1) --------------------------------
# Same contract as tools/cli_harness_eval/launch_cell.sh — see the long note
# there. Short version: HARNESS_DEFECTS §3.1's two hangs never reach `env.step`,
# so nothing in-process can time them out; tools/stall_watchdog.py watches
# `turns/` from outside. Armed per launcher invocation (== per batch) because
# the documented failure mode is a watchdog that exits when one batch's queue
# drains. `--parent-pid $$` works because the `exec` below keeps this PID.
# Opt-in so an unset env var leaves a normal foreground run untouched.
if [ -n "${STALL_WATCHDOG:-}" ]; then
  WD_PY="$(dirname "${EVAL_BIN}")/python"
  [ -x "$WD_PY" ] || WD_PY=python3
  "$WD_PY" "${REPO}/tools/stall_watchdog.py" \
    --turns-dir "${OUT_ABS}/turns" \
    --timeout "${STALL_TIMEOUT:-300}" \
    --poll "${STALL_POLL:-15}" \
    --parent-pid "$$" \
    ${STALL_EXTRA_ARGS:-} >>"${OUT_ABS}/stall_watchdog.log" 2>&1 &
  echo "[cell:watchdog] armed: pid=$! timeout=${STALL_TIMEOUT:-300}s" \
       "log=${OUT_ABS}/stall_watchdog.log quarantine=${OUT_ABS}/turns.stalled"
fi

exec "$EVAL_BIN" @ tools/encoding_eval/configs/encoding_base.toml \
  --num_tasks "${N}" \
  --output_dir "${OUT_ABS}" \
  "${MODEL_ARGS[@]}" \
  --args "${ARGS_JSON}"
