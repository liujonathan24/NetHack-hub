#!/usr/bin/env bash
# Cost/latency calibration for the encoding experiments.
#
# Runs the BASE cell — B0 uncompacted — on the real NetPlay action surface
# (`netplay_true`, the 31 vendored upstream skills) so the measured cost
# reflects what the actual exp1 rerun will spend, not the old 18-tool surface.
#
# Death terminates normally (the hp==0 path), so this measures real programme
# cost rather than a forced 400-turn ceiling. Per-turn cost is reported by the
# aggregator, so a full-length ceiling can still be extrapolated from it.
#
# Usage: tools/encoding_eval/launch_calib.sh [OUTDIR] [MAX_TURNS] [N] [MODEL]
set -uo pipefail

OUTDIR="${1:-outputs/encoding_eval/calib_pro_b0}"
MT="${2:-400}"
N="${3:-5}"
# NB: `google/gemini-3-pro-preview` DOES NOT EXIST on Prime — verified by API
# ("The model 'google/gemini-3-pro-preview' does not exist"). 3.1 is the real id
# and is the one listed in the funded prime-team block of configs/endpoints.toml.
MODEL="${4:-google/gemini-3.1-pro-preview}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
# The compiled engine (`nethack_core`) lives in the NetHack-engine repo and must
# lead PYTHONPATH; it is not pip-installed. Override with ENGINE_REPO if needed.
ENGINE_REPO="${ENGINE_REPO:-/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness}"
export PYTHONPATH="${ENGINE_REPO}:.:environments/nethack"
export PI_API_KEY="${PI_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")}"

# Absolute, so traces survive regardless of the runner's working directory.
mkdir -p "$OUTDIR"; OUTDIR="$(cd "$OUTDIR" && pwd)"; mkdir -p "$OUTDIR/trace"

SEEDS="[0,1,2,3,4]"
ARGS="{\"task_spec\":\"full_nle\",\"variant\":\"B0\",\"compact_obs\":false,\"skill_set\":\"netplay_true\",\"map_detail\":\"full\",\"max_turns\":${MT},\"character\":\"Val-hum-neu-fem\",\"explicit_seeds\":${SEEDS},\"trace_dir\":\"${OUTDIR}/trace\"}"

echo "[calib] model=${MODEL} variant=B0 compact_obs=false skill_set=netplay_true"
echo "[calib] max_turns=${MT} n=${N} seeds=${SEEDS}"
echo "[calib] out=${OUTDIR}"
# No `-p prime`: that overrides the registry and drops the X-Prime-Team-ID
# billing header. The model resolves from the prime-team block instead.
.venv-cli-eval/bin/vf-eval nethack --env-dir-path environments \
  -m "$MODEL" --endpoints-path configs/endpoints.toml \
  -a "$ARGS" -n "$N" -r 1 -c 3 --num-workers 3 --max-tokens 2048 \
  --save-results --output-dir "$OUTDIR" --disable-tui
echo "[calib] vf-eval exit=$?"
