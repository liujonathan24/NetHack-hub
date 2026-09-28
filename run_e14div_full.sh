#!/usr/bin/env bash
# E14 Diversity, full grid: 5 models x seeds 0-4 x 3 reps = 75 rollouts.
#
# Structure mirrors the arm exactly: one CELL per model per rep, each cell
# running the contract's 5 seeds. No SEEDS override -- the contract names
# [0,1,2,3,4] and N=5 matches it, so launch_cell does not mark the cell
# seeds_overridden the way the seed-1 pilot did.
#
# Waves by REP, not by model: 5 concurrent cells (25 concurrent rollouts) is
# the same shape the arm ran at, and staging keeps a wedged daemon from taking
# the whole grid down with it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export EVAL_BIN=/root/nld/.venv-e1315/bin/eval
export NETHACK_PA_AGENT_BIND_SRC=/root/nld/.prime-agent-e14div
export ENG=/root/NetHack-engine
MODELS="luna sol gemini37 qwen38"   # dsv4flash dropped 2026-09-01: 2/5 rollouts censored as agent_completed, ~5x the wall clock of any other cell
REP="${1:?usage: run_e14div_full.sh <rep>}"
for m in $MODELS; do
  OUT="outputs/e14_diversity_full/${m}_r${REP}__prime_agent"
  mkdir -p "$OUT"
  echo "[e14div] rep${REP} ${m} -> ${OUT}"
  env TOOL_TIER="e14div_${m}" INSTALL_DIR="/tmp/vf-pa-e14divfull-${m}-r${REP}" \
      ./tools/cli_harness_eval/launch_cell.sh prime_agent "$OUT" 0 5 \
      > "${OUT}/launch.log" 2>&1 &
  sleep 10
done
wait
echo "[e14div] rep${REP} complete"
