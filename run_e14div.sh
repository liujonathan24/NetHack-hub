#!/usr/bin/env bash
# E14 Diversity pilot: seed 1, one rep, one cell per model.
#
# Private stack throughout (the eval box is multi-tenant):
#   EVAL_BIN                   my byte-verified venv, not the shared one
#   NETHACK_PA_AGENT_BIND_SRC  my own copy of ~/.prime/agent, bound over the
#                              real path in-sandbox -- a booting experiment
#                              scanning the shared dir reaps live sessions
#   INSTALL_DIR                per-cell, so no two cells share a daemon
# TMPDIR is deliberately NOT exported: rollouts get private sockets from
# bwrap --tmpfs /tmp, and exporting it collapses them onto one socket.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export EVAL_BIN=/root/nld/.venv-e1315/bin/eval
export NETHACK_PA_AGENT_BIND_SRC=/root/nld/.prime-agent-e14div
export ENG=/root/NetHack-engine
MODELS="${*:-luna dsv4flash gemini37 qwen38 sonnet5 kimik3 sol fable5}"
for m in $MODELS; do
  OUT="outputs/e14_diversity/${m}__prime_agent"
  mkdir -p "$OUT"
  echo "[e14div] launching ${m} -> ${OUT}"
  env TOOL_TIER="e14div_${m}" SEEDS='[1]' \
      INSTALL_DIR="/tmp/vf-pa-e14div-${m}" \
      ./tools/cli_harness_eval/launch_cell.sh prime_agent "$OUT" 0 1 \
      > "${OUT}/launch.log" 2>&1 &
  sleep 8
done
wait
echo "[e14div] all cells returned"
