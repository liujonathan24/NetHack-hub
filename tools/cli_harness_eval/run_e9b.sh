#!/usr/bin/env bash
# E9b: NPCORE_v3 + reflect=on (per-turn reflection prompt). Byte-identical to the
# control otherwise (np_core surface, BBOX_MIN, 5 seeds). Same daemon-reset
# recipe as run_e9.sh.
set -uo pipefail
REPO=/root/nld/hub-eval
export ENG=/root/NetHack-engine
export EVAL_BIN=/root/NetHack-hub/.venv-cli-eval/bin/eval
cd "$REPO"

reset_daemon() {
  echo "[reset] $(date -u +%H:%M:%S) daemon teardown"
  prime-agent shutdown >/dev/null 2>&1 || true
  pkill -9 -f 'prime-agent'  2>/dev/null || true
  pkill -9 -f 'nethack_v1'   2>/dev/null || true
  pkill -9 -f 'kernel-venv'  2>/dev/null || true
  pkill -9 -f 'provider intercept' 2>/dev/null || true
  rm -rf /tmp/prime-agent-0 2>/dev/null || true
  local s; s=$(date +%s)
  [ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers "/root/.prime/agent/daemon-workers.bak-e9b-$s" 2>/dev/null || true
  [ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases "/root/.prime/agent/session-leases.bak-e9b-$s" 2>/dev/null || true
  sleep 3
}

OUT=outputs/e9_reflect/REFLECT__prime_agent
echo "[cell ] $(date -u +%H:%M:%S) -> $OUT"
reset_daemon
MAX_CONCURRENT=3 ENV_ARGS='{"skill_set":"np_core,request_map,search","reflect":"on"}'' VARIANT=BBOX_MIN \
  "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$OUT" 200 5 \
  && echo "[done ] $(date -u +%H:%M:%S) OK  $OUT" \
  || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $OUT"
echo "[all  ] $(date -u +%H:%M:%S) E9b finished"
