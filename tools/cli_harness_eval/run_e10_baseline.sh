#!/usr/bin/env bash
# E10: re-baseline NPCORE_v3 on the honest harness (post harness-honesty pass,
# commit fd8aa13). Three replicate cells x 5 seeds, identical config to the old
# NPCORE_v3 control so before/after isolates the docs/harness fixes.
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
  rm -rf /tmp/prime-agent-0 2>/dev/null || true
  local s; s=$(date +%s)
  [ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers "/root/.prime/agent/daemon-workers.bak-$s" 2>/dev/null || true
  [ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases "/root/.prime/agent/session-leases.bak-$s" 2>/dev/null || true
  sleep 3
}

run_cell() {
  local out="$1"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out"
  reset_daemon
  ENV_ARGS='{"skill_set":"np_core,request_map,search","auto_dismiss":"false","tune":{"reveal_map":1.0}}' VARIANT=BBOX_MIN \
    "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$out" 200 5 \
    && echo "[done ] $(date -u +%H:%M:%S) OK  $out" \
    || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $out"
}

run_cell outputs/e10_baseline/NPCORE_v3H_r1__prime_agent
run_cell outputs/e10_baseline/NPCORE_v3H_r2__prime_agent
run_cell outputs/e10_baseline/NPCORE_v3H_r3__prime_agent
echo "[all  ] $(date -u +%H:%M:%S) E10 re-baseline finished"
