#!/usr/bin/env bash
# E12: NPCORE_v3 on the honest harness + FIXED NetPlay tools (PR #29:
# rest works, trivia interrupts filtered, move/melee telemetry, '-' key).
# Same 5 seeds / config as the E10 baseline; measures recovered budget and
# whether working tools move survival. Run from fix/netplay-telemetry.
set -uo pipefail
REPO=/root/nld/hub-eval
export ENG=/root/NetHack-engine
export EVAL_BIN=/root/NetHack-hub/.venv-cli-eval/bin/eval
cd "$REPO"

echo "[reset] $(date -u +%H:%M:%S) daemon teardown"
prime-agent shutdown >/dev/null 2>&1 || true
pkill -9 -f 'prime-agent'  2>/dev/null || true
pkill -9 -f 'nethack_v1'   2>/dev/null || true
pkill -9 -f 'kernel-venv'  2>/dev/null || true
rm -rf /tmp/prime-agent-0 2>/dev/null || true
s=$(date +%s)
[ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers "/root/.prime/agent/daemon-workers.bak-$s" 2>/dev/null || true
[ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases "/root/.prime/agent/session-leases.bak-$s" 2>/dev/null || true
sleep 3

OUT=outputs/e12_fixed_tools/NPCORE_v3F__prime_agent
echo "[cell ] $(date -u +%H:%M:%S) -> $OUT"
ENV_ARGS='{"skill_set":"np_core,request_map,search","auto_dismiss":"false","tune":{"reveal_map":1.0}}' \
  VARIANT=BBOX_MIN \
  "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$OUT" 200 5 \
  && echo "[done ] $(date -u +%H:%M:%S) OK  $OUT" \
  || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $OUT"
echo "[all  ] $(date -u +%H:%M:%S) E12 finished"
