#!/usr/bin/env bash
# E10 re-baseline, PARALLEL: one daemon reset, then all three replicate cells
# launched concurrently (15 rollouts in flight). Cell starts staggered 60s so
# the first invocation brings the daemon + skill materialization up alone.
set -uo pipefail
REPO=/root/nld/hub-eval
export ENG=/root/NetHack-engine
export EVAL_BIN=/root/NetHack-hub/.venv-cli-eval/bin/eval
cd "$REPO"
ARGS='{"skill_set":"np_core,request_map,search","auto_dismiss":"false","tune":{"reveal_map":1.0}}'

echo "[reset] $(date -u +%H:%M:%S) daemon teardown (once, up front)"
prime-agent shutdown >/dev/null 2>&1 || true
pkill -9 -f 'prime-agent'  2>/dev/null || true
pkill -9 -f 'nethack_v1'   2>/dev/null || true
pkill -9 -f 'kernel-venv'  2>/dev/null || true
rm -rf /tmp/prime-agent-0 2>/dev/null || true
s=$(date +%s)
[ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers "/root/.prime/agent/daemon-workers.bak-$s" 2>/dev/null || true
[ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases "/root/.prime/agent/session-leases.bak-$s" 2>/dev/null || true
sleep 3

pids=()
for r in r1 r2 r3; do
  out="outputs/e10_baseline/NPCORE_v3H_${r}__prime_agent"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out (parallel)"
  ENV_ARGS="$ARGS" VARIANT=BBOX_MIN \
    "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$out" 200 5 \
    > "/tmp/e10_${r}.cell.log" 2>&1 &
  pids+=($!)
  [ "$r" != "r3" ] && sleep 60
done
rc=0
for i in 0 1 2; do
  wait "${pids[$i]}" && echo "[done ] $(date -u +%H:%M:%S) OK  cell $((i+1))" \
                     || { echo "[FAIL ] $(date -u +%H:%M:%S) cell $((i+1)) rc=$?"; rc=1; }
done
echo "[all  ] $(date -u +%H:%M:%S) E10 parallel re-baseline finished (rc=$rc)"
