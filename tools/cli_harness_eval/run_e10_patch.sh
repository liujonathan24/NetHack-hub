#!/usr/bin/env bash
# E10 patch runs: rerun the three flagged games of the clean re-baseline as
# fresh single-seed rollouts (full 200-call budget, same honest-harness config):
#   r2 seed 3  -- wedge-truncated at 52 calls (daemon stall, never retried)
#   r3 seed 0  -- flagged in the results table
#   r3 seed 2  -- early exit alive at 34 calls
# Separate patch dirs so the parent cells' traces.jsonl are untouched; fold the
# seed's turn file back at analysis time (the per-seed selector picks max rows).
set -uo pipefail
REPO=/root/nld/hub-eval
export ENG=/root/NetHack-engine
export EVAL_BIN=/root/NetHack-hub/.venv-cli-eval/bin/eval
cd "$REPO"
BASE='{"skill_set":"np_core,request_map,search","auto_dismiss":"false","tune":{"reveal_map":1.0}'

echo "[reset] $(date -u +%H:%M:%S) daemon teardown (once)"
prime-agent shutdown >/dev/null 2>&1 || true
pkill -9 -f 'prime-agent'  2>/dev/null || true
pkill -9 -f 'nethack_v1'   2>/dev/null || true
pkill -9 -f 'kernel-venv'  2>/dev/null || true
rm -rf /tmp/prime-agent-0 2>/dev/null || true
s=$(date +%s)
[ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers "/root/.prime/agent/daemon-workers.bak-$s" 2>/dev/null || true
[ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases "/root/.prime/agent/session-leases.bak-$s" 2>/dev/null || true
sleep 3

pids=(); names=()
launch() {  # <name> <seed>
  local name="$1" seed="$2"
  local out="outputs/e10_baseline/PATCH_${name}__prime_agent"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out (seed $seed)"
  ENV_ARGS="${BASE},\"explicit_seeds\":[${seed}]}" VARIANT=BBOX_MIN \
    "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$out" 200 1 \
    > "/tmp/e10_patch_${name}.log" 2>&1 &
  pids+=($!); names+=("$name")
}
launch r2s3 3
sleep 45
launch r3s0 0
sleep 45
launch r3s2 2

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" && echo "[done ] $(date -u +%H:%M:%S) OK  ${names[$i]}" \
                     || { echo "[FAIL ] $(date -u +%H:%M:%S) ${names[$i]} rc=$?"; rc=1; }
done
echo "[all  ] $(date -u +%H:%M:%S) E10 patch runs finished (rc=$rc)"
