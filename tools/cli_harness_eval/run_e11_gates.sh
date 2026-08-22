#!/usr/bin/env bash
# E11: the two descent-gate interventions rerun on the HONEST harness
# (post-fd8aa13 docs + tune.reveal_map=1.0 + auto_dismiss=false), against the
# E10 baseline (median 3.54 / mean 5.34 / 14 of 15 died / ceiling dl11):
#   NORM    -- ignorable gate (E8a): first descent per level returns the
#              human-norm line at zero cost; repeating the call descends.
#   ENFORCE -- forced gate (E9a): descent refused, no override, until
#              XL >= norm(Dlvl). Ascent never gated.
# Both cells launched in parallel (10 rollouts in flight), staggered 60s.
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
launch() {  # <name> <gate>
  local name="$1" gate="$2"
  local out="outputs/e11_gates/${name}__prime_agent"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out (descent_gate=$gate)"
  ENV_ARGS="${BASE},\"descent_gate\":\"${gate}\"}" VARIANT=BBOX_MIN \
    "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$out" 200 5 \
    > "/tmp/e11_${name}.cell.log" 2>&1 &
  pids+=($!); names+=("$name")
}
launch E11A_NORM norm
sleep 60
launch E11B_ENFORCE enforce

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" && echo "[done ] $(date -u +%H:%M:%S) OK  ${names[$i]}" \
                     || { echo "[FAIL ] $(date -u +%H:%M:%S) ${names[$i]} rc=$?"; rc=1; }
done
echo "[all  ] $(date -u +%H:%M:%S) E11 gate reruns finished (rc=$rc)"
