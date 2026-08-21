#!/usr/bin/env bash
# E9 launch: extend the NPCORE_v3 control to 3 runs/seed + the E9a enforce cell.
# Every cell is byte-identical to the NPCORE_v3 control (np_core reduced NetPlay
# surface, BBOX_MIN, GLM-5.2, Valkyrie, 200 turns, 5 seeds); the enforce cell
# adds the single `descent_gate=enforce` knob. Cells run SEQUENTIALLY with a
# full prime-agent daemon reset before each (the wedge recipe), which is the
# pattern that held across E8 -- seeds within a cell still run concurrently.
set -uo pipefail

REPO=/root/nld/hub-eval
export ENG=/root/NetHack-engine
export EVAL_BIN=/root/NetHack-hub/.venv-cli-eval/bin/eval
LAUNCH="$REPO/tools/cli_harness_eval/launch_cell.sh"
CORE='{"skill_set":"np_core,request_map,search"}'
ENFORCE='{"skill_set":"np_core,request_map,search","descent_gate":"enforce"}'
cd "$REPO"

reset_daemon() {
  echo "[reset] $(date -u +%H:%M:%S) tearing down prime-agent daemon"
  prime-agent shutdown >/dev/null 2>&1 || true
  pkill -9 -f 'prime-agent'  2>/dev/null || true
  pkill -9 -f 'nethack_v1'   2>/dev/null || true
  pkill -9 -f 'provider intercept' 2>/dev/null || true
  rm -rf /tmp/prime-agent-0 2>/dev/null || true
  local s; s=$(date +%s)
  [ -d /root/.prime/agent/daemon-workers ]  && mv /root/.prime/agent/daemon-workers  "/root/.prime/agent/daemon-workers.bak-e9-$s"  2>/dev/null || true
  [ -d /root/.prime/agent/session-leases ]  && mv /root/.prime/agent/session-leases  "/root/.prime/agent/session-leases.bak-e9-$s"  2>/dev/null || true
  sleep 3
}

run_cell() {  # <outdir> <env_args_json>
  local out="$1" env_args="$2"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out"
  reset_daemon
  ENV_ARGS="$env_args" VARIANT=BBOX_MIN "$LAUNCH" prime_agent "$out" 200 5 \
    && echo "[done ] $(date -u +%H:%M:%S) OK  $out" \
    || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $out"
}

run_cell outputs/e9_control_rep/NPCORE_v3_r2__prime_agent "$CORE"
run_cell outputs/e9_control_rep/NPCORE_v3_r3__prime_agent "$CORE"
run_cell outputs/e9_enforce/ENFORCE__prime_agent          "$ENFORCE"
echo "[all  ] $(date -u +%H:%M:%S) E9 launch batch finished"
