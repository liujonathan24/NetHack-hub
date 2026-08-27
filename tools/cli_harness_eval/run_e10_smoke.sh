#!/usr/bin/env bash
set -uo pipefail
REPO=/root/nld/hub-eval
export ENG=/root/NetHack-engine
export EVAL_BIN=/root/NetHack-hub/.venv-cli-eval/bin/eval
cd "$REPO"
prime-agent shutdown >/dev/null 2>&1 || true
pkill -9 -f 'prime-agent'  2>/dev/null || true
pkill -9 -f 'nethack_v1'   2>/dev/null || true
pkill -9 -f 'kernel-venv'  2>/dev/null || true
rm -rf /tmp/prime-agent-0 /tmp/vf-prime-agent 2>/dev/null || true
s=$(date +%s)
[ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers /root/.prime/agent/daemon-workers.bak-$s || true
[ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases /root/.prime/agent/session-leases.bak-$s || true
sleep 3
rm -rf outputs/e10_smoke
ENV_ARGS='{"skill_set":"np_core,request_map,search","auto_dismiss":"false","tune":{"reveal_map":1.0}}' VARIANT=BBOX_MIN \
  "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent outputs/e10_smoke/FIXED__prime_agent 30 1
echo "[smoke done] rc=$?"
