#!/usr/bin/env bash
# Tear down prime-agent state for ONE experiment, so two batches can overlap.
#
#   reset_daemon.sh <INSTALL_DIR>
#
# The stale-daemon wedge is a BOOT-time problem: a leftover daemon.sock hangs
# every rollout at startup. The standard recipe fixes it with
# `pkill -9 -f prime-agent`, which is global -- it would kill a concurrently
# running experiment's players too. That is the only thing stopping us from
# running a re-baseline alongside an experiment, because the interception layer
# already parallelises on its own (each `eval` process starts its own
# interception server, multiplex=32).
#
# Two changes make it safe:
#   * kill by ENVIRONMENT, not cmdline. The worker processes are named bare
#     `prime-agent`, so cmdline matching cannot distinguish experiments -- but
#     each carries PRIME_AGENT_CODING_AGENT_DIR, which lives under its own
#     install_dir. /proc/<pid>/environ is the only place that distinction exists.
#   * serialise the reset+boot window with flock. Concurrent batches may RUN
#     concurrently; they must not both be resetting and booting daemons at the
#     same moment, which is when the shared ~/.prime/agent state races.
set -uo pipefail

INSTALL_DIR="${1:?usage: reset_daemon.sh <INSTALL_DIR>}"
LOCK="/tmp/prime-agent-reset.lock"

exec 9>"$LOCK"
flock 9 || { echo "reset_daemon: could not take $LOCK" >&2; exit 1; }

echo "[reset] $(date -u +%H:%M:%S) scoped teardown for $INSTALL_DIR"

killed=0
for pid in $(pgrep -x prime-agent 2>/dev/null) $(pgrep -f 'nethack_v1' 2>/dev/null); do
  envfile="/proc/$pid/environ"
  [ -r "$envfile" ] || continue
  if tr '\0' '\n' < "$envfile" 2>/dev/null | grep -q "^PRIME_AGENT_CODING_AGENT_DIR=${INSTALL_DIR}/"; then
    kill -9 "$pid" 2>/dev/null && killed=$((killed + 1))
  fi
done
echo "[reset] killed $killed process(es) belonging to this experiment"

# Per-experiment daemon socket dir, if the sandbox is off and TMPDIR is shared.
rm -rf "${INSTALL_DIR}/prime-agent-0" 2>/dev/null || true

# The shared supervisor state. Moving it aside is what actually clears a wedge;
# it is shared, so it happens under the lock and only when no OTHER experiment
# has live workers.
if ! pgrep -x prime-agent >/dev/null 2>&1; then
  s=$(date +%s)
  for d in daemon-workers session-leases; do
    [ -d "/root/.prime/agent/$d" ] && mv "/root/.prime/agent/$d" "/root/.prime/agent/$d.bak-$s" 2>/dev/null || true
  done
  rm -rf /tmp/prime-agent-0 2>/dev/null || true
  echo "[reset] no other experiment is live -- cleared shared supervisor state too"
else
  echo "[reset] another experiment has live workers -- left shared state alone"
fi

sleep 3
exec 9>&-
