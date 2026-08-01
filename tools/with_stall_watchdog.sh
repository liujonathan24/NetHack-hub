#!/usr/bin/env bash
# Run any command with a stall watchdog armed for exactly its lifetime.
#
#   tools/with_stall_watchdog.sh <TURNS_DIR> [-- ] <command> [args...]
#
# e.g.
#   tools/with_stall_watchdog.sh outputs/run1/control/turns -- \
#       tools/cli_harness_eval/launch_cell.sh control outputs/run1/control 400 5
#
# WHY A WRAPPER AND NOT A LONG-LIVED DAEMON
# -----------------------------------------
# RUNBOOK.md: "Re-arm it per batch -- it exits when the queue drains." The
# documented failure mode is a watchdog started once for a whole sweep: it exits
# (or is forgotten) when the first batch's queue drains, and every later batch
# runs unguarded. Binding the watchdog's lifetime to ONE command makes arming
# and the batch the same event -- there is no state in which a batch is running
# and the watchdog is not.
#
# The command runs in the FOREGROUND and its exit code is this script's exit
# code, so this is a drop-in prefix for an existing launcher invocation.
#
# Env knobs (all optional):
#   STALL_TIMEOUT      seconds of silence before a kill (default 300)
#   STALL_POLL         seconds between scans (default 15)
#   STALL_EXTRA_ARGS   extra args passed through to stall_watchdog.py
#                      (e.g. --dry-run, --verbose, --quarantine-dir ...)
#   PY_BIN             interpreter (default: repo .venv-cli-eval/bin/python,
#                      else python3)
set -uo pipefail

usage() {
  echo "usage: with_stall_watchdog.sh <TURNS_DIR> [--] <command> [args...]" >&2
}

[ "$#" -ge 2 ] || { usage; exit 2; }

TURNS_DIR="$1"; shift
[ "${1:-}" = "--" ] && shift
[ "$#" -ge 1 ] || { usage; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Created up front: the watchdog needs the directory to exist to watch it, and
# the launchers create it anyway a moment later.
mkdir -p "${TURNS_DIR}"
TURNS_ABS="$(cd "${TURNS_DIR}" && pwd)"

PY_BIN="${PY_BIN:-${REPO}/.venv-cli-eval/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN=python3

# --parent-pid $$ is belt-and-braces: the trap below already stops the watchdog,
# but if this shell is killed with SIGKILL the trap never runs, and the watchdog
# would otherwise linger and guard a directory nobody is writing to.
# shellcheck disable=SC2086
"$PY_BIN" "${REPO}/tools/stall_watchdog.py" \
  --turns-dir "${TURNS_ABS}" \
  --timeout "${STALL_TIMEOUT:-300}" \
  --poll "${STALL_POLL:-15}" \
  --parent-pid "$$" \
  ${STALL_EXTRA_ARGS:-} &
WD_PID=$!
trap 'kill "${WD_PID}" 2>/dev/null || true' EXIT INT TERM

echo "[with_stall_watchdog] armed pid=${WD_PID} turns=${TURNS_ABS} timeout=${STALL_TIMEOUT:-300}s" >&2
echo "[with_stall_watchdog] run: $*" >&2

"$@"
RC=$?

kill "${WD_PID}" 2>/dev/null || true
wait "${WD_PID}" 2>/dev/null || true
echo "[with_stall_watchdog] command exited rc=${RC}; watchdog disarmed" >&2
exit "${RC}"
