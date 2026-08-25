#!/usr/bin/env bash
# Held-out evaluation of a frozen experiment.
#
#   eval_frozen.sh <outputs/e13/<run>/final> [OUTDIR]
#
# Run by US, after an experiment ends -- not by the experiment. It mounts that
# experiment's FROZEN store on the held-out seeds and asks the only question the
# corpus side cannot: do the accumulated lessons generalise to dungeons the agent
# never played and never reflected on?
#
# The store is mounted READ-ONLY (shared-ro, writable off). An evaluation that
# could write would contaminate the very artifact it is testing, and the next
# evaluation of the same snapshot would not be measuring the same thing.
set -uo pipefail

FINAL="${1:?usage: eval_frozen.sh <final-dir> [outdir]}"
[ -f "$FINAL/FROZEN.json" ] || { echo "eval_frozen: $FINAL is not a frozen run" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO}/harnesses/nethack-prime-agent${PYTHONPATH:+:${PYTHONPATH}}"
export ENG="${ENG:-/root/NetHack-engine}"
export EVAL_BIN="${EVAL_BIN:-/root/NetHack-hub/.venv-cli-eval/bin/eval}"
PY_BIN="${PY_BIN:-$(dirname "${EVAL_BIN}")/python}"

RUN="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["run"])' "$FINAL/FROZEN.json")"
SEEDS="$("$PY_BIN" -c 'import json,sys,tomllib;print(json.dumps(tomllib.load(open(sys.argv[1],"rb"))["eval_seeds"]))' "$FINAL/experiment.toml")"
N="$(printf '%s' "$SEEDS" | "$PY_BIN" -c 'import json,sys;print(len(json.load(sys.stdin)))')"
OUT="${2:-$REPO/outputs/e13/heldout/${RUN}}"

# The frozen store has to live under install_dir to be visible inside the
# sandbox, so it is COPIED there rather than mounted from outputs/. The copy is
# what the cell reads; the snapshot in outputs/ stays pristine.
INSTALL_DIR="${INSTALL_DIR:-/tmp/vf-prime-agent-heldout-${RUN}}"
CH="${INSTALL_DIR}/frozen-store"
mkdir -p "$CH"; cp "$FINAL/harness_state.json" "$CH/harness_state.json"
BEFORE="$(sha256sum "$CH/harness_state.json" | cut -c1-16)"
echo "[eval ] $RUN seeds=$SEEDS store=$BEFORE (read-only)"

# SCOPED teardown. This used to be a global `pkill -9 -f prime-agent`, which
# kills every experiment on the box, not just this one -- and a held-out
# evaluation is precisely the thing you run at the END, while other cells are
# still going. reset_daemon.sh matches PRIME_AGENT_CODING_AGENT_DIR under this
# evaluation's own install_dir and takes a flock around the reset+boot window,
# so a concurrent run keeps its players.
"$REPO/tools/cli_harness_eval/reset_daemon.sh" "$INSTALL_DIR"

mkdir -p "$OUT"; cp "$FINAL/FROZEN.json" "$OUT/evaluated_snapshot.json"
env TOOL_TIER=continual SEEDS="$SEEDS" INSTALL_DIR="$INSTALL_DIR" \
    CONTINUAL_HARNESS="$CH" CONTINUAL_RUN_ID="heldout-${RUN}" \
    CONTINUAL_HARNESS_MODE=shared-ro \
    "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$OUT" 200 "$N"
rc=$?

AFTER="$(sha256sum "$CH/harness_state.json" | cut -c1-16)"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "[eval ] FAILED: the frozen store CHANGED during evaluation ($BEFORE -> $AFTER)." >&2
  echo "  A held-out evaluation that writes has contaminated what it tested." >&2
  exit 4
fi
echo "[eval ] $(date -u +%H:%M:%S) rc=$rc; store unchanged; results in $OUT"
