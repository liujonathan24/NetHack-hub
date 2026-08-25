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

# Content hash of a directory tree, for the before/after contamination check.
# Sorted so it is order-independent, and names are included so a deletion or a
# rename is caught as well as an edit.
_tree_hash() {
  find "$1" -type f -not -path '*/.git/*' -not -name '*.pyc' -print0 \
    | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -c1-16
}

RUN="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["run"])' "$FINAL/FROZEN.json")"
# The tier the experiment actually ran. Recorded in FROZEN.json since the code
# arms landed; older frozen runs predate the field, and `continual` is the right
# fallback for every one of them because no other tier existed.
TIER="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("tier") or "continual")' "$FINAL/FROZEN.json")"
# A code arm is evaluated on the HELD-OUT tier, not on its own: that tier pins
# the tree (`netplay_code_mode = "pinned"`) so the harness materialises nothing
# and the agent's own final code is what runs. Evaluating on `continual-code`
# would let the cell rewrite the artifact under test; on `continual-code-frozen`
# it would overwrite the agent's composites with our repo seeds.
case "$TIER" in
  continual-code*) EVAL_TIER="continual-code-heldout" ;;
  *)               EVAL_TIER="$TIER" ;;
esac
# Recorded beside the store so a held-out number names the exact tree it scored,
# the way tool_tier_commit already names the exact tool surface.
NETPLAY_COMMIT="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("netplay_commit") or "")' "$FINAL/FROZEN.json")"
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

# The frozen CODE tree, same treatment as the store: copied under install_dir so
# it is visible in the sandbox, with outputs/ left pristine. It goes to the one
# path the kernel's sys.path carries -- `{install_dir}/skills/nethack/src` is a
# bare .pth entry, so the tree is importable there and nowhere else.
NETPLAY_DIR="${INSTALL_DIR}/skills/nethack/src/netplay"
NETPLAY_BEFORE=""
if [ -d "$FINAL/netplay" ]; then
  mkdir -p "$(dirname "$NETPLAY_DIR")"
  rm -rf "$NETPLAY_DIR"
  cp -a "$FINAL/netplay" "$NETPLAY_DIR"
  rm -rf "$NETPLAY_DIR/.git"          # the artifact, not its history
  NETPLAY_BEFORE="$(_tree_hash "$NETPLAY_DIR")"
  echo "[eval ] netplay tree pinned at $NETPLAY_BEFORE (commit ${NETPLAY_COMMIT:-unknown})"
elif [ "$EVAL_TIER" = "continual-code-heldout" ]; then
  echo "eval_frozen: $FINAL records tier=$TIER but has no netplay/ tree to pin." >&2
  echo "  A code arm evaluated without its code measures the wrong thing." >&2
  exit 3
fi

echo "[eval ] $RUN tier=$EVAL_TIER seeds=$SEEDS store=$BEFORE (read-only)"

prime-agent shutdown >/dev/null 2>&1 || true
pkill -9 -f 'prime-agent' 2>/dev/null || true
pkill -9 -f 'nethack_v1'  2>/dev/null || true
rm -rf /tmp/prime-agent-0 2>/dev/null || true
sleep 3

mkdir -p "$OUT"; cp "$FINAL/FROZEN.json" "$OUT/evaluated_snapshot.json"
env TOOL_TIER="$EVAL_TIER" SEEDS="$SEEDS" INSTALL_DIR="$INSTALL_DIR" \
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
# The same guard over the code tree. `pinned` mode plus this check are belt and
# braces on purpose: the mode is what SHOULD make writes impossible, and this is
# what tells us if it did not.
if [ -n "$NETPLAY_BEFORE" ]; then
  NETPLAY_AFTER="$(_tree_hash "$NETPLAY_DIR")"
  if [ "$NETPLAY_BEFORE" != "$NETPLAY_AFTER" ]; then
    echo "[eval ] FAILED: the frozen netplay tree CHANGED during evaluation" >&2
    echo "  ($NETPLAY_BEFORE -> $NETPLAY_AFTER). The evaluation rewrote the code" >&2
    echo "  it was measuring, so this result describes no fixed artifact." >&2
    exit 5
  fi
  echo "[eval ] netplay tree unchanged ($NETPLAY_AFTER)"
fi
echo "[eval ] $(date -u +%H:%M:%S) rc=$rc; store unchanged; results in $OUT"
