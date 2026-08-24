#!/usr/bin/env bash
# E13: does past play make future play cheaper?
#
#   run_e13.sh <RUN_ID>
#
# A RUN_ID names one continual-harness EXPERIMENT. Several run side by side --
# same base surface, different reflection instructions -- so everything is
# namespaced by it and nothing is shared between them:
#
#   store    /tmp/vf-prime-agent/continual-harness/<RUN_ID>
#   prompt   configs/continual/<RUN_ID>.md   (falls back to default.md)
#   outputs  outputs/e13/<RUN_ID>/round<r>/...
#
# EVERY RUN STARTS FROM THE BASE SET OF SKILLS. Round 0 refuses to begin against
# a non-empty store, and the [continual] tier carries base's flags (all fix-
# flags off), so a gain cannot come from the human tier or from another
# experiment's leftovers. RESUME=1 continues an existing run instead.
#
#   round 0:  corpus cell  (seeds 5-9, TOOL_TIER=base)  -- the games to reflect on
#             control cell (seeds 0-4, TOOL_TIER=base)  -- the concurrent baseline
#   round r:  orchestrate (reads the seeds 5-9 corpus, edits this run's store)
#             -> eval   cell (seeds 0-4, TOOL_TIER=continual + store)
#             -> corpus cell (seeds 5-9, same) -- next round's corpus
#
# Reflection and evaluation use DISJOINT seeds: the orchestrator only ever sees
# 5-9; the headline is on 0-4, which also puts E13 beside E10's baseline and
# E11's handcrafted gates.
set -uo pipefail

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ]; then
  echo "usage: run_e13.sh <RUN_ID>   (e.g. reflect-default, reflect-terse)" >&2
  echo "  a RUN_ID names one continual experiment; it is pinned into every" >&2
  echo "  cell's config.toml so two experiments cannot be confused on disk." >&2
  exit 2
fi
case "$RUN_ID" in
  *[!a-zA-Z0-9._-]*)
    echo "run_e13: RUN_ID '$RUN_ID' must be [A-Za-z0-9._-] -- it becomes a" >&2
    echo "  directory name and a config value." >&2
    exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The harness package is an editable install pointing at the MAIN checkout, so a
# worktree's harness changes are invisible without this (measured: the eval CLI
# rejects --harness.continual_harness_dir as "Extra inputs are not permitted",
# which reads like a bad flag rather than a stale package).
export PYTHONPATH="${REPO}/harnesses/nethack-prime-agent${PYTHONPATH:+:${PYTHONPATH}}"
export ENG="${ENG:-/root/NetHack-engine}"
export EVAL_BIN="${EVAL_BIN:-/root/NetHack-hub/.venv-cli-eval/bin/eval}"
PY_BIN="$(dirname "${EVAL_BIN}")/python"
LAUNCH="$REPO/tools/cli_harness_eval/launch_cell.sh"
ORCH="$REPO/tools/cli_harness_eval/e13_orchestrate.sh"

CH="${CH:-/tmp/vf-prime-agent/continual-harness/${RUN_ID}}"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs/e13/${RUN_ID}}"
PROMPT_FILE="${PROMPT_FILE:-$REPO/configs/continual/${RUN_ID}.md}"
[ -f "$PROMPT_FILE" ] || PROMPT_FILE="$REPO/configs/continual/default.md"
[ -f "$PROMPT_FILE" ] || { echo "run_e13: no reflection prompt at $PROMPT_FILE" >&2; exit 2; }
PROMPT_SHA="$(sha256sum "$PROMPT_FILE" | cut -c1-16)"

ROUNDS="${ROUNDS:-1}"      # 1 = the pilot (20 games). Each extra round is +10.
CELL_N=5
HELD_SEEDS='[5,6,7,8,9]'
cd "$REPO"

echo "[e13  ] run_id=$RUN_ID store=$CH"
echo "[e13  ] prompt=$PROMPT_FILE sha=$PROMPT_SHA rounds=$ROUNDS"

# --- the base-set guarantee --------------------------------------------------
mkdir -p "$CH"
if [ "${RESUME:-}" != "1" ] && [ -n "$(ls -A "$CH" 2>/dev/null)" ]; then
  echo "run_e13: store $CH is not empty, so this run would NOT start from the" >&2
  echo "  base set of skills -- it would inherit whatever is already there." >&2
  echo "  Use a fresh RUN_ID, RESUME=1 to continue this one, or delete it." >&2
  exit 3
fi

reset_daemon() {
  echo "[reset] $(date -u +%H:%M:%S) tearing down prime-agent daemon"
  prime-agent shutdown >/dev/null 2>&1 || true
  pkill -9 -f 'prime-agent'  2>/dev/null || true
  pkill -9 -f 'nethack_v1'   2>/dev/null || true
  rm -rf /tmp/prime-agent-0 2>/dev/null || true
  local s; s=$(date +%s)
  for d in daemon-workers session-leases; do
    [ -d "/root/.prime/agent/$d" ] && mv "/root/.prime/agent/$d" "/root/.prime/agent/$d.bak-e13-$s" 2>/dev/null || true
  done
  sleep 3
}

# The store is the independent variable, so every cell records the exact bytes
# it ran against -- before AND after. A differing pair means a player wrote to a
# store we believe is read-only, and the cell is not reproducible from either.
snapshot() {  # <label> <destdir>
  local label="$1" dest="$2"
  mkdir -p "$dest"
  if [ -e "$CH/harness_state.json" ]; then
    cp "$CH/harness_state.json" "$dest/harness_state.$label.json"
  else
    echo '{"note":"empty store"}' > "$dest/harness_state.$label.json"
  fi
  ( cd "$CH" 2>/dev/null && find . -type f -exec sha256sum {} \; | sort ) \
    > "$dest/harness_store.$label.sha256" 2>/dev/null || true
}

run_cell() {  # <outdir> <tier> <seeds-json|""> <mount-store: 0|1>
  local out="$1" tier="$2" seeds="$3" mount="$4"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out (tier=$tier mount=$mount)"
  reset_daemon
  mkdir -p "$out"
  snapshot before "$out"
  local -a env_pairs=(TOOL_TIER="$tier")
  [ -n "$seeds" ] && env_pairs+=(SEEDS="$seeds")
  if [ "$mount" = "1" ]; then
    env_pairs+=(CONTINUAL_HARNESS="$CH" CONTINUAL_RUN_ID="$RUN_ID" CONTINUAL_PROMPT_SHA="$PROMPT_SHA")
  fi
  env "${env_pairs[@]}" "$LAUNCH" prime_agent "$out" 200 "$CELL_N" \
    && echo "[done ] $(date -u +%H:%M:%S) OK  $out" \
    || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $out"
  snapshot after "$out"
  if ! cmp -s "$out/harness_store.before.sha256" "$out/harness_store.after.sha256"; then
    echo "[WARN ] $(date -u +%H:%M:%S) the store CHANGED during $out -- expected" \
         "only with CONTINUAL_HARNESS_WRITABLE=1." >&2
  fi
}

# A manifest per run, so two experiments are told apart from their outputs alone.
mkdir -p "$OUT_ROOT"
cat > "$OUT_ROOT/run_manifest.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "store": "${CH}",
  "reflection_prompt": "${PROMPT_FILE}",
  "reflection_prompt_sha256_16": "${PROMPT_SHA}",
  "rounds": ${ROUNDS},
  "eval_seeds": [0, 1, 2, 3, 4],
  "corpus_seeds": ${HELD_SEEDS},
  "corpus_tier": "base",
  "eval_tier": "continual",
  "started_from_empty_store": $( [ "${RESUME:-}" = "1" ] && echo false || echo true )
}
JSON
echo "[e13  ] manifest -> $OUT_ROOT/run_manifest.json"

# --- round 0: the corpus to reflect on, and the concurrent control -----------
if [ "${SKIP_ROUND0:-}" != "1" ]; then
  run_cell "$OUT_ROOT/round0/corpus__prime_agent"  base "$HELD_SEEDS" 0
  for rep in $(seq 1 "${CONTROL_REPS:-1}"); do
    run_cell "$OUT_ROOT/round0/control_r${rep}__prime_agent" base "" 0
  done
fi

# --- rounds 1..N -------------------------------------------------------------
for r in $(seq 1 "$ROUNDS"); do
  prev=$((r - 1))
  echo "[round] $(date -u +%H:%M:%S) === ${RUN_ID} round $r ==="
  reset_daemon   # the orchestrator is a prime-agent process too; start it clean
  "$ORCH" "$OUT_ROOT/round$prev/corpus__prime_agent" "$CH" "$OUT_ROOT/round$r" "$PROMPT_FILE" \
    || echo "[FAIL ] orchestrator round $r rc=$?" >&2
  for rep in $(seq 1 "${EVAL_REPS:-1}"); do
    run_cell "$OUT_ROOT/round$r/eval_r${rep}__prime_agent" continual "" 1
  done
  run_cell "$OUT_ROOT/round$r/corpus__prime_agent" continual "$HELD_SEEDS" 1
done

echo "[e13  ] $(date -u +%H:%M:%S) ${RUN_ID} finished; store at $CH"
