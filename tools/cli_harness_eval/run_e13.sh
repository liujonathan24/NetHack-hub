#!/usr/bin/env bash
# E13 launch: does a shared continual harness make later games cheaper?
#
# Every cell is the NPCORE_v3 control (np_core reduced NetPlay surface, BBOX_MIN,
# GLM-5.2, Valkyrie, 200 turns / 200 skill calls, 5 seeds) and varies ONE thing:
# whether Prime Agent's global continual-harness store is shared across rollouts
# (`CONTINUAL_HARNESS`) instead of being the per-rollout, always-empty directory
# it is by default.
#
# Round structure:
#
#   round 0:  corpus cell  (seeds 5-9, store OFF) -- the games to reflect on
#             control cell (seeds 0-4, store OFF) -- the concurrent baseline
#   round r:  orchestrate (reads the seeds 5-9 corpus, edits the store)
#             -> eval   cell (seeds 0-4, store ON)  <- the headline comparison
#             -> corpus cell (seeds 5-9, store ON)  <- next round's corpus
#
# Reflection and evaluation use DISJOINT seeds: the orchestrator only ever sees
# seeds 5-9, and the headline number is on seeds 0-4, which it has never read a
# trace from. Evaluating on 0-4 also puts E13 alongside E10's honest re-baseline
# and E11's handcrafted descent gates, which ran those same seeds.
#
# WHY A CONCURRENT CONTROL, when E10 already published a 3-rep baseline on seeds
# 0-4: the harness moved after E10/E11 ran. E10's cells finished 14:39 and
# E11's 18:11 on 2026-08-21; c1a0bec (NetPlay telemetry/affordances), aee5c43
# (SKILL.md coordinate frame -- a silent off-by-one) and ea8cc15 (melee hints)
# landed at 22:06, 22:59 and 23:03 the same day. The previous harness pass
# doubled measured performance, so comparing a cell run today against E10's
# published numbers would confound "reflection helped" with "three more fixes
# landed". E10/E11 stay as historical context; the comparison that carries the
# claim is control-vs-eval INSIDE one batch, on one harness commit.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# THE HARNESS PACKAGE DOES NOT COME FROM THIS WORKTREE unless we put it here.
# Unlike the env code (which launch_cell.sh puts on PYTHONPATH from $REPO),
# `nethack_prime_agent` is an EDITABLE INSTALL in the shared venv pointing at
# /root/NetHack-hub/harnesses/nethack-prime-agent -- the main checkout. Without
# this line the eval CLI validates against the OLD config class and dies with
#   "--continual-harness-dir  Extra inputs are not permitted"
# which reads like a typo in the flag rather than a stale package. Measured.
# PYTHONPATH beats the .pth-added site-packages entry, so this wins.
export PYTHONPATH="${REPO}/harnesses/nethack-prime-agent${PYTHONPATH:+:${PYTHONPATH}}"
export ENG="${ENG:-/root/NetHack-engine}"
export EVAL_BIN="${EVAL_BIN:-/root/NetHack-hub/.venv-cli-eval/bin/eval}"
LAUNCH="$REPO/tools/cli_harness_eval/launch_cell.sh"
ORCH="$REPO/tools/cli_harness_eval/e13_orchestrate.sh"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs/e13}"
# Default 1, not 3: one round is the pilot (20 games). Raise it to extend the
# learning curve -- each extra round is another 10 games.
ROUNDS="${ROUNDS:-1}"

# The shared store. MUST be under the harness's install_dir: the bwrap sandbox
# binds that path and little else, so a store outside it is invisible to the
# agent and the harness refuses to launch (see PrimeAgentHarnessConfig.
# continual_harness_dir).
CH="${CH:-/tmp/vf-prime-agent/continual-harness/e13}"

# The held-out half is selected by pinning seeds through ENV_ARGS, which wins
# over the TOML's `explicit_seeds` for ROW SELECTION (nethack_v1.py:817-847) --
# `--num_tasks N` alone only ever takes the FIRST N of the pinned list, so it
# cannot express "seeds 5-9" and an eval cell would silently re-run the training
# seeds instead.
# TIERS, NOT STRINGS. What the agent may reach for is declared in
# configs/tool_tiers.toml and resolved by tiers.py, which also folds the
# provenance pin (tier-file hash + version + code commit) into env_args -- so it
# lands in each cell's own resolved config.toml and the run replays from that
# file alone. Hand-editing an ENV_ARGS string here is exactly the drift the tier
# file exists to stop.
#
# Arms are never pooled: the eval cells run one stack, the control cells run
# another, and each is its own output directory.
PY_BIN="$(dirname "${EVAL_BIN}")/python"
TIERS="$PY_BIN $REPO/tools/cli_harness_eval/tiers.py"

EVAL_STACK="${EVAL_STACK:-base+human+continual}"   # the continual arm
CTL_STACK="${CTL_STACK:-base+human}"               # what it must beat
CORPUS_STACK="${CORPUS_STACK:-base+human}"         # games to reflect on

# Refuse a stack this tree cannot honestly produce (see [human.gap]: the PR #29
# fixes are still unconditional code, so a true `base` cell needs the pinned
# checkout, not a config flag). TIERS_ALLOW_BLOCKED=1 overrides, loudly.
for st in "$EVAL_STACK" "$CTL_STACK" "$CORPUS_STACK"; do
  if ! $TIERS check "$st" 2>/tmp/tiercheck.$$; then
    cat /tmp/tiercheck.$$ >&2
    if [ "${TIERS_ALLOW_BLOCKED:-}" != "1" ]; then
      echo "run_e13: refusing to launch a stack the tier file says this tree" >&2
      echo "  cannot produce. Set TIERS_ALLOW_BLOCKED=1 to override anyway." >&2
      rm -f /tmp/tiercheck.$$; exit 5
    fi
    echo "run_e13: TIERS_ALLOW_BLOCKED=1 -- launching a stack with known gaps." >&2
  fi
  rm -f /tmp/tiercheck.$$
done

# The held-out corpus half is selected by pinning seeds through ENV_ARGS, which
# wins over the TOML's `explicit_seeds` for ROW SELECTION (nethack_v1.py:817-847).
TIER_HASH="$($PY_BIN -c 'import sys;sys.path.insert(0,"'"$REPO"'/tools/cli_harness_eval");import tiers;print(tiers.tier_hash())')"
CORE="$($TIERS env-args "$CTL_STACK")"
EVAL_ARGS="$($TIERS env-args "$EVAL_STACK")"
HELD="$($PY_BIN -c '
import json,sys
d=json.loads(sys.argv[1]); d["explicit_seeds"]=[5,6,7,8,9]; print(json.dumps(d))
' "$($TIERS env-args "$CORPUS_STACK")")"
CELL_N=5

cd "$REPO"

reset_daemon() {
  echo "[reset] $(date -u +%H:%M:%S) tearing down prime-agent daemon"
  prime-agent shutdown >/dev/null 2>&1 || true
  pkill -9 -f 'prime-agent'  2>/dev/null || true
  pkill -9 -f 'nethack_v1'   2>/dev/null || true
  pkill -9 -f 'provider intercept' 2>/dev/null || true
  rm -rf /tmp/prime-agent-0 2>/dev/null || true
  local s; s=$(date +%s)
  [ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers "/root/.prime/agent/daemon-workers.bak-e13-$s" 2>/dev/null || true
  [ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases "/root/.prime/agent/session-leases.bak-e13-$s" 2>/dev/null || true
  sleep 3
}

# The store is the independent variable, so every cell records the exact bytes it
# ran against -- before AND after. `after` is what catches a player that wrote to
# a store we believe is read-only, and a cell whose two snapshots differ is not
# reproducible from either one.
snapshot() {  # <label> <destdir>
  local label="$1" dest="$2"
  mkdir -p "$dest"
  if [ -e "$CH/harness_state.json" ]; then
    cp "$CH/harness_state.json" "$dest/harness_state.$label.json"
  else
    echo '{"note":"no harness_state.json yet"}' > "$dest/harness_state.$label.json"
  fi
  [ -e "$CH/refinements.jsonl" ] && cp "$CH/refinements.jsonl" "$dest/refinements.$label.jsonl"
  ( cd "$CH" 2>/dev/null && find . -type f -exec sha256sum {} \; | sort ) \
    > "$dest/harness_store.$label.sha256" 2>/dev/null || true
}

run_cell() {  # <outdir> <env_args> <stack>
  # The stack decides whether the shared store is mounted -- `tiers.py flags`
  # emits CONTINUAL_HARNESS only for a stack that includes [continual]. Passing
  # a 0/1 by hand is how an arm silently becomes a different arm.
  local out="$1" env_args="$2" stack="$3"
  local n="$CELL_N"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out (stack=$stack n=$n)"
  reset_daemon
  mkdir -p "$out"
  snapshot before "$out"
  # shellcheck disable=SC2046
  env $($TIERS flags "$stack") ENV_ARGS="$env_args" VARIANT=BBOX_MIN \
    "$LAUNCH" prime_agent "$out" 200 "$n" \
    && echo "[done ] $(date -u +%H:%M:%S) OK  $out" \
    || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $out"
  snapshot after "$out"
  # Read back what the cell ACTUALLY resolved to. A cell whose config does not
  # match the stack it declared is not evidence, whatever it scored.
  "$PY_BIN" "$REPO/tools/cli_harness_eval/preflight_cell.py" "$out" \
    --stack "$stack" --tier-hash "$TIER_HASH" \
    || echo "[WARN ] $(date -u +%H:%M:%S) $out did not read back clean -- see above" >&2
  if ! cmp -s "$out/harness_store.before.sha256" "$out/harness_store.after.sha256"; then
    echo "[WARN ] $(date -u +%H:%M:%S) the shared store CHANGED during $out --" \
         "expected only when CONTINUAL_HARNESS_WRITABLE=1. Diff the snapshots." >&2
  fi
}

# MANDATORY PROTOCOL. No paid cell launches until a mock play on the same stack
# has read back clean. SKIP_PREFLIGHT=1 exists for a rerun in the same session
# on an unchanged tree -- it is not for "I'm fairly sure it's fine".
if [ "${SKIP_PREFLIGHT:-}" != "1" ]; then
  for st in $(printf '%s\n' "$CTL_STACK" "$EVAL_STACK" "$CORPUS_STACK" | sort -u); do
    echo "[pre  ] $(date -u +%H:%M:%S) preflight $st"
    "$REPO/tools/cli_harness_eval/preflight_cell.sh" "$st" \
      || { echo "run_e13: preflight failed for $st -- batch not launched." >&2; exit 7; }
  done
fi

mkdir -p "$CH"

# --- round 0: the corpus to reflect on, and the concurrent control -----------
# Seeds 5-9 have never been played by anything, so the corpus must be run. The
# 0-4 control is what the eval cells are actually compared against (see the
# harness-drift note above).
if [ "${SKIP_ROUND0:-}" != "1" ]; then
  run_cell "$OUT_ROOT/round0/corpus__prime_agent"  "$HELD" "$CORPUS_STACK"
  for rep in $(seq 1 "${CONTROL_REPS:-1}"); do
    run_cell "$OUT_ROOT/round0/control_r${rep}__prime_agent" "$CORE" "$CTL_STACK"
  done
fi

# --- rounds 1..N -------------------------------------------------------------
for r in $(seq 1 "$ROUNDS"); do
  prev=$((r - 1))
  echo "[round] $(date -u +%H:%M:%S) === round $r ==="
  reset_daemon   # the orchestrator is a prime-agent process too; start it clean
  # The corpus is ALWAYS a seeds 5-9 cell. Nothing from seeds 0-4 is ever passed
  # to the orchestrator -- that is the whole basis of the headline number.
  CORPUS="$OUT_ROOT/round$prev/corpus__prime_agent"
  echo "[round] corpus: $CORPUS"
  "$ORCH" "$CORPUS" "$CH" "$OUT_ROOT/round$r" \
    || { echo "[FAIL ] orchestrator round $r rc=$?" >&2; }
  for rep in $(seq 1 "${EVAL_REPS:-1}"); do
    run_cell "$OUT_ROOT/round$r/eval_r${rep}__prime_agent" "$EVAL_ARGS" "$EVAL_STACK"
  done
  # Next round's corpus, played WITH the store so the loop compounds.
  run_cell "$OUT_ROOT/round$r/corpus__prime_agent" "$HELD" "$EVAL_STACK"
done

echo "[all  ] $(date -u +%H:%M:%S) E13 batch finished; store at $CH"
