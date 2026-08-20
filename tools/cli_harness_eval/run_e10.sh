#!/usr/bin/env bash
# E10 launch: does a shared continual harness make later games cheaper?
#
# Every cell is the NPCORE_v3 control (np_core reduced NetPlay surface, BBOX_MIN,
# GLM-5.2, Valkyrie, 200 turns / 200 skill calls, 5 seeds) and varies ONE thing:
# whether Prime Agent's global continual-harness store is shared across rollouts
# (`CONTINUAL_HARNESS`) instead of being the per-rollout, always-empty directory
# it is by default.
#
# Round structure, per docs/EXPERIMENT_E10.md:
#
#   round r:  orchestrate (reads round r-1's TRAIN traces, edits the store)
#             -> train cell  (seeds 0-4, the corpus for round r+1)
#             -> eval  cell  (seeds 5-9, HELD OUT, never shown to the orchestrator)
#
# Cells run SEQUENTIALLY with a full prime-agent daemon reset before each (the
# wedge recipe that held across E8/E9); seeds within a cell run concurrently.
# The orchestrator runs BETWEEN cells and never while one is live -- the reset
# is `pkill -9 -f prime-agent`, which would take a resident orchestrator with it.
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
ORCH="$REPO/tools/cli_harness_eval/e10_orchestrate.sh"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs/e10}"
ROUNDS="${ROUNDS:-3}"

# The shared store. MUST be under the harness's install_dir: the bwrap sandbox
# binds that path and little else, so a store outside it is invisible to the
# agent and the harness refuses to launch (see PrimeAgentHarnessConfig.
# continual_harness_dir).
CH="${CH:-/tmp/vf-prime-agent/continual-harness/e10}"

# The held-out half is selected by pinning seeds through ENV_ARGS, which wins
# over the TOML's `explicit_seeds` for ROW SELECTION (nethack_v1.py:817-847) --
# `--num_tasks N` alone only ever takes the FIRST N of the pinned list, so it
# cannot express "seeds 5-9" and an eval cell would silently re-run the training
# seeds instead.
CORE='{"skill_set":"np_core,request_map,search"}'
HELD='{"skill_set":"np_core,request_map,search","explicit_seeds":[5,6,7,8,9]}'
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
  [ -d /root/.prime/agent/daemon-workers ] && mv /root/.prime/agent/daemon-workers "/root/.prime/agent/daemon-workers.bak-e10-$s" 2>/dev/null || true
  [ -d /root/.prime/agent/session-leases ] && mv /root/.prime/agent/session-leases "/root/.prime/agent/session-leases.bak-e10-$s" 2>/dev/null || true
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

run_cell() {  # <outdir> <env_args> <shared: 0|1>
  local out="$1" env_args="$2" shared="$3"
  local n="$CELL_N"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out (shared_harness=$shared n=$n)"
  reset_daemon
  mkdir -p "$out"
  snapshot before "$out"
  if [ "$shared" = "1" ]; then
    CONTINUAL_HARNESS="$CH" ENV_ARGS="$env_args" VARIANT=BBOX_MIN "$LAUNCH" prime_agent "$out" 200 "$n" \
      && echo "[done ] $(date -u +%H:%M:%S) OK  $out" \
      || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $out"
  else
    ENV_ARGS="$env_args" VARIANT=BBOX_MIN "$LAUNCH" prime_agent "$out" 200 "$n" \
      && echo "[done ] $(date -u +%H:%M:%S) OK  $out" \
      || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $out"
  fi
  snapshot after "$out"
  if ! cmp -s "$out/harness_store.before.sha256" "$out/harness_store.after.sha256"; then
    echo "[WARN ] $(date -u +%H:%M:%S) the shared store CHANGED during $out --" \
         "expected only when CONTINUAL_HARNESS_WRITABLE=1. Diff the snapshots." >&2
  fi
}

mkdir -p "$CH"

# --- round 0: the baselines, with the store deliberately NOT shared ----------
# Both halves need a no-harness baseline before anything is written: the eval
# seeds have never been measured, and the train cell is the corpus round 1 reads.
if [ "${SKIP_ROUND0:-}" != "1" ]; then
  run_cell "$OUT_ROOT/round0/train__prime_agent" "$CORE" 0
  run_cell "$OUT_ROOT/round0/eval__prime_agent"  "$HELD" 0
fi

# --- rounds 1..N -------------------------------------------------------------
for r in $(seq 1 "$ROUNDS"); do
  prev=$((r - 1))
  echo "[round] $(date -u +%H:%M:%S) === round $r ==="
  reset_daemon   # the orchestrator is a prime-agent process too; start it clean
  "$ORCH" "$OUT_ROOT/round$prev/train__prime_agent" "$CH" "$OUT_ROOT/round$r" \
    || { echo "[FAIL ] orchestrator round $r rc=$?" >&2; }
  run_cell "$OUT_ROOT/round$r/train__prime_agent" "$CORE" 1
  run_cell "$OUT_ROOT/round$r/eval__prime_agent"  "$HELD" 1
done

echo "[all  ] $(date -u +%H:%M:%S) E10 batch finished; store at $CH"
