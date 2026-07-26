#!/usr/bin/env bash
# Run the full CLI-harness comparison: every arm x the first N pinned seeds,
# then aggregate into the cross-arm results table.
#
# Usage:
#   tools/cli_harness_eval/run_sweep.sh <RUN_NAME> [MAX_CALLS] [N] [ARM ...]
#
#   tools/cli_harness_eval/run_sweep.sh run1                    # 400 calls, 5 seeds, all 3 arms
#   tools/cli_harness_eval/run_sweep.sh run1 400 5              # explicit
#   tools/cli_harness_eval/run_sweep.sh smoke 20 1              # cheap smoke, seed 0
#   tools/cli_harness_eval/run_sweep.sh run1 400 5 control      # one arm only
#   DRY_RUN=1 tools/cli_harness_eval/run_sweep.sh run1          # print the plan, spend nothing
#
# Each arm goes through launch_cell.sh so the fixed factors (model, seeds,
# character, task_spec, skill_set) cannot drift -- see tests/test_arm_configs.py.
# Arms run CONCURRENTLY with each other (they are independent processes and hold
# separate engines); seed-level concurrency inside an arm is the eval CLI's own.
#
# ---------------------------------------------------------------------------
# Why MAX_CALLS defaults ABOVE exp1's 150
# ---------------------------------------------------------------------------
# exp1 measured Alive@150 at 100/88/62/88% across its cells: most rollouts hit
# the cap still alive, so the cap -- not the agent -- decided where the episode
# ended. A censored measurement cannot separate a better scaffold from a worse
# one. Raising the budget is what lets depth differentiate.
#
# Raising it is also cheaper than it used to be: Task 14 made rollouts terminate
# at `hp == 0`, so a dead run no longer drains its remaining budget into a
# tombstone `--More--` screen (the committed arm-2 artifact burned 7 of 12 calls
# exactly that way). Runs that die now stop; only survivors spend the extra.
#
# MAX_CALLS binds a DIFFERENT knob per arm ON PURPOSE (see launch_cell.sh and
# configs/README.md Sec 8): `args.max_turns` for the control (LM turns -- a turn
# can execute zero skills, and parallel calls past the first are dropped, so the
# control gets AT MOST this many) vs `taskset.max_skill_calls` for the CLI arms
# (a toolset-side referee granting EXACTLY this many executed skills). They are
# pinned to the same nominal number by convention. aggregate.py normalizes on
# the MEASURED count per rollout; never quote the nominal one.
# ---------------------------------------------------------------------------
set -uo pipefail   # NOT -e: one arm failing must not abort the others

RUN_NAME="${1:?usage: run_sweep.sh <RUN_NAME> [MAX_CALLS] [N] [ARM ...]}"
MAX_CALLS="${2:-400}"
N="${3:-5}"
shift 3 2>/dev/null || shift $#
ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
  ARMS=(control claude_code prime_agent)
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

LAUNCH="tools/cli_harness_eval/launch_cell.sh"
[ -x "$LAUNCH" ] || { echo "missing or non-executable: $LAUNCH" >&2; exit 2; }

RUN_ROOT="outputs/cli_harness_eval/${RUN_NAME}"
mkdir -p "$RUN_ROOT"

# --- provenance: what exactly produced these numbers -------------------------
MANIFEST="${RUN_ROOT}/manifest.txt"
{
  echo "run_name    ${RUN_NAME}"
  echo "started     $(date -Is)"
  echo "max_calls   ${MAX_CALLS}"
  echo "seeds       first ${N} of the pinned explicit_seeds (shuffle=false => seeds 0..$((N-1)))"
  echo "arms        ${ARMS[*]}"
  echo "git_head    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty   $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
} | tee "$MANIFEST"

# --- wall-clock estimate, so nobody starts a 3-day run by accident -----------
# Measured on the committed acceptance artifacts: median ~10s/call (claude_code)
# and ~6s/call (prime_agent), with a heavy tail -- the slowest single call
# observed was 267s. Medians, not means, because means are outlier-dominated.
EST_MIN=$(python3 -c "print(round(${MAX_CALLS}*10*${N}/60))" 2>/dev/null || echo '?')
cat <<EST

  Rough wall-clock, per CLI arm, if every rollout runs to the cap:
    ${N} seeds x ${MAX_CALLS} calls x ~10s median  ~=  ${EST_MIN} min
  Arms run concurrently, so the sweep is about as long as its slowest arm.
  Rollouts that die now terminate early (Task 14), so this is an upper bound.

EST

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] would run, concurrently:"
  for arm in "${ARMS[@]}"; do
    echo "  $LAUNCH $arm ${RUN_ROOT}/${arm} ${MAX_CALLS} ${N}"
  done
  echo "[dry-run] then: python3 tools/cli_harness_eval/aggregate.py ${RUN_ROOT}"
  exit 0
fi

# --- fan out -----------------------------------------------------------------
declare -A PID_OF
for arm in "${ARMS[@]}"; do
  out="${RUN_ROOT}/${arm}"
  log="${RUN_ROOT}/${arm}.log"
  mkdir -p "$out"
  echo "[sweep] starting ${arm} -> ${out} (log: ${log})"
  "$LAUNCH" "$arm" "$out" "$MAX_CALLS" "$N" >"$log" 2>&1 &
  PID_OF["$arm"]=$!
done

FAILED=()
for arm in "${ARMS[@]}"; do
  if wait "${PID_OF[$arm]}"; then
    echo "[sweep] ${arm}: ok"
  else
    rc=$?
    echo "[sweep] ${arm}: FAILED (exit ${rc}) -- see ${RUN_ROOT}/${arm}.log" >&2
    FAILED+=("$arm")
  fi
done

echo "finished    $(date -Is)" >> "$MANIFEST"

# --- aggregate ---------------------------------------------------------------
# Aggregate whatever completed. A partial table beats no table, but say plainly
# which arms are missing so nobody reads a 2-arm table as a 3-arm result.
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[sweep] arms that FAILED: ${FAILED[*]}" | tee -a "$MANIFEST" >&2
  echo "[sweep] aggregating the arms that completed -- the table below is PARTIAL." >&2
fi

echo "[sweep] aggregating ${RUN_ROOT}"
python3 tools/cli_harness_eval/aggregate.py "$RUN_ROOT" | tee "${RUN_ROOT}/table.md"

[ "${#FAILED[@]}" -eq 0 ] || exit 1
