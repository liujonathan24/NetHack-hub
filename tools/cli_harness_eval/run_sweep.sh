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

# --- provenance: which ENGINE, not just which harness ------------------------
# `git_head` above pins the HARNESS only. The engine is a submodule plus a
# compiled .so, and both have drifted silently before: `third_party/NetHack`
# checked out to 66c84e6 against a fefd557 pin (~30 tests then failing as if the
# engine had logic bugs), and a libnethack.so built 2026-07-22 04:26 from a
# src/src/nle.c last touched 2026-07-31 18:52 -- a 230h gap that every rollout
# links and nothing reports. `results/run_provenance.json` recorded the harness
# HEAD only, so no past result can be attributed to an engine. New sweeps carry
# the full fingerprint; launch_cell.sh separately REFUSES to start a cell whose
# .so provably predates the source.
#
# The interpreter and PYTHONPATH here MUST match launch_cell.sh's, or this
# records a different .so than the cells load: the resolution goes through
# `nethack_core._engine.library_path()`, so a bare `python3` with no ENG on the
# path resolves NOTHING and writes an all-null fingerprint (measured -- the
# first version of this block did exactly that).
ENG="${ENG:-/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness}"
ENGINE_PROV="tools/cli_harness_eval/engine_provenance.py"
PROV_PY="${REPO}/.venv-cli-eval/bin/python"
[ -x "$PROV_PY" ] || PROV_PY=python3
if [ -f "$ENGINE_PROV" ]; then
  ENGINE_LINE="$(PYTHONPATH="${ENG}:${REPO}:${REPO}/environments/nethack" "$PROV_PY" "$ENGINE_PROV" \
      --json "${RUN_ROOT}/engine_provenance.json" \
      --run-record results/run_provenance.json \
      --run "${RUN_NAME}" \
      --driver "${RUN_NAME}.log" \
      --extra "max_calls=${MAX_CALLS} n=${N} arms=${ARMS[*]}" \
      2>/dev/null | head -1)"
  echo "engine      ${ENGINE_LINE:-capture failed}" | tee -a "$MANIFEST"
fi

# --- wall-clock estimate, so nobody starts a 3-day run by accident -----------
# Measured on the committed acceptance artifacts: median ~10s/call (claude_code)
# and ~6s/call (prime_agent), with a heavy tail -- the slowest single call
# observed was 267s. Medians, not means, because means are outlier-dominated.
# The eval CLI's --max-concurrent defaults to 128, so all N seeds of an arm are
# in flight at once; arms are fanned out concurrently here too. Wall-clock is
# therefore about ONE rollout, not N of them.
EST_MIN=$(python3 -c "print(round(${MAX_CALLS}*10/60))" 2>/dev/null || echo '?')
cat <<EST

  Rough wall-clock ~= ONE rollout, since seeds run concurrently (--max-concurrent
  defaults to 128) and the arms are fanned out here:
    ${MAX_CALLS} calls x ~10s median  ~=  ${EST_MIN} min
  Rollouts that die terminate early (Task 14), so this is an upper bound.
  Concurrency caveat: the prime_agent arm shares a Prime Agent daemon; even
  SEQUENTIAL rollouts on that arm are unproven (Task 10 concern 3). If it
  misbehaves at N>1, run it with MAX_CONCURRENT=1 via the eval CLI.

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
