#!/usr/bin/env bash
# Run the encoding sweep: every cell x the first N pinned seeds, then aggregate.
#
#   tools/encoding_eval/run_encoding_sweep.sh run1                 # 400 turns, 5 seeds, 4 cells
#   tools/encoding_eval/run_encoding_sweep.sh run1 400 5           # explicit
#   tools/encoding_eval/run_encoding_sweep.sh smoke 20 1 B0 JSON   # cheap, two cells
#   DRY_RUN=1 tools/encoding_eval/run_encoding_sweep.sh run1       # print the plan
#
# Every cell goes through launch_encoding_cell.sh and shares
# configs/encoding_base.toml, so only `variant` differs between them.
#
# Cost, measured: a 5-seed B0 calibration at max_turns=400 consumed 2.49M input
# tokens and moved the team wallet $0.93 — about $0.19 per rollout. So four
# cells x 5 seeds is roughly $4. The 400 cap is not the cost driver: all five
# calibration rollouts ended on DEATH at 29-140 turns with `is_truncated` false.
set -uo pipefail   # NOT -e: one cell failing must not abort the others

RUN_NAME="${1:?usage: run_encoding_sweep.sh <RUN_NAME> [MAX_TURNS] [N] [CELL ...]}"
MAX_TURNS="${2:-400}"
N="${3:-5}"
shift 3 2>/dev/null || shift $#
CELLS=("$@")
if [ "${#CELLS[@]}" -eq 0 ]; then
  # exp1's four cells. IMG (pixel tileset) is excluded: its renderer needs the
  # nle/MiniHack GlyphMapper tileset, which is absent here.
  CELLS=(B0 JSON TOON IMG_TTY)
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

LAUNCH="tools/encoding_eval/launch_encoding_cell.sh"
[ -x "$LAUNCH" ] || { echo "missing or non-executable: $LAUNCH" >&2; exit 2; }

RUN_ROOT="outputs/encoding_eval/${RUN_NAME}"
mkdir -p "$RUN_ROOT"

MANIFEST="${RUN_ROOT}/manifest.txt"
{
  echo "run_name    ${RUN_NAME}"
  echo "started     $(date -Is)"
  echo "max_turns   ${MAX_TURNS}"
  echo "seeds       first ${N} of the pinned explicit_seeds (seeds 0..$((N-1)))"
  echo "cells       ${CELLS[*]}"
  echo "skill_set   netplay_true (31 vendored upstream NetPlay skills)"
  echo "git_head    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty   $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
} | tee "$MANIFEST"

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] would run, concurrently:"
  for c in "${CELLS[@]}"; do
    echo "  $LAUNCH $c ${RUN_ROOT}/${c} ${MAX_TURNS} ${N}"
  done
  exit 0
fi

declare -A PID_OF
for c in "${CELLS[@]}"; do
  out="${RUN_ROOT}/${c}"
  log="${RUN_ROOT}/${c}.log"
  mkdir -p "$out"
  echo "[sweep] starting ${c} -> ${out} (log: ${log})"
  "$LAUNCH" "$c" "$out" "$MAX_TURNS" "$N" >"$log" 2>&1 &
  PID_OF["$c"]=$!
done

FAILED=()
for c in "${CELLS[@]}"; do
  if wait "${PID_OF[$c]}"; then echo "[sweep] ${c}: ok"
  else echo "[sweep] ${c}: FAILED -- see ${RUN_ROOT}/${c}.log" >&2; FAILED+=("$c"); fi
done

echo "finished    $(date -Is)" >> "$MANIFEST"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[sweep] cells that FAILED: ${FAILED[*]}" | tee -a "$MANIFEST" >&2
  echo "[sweep] the table below is PARTIAL." >&2
fi

echo "[sweep] aggregating ${RUN_ROOT}"
for c in "${CELLS[@]}"; do
  echo "--- ${c} ---"
  .venv-cli-eval/bin/python tools/encoding_eval/aggregate_calib.py "${RUN_ROOT}/${c}" 2>/dev/null
done | tee "${RUN_ROOT}/table.md"

[ "${#FAILED[@]}" -eq 0 ] || exit 1
