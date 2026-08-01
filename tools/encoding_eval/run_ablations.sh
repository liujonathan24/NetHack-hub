#!/usr/bin/env bash
# Run the encoding-family ablations: the remaining encodings, plus 1b (JSON
# cell content), 1c (memory), and 1d (observation timing).
#
#   tools/encoding_eval/run_ablations.sh abl1 400 5
#   DRY_RUN=1 tools/encoding_eval/run_ablations.sh abl1
#
# Every cell is encoding_base.toml plus a named override, so each differs from
# its baseline in exactly the keys listed here and nothing else.
#
# Cost: a 5-seed cell at max_turns=400 measured ~$0.19/rollout (2.49M input
# tokens, $0.93 wallet delta for 5). These 14 cells x 5 seeds ~= $13.
set -uo pipefail

RUN_NAME="${1:?usage: run_ablations.sh <RUN_NAME> [MAX_TURNS] [N] [CELL ...]}"
MAX_TURNS="${2:-400}"
N="${3:-5}"
shift 3 2>/dev/null || shift $#
ONLY=("$@")   # optional: rerun just these cell names

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
LAUNCH="tools/encoding_eval/launch_encoding_cell.sh"
RUN_ROOT="outputs/encoding_eval/${RUN_NAME}"
mkdir -p "$RUN_ROOT"

# name|variant|EXTRA_ARGS
#
# --- remaining encodings -----------------------------------------------------
# B is BALROG's own natural-language scene ("horizontal wall near north", no
# ASCII grid) — the closest thing we have to the observation their leaderboard
# run actually consumed, so it is the most direct like-for-like cell.
# B1 is compacted ASCII, included only as the compaction control exp1 excluded.
#
# --- 1b: JSON cell content ---------------------------------------------------
# Does enriching each map cell with spatial/exploration attributes close JSON's
# exp1 gap? Cells are cumulative so the marginal value of each layer is visible.
#
# --- 1c: memory --------------------------------------------------------------
# journal-only is the default. belief-state distils every N turns via a sub-LM.
# no-memory strips the journal entirely (no pinned objective, no notes).
#
# --- 1d: observation timing --------------------------------------------------
# Driven by the VARIANT, not by a kwarg. DM = delayed map (action feedback every
# turn, full map re-sent only on request/material change), DM_JSON = the same
# over the JSON body, BBOX = the map exposed only through a bounding-box query.
# There is no `obs_mode` parameter on `load_environment` — that field lives on
# the v1 taskset config — so passing one would land in **kwargs and be silently
# ignored, yielding a cell identical to its baseline.
CELLS=(
  # Base encodings, so a single low-concurrency invocation can cover both the
  # encoding sweep and the ablations.
  "enc_B0|B0|"
  "enc_TOON|TOON|"
  "enc_JSON|JSON|"
  "enc_IMG_TTY|IMG_TTY|"
  "enc_B|B|"
  "enc_B1|B1|"
  # BALROG-shaped: natural-language scene description AND the map, which is what
  # their published run CSVs actually carry. B alone drops the grid; B0/JSON
  # alone drop the description — neither reproduces the leaderboard's input.
  "enc_B_ascii|B_ASCII|"
  "enc_B_json|B_JSON|"
  "1b_none|JSON|{\"cell_schema\":[]}"
  "1b_seen|JSON|{\"cell_schema\":[\"seen\"]}"
  "1b_seen_visited|JSON|{\"cell_schema\":[\"seen\",\"visited\"]}"
  "1b_all|JSON|{\"cell_schema\":[\"seen\",\"visited\",\"reach\"]}"
  "1c_journal_only|B0|{\"belief_state_interval\":0,\"summarize_and_reset\":false}"
  "1c_belief_state|B0|{\"belief_state_interval\":25,\"sub_lm_model\":\"google/gemini-3-flash-preview\"}"
  "1c_summarize_reset|B0|{\"belief_state_interval\":25,\"sub_lm_model\":\"google/gemini-3-flash-preview\",\"summarize_and_reset\":true}"
  "1c_no_memory|B0|{\"belief_state_interval\":0,\"pin_objective_on_setup\":false,\"history_keep_full\":2}"
  "1d_dm|DM|"
  "1d_dm_json|DM_JSON|"
  "1d_bbox|BBOX|"
  # --- 1d, corrected -----------------------------------------------------------
  # The cells above ran WITHOUT `request_map`: it is deliberately absent from the
  # netplay presets, and the documented comma form ("<preset>,request_map") did
  # not expand the preset, so the arm collapsed to a single tool. Those cells
  # therefore measured "map only on a new floor, with no way to ask for it" —
  # not delayed delivery. These re-run with the tool actually exposed.
  "1d_dm_fixed|DM|{\"skill_set\":\"netplay_true,request_map\"}"
  "1d_bbox_fixed|BBOX|{\"skill_set\":\"netplay_true,reveal\"}"
  # --- combination of the three best independent treatments ---------------------
  # language description (B_ASCII) + delayed map (DM) + journal-only memory (1c).
  # The first two are complementary by construction: DM withholds the grid, and
  # the description is what keeps the agent oriented while it is withheld.
  "combo_lang_dm_journal|DM_B_ASCII|{\"skill_set\":\"netplay_true,request_map\",\"belief_state_interval\":0,\"summarize_and_reset\":false}"
)

if [ "${#ONLY[@]}" -gt 0 ]; then
  KEEP=()
  for spec in "${CELLS[@]}"; do
    IFS='|' read -r nm _ _ <<< "$spec"
    for want in "${ONLY[@]}"; do
      [ "$nm" = "$want" ] && KEEP+=("$spec")
    done
  done
  CELLS=("${KEEP[@]}")
fi

{
  echo "run_name  ${RUN_NAME}"
  echo "started   $(date -Is)"
  echo "max_turns ${MAX_TURNS}   seeds first ${N}"
  echo "cells     ${#CELLS[@]}"
  echo "git_head  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
} | tee "${RUN_ROOT}/manifest.txt"

if [ -n "${DRY_RUN:-}" ]; then
  for spec in "${CELLS[@]}"; do
    IFS='|' read -r name variant extra <<< "$spec"
    echo "[dry-run] ${name}: variant=${variant} extra=${extra:-none}"
  done
  exit 0
fi

# MAX_PARALLEL caps how many CELLS are in flight at once. Total concurrent
# rollouts = MAX_PARALLEL x N, and that product is what matters: fanning all 19
# cells out at N=5 put 95 rollouts on a shared login node, and the host killed
# them — silently, with SIGKILL, so the logs showed no error at all. Cells died
# staggered as each crossed the CPU threshold, which is why the failures looked
# encoding-specific when they were not.
MAX_PARALLEL="${MAX_PARALLEL:-2}"

FAILED=()
running=0
declare -A PID_OF

_reap_one() {
  local name pid
  for name in "${!PID_OF[@]}"; do
    pid="${PID_OF[$name]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then echo "[abl] ${name}: ok"
      else echo "[abl] ${name}: FAILED -- see ${RUN_ROOT}/${name}.log" >&2; FAILED+=("$name"); fi
      unset "PID_OF[$name]"
      running=$((running - 1))
      return 0
    fi
  done
  sleep 5
  return 1
}

for spec in "${CELLS[@]}"; do
  IFS='|' read -r name variant extra <<< "$spec"
  while [ "$running" -ge "$MAX_PARALLEL" ]; do _reap_one || true; done
  out="${RUN_ROOT}/${name}"
  mkdir -p "$out"
  echo "[abl] starting ${name} (variant=${variant} extra=${extra:-none}) [${running}/${MAX_PARALLEL} slots busy]"
  EXTRA_ARGS="$extra" "$LAUNCH" "$variant" "$out" "$MAX_TURNS" "$N" \
    >"${RUN_ROOT}/${name}.log" 2>&1 &
  PID_OF["$name"]=$!
  running=$((running + 1))
done

while [ "$running" -gt 0 ]; do _reap_one || true; done

echo "finished  $(date -Is)" >> "${RUN_ROOT}/manifest.txt"
[ "${#FAILED[@]}" -eq 0 ] || echo "[abl] FAILED cells: ${FAILED[*]}" | tee -a "${RUN_ROOT}/manifest.txt" >&2

for spec in "${CELLS[@]}"; do
  IFS='|' read -r name _ _ <<< "$spec"
  echo "--- ${name} ---"
  .venv-cli-eval/bin/python tools/encoding_eval/aggregate_calib.py "${RUN_ROOT}/${name}" 2>/dev/null
done | tee "${RUN_ROOT}/table.md"
