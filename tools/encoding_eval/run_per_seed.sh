#!/usr/bin/env bash
# Run cells ONE SEED PER PROCESS, so no single process accumulates enough CPU
# to trip the login node's per-process ceiling.
#
#   tools/encoding_eval/run_per_seed.sh abl5 400 5 1b_none 1b_seen 1b_seen_visited 1b_all
#
# Why this exists. The 1b cells were killed at MAX_PARALLEL=2 AND at 3, having
# written 80-170 turns each — so the limit is not concurrency, and lowering it
# again would not help. It is CPU accumulated by a single process: one eval
# invocation carrying all 5 rollouts of a heavy cell (JSON plus per-tile
# attribute layers, the largest observations in the sweep) crosses the ceiling
# before finishing. Splitting the same work across 5 processes keeps each one
# under it.
#
# Crucially this does NOT shorten the games: max_turns stays whatever you pass,
# every rollout still ends on death or the cap, and the scored result is
# identical to a single-process run. Only the process boundary moves.
#
# Each (cell, seed) writes its per-turn NDJSON into the SHARED <cell>/turns/
# directory, so the existing aggregator sees all 5 rollouts as one cell.
set -uo pipefail

RUN_NAME="${1:?usage: run_per_seed.sh <RUN_NAME> [MAX_TURNS] [NSEEDS] [CELL ...]}"
MAX_TURNS="${2:-400}"
NSEEDS="${3:-5}"
shift 3 2>/dev/null || shift $#
CELLS=("$@")
[ "${#CELLS[@]}" -gt 0 ] || { echo "no cells given" >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
LAUNCH="tools/encoding_eval/launch_encoding_cell.sh"
RUN_ROOT="outputs/encoding_eval/${RUN_NAME}"
mkdir -p "$RUN_ROOT"

# cell -> the EXTRA_ARGS that define it (mirrors run_ablations.sh).
declare -A EXTRA=(
  [1b_none]='{"cell_schema":[]}'
  [1b_seen]='{"cell_schema":["seen"]}'
  [1b_seen_visited]='{"cell_schema":["seen","visited"]}'
  [1b_all]='{"cell_schema":["seen","visited","reach"]}'
  [enc_B_json]=''
  [1d_dm_fixed]='{"skill_set":"netplay_true,request_map"}'
  [1d_dm_json]=''
  [1c_journal_only]='{"belief_state_interval":0,"summarize_and_reset":false}'
  # The two best-scoring treatments stacked: bounding-box on-demand map
  # (1d_bbox_fixed, 3.58) + journal-only memory (1c_journal_only, 3.22).
  # Unlike the earlier DM+language combo these are orthogonal — one governs how
  # the map is delivered, the other whether belief-state distillation runs — so
  # neither needs a new variant, just BBOX plus the memory kwargs. `reveal` must
  # be in the skill set or BBOX has no way to show the map at all.
  [combo_bbox_journal]='{"skill_set":"netplay_true,reveal","belief_state_interval":0,"summarize_and_reset":false}'
  # 2x2 closing the belief-state confound. `1d_bbox_fixed` (3.58) was labelled
  # "pure BBOX" but ran with belief_state_interval at its default 25 — 4,464
  # belief_state notes in its traces — while combo_bbox_journal inherited 0.
  # So that pair already gives the ASCII row; these two give the JSON row, with
  # the interval set EXPLICITLY in every cell rather than inherited.
  [bbox_json_bs25]='{"skill_set":"netplay_true,reveal","belief_state_interval":25}'
  [bbox_json_bs0]='{"skill_set":"netplay_true,reveal","belief_state_interval":0,"summarize_and_reset":false}'
  # TOON re-baseline. Every prior TOON number was scored with `grid: 2359x79` —
  # raw glyph ids — so the agent navigated on the entity lines alone. The grid
  # now renders readable characters; this establishes the real baseline.
  [toon_fixed]='{"belief_state_interval":25}'
  [toon_minimal]='{"map_detail":"minimal","belief_state_interval":25}'
  # ---- BALROG matched-action-space cells (`skill_set=balrog80`) ----
  # Their agent gets the 80 text commands in balrog/environments/nle/__init__.py
  # and nothing else -- no pathfinding, no closed loops, no in-skill item
  # selection. See tools/balrog_actions.py. Run at 2500 steps because one raw
  # keystroke does far less than one np_* macro call.
  #
  # These are the five highest-priority observation configs, re-run on that
  # surface, so each pairs against a known netplay_true number.
  [b80_bbox]='{"skill_set":"balrog80,reveal","belief_state_interval":25}'
  [b80_b0]='{"skill_set":"balrog80","belief_state_interval":25}'
  [b80_bbox_json]='{"skill_set":"balrog80,reveal","belief_state_interval":25}'
  # The 1c memory contrast needs the journal TOOLS present in both arms, or the
  # comparison degrades into "tools vs no tools" instead of distillation on/off.
  # balrog80 has none of its own (BALROG gives its agent no memory affordance),
  # so they are added explicitly to these two and to neither of the others.
  [b80_journal]='{"skill_set":"balrog80,add_note,recall,pin_objective","belief_state_interval":0,"summarize_and_reset":false}'
  [b80_belief]='{"skill_set":"balrog80,add_note,recall,pin_objective","belief_state_interval":25,"sub_lm_model":"google/gemini-3-flash-preview"}'
  # ---- model-strength arms: our best 4 configs on stronger models ----
  # Identical to the netplay_true cells that produced 3.58 / 3.49 / 3.22 /
  # 3.21 with gemini-3-flash. ONLY the model changes, so any delta is model
  # strength and not encoding, action surface, or memory policy.
  [m_f36_bbox_fixed]='{"skill_set":"netplay_true,reveal","belief_state_interval":25}'
  [m_f36_b0]='{"belief_state_interval":25}'
  [m_f36_journal]='{"belief_state_interval":0,"summarize_and_reset":false}'
  [m_f36_bboxjson]='{"skill_set":"netplay_true,reveal","belief_state_interval":25}'
  [m_p31_bbox_fixed]='{"skill_set":"netplay_true,reveal","belief_state_interval":25}'
  [m_p31_b0]='{"belief_state_interval":25}'
  [m_p31_journal]='{"belief_state_interval":0,"summarize_and_reset":false}'
  [m_p31_bboxjson]='{"skill_set":"netplay_true,reveal","belief_state_interval":25}'

)
declare -A VARIANT=(
  [1b_none]=JSON [1b_seen]=JSON [1b_seen_visited]=JSON [1b_all]=JSON
  [enc_B_json]=B_JSON [1d_dm_fixed]=DM [1d_dm_json]=DM_JSON
  [1c_journal_only]=B0 [combo_bbox_journal]=BBOX
  [bbox_json_bs25]=BBOX_JSON [bbox_json_bs0]=BBOX_JSON
  [toon_fixed]=TOON [toon_minimal]=TOON
  [b80_bbox]=BBOX [b80_b0]=B0 [b80_bbox_json]=BBOX_JSON
  [b80_journal]=B0 [b80_belief]=B0
  [m_f36_bbox_fixed]=BBOX
  [m_f36_b0]=B0
  [m_f36_journal]=B0
  [m_f36_bboxjson]=BBOX_JSON
  [m_p31_bbox_fixed]=BBOX
  [m_p31_b0]=B0
  [m_p31_journal]=B0
  [m_p31_bboxjson]=BBOX_JSON
)

# Per-cell model override (empty/absent => the base config's
# google/gemini-3-flash-preview). Only the model-strength arms set this.
declare -A MODEL_OF=(
  [m_f36_bbox_fixed]='google/gemini-3.6-flash'
  [m_f36_b0]='google/gemini-3.6-flash'
  [m_f36_journal]='google/gemini-3.6-flash'
  [m_f36_bboxjson]='google/gemini-3.6-flash'
  [m_p31_bbox_fixed]='google/gemini-3.1-pro-preview'
  [m_p31_b0]='google/gemini-3.1-pro-preview'
  [m_p31_journal]='google/gemini-3.1-pro-preview'
  [m_p31_bboxjson]='google/gemini-3.1-pro-preview'
)


MAX_PARALLEL="${MAX_PARALLEL:-3}"
running=0
declare -A PID_OF

_reap() {
  local k pid
  for k in "${!PID_OF[@]}"; do
    pid="${PID_OF[$k]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" && echo "[ps] ${k}: ok" || echo "[ps] ${k}: FAILED" >&2
      unset "PID_OF[$k]"; running=$((running - 1)); return 0
    fi
  done
  sleep 5; return 1
}

{
  echo "run_name  ${RUN_NAME}"
  echo "started   $(date -Is)"
  echo "mode      one seed per process (games run to completion; max_turns=${MAX_TURNS})"
  echo "cells     ${CELLS[*]}"
  echo "seeds     0..$((NSEEDS-1))"
  echo "parallel  ${MAX_PARALLEL}"
} | tee "${RUN_ROOT}/manifest.txt"

for cell in "${CELLS[@]}"; do
  out="${RUN_ROOT}/${cell}"
  mkdir -p "${out}/turns"
  OUT_ABS="$(cd "$out" && pwd)"
  for s in $(seq 0 $((NSEEDS-1))); do
    while [ "$running" -ge "$MAX_PARALLEL" ]; do _reap || true; done
    base="${EXTRA[$cell]:-}"
    # Pin this invocation to exactly one seed, and send its per-turn NDJSON to
    # the cell's shared turns/ dir so the aggregator sees one cell, not five.
    merged="$(python3 -c "
import json,sys
d = json.loads(sys.argv[1]) if sys.argv[1] else {}
d['explicit_seeds'] = [int(sys.argv[2])]
d['n_examples'] = 1
d['trace_dir'] = sys.argv[3]
print(json.dumps(d))" "$base" "$s" "${OUT_ABS}/turns")"
    echo "[ps] starting ${cell} seed ${s} [${running}/${MAX_PARALLEL} busy]"
    MODEL="${MODEL_OF[$cell]:-}" EXTRA_ARGS="$merged" "$LAUNCH" "${VARIANT[$cell]}" "${OUT_ABS}/s${s}" "$MAX_TURNS" 1 \
      >"${RUN_ROOT}/${cell}.s${s}.log" 2>&1 &
    PID_OF["${cell}/s${s}"]=$!
    running=$((running + 1))
  done
done
while [ "$running" -gt 0 ]; do _reap || true; done

echo "finished  $(date -Is)" >> "${RUN_ROOT}/manifest.txt"
for cell in "${CELLS[@]}"; do
  echo "--- ${cell} ---"
  .venv-cli-eval/bin/python tools/encoding_eval/aggregate_calib.py "${RUN_ROOT}/${cell}" 2>/dev/null
done | tee "${RUN_ROOT}/table.md"
