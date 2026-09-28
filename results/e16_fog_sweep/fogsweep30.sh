#!/usr/bin/env bash
# Fog-of-war Go-Explore on one seed, 30 attempts. Completes the five-seed set.
#
# WHAT THIS IS. The seed-parameterised form of fogrun30.sh, which ran seed 1 as
# e16_fog_s1_r1. Every environment variable below is copied from that script
# unchanged; the only difference is that E16_SEED and RUN_DIR come from the
# argument. Run it once per seed for 0, 2, 3 and 4 and the arm is complete at
# five seeds x 30 attempts, matching the unfogged sweep.
#
# CONFIRMED IDENTICAL TO THE SEED-1 RUN before launch, against served bytes
# rather than against this script: tool_tiers.toml hashes to 1f65e8a46431cd0b,
# the same tier_registry_sha256_16 that e16_fog_s1_r1 recorded, so the tier
# definition has not drifted. e16_fog_s1_r1/attempts/a001/config.toml is the
# reference for the rest:
#     tool_tier       = "e16_gewiki_norb_fog"
#     tool_tier_hash  = "1f65e8a46431cd0b"
#     skill_set       = "np_core,request_map,search,save,wiki"
#     reveal_map      = "0.0"
#     max_skill_calls = 0        (uncapped -- attempts end on death)
#     max_relaunches  = 2
#     model           = "z-ai/glm-5.2"
#     character       = "Val-hum-neu-fem"
#
# WHAT SEED 1 DID, for expectations. It stopped itself at 26 of 30 attempts on
# the plateau guard (stop_reason=no_new_frontier: twelve consecutive attempts
# without advancing the frontier), reached Dlvl 22 at XL 8, and cost about $50
# measured against the wallet. Expect these to stall early too; that is the
# guard working, not a failure.
#
#   bash fogsweep30.sh <SEED>
set -uo pipefail
SEED="${1:?usage: fogsweep30.sh <SEED>}"
RUN_DIR="/root/nld/e16_runs/e16_fog_s${SEED}_r1"
REPO=/root/nld/zombie-fix
mkdir -p "$RUN_DIR"
exec >>"${RUN_DIR}.driver.log" 2>&1

export E16_BUDGET=100000
export E16_NO_INFLIGHT_BUDGET=1
export E16_MAX_ATTEMPTS=30
export E16_STALL=12
export E16_MILESTONE_DLVL=0
export E16_MILESTONE_DUNGEON=-1
export E16_TIER=e16_gewiki_norb_fog
export E16_SEED="$SEED"
export ENG=/root/nld/e16-engine
export EVAL_BIN="${REPO}/.venv-cli-eval/bin/eval"
RUN="${REPO}/tools/cli_harness_eval/run_e16.sh"

echo "=============================================================="
echo "[launch] $(date -u +%FT%TZ) e16_fog_s${SEED}_r1 -- seed ${SEED}, N=30, stall=12"
echo "[launch] harness: $(cd "$REPO" && git log --oneline -1)"
echo "[launch] tier=$E16_TIER  reveal_map=0.0  milestones off"
echo "[launch] matches e16_fog_s1_r1; unfogged control is e16_s${SEED}_r1"
echo "=============================================================="
for step in prepare seed probe; do
  echo "----- [$step] $(date -u +%FT%TZ) -----"
  if [ "$step" = seed ] && [ -f "${RUN_DIR}/archive/c1/meta.json" ]; then
    echo "[launch] archive already seeded at ${RUN_DIR}/archive/c1 -- skipping"
    continue
  fi
  bash "$RUN" "$RUN_DIR" "$step" || { echo "[launch] FAILED at $step"; exit 1; }
done
echo "----- [run] $(date -u +%FT%TZ) -----"
bash "$RUN" "$RUN_DIR" run; rc=$?
echo "[launch] exited rc=$rc at $(date -u +%FT%TZ)"
exit $rc
