#!/usr/bin/env bash
# E16 Go-Explore at the observability floor: game seed 1, 30 attempts, fog on.
#
# WHAT THIS IS. One replica of [e16_gewiki_norb_fog], which is the tier the
# seven finished Go-Explore arms ran ([e16_gewiki_norb]) with `reveal_map` set
# to 0.0 and nothing else changed. Its matched control is the three replicas
# already on disk at the same game seed with the map revealed:
# treesmoke11, treesmoke11_r2, treesmoke11_r3.
#
# WHY NOT "GO-EXPLORE ON [base]". [base] inherits the global contract's
# skill_set = "np_core,request_map,search" and so does not publish `save`.
# Go-Explore's archive is written by `save`; without it there is no checkpoint,
# no archive, and no selection loop. A base-tier Go-Explore arm is not a
# cheaper version of this experiment, it is not an experiment.
#
# CONFIG DELTAS FROM THE CONTROL THREE -- there are two, and both are intended:
#   E16_TIER          e16_gewiki_norb_fog instead of e16_gewiki_norb
#   E16_MAX_ATTEMPTS  30 instead of 100
# E16_SEED is set explicitly to 1 rather than left to run_e16.sh's default,
# which is how the control three ended up on seed 1 without any launcher
# saying so. Stall and milestone settings match what the control three ran
# under after extend100*.sh, so stop semantics are shared.
#
# BUDGET. Wallet was $309.15 at launch. Measured cost on the control tier is
# $6.72/attempt calibrated to the wallet, so 30 attempts is ~$202 plus ~$7 of
# orchestrator. Fog is expected to run longer per attempt than the control did
# (v0_fog measured 1.5x the LM calls of its own control), so the stall guard
# and the attempt cap are what stand between this and an overdraft. Watch
# .driver.log; do not raise E16_MAX_ATTEMPTS without re-checking the wallet.
#
#   bash fogrun30.sh
set -uo pipefail
RUN_DIR="/root/nld/e16_runs/e16_fog_s1_r1"
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
export E16_SEED=1
export ENG=/root/nld/e16-engine
export EVAL_BIN="${REPO}/.venv-cli-eval/bin/eval"
RUN="${REPO}/tools/cli_harness_eval/run_e16.sh"

echo "=============================================================="
echo "[launch] $(date -u +%FT%TZ) e16_fog_s1_r1 -- seed 1, N=30, stall=12"
echo "[launch] harness: $(cd "$REPO" && git log --oneline -1)"
echo "[launch] tier=$E16_TIER  reveal_map=0.0  milestones off"
echo "[launch] control: treesmoke11, treesmoke11_r2, treesmoke11_r3"
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
