#!/usr/bin/env bash
# Run the E14 Diversity reps in sequence: wait for rep 1, then run 2 and 3.
#
# Staged by REP rather than launching all fifteen cells at once. A rep is five
# concurrent cells (25 concurrent rollouts), the same shape the base arm ran at;
# fifteen at once is 75, and a wedged daemon at that width takes the whole grid
# with it rather than one rep.
#
# LOGDIR defaults to logs/ beside this script. The 2026-09-01 run used an
# absolute scratch path; only the location differed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR"
until grep -q "rep1 complete" "$LOGDIR/rep1.log" 2>/dev/null; do sleep 60; done
./run_e14div_full.sh 2 > "$LOGDIR/rep2.log" 2>&1
./run_e14div_full.sh 3 > "$LOGDIR/rep3.log" 2>&1
echo "ALL THREE REPS COMPLETE"
