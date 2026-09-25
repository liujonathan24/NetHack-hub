#!/usr/bin/env bash
# Reproduce every check in REPORT.md. No LLM calls; CPU only.
#
#   SCRATCH=/some/dir bash tools/balrog_ckpt/run_all.sh            # full install + run
#   SKIP_INSTALL=1 SCRATCH=/some/dir bash tools/balrog_ckpt/run_all.sh
#
# Layout under $SCRATCH: BALROG/ (git clone), .venv/ (uv venv, python 3.12),
# mh_engine_<task>/ (patched NetHack data dirs for the fork-engine path).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${SCRATCH:-/tmp/claude-0/-root-NetHack-hub/5557e37f-5844-469d-83a8-5ac4ec77b191/scratchpad/balrog_ckpt}"
PY312="${PY312:-/root/NetHack-hub/.venv-cli-eval/bin/python}"   # any python3.12 works
UV="${UV:-$HOME/.local/bin/uv}"
ENGINE_ROOT="${NETHACK_ENGINE_ROOT:-/root/NetHack-engine}"
mkdir -p "$SCRATCH"; cd "$SCRATCH"

if [[ -z "${SKIP_INSTALL:-}" ]]; then
  [[ -d BALROG ]] || git clone --depth 1 https://github.com/balrog-ai/BALROG.git BALROG
  "$UV" venv --python "$PY312" .venv
  # BALROG's setup.py pulls balrog-nle, minihack/textworld/minigrid/baba forks, crafter, gym==0.23
  "$UV" pip install --python .venv/bin/python -e ./BALROG
  # balrog-nle imports pkg_resources (setuptools<70 provides it; not in a bare 3.12 venv)
  "$UV" pip install --python .venv/bin/python "setuptools<70"
  # Boxoban levels (into minihack/dat) + TextWorld games (tw_games/ next to the balrog package)
  (cd BALROG && ../.venv/bin/balrog-post-install)
fi
PY="$SCRATCH/.venv/bin/python"
export PYTHONPATH="$ENGINE_ROOT"

echo "### Crafter"
"$PY" "$HERE/verify_crafter.py" --seeds 1,2,3,4,5 --prefix 30 --branch 50

echo "### MiniHack, path 2: balrog-nle replay determinism"
"$PY" "$HERE/verify_minihack_replay.py" --task MiniHack-Quest-Easy-v0                                   # dies in lava at step 13 (deterministically)
"$PY" "$HERE/verify_minihack_replay.py" --task MiniHack-Quest-Easy-v0 --actions north,south,west,search # lava-safe: 30 + 50 steps
"$PY" "$HERE/verify_minihack_replay.py" --task MiniHack-Boxoban-Medium-v0

echo "### MiniHack, path 1: NetHack-engine fork + MiniHack .des (needs $ENGINE_ROOT built)"
for t in MiniHack-Boxoban-Medium-v0 MiniHack-Corridor-R3-v0 MiniHack-MazeWalk-9x9-v0 MiniHack-CorridorBattle-Dark-v0; do
  "$PY" "$HERE/verify_minihack_engine.py" --task "$t" --workdir "$SCRATCH/mh_engine_$t" || echo "!! $t: exit $?"
done
"$PY" "$HERE/verify_minihack_engine.py" --task MiniHack-Boxoban-Medium-v0 --no-redraw --workdir "$SCRATCH/mh_engine_MiniHack-Boxoban-Medium-v0" || true
# Quest-Easy / Quest-Medium abort inside nle_fr_restore -> NetHackRL::load_mirror (see REPORT.md)
"$PY" "$HERE/verify_minihack_engine.py" --task MiniHack-Quest-Easy-v0 --workdir "$SCRATCH/mh_engine_MiniHack-Quest-Easy-v0" || echo "!! Quest-Easy engine path: exit $? (expected: SIGABRT/SIGSEGV in load_mirror)"

echo "### TextWorld"
"$PY" "$HERE/verify_textworld.py" --tasks treasure_hunter,the_cooking_game,coin_collector

echo "results in $HERE/results/"
