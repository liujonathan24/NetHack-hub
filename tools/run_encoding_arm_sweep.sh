#!/usr/bin/env bash
# The encoding x vision x arm factorial.
#
#   4 encodings  {B0, BBOX, SPARSE_ONDEMAND, JSON}
# x 2 vision     {off, on}          (engine knob tune.reveal_map)
# x 2 arms       {prime_agent, claude_code}
# = 16 cells x SEEDS rollouts.
#
# Budget and timeout are both set deliberately NON-BINDING (1500 calls, 6 h) so a
# rollout ends for a REAL reason -- the character died, or the budget genuinely
# ran out. Exp 1 is the cautionary tale: at 800 calls / 2 h, the vision-off arm
# ended on `harness_timeout` while vision-on ended on `game_over`, so the two
# arms stopped for different reasons and the survival comparison was worthless.
#
#   bash tools/run_encoding_arm_sweep.sh [OUTROOT] [BATCH_CELLS]
#
# Cells run in batches: every rollout holds an MCP tool server (~291 MB) plus its
# agent (~265 MB), so ~0.6 GB each. BATCH_CELLS x SEEDS rollouts are in flight at
# once -- 8 cells x 5 seeds = 40 = ~24 GB against ~60 GB free. The API itself was
# measured flat to 16-way concurrency (median latency 0.76 s -> 0.54 s, throughput
# 1.2 -> 18 req/s), so the cap here is local memory, not the provider.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUTROOT="${1:-$REPO/outputs/e2_encoding_sweep}"
BATCH_CELLS="${2:-8}"
SEEDS="${SEEDS:-5}"
MAX_CALLS="${MAX_CALLS:-1500}"
export ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-21600}"   # 6 h
export MODEL="${MODEL:-z-ai/glm-5.2}"
export ENG="${ENG:-/root/NetHack-engine}"
export STALL_WATCHDOG=1
export MAX_CONCURRENT="${MAX_CONCURRENT:-16}"

SKILLS='"skill_set":"netplay_true,reveal,rollback"'
ENCODINGS=(B0 BBOX SPARSE_ONDEMAND JSON)
ARMS=(prime_agent claude_code)

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO/tools/pycompat:$ENG:$REPO:$REPO/environments/nethack"
if [ -z "${PI_API_KEY:-}" ]; then
  key="$("$REPO/.venv-cli-eval/bin/python" -c \
    "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")"
  export PI_API_KEY="$key" PRIME_API_KEY="$key"
fi

# Refuse to spend anything against an engine no commit identifies.
"$REPO/.venv-cli-eval/bin/python" tools/cli_harness_eval/engine_provenance.py --check || {
  echo "sweep: provenance preflight refused; not launching" >&2; exit 4; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/sweep.log"
echo "sweep start $(date -u +%FT%TZ) model=$MODEL seeds=$SEEDS calls=$MAX_CALLS timeout=${ROLLOUT_TIMEOUT}s" | tee -a "$SUMMARY"

cells=()
for arm in "${ARMS[@]}"; do
  for enc in "${ENCODINGS[@]}"; do
    for vis in off on; do
      cells+=("$arm|$enc|$vis")
    done
  done
done
echo "sweep: ${#cells[@]} cells, ${BATCH_CELLS} per batch, ${SEEDS} seeds each" | tee -a "$SUMMARY"

launch_cell() {
  local arm="$1" enc="$2" vis="$3"
  local name="${arm}__${enc}__vis${vis}"
  local out="$OUTROOT/$name"
  mkdir -p "$out"
  local env_args="{$SKILLS}"
  [ "$vis" = "on" ] && env_args="{$SKILLS,\"tune\":{\"reveal_map\":1.0}}"
  echo "  launch $name" | tee -a "$SUMMARY"
  VARIANT="$enc" ENV_ARGS="$env_args" \
    tools/cli_harness_eval/launch_cell.sh "$arm" "$out" "$MAX_CALLS" "$SEEDS" \
    > "$out/launch.log" 2>&1
  echo "  done   $name rc=$?" | tee -a "$SUMMARY"
}

i=0
while [ $i -lt ${#cells[@]} ]; do
  batch=("${cells[@]:$i:$BATCH_CELLS}")
  echo "== batch $((i/BATCH_CELLS+1)): ${#batch[@]} cells @ $(date -u +%T) ==" | tee -a "$SUMMARY"
  for spec in "${batch[@]}"; do
    IFS='|' read -r arm enc vis <<< "$spec"
    launch_cell "$arm" "$enc" "$vis" &
    sleep 5          # stagger: each cell writes a shared /tmp install lock on first use
  done
  wait
  echo "== batch $((i/BATCH_CELLS+1)) complete @ $(date -u +%T) ==" | tee -a "$SUMMARY"
  i=$((i+BATCH_CELLS))
done

echo "sweep end $(date -u +%FT%TZ)" | tee -a "$SUMMARY"
"$REPO/.venv-cli-eval/bin/python" tools/cli_harness_eval/aggregate.py "$OUTROOT" 2>&1 | tee -a "$SUMMARY"
