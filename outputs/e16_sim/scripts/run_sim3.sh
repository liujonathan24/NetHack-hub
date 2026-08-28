#!/usr/bin/env bash
# SIM 3: launch ONE real NetHack player rollout on seed 1 whose first served
# observation carries an [ORCHESTRATOR DIRECTIVE ...] block.
#
# The directive reaches the env through `E16_ARGS`, launch_cell.sh's
# whitelisted JSON knob (`launch_cell.sh:551-608`), which writes
# `taskset.env_args.directive`. nethack.py reads it into `self._directive`
# (line 543/556), sets `state["_directive_notice"]` in setup_state (799-800),
# and renders it into the FIRST observation's prefix via
# `DIRECTIVE_BLOCK_FORMAT` (line 61, 2278-2280). None of that is this
# simulation's code -- it is the production path, used unmodified.
#
# ISOLATION, because four experiments share this box:
#   EVAL_BIN                    the repo's own venv (not a foreign worktree's)
#   INSTALL_DIR                 private skill-package/kernel-venv install root
#   NETHACK_PA_AGENT_BIND_SRC   private ~/.prime/agent copy, bound over the
#                               shared one INSIDE the rollout's bwrap sandbox
#   OUTDIR                      the scratchpad, NOT the shared repo worktree
#                               (a concurrent agent git-cleaned outputs/ at
#                               10:31 and took this simulation's first set of
#                               artifacts with it)
#
# Usage: run_sim3.sh <tag> <directive-text> [max_calls]
set -euo pipefail

TAG="${1:?usage: run_sim3.sh <tag> <directive> [max_calls]}"
DIRECTIVE="${2:?usage: run_sim3.sh <tag> <directive> [max_calls]}"
MAX_CALLS="${3:-10}"

REPO=/root/nld/zombie-fix
SIM=/tmp/claude-0/-root/3382b9f1-b1ec-4e18-9c3e-4585c4167d08/scratchpad/e16sim
OUTDIR="$SIM/rollouts/$TAG"

export ENG=/root/NetHack-engine
export PYTHONPATH="$REPO/tools/pycompat:$ENG:$REPO:$REPO/environments/nethack"
export EVAL_BIN="$REPO/.venv-cli-eval/bin/eval"
export INSTALL_DIR="/tmp/vf-prime-agent-e16sim"
export NETHACK_PA_AGENT_BIND_SRC="/root/nld/.prime-agent-e16-sim"
key=$(python3 -c "import json;print(json.load(open('/root/.prime/config.json'))['api_key'])")
export PI_API_KEY="$key" PRIME_API_KEY="$key"
unset key

# base tier: the frozen Baseline-v2 surface. The tier names 5 seeds; SEEDS
# overrides row selection to seed 1 alone and TIER_SHORT_BUDGET waives the
# "N must equal the contract's seed count" guard, deliberately, for a smoke.
export TOOL_TIER=base
export SEEDS='[1]'
export TIER_SHORT_BUDGET=1
export STALL_WATCHDOG=1
# A directive of "-" is the NO-DIRECTIVE CONTROL: E16_ARGS is not exported at
# all, so the cell is byte-identical to a plain base-tier seed-1 rollout. It is
# the arm that decides whether the A/B difference is the directive or sampling.
mkdir -p "$OUTDIR"
if [ "$DIRECTIVE" = "-" ]; then
  DIRECTIVE=""
  printf '' > "$OUTDIR/directive.txt"
  printf '{}' > "$OUTDIR/e16_args.json"
  echo "[sim3] NO-DIRECTIVE CONTROL: E16_ARGS unset"
else
  export E16_ARGS
  E16_ARGS=$(DIRECTIVE="$DIRECTIVE" python3 -c '
import json, os
print(json.dumps({"directive": os.environ["DIRECTIVE"]}))')
  printf '%s' "$DIRECTIVE" > "$OUTDIR/directive.txt"
  printf '%s' "$E16_ARGS"  > "$OUTDIR/e16_args.json"
fi

echo "[sim3] tag=$TAG max_calls=$MAX_CALLS out=$OUTDIR"
echo "[sim3] directive: $DIRECTIVE"
cd "$REPO"
exec tools/cli_harness_eval/launch_cell.sh prime_agent "$OUTDIR" "$MAX_CALLS" 1 \
  --harness.path-prepend "/usr/bin:/root/.local/bin"
