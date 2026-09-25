#!/usr/bin/env bash
# Batch launcher for the MiniHack panel of the "PAE generalizes" experiment.
#
# For every (task, seed) it runs:
#   base : ONE unchanged BALROG episode  (--base-only, the leaderboard protocol)
#   pae  : ONE PAE run of N attempts     (orchestrator selection + directive)
#
# Every run gets its own wallet snapshot (before/after) and, because other jobs
# share the Prime wallet, the run's own provider-reported cost is recorded too
# (summary.json -> tokens.billed_by_provider_usd). Trust that one; the wallet
# delta is the upper bound.
#
# Nothing is launched by sourcing this file - you have to run it.
#
#   games/minihack/launch.sh --dry-run
#   games/minihack/launch.sh --tasks "MiniHack-Quest-Easy-v0" --seeds "0 1" --attempts 10
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"         # worktree root
PY="${PY:-$REPO/.venv-balrog/bin/python}"

TASKS="MiniHack-Quest-Easy-v0 MiniHack-Quest-Medium-v0 MiniHack-CorridorBattle-Dark-v0 MiniHack-Boxoban-Medium-v0 MiniHack-Boxoban-Hard-v0"
SEEDS="0 1 2 3 4"
ARMS="base pae"
ATTEMPTS=10
SELECT=orchestrator
DIRECTIVE=on
BLIND=""
CKPT_EVERY=10
PLATEAU=99            # MiniHack progression is binary, so the plateau guard
                      # would fire after P failed attempts; N is the real bound.
MODEL="z-ai/glm-5.2"
OUT_ROOT=/root/nld/gen_runs/minihack
TAG=""
DRY=0
JOBS=1

usage() { sed -n '2,20p' "$0"; exit 0; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks) TASKS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --attempts) ATTEMPTS="$2"; shift 2;;
    --select) SELECT="$2"; shift 2;;
    --directive) DIRECTIVE="$2"; shift 2;;
    --blind) BLIND="--blind"; shift;;
    --checkpoint-every) CKPT_EVERY="$2"; shift 2;;
    --plateau) PLATEAU="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --tag) TAG="_$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

wallet_usd() {
  # NB: must not pipe `prime wallet` into an awk that `exit`s early -- the early
  # close sends prime a SIGPIPE, and under `set -euo pipefail` that non-zero
  # status propagates out of the command substitution and aborts the whole
  # batch before a single run starts. Capture first, then scan the string.
  local out
  out=$(prime wallet 2>/dev/null) || return 0
  awk '/Balance:/ {gsub(/[$,]/,"",$2); print $2; exit}' <<<"$out"
}

BATCH="$OUT_ROOT/batch_$(date +%Y%m%d_%H%M%S)${TAG}"
[[ $DRY -eq 0 ]] && mkdir -p "$BATCH"
echo "batch dir : $BATCH"
echo "tasks     : $TASKS"
echo "seeds     : $SEEDS"
echo "arms      : $ARMS  (pae: N=$ATTEMPTS select=$SELECT directive=$DIRECTIVE $BLIND)"

run_one() {   # $1=task $2=seed $3=arm
  local task="$1" seed="$2" arm="$3"
  local dir="$OUT_ROOT/${task}_s${seed}_${arm}${TAG}"
  if [[ -f "$dir/summary.json" ]]; then echo "skip (done): $dir"; return 0; fi
  local args=(--game minihack --task "$task" --seed "$seed" --model "$MODEL"
              --checkpoint-every "$CKPT_EVERY" --run-dir "$dir")
  if [[ "$arm" == "base" ]]; then
    args+=(--base-only)
  else
    args+=(--attempts "$ATTEMPTS" --plateau "$PLATEAU" --select "$SELECT" --directive "$DIRECTIVE")
    [[ -n "$BLIND" ]] && args+=("$BLIND")
  fi
  if [[ $DRY -eq 1 ]]; then echo "DRY: $PY -m tools.balrog_pae.run ${args[*]}"; return 0; fi

  mkdir -p "$dir"
  local w0 t0 w1 t1 rc=0
  w0=$(wallet_usd); t0=$(date +%s)
  ( cd "$REPO" && "$PY" -m tools.balrog_pae.run "${args[@]}" ) > "$dir/run.log" 2>&1 || rc=$?
  w1=$(wallet_usd); t1=$(date +%s)
  "$PY" - "$dir" "$w0" "$w1" "$t0" "$t1" "$rc" <<'PYEOF'
import json, sys
from pathlib import Path
d, w0, w1, t0, t1, rc = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
def f(x):
    try: return float(x)
    except Exception: return None
snap = {"wallet_before_usd": f(w0), "wallet_after_usd": f(w1),
        "wallet_delta_usd": (f(w0) - f(w1)) if (f(w0) is not None and f(w1) is not None) else None,
        "started": t0, "finished": t1, "wall_s": t1 - t0, "returncode": rc,
        "note": "the wallet is shared with other jobs; tokens.billed_by_provider_usd in "
                "summary.json is this run's own cost, the wallet delta is an upper bound"}
(d / "wallet.json").write_text(json.dumps(snap, indent=2))
print(f"  {d.name}: rc={rc} wall={t1-t0}s wallet_delta={snap['wallet_delta_usd']}")
PYEOF
  return 0
}

for task in $TASKS; do
  for seed in $SEEDS; do
    for arm in $ARMS; do
      if [[ $JOBS -gt 1 && $DRY -eq 0 ]]; then
        run_one "$task" "$seed" "$arm" &
        while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do wait -n; done
      else
        run_one "$task" "$seed" "$arm"
      fi
    done
  done
done
wait

[[ $DRY -eq 1 ]] && { echo "dry run: nothing launched"; exit 0; }

echo
# NB: must run from the repo root -- `python -m tools.balrog_pae...` cannot
# resolve the package from $OUT_ROOT, and under this script's `set -euo
# pipefail` the resulting ModuleNotFoundError kills the batch AFTER the first
# task completes, silently skipping every task queued behind it.
( cd "$REPO" && "$PY" -m tools.balrog_pae.games.minihack.aggregate_minihack "$OUT_ROOT" \
      --json "$BATCH/results.json" --csv "$BATCH/results.csv" ) | tee "$BATCH/results.txt"
echo "aggregate: $BATCH/results.{json,csv,txt}"
