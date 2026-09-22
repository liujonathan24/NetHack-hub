#!/usr/bin/env bash
# Batch launcher for the TextWorld arm of the "PAE generalizes" experiment.
#
#   3 games x seeds 0-4 x { base, PAE N=10 }  = 30 runs
#
# Base  = one unchanged BALROG episode (the leaderboard protocol).
# PAE   = up to N attempts on the SAME episode, checkpoint every 10 steps,
#         orchestrator selection + directive (the arm the panel cares about).
#
# Writes under /root/nld/gen_runs/textworld/, takes a `prime wallet` snapshot
# before and after, and aggregates every summary.json at the end.
#
#   bash tools/balrog_pae/games/textworld/launch.sh            # everything
#   GAMES=the_cooking_game SEEDS="0 1" bash .../launch.sh      # a subset
#   DRY_RUN=1 bash .../launch.sh                               # print only
#
# NOTE: this script is NOT run as part of the calibration; launching the batch
# is a separate, explicit decision.
set -uo pipefail

WT=${WT:-/root/nld/gen-textworld}
PY=${PY:-/root/nld/gen-pae/.venv-balrog/bin/python}
OUT=${OUT:-/root/nld/gen_runs/textworld}
GAMES=${GAMES:-"treasure_hunter the_cooking_game coin_collector"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
ARMS=${ARMS:-"base pae"}
ATTEMPTS=${ATTEMPTS:-10}
CKPT_EVERY=${CKPT_EVERY:-10}
PLATEAU=${PLATEAU:-4}
SELECT=${SELECT:-orchestrator}
DIRECTIVE=${DIRECTIVE:-on}
MODEL=${MODEL:-z-ai/glm-5.2}
PARALLEL=${PARALLEL:-3}
DRY_RUN=${DRY_RUN:-0}
TAG=${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}

BATCH="$OUT/batch_$TAG"
mkdir -p "$BATCH"
export PYTHONPATH="$WT"
export PATH="$PATH:/root/.local/bin"

wallet_snapshot() {  # $1 = label
  { echo "=== prime wallet ($1) $(date -u +%FT%TZ) ==="
    prime wallet 2>&1 | head -5
  } | tee -a "$BATCH/wallet.txt"
}

run_one() {  # $1 game  $2 seed  $3 arm
  local game=$1 seed=$2 arm=$3
  local dir="$OUT/${game}_s${seed}_${arm}"
  local log="$dir.log"
  local -a extra
  if [ "$arm" = base ]; then
    extra=(--base-only)
  else
    extra=(--attempts "$ATTEMPTS" --checkpoint-every "$CKPT_EVERY"
           --plateau "$PLATEAU" --select "$SELECT" --directive "$DIRECTIVE")
  fi
  if [ -f "$dir/summary.json" ]; then
    echo "skip (done): $dir"; return 0
  fi
  local cmd=("$PY" -m tools.balrog_pae.run --game textworld --task "$game"
             --seed "$seed" --model "$MODEL" --run-dir "$dir" "${extra[@]}")
  if [ "$DRY_RUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
  echo "start $game s$seed $arm -> $dir"
  ( cd "$WT" && "${cmd[@]}" ) > "$log" 2>&1
  local rc=$?
  echo "$game,$seed,$arm,$rc,$dir" >> "$BATCH/runs.csv"
  [ $rc -ne 0 ] && echo "FAILED rc=$rc: $dir (see $log)"
  return 0
}

echo "batch $TAG -> $BATCH"
echo "game,seed,arm,rc,dir" > "$BATCH/runs.csv"
wallet_snapshot before

jobs_running=0
for game in $GAMES; do
  for seed in $SEEDS; do
    for arm in $ARMS; do
      run_one "$game" "$seed" "$arm" &
      jobs_running=$((jobs_running + 1))
      if [ "$jobs_running" -ge "$PARALLEL" ]; then wait -n 2>/dev/null || wait; jobs_running=$((jobs_running - 1)); fi
    done
  done
done
wait

wallet_snapshot after

# ---- aggregate ------------------------------------------------------------
"$PY" - "$OUT" "$BATCH" <<'PYAGG'
import json, os, sys, glob
out_root, batch = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(out_root, "*_s*_*", "summary.json"))):
    try:
        s = json.load(open(p))
    except Exception as e:
        print("unreadable", p, e); continue
    tok = s.get("tokens", {}) or {}
    tot = tok.get("total", {}) or {}
    rows.append({
        "run": os.path.basename(os.path.dirname(p)),
        "task": s.get("task"), "seed": s.get("seed"), "arm": s.get("arm"),
        "attempts": s.get("attempts"), "checkpoints": s.get("checkpoints"),
        "committed_steps_max": s.get("committed_steps_max"),
        "total_env_steps": s.get("total_env_steps"),
        "llm_steps": s.get("llm_steps"),
        "attempt1_progression": s.get("attempt1_progression"),
        "best_progression": s.get("pae_best_progression"),
        "aux_max": s.get("aux_max_measured"),
        "stop_reason": s.get("stop_reason"),
        "in_tok": tot.get("input_tokens"), "out_tok": tot.get("output_tokens"),
        "list_usd": tot.get("cost_usd"), "billed_usd": tok.get("billed_by_provider_usd"),
        "wall_s": s.get("wall_s"),
    })
json.dump(rows, open(os.path.join(batch, "aggregate.json"), "w"), indent=1)
if rows:
    keys = list(rows[0])
    with open(os.path.join(batch, "aggregate.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    def num(xs): return [x for x in xs if isinstance(x, (int, float))]
    for arm in ("base", "pae"):
        sub = [r for r in rows if r["arm"] == arm]
        for task in sorted({r["task"] for r in sub if r["task"]}):
            t = [r for r in sub if r["task"] == task]
            prog = num([r["best_progression"] for r in t])
            print(f"{arm:<5} {task:<18} n={len(t):>2} "
                  f"progression={(sum(prog)/len(prog) if prog else 0):.3f} "
                  f"list=${sum(num([r['list_usd'] for r in t])):.2f} "
                  f"billed=${sum(num([r['billed_usd'] for r in t])):.2f}")
    print("TOTAL list $%.2f, billed $%.2f, runs %d"
          % (sum(num([r["list_usd"] for r in rows])),
             sum(num([r["billed_usd"] for r in rows])), len(rows)))
print("->", os.path.join(batch, "aggregate.csv"))
PYAGG

echo "batch $TAG complete -> $BATCH"
