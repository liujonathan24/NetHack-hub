#!/usr/bin/env bash
# Crafter batch for the "PAE generalizes" panel: seeds 0-9 x {base, PAE N=10}.
#
# base arm : one unchanged BALROG episode per seed, at BALROG's own Crafter
#            horizon (2000 steps; the episode ends on death long before that).
# PAE arm  : N=10 attempts on the SAME episode, orchestrator selection +
#            directive, with an EXPLICIT --max-steps cap (tasks.md section 7:
#            PAE resumes before death, so an uncapped run drifts toward 2000
#            committed steps and costs ~8x the base episode).
#
# Prints the plan and the projected cost, then STOPS unless --go is passed.
# Takes a `prime wallet` snapshot before and after, and aggregates at the end.
#
#   games/crafter/launch.sh                 # plan + cost only, launches nothing
#   games/crafter/launch.sh --go            # actually run
#   games/crafter/launch.sh --go --seeds "0 1 2" --arms base
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="$REPO/.venv-balrog/bin/python"
OUT_ROOT="${OUT_ROOT:-/root/nld/gen_runs/crafter}"

SEEDS="0 1 2 3 4 5 6 7 8 9"
ARMS="base pae"
ATTEMPTS=10
PAE_MAX_STEPS=400          # explicit cap on the PAE arm (see tasks.md section 7)
BASE_MAX_STEPS=2000        # BALROG's own Crafter horizon
CHECKPOINT_EVERY=10
PLATEAU=4
SELECT=orchestrator
DIRECTIVE=on
MODEL="z-ai/glm-5.2"
TAG="$(date +%Y%m%d)"
GO=0
JOBS=1

while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --attempts) ATTEMPTS="$2"; shift 2 ;;
    --pae-max-steps) PAE_MAX_STEPS="$2"; shift 2 ;;
    --base-max-steps) BASE_MAX_STEPS="$2"; shift 2 ;;
    --select) SELECT="$2"; shift 2 ;;
    --directive) DIRECTIVE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

BATCH="$OUT_ROOT/batch_$TAG"
N_SEEDS=$(echo $SEEDS | wc -w)

# --- projected cost -----------------------------------------------------------
# MEASURED on this stack (games/crafter/calibration.json), GLM-5.2 over Prime.
IN_TOK_PER_STEP=${IN_TOK_PER_STEP:-1930}
OUT_TOK_PER_STEP=${OUT_TOK_PER_STEP:-150}
BASE_STEPS_EST=${BASE_STEPS_EST:-215}
PRICE_IN=1.54; PRICE_OUT=4.84
BILLED_RATIO=${BILLED_RATIO:-0.39}

read -r PLAN_TXT <<'EOF'
EOF
COST=$("$PY" - "$N_SEEDS" "$ATTEMPTS" "$PAE_MAX_STEPS" "$BASE_STEPS_EST" \
        "$IN_TOK_PER_STEP" "$OUT_TOK_PER_STEP" "$BILLED_RATIO" "$ARMS" <<'PYEOF'
import sys
n, att, cap, base_steps, tin, tout, ratio, arms = sys.argv[1:]
n, att, cap, base_steps, tin, tout, ratio = int(n), int(att), int(cap), int(base_steps), int(tin), int(tout), float(ratio)
PIN, POUT = 1.54, 4.84
def usd(steps):
    return steps * (tin * PIN + tout * POUT) / 1e6
# base: one episode per seed, ends on death at ~base_steps
base_llm = base_steps
# PAE: attempt 1 = a base episode; each later attempt resumes ~K steps before the
# end of the previous one and plays to the cap, so LLM calls per later attempt
# are bounded by the cap and floored by the measured resume length.
pae_llm = base_steps + (att - 1) * min(cap, base_steps)
rows, tot_l = [], 0.0
if "base" in arms:
    c = n * usd(base_llm); tot_l += c
    rows.append(("base", n, base_llm, c))
if "pae" in arms:
    c = n * usd(pae_llm) * 1.05; tot_l += c   # +5% orchestrator turns
    rows.append((f"PAE N={att} (cap {cap})", n, pae_llm, c))
out = []
for name, nn, steps, c in rows:
    out.append(f"  {name:<24} {nn:>2} runs x ~{steps:>5} LLM calls  ${c:8.2f} list  ${c*ratio:7.2f} billed")
out.append(f"  {'TOTAL':<24} {'':>2}                         ${tot_l:8.2f} list  ${tot_l*ratio:7.2f} billed")
print("\n".join(out))
PYEOF
)

cat <<EOF
================================================================================
Crafter batch plan
================================================================================
repo            $REPO
python          $PY
out root        $BATCH
seeds           $SEEDS   (n=$N_SEEDS)
arms            $ARMS
model           $MODEL
base arm        --base-only --max-steps $BASE_MAX_STEPS   (BALROG's own horizon)
PAE arm         --attempts $ATTEMPTS --select $SELECT --directive $DIRECTIVE \\
                --checkpoint-every $CHECKPOINT_EVERY --plateau $PLATEAU --max-steps $PAE_MAX_STEPS
env patches     crafter_balance_chunk_sorted, crafter_seed_pinned_to_episode_seed
                (both arms; recorded in every summary.json under env_patches)
parallel jobs   $JOBS

projected cost (measured ${IN_TOK_PER_STEP} in / ${OUT_TOK_PER_STEP} out tokens per step,
                base episode ~${BASE_STEPS_EST} steps, billed/list ${BILLED_RATIO}x):
$COST
================================================================================
EOF

if [ "$GO" -ne 1 ]; then
  echo "DRY RUN - nothing launched. Re-run with --go to execute."
  exit 0
fi

mkdir -p "$BATCH"
cd "$REPO" || exit 1
prime wallet > "$BATCH/wallet_before.txt" 2>&1 || echo "prime wallet failed" > "$BATCH/wallet_before.txt"
head -4 "$BATCH/wallet_before.txt"
git -C "$REPO" rev-parse HEAD > "$BATCH/commit.txt" 2>/dev/null

run_one() {  # arm seed
  local arm="$1" seed="$2"
  local dir="$BATCH/${arm}_s${seed}"
  if [ -f "$dir/summary.json" ]; then echo "skip $arm s$seed (done)"; return 0; fi
  rm -rf "$dir"
  if [ "$arm" = "base" ]; then
    "$PY" -m tools.balrog_pae.run --game crafter --task default --seed "$seed" \
      --model "$MODEL" --base-only --max-steps "$BASE_MAX_STEPS" \
      --run-dir "$dir" > "$BATCH/${arm}_s${seed}.log" 2>&1
  else
    "$PY" -m tools.balrog_pae.run --game crafter --task default --seed "$seed" \
      --model "$MODEL" --attempts "$ATTEMPTS" --select "$SELECT" --directive "$DIRECTIVE" \
      --checkpoint-every "$CHECKPOINT_EVERY" --plateau "$PLATEAU" --max-steps "$PAE_MAX_STEPS" \
      --run-dir "$dir" > "$BATCH/${arm}_s${seed}.log" 2>&1
  fi
  echo "done $arm s$seed rc=$? -> $dir"
}
export -f run_one 2>/dev/null || true

for arm in $ARMS; do
  for seed in $SEEDS; do
    if [ "$JOBS" -le 1 ]; then
      run_one "$arm" "$seed"
    else
      run_one "$arm" "$seed" &
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    fi
  done
done
wait

prime wallet > "$BATCH/wallet_after.txt" 2>&1 || echo "prime wallet failed" > "$BATCH/wallet_after.txt"
echo "--- wallet before/after"
grep -i balance "$BATCH/wallet_before.txt" "$BATCH/wallet_after.txt"

"$PY" -m tools.balrog_pae.games.crafter.aggregate "$BATCH" \
  --json "$BATCH/aggregate.json" --csv "$BATCH/aggregate.csv" | tee "$BATCH/aggregate.txt"
