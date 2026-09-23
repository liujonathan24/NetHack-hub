#!/usr/bin/env bash
# TextWorld arm: extend the panel from 5 to 10 seeds (BALROG's own default of
# 10 episodes per TextWorld task).  Same protocol as seeds 0-4 -- verified
# against /root/nld/gen_runs/panel/tw_treasure_hunter_s0/config.json.
# Resumable: a run whose summary.json exists is skipped.
set -u
REPO=/root/nld/gen-pilot
PY=/root/nld/gen-pae/.venv-balrog/bin/python     # same interpreter/BALROG as seeds 0-4
OUT=/root/nld/gen_runs/panel
JOBS=${JOBS:-5}
mkdir -p "$OUT"

wallet() { prime wallet 2>/dev/null | grep -oP 'Balance:\s+\$\K[0-9.]+' | head -1; }

run_one() {
  local name="$1"; shift
  local dir="$OUT/$name"
  if [ -f "$dir/summary.json" ]; then echo "SKIP $name (summary.json exists)"; return 0; fi
  if [ -f "$OUT/ABORT_TW10" ]; then echo "ABORT-SKIP $name"; return 0; fi
  local wb; wb=$(wallet)
  echo "START $name wallet=$wb $(date -Is)"
  ( cd "$REPO" && "$PY" -m tools.balrog_pae.run --run-dir "$dir" "$@" ) \
      > "$OUT/$name.log" 2>&1
  local rc=$?
  local wa; wa=$(wallet)
  echo "{\"run\":\"$name\",\"rc\":$rc,\"wallet_before\":\"$wb\",\"wallet_after\":\"$wa\",\"t\":\"$(date -Is)\"}" \
      >> "$OUT/wallet_per_run.jsonl"
  if [ $rc -ne 0 ]; then
    echo "{\"run\":\"$name\",\"rc\":$rc,\"tail\":$(tail -c 800 "$OUT/$name.log" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}" >> "$OUT/crashes.jsonl"
    echo "CRASH $name rc=$rc"
  else
    echo "DONE  $name wallet=$wa"
  fi
  return 0
}
export -f run_one wallet
export REPO PY OUT

JOBLIST=$OUT/joblist_tw10.txt
: > "$JOBLIST"
for g in the_cooking_game treasure_hunter coin_collector; do
  for s in 5 6 7 8 9; do
    echo "tw_${g}_s$s --game textworld --task $g --seed $s --attempts 10 --plateau 4 --select orchestrator --directive on" >> "$JOBLIST"
  done
done

echo "tw10: $(wc -l < "$JOBLIST") jobs, JOBS=$JOBS"
xargs -a "$JOBLIST" -d '\n' -P "$JOBS" -I{} bash -c 'run_one {}'
echo "TW10 COMPLETE $(date -Is)"
