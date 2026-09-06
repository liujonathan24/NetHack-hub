#!/usr/bin/env bash
# Regenerate every E16 artifact dataset and rebuild both pages.
#
# Publishing is NOT done here -- only Claude can call the Artifact tool. This
# script leaves two rebuilt HTML files ready to publish, and prints a short
# status block so the caller can say what actually changed.
#
#   bash refresh_e16.sh
set -uo pipefail
SP="${E16_ART_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO=/root/nld/zombie-fix
PY="$REPO/.venv-cli-eval/bin/python"
cd /root/nld/e16_runs || exit 1

# Runs to include, newest-interesting first. A run with no attempts.jsonl is
# skipped by the explorer generator but still picked up by the curve script,
# which reads the turn stream -- that is deliberate, so an in-flight first
# attempt still plots.
# THE TWO LIVE ARMS ONLY. The earlier runs are kept on disk and can be added
# back by name, but they are answering superseded questions and their presence
# made the plots read as six comparable arms when they are not:
#   treesmoke7  ran blind on hunger AND on the player's account
#   treesmoke6  publishes rollback, so its attempts are not single-life
#   nullsmoke1  the matched null for treesmoke7, not for these
#   treesmoke10 aborted; 2 of 4 attempts lost to infrastructure
#   e16_s{0,2,3,4}_r1 are the seed sweep: one replica per seed nobody had
#   played, capped at 30 attempts. The three treesmoke11 runs are all seed 1.
#   e16_gem_s1_r1 was DELETED. It was meant to run google/gemini-3.7-flash but
#   the MODEL env override is overwritten by the tier registry, so it played
#   glm-5.2 -- a duplicate seed-1 replica, not a Gemini arm. A real Gemini run
#   needs a declared tier; there is no env path that sets the player model.
#   e16_fog_s*_r1: the fog-of-war sweep (reveal_map=0.0), one replica per
#   seed, all finished 2026-09-05: s0/s3/s4 at the 30-attempt cap, s1 stalled
#   at 26, s2 stalled at 21.
RUNS="treesmoke8 treesmoke11 treesmoke11_r2 treesmoke11_r3 e16_s0_r1 e16_s2_r1 e16_s3_r1 e16_s4_r1 e16_fog_s0_r1 e16_fog_s1_r1 e16_fog_s2_r1 e16_fog_s3_r1 e16_fog_s4_r1"

"$PY" "$SP/gen_e16_data.py" $RUNS > "$SP/e16_data.json" 2>"$SP/.gen_data.err" || {
  echo "[refresh] FAILED generating e16_data.json"; tail -3 "$SP/.gen_data.err"; exit 1; }

"$PY" "$SP/gen_traces_data.py" $RUNS > "$SP/traces_data.json" 2>"$SP/.gen_tr.err" || {
  echo "[refresh] FAILED generating traces_data.json"; tail -3 "$SP/.gen_tr.err"; exit 1; }

ARGS=""; for r in $RUNS; do ARGS="$ARGS /root/nld/e16_runs/$r"; done
"$PY" "$REPO/tools/cli_harness_eval/e16_progress_curve.py" $ARGS -o "$SP/curves.json" >/dev/null 2>&1 || {
  echo "[refresh] FAILED generating curves.json"; exit 1; }

# BASELINE REFERENCE LINES: E14 uncapped base, seed 1, all three reps. Single
# uninterrupted lives -- no archive, no resume, no directive -- plotted so the
# E16 arms can be read against what one plain rollout does on the same seed.
BASE_DIR=$REPO/outputs/e14_uncapped
"$PY" "$REPO/tools/cli_harness_eval/artifacts/gen_baseline_curves.py" 1 \
  $BASE_DIR/base_r1__prime_agent $BASE_DIR/base_r2__prime_agent $BASE_DIR/base_r3__prime_agent \
  > "$SP/baseline_curves.json" 2>/dev/null || echo "{}" > "$SP/baseline_curves.json"
"$PY" - "$SP" <<'MERGE'
import json,sys
SP=sys.argv[1]
c=json.load(open(SP+'/curves.json'))
try: b=json.load(open(SP+'/baseline_curves.json'))
except Exception: b={}
c.update(b)
json.dump(c,open(SP+'/curves.json','w'),separators=(',',':'))
MERGE

(cd "$SP" && python3 build_explorer.py >/dev/null && python3 build_traces.py >/dev/null) || {
  echo "[refresh] FAILED rebuilding pages"; exit 1; }

# Status the caller can report without re-deriving anything.
"$PY" - "$SP" <<'PY'
import json,sys,os,glob
SP=sys.argv[1]
d=json.load(open(SP+'/e16_data.json')); c=json.load(open(SP+'/curves.json'))
print("[refresh] ok  %s" % os.popen("date -u +%FT%TZ").read().strip())
for r in c:
    v=d.get(r)
    cv=c[r]
    closed=len(v['attempts']) if v else 0
    print("  %-20s closed=%-3d turns=%-5d  BALROG max %-6s min %-5s"%(
        r,closed,cv.get('total_turns',0),cv.get('final_best_max'),cv.get('final_best_min')))
alive=0
for p in glob.glob('/proc/[0-9]*'):
    try: cm=open(p+'/cmdline','rb').read().decode(errors='replace')
    except Exception: continue
    if 'e16_orchestrator' in cm and '/bin/bash -c' not in cm: alive+=1
print("  orchestrators alive: %d"%alive)
PY

# Mirror the build chain out of /tmp on every tick. The scratchpad is session-
# scoped; this is the only thing standing between a cleared /tmp and rebuilding
# the generators from scratch. Never fails the refresh -- a backup problem must
# not stop the artifacts from being published.
bash /root/nld/e16_runs/backup_artifacts.sh 2>&1 | sed 's/^/  /' || true
