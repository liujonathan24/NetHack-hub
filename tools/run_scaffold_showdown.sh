#!/usr/bin/env bash
# Exp 3 — the scaffold comparison, and nothing else.
#
#   claude_code  vs  prime_agent
#   vision ON, search published, batching un-handicapped, encoding BBOX_MIN.
#   Two cells, one question: which scaffold plays better on identical terms.
#
#   bash tools/run_scaffold_showdown.sh [OUTROOT] [BATCH_CELLS]
#
# ---------------------------------------------------------------------------
# What this drops, and why
# ---------------------------------------------------------------------------
# Exp 2 ran 16 cells (4 encodings x 2 vision x 2 arms) at 5 seeds and answered
# neither question it was asked, because the wallet emptied 2h27m in and the
# surviving cells were n=1..5 against a documented 2.2-point variance floor.
# Three of its four factors are now settled well enough to FIX rather than
# sweep:
#
#   * vision: ON. +2.06 dungeon levels, deeper in 7 of 8 paired cells, reached
#     50% deeper in 42% fewer calls, and died LESS (57% vs 71%). Whatever is
#     left to learn about fog is not worth half a budget.
#   * encoding: BBOX. BBOX / BBOX_JSON / B0 came out at 5.87 / 5.83 / 5.64 --
#     inside each other's error bars -- and only SPARSE_ONDEMAND (3.73) is
#     distinguishable, downward. BBOX has the highest pooled mean of the three
#     and the smallest per-turn observation: at turn 1, seed 0, vision on, it
#     pushes 1,110 chars against B0's 2,648, because the ASCII grid is withheld
#     behind `reveal(x1,y1,x2,y2)` while the coordinate lists (VISIBLE FEATURES
#     / VISIBLE MONSTERS / ADJACENT / UNDER PLAYER) are still pushed every turn.
#
#     NOTE, because it is easy to over-claim: that 2.4x is the TURN-1
#     OBSERVATION, not the rollout. Realized cost per 200-call rollout is
#     $31.41 (BBOX) against $34.33 (B0) -- only ~9% apart -- because context is
#     dominated by accumulated history and by `reveal`'s own crops, not by the
#     turn-1 push. Choose BBOX on its mean and its delivery question; do not
#     budget it as though it were 2.4x cheaper.
#
#     Read the crossover honestly: on exp 2's vision-on cells B0 was
#     claude_code's best (11.50, n=2) and BBOX was prime_agent's (6.00, n=4).
#     At n=2..5 against a 2.2-point variance floor that is noise, but it does
#     mean the encoding choice is not neutral between the arms, and this cell
#     picks the one that flattered prime_agent. Say so wherever the result is
#     reported.
#   * the control arm: gone. The question is which CLI SCAFFOLD plays better;
#     the v0 control answers a different one and costs a third of the sweep.
#
# That buys ~3x the seeds per dollar on the comparison that matters.
#
# ---------------------------------------------------------------------------
# The cells: claude_code vs prime_agent on BBOX_MIN, nothing else varies
# ---------------------------------------------------------------------------
# Everything known-good is simply ON in both cells:
#
#   * `search(times<=20)` is published (34 tools). Exp 2 measured 1,058 raw `s`
#     keystrokes on the prime_agent arm in loops up to 50 long; one `search`
#     call replaces up to 20 of them for either arm.
#   * The "Do not batch blind sequences of calls" instruction is STRIPPED from
#     prime_agent's SKILL.md (ALLOW_BATCHING=1). Claude Code never had an
#     equivalent rule and was measured batching 2.31 tool calls per assistant
#     turn -- the restriction was our one-sided handicap. The paired knob
#     MAX_PARALLEL=1 stays off: at the 0.5s batch window it refuses 0.0% of
#     claude_code's calls, so it is a no-op.
#
# The encoding is BBOX_MIN (see nethack_harness/prompt/prompt_spec.py): a quiet
# turn pushes the action feedback line + STATUS (~330 chars, measured live at
# seed 0); ADJACENT / UNDER PLAYER / VISIBLE FEATURES / VISIBLE MONSTERS /
# MESSAGES / INVENTORY arrive only on the turn a `reveal(x1,y1,x2,y2)` executes,
# together with the map crop. `reveal` costs no game turn, so information has a
# price only in budget calls. The plain-BBOX baseline cell was cut deliberately
# (2026-08-01): exp 2 already carries n=4-5 BBOX/vison rollouts per arm at this
# surface's predecessor, and the money buys more here as seeds than as a
# delivery A/B this experiment is not asking.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUTROOT="${1:-$REPO/outputs/e3_scaffold_showdown}"
BATCH_CELLS="${2:-5}"
SEEDS="${SEEDS:-5}"

# 200 calls, not exp2's 1500. Marginal cost per 100 calls for the BBOX/vision-on
# cells specifically, recomputed from the exp2 traces with the REPAIRED meter
# (tools/eval_metrics.py) -- not the all-cell pool, which mixes encodings that
# accumulate context at different rates:
#
#     calls   0-49   $5.32 / 100       rollout of 150 calls  ~$19.35
#            50-99  $13.29 / 100                  200 calls  ~$31.41
#           100-149 $20.08 / 100
#           150-199 $24.14 / 100
#
# Context, not concurrency, drives that curve: median context goes 15k -> 138k
# tokens between call bucket 0-24 and 175-199, and 98% of spend is re-sent
# input. So for a fixed budget MORE SEEDS AT 200 beats fewer at 400, and the
# variance floor is what this experiment is actually fighting.
#
# 200 is also chosen to stay NON-BINDING: exp 2's vision-on rollouts averaged
# 133-175 measured calls and mostly ended on death, so the cap should not be
# what stops a cell. If a rerun starts ending on `budget_exhausted`, raise it
# rather than reading the result -- a censored rollout and a dead one are not
# the same measurement.
MAX_CALLS="${MAX_CALLS:-200}"

# 2h. Exp 2 set 6h to make the bound non-binding and got zero `harness_timeout`
# with a 1500-call cap; at 200 calls the observed worst case is well inside 2h.
# Raise it, do not lower it, if a cell starts ending on the clock -- a timeout
# and a death are not the same measurement.
export ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-7200}"
export MODEL="${MODEL:-z-ai/glm-5.2}"
export ENG="${ENG:-/root/NetHack-engine}"
export STALL_WATCHDOG=1
export MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

SURFACE='netplay_true,reveal,rollback,search'
VISION='"tune":{"reveal_map":1.0}'

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO/tools/pycompat:$ENG:$REPO:$REPO/environments/nethack"
if [ -z "${PI_API_KEY:-}" ]; then
  key="$("$REPO/.venv-cli-eval/bin/python" -c \
    "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")"
  export PI_API_KEY="$key" PRIME_API_KEY="$key"
fi

# --- preflight 1: is anyone going to pay for this? ---------------------------
# Exp 2 died at 19:20:28Z when the balance went to -$15.01, and 31 of its 80
# records are the 402 stubs that followed. The personal balance is STILL empty;
# what works is team billing, and the arm configs already carry
# `[client.headers] X-Prime-Team-ID`. A one-token probe up front is a few cents
# against the alternative of discovering it 2h in.
echo "preflight: probing team billing with a 1-token completion..."
probe="$(curl -s --max-time 30 -X POST "https://api.pinference.ai/api/v1/chat/completions" \
  -H "Authorization: Bearer ${PI_API_KEY}" \
  -H "X-Prime-Team-ID: cmotasmp5005ppyp07dcoh50u" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}")"
case "$probe" in
  *insufficient_funds*|*Insufficient*)
    echo "preflight: FUNDS. The team balance is empty -- this is exactly how exp2" >&2
    echo "  ended. Add funds before launching; not spending anything." >&2
    exit 5 ;;
  *'"choices"'*) echo "preflight: billing ok" ;;
  *) echo "preflight: unexpected reply from the provider, refusing to launch:" >&2
     echo "  ${probe:0:300}" >&2
     exit 5 ;;
esac

# --- preflight 2: which engine ----------------------------------------------
"$REPO/.venv-cli-eval/bin/python" tools/cli_harness_eval/engine_provenance.py --check || {
  echo "sweep: provenance preflight refused; not launching" >&2; exit 4; }

# --- preflight 3: the prime_agent sandbox is ON and works --------------------
# `prime_agent.toml` now sets `sandbox = true`. That is what stops a rollout
# doing what one did in exp2: with the sandbox off, an agent wrote eight
# scripts into /tmp, monkey-patched `NetHackToolset._setup_task_from_channel`
# to hardcode seed 42, resurrected its own game and ran 3.5h outside any cell,
# billing tokens attributable to no experiment.
#
# The sandbox needs BOTH repairs to work at all, and a sweep must not discover
# a missing one at rollout 1 of 25:
#   `--ro-bind /run /run`  -- /etc/resolv.conf symlinks into /run on this host,
#                             so without it DNS dies and prime-agent reports the
#                             generic "Connection error."
#   `~/.prime` bound       -- holds config.json (the API key) and the IPython
#                             kernel venv the agent boots into.
for arm_cfg in prime_agent prime_agent_b80; do
  grep -q '^sandbox = true' "tools/cli_harness_eval/configs/${arm_cfg}.toml" || {
    echo "sweep: ${arm_cfg}.toml does not set sandbox = true -- refusing to launch" >&2
    echo "  an unsandboxed prime_agent arm. See exp2 section 07." >&2; exit 6; }
done
"$REPO/.venv-cli-eval/bin/python" - <<'PYSANDBOX' || exit 6
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.getcwd(), "harnesses", "nethack-prime-agent"))
from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig
workdir = tempfile.mkdtemp(prefix="sandbox-preflight-")
h = PrimeAgentHarness(PrimeAgentHarnessConfig(sandbox=True))
try:
    prefix = h._sandbox_prefix(workdir)
except Exception as exc:                                   # noqa: BLE001
    print(f"sweep: cannot build the bwrap prefix ({exc}); skipping the DNS probe",
          file=sys.stderr)
    raise SystemExit(0)
rc = subprocess.run(list(prefix) + ["getent", "hosts", "api.pinference.ai"],
                    capture_output=True, text=True)
if rc.returncode != 0 or not rc.stdout.strip():
    print("sweep: DNS does NOT resolve inside the bwrap sandbox. This surfaces "
          "later as prime-agent's generic 'Connection error.' at 0 turns. "
          "Check the `--ro-bind /run /run` bind.", file=sys.stderr)
    raise SystemExit(6)
print(f"preflight: sandbox DNS ok ({rc.stdout.split()[0]})")
PYSANDBOX

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/sweep.log"
{
  echo "sweep start $(date -u +%FT%TZ)"
  echo "  model=$MODEL cells=BBOX_MIN x cc/pa vision=on seeds=$SEEDS calls=$MAX_CALLS timeout=${ROLLOUT_TIMEOUT}s"
  echo "  git_head $(git rev-parse HEAD 2>/dev/null || echo unknown) dirty=$(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "  est cost ~\$$("$REPO/.venv-cli-eval/bin/python" -c "
# Per-rollout at 200 calls on plain BBOX (exp2 curves, repaired meter):
# cc \$26.50, pa \$38.30. BBOX_MIN scales those by the measured context-volume
# ratio, which depends on how often the agent buys the feed with reveal --
# 0.44/0.50 (cc/pa) at the observed ~2% rate, 0.68/0.72 at every-4th-turn,
# 0.92/0.93 at every-2nd. Mid scenario quoted, range alongside.
cc, pa = 26.50, 38.30
lo  = ${SEEDS}*(cc*0.44 + pa*0.50)
mid = ${SEEDS}*(cc*0.68 + pa*0.72)
hi  = ${SEEDS}*(cc*0.92 + pa*0.93)
print(f'{mid:.0f} (range {lo:.0f}-{hi:.0f})')") for 2 cells x $SEEDS seeds"
} | tee -a "$SUMMARY"

# cell name | arm | variant | extra env. The prime_agent cell strips the
# no-batching rule; both cells get the search skill via $SURFACE.
cells=(
  "BBOX_MIN__claude_code|claude_code|BBOX_MIN|"
  "BBOX_MIN__prime_agent|prime_agent|BBOX_MIN|ALLOW_BATCHING=1"
)

launch_cell() {
  local name="$1" arm="$2" variant="$3" extra="$4"
  local out="$OUTROOT/$name"
  mkdir -p "$out"
  # SKILL_SET goes INSIDE env_args: launch_cell.sh refuses the two together,
  # because a whole-object env_args override replaces the table and would drop
  # a separately-passed skill_set.
  local env_args="{\"skill_set\":\"${SURFACE}\",${VISION}}"
  echo "  launch $name (arm=$arm variant=$variant ${extra:-no-extra})" | tee -a "$SUMMARY"
  env $extra VARIANT="$variant" ENV_ARGS="$env_args" \
    tools/cli_harness_eval/launch_cell.sh "$arm" "$out" "$MAX_CALLS" "$SEEDS" \
    > "$out/launch.log" 2>&1
  echo "  done   $name rc=$?" | tee -a "$SUMMARY"
}

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] preflights passed. Would launch, ${BATCH_CELLS} at a time:"
  for spec in "${cells[@]}"; do
    IFS='|' read -r name arm variant extra <<< "$spec"
    echo "  ${name}: arm=${arm} variant=${variant} seeds=${SEEDS} calls=${MAX_CALLS} surface=${SURFACE} ${extra:-}"
  done
  echo "[dry-run] then: stall_watchdog.py --backfill && aggregate.py ${OUTROOT}"
  exit 0
fi

i=0
while [ $i -lt ${#cells[@]} ]; do
  batch=("${cells[@]:$i:$BATCH_CELLS}")
  echo "== batch $((i/BATCH_CELLS+1)): ${#batch[@]} cells @ $(date -u +%T) ==" | tee -a "$SUMMARY"
  for spec in "${batch[@]}"; do
    IFS='|' read -r name arm variant extra <<< "$spec"
    launch_cell "$name" "$arm" "$variant" "$extra" &
    sleep 5          # stagger: cells share a /tmp install lock on first use
  done
  wait
  echo "== batch $((i/BATCH_CELLS+1)) complete @ $(date -u +%T) ==" | tee -a "$SUMMARY"
  i=$((i+BATCH_CELLS))
done

echo "sweep end $(date -u +%FT%TZ)" | tee -a "$SUMMARY"

# Recover the attempts the watchdog killed BEFORE aggregating: a SIGKILLed
# rollout never gets a `traces.jsonl` record, and exp2 lost 50 attempts
# (6,195 turn records, 32,145 in-game turns) that way while still being billed.
"$REPO/.venv-cli-eval/bin/python" tools/stall_watchdog.py --backfill "$OUTROOT" \
  2>&1 | tee -a "$SUMMARY"
"$REPO/.venv-cli-eval/bin/python" tools/cli_harness_eval/aggregate.py "$OUTROOT" 2>&1 | tee -a "$SUMMARY"

# Did the CAP decide where these rollouts ended? On exp 2's vision-on cells it
# did not -- only 3 of 22 claude_code and 2 of 26 prime_agent rollouts ever
# reached 200 calls, and every BALROG milestone through 20% was reached by call
# ~186 -- which is the entire argument for a 200-call horizon. If the fixes in
# this sweep keep rollouts alive longer, that stops being true, and the run
# after this one needs a bigger cap. This is the check that tells you, instead
# of a censored table that looks like a result.
"$REPO/.venv-cli-eval/bin/python" - "$OUTROOT" <<'PYBIND' 2>&1 | tee -a "$SUMMARY"
import glob, json, os, sys
root = sys.argv[1]
tot = binding = 0
for path in glob.glob(os.path.join(root, "*", "traces.jsonl")):
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        m = t.get("metrics") or {}
        if not m:
            continue
        tot += 1
        if m.get("budget_exhausted"):
            binding += 1
if not tot:
    print("[cap-check] no scored rollouts")
elif binding / tot > 0.2:
    print(f"[cap-check] WARNING: {binding}/{tot} rollouts hit the call cap "
          f"({100*binding/tot:.0f}%). The BUDGET, not the game, ended them -- "
          f"depth here is censored. Re-run at a higher MAX_CALLS before "
          f"quoting these depths against anything.")
else:
    print(f"[cap-check] ok: {binding}/{tot} rollouts hit the call cap "
          f"({100*binding/tot:.0f}%); the cap is not what ended this sweep.")
PYBIND
