#!/usr/bin/env bash
# E16 Go-Explore on one seed -- the run launcher.
#
#   run_e16.sh <RUN_DIR> [SUBCOMMAND]
#
# Subcommands:
#   preflight the three silent hangs, exercised for real against a stub
#             prime-agent: a --resume from a foreign cwd, a call that blocks
#             past its deadline, the shared daemon socket, and json mode +
#             --resume. NO model calls, no spend. `run` refuses to start until
#             this passes, because each of those three fails by HANGING rather
#             than erroring, and a hang on this budget is the whole run.
#   prepare   create the run tree, copy the wiki in, write provenance.json.
#             NO model calls. Safe to run any number of times.
#   seed      write checkpoint c1 (a fresh game at the dungeon entrance on the
#             run's seed) through the normal checkpoint path. NO model calls.
#   probe     the two-call `--resume` continuity check. THE ONLY subcommand
#             that spends inference, and it spends about two sentences of it.
#             Writes `session_resume_verified` into provenance.json.
#   run       the loop: orchestrator round -> player attempt -> ingest, until
#             budget / stall / milestone. SPENDS THE RUN'S BUDGET.
#   resume    CONTINUE a run whose orchestrator died. Reconciles the archive
#             against the attempt rows, finalizes whatever attempt was in
#             flight as censored:interrupted (attributing the checkpoints it
#             left behind), re-attaches the orchestrator's recorded session so
#             the optimization conversation continues rather than restarting,
#             and carries spend forward so nothing is charged twice. SPENDS
#             THE REMAINING BUDGET.
#   reconcile reconciliation ONLY -- no session, no launches, NO SPEND. Run it
#             on any run directory to attribute orphaned archive checkpoints,
#             finalize interrupted attempts, and be told what it could not
#             attribute. Exits 1 if anything was left unattributed.
#   status    print the current summary.json.
#   watch     tail the live progress stream (progress.jsonl).
#
# ENVIRONMENT
#   The isolation rules of this box are not optional here; every one of them
#   was written after an incident.
#
#   ENG                 engine checkout (default /root/NetHack-engine). The
#                       launcher's PYTHONPATH is built from it, and WITHOUT
#                       that PYTHONPATH a different nethack_harness (from
#                       /root/NetHack-hub) is imported and nothing works.
#   EVAL_BIN            the eval binary. Point it at a PRIVATE venv: the
#                       shared hub venv's editable install resolves harness
#                       code out of another agent's worktree, which a
#                       `uv pip install` there can rewrite mid-run.
#   NETHACK_PA_AGENT_BIND_SRC
#                       per-experiment copy of ~/.prime/agent, bound over
#                       /root/.prime/agent inside each PLAYER sandbox. The
#                       shared agent dir holds daemon-workers/ and
#                       session-leases/; a booting experiment scanning it has
#                       been measured reaping another experiment's live
#                       rollouts.
#   ORCH_MODEL / ORCH_PROVIDER
#                       the ORCHESTRATOR's model route. The provider is pinned
#                       explicitly because `--model z-ai/glm-5.2` alone is a
#                       model PATTERN: it matched openrouter's catalog entry
#                       first and died with "No API key found for openrouter"
#                       even though settings.json names prime-inference.
#   E16_SELECTOR        llm (default, the design) | scripted (the ablation).
#   E16_BUDGET          USD ceiling for this run. Hard: checked before every
#                       launch. Default 385 (the design's ~55% of $700).
#   E16_NO_DIRECTIVE=1  run-wide control: same archive, same selection, no
#                       instructions to players.
#   E16_PAIRED_CONTROL=1
#                       PER-DIRECTIVE-KIND control: run every directive twice
#                       from the same checkpoint, with and without it. This is
#                       the one the sims showed is necessary -- a run-wide
#                       control cannot tell a directive that changed behaviour
#                       from one that prohibited something the player was never
#                       going to do. (`descend_fast` inverted the first
#                       decision causally; `no_descend` was indistinguishable
#                       from the control, because the control did not descend
#                       either.)
#   E16_EXP_ARM         go_explore (default, the method)
#                     | matched_restart (THE NULL: N independent attempts from
#                       ONE fixed start state, no archive, no selection, no
#                       directives, no lessons). Every other control here is an
#                       ablation WITHIN the method and is beaten by "we got N
#                       tries instead of one"; the headline number is a running
#                       maximum over attempts and cannot fall. Without this arm
#                       at matched N there is no reading under which the method
#                       could have failed.
#   E16_PLAYER_ARM      the LAUNCHER's arm (default prime_agent). Distinct from
#                       E16_EXP_ARM, which names the experiment condition.
#   E16_START_CHECKPOINT
#                       matched_restart only: the fixed state every attempt
#                       restarts from. Default: the archive's seed checkpoint.
#   E16_RESEED=1        reseed the engine RNG after every restore. OFF by
#                       default, deliberately: a restore otherwise resets to
#                       the game's original seed and replays no history, so two
#                       attempts from one checkpoint differ only in what the
#                       model chose -- which is what makes a directive's effect
#                       attributable to the directive rather than to dice.
#                       Recorded in provenance.json either way.
#   E16_ORCH_TMPDIR     the orchestrator's PRIVATE TMPDIR (default
#                       <agent-dir>/tmp). The unsandboxed orchestrator
#                       otherwise shares /tmp/prime-agent-0/daemon.sock with
#                       every prime-agent on the box: measured 900s of hang
#                       against 3.5s with a private one. Set on the
#                       orchestrator's own env only -- see the DO NOT export
#                       note below.
#   E16_ORCH_JSON_MODE  discovery_only (default) | always | never. json mode +
#                       --resume was measured hanging where the identical text
#                       --print call answered 30s later, so json mode runs only
#                       on the round that has no session id to resume.
#   E16_ORCH_TIMEOUT    hard per-round deadline in seconds (default 900). The
#                       call's whole process group is killed at the deadline
#                       and the round is recorded as an error, so no
#                       orchestrator call can consume the run's wall clock.
#
# DO NOT export TMPDIR from here. It leaks into the sandboxed rollouts, where
# the path does not exist, and pointing it at a bind-mounted directory
# collapses concurrent rollouts onto one daemon socket. Players already get
# private sockets from bwrap --tmpfs /tmp.
set -uo pipefail

RUN_DIR="${1:?usage: run_e16.sh <RUN_DIR> [prepare|seed|probe|run|status]}"
CMD="${2:-run}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export ENG="${ENG:-/root/NetHack-engine}"
export EVAL_BIN="${EVAL_BIN:-${REPO}/.venv-cli-eval/bin/eval}"
PY_BIN="${PY_BIN:-$(dirname "${EVAL_BIN}")/python}"
[ -x "$PY_BIN" ] || PY_BIN="${REPO}/.venv-cli-eval/bin/python"

# THE import path, in this order. tools/pycompat first (its sitecustomize
# carries the two vendor quirk fixes every CLI arm needs); then the engine;
# then the repo; then environments/nethack. Getting this wrong imports the
# wrong nethack_harness and everything downstream is quietly the wrong code.
export PYTHONPATH="${REPO}/tools/pycompat:${ENG}:${REPO}:${REPO}/environments/nethack:${REPO}/tools/cli_harness_eval${PYTHONPATH:+:${PYTHONPATH}}"

WIKI_SRC="${E16_WIKI_SRC:-/root/nld/e15-wiki/configs/continual/wiki}"
[ -d "$WIKI_SRC" ] || {
  echo "run_e16: no curated wiki subset at ${WIKI_SRC}" >&2
  exit 2
}

ORCH="${REPO}/tools/cli_harness_eval/e16_orchestrator.py"

PREFLIGHT="${REPO}/tools/cli_harness_eval/e16_preflight.py"

# E16_ARM used to name the LAUNCHER's arm. It now names the EXPERIMENT arm, and
# the launcher's is E16_PLAYER_ARM. An old E16_ARM=prime_agent would silently
# select an arm that does not exist, so it is caught rather than translated:
# a launch is not the place to guess which of two things someone meant.
if [ -n "${E16_ARM:-}" ]; then
  echo "run_e16: E16_ARM is no longer the launcher's arm." >&2
  echo "  E16_EXP_ARM    = go_explore | matched_restart  (the experiment arm)" >&2
  echo "  E16_PLAYER_ARM = prime_agent | claude_code ... (the launcher's arm)" >&2
  echo "  You set E16_ARM=${E16_ARM}. Say which you meant." >&2
  exit 2
fi

ARGS=(
  "$RUN_DIR"
  --wiki-src "$WIKI_SRC"
  --tier "${E16_TIER:-e16_gewiki}"
  --arm "${E16_EXP_ARM:-go_explore}"
  --player-arm "${E16_PLAYER_ARM:-prime_agent}"
  --selector "${E16_SELECTOR:-llm}"
  --game-seed "${E16_SEED:-1}"
  --rng-seed "${E16_RNG_SEED:-20260828}"
  --budget "${E16_BUDGET:-385}"
  --min-headroom "${E16_MIN_HEADROOM:-5}"
  --max-attempts "${E16_MAX_ATTEMPTS:-200}"
  --stall-attempts "${E16_STALL:-8}"
  --orch-json-mode "${E16_ORCH_JSON_MODE:-discovery_only}"
  --orch-timeout "${E16_ORCH_TIMEOUT:-900}"
  --progress-interval "${E16_PROGRESS_INTERVAL:-15}"
)
[ -n "${ORCH_MODEL:-}" ] && ARGS+=(--orch-model "$ORCH_MODEL")
[ -n "${E16_ORCH_AGENT_DIR:-}" ] && ARGS+=(--orch-agent-dir "$E16_ORCH_AGENT_DIR")
[ -n "${E16_ORCH_TMPDIR:-}" ] && ARGS+=(--orch-tmpdir "$E16_ORCH_TMPDIR")
[ -n "${E16_START_CHECKPOINT:-}" ] && ARGS+=(--start-checkpoint "$E16_START_CHECKPOINT")
[ "${E16_NO_DIRECTIVE:-0}" = "1" ] && ARGS+=(--no-directive)
# E16_NO_INFLIGHT_BUDGET=1 disables the mid-attempt budget guard. The guard
# charges wall clock at DEFAULT_SPEND_RATE_USD_PER_HOUR ($90/hr) x a 2.0 safety
# factor and there is NO in-flight usage source for prime_agent rollouts
# (extra_usage is empty), so `spend_rate_source` never leaves `prior`: the rate
# it enforces is a guess that no run can ever correct. Measured in treesmoke4
# at E16_BUDGET=45 it capped attempt 1 at 800s of wall clock (it ran 811s) and
# attempts 2/3 at 394s/191s, which is ~1/5 of the ~64 min a natural death takes
# on this harness -- i.e. it made "play until the character dies" impossible by
# construction. Turn it off when the run's purpose is to reach a natural end or
# to MEASURE the real rate; leave it on for cost-controlled sweeps.
[ "${E16_NO_INFLIGHT_BUDGET:-0}" = "1" ] && ARGS+=(--no-inflight-budget)
[ "${E16_PAIRED_CONTROL:-0}" = "1" ] && ARGS+=(--paired-control)
[ "${E16_RESEED:-0}" = "1" ] && ARGS+=(--reseed)

case "$CMD" in
  preflight) exec "$PY_BIN" "$PREFLIGHT" ;;
  prepare) exec "$PY_BIN" "$ORCH" "${ARGS[@]}" --prepare-only ;;
  seed)    exec "$PY_BIN" "$ORCH" "${ARGS[@]}" --seed-archive ;;
  probe)
    echo "[e16] the resume-continuity probe spends TWO small model calls." >&2
    exec "$PY_BIN" "$ORCH" "${ARGS[@]}" --probe-session
    ;;
  status)
    exec "$PY_BIN" -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])),indent=2))" \
      "${RUN_DIR}/summary.json"
    ;;
  watch)
    # THE LIVE SIGNAL. The orchestrator is blocked in the player's subprocess
    # for the whole rollout, so before progress.jsonl existed the only way to
    # tell a playing run from a wedged one was to stat the archive. Watch
    # `idle_s`: it is seconds since the harness last wrote a turn, and a run
    # sitting in a harness relaunch shows a climbing idle_s while every other
    # column holds still.
    exec tail -n +1 -F "${RUN_DIR}/progress.jsonl"
    ;;
  reconcile) exec "$PY_BIN" "$ORCH" "${ARGS[@]}" --reconcile ;;
  run|resume)
    # THE HANG GATE, FIRST. Three of E16's known failure modes do not error --
    # they hang, and a hang on an uncapped player behind a $385 ceiling is the
    # run. The preflight exercises all three against a stub binary for nothing,
    # so there is no argument for skipping it before a real launch.
    if [ "${E16_SKIP_PREFLIGHT:-0}" != "1" ]; then
      if ! "$PY_BIN" "$PREFLIGHT"; then
        echo "run_e16: PREFLIGHT FAILED -- refusing to launch." >&2
        echo "  Every check above runs with no inference; a failure means one" >&2
        echo "  of the three silent-hang classes is live in this checkout." >&2
        echo "  (E16_SKIP_PREFLIGHT=1 overrides, knowing that.)" >&2
        exit 4
      fi
    fi
    # REFUSE TO SPEND THE BUDGET ON AN UNVERIFIED ASSUMPTION. The whole point
    # of the LLM orchestrator is that its conversation accumulates across
    # rounds; if `--resume` does not actually carry history under `--print`,
    # the run is N independent calls wearing a session's name and its headline
    # claim is false. `probe` settles it for two sentences of inference. The
    # scripted arm needs no session and is exempt.
    # matched_restart has NO orchestrator session by construction (the null arm
    # is not "an LM told to sit still"), so the resume probe does not apply.
    if [ "${E16_SELECTOR:-llm}" = "llm" ] \
       && [ "${E16_EXP_ARM:-go_explore}" != "matched_restart" ] \
       && [ "${E16_SKIP_PROBE:-0}" != "1" ]; then
      VERIFIED="$("$PY_BIN" - "$RUN_DIR" <<'PYV'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "provenance.json"
try:
    v = json.loads(p.read_text())["orchestrator_session"]["session_resume_verified"]
except Exception:
    v = None
print("yes" if v else "no")
PYV
)"
      if [ "$VERIFIED" != "yes" ]; then
        echo "run_e16: session resume is NOT verified for ${RUN_DIR}." >&2
        echo "  Run:  $0 ${RUN_DIR} probe" >&2
        echo "  (or E16_SKIP_PROBE=1 to proceed knowing the orchestrator's" >&2
        echo "   continuity is an assumption, which the run will record.)" >&2
        exit 3
      fi
    fi
    if [ "$CMD" = "resume" ]; then
      [ -d "${RUN_DIR}/archive" ] || {
        echo "run_e16: ${RUN_DIR} has no archive/ -- there is nothing to resume." >&2
        echo "  Use '$0 ${RUN_DIR} run' to start it." >&2
        exit 2
      }
      ARGS+=(--resume)
    fi
    exec "$PY_BIN" "$ORCH" "${ARGS[@]}"
    ;;
  *)
    echo "run_e16: unknown subcommand '${CMD}'" >&2
    exit 2
    ;;
esac
