#!/usr/bin/env bash
# E16 Go-Explore on one seed -- the run launcher.
#
#   run_e16.sh <RUN_DIR> [SUBCOMMAND]
#
# Subcommands:
#   prepare   create the run tree, copy the wiki in, write provenance.json.
#             NO model calls. Safe to run any number of times.
#   seed      write checkpoint c1 (a fresh game at the dungeon entrance on the
#             run's seed) through the normal checkpoint path. NO model calls.
#   probe     the two-call `--resume` continuity check. THE ONLY subcommand
#             that spends inference, and it spends about two sentences of it.
#             Writes `session_resume_verified` into provenance.json.
#   run       the loop: orchestrator round -> player attempt -> ingest, until
#             budget / stall / milestone. SPENDS THE RUN'S BUDGET.
#   status    print the current summary.json.
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
#   E16_NO_DIRECTIVE=1  the control arm: same archive, same selection, no
#                       instructions to players.
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

ARGS=(
  "$RUN_DIR"
  --wiki-src "$WIKI_SRC"
  --tier "${E16_TIER:-e16_gewiki}"
  --arm "${E16_ARM:-prime_agent}"
  --selector "${E16_SELECTOR:-llm}"
  --game-seed "${E16_SEED:-1}"
  --rng-seed "${E16_RNG_SEED:-20260828}"
  --budget "${E16_BUDGET:-385}"
  --min-headroom "${E16_MIN_HEADROOM:-5}"
  --max-attempts "${E16_MAX_ATTEMPTS:-200}"
  --stall-attempts "${E16_STALL:-8}"
)
[ -n "${ORCH_MODEL:-}" ] && ARGS+=(--orch-model "$ORCH_MODEL")
[ -n "${E16_ORCH_AGENT_DIR:-}" ] && ARGS+=(--orch-agent-dir "$E16_ORCH_AGENT_DIR")
[ "${E16_NO_DIRECTIVE:-0}" = "1" ] && ARGS+=(--no-directive)

case "$CMD" in
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
  run)
    # REFUSE TO SPEND THE BUDGET ON AN UNVERIFIED ASSUMPTION. The whole point
    # of the LLM orchestrator is that its conversation accumulates across
    # rounds; if `--resume` does not actually carry history under `--print`,
    # the run is N independent calls wearing a session's name and its headline
    # claim is false. `probe` settles it for two sentences of inference. The
    # scripted arm needs no session and is exempt.
    if [ "${E16_SELECTOR:-llm}" = "llm" ] && [ "${E16_SKIP_PROBE:-0}" != "1" ]; then
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
    exec "$PY_BIN" "$ORCH" "${ARGS[@]}"
    ;;
  *)
    echo "run_e16: unknown subcommand '${CMD}'" >&2
    exit 2
    ;;
esac
