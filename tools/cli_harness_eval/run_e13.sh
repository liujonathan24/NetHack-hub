#!/usr/bin/env bash
# E13: does past play make future play cheaper?
#
#   run_e13.sh              run this worktree's experiment
#   run_e13.sh --list       what exists on disk for it
#   RESUME=1 run_e13.sh     continue where it stopped
#
# ONE EXPERIMENT PER WORKTREE. Identity comes from configs/continual/
# experiment.toml, not from an argument, so anything that is a code or content
# edit -- the starting skill package, the env, the reflection prompt -- is just
# an edit in this worktree.
#
#   round r:  N corpus games (the experiment's seeds), players writing lessons
#             -> merge their private stores into canonical
#             -> orchestrate: read the round's traces, curate the merged store
#
# The held-out evaluation is NOT part of a round. When the experiment ends its
# final store is frozen into outputs/, and we evaluate that snapshot on seeds
# 0-4 ourselves with tools/cli_harness_eval/eval_frozen.sh.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The harness package is an editable install pointing at the MAIN checkout, so a
# worktree's harness changes are invisible without this (measured: the eval CLI
# rejects --harness.continual_harness_dir as "Extra inputs are not permitted",
# which reads like a bad flag rather than a stale package).
export PYTHONPATH="${REPO}/harnesses/nethack-prime-agent${PYTHONPATH:+:${PYTHONPATH}}"
export ENG="${ENG:-/root/NetHack-engine}"
export EVAL_BIN="${EVAL_BIN:-/root/NetHack-hub/.venv-cli-eval/bin/eval}"
PY_BIN="${PY_BIN:-$(dirname "${EVAL_BIN}")/python}"
SPEC="$REPO/configs/continual/experiment.toml"
[ -f "$SPEC" ] || { echo "run_e13: no experiment spec at $SPEC" >&2; exit 2; }

read_spec() { "$PY_BIN" - "$SPEC" "$1" <<'PYS'
import json, sys, tomllib
d = tomllib.load(open(sys.argv[1], "rb"))
v = d[sys.argv[2]]
print(json.dumps(v) if isinstance(v, (list, dict, bool)) else v)
PYS
}
EXP_ID="$(read_spec id)"; REPLICATE="$(read_spec replicate)"
ROUNDS="${ROUNDS:-$(read_spec rounds)}"
CORPUS_SEEDS="$(read_spec corpus_seeds)"
PLAYERS_EDIT="$(read_spec players_may_edit)"
PROMPT_NAME="$(read_spec prompt)"; TIER="$(read_spec tier)"
SPEC_SHA="$(sha256sum "$SPEC" | cut -c1-16)"
RUN="${EXP_ID}-r${REPLICATE}"

# Per-experiment install_dir. FIXED per experiment, not globally: the kernel venv
# is keyed on the Python-skill paths, and two worktrees sharing /tmp/vf-prime-agent
# would overwrite each other's skill package mid-run.
INSTALL_DIR="${INSTALL_DIR:-/tmp/vf-prime-agent-${RUN}}"
CH="${CH:-${INSTALL_DIR}/continual-harness}"

# Skills-as-code arm wiring. NETPLAY_CANONICAL is the ONE git repo the agent's
# code accumulates in -- the same fixed skills path the kernel imports from, so
# canonical IS the live tree. It is left unset for every other arm, which keeps
# the code-merge and code-freeze blocks below inert for them.
#
# MAX_CONCURRENT=1 is not a tuning knob here, it is correctness: the mutable
# arm's rollouts share this one tree and this one repo, so they MUST commit one
# at a time (see PrimeAgentHarness._materialise_netplay). Parallel per-rollout
# isolation is deferred; until it exists, this serialises.
case "$TIER" in
  continual-code)
    # Canonical is a SEPARATE git repo; each rollout edits a private clone bound
    # over the fixed skill path inside its sandbox, so rollouts run in PARALLEL.
    export NETPLAY_CANONICAL="${INSTALL_DIR}/netplay-canonical"
    export NETPLAY_WORK="${INSTALL_DIR}/netplay-work"
    # Seed canonical ONCE, serially, before any rollout -- a rollout cannot seed
    # it safely because concurrent clones would race to create it.
    if [ ! -d "$NETPLAY_CANONICAL/.git" ]; then
      SEED="$REPO/harnesses/nethack-prime-agent/nethack_prime_agent/skill/src/netplay"
      mkdir -p "$NETPLAY_CANONICAL"
      cp "$SEED"/*.py "$NETPLAY_CANONICAL/"
      ( cd "$NETPLAY_CANONICAL"
        git init -q -b netplay-canonical .
        printf '__pycache__/\n*.pyc\n' > .gitignore
        git config user.email netplay@localhost
        git config user.name 'netplay seed'
        git add -A && git commit -q -m 'round-0 seed' && git tag -f round-0 )
      echo "[e13  ] seeded netplay canonical at $NETPLAY_CANONICAL"
    fi
    echo "[e13  ] netplay canonical=$NETPLAY_CANONICAL (PARALLEL: private per-rollout clones)"
    ;;
esac
OUT_ROOT="${OUT_ROOT:-$REPO/outputs/e13/${RUN}}"
PROMPT_FILE="$REPO/configs/continual/${PROMPT_NAME}"
[ -f "$PROMPT_FILE" ] || { echo "run_e13: no reflection prompt at $PROMPT_FILE" >&2; exit 2; }
PROMPT_SHA="$(sha256sum "$PROMPT_FILE" | cut -c1-16)"
[ "$PLAYERS_EDIT" = "true" ] && MODE="copy-merge" || MODE="shared-ro"
N_SEEDS="$(printf '%s' "$CORPUS_SEEDS" | "$PY_BIN" -c 'import json,sys;print(len(json.load(sys.stdin)))')"

if [ "${1:-}" = "--list" ]; then
  echo "experiment : $RUN   (spec $SPEC_SHA, prompt $PROMPT_NAME $PROMPT_SHA)"
  echo "store      : $CH"
  echo "outputs    : $OUT_ROOT"
  if [ -d "$OUT_ROOT" ]; then
    for d in "$OUT_ROOT"/round*; do
      [ -d "$d" ] || continue
      n=$(ls -d "$d"/*__prime_agent 2>/dev/null | wc -l)
      echo "  $(basename "$d"): $n cell(s)$( [ -f "$d/merge_report.json" ] && echo ', merged' )"
    done
  else
    echo "  (nothing run yet)"
  fi
  [ -d "$OUT_ROOT/final" ] && echo "  FROZEN: $OUT_ROOT/final"
  exit 0
fi

echo "[e13  ] experiment=$RUN rounds=$ROUNDS seeds=$CORPUS_SEEDS mode=$MODE"
echo "[e13  ] spec=$SPEC_SHA prompt=$PROMPT_NAME($PROMPT_SHA) install_dir=$INSTALL_DIR"

mkdir -p "$CH" "$OUT_ROOT"
MANIFEST="$OUT_ROOT/run_manifest.json"
if [ -n "$(ls -A "$CH" 2>/dev/null)" ] || [ -f "$MANIFEST" ]; then
  if [ "${RESUME:-}" != "1" ]; then
    echo "run_e13: $RUN has already started (store or manifest present), so this" >&2
    echo "  run would NOT begin from the base set of skills. RESUME=1 to continue," >&2
    echo "  bump 'replicate' in the spec for an independent repeat, or clear" >&2
    echo "  $CH and $OUT_ROOT." >&2
    exit 3
  fi
  # Resuming with different instructions would mix two reflection regimes in one
  # store, and nothing downstream could tell.
  if [ -f "$MANIFEST" ]; then
    prev_spec="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["spec_sha256_16"])' "$MANIFEST")"
    prev_prompt="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["prompt_sha256_16"])' "$MANIFEST")"
    if [ "$prev_spec" != "$SPEC_SHA" ] || [ "$prev_prompt" != "$PROMPT_SHA" ]; then
      echo "run_e13: refusing to resume -- the spec or the reflection prompt changed" >&2
      echo "  since this run started (spec $prev_spec -> $SPEC_SHA, prompt" >&2
      echo "  $prev_prompt -> $PROMPT_SHA). Resuming would mix two reflection" >&2
      echo "  regimes in one store. Start a new replicate instead." >&2
      exit 3
    fi
  fi
fi

cat > "$MANIFEST" <<JSON
{
  "experiment": "${EXP_ID}", "replicate": ${REPLICATE}, "run": "${RUN}",
  "spec": "${SPEC}", "spec_sha256_16": "${SPEC_SHA}",
  "reflection_prompt": "${PROMPT_NAME}", "prompt_sha256_16": "${PROMPT_SHA}",
  "rounds": ${ROUNDS}, "corpus_seeds": ${CORPUS_SEEDS},
  "eval_seeds": $(read_spec eval_seeds),
  "players_may_edit": ${PLAYERS_EDIT}, "store_mode": "${MODE}",
  "tier": "${TIER}", "store": "${CH}", "install_dir": "${INSTALL_DIR}"
}
JSON

reset_daemon() {
  # Scoped by install_dir and serialised with flock, so a re-baseline can run
  # alongside an experiment. See tools/cli_harness_eval/reset_daemon.sh.
  "$REPO/tools/cli_harness_eval/reset_daemon.sh" "$INSTALL_DIR"
}

snapshot() { # <label> <destdir>
  local label="$1" dest="$2"; mkdir -p "$dest"
  if [ -e "$CH/harness_state.json" ]; then cp "$CH/harness_state.json" "$dest/harness_state.$label.json"
  else echo '{"note":"empty store"}' > "$dest/harness_state.$label.json"; fi
}

play_round() { # <round>
  local r="$1" out="$OUT_ROOT/round${r}/corpus__prime_agent"
  echo "[cell ] $(date -u +%H:%M:%S) -> $out (tier=$TIER mode=$MODE seeds=$CORPUS_SEEDS)"
  reset_daemon; mkdir -p "$out"; snapshot before "$out"
  env TOOL_TIER="$TIER" SEEDS="$CORPUS_SEEDS" \
      INSTALL_DIR="$INSTALL_DIR" \
      CONTINUAL_HARNESS="$CH" CONTINUAL_RUN_ID="$RUN" \
      CONTINUAL_PROMPT_SHA="$PROMPT_SHA" CONTINUAL_SPEC_SHA="$SPEC_SHA" \
      CONTINUAL_HARNESS_MODE="$MODE" \
      ${PLAYERS_EDIT:+CONTINUAL_SELF_EDIT="$PLAYERS_EDIT"} \
      "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$out" 200 "$N_SEEDS" \
    && echo "[done ] $(date -u +%H:%M:%S) OK  $out" \
    || echo "[FAIL ] $(date -u +%H:%M:%S) rc=$? $out"
  # Code write-back, in PARALLEL with the store write-back below and not
  # instead of it: the two merge different artifacts. The store keeps the prose
  # (memories, and the create_skill entries that ROUTE to netplay.<fn>); this
  # keeps the function bodies. Both run at every round boundary.
  #
  # NETPLAY_CANONICAL is unset for every arm that is not a code arm, so this is
  # inert for [base], [human] and [continual] -- no new command, no new file.
  if [ -n "${NETPLAY_CANONICAL:-}" ]; then
    "$PY_BIN" "$REPO/tools/cli_harness_eval/merge_netplay_code.py" \
      --canonical "$NETPLAY_CANONICAL" \
      --collect "${NETPLAY_WORK:-${INSTALL_DIR}/netplay-work}" \
      --frozen-reference "$REPO/harnesses/nethack-prime-agent/nethack_prime_agent/skill/src/netplay" \
      --round "round-${r}" \
      --report "$OUT_ROOT/round${r}/netplay_merge_report.json" | sed 's/^/[nmrg] /'
    # Clear the per-rollout clones so the next round starts clean.
    rm -rf "${NETPLAY_WORK:-${INSTALL_DIR}/netplay-work}"/* 2>/dev/null || true
  fi
  if [ "$MODE" = "copy-merge" ]; then
    # Single-threaded reconciliation of the private copies. Without this the
    # players' lessons stay in per-rollout directories and never compound.
    "$PY_BIN" "$REPO/tools/cli_harness_eval/merge_harness_stores.py" \
      --canonical "$CH" --run-dir "$out" --install-dir "$INSTALL_DIR" \
      --baseline "$out/harness_state.before.json" \
      --report "$OUT_ROOT/round${r}/merge_report.json" | sed 's/^/[merge] /'
  fi
  snapshot after "$out"
}

# MANDATORY PROTOCOL. No paid round starts until a mock play on this tier has
# read back clean. SKIP_PREFLIGHT=1 is for a rerun in the same session on an
# unchanged tree -- not for "I am fairly sure it is fine". Note the first
# preflight of a NEW experiment also pays the one-time kernel-venv build for its
# install_dir (~10 minutes, measured), which is exactly the cost you want to hit
# on a 3-call mock rather than on round 1.
if [ "${SKIP_PREFLIGHT:-}" != "1" ]; then
  echo "[pre  ] $(date -u +%H:%M:%S) preflight $TIER"
  INSTALL_DIR="$INSTALL_DIR" "$REPO/tools/cli_harness_eval/preflight_cell.sh" "$TIER" \
    || { echo "run_e13: preflight failed -- batch not launched." >&2; exit 7; }
fi

for r in $(seq 1 "$ROUNDS"); do
  echo "[round] $(date -u +%H:%M:%S) === $RUN round $r/$ROUNDS ==="
  if [ "${RESUME:-}" = "1" ] && [ -f "$OUT_ROOT/round${r}/corpus__prime_agent/traces.jsonl" ]; then
    echo "[round] round $r already has traces -- skipping (RESUME)"
    continue
  fi
  play_round "$r"
  reset_daemon   # the orchestrator is a prime-agent process too

  if [ "${NO_REFLECT:-}" = "1" ]; then
    echo "[round] NO_REFLECT=1 -- skipping both orchestrators (mechanical merges only)"
    continue
  fi
  # STORE orchestrator (memory harness): reflects on traces -> writes the store.
  # Runs for every continual arm; its store is what continual-code layers ON TOP
  # of the code channel (both accumulate, they are not exclusive).
  "$REPO/tools/cli_harness_eval/e13_orchestrate.sh" \
    "$OUT_ROOT/round${r}/corpus__prime_agent" "$CH" "$OUT_ROOT/round${r}" "$PROMPT_FILE" \
    || echo "[FAIL ] store orchestrator round $r rc=$?" >&2

  # CODE orchestrator: reflects on traces + the code lineage -> EDITS netplay.
  # Only for the code arm (NETPLAY_CANONICAL set). Runs AFTER the store one and
  # AFTER any player-edit merge, so it sees the round's final code and traces.
  if [ -n "${NETPLAY_CANONICAL:-}" ] && [ -d "${NETPLAY_CANONICAL}/.git" ]; then
    reset_daemon
    CODE_PROMPT="${CODE_PROMPT_FILE:-$REPO/configs/continual/code_default.md}"
    "$REPO/tools/cli_harness_eval/code_orchestrate.sh" \
      "$OUT_ROOT/round${r}/corpus__prime_agent" "$NETPLAY_CANONICAL" "$OUT_ROOT" "$r" \
      "$CODE_PROMPT" \
      || echo "[FAIL ] code orchestrator round $r rc=$?" >&2
  fi
done

# --- freeze ------------------------------------------------------------------
# Into the repo, not /tmp: the sandbox has no persistent volume, so a /tmp store
# is one rebuild away from gone -- and this snapshot is what the held-out
# evaluation is run against.
FINAL="$OUT_ROOT/final"; mkdir -p "$FINAL"
cp "$CH/harness_state.json" "$FINAL/harness_state.json" 2>/dev/null || echo '{}' > "$FINAL/harness_state.json"
cp "$SPEC" "$FINAL/experiment.toml"; cp "$PROMPT_FILE" "$FINAL/reflection_prompt.md"
# The code tree is frozen the same way and for the same reason: the held-out
# evaluation must run against a FIXED sha, not a moving tree. `netplay_commit`
# is recorded beside it so a cell is replayable from its own artifact, exactly
# as `tool_tier_commit` already makes the tool surface replayable.
if [ -n "${NETPLAY_CANONICAL:-}" ] && [ -d "$NETPLAY_CANONICAL/.git" ]; then
  cp -a "$NETPLAY_CANONICAL" "$FINAL/netplay"
  git -C "$NETPLAY_CANONICAL" rev-parse --short HEAD > "$FINAL/netplay_commit" 2>/dev/null || true
  echo "[freeze] netplay tree at $(cat "$FINAL/netplay_commit" 2>/dev/null || echo unknown)"
fi
# The document this arm actually served. A code arm serves SKILL.code.md, so
# copying SKILL.md unconditionally would freeze a doc the agent never saw --
# and it is the doc that tells the agent its code is editable at all.
case "$TIER" in
  continual-code*) _SKILL_SRC="SKILL.code.md" ;;
  *)               _SKILL_SRC="SKILL.md" ;;
esac
cp "$REPO/harnesses/nethack-prime-agent/nethack_prime_agent/skill/$_SKILL_SRC" \
   "$FINAL/SKILL.md" 2>/dev/null || true
"$PY_BIN" - "$FINAL" "$RUN" "$SPEC_SHA" "$PROMPT_SHA" "$TIER" <<'PYF'
import json, pathlib, sys, hashlib
final, run, spec_sha, prompt_sha, tier = pathlib.Path(sys.argv[1]), *sys.argv[2:]
state = json.loads((final / "harness_state.json").read_text())
counts = {k: len(v) for k, v in state.get("entries", {}).items()}
(final / "FROZEN.json").write_text(json.dumps({
    "run": run, "spec_sha256_16": spec_sha, "prompt_sha256_16": prompt_sha,
    # The tier is recorded so eval_frozen.sh does not have to guess it. Before
    # this, that script hardcoded `continual`, which would have evaluated a code
    # arm with the agent's code absent and reported the number as the arm's.
    "tier": tier,
    "netplay_commit": (final / "netplay_commit").read_text().strip()
                      if (final / "netplay_commit").exists() else None,
    "entry_counts": counts,
    "harness_state_sha256": hashlib.sha256((final / "harness_state.json").read_bytes()).hexdigest(),
    "note": "Evaluate with tools/cli_harness_eval/eval_frozen.sh; the store is "
            "mounted read-only so the evaluation cannot contaminate what it tests.",
}, indent=2))
print(f"[freeze] {run}: {counts}")
PYF
echo "[e13  ] $(date -u +%H:%M:%S) $RUN finished; frozen at $FINAL"
