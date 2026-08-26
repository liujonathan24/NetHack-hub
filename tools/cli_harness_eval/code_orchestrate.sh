#!/usr/bin/env bash
# CODE orchestrator: one Prime Agent process, run BETWEEN rounds, that reads how
# the round's games went and EDITS the agent's netplay code accordingly. It is
# the code counterpart of e13_orchestrate.sh -- same scaffold, but the thing it
# writes is the netplay git repo, not the harness store.
#
#   code_orchestrate.sh <train_dir> <netplay_canonical> <out_root> <round_n> [prompt_file]
#
# Why a fresh --print process (mirrors e13_orchestrate.sh):
#   * every cell is preceded by a daemon reset that would kill a resident one;
#   * nothing needs to survive in its head -- the git repo IS the state;
#   * a fresh process re-reads the repo and the traces, so it always sees what
#     it (and the players) wrote.
# It runs UNSANDBOXED: it must read the run directory (which the players' sandbox
# hides) and commit into the canonical repo. The prompt is explicit about the
# only two things it may write: the netplay code, and its rationale file.
set -uo pipefail

TRAIN_DIR="${1:?usage: code_orchestrate.sh <train_dir> <netplay_canonical> <out_root> <round_n> [prompt_file]}"
CANON="${2:?}"
OUT_ROOT="${3:?}"
ROUND_N="${4:?}"
PROMPT_FILE="${5:-}"
MODEL="${ORCH_MODEL:-z-ai/glm-5.2}"
PROVIDER="${ORCH_PROVIDER:-prime-inference}"
ROUND_OUT="$OUT_ROOT/round${ROUND_N}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[ -d "$TRAIN_DIR" ]   || { echo "code_orchestrate: no such train dir: $TRAIN_DIR" >&2; exit 2; }
[ -d "$CANON/.git" ]  || { echo "code_orchestrate: $CANON is not a git repo (seed it first)" >&2; exit 2; }
mkdir -p "$ROUND_OUT"

# A private config dir so the orchestrator never edits the operator's
# ~/.prime/agent. No harness symlink here (unlike the memory orchestrator): the
# code orchestrator's state is the git repo, not the store.
ORCH_DIR="$ROUND_OUT/code-orch-agent"
mkdir -p "$ORCH_DIR"
for f in settings.json auth.json models.json; do
  [ -e "$HOME/.prime/agent/$f" ] && cp "$HOME/.prime/agent/$f" "$ORCH_DIR/$f"
done

GATE="$REPO/tools/cli_harness_eval/netplay_gate.py"
FROZEN_REF="$REPO/harnesses/nethack-prime-agent/nethack_prime_agent/skill/src/netplay"

# UNQUOTED heredoc on purpose (like e13_orchestrate.sh): ${...} interpolate.
# Keep the prompt free of backticks and of $ before anything but our vars.
read -r -d '' PROMPT <<PROMPT_EOF
You are the code orchestrator for a NetHack agent experiment. You are NOT
playing the game. Your job is to read how this round's games went and to make
the agent's own policy code better for the next round -- and to UNDO changes
that made it worse.

THE CODE YOU EDIT (read-write): ${CANON}
  A git repo of the agent's netplay policies -- explore.py, descend.py,
  fight.py, survive.py compose a frozen primitive floor (_base.py) into the
  policies the player calls (netplay.explore(), netplay.descend(), etc.).
  _base.py, __init__.py, pyproject.toml and the nethack shim are FROZEN: do not
  edit them; the gate rejects a tree that does. Its history is the round
  lineage -- git -C ${CANON} log --oneline, and tags round-0, round-1, ...

THIS ROUND'S GAMES (read-only): ${TRAIN_DIR}
  traces.jsonl   one JSON object per rollout: seed, metrics (max_dlvl_reached,
                 died, scout/descent/reward, skill_calls) and the model calls.
  turns/*.ndjson one line per turn: the observation, the skill called, result.

THE PLAYERS' OWN EDITS (read-only clones): ${NETPLAY_WORK}/<rollout id>/
  Each rollout plays in its own git clone of the canonical repo and may edit
  its policies MID-GAME; the teardown commits whatever it changed (look for
  commits named "rollout <id>: gate=pass ..."). These are candidate
  improvements you MUST consider, not noise:
    for d in ${NETPLAY_WORK}/*/ : git -C <d> log --oneline -3
    git -C <d> diff <base>..HEAD -- <file>   (base = the round tag it cloned)
  Match clones to this round's rollouts via the ids in traces.jsonl; clones
  from earlier rounds persist and an unmerged good idea from any round is
  fair game. Judge each player edit by the SAME differential standard: did
  the seed that wrote it do better on the mechanism it touches (fewer
  wall-bumps, fewer stuck loops, cleaner descents)? Adopt the good ones into
  ${CANON} (re-apply with the edit skill, or cherry-pick the hunk by hand),
  reject the bad ones, and record every verdict in the rationale as
  {"action": "adopt-player"|"reject-player", ...} with the rollout id.

PRIOR ROUNDS (read-only, for the DIFFERENTIAL): ${OUT_ROOT}/round*/corpus__prime_agent/traces.jsonl
  Same shape, one directory per earlier round. This is how you tell whether a
  code change HELPED. Parse them all in your IPython kernel and build a table:
  per SEED, the metric (e.g. max_dlvl_reached) at each round. Line that up
  against what the code was at each round (git tag round-N marks it, and
  git -C ${CANON} diff round-\$((${ROUND_N}-1)) round-${ROUND_N} -- shows the
  last change). A change that lifted the seeds that exercise it is worth
  keeping; one that dropped them is worth reverting. A single unlucky game is
  not evidence -- look for the effect ACROSS the seeds a change should touch.

DECIDE, PER CHANGE (differential, then act):
  * If the last round's edit to a policy improved the seeds it affects: keep it,
    and consider extending the same idea.
  * If it hurt them: revert it (git -C ${CANON} checkout round-\$((${ROUND_N}-1)) -- <file>,
    or edit it back), and try a different fix.
  * If a policy is failing in a way no past change addressed (a loop, an early
    give-up, an avoidable death that repeats across seeds): edit it to fix that.
  Prefer small, surgical, evidence-backed edits over rewrites. Every edit must
  name the seeds and turns that justify it.

HOW TO EDIT AND VALIDATE (do this from your IPython kernel):
  1. Read the file first. Use the built-in edit skill for a surgical change:
        await edit(path="${CANON}/explore.py", old_str=..., new_str=...)
     (whole-file writes work too; edit keeps the diff small.)
  2. VALIDATE before committing -- the gate is the same one the players' teardown
     runs, and a tree that fails it is refused:
        import subprocess
        r = subprocess.run(["python", "${GATE}", "${CANON}",
                            "--frozen-reference", "${FROZEN_REF}", "--json"],
                           capture_output=True, text=True)
        print(r.stdout)   # must show "ok": true
     Fix anything it flags. Do NOT commit a tree that fails the gate.
  3. Commit and tag this round, so the lineage stays diffable and revertible:
        cd ${CANON}
        git add -A
        git commit -q -m "round-${ROUND_N}: <one line on what you changed and why>"
        git tag -f round-${ROUND_N}
     If after your analysis the right call is to change NOTHING, commit
     --allow-empty with that reasoning and still tag the round.

THE OTHER THING YOU WRITE: ${ROUND_OUT}/code_orchestrator_rationale.json
  A JSON list, one object per decision:
    {"action": "keep|revise|revert|new|noop", "file": "...",
     "seeds": [...], "evidence": "rollouts + turn numbers",
     "expected_effect": "...", "differential": "seed->metric before/after"}

Work entirely in your IPython kernel: parse the traces yourself, do the
differential, edit, gate, commit, write the rationale. Nothing else is yours to
touch.
PROMPT_EOF

# Same assembly guard as e13_orchestrate.sh: refuse a truncated prompt.
[ "${#PROMPT}" -gt 800 ] || { echo "code_orchestrate: PROMPT did not assemble (${#PROMPT} chars) -- refusing" >&2; exit 3; }

# Experiment-specific reflection guidance, appended AFTER the invariant contract
# so a variant can only ADD to the protocol, never drop the gate/commit/format.
if [ -n "$PROMPT_FILE" ] && [ -f "$PROMPT_FILE" ]; then
  PROMPT="${PROMPT}

=== EXPERIMENT-SPECIFIC CODE-REFLECTION INSTRUCTIONS ($(basename "$PROMPT_FILE")) ===
$(cat "$PROMPT_FILE")"
  echo "[corch] reflection prompt: $PROMPT_FILE ($(sha256sum "$PROMPT_FILE" | cut -c1-16))"
fi

BEFORE="$(git -C "$CANON" rev-parse --short HEAD)"
echo "[corch] $(date -u +%H:%M:%S) round ${ROUND_N}: reading $TRAIN_DIR -> code $CANON @ $BEFORE (provider=$PROVIDER model=$MODEL)"
PRIME_AGENT_CODING_AGENT_DIR="$ORCH_DIR" \
PRIME_AGENT_KERNEL_VENV="${PRIME_AGENT_KERNEL_VENV:-$HOME/.prime/agent/kernel-venv}" \
  "${PA38_BIN:-prime-agent}" --print --provider "$PROVIDER" --model "$MODEL" -- "$PROMPT" \
  > "$ROUND_OUT/code_orchestrator.stdout.txt" 2> "$ROUND_OUT/code_orchestrator.stderr.txt"
rc=$?
AFTER="$(git -C "$CANON" rev-parse --short HEAD)"
echo "[corch] $(date -u +%H:%M:%S) rc=$rc  code $BEFORE -> $AFTER"

# Post-check: the orchestrator must leave a valid, committed tree. If it left the
# gate failing or an uncommitted mess, say so loudly -- the next round reads this.
if [ "$(python3 "$GATE" "$CANON" --frozen-reference "$FROZEN_REF" --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["ok"])' 2>/dev/null)" != "True" ]; then
  echo "[corch] WARNING: canonical FAILS the gate after orchestration -- next round would boot broken code." >&2
fi
if [ -n "$(git -C "$CANON" status --porcelain)" ]; then
  echo "[corch] WARNING: orchestrator left canonical dirty (uncommitted changes). Committing a safety snapshot." >&2
  git -C "$CANON" add -A && git -C "$CANON" commit -q -m "round-${ROUND_N}: orchestrator left uncommitted changes (auto-snapshot)" && git -C "$CANON" tag -f "round-${ROUND_N}"
fi
if [ "$BEFORE" = "$AFTER" ] && [ ! -f "$ROUND_OUT/code_orchestrator_rationale.json" ]; then
  echo "[corch] WARNING: no code change and no rationale -- the orchestrator wrote nothing." >&2
  echo "        Check $ROUND_OUT/code_orchestrator.stdout.txt before spending a cell." >&2
fi
exit "$rc"
