#!/usr/bin/env bash
# E13 orchestrator: one Prime Agent process, run BETWEEN cells, that reads the
# previous round's TRAINING traces and edits the shared continual-harness store
# the next round's players will boot with.
#
#   e13_orchestrate.sh <train_dir> <ch_dir> <round_out>
#
# Why a fresh `--print` process per round rather than a resident session:
#   * every cell is preceded by `pkill -9 -f prime-agent` (the daemon wedge
#     recipe), which would kill a resident orchestrator;
#   * nothing needs to survive in its head -- the store IS the state;
#   * a fresh process re-reads the store, so it always sees what it wrote.
#
# It runs UNSANDBOXED (it has to read the run directory, which the players'
# sandbox deliberately hides), so the prompt below is explicit about the one
# thing it may write.
set -uo pipefail

TRAIN_DIR="${1:?usage: e13_orchestrate.sh <train_dir> <ch_dir> <round_out>}"
CH="${2:?}"
ROUND_OUT="${3:?}"
# The reflection instructions, as a FILE. This is the knob that distinguishes
# one continual experiment from another: same base surface, same seeds, same
# code -- different instructions to the orchestrator. Its sha is pinned into
# every cell of the run, so two experiments are told apart from their artifacts.
PROMPT_FILE="${4:-}"
MODEL="${ORCH_MODEL:-z-ai/glm-5.2}"
# PIN THE PROVIDER. `--model z-ai/glm-5.2` alone is a model *pattern*: it matched
# openrouter's catalog entry first and died with "No API key found for
# openrouter", even though settings.json names prime-inference as the default.
# Credentials for prime-inference come from ~/.prime/config.json, which is
# outside the agent dir, so the private ORCH_DIR does not lose them.
PROVIDER="${ORCH_PROVIDER:-prime-inference}"

[ -d "$TRAIN_DIR" ] || { echo "e13_orchestrate: no such train dir: $TRAIN_DIR" >&2; exit 2; }
mkdir -p "$ROUND_OUT" "$CH"

# A private config directory, so the orchestrator never edits the operator's
# `~/.prime/agent` -- except that its GLOBAL harness directory is a symlink to
# the shared store, which is the whole point. `getGlobalHarnessStateDir()` is
# `join(agentDir, "harness")`, so this one link is all it takes.
ORCH_DIR="$ROUND_OUT/orch-agent"
mkdir -p "$ORCH_DIR"
for f in settings.json auth.json models.json; do
  [ -e "$HOME/.prime/agent/$f" ] && cp "$HOME/.prime/agent/$f" "$ORCH_DIR/$f"
done
rm -rf "$ORCH_DIR/harness"
ln -sfn "$CH" "$ORCH_DIR/harness"

# NOTE: this heredoc is UNQUOTED on purpose, so \${TRAIN_DIR} and \${ROUND_OUT}
# interpolate. That also makes backticks command substitutions -- bash executed
# `refine.run(...)` as a command the first time this ran. Keep the prompt free
# of backticks and of $ followed by anything that is not one of those two vars.
read -r -d '' PROMPT <<PROMPT_EOF
You are the E13 orchestrator for a NetHack agent experiment. You are not playing
the game. Your job is to read how previous games went and to leave better notes
for the next ones.

INPUT (read-only): ${TRAIN_DIR}
  turns/*.ndjson  one line per turn per rollout: the observation the player saw,
                  the skill it called, and the result.
  traces.jsonl    one JSON object per rollout: metrics (max_dlvl_reached, died,
                  skill_calls) and the full model call list.
Use your IPython kernel to parse these yourself. Look for what repeats ACROSS
rollouts: skills called with arguments that always fail, loops the player never
escapes, deaths that share a cause, budget spent on probing rather than progress.
A single unlucky game is not evidence.

OUTPUT (the only things you may write):
  1. The global continual-harness store, through its Python API. Every call needs
     global_=True, because only global entries survive into another session:
        rlm.harness.create_memory(title=..., content=..., global_=True)
        rlm.harness.update_memory(id=..., content=..., global_=True)
        rlm.harness.delete_memory(id=..., global_=True)
     and the same create/update/delete trio for skill, prompt_note and subagent.
     A skill entry is validated and REJECTED without a Python reference; the
     shape that passes is
        rlm.harness.create_skill(title=..., content=...,
            reference={"type": "python", "import": "nethack",
                       "call_pattern": "await nethack.explore_and_descend()"},
            arguments={}, global_=True)
     A memory or prompt_note takes no reference. Verified round-trip, so a
     ValueError here means your call shape is wrong, not that the store is.
     Read what is already there first: rlm.get_harness_state(global_=True)
     These are SYNCHRONOUS methods on a local JSON store -- do NOT await them.
     ('refine.run(...)' and the 'rlm(...)' subagent call ARE async; the harness
     store is not.)
  2. A rationale file at ${ROUND_OUT}/orchestrator_rationale.json, a JSON list of
     {"action","kind","id","title","evidence","expected_effect"} -- one object per
     edit you made, where "evidence" cites the rollouts and turn numbers that
     justify it.

HARD BUDGET, because of how the store is rendered into the player's system
prompt: at most 6 entries PER KIND are shown, each truncated to 180 characters,
plus the 5 most recent refinement events. Anything past that is invisible to the
player and is wasted. So:
  * keep at most 6 memories and at most 6 skill entries in total;
  * write each entry's content to be USEFUL AT 180 CHARACTERS -- concrete and
    imperative ("call X with Y before Z", not "consider being careful");
  * to add an entry once you are at the cap you must DELETE one. Delete the entry
    the evidence supports least, and say so in the rationale.
  * delete any existing entry this round's traces CONTRADICT, even if you wrote it
    yourself last round. An entry that made things worse is the most valuable
    thing you can find.

Do not edit the repository, the outputs directory, the game environment, or any
skill package. Do not run the game. Do not launch evaluations. When you are done,
print a one-paragraph summary of what you changed and why.
PROMPT_EOF

# The prompt is the whole instrument; a mangled one silently produces a useless
# round. (First run: the unquoted heredoc executed the backticked spans and left
# PROMPT truncated.) Fail before spending a model call on it.
case "$PROMPT" in
  *"HARD BUDGET"*) prompt_ok=1 ;;
  *) prompt_ok=0 ;;
esac
case "$PROMPT" in
  *"orchestrator_rationale.json"*) : ;;
  *) prompt_ok=0 ;;
esac
case "$prompt_ok" in
  1) : ;;
  *)
    echo "e13_orchestrate: PROMPT did not assemble (${#PROMPT} chars) -- refusing" >&2
    echo "  to run a round on a truncated prompt." >&2
    exit 3
    ;;
esac

# Per-experiment instructions, appended AFTER the invariant contract above so a
# variant cannot quietly drop the output format, the budget, or the read-only
# rule -- only add to them. That keeps runs comparable: what differs between
# experiments is the reflection guidance, not the protocol.
if [ -n "$PROMPT_FILE" ] && [ -f "$PROMPT_FILE" ]; then
  PROMPT="${PROMPT}

=== EXPERIMENT-SPECIFIC REFLECTION INSTRUCTIONS ($(basename "$PROMPT_FILE")) ===
$(cat "$PROMPT_FILE")"
  echo "[orch ] reflection prompt: $PROMPT_FILE ($(sha256sum "$PROMPT_FILE" | cut -c1-16))"
fi

echo "[orch ] $(date -u +%H:%M:%S) reading $TRAIN_DIR -> store $CH (provider=$PROVIDER model=$MODEL)"
PRIME_AGENT_CODING_AGENT_DIR="$ORCH_DIR" \
PRIME_AGENT_KERNEL_VENV="${PRIME_AGENT_KERNEL_VENV:-$HOME/.prime/agent/kernel-venv}" \
  prime-agent --print --provider "$PROVIDER" --model "$MODEL" -- "$PROMPT" \
  > "$ROUND_OUT/orchestrator.stdout.txt" 2> "$ROUND_OUT/orchestrator.stderr.txt"
rc=$?
echo "[orch ] $(date -u +%H:%M:%S) rc=$rc"

if [ ! -e "$CH/harness_state.json" ]; then
  echo "[orch ] WARNING: no harness_state.json in $CH -- the orchestrator wrote" >&2
  echo "        nothing, so the next round is identical to this one. Check" >&2
  echo "        $ROUND_OUT/orchestrator.stdout.txt before spending a cell on it." >&2
fi
exit "$rc"
