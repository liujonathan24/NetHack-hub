#!/usr/bin/env bash
# E10 orchestrator: one Prime Agent process, run BETWEEN cells, that reads the
# previous round's TRAINING traces and edits the shared continual-harness store
# the next round's players will boot with.
#
#   e10_orchestrate.sh <train_dir> <ch_dir> <round_out>
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

TRAIN_DIR="${1:?usage: e10_orchestrate.sh <train_dir> <ch_dir> <round_out>}"
CH="${2:?}"
ROUND_OUT="${3:?}"
MODEL="${ORCH_MODEL:-z-ai/glm-5.2}"

[ -d "$TRAIN_DIR" ] || { echo "e10_orchestrate: no such train dir: $TRAIN_DIR" >&2; exit 2; }
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

read -r -d '' PROMPT <<PROMPT_EOF
You are the E10 orchestrator for a NetHack agent experiment. You are not playing
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
     (`refine.run(...)` and the `rlm(...)` subagent call ARE async; the harness
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

echo "[orch ] $(date -u +%H:%M:%S) reading $TRAIN_DIR -> store $CH"
PRIME_AGENT_CODING_AGENT_DIR="$ORCH_DIR" \
PRIME_AGENT_KERNEL_VENV="${PRIME_AGENT_KERNEL_VENV:-$HOME/.prime/agent/kernel-venv}" \
  prime-agent --print --model "$MODEL" -- "$PROMPT" \
  > "$ROUND_OUT/orchestrator.stdout.txt" 2> "$ROUND_OUT/orchestrator.stderr.txt"
rc=$?
echo "[orch ] $(date -u +%H:%M:%S) rc=$rc"

if [ ! -e "$CH/harness_state.json" ]; then
  echo "[orch ] WARNING: no harness_state.json in $CH -- the orchestrator wrote" >&2
  echo "        nothing, so the next round is identical to this one. Check" >&2
  echo "        $ROUND_OUT/orchestrator.stdout.txt before spending a cell on it." >&2
fi
exit "$rc"
