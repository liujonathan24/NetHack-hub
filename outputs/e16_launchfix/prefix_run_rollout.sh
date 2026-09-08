#!/usr/bin/env bash
# TASK 4, step 2: ONE real player rollout resumed from a checkpoint whose
# prefix.jsonl is non-empty, launched through the PRODUCTION path.
#
# The prefix reaches the player exactly the way a real E16 attempt sends it:
# e16_orchestrator._launch_one appends render_prefix(ck) to `ledger_text`, that
# string goes into E16_ARGS, launch_cell.sh writes it as
# `taskset.env_args.ledger_text`, nethack.py puts it in `state["_resume_notice"]`
# and renders it into the FIRST observation. Nothing here is a special path for
# the test -- the only thing this script does that the orchestrator would not is
# cap the call budget so the check costs ~$0.25 instead of ~$35.
#
# ISOLATION (four experiments share this box; every line was written after an
# incident): the repo's own eval venv, a private skill-install root, a private
# ~/.prime/agent copy bound over the shared one inside the rollout's sandbox,
# and an output directory in the scratchpad rather than the shared worktree.
set -euo pipefail

REPO=/root/nld/zombie-fix
WORK="${WORK:-/tmp/claude-0/-root/3382b9f1-b1ec-4e18-9c3e-4585c4167d08/scratchpad/e16pfx}"
MAX_CALLS="${MAX_CALLS:-12}"
OUTDIR="$WORK/rollout"

export ENG=/root/NetHack-engine
export PYTHONPATH="$REPO/tools/pycompat:$ENG:$REPO:$REPO/environments/nethack:$REPO/tools/cli_harness_eval"
export EVAL_BIN="$REPO/.venv-cli-eval/bin/eval"
export INSTALL_DIR="/tmp/vf-prime-agent-e16pfx"
export NETHACK_PA_AGENT_BIND_SRC="/root/nld/.prime-agent-e16-sim"
key=$(python3 -c "import json;print(json.load(open('/root/.prime/config.json'))['api_key'])")
export PI_API_KEY="$key" PRIME_API_KEY="$key"
unset key

export TOOL_TIER=e16_gewiki
export SEEDS='[1]'
export TIER_SHORT_BUDGET=1
export STALL_WATCHDOG=1

mkdir -p "$OUTDIR"
# Build E16_ARGS with the SAME code the orchestrator uses, so ledger_text is
# assembled by render_ledger + render_lessons + render_prefix rather than by
# this script's idea of what those produce.
E16_ARGS=$("$REPO/.venv-cli-eval/bin/python" - "$WORK" <<'PYE'
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
sys.path.insert(0, "/root/nld/zombie-fix/tools/cli_harness_eval")
sys.path.insert(0, "/root/nld/zombie-fix/environments/nethack")
sys.path.insert(0, "/root/nld/zombie-fix")
import e16_orchestrator as E
archive = work / "archive"
ck = archive / "c2"
rows = E.ledger_rows(archive)
cfg = E.OrchestratorConfig(run_dir=work)
text = E.render_ledger(rows, cfg, current_id="2", attempts=[])
for extra in (E.render_lessons(ck, 8), E.render_prefix(ck)):
    if extra:
        text += "\n\n" + extra
(work / "ledger_text.txt").write_text(text)
print(json.dumps({
    "resume_checkpoint": str(ck),
    "checkpoint_archive": str(archive),
    "wiki_dir": "/root/nld/e15-wiki/configs/continual/wiki",
    "ledger_text": text,
    "fidelity_log": str(work / "restore_fidelity.jsonl"),
    "directive": "Follow the plan quoted from the earlier session.",
}))
PYE
)
export E16_ARGS
printf '%s' "$E16_ARGS" > "$WORK/e16_args.json"

echo "[pfx] ledger_text chars: $(wc -c < "$WORK/ledger_text.txt")"
echo "[pfx] max_calls=$MAX_CALLS out=$OUTDIR"
cd "$REPO"
exec tools/cli_harness_eval/launch_cell.sh prime_agent "$OUTDIR" "$MAX_CALLS" 1 \
  --harness.path-prepend "/usr/bin:/root/.local/bin"
