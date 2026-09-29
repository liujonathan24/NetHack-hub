#!/usr/bin/env python3
"""TASK 4, step 1: build a checkpoint whose `prefix.jsonl` is NOT empty.

The builder's report says prefix continuity is "text replay only" and concluded
H4 is untestable. "Text replay only" is a true statement about the MECHANISM
(no scaffold can be handed a prior conversation; both player CLIs launch with
sessions off). It is NOT a statement about whether the replayed text reaches
the model, and those are different questions with different answers.

This writes the checkpoint the test needs, through the PRODUCTION write path:
the transcript is put on the env under `CONVERSATION_PREFIX_ATTR`, exactly
where `nethack.py:_append_conversation_prefix` puts it after every turn, and
`checkpoint_save` picks it up from there with no argument passed. So the write
side is exercised, not stubbed.

The transcript's content is synthesized rather than harvested from a real
player, and that is deliberate: it carries a CANARY string that appears nowhere
else in the harness, the wiki, the ledger or the directive, so a hit in the
served bytes cannot be anything else. What is under test is the READ side.
"""
import json
import sys
from pathlib import Path

REPO = Path("/root/nld/zombie-fix")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "environments" / "nethack"))
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))

from nethack_core.env import NetHackCoreEnv
from nethack_harness.checkpoints import (
    CONVERSATION_PREFIX_ATTR, PREFIX_JSONL, checkpoint_save, checkpoint_meta,
)
import e16_orchestrator as E

CANARY = "QUILLFEATHER-7731"

#: Shaped exactly like `_append_conversation_prefix` writes them: alternating
#: assistant (with tool_calls) and user (the served observation), each already
#: whitespace-collapsed and truncated.
PREFIX = [
    {"role": "assistant",
     "content": f"Plan {CANARY}: the east corridor is a dead end, I checked it "
                f"twice. Going west along the wall to the room with the "
                f"fountain, then down.",
     "tool_calls": [{"name": "np_move_to", "arguments": {"x": 12, "y": 9}}]},
    {"role": "user",
     "content": "You move west. There is a fountain here. The corridor east is "
                "unlit."},
    {"role": "assistant",
     "content": f"Confirmed. Sticking to plan {CANARY}: do not re-enter the "
                f"east corridor, it cost me eleven turns last time.",
     "tool_calls": [{"name": "np_search", "arguments": {"turns": 5}}]},
    {"role": "user",
     "content": "You find nothing. Your HP is 14(14). You feel hungry."},
]


def main(archive: Path) -> int:
    archive.mkdir(parents=True, exist_ok=True)
    env = NetHackCoreEnv(task_name="NetHackChallenge-v0",
                         max_episode_steps=100_000)
    env.seed(core=1, disp=1)
    env.reset(character="Val-hum-neu-fem")
    env.step(13)
    for _ in range(3):
        env.step(ord("s"))
    env.step(27)

    # THE PRODUCTION WRITE PATH: the attribute nethack.py maintains, read by
    # checkpoint_save with no `prefix=` argument passed.
    setattr(env, CONVERSATION_PREFIX_ATTR, PREFIX)
    target = archive / "c2"
    meta = checkpoint_save(env, target, name="fountain room",
                           note="saved mid-plan, to test prefix replay",
                           created_by="save")

    body = (target / PREFIX_JSONL).read_bytes()
    rendered = E.render_prefix(target)
    out = {
        "checkpoint": str(target),
        "prefix_jsonl_bytes": len(body),
        "prefix_jsonl_records": len([l for l in body.decode().splitlines() if l.strip()]),
        "canary": CANARY,
        "render_prefix_chars": len(rendered),
        "render_prefix_contains_canary": CANARY in rendered,
        "render_prefix_text": rendered,
        "meta_dlvl": meta.get("dlvl"), "meta_hp": meta.get("hp"),
    }
    print(json.dumps(out, indent=2))
    ok = (out["prefix_jsonl_bytes"] > 0
          and out["render_prefix_contains_canary"])
    if not ok:
        print("FAILED: the write side did not produce a replayable prefix",
              file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
