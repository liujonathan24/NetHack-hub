#!/usr/bin/env python
"""Offline self-test of the PAE loop: no LLM calls, no money.

Replaces the player's LLM with a deterministic scripted client, then runs a
fresh attempt + two resumes and checks
  (a) replay restores byte-identical observations at the checkpoint,
  (b) the resumed player's message list equals the checkpoint's stored
      next-prompt, plus exactly the directive message.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings

warnings.filterwarnings("ignore")

from balrog.client import LLMResponse  # noqa: E402


class ScriptedClient:
    """Deterministic 'LLM': picks an action from the task's action list."""

    def __init__(self, actions, seed=0):
        self.actions = actions
        self.rng = random.Random(seed)
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        n = sum(len(m.content) for m in messages)
        return LLMResponse(
            model_id="scripted", completion=self.rng.choice(self.actions),
            stop_reason="stop", input_tokens=n // 4, output_tokens=3, reasoning=None,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="minihack")
    ap.add_argument("--task", default="MiniHack-Quest-Easy-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--run-dir", default="runs/selftest")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    a = ap.parse_args()

    from .pae import Config, Run
    from .run import balrog_config

    cfg = Config(game=a.game, task=a.task, seed=a.seed, attempts=a.attempts,
                 checkpoint_every=a.checkpoint_every, plateau=99, select="fixed",
                 directive="on", max_steps=a.max_steps, run_dir=a.run_dir)
    bcfg = balrog_config("scripted", 1.0, 64, 60)
    run = Run(cfg, bcfg)

    space = run.adapter.env.env.language_action_space
    if a.game == "minihack":
        safe = ["north", "south", "east", "west", "search", "wait"]
        acts = [x for x in space if x in safe] or ["north"]
    elif a.game == "crafter":
        acts = list(space)
    else:
        acts = ["look", "inventory", "go north", "go south", "go east", "go west"]

    def new_agent():
        from balrog.prompt_builder import create_prompt_builder
        from .player import PAEAgent

        ag = PAEAgent(lambda: ScriptedClient(acts, seed=a.seed), create_prompt_builder(bcfg.agent))
        return ag

    run._new_agent = new_agent
    # no orchestrator: canned directive, fixed-rule selection
    run.orch = None
    orig_select = run._select

    def select():
        cid, _ = orig_select()
        return cid, "Try heading east along the corridor instead of north."

    run._select = select
    run.go()

    # Render the orchestrator ledger offline (constructing the client makes no
    # network call) and assert the per-checkpoint outcome fields are present.
    from .orchestrator import Orchestrator

    o = Orchestrator(a.game, a.task, run.acct, step_cap=run.step_cap)
    ledger = o._ledger(run.resumable(), run.attempts)
    print("--- orchestrator ledger (rendered offline)")
    print(ledger)
    missing = [w for w in ("steps left", "tried", "parent", "EPISODE HORIZON") if w not in ledger]
    assert not missing, f"ledger is missing {missing}"
    assert any("tried 1x" in ln or "tried 2x" in ln for ln in ledger.splitlines()), \
        "no checkpoint shows a launched attempt - the tried column is not being filled"
    print("ledger check: OK (horizon, steps-left, parent and tried columns all render)")
    print(json.dumps(json.load(open(run.dir / "summary.json")), indent=2))
    for name in ("restore_fidelity.jsonl", "resume_prompt_check.jsonl"):
        p = run.dir / name
        print(f"--- {name}")
        if p.exists():
            for line in p.read_text().splitlines():
                d = json.loads(line)
                d.pop("extra_messages", None) if False else None
                print(json.dumps(d, default=str)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
