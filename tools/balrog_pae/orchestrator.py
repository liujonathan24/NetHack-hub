"""The PAE orchestrator: a separate GLM-5.2 chat that picks a checkpoint and
writes one short directive.

It sees only a ledger (checkpoint id, attempt, step, BALROG progression, the
dense exploration proxy, one line of state) and the attempt history.  It never
sees the player's conversation.
"""
from __future__ import annotations

import json
import re

from .prime_client import MODEL_ID, make_openai_client

SYSTEM = """You are the explorer in a checkpointed-exploration loop for the game {game} ({task}).

A player agent plays the game one action at a time. Whenever it stops (death, step
cap, or the task ends) you choose which SAVED CHECKPOINT it should resume from and
write ONE short directive telling it what to do differently. The player resumes
with its own conversation history exactly as it was at that checkpoint, plus your
directive as one extra message; it cannot see this conversation or the ledger.

Pick checkpoints that are close to the frontier of what has been achieved but
early enough to still change the outcome; do not always pick the deepest one if the
attempts from there keep failing the same way.

Reply with ONLY a JSON object, no prose, no code fence:
{{"checkpoint": "<checkpoint id>", "directive": "<at most 3 sentences of concrete advice>"}}"""


class Orchestrator:
    def __init__(self, game, task, accountant, model_id=MODEL_ID, temperature=1.0, max_tokens=1024):
        self.client = make_openai_client()
        self.game, self.task = game, task
        self.accountant = accountant
        self.model_id, self.temperature, self.max_tokens = model_id, temperature, max_tokens
        self.rounds: list[dict] = []

    def _ledger(self, archive, attempts) -> str:
        lines = ["CHECKPOINT LEDGER (id | from attempt | step | balrog progression | cells/achievements seen | state)"]
        for c in archive:
            lines.append(
                f"{c['id']} | a{c['attempt']} | step {c['step']} | {c['progression']:.3f} | "
                f"{c['aux_progress']:.0f} | {c['summary']}"
            )
        lines.append("")
        lines.append("ATTEMPT HISTORY (attempt | resumed from | steps played | final progression | outcome | directive given)")
        for a in attempts:
            lines.append(
                f"a{a['attempt']} | {a['from_checkpoint'] or 'fresh episode'} | {a['steps_played']} | "
                f"{a['progression']:.3f} | {a['outcome']} | {(a['directive'] or '-')[:160]}"
            )
        return "\n".join(lines)

    def decide(self, archive, attempts, want_directive: bool, fixed_checkpoint: str | None = None) -> dict:
        ledger = self._ledger(archive, attempts)
        if fixed_checkpoint is not None:
            # ablation arm: the branch point is fixed by rule, the orchestrator
            # only writes the directive.
            ask = (
                f"The player will resume from checkpoint {fixed_checkpoint} (fixed by rule; you do not choose it). "
                f"Write the directive. Set \"checkpoint\" to {fixed_checkpoint}."
            )
        elif want_directive:
            ask = "Choose the checkpoint the player should resume from and write the directive."
        else:
            ask = "Choose the checkpoint the player should resume from. Leave \"directive\" as an empty string."
        messages = [
            {"role": "system", "content": SYSTEM.format(game=self.game, task=self.task)},
            {"role": "user", "content": f"{ledger}\n\n{ask}"},
        ]
        resp = self.client.chat.completions.create(
            model=self.model_id, messages=messages,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        self.accountant.add("orchestrator", resp.usage.prompt_tokens, resp.usage.completion_tokens)
        parsed, err = self._parse(text, {c["id"] for c in archive})
        self.rounds.append(
            {"ledger": ledger, "raw": text, "parsed": parsed, "error": err,
             "input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        )
        return parsed

    @staticmethod
    def _parse(text, valid_ids):
        err = None
        m = re.search(r"\{.*\}", text, re.S)
        obj = {}
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception as e:  # noqa: BLE001
                err = f"json: {e}"
        else:
            err = "no json object in reply"
        cid = str(obj.get("checkpoint", "")).strip()
        if cid not in valid_ids:
            err = (err or "") + f" | unknown checkpoint {cid!r}"
            cid = None
        return {"checkpoint": cid, "directive": str(obj.get("directive", "")).strip()}, err
