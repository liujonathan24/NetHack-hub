"""The PAE orchestrator: a separate GLM-5.2 chat that picks a checkpoint and
writes one short directive.

It sees only a ledger (checkpoint id, parent, attempt, step, BALROG
progression, a labelled dense measurement, one line of state) and the attempt
history.  It never sees the player's conversation.

Directives are STRATEGY, never keystrokes - the same rule the NetHack
orchestrator runs under.  Every directive is validated (1-2 sentences, <=300
chars, no literal key/action tokens); a violation buys exactly one re-ask, and
every rejection is logged to orchestrator_rounds.jsonl.
"""
from __future__ import annotations

import json
import re

from .prime_client import MODEL_ID, make_openai_client

MAX_DIRECTIVE_CHARS = 300
MAX_DIRECTIVE_SENTENCES = 2

# Literal-keystroke tells. Named actions ("head north", "search the wall") are
# strategy and stay legal; spelling out keys or quoting single characters is not.
_KEY_PATTERNS = [
    (r"\bpress(?:es|ed|ing)?\b", "press"),
    (r"\bkeystrokes?\b", "keystroke"),
    (r"\bhit\s+the\s+\w+\s+key\b", "hit the ... key"),
    (r"\bkeys?\b(?!\s*(?:to|of|item|ring))", "key"),
    (r"\baction\s*:", "action:"),
    (r"\btype\s+(?:the\s+)?(?:letter|character|key)\b", "type the letter"),
    (r"(?<![\w])['\"`]\w['\"`](?![\w])", "single quoted character"),
    (r"\boutput\s+(?:the\s+)?(?:exact|literal)\b", "output the literal"),
    (r"\bctrl\s*[-+]\s*\w\b", "ctrl-key"),
    (r"<\s*\w+\s*>", "angle-bracket key"),
]

SYSTEM = """You are the explorer in a checkpointed-exploration loop for the game {game} ({task}).

A player agent plays the game one action at a time. Whenever it stops (death, step
cap, or the task ends) you choose which SAVED CHECKPOINT it should resume from and
write ONE short directive telling it what to do differently. The player resumes
with its own conversation history exactly as it was at that checkpoint, plus your
directive as one extra message; it cannot see this conversation or the ledger.

CHOOSING A CHECKPOINT. Every id in the ledger is choosable, not just the deepest
or the newest. Prefer a branch point close to the frontier of what has been
achieved but early enough to still change the outcome; if attempts from one
checkpoint keep failing the same way, step further back. The ROOT checkpoint
(the one with no parent) is a FULL RESTART of the episode: it throws away every
step taken so far, so choose it only when you can say why a restart beats
branching, and never as a default.

WRITING A DIRECTIVE. A directive is STRATEGY - what to aim for, what to avoid,
what the last attempt got wrong. It is NEVER keystrokes, key names, literal
characters to output, or a transcription of the action the player should emit;
the player already knows its own action list. At most 2 sentences and at most
{maxchars} characters.

Reply with ONLY a JSON object, no prose, no code fence:
{{"checkpoint": "<checkpoint id>", "directive": "<at most 2 sentences of strategy>"}}"""


def validate_directive(text: str) -> tuple[bool, list[str]]:
    """(ok, reasons). Empty directives are legal (the no-directive arms)."""
    problems: list[str] = []
    if not text:
        return True, problems
    if len(text) > MAX_DIRECTIVE_CHARS:
        problems.append(f"too long ({len(text)} > {MAX_DIRECTIVE_CHARS} chars)")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sentences) > MAX_DIRECTIVE_SENTENCES:
        problems.append(f"too many sentences ({len(sentences)} > {MAX_DIRECTIVE_SENTENCES})")
    for pattern, label in _KEY_PATTERNS:
        if re.search(pattern, text, re.I):
            problems.append(f"literal key/action token: {label}")
    return not problems, problems


class Orchestrator:
    def __init__(self, game, task, accountant, model_id=MODEL_ID, temperature=1.0, max_tokens=1024,
                 step_cap: int | None = None):
        self.client = make_openai_client()
        self.game, self.task = game, task
        #: the run's EFFECTIVE horizon - cfg.max_steps when given, else the env
        #: default. Never adapter.max_steps directly: a run with --max-steps 20
        #: would otherwise be told it has the env's 80 steps to play with.
        self.step_cap = step_cap
        self.accountant = accountant
        self.model_id, self.temperature, self.max_tokens = model_id, temperature, max_tokens
        self.rounds: list[dict] = []
        self.rejections = 0

    def _steps_left(self, step) -> str:
        return "?" if self.step_cap is None else str(max(0, self.step_cap - step))

    def _ledger(self, archive, attempts) -> str:
        aux = attempts[0].get("aux_label", "measured") if attempts else "measured"
        cap = "unknown" if self.step_cap is None else str(self.step_cap)
        lines = [
            f"EPISODE HORIZON: {cap} steps total. A checkpoint taken at step S leaves {cap} - S steps"
            " to play, so a deep checkpoint buys progress but little room to change course.",
            "",
            f"CHECKPOINT LEDGER (id | parent | from attempt | step | steps left | BALROG progression | {aux} | state)",
            "the checkpoint with parent '-' is the ROOT: resuming it restarts the episode from scratch",
        ]
        for c in archive:
            lines.append(
                f"{c['id']} | {c.get('parent') or '-'} | a{c['attempt']} | step {c['step']} | "
                f"{self._steps_left(c['step'])} | "
                f"{c['progression']:.3f} | {c['aux_progress']:.0f} | {c['summary']}"
            )
        lines.append("")
        lines.append("ATTEMPT HISTORY (attempt | resumed from | steps played | final progression | outcome | directive given)")
        for a in attempts:
            lines.append(
                f"a{a['attempt']} | {a['from_checkpoint'] or 'fresh episode'} | {a['steps_played']} | "
                f"{a['progression']:.3f} | {a['outcome']} | {(a['directive'] or '-')[:160]}"
            )
        return "\n".join(lines)

    def _call(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model_id, messages=messages,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        self.accountant.add("orchestrator", resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return (resp.choices[0].message.content or "").strip(), resp.usage

    def decide(self, archive, attempts, want_directive: bool, fixed_checkpoint: str | None = None) -> dict:
        ledger = self._ledger(archive, attempts)
        if fixed_checkpoint is not None:
            ask = (
                f"The player will resume from checkpoint {fixed_checkpoint} (fixed by rule; you do not choose it). "
                f"Write the directive. Set \"checkpoint\" to {fixed_checkpoint}."
            )
        elif want_directive:
            ask = "Choose the checkpoint the player should resume from and write the directive."
        else:
            ask = "Choose the checkpoint the player should resume from. Leave \"directive\" as an empty string."

        valid_ids = {c["id"] for c in archive}
        root_ids = {c["id"] for c in archive if not c.get("parent")}
        messages = [
            {"role": "system", "content": SYSTEM.format(game=self.game, task=self.task, maxchars=MAX_DIRECTIVE_CHARS)},
            {"role": "user", "content": f"{ledger}\n\n{ask}"},
        ]

        text, usage = self._call(messages)
        parsed, err = self._parse(text, valid_ids)
        in_tok, out_tok = usage.prompt_tokens, usage.completion_tokens
        rejected: list[dict] = []

        if want_directive:
            ok, problems = validate_directive(parsed["directive"])
            if not ok:
                # exactly one re-ask, with the reasons quoted back
                rejected.append({"directive": parsed["directive"], "problems": problems, "round": 1})
                self.rejections += 1
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content":
                        "That directive was rejected: " + "; ".join(problems) +
                        ". Rewrite it as strategy only - no key names, no literal characters to output, "
                        f"at most {MAX_DIRECTIVE_SENTENCES} sentences and {MAX_DIRECTIVE_CHARS} characters. "
                        "Reply with the same JSON object and the same checkpoint."},
                ]
                text2, usage2 = self._call(messages)
                in_tok += usage2.prompt_tokens
                out_tok += usage2.completion_tokens
                parsed2, err2 = self._parse(text2, valid_ids)
                ok2, problems2 = validate_directive(parsed2["directive"])
                if ok2:
                    parsed, err, text = parsed2, err2, text2
                else:
                    rejected.append({"directive": parsed2["directive"], "problems": problems2, "round": 2})
                    self.rejections += 1
                    # keep the checkpoint choice, drop the unusable directive
                    parsed = {"checkpoint": parsed2["checkpoint"] or parsed["checkpoint"], "directive": ""}
                    err = (err or "") + " | directive dropped after 2 failed validations"

        self.rounds.append({
            "ledger": ledger, "raw": text, "parsed": parsed, "error": err,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "fixed_checkpoint": fixed_checkpoint,
            "chosen_is_root": parsed["checkpoint"] in root_ids if parsed["checkpoint"] else None,
            "n_candidates": len(archive),
            "directive_rejections": rejected,
        })
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
