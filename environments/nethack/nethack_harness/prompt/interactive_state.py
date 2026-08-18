"""Surface blocking NetHack UI state (open menu / waiting prompt) in the observation.

Why this exists — a measured, total-loss failure mode
------------------------------------------------------
`ascii_map.render_map_from_chars` renders the `=== MAP ===` block from the
glyph plane (`chars`) rather than `tty_chars`, which deleted the menu-bleed
defect class documented in that module's docstring. The cost of that choice:
`chars` has *no representation for an overlay at all*, so when NetHack opens a
menu or a prompt the map block keeps rendering the dungeon exactly as before.
The agent gets a normal-looking observation while the game is frozen.

Measured consequence (`outputs/encoding_eval/b80/b80_b0`, skill_set=balrog80,
glm-5.2). Three of five seeds pressed `i` (inventory) as their *first* action
and then never pressed `esc`:

    seed 0:  2,499 calls,  game time 1,  2,494 of them the single key `k`
    seed 1:  2,499 calls,  game time 1
    seed 2:  2,499 calls,  game time 1

The inventory menu swallows every movement key, the clock never advances, and
because the rendered observation is byte-identical turn over turn the model
emits the identical next action — a deterministic infinite loop. Those three
rollouts are a 100% loss: 7,497 LLM calls that played one game turn.

The same shape recurs with `_` (travel), which opens `Where do you want to
travel to?` and blocks: it precedes the wedge point in 5 of the 5 non-inventory
stalls across `b80_b0`, `b80_bbox` and `b80_bbox_json`.

This is not confined to the 80-key surface. Any skill set that can reach a
menu- or prompt-opening command can hit it; the keystroke surface just reaches
them far more often.

Detection
---------
`misc` is NLE's UI-state triple. Verified against a live engine (seed 0/0):

    reset                       misc=[0, 0, 0]   clock runs
    `i`   inventory menu        misc=[0, 0, 1]   clock FROZEN until esc
    `/`   whatis overlay        misc=[0, 0, 1]   clock FROZEN until esc
    `r`   "What do you want to read? [de or ?*]"
                                misc=[1, 0, 0]   clock FROZEN until answer/esc

Any nonzero entry means the engine is waiting on UI input and `blstats[20]`
(game time) cannot advance. That is the signal we key on — a flag the engine
sets, not a string we pattern-match.

`misc` does *not* cover the travel prompt (`_` measured as `[0, 0, 0]`), so
tty row 0 is checked for the small set of prompt shapes NetHack writes to the
message line. That half is a heuristic and is deliberately narrow: a false
positive would tell the agent to press `esc` when nothing is open, which costs
one call, whereas a false negative costs the entire rollout.
"""
from __future__ import annotations

import re

import numpy as np

# Prompt text NetHack writes to tty row 0 while blocking for input, for the
# cases `misc` leaves at zero. Matched case-insensitively as substrings.
_PROMPT_MARKERS = (
    "--more--",
    "where do you want to travel to?",
    "(for instructions type a",
    "(end)",
    "[yn",  # [yn], [ynq], [yn q] ...
)


def _tty_row0(raw_obs) -> str:
    tty = getattr(raw_obs, "tty_chars", None)
    if tty is None:
        return ""
    try:
        return "".join(chr(int(c)) for c in tty[0]).strip()
    except (IndexError, TypeError, ValueError):
        return ""


def detect_blocking_ui(raw_obs, published_tools=None) -> str | None:
    """Return a warning block if the game is blocked on UI input, else None.

    The returned string is inserted directly above `=== MAP ===` so the agent
    reads it before the (stale) dungeon view it would otherwise act on.
    """
    if raw_obs is None:
        return None

    row0 = _tty_row0(raw_obs)

    misc = getattr(raw_obs, "misc", None)
    blocked = False
    if misc is not None:
        try:
            blocked = bool(np.asarray(misc).any())
        except (TypeError, ValueError):
            blocked = False

    if not blocked:
        low = row0.lower()
        blocked = any(marker in low for marker in _PROMPT_MARKERS)

    if not blocked:
        return None

    lines = [
        "=== GAME IS WAITING FOR INPUT — THE MAP BELOW IS STALE ===",
        "A menu or prompt is open. The game clock is FROZEN: movement and",
        "action keys are being swallowed and change nothing. Repeating your",
        "last action will loop forever.",
    ]
    if row0:
        lines.append(f"Prompt line: {row0}")

    answers = _offered_answers(row0, published_tools)
    if answers:
        lines.append(f"ONLY these answers do anything right now: {answers}.")
        lines.append("Any other key is swallowed and changes nothing.")
    else:
        lines.append("Answer the prompt above, or press `esc` to cancel it.")
    lines.append("")
    return "\n".join(lines)


# NetHack advertises a prompt's legal replies inline: "What do you want to
# wield? [- bc or ?*]", "Really attack? [yn] (n)". Everything inside the first
# bracket group is a key that does something; every other key is discarded.
_OFFER_RE = re.compile(r"\[([^\]]+)\]")


def _offered_answers(row0: str, published_tools=None) -> str:
    """Name the prompt's legal replies as callable tools, not as raw keys.

    Measured need (`p1/nle_lang_glm`, glm-5.2 on the 80-command surface): the
    agent hit `What do you want to wield? [- bc or ?*]` and called `bal_e`
    **90 times in 103 turns** — game clock frozen at 15, 91% of calls wasted.
    `bal_b`, `bal_c` and `bal_minus` were all published and would each have
    worked. The old wording ("press `esc` (or answer the prompt shown above)")
    named only the cancel key, leaving the model to infer that the bracket
    group maps onto tool names. It did not make that leap.

    This is an OBSERVATION change, not an action-surface one: the agent still
    answers its own prompts, which is deliberate on this surface (see
    `nethack.py`'s `_balrog_raw_prompts` — auto-dismissing would mean `bal_eat`
    could never eat). We only stop hiding which of its existing tools apply.

    Ranges are expanded ("[a-c or ?*]" -> a, b, c) because NetHack abbreviates
    inventory letters that way and the tool names are per-letter.
    """
    m = _OFFER_RE.search(row0 or "")
    if not m:
        return ""
    body = m.group(1).replace(" or ", "")

    keys: list[str] = []
    i = 0
    while i < len(body):
        # "a-c" is a closed range of inventory letters, not three literal keys.
        if i + 2 < len(body) and body[i + 1] == "-" and body[i].isalnum() and body[i + 2].isalnum():
            for o in range(ord(body[i]), ord(body[i + 2]) + 1):
                keys.append(chr(o))
            i += 3
            continue
        if not body[i].isspace():
            keys.append(body[i])
        i += 1

    # Name keys in the vocabulary of whatever answer surface is actually
    # published: `np_press_key(key='b')` on the netplay_true / np_core raw
    # surfaces, `bal_b` on the BALROG-80 surface (the historical default,
    # kept verbatim so those arms render exactly as before).
    tools = {str(t) for t in (published_tools or ())}
    use_np = "np_press_key" in tools
    named = []
    for k in keys:
        if k == "*":
            named.append("'*' to list every item")
        elif k == "?":
            named.append("'?' to list valid choices")
        elif use_np and (k.isalnum() or k in "-$#,.<>"):
            named.append(f"`np_press_key(key='{k}')`")
        elif not use_np and k.isalnum():
            named.append(f"`bal_{k}`")
        elif not use_np and k == "-":
            named.append("`bal_minus` (the '-' key)")
    if not named:
        return ""
    named.append("`np_press_key(key='esc')` to cancel" if use_np
                 else "`bal_esc` to cancel")
    # Dedupe while preserving order; NetHack sometimes repeats a key across the
    # bracket group and the default hint "[yn] (n)".
    seen, out = set(), []
    for n in named:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return ", ".join(out)
