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


def detect_blocking_ui(raw_obs) -> str | None:
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
    lines.append("Dismiss it FIRST — press `esc` (or answer the prompt shown above).")
    lines.append("")
    return "\n".join(lines)
