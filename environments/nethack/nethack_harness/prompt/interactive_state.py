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

The engine leaves the last message on row 0 after it is answered, so when
the observation has a cursor, a row-0 prompt counts only while the cursor is
on row 0.
"""
from __future__ import annotations

import re

from nethack_harness.prompt.tty_screen import (
    find_window, format_window, is_getpos_text, misc_flags, tty_cursor,
)

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


def _hero_tty_cell(raw_obs):
    """The hero's terminal cell (row, col): map (x, y) is tty (y + 1, x)."""
    bl = getattr(raw_obs, "blstats", None)
    if bl is None:
        return None
    try:
        return int(bl[1]) + 1, int(bl[0])
    except (IndexError, TypeError, ValueError):
        return None


# Item prompts end in a bracket group like "[bcde or ?*]" or "[- bh or ?*]".
_ITEM_PROMPT_RE = re.compile(r"\[([^\]]*?)(?:\s*or\s*)?\??\*\]")


def _item_prompt_letters(prompt: str):
    """Letters an item prompt suggests, ranges expanded, or None if not one."""
    m = _ITEM_PROMPT_RE.search(prompt or "")
    if not m:
        return None
    body = m.group(1)
    keys: list[str] = []
    i = 0
    while i < len(body):
        if (i + 2 < len(body) and body[i + 1] == "-" and body[i].isalpha()
                and body[i + 2].isalpha()):
            keys.extend(chr(o) for o in range(ord(body[i]), ord(body[i + 2]) + 1))
            i += 3
            continue
        if not body[i].isspace():
            keys.append(body[i])
        i += 1
    return keys


def _item_prompt_lines(prompt: str, inventory, published_tools=None) -> list[str]:
    keys = _item_prompt_letters(prompt)
    if keys is None:
        return []
    tools = {str(t) for t in (published_tools or ())}
    if "np_press_key" in tools:
        how = "`np_press_key(key='<letter>')`"
    elif any(t.startswith("bal_") for t in tools):
        how = "`bal_<letter>`"
    else:
        how = "pressing that letter"
    by_letter = {}
    for it in inventory or ():
        try:
            by_letter[str(it.letter)] = str(it.description)
        except AttributeError:
            continue
    out = ["This is an ITEM prompt: answer with an inventory letter, e.g. "
           f"{how}. ANY letter from your inventory is "
           "accepted, not only the suggested ones."]
    sugg = [k for k in keys if k.isalpha()]
    if sugg:
        out.append("The game suggests:")
        for k in sugg:
            out.append(f"  {k}: {by_letter.get(k, '(not in your inventory list)')}")
    if "-" in keys:
        out.append("  -: nothing / your bare hands or fingers")
    if "$" in keys:
        out.append("  $: gold")
    out.append("'?' lists the suggested items, '*' lists your whole inventory, "
               "`esc` cancels.")
    return out


def _window_lines(win) -> list[str]:
    if win.is_menu:
        title = "A MENU IS OPEN"
    else:
        title = "A TEXT WINDOW IS OPEN"
    page = win.page
    if page:
        title += f" (page {page[0]} of {page[1]})"
    lines = [f"=== GAME IS WAITING FOR INPUT: {title} ===",
             "The game clock is FROZEN until this is answered or closed; "
             "movement keys do nothing here. On screen:"]
    lines.extend(format_window(win))
    if win.is_menu:
        sel = [f"{l}" for (l, mark, _t) in win.entries if mark in "+#"]
        lines.append(
            "Press an entry's letter to select it (in a pick-one menu that "
            "chooses it immediately; in a pick-several menu it toggles, and a "
            "selected entry shows '+' instead of '-' after its letter). "
            "`enter` confirms, `esc` cancels.")
        lines.append("Currently selected: " + (", ".join(sel) if sel else "nothing"))
    else:
        lines.append("Press `enter` to continue past it, or `esc` to close it.")
    if page and page[0] < page[1]:
        nxt = "'>'" if win.is_menu else "'>' (or `enter`)"
        lines.append(f"More follows: press {nxt} for page {page[0] + 1}, "
                     "'<' to go back.")
    elif page and page[0] > 1:
        lines.append("'<' goes back to the previous page.")
    lines.append("")
    return lines


_FIND = object()


def detect_blocking_ui(raw_obs, published_tools=None, inventory=None,
                       getpos_pending: bool = False, window=_FIND) -> str | None:
    """Return a warning block if the game is blocked on UI input, else None.

    The returned string is inserted directly above `=== MAP ===` so the agent
    reads it before the (stale) dungeon view it would otherwise act on.

    `window` is the open window tracked across the call (default: read off
    `raw_obs`). `getpos_pending` means a position prompt (travel `_`) was the
    last thing this call printed; that prompt leaves `misc` clear and the
    cursor on the hero.
    """
    if raw_obs is None:
        return None

    row0 = _tty_row0(raw_obs)
    low = row0.lower()
    flags = misc_flags(raw_obs)
    cur = tty_cursor(raw_obs)
    win = find_window(raw_obs) if window is _FIND else window

    # No cursor (older engines, test fakes): match prompt text on row 0.
    if cur is None:
        blocked = bool(flags and any(flags)) or any(m in low for m in _PROMPT_MARKERS)
    else:
        hero = _hero_tty_cell(raw_obs)
        getpos = is_getpos_text(row0) and (
            getpos_pending or (hero is not None and cur != hero and cur[0] != 0))
        blocked = (bool(flags and any(flags)) or cur[0] == 0
                   or win is not None or getpos)
    if not blocked:
        return None

    if win is not None:
        return "\n".join(_window_lines(win))

    lines = [
        "=== GAME IS WAITING FOR INPUT — THE MAP BELOW IS STALE ===",
        "A prompt is open. The game clock is FROZEN until you answer it:",
        "movement and action keys are taken as answers, not as moves.",
    ]
    if row0:
        lines.append(f"Prompt line: {row0}")

    item_lines = _item_prompt_lines(row0, inventory, published_tools)
    if item_lines:
        lines.extend(item_lines)
    elif "--more--" in low or (flags is not None and flags[2]):
        # Waiting for space with no window drawn is a --More--, which a long
        # message can wrap off row 0.
        lines.append("The game is showing --More--: press `enter` to see the "
                     "next message (`esc` skips the rest of them).")
    elif "in what direction" in low:
        lines.append("Answer with a direction key: h j k l y u b n (west, "
                     "south, north, east, NW, NE, SW, SE), or < > . for "
                     "up, down, yourself; `esc` cancels.")
    elif is_getpos_text(row0):
        lines.append("Move the cursor with direction keys, then press '.' or "
                     "',' to pick the spot; `esc` cancels.")
    elif flags is not None and flags[1] and not flags[0]:
        lines.append("It wants typed text: press the keys one at a time, then "
                     "`enter`; `esc` cancels.")
    else:
        answers = _offered_answers(row0, published_tools)
        if answers:
            lines.append(f"Suggested answers (from the prompt): {answers}.")
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
