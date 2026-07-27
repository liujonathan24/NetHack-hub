"""Regression tests for two HINT-ladder defects found in the baseline (B0)
control-arm run `outputs/encoding_eval/abl2/enc_B0/turns/*.ndjson`.

Bug 1 (hunger): 40/40 turns that rendered the hunger HINT told the agent to
`eat(item=<food letter>)` with a literal placeholder while the actual
inventory held zero edible items -- an instruction the agent cannot comply
with. Two of five seeds starved.

Bug 2 (exits): the "no `>` visible, nearest exit" HINT has no memory, so it
cycles back to exits already recommended and already failed to make
progress (13,10 -> 26,10 -> 34,4 -> ... -> back to 13,10). 26% of all turns
in the run carried the "same suggestion N turns running" escalation without
the underlying exit target ever actually changing.
"""
from __future__ import annotations

import re
import types

import numpy as np

from nethack_core.observations import InventoryItem, StructuredObservation
from nethack_harness.prompt.rendering import format_observation_as_chat

_HINT_RE = re.compile(r"=== HINT === (.*)")


def _status(hunger_state=1, hp=20, hp_max=20, depth=1, x=10, y=3, time=1):
    return {
        "x": x, "y": y, "hitpoints": hp, "max_hitpoints": hp_max,
        "armor_class": 6, "depth": depth, "time": time,
        "experience_level": 1, "gold": 0, "hunger_state": hunger_state,
    }


# --------------------------------------------------------------------------- #
# Bug 1 -- hunger hint must be inventory-aware                                 #
# --------------------------------------------------------------------------- #

def _weak_structured(inventory=()):
    return StructuredObservation(
        map_view="", messages=[], inventory=list(inventory),
        status=_status(hunger_state=3),  # 3 == Weak
        character={}, adjacent={},
    )


def test_hunger_hint_does_not_tell_agent_to_eat_when_no_food():
    rendered = format_observation_as_chat(
        _weak_structured(inventory=[]), None, state={},
        compact=False, include_map=False,
    )
    assert "=== HINT ===" in rendered, rendered
    assert "eat(item=" not in rendered, rendered
    assert "<food letter>" not in rendered, rendered
    assert "no food" in rendered.lower(), rendered


def test_hunger_hint_names_the_actual_food_letter_when_present():
    food = InventoryItem(letter="f", description="a food ration", glyph=0)
    rendered = format_observation_as_chat(
        _weak_structured(inventory=[food]), None, state={},
        compact=False, include_map=False,
    )
    assert "=== HINT ===" in rendered, rendered
    assert '"f"' in rendered, rendered
    assert "food ration" in rendered, rendered
    assert "<food letter>" not in rendered, rendered


# --------------------------------------------------------------------------- #
# Bug 2 -- exit hint must not thrash between exits already tried               #
# --------------------------------------------------------------------------- #

def _room_chars() -> np.ndarray:
    """A single room, walls of `-`/`|`, two open doorways at (5,1) and
    (14,1) (a `.` gap in the horizontal top wall run)."""
    rows = [" " * 20]
    rows.append("-" * 5 + "." + "-" * 8 + "." + "-" * 5)
    rows.extend(["|" + "." * 18 + "|"] * 4)
    rows.append("-" * 20)
    rows.append(" " * 20)
    return np.array([[ord(c) for c in row] for row in rows], dtype=np.int16)


def _exit_structured(depth=1):
    return StructuredObservation(
        map_view="", messages=[], inventory=[],
        status=_status(hunger_state=1, depth=depth, x=10, y=3),
        character={}, adjacent={},
    )


def _render_exit_hint(state, depth=1):
    rendered = format_observation_as_chat(
        _exit_structured(depth=depth), None, state=state,
        compact=False, include_map=False,
    )
    m = _HINT_RE.search(rendered)
    assert m, rendered
    return m.group(1)


def test_exit_hint_rotates_after_no_progress_then_recommends_search():
    state = {"raw_obs": types.SimpleNamespace(chars=_room_chars())}
    hints = [_render_exit_hint(state) for _ in range(8)]

    # Nearest exit is (14,1) (Chebyshev 4 vs 5 for (5,1)) -- recommended
    # three turns running, the third carrying the escalation warning.
    assert "(14,1)" in hints[0], hints[0]
    assert "(14,1)" in hints[1], hints[1]
    assert "(14,1)" in hints[2], hints[2]
    assert "now been made 3 turns running" in hints[2], hints[2]

    # The FOURTH recommendation must differ -- repeating the exact same
    # suggestion a fourth time after already flagging it as "not working"
    # is exactly the thrash this bug report is about.
    assert "(14,1)" not in hints[3], hints[3]
    assert "(5,1)" in hints[3], hints[3]

    # Exhaust the second exit the same way...
    assert "(5,1)" in hints[4], hints[4]
    assert "(5,1)" in hints[5], hints[5]
    assert "now been made 3 turns running" in hints[5], hints[5]

    # ...and once both are tried, stop cycling between them -- recommend
    # `search` instead of looping back to (14,1) or (5,1).
    assert "(14,1)" not in hints[6], hints[6]
    assert "(5,1)" not in hints[6], hints[6]
    assert "search" in hints[6].lower(), hints[6]
    assert "search" in hints[7].lower(), hints[7]


def test_exit_hint_memory_is_keyed_by_dungeon_level():
    state = {"raw_obs": types.SimpleNamespace(chars=_room_chars())}
    for _ in range(4):
        hint = _render_exit_hint(state, depth=1)
    # (14,1) was recommended 3 turns running (the 3rd carries the escalation
    # warning and marks it tried); the 4th call rotates off it.
    assert "(14,1)" not in hint, hint
    assert "(5,1)" in hint, hint

    # A NEW level must not inherit Dlvl 1's tried-exit memory (the same
    # class of bug as the `>` memo leak fixed in Task 17): (14,1) is fair
    # game again on Dlvl 2 even though the underlying map (for this test)
    # happens to be geometrically identical.
    hint_l2 = _render_exit_hint(state, depth=2)
    assert "(14,1)" in hint_l2, hint_l2
