"""Text left on screen from something already over must not end or block the game."""

import numpy as np
import pytest

from nethack_harness.helpers import _detect_terminal_outcome
from nethack_harness.prompt.tty_screen import find_window


class _Screen:
    def __init__(self, rows, misc=(0, 0, 0), cursor=None):
        rows = list(rows) + [""] * (24 - len(rows))
        self.tty_chars = np.array([[ord(c) for c in r.ljust(80)[:80]] for r in rows], dtype=np.uint8)
        self.misc = np.asarray(misc)
        if cursor is not None:
            self.tty_cursor = np.asarray(cursor)


@pytest.mark.parametrize("row0,misc,died", [
    ("The gnome is killed by the magic missile!  The magic missile hits you!", (0, 0, 0), False),
    ("Do you want your possessions identified? [ynq] (n)", (1, 0, 0), True),
], ids=["monster-killed-by-something", "death-prompt"])
def test_only_the_death_prompt_is_a_death(row0, misc, died):
    state = {"terminated": False}
    _detect_terminal_outcome(_Screen([row0], misc=misc), state)
    assert state["died"] is died
    assert state["terminated"] is died


_QUEST_TEXT = [
    "",
    "                    You materialize at the base of a snowy hill.",
    "                    Atop the hill sits a place you know well.",
    "                    --More--",
]


def test_a_text_window_is_open_only_while_the_engine_waits_on_it():
    # Dismissed: the engine waits for nothing; the hero (cursor) stands inside the leftover text.
    assert find_window(_Screen(_QUEST_TEXT, misc=(0, 0, 0), cursor=(2, 30))) is None
    win = find_window(_Screen(_QUEST_TEXT, misc=(0, 0, 1), cursor=(3, 29)))
    assert win is not None and win.terminator == "--More--"
