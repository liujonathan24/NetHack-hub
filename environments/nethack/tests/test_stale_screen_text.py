"""Text left on screen from something already over must not end or block the game.

Both cases come from DeepSeek-v4-Flash rollouts on the np_core sandbox:

* "The gnome is killed by the magic missile!  The magic missile hits you!"
  matched the death marker "killed by"; the game was reported over, and the
  rollout ended, with the character alive at HP 46/65.
* After the Quest portal, the Quest entry text window stayed drawn on screen
  once it had been dismissed. With the hero standing inside that remnant, the
  observation announced an open text window and a frozen clock for the rest of
  the game while the clock kept running.
"""
from __future__ import annotations

import numpy as np

from nethack_harness.helpers import _detect_terminal_outcome
from nethack_harness.prompt.tty_screen import find_window


class _Screen:
    def __init__(self, rows, misc=(0, 0, 0), cursor=None):
        rows = list(rows) + [""] * (24 - len(rows))
        self.tty_chars = np.array([[ord(c) for c in r.ljust(80)[:80]] for r in rows], dtype=np.uint8)
        self.misc = np.asarray(misc)
        if cursor is not None:
            self.tty_cursor = np.asarray(cursor)


def test_a_monster_killed_by_something_is_not_a_death():
    obs = _Screen(["The gnome is killed by the magic missile!  The magic missile hits you!"])
    state = {"terminated": False}
    _detect_terminal_outcome(obs, state)
    assert state["died"] is False
    assert state["terminated"] is False


def test_the_death_prompt_is_still_a_death():
    obs = _Screen(["Do you want your possessions identified? [ynq] (n)"], misc=(1, 0, 0))
    state = {"terminated": False}
    _detect_terminal_outcome(obs, state)
    assert state["died"] is True
    assert state["terminated"] is True


_QUEST_TEXT = [
    "",
    "                    You materialize at the base of a snowy hill.",
    "                    Atop the hill sits a place you know well.",
    "                    --More--",
]


def test_a_dismissed_text_window_is_not_open():
    # The hero (cursor) stands inside the leftover text; the engine waits for nothing.
    assert find_window(_Screen(_QUEST_TEXT, misc=(0, 0, 0), cursor=(2, 30))) is None


def test_a_text_window_the_engine_waits_on_is_open():
    win = find_window(_Screen(_QUEST_TEXT, misc=(0, 0, 1), cursor=(3, 29)))
    assert win is not None and win.terminator == "--More--"
