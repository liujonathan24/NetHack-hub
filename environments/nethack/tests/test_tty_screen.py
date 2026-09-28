"""What the NetCore agent sees of NetHack's own UI, read off the terminal.

Engine-backed: each test drives a real seeded game (Valkyrie, seed 12) and
asserts on what the observation layer reports. The play-test failures these
pin (all on the np_core surface):

* `i` rendered as one garbled entry; the pick-up menu had no entries; the
  "Things that are here:" window and the ^X / enlightenment pages were lost.
* item prompts listed `[0] [1] ...` instead of the inventory letters.
* only the last message of a call was shown, and an OLD message was shown
  again as if new (this engine never clears the message line).
* "eat it? [ynq]" stayed in the waiting-for-input banner after it was answered.
* ADJACENT described the tiles around the FIRST `@` on the screen.
* inventory text stayed stale after "Your dagger gets dull."
"""
from __future__ import annotations

import numpy as np
import pytest

from nethack_harness.prompt import tty_screen as ts
from nethack_harness.prompt.interactive_state import detect_blocking_ui


def _game():
    from nethack_core import NetHackCoreEnv

    env = NetHackCoreEnv(task_name="NetHackScore-v0", max_episode_steps=100_000,
                         tune={"reveal_map": 1.0})
    env.seed(core=12, disp=12)
    obs, _ = env.reset(character="Val-hum-neu-fem")
    for _ in range(10):  # the server acknowledges the welcome --More-- chain
        if not any("--More--" in r for r in ts.tty_rows(obs)):
            break
        obs, *_ = env.step(13)
    return env, obs


def _keys(s):
    return [{"ESC": 27, "CR": 13, "SP": 32}.get(t, None) or (int(t[1:]) if t.startswith("^") else ord(t))
            for t in s.split()]


class _Game:
    """Press keys one call at a time, like np_press_key, with a CallScreenLog."""

    def __init__(self):
        self.env, self.obs = _game()
        self.log = None

    def call(self, keys):
        prev = self.log.window if self.log is not None else None
        self.log = ts.CallScreenLog(self.env, baseline_obs=self.obs, prev_window=prev)
        with self.log:
            for k in _keys(keys):
                self.obs, *_ = self.env.step(k)
        return self.log

    def inventory(self):
        from nethack_core.observations import shape
        return shape(self.obs, {}).inventory

    def banner(self):
        return detect_blocking_ui(self.obs, {"np_press_key"}, self.inventory(),
                                  window=self.log.window if self.log else None)


def test_inventory_menu_renders_every_entry_with_real_letters():
    g = _Game()
    g.call("i")
    out = g.banner()
    assert out is not None and "A MENU IS OPEN" in out
    for line in ("Weapons", "b - a +1 long sword (weapon in hand)",
                 "c - a +0 dagger (alternate weapon; not wielded)", "Armor",
                 "d - an uncursed +3 small shield (being worn)", "Comestibles",
                 "e - an uncursed food ration", "(end)"):
        assert f"  {line}" in out, line
    g.call("ESC")
    assert g.banner() is None


def test_pickup_menu_keeps_entries_and_selection_across_partial_redraws():
    g = _Game()
    for k in ("d", "c", "d", "e"):
        g.call(k)
    g.call(",")
    out = g.banner()
    assert "Pick up what?" in out and "a - a +0 dagger" in out
    assert "b - an uncursed food ration" in out
    # Toggling an entry repaints only that row on this engine's tty.
    g.call("a")
    out = g.banner()
    assert "a + a +0 dagger" in out, out
    assert "b - an uncursed food ration" in out
    assert "Currently selected: a" in out
    log = g.call("CR")
    assert g.banner() is None
    assert any("a +0 dagger" in m for m in log.lines), log.lines


def test_item_prompt_lists_inventory_letters_not_indices():
    g = _Game()
    g.call("d")
    out = g.banner()
    assert "What do you want to drop? [bcde or ?*]" in out
    assert "  b: a +1 long sword (weapon in hand)" in out
    assert "  e: an uncursed food ration" in out
    assert "ANY letter from your inventory is accepted" in out
    assert "[0]" not in out and "ONLY these" not in out


def test_look_here_window_is_reported_as_messages():
    g = _Game()
    for k in ("d", "c", "d", "e"):
        g.call(k)
    log = g.call(": SP")  # ":" opens the window; space dismisses it, as skip_more does
    assert log.lines[:1] == ["There is a staircase up here."], log.lines
    assert "Things that are here:" in log.lines
    assert "a +0 dagger" in log.lines and "an uncursed food ration" in log.lines


def test_more_chain_reports_every_message_in_order():
    g = _Game()
    log = g.call("^240 y CR CR CR")  # M-p (pray), confirm, acknowledge --More--s
    joined = " | ".join(log.lines)
    assert log.lines[0].startswith("You begin praying"), joined
    assert len(log.lines) >= 2, joined
    assert not any("Are you sure you want to pray" in m for m in log.lines), \
        "an answered prompt is not a message"


def test_old_message_is_not_reported_again():
    g = _Game()
    g.call("d")
    log = g.call("c")
    assert log.lines == ["You drop a +0 dagger."]
    # `s` prints nothing; the message line still shows the drop message.
    log = g.call("s")
    assert log.lines == [], log.lines
    assert "You drop a +0 dagger." in ts.obs_message(g.obs), "engine keeps old text"


def test_answered_yn_prompt_does_not_leave_a_stale_banner():
    g = _Game()
    g.call("d")
    g.call("e")                        # food ration on the floor
    g.call("e")
    out = g.banner()
    assert out is not None and "eat it? [ynq]" in out
    assert "Suggested answers" in out
    g.call("q")
    g.call("s")                        # the clock advances: nothing is pending
    assert "eat it?" in ts.row0(g.obs), "row 0 still shows the old prompt"
    assert g.banner() is None


def test_attributes_pages_render_with_page_marker():
    g = _Game()
    g.call("^24")                      # ^X
    out = g.banner()
    assert "A TEXT WINDOW IS OPEN (page 1 of 2)" in out, out
    assert "Agent the Valkyrie's attributes:" in out
    assert "You are a Stripling, a level 1 human Valkyrie." in out
    g.call(">")
    out = g.banner()
    assert "(page 2 of 2)" in out and "Current Status:" in out
    g.call("ESC")
    assert g.banner() is None


def test_adjacent_is_computed_around_the_hero_not_the_first_at_sign():
    from nethack_core.observations import extract_adjacent
    from nethack_harness.prompt.features import adjacent_tiles

    env, obs = _game()
    x, y = int(obs.blstats[0]), int(obs.blstats[1])
    assert adjacent_tiles(obs)["S"] == "f(kitten)[PET — don't attack]"
    # A human monster above the hero on screen (the Oracle, a shopkeeper...).
    obs.tty_chars[2, 12] = ord("@")
    old = extract_adjacent(obs.tty_chars, obs.glyphs)
    new = adjacent_tiles(obs)
    assert "kitten" in new["S"]
    assert "f(" not in old.get("S", ""), "the old extractor looked elsewhere"


def test_inventory_listing_overrides_stale_inv_strs():
    g = _Game()
    g.call("w")
    g.call("c CR")
    g.call("E")
    g.call("c CR")
    for k in "nxyz":
        g.call(k)
    log = g.call("CR")
    assert "Your dagger gets dull." in " ".join(log.lines)
    inv = {it.letter: it.description for it in g.inventory()}
    assert inv["c"].startswith("a +0 dagger"), "engine inv_strs is stale here"
    log = g.call("i")
    assert log.inventory_listing and log.inventory_listing["c"].startswith("a -2 dagger")
    state: dict = {}
    items = g.inventory()
    ts.apply_inventory_listing(state, items, log)
    assert {it.letter: it.description for it in items}["c"].startswith("a -2 dagger")


def test_bound_lines_elides_the_middle():
    lines = [f"m{i}" for i in range(100)]
    out = ts.bound_lines(lines)
    assert len(out) == ts.MAX_MESSAGE_LINES + 1
    assert out[0] == "m0" and out[-1] == "m99"
    assert "elided" in out[ts._HEAD_LINES]
