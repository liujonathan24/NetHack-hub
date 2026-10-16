"""What the NetCore agent sees of NetHack's own UI, read off the terminal of a real seeded game."""

from nethack_harness.prompt import tty_screen as ts
from nethack_harness.prompt.interactive_state import detect_blocking_ui


class _Game:
    """A Valkyrie game (seed 12), played one np_press_key-like call at a time."""

    def __init__(self):
        from nethack_core import NetHackCoreEnv

        self.env = NetHackCoreEnv(task_name="NetHackScore-v0", max_episode_steps=100_000,
                                  tune={"reveal_map": 1.0})
        self.env.seed(core=12, disp=12)
        self.obs, _ = self.env.reset(character="Val-hum-neu-fem")
        for _ in range(10):  # acknowledge the welcome --More-- chain
            if not any("--More--" in r for r in ts.tty_rows(self.obs)):
                break
            self.obs, *_ = self.env.step(13)
        self.log = None

    def call(self, keys):
        """`keys` is space-separated: ESC, CR, SP, ^<code>, or one character each."""
        prev = self.log.window if self.log is not None else None
        self.log = ts.CallScreenLog(self.env, baseline_obs=self.obs, prev_window=prev)
        with self.log:
            for t in keys.split():
                named = {"ESC": 27, "CR": 13, "SP": 32}.get(t)
                self.obs, *_ = self.env.step(named or (int(t[1:]) if t.startswith("^") else ord(t)))
        return self.log

    def drop(self, letters):
        for letter in letters:
            self.call("d")
            log = self.call(letter)
        return log

    def inventory(self):
        from nethack_core.observations import shape
        return shape(self.obs, {}).inventory

    def banner(self):
        return detect_blocking_ui(self.obs, {"np_press_key"}, self.inventory(),
                                  window=self.log.window if self.log else None)

    def shows(self, *texts):
        out = self.banner() or ""
        assert [t for t in texts if t not in out] == [], out
        return out


def test_inventory_menu_renders_every_entry_with_real_letters():
    g = _Game()
    g.call("i")
    g.shows("A MENU IS OPEN", "  Weapons", "  a - a +1 long sword (weapon in hand)",
            "  b - a +0 dagger (alternate weapon; not wielded)", "  Armor",
            "  c - an uncursed +3 small shield (being worn)", "  Comestibles",
            "  d - an uncursed food ration", "  (end)")
    g.call("ESC")
    assert g.banner() is None


def test_pickup_menu_keeps_entries_and_selection_across_partial_redraws():
    g = _Game()
    g.drop("bd")
    g.call(",")
    g.shows("Pick up what?", "a - a +0 dagger", "b - an uncursed food ration")
    g.call("a")  # toggling an entry repaints only that row on this engine's tty
    g.shows("a + a +0 dagger", "b - an uncursed food ration", "Currently selected: a")
    log = g.call("CR")
    assert g.banner() is None
    assert any("a +0 dagger" in m for m in log.lines), log.lines


def test_item_prompt_lists_inventory_letters_not_indices():
    g = _Game()
    g.call("d")
    out = g.shows("What do you want to drop? [abcd or ?*]", "  a: a +1 long sword (weapon in hand)",
                  "  d: an uncursed food ration", "ANY letter from your inventory is accepted")
    assert "[0]" not in out and "ONLY these" not in out


def test_look_here_window_is_reported_as_messages():
    g = _Game()
    g.drop("bd")
    lines = g.call(": SP").lines  # space dismisses the window, as skip_more does
    assert lines[:1] == ["There is a staircase up here."], lines
    assert {"Things that are here:", "a +0 dagger", "an uncursed food ration"} <= set(lines), lines


def test_more_chain_reports_every_message_in_order():
    lines = _Game().call("^240 y CR CR CR").lines  # M-p (pray), confirm, dismiss each --More--
    assert len(lines) >= 2 and lines[0].startswith("You begin praying"), lines
    assert all("Are you sure you want to pray" not in m for m in lines), lines  # an answered prompt


def test_old_message_is_not_reported_again():
    g = _Game()
    assert g.drop("b").lines == ["You drop a +0 dagger."]
    assert g.call("s").lines == []  # `s` prints nothing


def test_a_message_printed_again_by_the_next_command_is_reported_again():
    g = _Game()
    assert g.call(">").lines == ["You can't go down here."]
    assert g.call(">").lines == ["You can't go down here."]
    assert g.call("> >").lines == ["You can't go down here."] * 2


def test_a_count_or_prefix_key_does_not_repeat_the_previous_message():
    g = _Game()
    g.call(">")
    assert g.call("2 .").lines == []  # the count digit leaves the line as it was
    assert g.call("F h").lines == ["You attack thin air."]


def test_a_key_ignored_at_a_more_prompt_does_not_repeat_the_message():
    g = _Game()
    g.call("^240 y")  # M-p (pray), confirm: the first message waits at --More--
    assert g.call("s").lines == []  # swallowed by the --More--; nothing new
    assert g.call("CR").lines, "dismissing it prints the next message"


def test_answered_yn_prompt_does_not_leave_a_stale_banner():
    g = _Game()
    g.drop("d")
    g.call("e")  # eat the food ration on the floor
    g.shows("eat it? [ynq]", "Suggested answers")
    g.call("q")
    g.call("s")  # the clock advances: nothing is pending
    assert g.banner() is None


def test_attributes_pages_render_with_page_marker():
    g = _Game()
    g.call("^24")  # ^X
    g.shows("A TEXT WINDOW IS OPEN (page 1 of 2)", "Agent the Valkyrie's attributes:",
            "You are a Stripling, a level 1 human Valkyrie.")
    g.call(">")
    g.shows("(page 2 of 2)", "Current Status:")
    g.call("ESC")
    assert g.banner() is None


def test_adjacent_is_computed_around_the_hero_not_the_first_at_sign():
    from nethack_harness.prompt.features import adjacent_tiles

    obs = _Game().obs
    assert adjacent_tiles(obs)["S"] == "f(kitten)[PET — don't attack]"
    obs.tty_chars[2, 12] = ord("@")  # a human monster above the hero on screen
    assert "kitten" in adjacent_tiles(obs)["S"]


def test_inventory_listing_overrides_stale_inv_strs():
    g = _Game()
    for keys in ("w", "b CR", "E", "b CR", *"nxyz"):  # wield the dagger (b), engrave "nxyz" with it
        g.call(keys)
    assert "Your dagger gets dull." in " ".join(g.call("CR").lines)
    inv = {it.letter: it.description for it in g.inventory()}
    assert inv["b"].startswith("a +0 dagger"), "inv_strs is refreshed only by update_inventory()"
    log = g.call("i")
    assert log.inventory_listing and log.inventory_listing["b"].startswith("a -2 dagger")
    items = g.inventory()
    ts.apply_inventory_listing({}, items, log)
    assert {it.letter: it.description for it in items}["b"].startswith("a -2 dagger")


def test_bound_lines_elides_the_middle():
    out = ts.bound_lines([f"m{i}" for i in range(100)])
    assert len(out) == ts.MAX_MESSAGE_LINES + 1
    assert out[0] == "m0" and out[-1] == "m99" and "elided" in out[ts._HEAD_LINES]
