"""An open menu / waiting prompt must be visible in the observation.

Regression test for the total-loss wedge measured in
`outputs/encoding_eval/b80/b80_b0`: three of five seeds pressed `i`
(inventory) as their first action, never pressed `esc`, and then emitted the
same key 2,494 more times while the game clock sat at 1. The `=== MAP ===`
block renders from the glyph plane, which cannot represent an overlay, so the
observation was byte-identical every turn and the model had no reason to do
anything different.

The engine-backed test below is the real repro — it drives an actual NetHack
and asserts on what the agent would see. The unit tests pin the detector's
contract without paying for an engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from nethack_harness.prompt.interactive_state import detect_blocking_ui


class _FakeObs:
    def __init__(self, misc=(0, 0, 0), row0=""):
        self.misc = np.asarray(misc)
        padded = row0.ljust(80)[:80]
        self.tty_chars = np.array([[ord(c) for c in padded]] * 24, dtype=np.uint8)


def test_quiet_when_nothing_is_blocking():
    assert detect_blocking_ui(_FakeObs()) is None


def test_none_obs_is_tolerated():
    assert detect_blocking_ui(None) is None


@pytest.mark.parametrize(
    "misc",
    [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
    ids=["menu-open", "prompt-awaiting-response", "waiting-for-space"],
)
def test_any_nonzero_misc_flag_warns(misc):
    """`misc` is the engine's own UI-state triple; any nonzero entry means the
    clock cannot advance. Keyed on the flag, not on message text."""
    out = detect_blocking_ui(_FakeObs(misc=misc))
    assert out is not None
    assert "FROZEN" in out
    assert "esc" in out


def test_travel_prompt_warns_despite_all_zero_misc():
    """`_` (travel) measured as misc=[0,0,0] on a live engine, so the tty
    message line is the only signal. This is the case the `misc` check alone
    would miss."""
    out = detect_blocking_ui(
        _FakeObs(row0="Where do you want to travel to?  (For instructions type a '?')")
    )
    assert out is not None
    assert "travel" in out.lower()


def test_more_prompt_warns():
    assert detect_blocking_ui(_FakeObs(row0="You see here a food ration.--More--")) is not None


def test_ordinary_message_line_does_not_warn():
    """A false positive costs one wasted `esc` call, but the detector should
    still not fire on routine message-line text."""
    assert detect_blocking_ui(_FakeObs(row0="You kill the newt!")) is None
    assert detect_blocking_ui(_FakeObs(row0="It's a wall.")) is None


@pytest.mark.slow
def test_live_engine_inventory_wedge_is_surfaced():
    """The measured failure, end to end: open the inventory, then press a
    movement key twice. The clock must stay frozen (proving the wedge is real)
    and the detector must fire on every frozen turn (proving the agent is told)."""
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.seed(0, 0)
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset

    # The reset itself may land on a `--More--` (the welcome banner is one
    # message on some roles and two on others), which the detector correctly
    # flags. Clear whatever is pending BEFORE asserting a clean baseline —
    # otherwise this test passes or fails on the engine's random role.
    for _ in range(4):
        if detect_blocking_ui(obs) is None:
            break
        obs, _, _ = env.step(27)  # esc
    assert detect_blocking_ui(obs) is None, "a settled reset must not warn"
    t0 = int(obs.blstats[20])

    obs, _, _ = env.step(105)  # 'i' — open inventory
    assert detect_blocking_ui(obs) is not None, "open inventory must be surfaced"

    for _ in range(2):
        obs, _, _ = env.step(107)  # 'k' — north; swallowed by the menu
        assert int(obs.blstats[20]) == t0, "clock must be frozen while the menu is up"
        assert detect_blocking_ui(obs) is not None, "warning must persist, not fire once"

    obs, _, _ = env.step(27)  # esc — the escape the wedged rollouts never found
    assert detect_blocking_ui(obs) is None, "esc must clear the warning"

    obs, _, _ = env.step(107)
    assert int(obs.blstats[20]) > t0, "the game must move again once dismissed"


# --- naming the prompt's legal answers -------------------------------------
# Detection alone was not enough. Measured in `p1/nle_lang_glm` (glm-5.2,
# balrog80): the agent hit `What do you want to wield? [- bc or ?*]`, saw the
# "press esc" warning on every turn, and called `bal_e` 90 times in 103 turns
# with the clock frozen at 15. `bal_b`/`bal_c`/`bal_minus` were all published.
# The warning must name which of its EXISTING tools apply.

def test_offered_answers_names_the_published_tools():
    from nethack_harness.prompt.interactive_state import _offered_answers

    out = _offered_answers("What do you want to wield? [- bc or ?*]")
    assert "`bal_b`" in out and "`bal_c`" in out
    assert "`bal_minus`" in out
    assert "`bal_esc` to cancel" in out
    # The key the agent actually spammed is NOT offered and must not be named.
    assert "`bal_e`" not in out


def test_inventory_letter_ranges_are_expanded():
    """NetHack abbreviates "[d-f]"; tool names are per-letter, so a model told
    only "d-f" has to infer three tools from one token."""
    from nethack_harness.prompt.interactive_state import _offered_answers

    out = _offered_answers("What do you want to eat? [d-f or ?*]")
    for letter in "def":
        assert f"`bal_{letter}`" in out
    assert "`bal_g`" not in out


def test_yn_prompt_offers_both_and_no_more():
    from nethack_harness.prompt.interactive_state import _offered_answers

    out = _offered_answers("Really attack? [yn] (n)")
    assert "`bal_y`" in out and "`bal_n`" in out
    assert out.count("`bal_n`") == 1, "the '(n)' default must not duplicate the entry"


def test_no_bracket_group_yields_no_answer_list():
    from nethack_harness.prompt.interactive_state import _offered_answers

    assert _offered_answers("You kill the newt!") == ""
    assert _offered_answers("You see here a food ration.--More--") == ""


def test_warning_embeds_the_answer_list_when_one_exists():
    from nethack_harness.prompt.interactive_state import detect_blocking_ui

    out = detect_blocking_ui(_FakeObs(misc=(1, 0, 0), row0="What do you want to read? [de or ?*]"))
    assert out is not None
    assert "ONLY these answers do anything right now" in out
    assert "`bal_d`" in out and "`bal_e`" in out
