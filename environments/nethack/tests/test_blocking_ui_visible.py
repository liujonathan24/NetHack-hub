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

    assert detect_blocking_ui(obs) is None, "clean reset must not warn"
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
