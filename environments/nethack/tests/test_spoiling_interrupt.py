"""The spoiling-food interrupt must actually fire, and its guards must not lie.

Regression for docs/HARNESS_DEFECTS.md 3.2. `_spoiling_now` handed a
three-field `_RawView` (chars/glyphs/blstats) to `shape()`, which reads
`.message` first; the resulting `AttributeError` was eaten by a bare `except`
and returned `[]`. So the interrupt never fired in a single rollout, and
"nothing is spoiling" was indistinguishable from "this function is broken".

Both halves are covered here:
  * the interrupt fires, on the conditions `prompt/corpse_age.py` defines
    (that module owns the rot arithmetic; nothing is re-derived here);
  * a guard that swallows an exception counts and logs it, and
    `NETHACK_STRICT_SKILL_ERRORS=1` makes it raise instead.
"""

import pathlib
import sys

import numpy as np
import pytest

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_core.observations import shape  # noqa: E402
from nethack_harness.helpers import classify_tool_result  # noqa: E402
from nethack_harness.prompt import corpse_age  # noqa: E402
from nethack_harness.tools import netplay_true as npt  # noqa: E402


@pytest.fixture()
def env():
    from nethack_core.env import NetHackCoreEnv
    e = NetHackCoreEnv(task_name="NetHackScore-v0")
    e.seed(42, 42)
    e.reset()
    npt.reset_swallowed_exceptions()
    npt.reset_agent_cache()
    try:
        yield e
    finally:
        e.close()


def _advance_clock(env, at_least: int = 40) -> int:
    """Search ('s') until the in-game clock is past `at_least`."""
    for _ in range(200):
        if int(env._last_observation.blstats[20]) >= at_least:
            break
        env.step(115)
    return int(env._last_observation.blstats[20])


def _carry_corpse_into(obs, description: str = "a newt corpse", letter: str = "e") -> None:
    """Put a corpse in a free inventory slot of an observation.

    NLE pre-allocates 55 slots; writing into an empty one is the only way to
    hand the harness a carried corpse without playing out a hunt.
    """
    free = int(np.nonzero(obs.inv_letters == 0)[0][0])
    raw = description.encode("ascii")
    obs.inv_strs[free][:] = 0
    obs.inv_strs[free][:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    obs.inv_letters[free] = ord(letter)


def _carry_corpse(env, description: str = "a newt corpse", letter: str = "e") -> None:
    _carry_corpse_into(env._last_observation, description, letter)


# ---------------------------------------------------------------------------
# The mechanism that was broken: the view handed to shape().
# ---------------------------------------------------------------------------

def test_raw_view_is_something_shape_can_actually_consume(env):
    """`shape()` needs message/inv_*/tty_chars, not just chars/glyphs/blstats."""
    view = npt._raw_view(env)
    assert view is not None
    so = shape(view, None)             # used to raise AttributeError: message
    assert so.status and "time" in so.status


def test_raw_view_from_the_list_form_is_also_complete(env):
    """The list fallback must not rebuild a crippled three-field view."""

    class _ListOnly:
        observation_keys = env.observation_keys
        last_observation = env.last_observation

    view = npt._raw_view(_ListOnly())
    assert isinstance(view, npt._RawView)
    for field in ("message", "inv_strs", "inv_letters", "inv_glyphs",
                  "tty_chars", "chars", "glyphs", "blstats"):
        assert getattr(view, field) is not None, field
    assert shape(view, None).status["time"] >= 0


def test_a_missing_field_raises_rather_than_reading_as_empty(env):
    """The old failure was a *silent* None. An absent field must be loud."""
    view = npt._RawView(chars=1, glyphs=2, blstats=3)
    with pytest.raises(AttributeError):
        view.message


# ---------------------------------------------------------------------------
# The interrupt itself.
# ---------------------------------------------------------------------------

def test_no_corpse_carried_means_nothing_urgent_and_nothing_swallowed(env):
    assert npt._spoiling_now(env) == []
    assert npt.swallowed_exceptions() == {}, "a clean turn must swallow nothing"


def test_a_corpse_about_to_spoil_is_reported(env):
    """Fires when the kill is `RISKY_AGE - URGENT_WINDOW` .. `RISKY_AGE` old.

    Thresholds come from `corpse_age` (RISKY_AGE = 30, derived from
    eat.c:1673; URGENT_WINDOW = 5) -- this test reads them rather than
    hard-coding, so the two layers cannot drift apart.
    """
    now = _advance_clock(env)
    assert now >= corpse_age.RISKY_AGE, now
    _carry_corpse(env)
    env._kill_log = {"newt": now - (corpse_age.RISKY_AGE - 2)}   # 2 moves left

    urgent = npt._spoiling_now(env)

    assert urgent, "the interrupt did not fire on a corpse 2 moves from rotting"
    assert "newt corpse" in urgent[0]
    assert "rots in 2 move(s)" in urgent[0]
    assert npt.swallowed_exceptions() == {}


def test_a_fresh_corpse_does_not_interrupt(env):
    now = _advance_clock(env)
    _carry_corpse(env)
    env._kill_log = {"newt": now}
    assert npt._spoiling_now(env) == []


def test_an_already_rotten_corpse_does_not_interrupt(env):
    """Past the window there is nothing to save -- breaking the macro then is
    pure cost. `corpse_age.annotate_corpse` still labels it RISKY in the
    inventory, which is where that warning belongs."""
    now = _advance_clock(env)
    _carry_corpse(env)
    env._kill_log = {"newt": now - (corpse_age.RISKY_AGE + 10)}
    assert npt._spoiling_now(env) == []


def test_a_corpse_of_unknown_age_does_not_interrupt(env):
    """No kill log entry -> no age -> no guess. Same contract as
    `annotate_corpse`, which reports `age UNKNOWN` rather than inventing one."""
    _advance_clock(env)
    _carry_corpse(env)
    env._kill_log = {}
    assert npt._spoiling_now(env) == []


def test_a_never_rotting_corpse_does_not_interrupt(env):
    now = _advance_clock(env)
    _carry_corpse(env, "a lichen corpse")
    env._kill_log = {"lichen": now - (corpse_age.RISKY_AGE - 2)}
    assert npt._spoiling_now(env) == []


def test_the_macro_loop_breaks_when_the_interrupt_fires(env):
    """End-to-end wiring: `run_netplay_skill` must stop the skill and say why.

    `np_explore_level` runs up to 100 game turns inside one agent turn, which
    outlasts the entire edible window of a corpse.
    """
    from nethack_harness.tools.netplay_true import (
        NETPLAY_SKILL_REPOSITORY, run_netplay_skill)

    fired = []

    def _fake_spoiling(core_env):
        fired.append(1)
        return ["newt corpse (e) rots in 2 move(s)"]

    original = npt._spoiling_now
    npt._spoiling_now = _fake_spoiling
    try:
        res = run_netplay_skill(
            env, NETPLAY_SKILL_REPOSITORY.skills["explore_level"], {})
    finally:
        npt._spoiling_now = original

    assert fired, "the macro loop never consulted the spoiling check"
    assert "[INTERRUPTED:" in res.feedback
    assert "rots in 2 move(s)" in res.feedback
    assert classify_tool_result(res.feedback) == "interrupted"


def test_the_real_spoiling_check_fires_inside_a_running_macro(env):
    """The same wiring, but with the REAL `_spoiling_now` doing the deciding.

    Only the corpse is fabricated: each engine step hands back a fresh
    observation from the engine, so the carried corpse is re-injected into
    every observation the check sees (there is no way to make the engine hand
    us one mid-macro). `shape`, `urgent_corpses`, the kill log and the loop
    break are all the production code path.
    """
    from nethack_harness.tools.netplay_true import (
        NETPLAY_SKILL_REPOSITORY, run_netplay_skill)

    now = _advance_clock(env)
    # Five moves of edibility left, so the check has slack for the first few
    # engine steps of the macro before the window closes.
    env._kill_log = {"newt": now - (corpse_age.RISKY_AGE - corpse_age.URGENT_WINDOW)}

    original_view = npt._raw_view

    def _view_with_corpse(core_env):
        view = original_view(core_env)
        if view is not None:
            _carry_corpse_into(view)
        return view

    npt._raw_view = _view_with_corpse
    try:
        res = run_netplay_skill(
            env, NETPLAY_SKILL_REPOSITORY.skills["explore_level"], {})
    finally:
        npt._raw_view = original_view

    assert "[INTERRUPTED:" in res.feedback, res.feedback
    assert "newt corpse" in res.feedback
    assert classify_tool_result(res.feedback) == "interrupted"
    assert npt.swallowed_exceptions() == {}


# ---------------------------------------------------------------------------
# Swallowed exceptions.
# ---------------------------------------------------------------------------

def test_a_swallowed_exception_is_counted_instead_of_vanishing(env, monkeypatch):
    def _boom(_core_env):
        raise RuntimeError("boom")

    monkeypatch.setattr(npt, "_raw_view", _boom)
    npt.reset_swallowed_exceptions()

    assert npt._spoiling_now(env) == []          # still total: no rollout dies
    assert npt.swallowed_exceptions() == {"_spoiling_now:RuntimeError": 1}

    npt._spoiling_now(env)
    assert npt.swallowed_exceptions()["_spoiling_now:RuntimeError"] == 2


def test_a_swallowed_exception_is_logged_the_first_time(env, monkeypatch, caplog):
    def _boom(_core_env):
        raise RuntimeError("boom")

    monkeypatch.setattr(npt, "_raw_view", _boom)
    npt.reset_swallowed_exceptions()
    with caplog.at_level("WARNING", logger=npt.logger.name):
        npt._spoiling_now(env)

    assert any("_spoiling_now" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records), caplog.text
    assert any(r.exc_info for r in caplog.records), "log the traceback, not just the name"


def test_strict_mode_refuses_to_swallow(env, monkeypatch):
    """What a test run wants: a programming error that fails, loudly.

    The exact defect this file exists for -- `shape()` raising AttributeError
    on every single call -- would have been a red test on day one under this
    flag instead of an empty list for months.
    """
    def _boom(_core_env):
        raise RuntimeError("boom")

    monkeypatch.setattr(npt, "_raw_view", _boom)
    monkeypatch.setenv("NETHACK_STRICT_SKILL_ERRORS", "1")
    with pytest.raises(RuntimeError):
        npt._spoiling_now(env)


def test_the_pet_guard_also_reports_when_it_swallows(env, monkeypatch):
    """1.2's peaceful/pet guard is the other place a silent `[]` is dangerous."""
    import nethack_harness.prompt.features as features

    def _boom(_raw):
        raise ValueError("nope")

    monkeypatch.setattr(features, "visible_monsters", _boom)
    npt.reset_swallowed_exceptions()

    assert npt._refuse_attack(env, {"x": 1, "y": 1}) is None
    assert npt.swallowed_exceptions() == {"_refuse_attack:ValueError": 1}


def test_a_skill_called_without_coordinates_is_not_an_error(env):
    """The narrow guard must stay narrow: no x/y is a control path, not a bug."""
    npt.reset_swallowed_exceptions()
    assert npt._refuse_attack(env, {}) is None
    assert npt.swallowed_exceptions() == {}
