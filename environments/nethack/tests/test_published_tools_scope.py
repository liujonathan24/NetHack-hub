"""The published-tool set must be per-render, never a process global.

DEFECT (docs/HARNESS_DEFECTS.md 3.7). `render_system_prompt(published_tools=...)`
wrote a module-level `rendering._PUBLISHED_TOOLS`, and `_fix_hint_vocabulary`
read it to DELETE hint sentences naming unbound tools (`search` and
`engrave_elbereth` are bound to nothing under `netplay_true`). Nothing ever put
it back. So *constructing an environment changed how every later render in the
same process behaved* -- including renders belonging to a different game, a
different skill set, or a hand-built synthetic observation.

The observable symptom was an order-dependent test suite:
`test_hint_actionability.py::test_exit_hint_rotates_after_no_progress_then_
recommends_search` passed alone and failed after `test_golden_obs.py`, because
booting the golden matrix's environments deleted the `search` sentence that test
asserts on. `tests/golden/obs_configs.py` carried a save/restore workaround.

These tests pin the fix from both ends: `render_system_prompt` is pure, and the
renderer takes its published set from per-rollout state.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nethack_core.observations import StructuredObservation
from nethack_harness.prompt import rendering
from nethack_harness.prompt.rendering import (
    PUBLISHED_TOOLS_STATE_KEY,
    _fix_hint_vocabulary,
    format_observation_as_chat,
    published_tools_for,
    render_system_prompt,
)

# The netplay_true surface: `search` / `engrave_elbereth` / `attack` / `kick`
# are NOT in it, `np_look` and `np_down` are.
_NETPLAY_TRUE_ISH = {"np_look", "np_down", "np_move_to", "np_kick", "rollback"}

_SEARCH_HINT = (
    "Every visible exit from this area has already been tried without "
    "progress. Call `search(times=10)` for a hidden passage instead of "
    "retrying a known exit."
)


def test_the_module_global_is_gone():
    """The leak was the global's existence. Nothing may reintroduce it."""
    assert not hasattr(rendering, "_PUBLISHED_TOOLS"), (
        "rendering._PUBLISHED_TOOLS is back. The published-tool set must travel "
        "in per-rollout state (rendering.PUBLISHED_TOOLS_STATE_KEY), not in a "
        "module global that no caller restores -- see this file's docstring."
    )


def test_fix_hint_vocabulary_takes_the_set_explicitly():
    """Unknown publisher -> leave the text alone; known -> rewrite it."""
    assert _fix_hint_vocabulary(_SEARCH_HINT, set()) == _SEARCH_HINT
    assert _fix_hint_vocabulary(_SEARCH_HINT, None) == _SEARCH_HINT

    trimmed = _fix_hint_vocabulary(_SEARCH_HINT, _NETPLAY_TRUE_ISH)
    assert "search" not in trimmed, trimmed
    assert "already been tried" in trimmed, trimmed

    renamed = _fix_hint_vocabulary("You are on stairs down. Call `descend` now.",
                                   _NETPLAY_TRUE_ISH)
    assert "`np_down`" in renamed, renamed


def test_render_system_prompt_is_pure():
    """Assembling a prompt must not reprogram the observation renderer."""
    before = _fix_hint_vocabulary(_SEARCH_HINT, published_tools_for({}))
    render_system_prompt(published_tools=_NETPLAY_TRUE_ISH)
    after = _fix_hint_vocabulary(_SEARCH_HINT, published_tools_for({}))
    assert before == after == _SEARCH_HINT, (
        "render_system_prompt leaked its published_tools into later renders"
    )


def _room_chars():
    """One room with two open doorways, at (5,1) and (14,1).

    Same fixture as `test_hint_actionability._room_chars` -- deliberately, since
    that is the test the leak used to break.
    """
    import numpy as np

    rows = [" " * 20]
    rows.append("-" * 5 + "." + "-" * 8 + "." + "-" * 5)
    rows.extend(["|" + "." * 18 + "|"] * 4)
    rows.append("-" * 20)
    rows.append(" " * 20)
    return np.array([[ord(c) for c in row] for row in rows], dtype=np.int16)


def _structured():
    return StructuredObservation(
        map_view="", messages=[], inventory=[],
        status={"x": 10, "y": 3, "hitpoints": 20, "max_hitpoints": 20,
                "armor_class": 6, "depth": 1, "time": 5,
                "experience_level": 1, "gold": 0, "hunger_state": 1},
        character={}, adjacent={},
    )


def _search_hint(state=None) -> str:
    """Drive the exit ladder until it recommends `search`, return that hint.

    Both exits get recommended three turns each and then marked tried, at which
    point the ladder falls through to the `search(times=10)` advice -- the one
    sentence `_fix_hint_vocabulary` deletes when `search` is unbound. Renders
    seven times against a fresh state so the ladder state is self-contained.
    """
    import types

    st = dict(state or {})
    st["raw_obs"] = types.SimpleNamespace(chars=_room_chars())
    hint = ""
    for _ in range(7):
        rendered = format_observation_as_chat(
            _structured(), None, state=st, compact=False, include_map=False,
        )
        hint = ""
        for line in rendered.splitlines():
            if line.startswith("=== HINT ==="):
                hint = line
    return hint


def test_state_selects_the_vocabulary_not_a_global():
    """Two renders, two skill sets, same process -- neither sees the other."""
    # Hand the ladder a pre-made hint by seeding the repeat memo it reads.
    unaware = {"_last_hint": _SEARCH_HINT, "_hint_repeats": 0}
    aware = {"_last_hint": _SEARCH_HINT, "_hint_repeats": 0,
             PUBLISHED_TOOLS_STATE_KEY: _NETPLAY_TRUE_ISH}
    assert published_tools_for(unaware) == set()
    assert published_tools_for(aware) == _NETPLAY_TRUE_ISH
    # And the reverse order, to prove neither call leaves residue behind.
    assert published_tools_for(aware) == _NETPLAY_TRUE_ISH
    assert published_tools_for(unaware) == set()


@pytest.mark.skipif(os.environ.get("NETHACK_SKIP_ENGINE") == "1",
                    reason="engine-backed")
def test_booting_an_environment_does_not_change_a_later_render():
    """The exact order-dependence that made the suite flaky.

    `load_environment` resolves the skill set and renders the system prompt from
    it -- the call that used to poison the process. A synthetic render performed
    afterwards must be byte-identical to one performed before.
    """
    import nethack as m

    before = _search_hint()
    assert "search" in before, (
        f"fixture is vacuous: the hint under test must name `search`: {before!r}"
    )

    m.load_environment(
        variant="B0", skill_set="netplay_true,reveal,rollback",
        max_turns=5, explicit_seeds=[0], n_examples=1,
        character="Val-hum-neu-fem", compact_obs=False,
    )

    after = _search_hint()
    assert before == after, (
        "constructing an environment changed an unrelated render:\n"
        f"  before: {before!r}\n  after:  {after!r}"
    )


def test_a_render_that_declares_its_tools_still_gets_the_rewrite():
    """The production behaviour the leak was accidentally providing.

    Under `netplay_true` nothing binds `search`, so the sentence must go -- but
    because THIS render said so, not because some earlier one did.
    """
    hint = _search_hint({PUBLISHED_TOOLS_STATE_KEY: _NETPLAY_TRUE_ISH})
    assert "search" not in hint, hint
    assert "already been tried" in hint, hint
