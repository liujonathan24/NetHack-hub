"""Death must be announced on the turn it happens.

DEFECT (docs/HARNESS_DEFECTS.md 3.6). The death turn rendered `HP: 0/N`, a
normal-looking MAP and whatever the HINT ladder produced -- observed:
"Hostile adjacent (SE). Call `attack(...)` — your HP is healthy." on a corpse.
Nothing said the character was dead. `state["died"]` is set at the BOTTOM of
`env_response`, after this render, and the "[Your character is dead...]" note is
appended by the NEXT `_apply_tool_call` -- at which point every tool is gated
and the one action that still works (`rollback`) has never been mentioned.

So the observation renderer announces it, from the observation itself rather
than from harness bookkeeping, so the ordering inside `env_response` cannot
matter. See `rendering._game_over_block`.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import verifiers as vf

from nethack_core.observations import StructuredObservation
from nethack_harness.prompt.rendering import (
    PUBLISHED_TOOLS_STATE_KEY, format_observation_as_chat,
)


def _structured(hp, messages=(), hp_max=16):
    return StructuredObservation(
        map_view="", messages=list(messages), inventory=[],
        status={"x": 10, "y": 3, "hitpoints": hp, "max_hitpoints": hp_max,
                "armor_class": 6, "depth": 3, "time": 812,
                "experience_level": 2, "gold": 0, "hunger_state": 1},
        # A hostile one step SE, so the HINT ladder has something to say. On a
        # live death turn it said exactly that, next to `HP: 0/16`.
        character={}, adjacent={"SE": "j"},
    )


def _render(structured, state=None):
    return format_observation_as_chat(
        structured, None, state=state if state is not None else {},
        compact=False, include_map=False,
    )


def test_zero_hp_is_announced_in_the_observation():
    out = _render(_structured(hp=0))
    assert "=== GAME OVER ===" in out, out
    assert "DEAD" in out, out
    assert "HP 0/16" in out, out
    assert "Dlvl 3" in out and "turn 812" in out, out


def test_the_death_turn_does_not_also_recommend_an_attack():
    """The bug's most visible face: dead, and told to fight."""
    alive = _render(_structured(hp=12))
    assert "=== HINT ===" in alive, "fixture is vacuous: a live turn must hint"
    assert "attack" in alive

    dead = _render(_structured(hp=0))
    assert "=== HINT ===" not in dead, dead


def test_game_over_is_the_first_thing_in_the_observation():
    out = _render(_structured(hp=0))
    assert out.lstrip().startswith("=== GAME OVER ==="), out[:200]


def test_the_cause_of_death_is_quoted_from_the_game():
    out = _render(_structured(hp=0, messages=["You die...", "Killed by a jackal"]))
    assert "Killed by a jackal" in out, out


def test_rollback_is_offered_only_when_it_is_published():
    without = _render(_structured(hp=0))
    assert "rollback" not in without, without

    with_rb = _render(_structured(hp=0),
                      state={PUBLISHED_TOOLS_STATE_KEY: {"rollback", "np_look"}})
    assert "`rollback(n)`" in with_rb, with_rb


def test_death_message_without_zero_hp_still_announces():
    """NetHack's death screens do not always leave `hitpoints == 0` behind."""
    out = _render(_structured(hp=4, messages=[
        "Do you want your possessions identified? [ynq] (n)"]))
    assert "=== GAME OVER ===" in out, out


def test_ascension_is_not_reported_as_death():
    out = _render(_structured(hp=4, messages=["You ascended to demigoddess!"]))
    assert "GAME OVER" not in out, out


def test_live_death_turn_announces_before_the_harness_knows():
    """End-to-end on the real engine.

    `modify={"hp": 0}` is the same deterministic kill `test_death_termination`
    uses. The assertion that matters is that the content returned BY THE DEATH
    TURN ITSELF says so -- previously the agent only found out on the next call,
    from a refusal.
    """
    import nethack as m

    env = m.load_environment(
        task_spec="full_nle", skill_set="netplay_true,reveal,rollback",
        n_examples=1, explicit_seeds=[0], character="Val-hum-neu-fem",
        modify={"hp": 0},
    )
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    assert not state.get("died"), "fixture: the harness must not know yet"

    content = asyncio.run(env._apply_tool_call(state, "np_look", {}))
    text = content if isinstance(content, str) else str(content)

    assert "=== GAME OVER ===" in text, text[:1500]
    assert "REFUSED" in text, text[:1500]
    assert "`rollback(n)`" in text, text[:1500]
    assert "=== HINT ===" not in text, text[:1500]
