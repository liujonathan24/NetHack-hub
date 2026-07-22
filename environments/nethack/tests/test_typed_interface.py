"""Typed Action layer (moved from the engine during the coupling decouple).

action_spec() derives from the skill registry; TypedNetHackInterface steps a
typed Action via skill dispatch and inherits the engine's raw RawAction path.
Drives a real NetHackCoreEnv (slow, that's fine)."""
from __future__ import annotations

from nethack_interface import Observation, RawAction  # engine raw substrate
from nethack_harness.interface import Action, action_spec, TypedNetHackInterface


def _make_core_env():
    from nethack_core import NetHackCoreEnv

    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(core=42, disp=42)
    return env


def test_action_spec_derives_core_actions_from_registry():
    spec = action_spec()
    for name in ("move", "move_to"):
        assert name in spec
    assert isinstance(spec["move_to"], dict)  # the registry schema


def test_typed_action_and_raw_action_dataclasses():
    a = Action("move_to", {"x": 5, "y": 9})
    assert a.name == "move_to" and a.args == {"x": 5, "y": 9}
    assert RawAction(12).index == 12


def test_typed_step_via_skill_dispatch_and_raw():
    iface = TypedNetHackInterface(_make_core_env())
    obs = iface.reset()
    assert isinstance(obs, Observation)

    obs2, reward, done, info = iface.step(Action("search"))
    assert isinstance(obs2, Observation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "feedback" in info

    obs3, *_ = iface.step(RawAction(0))  # inherited raw escape hatch
    assert isinstance(obs3, Observation)
