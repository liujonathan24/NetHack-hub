"""Fix1 — the descent gate fires BEFORE the skill executes (E15 P1).

In P1 r1 the gate check sat after the registry dispatch, but the netplay
skills are closed-loop: `run_netplay_skill` steps the engine inside the
dispatch, so all 30/30 gate lines were served AFTER the descent had already
happened, and the instructed "repeat" '>' landed on the NEW level ("You
can't go down here"). These tests pin the fixed ordering with a live engine:
a gated descent consumes zero engine steps, and the gate only fires when the
character actually lags the norm (r1 fired at deficit 0 too, teaching the
model the interrupt class was ignorable noise).
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack as m
import verifiers as vf


def _env_and_state(**kw):
    env = m.load_environment(
        task_spec="full_nle", skill_set="np_core,request_map,search",
        n_examples=1, explicit_seeds=[0],
        character="Val-hum-neu-fem", **kw,
    )
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    state.setdefault("trajectory", [])
    return env, state


def _status(state):
    return state["structured_obs"].status or {}


def test_gated_descent_consumes_no_engine_step():
    env, state = _env_and_state(descent_gate="norm")
    # XL 1 on Dlvl 4: deficit vs the leaving norm (3) -> the gate must fire.
    state["env"].modify(goto_depth=4)
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    depth0 = _status(state).get("depth")
    time0 = _status(state).get("time")
    assert depth0 == 4

    content = asyncio.run(
        env._apply_tool_call(state, "np_press_key", {"key": ">"}))
    text = str(content)
    assert "[descent check:" in text
    # THE fix: the gated call must not have stepped the engine — same depth,
    # same clock, character still standing where it was.
    assert _status(state).get("depth") == depth0
    assert _status(state).get("time") == time0


def test_repeat_passes_the_gate():
    env, state = _env_and_state(descent_gate="norm")
    state["env"].modify(goto_depth=4)
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    asyncio.run(env._apply_tool_call(state, "np_press_key", {"key": ">"}))
    content = asyncio.run(
        env._apply_tool_call(state, "np_press_key", {"key": ">"}))
    # Acked depth: the repeat reaches the engine (no second gate line). The
    # descent itself may still fail off-stairs — that is the game's verdict,
    # not the gate's.
    assert "[descent check:" not in str(content)


def test_no_firing_at_or_above_norm():
    env, state = _env_and_state(descent_gate="norm")
    # XL 1 on Dlvl 1: leaving norm is 1, deficit 0 -> fix1 stays silent
    # (r1 fired here, and that dismissable first exposure anchored the model
    # into ignoring every later, real warning).
    content = asyncio.run(
        env._apply_tool_call(state, "np_press_key", {"key": ">"}))
    assert "[descent check:" not in str(content)


def test_enforce_blocks_before_the_engine_steps():
    env, state = _env_and_state(descent_gate="enforce")
    state["env"].modify(goto_depth=4)
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    time0 = _status(state).get("time")
    content = asyncio.run(
        env._apply_tool_call(state, "np_press_key", {"key": ">"}))
    assert "[descent BLOCKED:" in str(content)
    assert _status(state).get("depth") == 4
    assert _status(state).get("time") == time0
