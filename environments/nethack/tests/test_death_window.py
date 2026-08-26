"""Fix1 — the one-turn post-death rollback window (E15 P3).

In P3 r1, `is_completed` returned True the instant `terminated` was set, so
the rollout ended before the model ever saw "You die..." — the post-death
revive path in `_apply_tool_call_inner` and the rendered "ONE action still
works: rollback(n)" banner were dead code in all four deaths. The fix defers
completion by exactly one LM turn when (a) the character DIED (not ascended /
truncated) and (b) `rollback` is published: the death observation reaches the
model, and a rollback that restores HP > 0 clears `died`/`terminated` and
the run genuinely resumes. Any other post-death call ends the rollout on the
next `is_completed` check.

Deterministic recipe: live calls accumulate snapshots, then a mid-run
`modify(hp=0)` poke (same hook test_death_termination.py uses at setup)
kills the character with no death message and no NLE terminated flag.
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack as m
import verifiers as vf

_P3_SKILLS = "np_core,request_map,search,rollback"


def _env_and_state(skill_set=_P3_SKILLS):
    env = m.load_environment(
        task_spec="full_nle", skill_set=skill_set,
        n_examples=1, explicit_seeds=[0],
        character="Val-hum-neu-fem",
    )
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    # The real rollout loop maintains this; is_completed's max_turns branch
    # reads it after the terminated branch stops short-circuiting (revive).
    state.setdefault("trajectory", [])
    return env, state


def _die_after_live_turns(env, state, live_calls=3):
    for _ in range(live_calls):
        asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    assert not state.get("died"), "setup: character must be alive pre-poke"
    state["env"].modify(hp=0)
    return asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))


def test_death_defers_completion_by_one_turn():
    env, state = _env_and_state()
    content = _die_after_live_turns(env, state)
    assert state["died"] is True and state["terminated"] is True
    # The window: first is_completed check returns False exactly once.
    assert asyncio.run(env.is_completed(state)) is False
    assert state["_death_window_spent"] is True
    # The death observation the model receives names the killing call and
    # offers rollback.
    text = str(content)
    assert "GAME OVER" in text
    assert "after calling search" in text
    assert "rollback" in text


def test_post_death_rollback_revives_and_run_continues():
    env, state = _env_and_state()
    _die_after_live_turns(env, state)
    assert asyncio.run(env.is_completed(state)) is False  # window opens
    asyncio.run(env._apply_tool_call(state, "rollback", {"n": 2}))
    assert state["died"] is False
    assert state["terminated"] is False
    assert (state["structured_obs"].status or {}).get("hitpoints", 0) > 0
    # Window re-armed for a later death; run continues.
    assert "_death_window_spent" not in state
    assert asyncio.run(env.is_completed(state)) is False


def test_post_death_non_rollback_call_ends_the_rollout():
    env, state = _env_and_state()
    _die_after_live_turns(env, state)
    assert asyncio.run(env.is_completed(state)) is False  # window opens
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))  # refused
    assert state["died"] is True
    assert asyncio.run(env.is_completed(state)) is True  # window spent -> over


def test_no_window_without_rollback_published():
    env, state = _env_and_state(skill_set="netplay")
    for _ in range(2):
        asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    state["env"].modify(hp=0)
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    assert state["died"] is True
    # Base-style arms terminate immediately, exactly as before the fix.
    assert asyncio.run(env.is_completed(state)) is True
