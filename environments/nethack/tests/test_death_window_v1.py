"""Fix1 — the post-death rollback window at the V1 TASKSET layer (E15 P3).

The fix1 relaunch proved the v0-level `is_completed` deferral is inert for
the CLI harness: every p3_rollback_fix1 rollout recorded
``stop_condition="game_over"`` — the ``@vf.stop game_over`` in nethack_v1.py,
which fired on ``trace.state.terminated`` before the next LM call, so the
death observation never entered the message list (trace node tails end on
the pre-death tool result). Two v1-side defects had to fall:

  * ``_publish`` latched ``terminated`` (``or state.terminated``), so a
    revive that cleared the v0 flag stayed terminated at this layer forever;
  * ``game_over`` had no window: it stopped on the very first check.

These tests drive the REAL v1 toolset (engine included) through death,
window, revive, and the non-rollback termination path, evaluating the actual
``NetHackTask.game_over`` stop exactly as v1/session.py's ``refused`` does.
"""

import asyncio
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack_v1 as m

_P3_SKILLS = "np_core,request_map,search,rollback"


def _toolset(skill_set=_P3_SKILLS):
    cfg = m.NetHackTasksetConfig(
        task_spec="full_nle",
        n_examples=1,
        explicit_seeds=[0],
        character="Val-hum-neu-fem",
        env_args={"skill_set": skill_set},
    )
    task = m.load_taskset(cfg).select(1)[0]
    toolset = task.tool_servers()[0]
    asyncio.run(toolset.setup_task(task.data))
    return toolset


def _stop(toolset) -> bool:
    trace = SimpleNamespace(state=toolset.state)
    return asyncio.run(m.NetHackTask.game_over(None, trace))


def _die(toolset, live_calls=3):
    tools = toolset.tool_functions()
    for _ in range(live_calls):
        asyncio.run(tools["search"](times=1))
    assert not toolset.state.died, "setup: character must be alive pre-poke"
    toolset.v0_state["env"].modify(hp=0)
    return asyncio.run(tools["search"](times=1))


def test_v1_stop_defers_once_and_death_obs_names_the_killer():
    toolset = _toolset()
    content = _die(toolset)
    s = toolset.state
    assert s.died and s.terminated and s.rollback_published
    assert not s.death_window_spent
    # The stop — evaluated exactly as session.refused does — must defer.
    assert _stop(toolset) is False
    # And the fatal call's returned observation is the death screen the next
    # LM call will carry: banner, attribution, rollback affordance.
    text = str(content)
    assert "GAME OVER" in text
    assert "after calling search" in text
    assert "rollback" in text


def test_v1_post_death_rollback_revives_through_the_stop():
    toolset = _toolset()
    _die(toolset)
    assert _stop(toolset) is False  # window open
    tools = toolset.tool_functions()
    asyncio.run(tools["rollback"](n=2))
    s = toolset.state
    assert s.died is False
    assert s.terminated is False, (
        "the old `or state.terminated` latch would keep this True forever"
    )
    assert _stop(toolset) is False  # run genuinely continues
    # And the engine is really live again: a further call steps the game.
    asyncio.run(tools["search"](times=1))
    assert toolset.state.died is False


def test_v1_post_death_non_rollback_ends_on_next_check():
    toolset = _toolset()
    _die(toolset)
    assert _stop(toolset) is False  # window open
    tools = toolset.tool_functions()
    asyncio.run(tools["search"](times=1))  # refused by the death guard
    s = toolset.state
    assert s.died and s.terminated and s.death_window_spent
    assert _stop(toolset) is True  # window spent -> rollout over


def test_v1_no_window_when_rollback_unpublished():
    toolset = _toolset(skill_set="netplay")
    tools = toolset.tool_functions()
    for _ in range(2):
        asyncio.run(tools["search"](times=1))
    toolset.v0_state["env"].modify(hp=0)
    asyncio.run(tools["search"](times=1))
    s = toolset.state
    assert s.died and s.terminated and not s.rollback_published
    assert _stop(toolset) is True  # terminates immediately, as before


def test_v1_budget_termination_still_latches():
    # The `or state.budget_exhausted` half of the latch fix: budget-driven
    # termination is v1-only state and must survive every later _publish.
    toolset = _toolset()
    toolset.state.budget_exhausted = True
    toolset.state.terminated = True
    tools = toolset.tool_functions()
    asyncio.run(tools["search"](times=1))  # "[The episode is over.]"
    assert toolset.state.terminated is True
