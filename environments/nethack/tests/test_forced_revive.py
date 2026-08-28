"""Fix2 — FORCED ROLLBACK ON DEATH (E15 P3, user-specified design).

When `rollback` is published and the character dies, the env itself restores
the most recent live snapshot — no model choice, no reliance on any pipeline
layer delivering a post-death turn (P3 r1 and the fix1 one-turn window both
foundered on exactly that delivery: every death recorded
stop_condition="game_over" with the death observation never entering the
message list). The next observation opens with
"[You died: <cause> -- after calling <skill>(<args>). The game has been
rewound to game turn T. (death N/3 on this dungeon level ...)]".
The THIRD death on the same dungeon level is final. This file is the
mandatory simulated-death gate that must pass before any paid rollouts.

Driven at BOTH layers: the v0 env (engine + state flags + rendered note) and
the v1 taskset (typed state + the game_over stop exactly as v1/session.py
evaluates it).
"""

import asyncio
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack as m0
import nethack_v1 as m1
import verifiers as vf

_P3_SKILLS = "np_core,request_map,search,rollback"


# ------------------------------- v0 layer ---------------------------------- #

def _v0_env_and_state(skill_set=_P3_SKILLS):
    env = m0.load_environment(
        task_spec="full_nle", skill_set=skill_set,
        n_examples=1, explicit_seeds=[0],
        character="Val-hum-neu-fem",
    )
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    state.setdefault("trajectory", [])
    return env, state


def _call(env, state, name="search", **kw):
    return asyncio.run(env._apply_tool_call(state, name, kw or {"times": 1}))


def _kill(env, state):
    """Scripted death: poke hp to 0 mid-run, then make a call."""
    state["env"].modify(hp=0)
    return _call(env, state)


def _clock(state):
    return (state["structured_obs"].status or {}).get("time")


def test_death_one_auto_revives_with_cause_attribution_and_counter():
    env, state = _v0_env_and_state()
    for _ in range(3):
        _call(env, state)  # live turns -> live snapshots
    t_alive = _clock(state)
    content = _kill(env, state)
    text = str(content)
    # (a) auto-revive: the rollout state is ALIVE after the fatal call.
    assert state["died"] is False and not state.get("terminated")
    assert (state["structured_obs"].status or {}).get("hitpoints", 0) > 0
    # The injected line: death fact, attribution, rewind clock, counter.
    assert "[You died" in text
    assert "after calling search" in text
    assert "The game has been rewound to game turn" in text
    assert "(death 1/3 on this dungeon level" in text
    # Clock rewound: restored time is not ahead of the last live time.
    assert _clock(state) is not None and _clock(state) <= t_alive
    # And the run continues: a further call steps a live engine.
    _call(env, state)
    assert state["died"] is False


def test_death_two_same_level_counts_two_of_three():
    env, state = _v0_env_and_state()
    for _ in range(3):
        _call(env, state)
    _kill(env, state)                     # death 1 -> revived
    text = str(_kill(env, state))         # death 2, same level
    assert state["died"] is False
    assert "(death 2/3 on this dungeon level" in text


def test_death_three_same_level_is_final_game_over():
    env, state = _v0_env_and_state()
    for _ in range(3):
        _call(env, state)
    _kill(env, state)                     # 1/3
    _kill(env, state)                     # 2/3
    content = _kill(env, state)           # 3/3 -> final
    text = str(content)
    assert state["died"] is True
    assert state["terminated"] is True
    assert "death 3/3 on this dungeon level; the game is over for good" in text
    assert "GAME OVER" in text
    # Every stop layer fires immediately: no stray fix1-window deferral.
    assert state.get("_death_window_spent") is True
    assert asyncio.run(env.is_completed(state)) is True
    # And post-mortem calls are refused without touching the engine.
    _call(env, state)
    assert state["died"] is True


def test_normal_rollout_unaffected():
    env, state = _v0_env_and_state()
    for _ in range(3):
        _call(env, state)
    assert state["died"] is False
    assert not state.get("terminated")
    assert "_death_counts" not in state or not state["_death_counts"]
    assert asyncio.run(env.is_completed(state)) is False


def test_no_forced_revive_without_rollback_published():
    env, state = _v0_env_and_state(skill_set="netplay")
    for _ in range(2):
        _call(env, state)
    _kill(env, state)
    # Base-style arms die exactly as before: no revive, immediate game over.
    assert state["died"] is True and state["terminated"] is True
    assert asyncio.run(env.is_completed(state)) is True


def test_deaths_on_different_levels_have_separate_counters():
    env, state = _v0_env_and_state()
    for _ in range(2):
        _call(env, state)
    _kill(env, state)                     # death 1 on Dlvl 1 -> revived
    _kill(env, state)                     # death 2 on Dlvl 1 -> revived
    state["env"].modify(goto_depth=2)
    _call(env, state)                     # live turn on Dlvl 2 (fresh snapshot)
    text = str(_kill(env, state))         # first death on Dlvl 2
    assert state["died"] is False
    assert "(death 1/3 on this dungeon level" in text


# ------------------------------- v1 layer ---------------------------------- #

def _v1_toolset(skill_set=_P3_SKILLS):
    cfg = m1.NetHackTasksetConfig(
        task_spec="full_nle",
        n_examples=1,
        explicit_seeds=[0],
        character="Val-hum-neu-fem",
        env_args={"skill_set": skill_set},
    )
    task = m1.load_taskset(cfg).select(1)[0]
    toolset = task.tool_servers()[0]
    asyncio.run(toolset.setup_task(task.data))
    return toolset


def _v1_stop(toolset) -> bool:
    trace = SimpleNamespace(state=toolset.state)
    return asyncio.run(m1.NetHackTask.game_over(None, trace))


def test_v1_pipeline_never_stops_until_third_death():
    toolset = _v1_toolset()
    tools = toolset.tool_functions()
    for _ in range(3):
        asyncio.run(tools["search"](times=1))
    for expected in (1, 2):
        toolset.v0_state["env"].modify(hp=0)
        text = str(asyncio.run(tools["search"](times=1)))
        assert f"(death {expected}/3 on this dungeon level" in text
        s = toolset.state
        assert s.died is False and s.terminated is False
        assert _v1_stop(toolset) is False, (
            "the pipeline must not see a terminated state on a revived death"
        )
    toolset.v0_state["env"].modify(hp=0)
    text = str(asyncio.run(tools["search"](times=1)))
    assert "death 3/3" in text
    s = toolset.state
    assert s.died is True and s.terminated is True
    assert _v1_stop(toolset) is True  # third death: game_over fires at once


# --------------------- deaths_per_level dose arm (cap=5) -------------------- #

def test_cap5_deaths_one_through_four_revive_with_n_of_5_lines():
    env, state = _v0_env_and_state()
    env.deaths_per_level = 5  # tier contract: [p3_revive_d5.contract] deaths_per_level = 5
    for _ in range(3):
        _call(env, state)
    for k in (1, 2, 3, 4):
        text = str(_kill(env, state))
        assert state["died"] is False, f"death {k} should revive under cap 5"
        assert f"(death {k}/5 on this dungeon level -- the fifth is final)" in text
        _call(env, state)  # push a fresh live snapshot between deaths
    assert asyncio.run(env.is_completed(state)) is False


def test_cap5_death_five_is_final_game_over():
    env, state = _v0_env_and_state()
    env.deaths_per_level = 5
    for _ in range(3):
        _call(env, state)
    for _ in range(4):
        _kill(env, state)
        _call(env, state)
    text = str(_kill(env, state))         # 5/5 -> final
    assert state["died"] is True and state["terminated"] is True
    assert "death 5/5 on this dungeon level; the game is over for good" in text
    assert asyncio.run(env.is_completed(state)) is True


def test_cap_default_is_three_and_ctor_parses_strings():
    env, state = _v0_env_and_state()
    assert env.deaths_per_level == 3      # default: fix2 arm unchanged
    env2 = m0.load_environment(
        task_spec="full_nle", skill_set=_P3_SKILLS, n_examples=1,
        explicit_seeds=[0], character="Val-hum-neu-fem", deaths_per_level="5",
    )
    assert env2.deaths_per_level == 5     # CLI strings coerce
