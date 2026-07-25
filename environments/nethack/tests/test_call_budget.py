import asyncio, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nethack_v1 as m
import verifiers as vf


def test_budget_refuses_past_the_cap():
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True, max_skill_calls=3,
                                 explicit_seeds=[0],
                                 env_args={"skill_set": "netplay"})
    ts = m.load_taskset(cfg)
    tool = next(t for t in ts.toolsets[0].tools if t.__name__ == "search")
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(ts.toolsets[0].setups[0](None, state))

    for _ in range(3):
        asyncio.run(tool(state, times=1))
    assert state["skill_calls"] == 3

    out = asyncio.run(tool(state, times=1))
    assert "budget" in str(out).lower()
    assert state["skill_calls"] == 3          # refused calls do not count
    assert state["terminated"] is True


def _run_disabled_cap(max_skill_calls, n_calls=5):
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True,
                                 max_skill_calls=max_skill_calls,
                                 explicit_seeds=[0],
                                 env_args={"skill_set": "netplay"})
    ts = m.load_taskset(cfg)
    tool = next(t for t in ts.toolsets[0].tools if t.__name__ == "search")
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(ts.toolsets[0].setups[0](None, state))

    for _ in range(n_calls):
        out = asyncio.run(tool(state, times=1))
        assert "budget" not in str(out).lower()
    return state


def test_zero_budget_disables_the_cap():
    # max_skill_calls <= 0 means "no cap" — an unbounded arm must keep
    # succeeding past any budget a positive value would have enforced, and
    # must never terminate on the budget path.
    state = _run_disabled_cap(0, n_calls=5)
    assert state["skill_calls"] == 5
    assert not state.get("terminated")


def test_negative_budget_disables_the_cap():
    state = _run_disabled_cap(-1, n_calls=5)
    assert state["skill_calls"] == 5
    assert not state.get("terminated")
