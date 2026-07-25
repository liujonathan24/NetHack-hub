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
