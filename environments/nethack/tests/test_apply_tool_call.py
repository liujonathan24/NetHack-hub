import asyncio, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nethack as m
import verifiers as vf


def _env_and_state():
    env = m.load_environment(task_spec="full_nle", skill_set="netplay",
                             n_examples=1, explicit_seeds=[0],
                             character="Val-hum-neu-fem")
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    return env, state


def test_apply_tool_call_returns_rendered_observation():
    env, state = _env_and_state()
    content = asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    text = m.content_to_text(content) if hasattr(m, "content_to_text") else str(content)
    assert "=== STATUS ===" in text


def test_gate_still_rejects_withheld_move_through_the_seam():
    env, state = _env_and_state()
    content = asyncio.run(env._apply_tool_call(state, "move", {"direction": "N"}))
    text = m.content_to_text(content) if hasattr(m, "content_to_text") else str(content)
    assert "is not available" in text
