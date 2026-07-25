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


def test_env_response_reports_missing_tool_call():
    # Regression test for the _parse_tool_call <-> _apply_tool_call sentinel
    # handoff (m._NO_TOOL_CALL_SENTINEL): when the assistant message carries no
    # tool_calls at all, env_response must still return the pre-refactor
    # "You must call a tool" text, routed through both new methods rather than
    # short-circuited in the single pre-refactor env_response body.
    env, state = _env_and_state()
    messages = [{"role": "assistant", "content": "thinking...", "tool_calls": []}]
    result = asyncio.run(env.env_response(messages, state))
    content = getattr(result[0], "content", None)
    if content is None:
        content = result[0]["content"]
    text = m.content_to_text(content) if hasattr(m, "content_to_text") else str(content)
    expected_tools = [s for s in m.list_skills() if s not in ("menu_option", "inventory_item")]
    assert text == "You must call a tool. Available tools: " + ", ".join(expected_tools)
