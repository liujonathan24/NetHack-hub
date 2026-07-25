"""Tests for the self-dispatching toolset mode (``self_dispatch`` / ``obs_mode``).

Today ``_build_toolset`` wraps schema-only v0 tool adapters: they carry the JSON
schema (name/signature/docstring) but do NOT touch the engine — dispatch lives
in ``env_response``. External CLI harnesses (Codex, Prime Agent) drive the game
over MCP and expect the result back from the tool call itself, so
``self_dispatch=True`` wraps each adapter to actually execute the skill via
``NetHackVerifiersEnv._apply_tool_call`` and return the observation.

``self_dispatch=False`` (the default) must remain byte-identical to today's
harness-driven behavior — this is what ``test_v1_taskset.py`` guards.
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack_v1 as m
import verifiers as vf


def test_default_toolset_is_harness_driven():
    ts = m.load_taskset({"task_spec": "full_nle", "n_examples": 1})
    assert ts.toolsets[0].scope == "rollout"


def test_self_dispatch_tools_are_wrapped():
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True,
                                 env_args={"skill_set": "netplay"})
    ts = m.load_taskset(cfg)
    names = {t.__name__ for t in ts.toolsets[0].tools}
    assert "explore_and_descend" in names
    assert "move" not in names          # netplay withholds it


def test_self_dispatch_and_obs_mode_default():
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1)
    assert cfg.self_dispatch is False
    assert cfg.obs_mode == "push"

    # The default config must not wrap the tools: no `_self_dispatching`
    # wrapper means no `__wrapped__` attribute (functools.wraps is what sets
    # it), so this is a real signal the schema-only adapters pass through
    # untouched — i.e. arm 0 stays byte-identical to today.
    ts = m.load_taskset(cfg)
    assert all(not hasattr(t, "__wrapped__") for t in ts.toolsets[0].tools)


def test_self_dispatch_tool_signature_exposes_state_for_runtime_injection():
    # The v1 Runtime's generic tool-calling path (used when an external CLI
    # harness drives these tools over MCP, e.g. Codex/Prime Agent — Tasks 9/10)
    # decides whether to inject `state` by checking
    # `"state" in inspect.signature(tool).parameters`. `functools.wraps` sets
    # `__wrapped__`, which makes plain `inspect.signature` resolution follow
    # straight through to the original schema-only adapter's signature (no
    # `state` param) unless `_self_dispatching` overrides `__signature__`
    # explicitly. Assert on the introspected signature, not on `__wrapped__`
    # presence/absence — a test on `__wrapped__` would pass even if this
    # regressed.
    import inspect

    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True,
                                 env_args={"skill_set": "netplay"})
    ts = m.load_taskset(cfg)
    wrapped = next(t for t in ts.toolsets[0].tools if t.__name__ == "search")
    assert "state" in inspect.signature(wrapped).parameters


def test_self_dispatch_tool_actually_executes_and_returns_observation():
    # The whole point: unlike the schema-only v0 adapters (which raise if
    # called directly), a self_dispatch-wrapped tool must run the skill
    # against the engine and hand back the rendered observation from the call
    # itself, since MCP-driven CLI harnesses never see `env_response`.
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True,
                                 explicit_seeds=[0],
                                 env_args={"skill_set": "netplay"})
    ts = m.load_taskset(cfg)
    toolset = ts.toolsets[0]
    search = next(t for t in toolset.tools if t.__name__ == "search")

    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(toolset.setups[0](None, state))

    result = asyncio.run(search(state, times=1))
    text = result if isinstance(result, str) else str(result)
    assert "=== STATUS ===" in text


def test_terse_drops_the_map_but_keeps_messages_and_feedback():
    from nethack_v1 import _terse
    text = "=== MAP ===\n#####\n\n=== MESSAGES ===\nYou hit it.\n[search: nothing found]"
    out = _terse(text)
    assert "#####" not in out
    assert "You hit it." in out
    assert "[search: nothing found]" in out
