"""Tests for the MCP toolset (``self_dispatch`` / ``obs_mode`` / the netplay gate).

The v0 tool adapters carry the JSON schema (name/signature/docstring) but do NOT
touch the engine — in v0, dispatch lives in ``env_response``. External CLI
harnesses (Claude Code, Prime Agent) drive the game over MCP and expect the
result back from the tool call itself, so :class:`nethack_v1.NetHackToolset`
binds each adapter to ``NetHackVerifiersEnv._apply_tool_call`` and returns the
observation.

Under verifiers 0.2.x a toolset IS an MCP server (`v1/mcp/server.py`), so these
tests exercise the real seams: what ``_register`` publishes (the netplay gate),
what ``_with_state`` advertises to FastMCP, and how a tool reads the rollout
state. The harness-driven (``self_dispatch=False``) arm has no v1 path in 0.2.x
— it is the v0 legacy bridge — and the toolset refuses to pretend otherwise.
"""

import asyncio
import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack_v1 as m


class _Recorder:
    """Stands in for FastMCP: records exactly what ``_register`` publishes."""

    def __init__(self):
        self.tools = {}

    def add_tool(self, fn, name=None, description=None):
        self.tools[name or fn.__name__] = fn


def _toolset(**cfg_kwargs):
    cfg = m.NetHackTasksetConfig(
        task_spec="full_nle",
        n_examples=1,
        explicit_seeds=[0],
        env_args={"skill_set": "netplay"},
        **cfg_kwargs,
    )
    task = m.load_taskset(cfg).select(1)[0]
    toolset = task.tool_servers()[0]
    asyncio.run(toolset.setup_task(task.data))
    return toolset


def test_toolset_is_declared_per_rollout():
    # 0.1.14's `scope="rollout"` is now the choice between `Task.tools`
    # (per-rollout server, its own engine) and `Taskset.tools` (one server
    # shared by a worker's rollouts). The engine must be per-rollout.
    assert m.NetHackTask.tools == (m.NetHackToolset,)
    assert m.NetHackTaskset.tools == ()


def test_self_dispatch_tools_are_wrapped():
    toolset = _toolset()
    names = set(toolset.tool_functions())
    assert "explore_and_descend" in names
    assert "move" not in names          # netplay withholds it
    # Every published tool is a wrapper around a v0 adapter, not the schema-only
    # adapter itself (`functools.wraps` sets `__wrapped__`).
    assert all(hasattr(fn, "__wrapped__") for fn in toolset.tool_functions().values())


def test_netplay_gate_holds_at_the_mcp_registration_point():
    # The gate that matters is what actually reaches the model over MCP, so
    # assert on `_register` — the real publication seam (`v1/mcp/toolset.py:28`).
    toolset = _toolset()
    recorder = _Recorder()
    toolset._register(recorder)
    assert "move" not in recorder.tools
    assert "explore_and_descend" in recorder.tools
    assert set(recorder.tools) == set(toolset.tool_functions())


def test_calls_outside_the_exposed_set_are_refused_without_stepping():
    # Defense in depth: `_apply_tool_call` re-checks the exposed set and
    # refuses without consuming an engine step (nethack.py:754).
    toolset = _toolset()
    before = toolset.v0_state["structured_obs"]
    out = asyncio.run(
        toolset.v0env._apply_tool_call(toolset.v0_state, "move", {"direction": "N"})
    )
    text = out if isinstance(out, str) else str(out)
    assert "is not available" in text
    assert toolset.v0_state["structured_obs"] is before


def test_obs_mode_default_and_self_dispatch_is_the_only_v1_mode():
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1)
    assert cfg.obs_mode == "push"
    # 0.1.14's default was `self_dispatch=False` = "arm 0, harness-driven". In
    # 0.2.x a v1 toolset is an MCP server and the ONLY way a tool call reaches
    # the engine is the tool call itself, so the harness-driven arm moved to the
    # v0 legacy bridge. The config must refuse rather than silently run arm 0's
    # semantics over MCP.
    assert cfg.self_dispatch is True
    with pytest.raises(ValueError, match="legacy bridge"):
        m.NetHackToolset(m.NetHackToolsetConfig(self_dispatch=False))


def test_state_reaches_the_tool_without_leaking_into_the_tool_schema():
    # Replaces 0.1.14's `"state" in inspect.signature(tool).parameters` test.
    # That mechanism (signature-driven `state` injection) was DELETED in 0.2.x:
    # `ServerBase._with_state` (`v1/mcp/server.py:190-216`) pulls the rollout
    # state, publishes it in a contextvar, and the tool body reads it as
    # `self.state`. Because `_with_state` re-advertises the wrapped signature to
    # FastMCP (`server.py:215`), a `state` parameter would now be published to
    # the model as a tool argument — so the guarantee is the inverse: state must
    # reach the tool, and must NOT appear in the schema.
    toolset = _toolset()
    seen = {}

    async def probe(times: int = 1):
        seen["state"] = toolset.state
        toolset.state.skill_calls += 5
        return "ok"

    wrapped = toolset._with_state(probe)
    assert asyncio.run(wrapped(times=1)) == "ok"
    # The tool saw a real, per-call NetHackState (not the inert fallback) ...
    assert isinstance(seen["state"], m.NetHackState)
    assert seen["state"] is not toolset._inert_state
    # ... and its mutation landed on that state.
    assert seen["state"].skill_calls == 5
    # The advertised MCP schema carries only the tool's own parameters.
    assert list(inspect.signature(wrapped).parameters) == ["times"]

    # Same for a real published tool.
    published = toolset._with_state(toolset.tool_functions()["search"])
    assert "state" not in inspect.signature(published).parameters
    assert "times" in inspect.signature(published).parameters


def test_self_dispatch_tool_actually_executes_and_returns_observation():
    # The whole point: unlike the schema-only v0 adapters (which raise if
    # called directly), a published tool must run the skill against the engine
    # and hand back the rendered observation from the call itself, since
    # MCP-driven CLI harnesses never see `env_response`.
    toolset = _toolset()
    result = asyncio.run(toolset.tool_functions()["search"](times=1))
    text = result if isinstance(result, str) else str(result)
    assert "=== STATUS ===" in text


def test_cleanup_releases_the_engine_and_leaves_state_serializable():
    toolset = _toolset()
    asyncio.run(toolset.tool_functions()["search"](times=1))
    engine = toolset.v0_state["env"]
    asyncio.run(toolset.close())
    assert toolset.v0_state is None
    assert toolset.v0env is None
    assert engine is not None
    # The typed state is serializable by construction (it never held the engine).
    assert toolset.state.model_dump_json()


def test_terse_drops_the_map_but_keeps_messages_and_feedback():
    from nethack_v1 import _terse
    text = "=== MAP ===\n#####\n\n=== MESSAGES ===\nYou hit it.\n[search: nothing found]"
    out = _terse(text)
    assert "#####" not in out
    assert "You hit it." in out
    assert "[search: nothing found]" in out
