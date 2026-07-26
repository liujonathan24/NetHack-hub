"""The toolset-side skill-call budget: the cross-arm referee.

`max_turns` only binds harnesses we control, but every arm — including external
CLI agents whose internal loop we do not control — passes through the MCP
toolset, so the cap lives there. One executed call == one v0 LM turn.

0.2.x construction notes: the counter lives on the typed :class:`NetHackState`
rather than a free-form ``state["skill_calls"]`` dict key, and a tool reads it
as ``self.state`` off a contextvar (`v1/mcp/server.py:132-135`). Calling the
published tool directly — the same callable ``_register`` hands to FastMCP —
uses the toolset's inert state, which persists across calls exactly as a
channel-backed rollout state does. Every assertion below keeps its 0.1.14
meaning.
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack_v1 as m


def _toolset(max_skill_calls):
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True,
                                 max_skill_calls=max_skill_calls,
                                 explicit_seeds=[0],
                                 env_args={"skill_set": "netplay"})
    task = m.load_taskset(cfg).select(1)[0]
    toolset = task.tool_servers()[0]
    asyncio.run(toolset.setup_task(task.data))
    return toolset


def test_budget_refuses_past_the_cap():
    toolset = _toolset(3)
    tool = toolset.tool_functions()["search"]

    for _ in range(3):
        asyncio.run(tool(times=1))
    assert toolset.state.skill_calls == 3

    out = asyncio.run(tool(times=1))
    assert "budget" in str(out).lower()
    assert toolset.state.skill_calls == 3     # refused calls do not count
    assert toolset.state.budget_exhausted is True
    assert toolset.state.terminated is True


def test_refusal_happens_before_any_engine_step():
    # The refusal must not reach the engine: the observation the engine would
    # have produced is unchanged across the refused call.
    toolset = _toolset(1)
    tool = toolset.tool_functions()["search"]
    asyncio.run(tool(times=1))
    before = toolset.v0_state["structured_obs"]
    out = asyncio.run(tool(times=1))
    assert "budget" in str(out).lower()
    assert toolset.v0_state["structured_obs"] is before


def test_exhaustion_is_the_named_terminal_reason():
    # `RolloutSession.refused` records the firing @stop's `__name__` as the
    # trace's stop condition (`v1/session.py:115`), so the terminal reason is
    # still exactly "call_budget_exhausted".
    task = m.load_taskset({"task_spec": "full_nle", "n_examples": 1}).select(1)[0]
    from verifiers.v1.decorators import discover_decorated

    stops = {fn.__name__: fn for fn in discover_decorated(task, "stop")}
    assert "call_budget_exhausted" in stops

    class _Trace:
        def __init__(self, state):
            self.state = state

    assert asyncio.run(
        stops["call_budget_exhausted"](_Trace(m.NetHackState(budget_exhausted=True)))
    ) is True
    assert asyncio.run(
        stops["call_budget_exhausted"](_Trace(m.NetHackState()))
    ) is False


def _run_disabled_cap(max_skill_calls, n_calls=5):
    toolset = _toolset(max_skill_calls)
    tool = toolset.tool_functions()["search"]
    for _ in range(n_calls):
        out = asyncio.run(tool(times=1))
        assert "budget" not in str(out).lower()
    return toolset.state


def test_zero_budget_disables_the_cap():
    # max_skill_calls <= 0 means "no cap" — an unbounded arm must keep
    # succeeding past any budget a positive value would have enforced, and
    # must never terminate on the budget path.
    state = _run_disabled_cap(0, n_calls=5)
    assert state.skill_calls == 5
    assert state.budget_exhausted is False
    assert state.terminated is False


def test_negative_budget_disables_the_cap():
    state = _run_disabled_cap(-1, n_calls=5)
    assert state.skill_calls == 5
    assert state.budget_exhausted is False
    assert state.terminated is False


# --------------------------------------------------------------------------- #
# Concurrency: the referee under the condition it actually exists for          #
# --------------------------------------------------------------------------- #
# Everything above drives the tool callables directly and serially, which is
# how the budget was originally specified. That is not how a CLI agent calls
# them. Claude Code emits several `tool_use` blocks in one assistant turn, and
# `ServerBase._with_state` (`v1/mcp/server.py:190-216`) is documented
# last-write-wins: it GETs the rollout state, runs the tool, then PUTs the whole
# state back. Two concurrent calls that both read `skill_calls = N` and both
# write `N + 1` would let an arm execute more skills than `max_skill_calls`
# allows — a silent invalidation of the cross-arm comparison — and would also
# re-enter one C-extension engine concurrently.
#
# These tests need the real seam (the HTTP `/state` channel), because offline
# `_pull_state` hands out a fresh state per call (`server.py:145-148`) and the
# race cannot appear. So they boot `python -m nethack_v1` for real; see
# `_mcp_server_harness.py`.

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _mcp_server_harness import booted_toolset, build_task_and_trace  # noqa: E402


def _fire_concurrently(n_calls, max_skill_calls):
    """Issue `n_calls` overlapping MCP tool calls; return (trace, results)."""
    task, trace = build_task_and_trace(max_skill_calls=max_skill_calls)

    async def main():
        async with booted_toolset(task, trace) as (session, mini):
            results = await asyncio.gather(
                *(session.call_tool("search", {"times": 1}) for _ in range(n_calls))
            )
            texts = [
                "".join(getattr(b, "text", "") for b in (r.content or []))
                for r in results
            ]
            return texts, mini.gets, mini.puts

    texts, gets, puts = asyncio.run(main())
    return trace, texts, gets, puts


def test_concurrent_calls_do_not_overrun_the_cap():
    """N overlapping calls against a cap of K execute EXACTLY K skills.

    This is the regression that matters: without serialization the pushed
    counter lands somewhere below K while more than K calls have already
    stepped the engine.
    """
    n_calls, cap = 8, 3
    trace, texts, _gets, _puts = _fire_concurrently(n_calls, cap)

    executed = [t for t in texts if "budget exhausted" not in t.lower()]
    refused = [t for t in texts if "budget exhausted" in t.lower()]

    assert trace.state.skill_calls == cap, (
        f"referee counted {trace.state.skill_calls}, expected exactly {cap} "
        f"({len(executed)} executed / {len(refused)} refused of {n_calls})"
    )
    assert len(executed) == cap, f"{len(executed)} calls stepped the engine, cap was {cap}"
    assert len(refused) == n_calls - cap
    assert trace.state.budget_exhausted is True
    # Every executed call returned a real observation, not a truncated/garbled
    # one from two engine steps interleaving.
    for text in executed:
        assert "=== STATUS ===" in text


def test_concurrent_calls_are_counted_exactly_when_under_the_cap():
    """Below the cap, N overlapping calls must count as exactly N.

    The complement of the test above: it proves the count is not merely
    clamped by the cap but is actually exact, i.e. no increment is lost to a
    last-write-wins overwrite.
    """
    n_calls = 6
    trace, texts, gets, puts = _fire_concurrently(n_calls, max_skill_calls=100)

    assert trace.state.skill_calls == n_calls, (
        f"lost increments: counted {trace.state.skill_calls} of {n_calls}"
    )
    assert trace.state.budget_exhausted is False
    assert all("=== STATUS ===" in t for t in texts)
    # Each call did its own state round-trip, so the channel — not a local
    # variable — carried the count.
    assert gets == n_calls and puts == n_calls
