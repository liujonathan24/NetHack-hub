"""`max_parallel_skill_calls` makes one budget unit mean one decision.

The call budget counts CALLS, not decisions, so a client that batches tool calls
gets fewer decisions for the same budget. Measured on identical GLM-5.2 / B0 /
seed-2 runs: the older cell emitted exactly 1.00 tool calls per assistant turn
and found the down-stair around decision 250; the newer cell batched (404 calls
in 175 turns) and exhausted the same 400-call budget after 175 decisions without
leaving dlvl 1. The v0 control arm never had this problem — it drops parallel
calls past the first — so `1` is what makes a CLI arm comparable to it.
"""
from __future__ import annotations

import asyncio

import pytest

from nethack_v1 import NetHackToolsetConfig


def test_flag_defaults_to_unlimited():
    """Off by default: existing cells must not change behaviour silently."""
    assert NetHackToolsetConfig().max_parallel_skill_calls == 0
    assert NetHackToolsetConfig().parallel_batch_window_s == 0.5


def test_taskset_config_passes_the_flag_through():
    from nethack_v1 import NetHackTasksetConfig

    ts = NetHackTasksetConfig(max_parallel_skill_calls=1, parallel_batch_window_s=0.25)
    tc = ts.toolset_config()
    assert tc.max_parallel_skill_calls == 1
    assert tc.parallel_batch_window_s == 0.25


class _FakeState:
    def __init__(self):
        self.skill_calls = 0
        self.budget_exhausted = False
        self.terminated = False
        self.moves_executed = 0
        self.batch_started_at = 0.0
        self.batch_count = 0
        self.parallel_refusals = 0


class _Toolset:
    """Minimal stand-in exercising the real `_executing` wrapper."""

    def __init__(self, **cfg):
        self.config = NetHackToolsetConfig(**cfg)
        self.state = _FakeState()
        self.executed = []
        self.v0env = self
        self.v0_state = {}

    def _publish(self, state):
        pass

    async def _apply_tool_call(self, state, name, kwargs):
        self.executed.append(name)
        return f"ok:{name}"

    _executing = None  # bound below


def _bind():
    from nethack_v1 import NetHackToolset

    _Toolset._executing = NetHackToolset._executing
    return _Toolset


async def _skill():  # pragma: no cover - schema carrier only
    ...


_skill.__name__ = "np_search"


def test_batch_past_the_cap_is_refused_and_costs_no_budget():
    ts = _bind()(max_parallel_skill_calls=1, max_skill_calls=100)
    run = ts._executing(_skill)
    out = [asyncio.get_event_loop().run_until_complete(run()) for _ in range(4)]

    assert out[0] == "ok:np_search", "first call of the batch must execute"
    for later in out[1:]:
        assert "dropped" in later, "later calls in the same batch must be refused"
    assert ts.executed == ["np_search"], "only one skill reached the engine"
    assert ts.state.skill_calls == 1, "refused calls must not consume budget"
    assert ts.state.parallel_refusals == 3, "refusals are counted for the trace"


def test_a_new_turn_after_the_window_executes_again():
    """A genuine next turn costs an LLM round-trip (~5s measured); the window is
    0.5s, so it must never swallow a real decision."""
    import time as _t

    ts = _bind()(max_parallel_skill_calls=1, parallel_batch_window_s=0.05, max_skill_calls=100)
    run = ts._executing(_skill)
    loop = asyncio.get_event_loop()
    assert loop.run_until_complete(run()) == "ok:np_search"
    assert "dropped" in loop.run_until_complete(run())
    _t.sleep(0.08)  # quiescence -> new turn
    assert loop.run_until_complete(run()) == "ok:np_search"
    assert ts.state.skill_calls == 2
    assert ts.state.parallel_refusals == 1


def test_unlimited_mode_executes_every_call():
    ts = _bind()(max_parallel_skill_calls=0, max_skill_calls=100)
    run = ts._executing(_skill)
    loop = asyncio.get_event_loop()
    for _ in range(5):
        assert loop.run_until_complete(run()) == "ok:np_search"
    assert ts.state.skill_calls == 5
    assert ts.state.parallel_refusals == 0
