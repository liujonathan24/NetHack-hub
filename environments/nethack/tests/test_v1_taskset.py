"""Tests for the Verifiers v1 taskset port of NetHack (``nethack_v1``).

Covers:
  * ``load_v1_environment`` builds a ``vf.Env`` for BOTH curriculum tasks
    (``full_nle`` and ``six_floor_primitives``).
  * The v1 rewards / toolset / custom harness are wired.
  * A minimal, keyless (mock-model) v1 rollout runs end-to-end: setup creates the
    engine, the custom harness drives model turns, ``env_response`` steps the
    engine, the completion is populated, rewards are scored, and the final state
    is JSON-serializable.
  * The v0 path (``load_environment`` / ``NetHackVerifiersEnv``) still imports and
    builds — no regression.

Run with the engine on the path, e.g.:
    NLE_LIB_PATH=/.../libnethack.so \
    PYTHONPATH=environments/nethack:/path/to/NetHackHarness \
    pytest environments/nethack/tests/test_v1_taskset.py
"""

import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[2] / "environments" / "nethack")
)

import nethack_v1 as m  # noqa: E402
from verifiers.types import Response, ResponseMessage, ToolCall  # noqa: E402


class _MockClient:
    """Keyless mock model: always emits a single valid skill tool call."""

    def __init__(self, tool_name: str = "search", arguments: str = "{}"):
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        self.calls += 1
        msg = ResponseMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.calls}",
                    name=self.tool_name,
                    arguments=self.arguments,
                )
            ],
            finish_reason="tool_calls",
            is_truncated=False,
        )
        return Response(id=f"resp_{self.calls}", created=0, model=model, message=msg)


def test_v1_env_builds_for_both_tasks():
    for task_spec in ("full_nle", "six_floor_primitives"):
        cfg = m.NetHackTasksetConfig(task_spec=task_spec, n_examples=2, max_turns=3)
        env = m.load_v1_environment(cfg)
        # Taskset wiring.
        assert len(env.taskset.get_dataset()) == 2
        assert [r.__name__ for r in env.taskset.rewards] == [
            "scout_reward",
            "descent_reward",
            "success_reward",
            "ascension_reward",
        ]
        assert env.taskset.system_prompt  # non-empty spec system prompt
        assert len(env.taskset.toolsets) == 1
        # Custom harness.
        assert isinstance(env.harness, m.NetHackHarness)
        # Rows carry a seed and per-task max_turns.
        task0 = list(env.taskset)[0]
        assert isinstance(task0.get("seed"), int)
        assert task0.get("max_turns") == 3


def test_v1_mock_rollout_full_nle():
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1, max_turns=2)
    env = m.load_v1_environment(cfg)
    task = list(env.taskset)[0]

    async def _run():
        return await env.rollout(
            task, client=_MockClient("search"), model="mock-model"
        )

    state = asyncio.run(_run())

    # The mock made exactly max_turns model requests.
    assert state.get("num_model_requests") == 2
    # A completion (the game conversation) was produced.
    completion = state.get("completion")
    assert isinstance(completion, list) and len(completion) >= 1
    roles = {msg.get("role") for msg in completion if isinstance(msg, dict)}
    assert "assistant" in roles  # the model turns
    # Rewards were scored and recorded.
    assert isinstance(state.get("reward"), (int, float))
    metrics = state.get("metrics") or {}
    for name in ("scout_reward", "descent_reward", "success_reward", "ascension_reward"):
        assert name in metrics
    # Engine was freed and the state is JSON-serializable (no engine/numpy/sets).
    assert "env" not in state
    json.dumps(state)


def test_v0_path_still_builds():
    # No-regression: the v0 entrypoint still constructs a StatefulToolEnv.
    from nethack import load_environment, NetHackVerifiersEnv

    env = load_environment(n_examples=1, max_turns=2)
    assert isinstance(env, NetHackVerifiersEnv)


if __name__ == "__main__":
    test_v1_env_builds_for_both_tasks()
    print("build-both: OK")
    test_v1_mock_rollout_full_nle()
    print("mock-rollout: OK")
    test_v0_path_still_builds()
    print("v0-path: OK")
    print("ALL PASS")
