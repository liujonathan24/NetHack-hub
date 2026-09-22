#!/usr/bin/env python
"""Offline test: a client that exhausts its retries must not kill the episode.

BALROG's ``execute_with_retries`` raises after ``max_retries``. On the panel's
path (``PrimeOpenAIWrapper`` -> ``OpenAIWrapper.generate``) that exception used
to propagate out of the episode loop, so a provider blip - or a GLM completion
that spends its whole ``max_tokens`` on reasoning and comes back with
``finish_reason="length"`` and ``content=None`` - ended the episode at a random
step for a harness reason. That truncation is not arm-neutral: it costs the arm
that plays more steps, which is PAE, so it would bias the base-vs-PAE contrast.

The fix makes the exhausted-retry case a recorded, non-fatal event: every
wrapper returns an empty completion stamped ``stop_reason =
RETRY_EXHAUSTED_STOP_REASON``, and the PAE loop counts the step as invalid under
its own counter and plays on.

  python -m tools.balrog_pae.test_client_retry_exhausted [--game minihack ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import types
import warnings

warnings.filterwarnings("ignore")

from balrog.client import (  # noqa: E402
    RETRY_EXHAUSTED_STOP_REASON,
    AWSBedrockWrapper,
    ClaudeWrapper,
    GoogleGenerativeAIWrapper,
    LLMResponse,
    OpenAIWrapper,
)


def _client_config(**over):
    cfg = types.SimpleNamespace(
        client_name="openai", model_id="test-model", base_url="", timeout=1.0,
        generate_kwargs={"temperature": 1.0, "max_tokens": 64}, max_retries=2,
        delay=0, alternate_roles=False,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _msg(text="hi"):
    from balrog.prompt_builder.history import Message

    return [Message(role="user", content=text)]


# --- 1. the real Boxoban shape, end to end through OpenAIWrapper -------------
def test_openai_length_capped_completion():
    """finish_reason='length' with content=None -> empty completion, no raise."""

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            # Exactly what Prime returned for GLM-5.2 on Boxoban: the whole
            # token budget went to `reasoning`, `content` came back None.
            message = types.SimpleNamespace(content=None, reasoning="thinking..." * 10)
            choice = types.SimpleNamespace(finish_reason="length", index=0, message=message)
            usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=16384)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    w = OpenAIWrapper(_client_config())
    fake = FakeCompletions()
    w.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=fake))
    w._initialized = True

    r = w.generate(_msg())
    assert isinstance(r, LLMResponse), f"expected an LLMResponse, got {type(r)}"
    assert r.completion == "", f"expected an empty completion, got {r.completion!r}"
    assert r.stop_reason == RETRY_EXHAUSTED_STOP_REASON, f"stop_reason={r.stop_reason!r}"
    assert fake.calls == w.max_retries, f"retried {fake.calls}x, expected {w.max_retries}"
    print(f"  OpenAIWrapper (length-capped, {fake.calls} retries): "
          f"completion={r.completion!r} stop_reason={r.stop_reason!r}")


# --- 2. every wrapper agrees on the contract ---------------------------------
def test_all_wrappers_agree():
    """No wrapper may raise out of generate() once retries are exhausted."""
    cases = [
        ("OpenAIWrapper", OpenAIWrapper, {}),
        ("ClaudeWrapper", ClaudeWrapper, {"client_name": "claude"}),
        ("AWSBedrockWrapper", AWSBedrockWrapper, {"client_name": "aws-bedrock"}),
        ("GoogleGenerativeAIWrapper", GoogleGenerativeAIWrapper, {"client_name": "gemini"}),
    ]
    for name, cls, over in cases:
        w = cls.__new__(cls)
        LLMClientWrapperInit = type(w).__mro__[-2]  # LLMClientWrapper
        LLMClientWrapperInit.__init__(w, _client_config(**over))
        w._initialized = True
        w.client = object()
        w.generation_config = None
        w.convert_messages = lambda messages: []
        # the provider is down: every attempt raises, so retries are exhausted
        w.execute_with_retries = lambda func, *a, **k: (_ for _ in ()).throw(
            Exception(f"Failed to execute {func.__name__} after {w.max_retries} retries.")
        )
        r = w.generate(_msg())
        assert isinstance(r, LLMResponse), f"{name} returned {type(r)}"
        assert r.completion == "", f"{name} completion={r.completion!r}"
        assert r.stop_reason == RETRY_EXHAUSTED_STOP_REASON, f"{name} stop_reason={r.stop_reason!r}"
        assert r.input_tokens == 0 and r.output_tokens == 0, f"{name} billed tokens for a failed call"
        print(f"  {name}: completion={r.completion!r} stop_reason={r.stop_reason!r}")


# --- 3. the PAE loop keeps playing and counts it apart -----------------------
class FlakyClient:
    """A scripted 'LLM' whose client gives up on a fixed set of steps."""

    def __init__(self, actions, fail_on, seed=0):
        self.actions = actions
        self.fail_on = set(fail_on)
        self.rng = random.Random(seed)
        self.calls = 0

    def generate(self, messages):
        n = sum(len(m.content) for m in messages)
        self.calls += 1
        if self.calls in self.fail_on:
            # exactly what the patched wrappers now return
            return LLMResponse(model_id="scripted", completion="",
                               stop_reason=RETRY_EXHAUSTED_STOP_REASON,
                               input_tokens=0, output_tokens=0, reasoning=None)
        return LLMResponse(model_id="scripted", completion=self.rng.choice(self.actions),
                           stop_reason="stop", input_tokens=n // 4, output_tokens=3,
                           reasoning=None)


def test_episode_survives(game, task, run_dir, max_steps, fail_on):
    from balrog.prompt_builder import create_prompt_builder

    from .pae import Config, Run
    from .player import PAEAgent
    from .run import balrog_config

    cfg = Config(game=game, task=task, seed=0, attempts=1, checkpoint_every=max_steps + 1,
                 plateau=99, select="fixed", directive="off", max_steps=max_steps,
                 run_dir=run_dir, base_only=True)
    bcfg = balrog_config("scripted", 1.0, 64, 60)
    run = Run(cfg, bcfg)

    space = run.adapter.env.env.language_action_space
    if game == "minihack":
        acts = [x for x in space if x in ("north", "south", "east", "west", "search", "wait")] or ["north"]
    elif game == "crafter":
        acts = list(space)
    else:
        acts = ["look", "inventory", "go north", "go south", "go east", "go west"]

    client = FlakyClient(acts, fail_on)
    run._new_agent = lambda: PAEAgent(lambda: client, create_prompt_builder(bcfg.agent))
    run.orch = None
    run.go()

    summary = json.loads((run.dir / "summary.json").read_text())
    trace = [json.loads(l) for l in (run.dir / "attempts" / "a001" / "trace.jsonl").read_text().splitlines()]

    # (a) the episode was NOT truncated by the client failure
    last_failed_step = max(fail_on) - 1
    assert len(trace) > last_failed_step + 1, \
        f"episode stopped at {len(trace)} steps; the failure at step {last_failed_step} truncated it"
    if trace[-1]["done"]:
        print(f"  note: episode ended naturally at step {trace[-1]['step']} ({summary['stop_reason']})")
    else:
        assert len(trace) == max_steps, f"{len(trace)} steps played, expected the full {max_steps}"

    # (b) each failed step is marked invalid and flagged, and the env advanced
    failed = [r for r in trace if r["client_retry_exhausted"]]
    assert len(failed) == len(fail_on), f"{len(failed)} flagged steps, expected {len(fail_on)}"
    for r in failed:
        assert r["completion"] == "", f"step {r['step']} completion={r['completion']!r}"
        assert r["valid"] is False, f"step {r['step']} was not marked invalid"
        assert r["action"], f"step {r['step']} did not fall back to a default action"

    # (c) the counters: distinct from genuine invalid actions, and nothing lost
    assert summary["client_retry_exhausted"] == len(fail_on), \
        f"summary counter {summary['client_retry_exhausted']} != {len(fail_on)}"
    genuine = sum(1 for r in trace if not r["valid"] and not r["client_retry_exhausted"])
    assert summary["invalid_actions"] == genuine, \
        f"invalid_actions {summary['invalid_actions']} != {genuine} genuine parse failures"
    assert (run.dir / "client_failures.jsonl").exists(), "no client_failures.jsonl was written"

    print(f"  {game}/{task}: {len(trace)} steps played, "
          f"client_retry_exhausted={summary['client_retry_exhausted']}, "
          f"invalid_actions={summary['invalid_actions']} (genuine), "
          f"stop_reason={summary['stop_reason']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="minihack")
    ap.add_argument("--task", default="MiniHack-Quest-Easy-v0")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--fail-on", default="4,7", help="1-based LLM call numbers that exhaust retries")
    ap.add_argument("--run-dir", default="/tmp/st_retry")
    a = ap.parse_args(argv)

    print("1. OpenAIWrapper on the real Boxoban response shape")
    test_openai_length_capped_completion()
    print("2. all wrappers return the same empty completion on exhausted retries")
    test_all_wrappers_agree()
    print("3. the PAE episode survives, marks the step invalid and counts it apart")
    test_episode_survives(a.game, a.task, a.run_dir, a.max_steps,
                          [int(x) for x in a.fail_on.split(",")])
    print("PASS: retry exhaustion is recorded and non-fatal in every wrapper and in the loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
