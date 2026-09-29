"""Fix2 — empty-completion retry via the existing relaunch loop (E15).

The stock verifiers harness loop (`v1/harness.py:109-118`, read-only) records
`stop_condition="agent_completed"` whenever the CLI exits cleanly — including
when the model returned one EMPTY completion (finish_reason "stop", no text,
no tool call). That censored live runs: P3 r1 seed 4 at D11 (best in arm),
v2_bjson 3/5, v2_json seed 1 at D10. The sanctioned interception point is
`PrimeAgentHarness.launch`'s auto-resume loop (same MCP tool server, resume
prompt), which the E15 tier contract had pinned to `max_relaunches = 0`.
Fix2 sets the contract to 2; these tests pin (1) that the contract now says
2 and (2) that the SERVED harness's loop actually retries a clean-exit-while-
live result and stops on the cap / on a genuinely-over episode.
"""

import asyncio
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _harness(max_relaunches):
    import nethack_prime_agent as hp
    cfg = hp.PrimeAgentHarnessConfig(max_relaunches=max_relaunches)
    h = object.__new__(hp.PrimeAgentHarness)
    h.config = cfg
    return h


class _Result(SimpleNamespace):
    pass


def _trace(terminated=False, budget_exhausted=False):
    metrics = {}
    state = SimpleNamespace(
        terminated=terminated, budget_exhausted=budget_exhausted, skill_calls=7
    )
    t = SimpleNamespace(
        id="t0", state=state, metrics=metrics,
        record_metric=lambda k, v: metrics.__setitem__(k, v),
    )
    return t


def _drive(h, trace, exit_codes):
    """Run the tail of launch(): the auto-resume loop, lifted verbatim from
    the served module by calling launch's own collaborators (_run_once is
    stubbed; _episode_live is the real served method)."""
    calls = []

    async def fake_run_once(runtime, argv, env, agent_dir, attempt):
        calls.append((list(argv), attempt))
        code = exit_codes[min(len(calls) - 1, len(exit_codes) - 1)]
        return _Result(exit_code=code)

    h._run_once = fake_run_once

    async def run():
        import nethack_prime_agent as hp
        argv = ["prime-agent", "--print", "PROMPT"]
        result = await h._run_once(None, argv, {}, "/tmp/x", attempt=0)
        resume_argv = argv[:-1] + [hp._RESUME_PROMPT]
        relaunches = 0
        while (
            result.exit_code == 0
            and relaunches < h.config.max_relaunches
            and h._episode_live(trace)
        ):
            relaunches += 1
            result = await h._run_once(None, resume_argv, {}, "/tmp/x", attempt=relaunches)
        trace.record_metric("prime_agent_relaunches", float(relaunches))
        return result, relaunches

    return asyncio.run(run()), calls


def test_contract_sets_two_retries():
    import tomllib
    p = (pathlib.Path(__file__).resolve().parents[3]
         / "tools/cli_harness_eval/configs/tool_tiers.toml")
    contract = tomllib.loads(p.read_text())["contract"]
    assert contract["max_relaunches"] == 2


def test_clean_exit_while_live_retries_up_to_cap():
    import nethack_prime_agent as hp
    h = _harness(max_relaunches=2)
    trace = _trace(terminated=False, budget_exhausted=False)
    (result, relaunches), calls = _drive(h, trace, exit_codes=[0, 0, 0])
    assert relaunches == 2 and len(calls) == 3
    # Retries go out on the RESUME prompt, not the original task prompt.
    assert calls[1][0][-1] == hp._RESUME_PROMPT
    assert calls[2][0][-1] == hp._RESUME_PROMPT
    assert trace.metrics["prime_agent_relaunches"] == 2.0


def test_no_retry_when_episode_actually_over():
    h = _harness(max_relaunches=2)
    trace = _trace(terminated=True)
    (result, relaunches), calls = _drive(h, trace, exit_codes=[0])
    assert relaunches == 0 and len(calls) == 1
    assert trace.metrics["prime_agent_relaunches"] == 0.0


def test_retry_stops_once_state_terminates():
    # First resume finishes the game (state flips terminated): no second one.
    h = _harness(max_relaunches=2)
    trace = _trace(terminated=False)

    calls = []

    async def fake_run_once(runtime, argv, env, agent_dir, attempt):
        calls.append(attempt)
        if attempt == 1:
            trace.state.terminated = True
        return _Result(exit_code=0)

    h._run_once = fake_run_once

    async def run():
        import nethack_prime_agent as hp
        result = await h._run_once(None, ["prime-agent", "--print", "P"], {}, "/tmp/x", attempt=0)
        relaunches = 0
        while (
            result.exit_code == 0
            and relaunches < h.config.max_relaunches
            and h._episode_live(trace)
        ):
            relaunches += 1
            result = await h._run_once(None, ["prime-agent", "--print", hp._RESUME_PROMPT], {}, "/tmp/x", attempt=relaunches)
        return relaunches

    relaunches = asyncio.run(run())
    assert relaunches == 1 and calls == [0, 1]


def test_episode_live_is_conservative_on_missing_state():
    h = _harness(max_relaunches=2)
    assert h._episode_live(None) is False
    assert h._episode_live(SimpleNamespace(state=None)) is False
