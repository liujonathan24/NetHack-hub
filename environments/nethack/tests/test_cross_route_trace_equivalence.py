"""Cross-route **trace-field** equivalence: control route vs native MCP route.

`test_cross_route_equivalence.py` closes the *engine* half of design-doc §10 in
process: the same forced skill sequence from the same seed produces identical
engine state and identical rendered observations down both routes. It explicitly
does not cover the process boundary. This file closes the other half:

    the same forced skill sequence, from the same seed, must produce the same
    **scored trace** whether it is dispatched by the control route (the v0
    legacy bridge) or the native route (a real MCP tool server over a real
    `/state` channel, scored by `NetHackTask.score`).

That is the check that makes "the arms play the same game" mean something at the
level the experiment actually reports on — the numbers in `traces.jsonl`.

WHAT IS REAL HERE, AND WHAT IS NOT
----------------------------------
* Control route: the v0 rollout loop verbatim (`env.run_rollout`, which is what
  `v1/legacy.py:517` calls) driven by a keyless scripted model, then mapped
  through the bridge's own `rollout_output_to_trace`.
* Native route: `python -m nethack_v1` booted as a real MCP server subprocess,
  reached with a real MCP client, with a real HTTP `/state` + `/task` channel
  standing in for the interception server's (`v1/interception/server.py:193-197`).
  Scoring is the framework's own `NetHackTask.score`.
* **No harness program runs.** `Trace.num_turns` is therefore not compared: see
  `test_num_turns_is_not_a_cross_route_comparable` below, which pins *why* — it
  is a property of the agent's loop, not of the game, and the experiment
  normalizes on executed skill calls instead. The `null` harness, which would
  have made the two loops shape-comparable, cannot run on this machine
  (`prepare_uv_script` writes to a hard-coded `/tmp/vf-scripts` owned by another
  user); that is recorded in `tools/cli_harness_eval/configs/README.md`.
"""

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import uuid

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack_v1 as m  # noqa: E402

# Deterministic, engine-stepping, and inside the netplay skill set.
SEQUENCE = [
    ("search", {"times": 1}),
    ("search", {"times": 2}),
    ("search", {"times": 1}),
]
SEED = 0
CHARACTER = "Val-hum-neu-fem"
ENV_ARGS = {"skill_set": "netplay"}
REWARD_NAMES = ("scout_reward", "descent_reward", "success_reward", "ascension_reward")
SECRET = "cross-route-secret"


# --------------------------------------------------------------------------- #
# Control route: the v0 loop the legacy bridge runs, then the bridge's mapping #
# --------------------------------------------------------------------------- #
class _ScriptedClient:
    """Keyless model that emits SEQUENCE[i] on turn i, then stops calling tools.

    Subclasses the v0 `Client` ABC lazily (inside `__init_subclass__`-free code)
    because `run_rollout` resolves its client through `resolve_client`
    (`verifiers/clients/__init__.py:15`), which rejects duck-typed objects.
    """


def _control_trace():
    from nethack import load_environment
    from verifiers.clients import Client
    from verifiers.types import Response, ResponseMessage, ToolCall
    from verifiers.v1.legacy import rollout_output_to_trace

    class Scripted(Client):
        def __init__(self):
            self._config = None
            self._client = None
            self.calls = 0

        def setup_client(self, config):
            return None

        async def close(self):
            return None

        async def to_native_tool(self, tool):
            return tool

        async def to_native_prompt(self, messages):
            return messages, {}

        async def get_native_response(self, *a, **kw):
            return None

        async def raise_from_native_response(self, response):
            return None

        async def from_native_response(self, response):
            return response

        async def get_response(self, prompt, model, sampling_args, tools=None, **kw):
            i = self.calls
            self.calls += 1
            if i < len(SEQUENCE):
                name, args = SEQUENCE[i]
                calls = [
                    ToolCall(id=f"call_{i}", name=name, arguments=json.dumps(args))
                ]
                finish = "tool_calls"
            else:
                calls, finish = None, "stop"
            msg = ResponseMessage(
                role="assistant",
                content="" if calls else "done",
                tool_calls=calls,
                finish_reason=finish,
                is_truncated=False,
            )
            return Response(id=f"resp_{i}", created=0, model=model, message=msg)

    env = load_environment(
        task_spec="full_nle",
        n_examples=1,
        # One LM turn dispatches at most one skill call in the v0 loop, so the
        # turn cap IS the skill-call budget. +1 for the closing no-tool turn.
        max_turns=len(SEQUENCE) + 1,
        explicit_seeds=[SEED],
        character=CHARACTER,
        **ENV_ARGS,
    )
    client = Scripted()
    out = asyncio.run(
        env.run_rollout(
            input=dict(env.get_eval_dataset()[0]),
            client=client,
            model="scripted",
            sampling_args={},
            state_columns=["trajectory"],
        )
    )
    return rollout_output_to_trace(out, 0), client.calls


# --------------------------------------------------------------------------- #
# Native route: a real MCP tool server over a real /state + /task channel      #
# --------------------------------------------------------------------------- #
class _MiniInterception:
    """The two channels a launched tool server needs, backed by a real Trace.

    Mirrors `InterceptionServer.handle_state_get/put` and `handle_task_get`
    (`v1/interception/server.py:745-795`) exactly: bearer auth, the state as the
    typed model's JSON, and the task as `{"cls": "<module>:<qualname>", "task":
    <json>}`. The backing store IS `trace.state`, so what the tools push is what
    `NetHackTask.score` later reads — no shim in between.
    """

    def __init__(self, trace, task_data):
        self.trace = trace
        self.task_data = task_data
        self.puts = 0

    def app(self):
        from aiohttp import web
        from pydantic import TypeAdapter

        adapter = TypeAdapter(m.NetHackState)

        def _auth(request):
            assert request.headers.get("Authorization") == f"Bearer {SECRET}"

        async def state_get(request):
            _auth(request)
            return web.Response(
                body=adapter.dump_json(self.trace.state),
                content_type="application/json",
                charset="utf-8",
            )

        async def state_put(request):
            _auth(request)
            self.puts += 1
            self.trace.state = m.NetHackState.model_validate_json(await request.read())
            return web.json_response({"ok": True})

        async def task_get(request):
            _auth(request)
            data = self.task_data
            return web.json_response(
                {
                    "cls": f"{type(data).__module__}:{type(data).__qualname__}",
                    "task": data.model_dump_json(),
                }
            )

        app = web.Application()
        app.router.add_get("/state", state_get)
        app.router.add_put("/state", state_put)
        app.router.add_get("/task", task_get)
        return app


async def _drive_native(trace, task):
    """Boot the tool server, issue SEQUENCE over MCP, return the advertised names."""
    from aiohttp import web
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    mini = _MiniInterception(trace, task.data)
    runner = web.AppRunner(mini.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    state_url = f"http://127.0.0.1:{runner.addresses[0][1]}/state"

    toolset = task.tool_servers()[0]
    port_file = pathlib.Path(tempfile.gettempdir()) / f"vf-port-{uuid.uuid4().hex}"
    env = {
        **os.environ,
        # Exactly what `serve_in_runtime` sets (`v1/mcp/launch.py:186-195`).
        "VF_CONFIG": toolset.config.model_dump_json(),
        "VF_STATE_URL": state_url,
        "VF_STATE_SECRET": SECRET,
        "MCP_PORT_FILE": str(port_file),
    }
    log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "nethack_v1"], env=env, stdout=log, stderr=log
    )
    try:
        port = None
        for _ in range(300):
            if port_file.exists() and port_file.read_text().strip().isdigit():
                port = int(port_file.read_text().strip())
                break
            assert proc.poll() is None, _fail(log, proc, "server exited before binding")
            await asyncio.sleep(1)
        assert port, _fail(log, proc, "server never reported a port")
        url = f"http://127.0.0.1:{port}/mcp"
        # The port file is written before setup (`v1/mcp/server.py:243-246`), so
        # wait for the socket to listen — verifiers' own `_PROBE` does the same.
        import urllib.error
        import urllib.request

        for _ in range(300):
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                assert proc.poll() is None, _fail(log, proc, "server exited in setup")
                await asyncio.sleep(1)
        else:
            raise AssertionError(_fail(log, proc, "server never listened"))

        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                advertised = sorted(t.name for t in (await session.list_tools()).tools)
                for name, args in SEQUENCE:
                    result = await session.call_tool(name, args)
                    assert not result.isError, f"{name} errored: {result.content}"
                return advertised
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()
        await runner.cleanup()


def _fail(log, proc, why):
    log.flush()
    tail = pathlib.Path(log.name).read_bytes()[-3000:].decode(errors="replace")
    return f"{why} (rc={proc.returncode})\n--- tool server log ---\n{tail}"


def _native_trace():
    from verifiers.v1.state import state_cls
    from verifiers.v1.trace import Trace, TraceTask

    cfg = m.NetHackTasksetConfig(
        task_spec="full_nle",
        n_examples=1,
        explicit_seeds=[SEED],
        character=CHARACTER,
        max_turns=len(SEQUENCE) + 1,
        env_args=dict(ENV_ARGS),
        # Nothing runs the agent here, so there is no workspace to seed.
        seed_workspace=False,
    )
    task = m.load_taskset(cfg).select(1)[0]
    # Built the way `Rollout.run` builds it (`v1/rollout.py:106-110`).
    trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        state=state_cls(type(task))(),
    )
    advertised = asyncio.run(_drive_native(trace, task))
    asyncio.run(task.finalize(trace, None))
    asyncio.run(task.score(trace, None))
    return trace, task, advertised


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def routes():
    control, control_calls = _control_trace()
    native, task, advertised = _native_trace()
    return control, control_calls, native, task, advertised


def test_the_forced_sequence_actually_ran_on_both_routes(routes):
    """Guard against a vacuous pass: neither route may have short-circuited."""
    control, control_calls, native, _task, advertised = routes
    # The scripted model was asked for every turn, including the closing one.
    assert control_calls == len(SEQUENCE) + 1
    # Every call reached the engine through the MCP transport and the /state
    # channel — this is the counter the toolset-side referee increments.
    assert native.state.skill_calls == len(SEQUENCE)
    assert native.state.budget_exhausted is False
    # And the gate held on the wire, not just in a config.
    assert "move" not in advertised
    assert "explore_and_descend" in advertised


def test_the_two_routes_score_the_same_game_the_same_way(routes):
    """`reward` and the four reward metrics agree across the routes.

    The two routes reach `Trace` through completely different code — the v0
    `Rubric` plus `rollout_output_to_trace` on one side, `@vf.reward` methods
    plus `Task.score` on the other — so this is where a scoring divergence
    between the arms would show up, and nothing else in the suite would catch it.

    The bridge records the v0 rubric's per-func values in `Trace.metrics` and its
    weighted total under `rewards["reward"]` (`v1/legacy.py:264-277`); the native
    route records per-reward weighted values in `Trace.rewards`. So the shapes
    differ and the *values* are what must agree.
    """
    control, _control_calls, native, _task, _advertised = routes

    control_per_reward = {n: control.metrics.get(n) for n in REWARD_NAMES}
    native_per_reward = {n: native.rewards.get(n) for n in REWARD_NAMES}
    assert None not in control_per_reward.values(), control.metrics
    assert None not in native_per_reward.values(), native.rewards

    assert control_per_reward == pytest.approx(native_per_reward), (
        f"per-reward divergence:\n  control(metrics)={control_per_reward}\n"
        f"  native(rewards)={native_per_reward}"
    )
    assert control.reward == pytest.approx(native.reward), (
        f"total reward diverged: control={control.reward} native={native.reward}"
    )


def test_the_total_reward_is_not_cross_arm_comparable_because_the_weights_differ():
    """The one place the two routes genuinely disagree — pinned, not papered over.

    `test_the_two_routes_score_the_same_game_the_same_way` passes on the forced
    sequence only because three of the four rewards are zero there. They are not
    zero in a real rollout, and the two routes apply **different weights**:

    * control: `nethack.py` builds `vf.Rubric(funcs=[...], weights=None)`, and a
      v0 Rubric's default is `[1.0] * n` — verified by construction below. The
      `@vf.reward(weight=...)` annotations on the v0 functions are NOT read by
      the rubric (the comment at `nethack.py:1671-1676` says so explicitly, and
      that behavior is what experiment 1's numbers were produced under).
    * native: `NetHackTask`'s `@vf.reward(weight=...)` methods are weighted
      1 / 10 / 100 / 1000 by `Task.score` (`v1/task.py:303-307`).

    So `Trace.reward` means "sum of raw rewards" on the control arm and "sum of
    1/10/100/1000-weighted rewards" on the CLI arms. **Comparing `trace.reward`
    across the arms is invalid.** Compare the per-reward values instead — those
    do agree (the test above) and are present on both routes.

    Neither side is changed here: re-weighting the control arm would alter the
    semantics Task 12 preserved on purpose, and re-weighting the native arm
    would drop the weights the reward authors declared. This test fails if
    anyone silently changes either, forcing the decision to be made explicitly.
    """
    import verifiers as vf0
    from nethack_harness.helpers import (
        ascension_reward,
        descent_reward,
        scout_reward,
        success_reward,
    )

    funcs = [scout_reward, descent_reward, success_reward, ascension_reward]
    control_weights = vf0.Rubric(funcs=funcs, weights=None).weights
    assert control_weights == [1.0, 1.0, 1.0, 1.0]

    from verifiers.v1.decorators import discover_decorated

    task = m.load_taskset({"task_spec": "full_nle", "n_examples": 1}).select(1)[0]
    native_weights = {
        fn.__name__: getattr(fn, "_vf_weight") for fn in discover_decorated(task, "reward")
    }
    assert native_weights == {
        "scout_reward": 1.0,
        "descent_reward": 10.0,
        "success_reward": 100.0,
        "ascension_reward": 1000.0,
    }

    # Concretely: one descent scores 1.0 on the control arm and 10.0 on a CLI arm.
    assert control_weights[1] != native_weights["descent_reward"]


def test_the_two_routes_agree_on_the_stop_condition(routes):
    """Neither route hit a *game* terminal condition over this sequence.

    The native route's `@vf.stop` predicates read the state scalars; the control
    route's terminal bookkeeping is the same scalars on the v0 side. What must
    agree is that the game did not end — a divergence here would mean one route
    killed, descended, or budget-capped the episode and the other did not.

    The literal `stop_condition` STRINGS are deliberately not compared; see
    `test_the_control_routes_stop_condition_string_is_a_bridge_artifact`.
    """
    control, _control_calls, native, task, _advertised = routes

    assert asyncio.run(task.call_budget_exhausted(native)) is False
    assert asyncio.run(task.game_over(native)) is False
    assert native.state.died is False
    assert native.state.terminated is False
    # The control route ended on its own turn cap, not on a game event.
    assert control.num_turns == len(SEQUENCE) + 1
    assert not control.errors, control.errors


def test_the_control_routes_stop_condition_string_is_a_bridge_artifact(routes):
    """Pin a real trap for whoever aggregates these runs.

    A v0 rollout that ends on `max_turns` comes back `is_truncated=True` with no
    v0 stop name, and `_v1_stop_condition` (`v1/legacy.py:233-241`) falls back to
    **`"max_output_tokens"`** for anything it cannot map. So every control-arm
    rollout that simply spends its skill-call budget is labelled as a token
    truncation on the trace, while the native arm reports its own
    `"call_budget_exhausted"`.

    Nothing is broken — but an aggregator that groups by `stop_condition` will
    silently claim the control arm ran out of tokens on every seed. This test
    exists so that fact is discovered here rather than in a results table.
    """
    control, _control_calls, _native, _task, _advertised = routes
    assert control.stop_condition == "max_output_tokens"


def test_the_referee_bookkeeping_reaches_the_trace(routes):
    """`Trace.state` is `exclude=True`, so the referee's numbers only reach
    `traces.jsonl` through `NetHackTask.finalize`. Without this the executed-call
    count — the quantity the arm comparison is normalized on — would be
    invisible in a run's output."""
    _control, _control_calls, native, _task, _advertised = routes
    assert native.metrics["skill_calls"] == float(len(SEQUENCE))
    # The netplay gate, measured rather than asserted by construction.
    assert native.metrics["moves_executed"] == 0.0
    assert native.metrics["budget_exhausted"] == 0.0


def test_num_turns_is_not_a_cross_route_comparable(routes):
    """Pin the reason `num_turns` is deliberately excluded above.

    On the control route one model turn dispatches at most one skill call, so
    `num_turns` counts game actions. On the native route the tool call *is* the
    dispatch and the agent's turn structure is its own business — an MCP-driven
    CLI agent can batch several skill calls into one turn or spend turns on no
    tool call at all. Asserting `num_turns` equality across the routes would
    assert that two different scaffolds take the same number of turns, which is
    the very thing this experiment measures. `skill_calls` is the comparable.
    """
    control, _control_calls, native, _task, _advertised = routes
    assert control.num_turns == len(SEQUENCE) + 1
    # No harness program ran on the native side, so it has no model turns at all
    # — the clearest possible demonstration that the two counts measure
    # different things.
    assert native.num_turns == 0
    # What IS comparable: the number of game actions each route executed.
    assert native.state.skill_calls == control.num_turns - 1 == len(SEQUENCE)
