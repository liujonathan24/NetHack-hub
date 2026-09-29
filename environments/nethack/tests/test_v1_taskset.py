"""Tests for the Verifiers v1 (0.2.x) taskset port of NetHack (``nethack_v1``).

Covers:
  * ``load_taskset`` / ``load_harness`` / ``load_v1_environment`` build for BOTH
    curriculum tasks (``full_nle`` and ``six_floor_primitives``).
  * The v1 rewards / per-rollout toolset / default harness are wired.
  * **The control arm (arm 0)** runs end-to-end through 0.2.x's v0 legacy
    bridge with a keyless mock model: setup creates the engine, the v0 loop
    steps it, rewards are scored, and the result maps to a serializable v1
    ``Trace``. This is the 0.2.x replacement for the old in-process
    ``env.rollout(...)`` test — ``Environment`` has no ``rollout()`` in 0.2.x
    (rollouts launch an external harness program), and the control arm no
    longer runs through the v1 taskset at all (see the ``nethack_v1`` module
    docstring).
  * The v0 path (``load_environment`` / ``NetHackVerifiersEnv``) still imports
    and builds — no regression.

Run with the engine on the path, e.g.:
    NLE_LIB_PATH=/.../libnethack.so \
    PYTHONPATH=environments/nethack:/path/to/NetHackHarness \
    pytest environments/nethack/tests/test_v1_taskset.py
"""

import asyncio
import pathlib
import sys

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[2] / "environments" / "nethack")
)

import nethack_v1 as m  # noqa: E402
import nethack_v1  # noqa: E402  (module name used by the crash-capture tests)
from verifiers.clients import Client  # noqa: E402
from verifiers.types import Response, ResponseMessage, ToolCall  # noqa: E402
from verifiers.v1.legacy import rollout_output_to_trace  # noqa: E402


class _MockClient(Client):
    """Keyless mock model: always emits a single valid skill tool call.

    Subclasses the v0 ``Client`` ABC because ``run_rollout`` resolves its client
    through ``resolve_client`` (`verifiers/clients/__init__.py:15`), which
    rejects duck-typed objects. Only ``get_response`` is exercised.
    """

    def __init__(self, tool_name: str = "search", arguments: str = "{}"):
        self._config = None
        self._client = None
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0

    def setup_client(self, config):
        return None

    async def close(self):
        return None

    async def to_native_tool(self, tool):
        return tool

    async def to_native_prompt(self, messages):
        return messages, {}

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kw):
        return None

    async def raise_from_native_response(self, response):
        return None

    async def from_native_response(self, response):
        return response

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
        # Taskset wiring. `select()` materializes the rows (0.2.x replaced
        # `Taskset.get_dataset()`; `taskset.py:79-103`).
        tasks = env.taskset.select()
        assert len(tasks) == 2
        # Rewards are @reward methods on the Task in 0.2.x, discovered by name
        # and scored concurrently (`task.py:301-308`), so what must hold is the
        # exact set of names AND their v0 weights — the declaration order the
        # 0.1.14 `Taskset(rewards=[...])` list carried is not a guarantee the
        # new API has (discovery sorts by priority then name).
        from verifiers.v1.decorators import discover_decorated

        discovered = discover_decorated(tasks[0], "reward")
        assert {fn.__name__: getattr(fn, "_vf_weight") for fn in discovered} == {
            "scout_reward": 1.0,
            "descent_reward": 10.0,
            "success_reward": 100.0,
            "ascension_reward": 1000.0,
        }
        # The spec system prompt rides on the row in 0.2.x (`task.py:153`).
        assert tasks[0].data.system_prompt
        # Exactly one tool server, declared PER ROLLOUT (Task.tools, not
        # Taskset.tools) so each rollout gets its own engine.
        assert type(tasks[0]).tools == (m.NetHackToolset,)
        assert type(env.taskset).tools == ()
        # Default harness.
        assert isinstance(env.harness, m.NetHackHarness)
        assert env.harness.SUPPORTS_MCP is True
        # Rows carry a seed and a tier; max_turns is framework-level now.
        assert isinstance(tasks[0].data.seed, int)
        assert tasks[0].data.tier
        assert env.config.max_turns == 3


def test_rewards_read_the_typed_state_and_keep_their_v0_meaning():
    # The four rewards delegate to the v0 implementations verbatim; only their
    # input moved (typed `trace.state` scalars instead of the v0 state dict).
    task = m.load_taskset({"task_spec": "full_nle", "n_examples": 1}).select(1)[0]

    class _Trace:
        def __init__(self, state):
            self.state = state

    trace = _Trace(
        m.NetHackState(
            scout_reward_total=0.25,
            descent_count=3.0,
            succeeded=True,
            ascended=False,
        )
    )
    assert asyncio.run(task.scout_reward(trace)) == 0.25
    assert asyncio.run(task.descent_reward(trace)) == 3.0
    assert asyncio.run(task.success_reward(trace)) == 1.0
    assert asyncio.run(task.ascension_reward(trace)) == 0.0


def test_control_arm_runs_through_the_v0_legacy_bridge():
    """Arm 0 end-to-end on 0.2.x, with NO v1 port in its path.

    ``EnvConfig.id`` + no ``taskset.id`` selects the bridge (`v1/env.py:128`),
    which runs the v0 rollout verbatim (`v1/legacy.py:517`) and maps it to a v1
    ``Trace``. This test drives exactly those two calls, so a change that broke
    the control arm's v0 loop, its engine lifecycle, or its reward wiring fails
    here.
    """
    from nethack import load_environment

    env = load_environment(
        n_examples=1,
        max_turns=2,
        explicit_seeds=[0],
        skill_set="netplay",
        character="Val-hum-neu-fem",
    )
    row = dict(env.get_eval_dataset()[0])
    client = _MockClient("search")
    out = asyncio.run(
        env.run_rollout(
            input=row,
            client=client,
            model="mock-model",
            sampling_args={},
            state_columns=["trajectory"],
        )
    )
    trace = rollout_output_to_trace(out, 0)

    # The mock made exactly max_turns model requests.
    assert client.calls == 2
    assert trace.num_turns == 2
    # Rewards were scored and recorded.
    assert isinstance(trace.reward, (int, float))
    for name in (
        "scout_reward",
        "descent_reward",
        "success_reward",
        "ascension_reward",
    ):
        assert name in (trace.metrics or {})
    # The engine is gone and the trace is serializable (no engine/numpy/sets).
    assert trace.model_dump_json()


def test_v0_path_still_builds():
    # No-regression: the v0 entrypoint still constructs a StatefulToolEnv.
    from nethack import load_environment, NetHackVerifiersEnv

    env = load_environment(n_examples=1, max_turns=2)
    assert isinstance(env, NetHackVerifiersEnv)


def test_dataset_rows_do_not_shadow_the_reserved_task_field():
    # `flatten_task_input` (verifiers/types.py:811) REPLACES the whole rollout
    # input with `input["task"]` when present, so a row carrying a bare
    # `{"tier": ..., "seed": ...}` under `task` silently loses its prompt and
    # `init_state` raises KeyError('prompt'). The seed rides in `info` instead.
    from nethack import load_environment
    from verifiers.types import flatten_task_input

    env = load_environment(n_examples=1, explicit_seeds=[123])
    row = dict(env.get_eval_dataset()[0])
    assert "task" not in row
    assert row["info"]["seed"] == 123
    assert "prompt" in flatten_task_input(row)


# --------------------------------------------------------------------------- #
# a fatal NATIVE crash of the tool server must leave evidence
# --------------------------------------------------------------------------- #

def test_a_fatal_native_signal_leaves_a_dump_in_the_attempt_directory(tmp_path):
    """The whole postmortem for a segfaulted tool server used to be deleted.

    The env drives NetHack through a C extension. A SIGSEGV/SIGABRT there
    kills `python -m nethack_v1` with no Python traceback; nothing in the eval
    CLI waits on the child, and its merged stdout/stderr lives in the runtime
    workdir that `SubprocessRuntime.cleanup()` `shutil.rmtree`s -- so the
    evidence is destroyed by the same teardown that fails to notice the death.
    Measured on run treesmoke2: three attempts, three fatal signals (11/6/11 in
    /var/log/apport.log), zero bytes of evidence in the run directory.

    `arm_crash_capture()` arms faulthandler on a file in the ATTEMPT
    directory, which outlives the runtime. A subprocess is the only honest way
    to assert this: the signal has to actually be fatal.
    """
    import json
    import os
    import subprocess

    crash_dir = tmp_path / "a001"
    script = (
        "import faulthandler, nethack_v1\n"
        "print(nethack_v1.arm_crash_capture())\n"
        "faulthandler._sigsegv()\n"
    )
    env = dict(os.environ)
    env["NETHACK_ENV_CRASH_DIR"] = str(crash_dir)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=300)

    assert proc.returncode < 0, f"expected a fatal signal, got {proc.returncode}"
    log = crash_dir / nethack_v1.CRASH_LOG_NAME
    assert log.is_file(), "the crash log must be where the ATTEMPT can find it"
    text = log.read_text(errors="replace")
    assert "[env-server] start pid=" in text, "the header identifies the server"
    assert nethack_v1.CRASH_MARKER in text
    assert "Segmentation fault" in text


def test_arming_the_crash_capture_takes_its_path_from_the_served_config():
    """`trace_dir` is the one durable directory the tool server is told about,
    so the dump lands next to the attempt's turn files with no new plumbing."""
    import json
    import os
    import tempfile

    keys = ("VF_CONFIG", "NETHACK_ENV_CRASH_DIR", "NLE_CRASH_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        turns = os.path.join(tmp, "turns")
        prev = {k: os.environ.get(k) for k in keys}
        for k in ("NETHACK_ENV_CRASH_DIR", "NLE_CRASH_DIR"):
            os.environ.pop(k, None)
        os.environ["VF_CONFIG"] = json.dumps({"trace_dir": turns})
        try:
            path = nethack_v1.arm_crash_capture()
            nle_crash_dir = os.environ.get("NLE_CRASH_DIR")
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert path == os.path.join(tmp, nethack_v1.CRASH_LOG_NAME)
        assert os.path.isfile(path)
        # The ENGINE's own dumper is pointed at the same durable directory.
        # It defaults to "." -- the runtime workdir teardown deletes -- and it
        # is the only channel that fires for a fault inside libnethack.so,
        # because the sentinel's handlers override CPython's faulthandler.
        assert nle_crash_dir == tmp


if __name__ == "__main__":
    test_v1_env_builds_for_both_tasks()
    print("build-both: OK")
    test_control_arm_runs_through_the_v0_legacy_bridge()
    print("legacy-bridge rollout: OK")
    test_v0_path_still_builds()
    print("v0-path: OK")
    print("ALL PASS")
