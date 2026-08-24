"""The Prime Agent arm: plugin resolution and the wiring the arm depends on.

Nothing here launches Prime Agent — the acceptance evidence that a real agent
played is the committed rollout under `tools/cli_harness_eval/acceptance/` and
Task 10's report. These tests pin the parts that are cheap to break silently:
the module contract the stock loader needs, the config parity with the other two
arms, and the three per-rollout files whose contents decide *which model* and
*which server* the agent actually reaches.
"""

from __future__ import annotations

import json
import logging
import pathlib
import tomllib

import pytest

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "tools/cli_harness_eval/configs"


# -- plugin resolution ------------------------------------------------------


def test_module_exports_exactly_one_harness():
    import nethack_prime_agent as m
    from verifiers.v1.harness import Harness

    exported = [getattr(m, n) for n in m.__all__]
    harnesses = [
        o
        for o in exported
        if isinstance(o, type) and issubclass(o, Harness) and o is not Harness
    ]
    assert len(harnesses) == 1, "the loader refuses a module exporting 0 or 2+"
    assert harnesses[0].__name__ == "PrimeAgentHarness"


def test_id_resolves_through_the_stock_loader():
    """`harness.id = "nethack-prime-agent"` -> the external top-level module.

    Stock 0.2.1 does this on its own (`loaders.py:33-45` probes the namespaced
    candidate, then falls back to the bare module name); no vendored patch.
    """
    from verifiers.v1.loaders import harness_class

    cls = harness_class("nethack-prime-agent")
    assert cls.__name__ == "PrimeAgentHarness"
    assert cls.__module__ == "nethack_prime_agent"
    assert cls.SUPPORTS_MCP is True


def test_the_harness_declares_the_capabilities_the_arm_needs():
    from nethack_prime_agent import PrimeAgentHarness

    # Without SUPPORTS_MCP the env hands the harness no `mcp_urls` at all.
    assert PrimeAgentHarness.SUPPORTS_MCP is True
    # `--append-system-prompt` exists, so the NetHack spec prompt is a real
    # system message here as on arm 1 — not folded into the user turn.
    assert PrimeAgentHarness.APPENDS_SYSTEM_PROMPT is True
    assert PrimeAgentHarness.SUPPORTS_MESSAGE_PROMPT is False


# -- the skill package shipped inside the distribution ----------------------


def test_the_skill_package_ships_and_targets_the_right_import_name():
    """The kernel import name IS the `mcpServers` key (mcp-integrations.md), and
    the toolset's `TOOL_PREFIX` fixes that key to `nethack`."""
    from importlib import resources

    skill = resources.files("nethack_prime_agent") / "skill"
    module = (skill / "src/nethack/__init__.py").read_text()
    assert 'server = "nethack"' in module
    assert "McpIntegration" in module
    # `url` must stay unset: the port is assigned per rollout and resolved from
    # the host's mcpServers entry. A literal URL here would pin one rollout's
    # port into every later one.
    assert "url = None" in module
    assert (skill / "SKILL.md").read_text().startswith("---\nname: nethack\n")
    assert 'name = "prime-agent-skill-nethack"' in (skill / "pyproject.toml").read_text()


def test_disabled_tools_is_refused_rather_than_silently_ignored():
    """Prime Agent has one built-in tool and an allowlist, no denylist. A config
    that asked for a denylist would otherwise run an arm that differs from what
    the config says."""
    import asyncio

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="nethack-prime-agent", disabled_tools=["ipython"])
    )
    with pytest.raises(ValueError, match="disabled_tools"):
        asyncio.run(harness.launch(None, None, None, "", "", {}))


# -- arm configs ------------------------------------------------------------


def _load(name):
    return tomllib.loads((CONFIGS / name).read_text())


def test_the_three_arms_pin_the_same_experiment():
    control, claude, prime = (
        _load("control.toml"),
        _load("claude_code.toml"),
        _load("prime_agent.toml"),
    )
    # The control arm's knobs live under [args] (legacy bridge); the CLI arms'
    # under [taskset].
    c_args, cc, pa = control["args"], claude["taskset"], prime["taskset"]

    assert control["model"] == claude["model"] == prime["model"] == "z-ai/glm-5.2"
    for key in ("task_spec", "character", "explicit_seeds", "n_examples", "interface",
                "variant", "map_detail"):
        assert c_args[key] == cc[key] == pa[key], key
    # Parity is the invariant; the SET is an experiment variable (18-tool `netplay`
    # vs the 31 vendored upstream skills in `netplay_true`).
    assert (
        c_args["skill_set"]
        == claude["taskset"]["env_args"]["skill_set"]
        == prime["taskset"]["env_args"]["skill_set"]
    )
    assert c_args["skill_set"] in ("netplay", "netplay_true")
    assert control["num_tasks"] == claude["num_tasks"] == prime["num_tasks"] == 16
    # Budget: exactly-150 on both CLI arms, at-most-150 on the control arm.
    assert cc["max_skill_calls"] == pa["max_skill_calls"] == 150
    assert c_args["max_turns"] == 150
    # Both CLI arms must seed the workspace, or they are not capability-matched
    # to the control arm's system prompt / wiki tools / Journal.
    assert cc["seed_workspace"] is True and pa["seed_workspace"] is True
    # Distinct per-turn NDJSON directories, or one arm overwrites the other.
    # A `a != b != c` chain here would only check (a != b) and (b != c) and
    # never compare a against c, so it would pass even if cc and control
    # shared a directory -- check pairwise via set cardinality instead. (Full
    # coverage, including the absolute-path requirement for all three arms,
    # lives in tests/test_arm_configs.py.)
    trace_dirs = {cc["trace_dir"], pa["trace_dir"], control["args"]["trace_dir"]}
    assert len(trace_dirs) == 3, trace_dirs


def test_the_prime_agent_arms_trace_dir_is_absolute():
    """REGRESSION, measured. `trace_dir` is resolved by the TOOL SERVER process,
    whose cwd is its own runtime workdir (`/tmp/vf-<id>`), and that directory is
    deleted at teardown. A relative value therefore discards the entire per-turn
    NDJSON with no error: two otherwise identical rollouts produced no file
    (relative) and 12 lines (absolute).

    Only this arm's config is asserted here. `control.toml` and
    `claude_code.toml` got the same absolute-path fix in Task 14 (commit
    659601e); `tests/test_arm_configs.py` covers all three arms.
    """
    trace_dir = _load("prime_agent.toml")["taskset"]["trace_dir"]
    assert pathlib.PurePosixPath(trace_dir).is_absolute(), trace_dir


def test_the_prime_agent_arm_selects_the_external_harness_on_the_subprocess_runtime():
    prime = _load("prime_agent.toml")
    assert prime["harness"]["id"] == "nethack-prime-agent"
    assert prime["harness"]["runtime"]["type"] == "subprocess"
    # An unpinned CLI version means a background `npm -g` upgrade silently
    # changes the scaffold under test between seeds.
    assert prime["harness"]["version"]
    assert "disabled_tools" not in prime["harness"]


# -- the per-rollout files that decide model + server -----------------------


class _RecordingRuntime:
    """Captures `runtime.write` / `run` instead of touching a filesystem.

    `exit_code` is constant across every `run_program` call in a test: enough
    to exercise the relaunch loop's own decision (episode-live vs. not), since
    a non-zero exit is never reached mid-loop in these tests -- a crash on the
    first attempt already keeps the loop from starting (`PrimeAgentHarness.
    launch` only relaunches while `result.exit_code == 0`).
    """

    def __init__(self, exit_code: int = 0):
        self.files: dict[str, bytes] = {}
        self.programs: list[tuple[list[str], dict[str, str]]] = []
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self._exit_code = exit_code

    async def write(self, path, data):
        self.files[path] = data

    async def run(self, argv, env):
        from verifiers.v1.runtimes import ProgramResult

        self.commands.append((argv, env))
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def run_program(self, argv, env):
        from verifiers.v1.runtimes import ProgramResult

        self.programs.append((argv, env))
        return ProgramResult(exit_code=self._exit_code, stdout="ok", stderr="")


class _FakeNetHackState:
    """Stands in for `nethack_v1.NetHackState` as synced onto `trace.state` over
    the interception `/state` channel -- only the fields `PrimeAgentHarness.
    _episode_live` reads."""

    def __init__(self, *, terminated: bool = True, budget_exhausted: bool = False,
                 skill_calls: int = 0):
        self.terminated = terminated
        self.budget_exhausted = budget_exhausted
        self.skill_calls = skill_calls


class _FakeTrace:
    """Stands in for `verifiers.v1.trace.Trace`: carries `.state` (relaunch's
    liveness read) and a `record_metric` matching `Trace.record_metric`'s exact
    semantics (last-write, coerced to float), so `trace.metrics` is inspectable
    after `launch` returns."""

    def __init__(self, *, id="trace-abc", task=None, state=None):
        self.id = id
        self.task = task
        self.state = state if state is not None else _FakeNetHackState()
        self.metrics: dict[str, float] = {}

    def record_metric(self, name, value):
        self.metrics[name] = float(value)


def _build_trace(state=None):
    import types

    from verifiers.v1.task import TaskData

    return _FakeTrace(
        task=types.SimpleNamespace(
            data=TaskData(
                idx=0, prompt="Play NetHack.", system_prompt="You are a Valkyrie."
            )
        ),
        state=state,
    )


def _launch(**config_overrides):
    """Drive `launch` against the recording runtime and return what it wrote.

    Uses a `terminated=True` fake state by default (the referee's own read on
    "the episode is over"), so every pre-existing test here -- none of which
    knows about relaunching -- keeps seeing exactly the one `run_program` call
    it always has.
    """
    import asyncio
    import types

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="nethack-prime-agent", **config_overrides)
    )
    ctx = types.SimpleNamespace(
        model="z-ai/glm-5.2",
        sampling=types.SimpleNamespace(reasoning_effort=None),
    )
    trace = _build_trace()
    runtime = _RecordingRuntime()
    asyncio.run(
        harness.launch(
            ctx,
            trace,
            runtime,
            "http://127.0.0.1:9999/v1",
            "vf-secret",
            {"nethack": "http://127.0.0.1:41449"},
        )
    )
    return runtime


def _launch_with_state(state, runtime=None, **config_overrides):
    """Like `_launch`, but returns `(runtime, trace)` and takes an explicit
    fake `NetHackState`, for the relaunch-decision tests."""
    import asyncio
    import types

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="nethack-prime-agent", **config_overrides)
    )
    ctx = types.SimpleNamespace(
        model="z-ai/glm-5.2",
        sampling=types.SimpleNamespace(reasoning_effort=None),
    )
    trace = _build_trace(state=state)
    runtime = runtime if runtime is not None else _RecordingRuntime()
    asyncio.run(
        harness.launch(
            ctx,
            trace,
            runtime,
            "http://127.0.0.1:9999/v1",
            "vf-secret",
            {"nethack": "http://127.0.0.1:41449"},
        )
    )
    return runtime, trace


def test_models_json_routes_the_agent_through_interception_not_a_vendor():
    """The whole validity of the arm rests on this file: if Prime Agent resolved
    a built-in provider instead, the arm would quietly run a different model."""
    runtime = _launch()
    models = json.loads(next(v for k, v in runtime.files.items() if k.endswith("models.json")))
    provider = models["providers"]["intercept"]
    assert provider["baseUrl"] == "http://127.0.0.1:9999/v1"
    assert provider["api"] == "openai-completions"
    # The secret is passed by env-var NAME, so it never lands on disk.
    assert provider["apiKey"] == "PRIME_AGENT_INTERCEPT_KEY"
    assert b"vf-secret" not in b"".join(runtime.files.values())
    assert [m["id"] for m in provider["models"]] == ["z-ai/glm-5.2"]

    argv, env = runtime.programs[0]
    assert argv[argv.index("--provider") + 1] == "intercept"
    assert argv[argv.index("--model") + 1] == "z-ai/glm-5.2"
    assert env["PRIME_AGENT_INTERCEPT_KEY"] == "vf-secret"


def test_settings_json_wires_the_per_rollout_mcp_server_and_the_skill():
    runtime = _launch()
    settings = json.loads(
        next(v for k, v in runtime.files.items() if k.endswith("settings.json"))
    )
    server = settings["mcpServers"]["nethack"]
    # Only remote "http" servers are wired through to Prime Agent's kernel.
    assert server["type"] == "http"
    assert server["url"] == "http://127.0.0.1:41449"
    assert server["bearerTokenEnvVar"] == "NETHACK_MCP_TOKEN"
    # Fallback pins too, so no resolution path can reach a built-in provider.
    assert settings["defaultProvider"] == "intercept"
    assert settings["defaultModel"] == "z-ai/glm-5.2"
    # Capability match with arm 1's `--bare`: no networked search skill.
    assert settings["bundledSkills"]["websearch"] is False
    # The skill directory must NOT be per-rollout, or the kernel venv is rebuilt
    # on every seed (Prime Agent keys it on the Python-skill paths).
    assert settings["skills"] == ["/tmp/vf-prime-agent/skills"]
    assert "trace-abc" not in json.dumps(settings)

    _, env = runtime.programs[0]
    assert env["NETHACK_MCP_TOKEN"]
    assert env["PRIME_AGENT_CODING_AGENT_DIR"].endswith("agent-trace-abc")


def test_the_prompt_is_passed_after_an_option_terminator():
    """Prime Agent treats bare `@word` as a file attachment and a leading `-` as
    a flag; the task prompt is neither."""
    runtime = _launch()
    argv, _ = runtime.programs[0]
    assert argv[-2:] == ["--", "Play NetHack."]
    assert "--no-session" in argv and "--print" in argv
    # `--append-system-prompt` was REMOVED on 2026-08-21 (see the de-dup note in
    # __init__.py): AGENTS.md already carries the resolved system prompt and
    # Prime Agent embeds it as Project Context, so passing both delivered the
    # gameplay block twice. This asserts the flag stays gone -- re-adding it
    # silently doubles the prompt, which is what the v3 seed-2 post-mortem found.
    assert "--append-system-prompt" not in argv


def test_pythonpath_is_unset_before_the_agent_starts():
    """REGRESSION, measured in the first smoke rollout.

    The tool server needs `environments/nethack` on PYTHONPATH; the agent must
    not have it. The kernel import name for an MCP integration is the
    `mcpServers` key, which the toolset's `TOOL_PREFIX` fixes to `nethack` — and
    `environments/nethack/nethack.py` is a module of that exact name. With
    PYTHONPATH inherited, `import nethack` in the kernel resolved to the v0
    environment module, Prime Agent reported `<unavailable Python skill
    'nethack': No module named 'verifiers'>`, and the agent spent 131 model turns
    reading the experiment's own source tree instead of playing (2 skill calls).
    """
    runtime = _launch()
    argv, _ = runtime.programs[0]
    assert argv[:3] == ["sh", "-c", 'unset PYTHONPATH; exec "$@"']
    # $0 is a label, not a program; the real argv starts after it.
    assert argv[4] == "prime-agent"


def test_the_daemon_socket_is_left_shared_on_purpose():
    """Prime Agent is daemon-backed even under `--print`, and its supervisor socket
    dir is per-USER (`<tmpdir()>/prime-agent-<uid>`). Two isolation attempts were
    measured and rejected: a per-rollout TMPDIR stops the daemon coming up at all
    ("Timed out waiting for daemon to start", reproduced outside the harness), and
    a `shutdown --force` in teardown would kill every concurrent rollout because
    the socket is shared. This test pins the decision so neither is reintroduced
    without re-measuring.
    """
    runtime = _launch()
    _, env = runtime.programs[0]
    assert "TMPDIR" not in env
    assert not [argv for argv, _ in runtime.commands if "shutdown" in argv]
    # And the config dir must survive: the shared supervisor keeps live state
    # under it, so deleting it kills the daemon and the NEXT rollout fails with
    # `DaemonSocketClosedError`. Measured twice; see the comment in `launch`.
    assert not [argv for argv, _ in runtime.commands if argv[:2] == ["rm", "-rf"]]


# -- the declared context window (the run-1 mid-action cutoffs) --------------
#
# Three of five run-1 rollouts ended `agent_completed` mid-action, on a tool call
# that never got a follow-up, with 107-130 of 400 skill calls and 34-43 of 120
# minutes used. Cause: `models.json` declared no `contextWindow`, Prime Agent
# normalizes that to 128000 (`definition.contextWindow ?? 128e3`), and its
# session host ends the agent loop the first time context exceeds
# `contextWindow - reserveTokens` = 111,616 tokens
# (`_shouldStopForThresholdCompaction` -> `shouldStopAfterTurn`). Under `--print`
# nothing resumes it: the CLI exits 0 with no output, which verifiers can only
# record as a clean completion. Reproduced against a stub model reporting a
# controlled `usage.total_tokens` ramp: omitted -> stops at 120,000; declared as
# 300000 -> stops at 290,000 (i.e. at `contextWindow - 16384` either way).


def test_the_declared_context_window_reaches_models_json():
    runtime = _launch(context_window=1048576)
    models = json.loads(
        next(v for k, v in runtime.files.items() if k.endswith("models.json"))
    )
    model_def = models["providers"]["intercept"]["models"][0]
    assert model_def["contextWindow"] == 1048576


def test_an_unset_context_window_is_omitted_and_warned_about(caplog):
    """0 keeps the old wire format rather than guessing a window, but it must
    not pass silently -- the failure it causes is invisible in the trace."""
    with caplog.at_level(logging.WARNING, logger="nethack_prime_agent"):
        runtime = _launch()
    models = json.loads(
        next(v for k, v in runtime.files.items() if k.endswith("models.json"))
    )
    assert "contextWindow" not in models["providers"]["intercept"]["models"][0]
    assert any("context_window is unset" in r.getMessage() for r in caplog.records)


def test_the_shipped_arm_config_declares_the_models_real_context_window():
    """A regression guard on the config, not just the code: the cutoff comes
    back the moment this value goes missing."""
    prime = _load("prime_agent.toml")
    assert prime["harness"]["context_window"] == 1048576
    # Threshold headroom sanity: the cutoffs happened at ~110k tokens, so the
    # declared window must leave far more than that before `- reserveTokens`.
    assert prime["harness"]["context_window"] - 16384 > 500_000


def test_the_cli_streams_are_persisted_for_the_next_silent_exit():
    """Verifiers only surfaces a harness program's output on a non-zero exit, so
    an exit-0 truncation leaves nothing to read. Keep both streams."""
    runtime = _launch()
    assert runtime.files["/tmp/vf-prime-agent/agent-trace-abc/program.stdout.txt"] == b"ok"
    assert runtime.files["/tmp/vf-prime-agent/agent-trace-abc/program.stderr.txt"] == b""


def test_output_capture_can_be_turned_off():
    runtime = _launch(capture_output=False)
    assert not [k for k in runtime.files if k.endswith("program.stdout.txt")]


# -- auto-resume relaunching --------------------------------------------------
#
# `stop_condition = "agent_completed"` (`verifiers/v1/harness.py:109-118`,
# stock, read-only) only means the CLI process exited 0 with no `@stop` firing
# -- under Prime Agent's `--print` mode that happens the instant the model
# stops emitting tool calls, whether or not the character is alive. Measured:
# 15-25% of rollouts stop this way, sessions ending anywhere from 0 to ~170 of
# a 400-call budget. `launch` cannot rely on the stock `Harness.run` loop to
# fix this (read-only, not vendored) -- it has to decide for itself, from
# `trace.state` (synced live from the toolset over the `/state` channel), and
# loop.


def test_relaunches_while_the_game_is_still_live_and_budget_remains():
    live = _FakeNetHackState(terminated=False, budget_exhausted=False, skill_calls=42)
    runtime, trace = _launch_with_state(live, max_relaunches=3)
    # One initial attempt plus exactly `max_relaunches` more: the fake state
    # never changes, so "still alive" holds at every check and only the cap
    # stops the loop.
    assert len(runtime.programs) == 1 + 3
    assert trace.metrics["prime_agent_relaunches"] == 3.0


def test_does_not_relaunch_after_the_character_dies():
    dead = _FakeNetHackState(terminated=True, budget_exhausted=False, skill_calls=42)
    runtime, trace = _launch_with_state(dead, max_relaunches=5)
    assert len(runtime.programs) == 1
    assert trace.metrics["prime_agent_relaunches"] == 0.0


def test_does_not_relaunch_once_the_call_budget_is_exhausted():
    """Alive but out of calls is also a real stop, not a false one -- the
    toolset's own referee (`NetHackState.budget_exhausted`) binds every arm,
    including a CLI agent whose internal loop this harness does not control."""
    out_of_calls = _FakeNetHackState(terminated=False, budget_exhausted=True)
    runtime, trace = _launch_with_state(out_of_calls, max_relaunches=5)
    assert len(runtime.programs) == 1
    assert trace.metrics["prime_agent_relaunches"] == 0.0


def test_a_crashing_first_attempt_is_not_relaunched():
    """A non-zero exit is a real failure the stock `Harness.run` must still
    raise on (`result.exit_code != 0` -> `HarnessError`) -- relaunching over it
    would mask a crash as a resumable pause."""
    live = _FakeNetHackState(terminated=False, budget_exhausted=False)
    crashing = _RecordingRuntime(exit_code=1)
    runtime, trace = _launch_with_state(live, runtime=crashing, max_relaunches=5)
    assert len(runtime.programs) == 1
    assert trace.metrics["prime_agent_relaunches"] == 0.0


def test_relaunches_still_respect_the_declared_zero_cap():
    """`max_relaunches = 0` is a valid way to disable the feature entirely
    without touching the relaunch-decision code path."""
    live = _FakeNetHackState(terminated=False, budget_exhausted=False)
    runtime, trace = _launch_with_state(live, max_relaunches=0)
    assert len(runtime.programs) == 1
    assert trace.metrics["prime_agent_relaunches"] == 0.0


def test_relaunch_argv_swaps_in_the_resume_prompt_not_the_original_task():
    """The MCP tool server (and the game it owns) is per-rollout, not
    per-CLI-process, so it is exactly where the last attempt left it -- only
    the agent's own conversation is gone. Re-sending "Task: ... Begin." after
    hundreds of tool calls would read as an instruction to restart, so a
    relaunch gets a distinct resume prompt instead."""
    from nethack_prime_agent import _RESUME_PROMPT

    live = _FakeNetHackState(terminated=False, budget_exhausted=False)
    runtime, _ = _launch_with_state(live, max_relaunches=2)
    assert len(runtime.programs) == 3
    first_argv, _ = runtime.programs[0]
    assert first_argv[-2:] == ["--", "Play NetHack."]
    for argv, _ in runtime.programs[1:]:
        assert argv[-2:] == ["--", _RESUME_PROMPT]
        # Everything else about the relaunch invocation is unchanged.
        assert argv[:-1] == first_argv[:-1]


def test_relaunch_output_is_persisted_without_overwriting_the_first_attempt():
    live = _FakeNetHackState(terminated=False, budget_exhausted=False)
    runtime, _ = _launch_with_state(live, max_relaunches=2)
    prefix = "/tmp/vf-prime-agent/agent-trace-abc/"
    assert runtime.files[f"{prefix}program.stdout.txt"] == b"ok"
    assert runtime.files[f"{prefix}program.stdout.relaunch1.txt"] == b"ok"
    assert runtime.files[f"{prefix}program.stdout.relaunch2.txt"] == b"ok"


# --- allow_batching -------------------------------------------------------
# Prime obeys SKILL.md's "Do not batch blind sequences of calls" (measured
# 0.80-0.95 skills per ipython call across m2/m3/g2). Claude Code was never
# given an equivalent rule and batched at 2.31 tool calls per assistant turn,
# buying ~2.3x the game actions per decision on the same budget. The asymmetry
# is ours; this flag lets Prime run unconstrained so both directions can be
# measured.

def test_allow_batching_defaults_off():
    from nethack_prime_agent import PrimeAgentHarnessConfig

    assert PrimeAgentHarnessConfig().allow_batching is False


def test_strip_removes_the_rule_and_leaves_the_rest_intact():
    from importlib import resources

    import nethack_prime_agent as pkg
    from nethack_prime_agent import _strip_no_batch_rule

    raw = (resources.files(pkg) / "skill" / "SKILL.md").read_bytes()
    # Assert against the CONSTANT, not a hardcoded phrase. This test duplicated
    # the wording ("Do not batch"), so when the honesty pass reworded the rule
    # both the constant and this fixture went stale together -- and the red test
    # was read as pre-existing noise rather than as "allow_batching now raises".
    rule = pkg._NO_BATCH_RULE
    assert rule in raw.decode(), "fixture assumption: the rule ships in SKILL.md"

    out = _strip_no_batch_rule(raw)
    assert rule not in out.decode()
    # Everything else the agent needs must survive — this file is its only
    # instruction sheet.
    # Re-anchored 2026-08-23: the honesty pass rewrote SKILL.md and these three
    # phrases went with it, so this guard had been asserting against a document
    # that no longer existed. Anchors below are lines the current doc actually
    # carries; keep them in sync with the doc, not with memory of it.
    for keep in (b"Every tool is **async**", b"await nethack.request_map()",
                 b"np_melee_attack", b"=== MAP ==="):
        assert keep in out, f"stripping removed unrelated guidance: {keep!r}"


def test_strip_fails_loudly_if_the_rule_text_drifts():
    """A silent no-op would ship the constrained prompt while the config claims
    batching is enabled — an invalid experiment that looks like a valid one."""
    import pytest

    from nethack_prime_agent import _strip_no_batch_rule

    with pytest.raises(RuntimeError, match="stale"):
        _strip_no_batch_rule(b"# SKILL\nsome other content\n")


# -- E13: the shared continual-harness store ---------------------------------
#
# Prime Agent's continual harness (prompt notes, memories, reusable skill
# descriptions, sub-agent specs) is rendered into the system prompt of every new
# session, which makes it the cross-episode learning channel E13 is built on.
# Under this arm it is inert by default: `--no-session` kills the local store,
# and the global one is `<agent_dir>/harness` with `agent_dir` per-rollout. The
# tests below pin the one link that changes that, and the two ways it could
# silently do nothing.


class _RecordingRuntimeWithWorkdir(_RecordingRuntime):
    """`_RecordingRuntime` plus the `workdir` attribute `launch` requires before
    it will build a sandbox prefix."""

    workdir = "/tmp/vf-workdir"


def _launch_sandboxed(**config_overrides):
    """`_launch`, but on a runtime that has a `workdir` -- the precondition for
    `launch` to build a bwrap prefix at all."""
    import asyncio
    import types

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="nethack-prime-agent", **config_overrides)
    )
    ctx = types.SimpleNamespace(
        model="z-ai/glm-5.2", sampling=types.SimpleNamespace(reasoning_effort=None)
    )
    runtime = _RecordingRuntimeWithWorkdir()
    asyncio.run(
        harness.launch(
            ctx,
            _build_trace(),
            runtime,
            "http://127.0.0.1:9999/v1",
            "vf-secret",
            {"nethack": "http://127.0.0.1:41449"},
        )
    )
    return runtime


def test_no_continual_harness_dir_means_no_symlink_and_no_extra_command():
    """The default must leave every existing arm byte-identical: E7-E9 cells and
    the E13 control differ by nothing at all, not by a stray `mkdir`."""
    runtime = _launch()
    assert not any("harness" in " ".join(argv) for argv, _ in runtime.commands), (
        "an unset continual_harness_dir must not touch the filesystem"
    )


def test_the_shared_store_is_linked_at_the_one_name_prime_agent_reads():
    """`getGlobalHarnessStateDir()` is `join(agentDir, "harness")` and the kernel
    gets the same path as `RLM_GLOBAL_HARNESS_STATE_DIR`, so the link has to land
    on exactly that name -- not `harness_state.json`, not a copy."""
    store = "/tmp/vf-prime-agent/continual-harness"
    runtime = _launch(continual_harness_dir=store)

    linked = [argv for argv, _ in runtime.commands if "ln -sfn" in " ".join(argv)]
    assert len(linked) == 1, "expected exactly one link command"
    script = " ".join(linked[0])
    assert f"mkdir -p {store}" in script, "the store must be created if absent"
    assert f"ln -sfn {store} /tmp/vf-prime-agent/agent-trace-abc/harness" in script


def test_the_per_rollout_files_stay_per_rollout():
    """Only the harness state is shared. Seeds run CONCURRENTLY and each carries
    its own MCP URL and interception secret, so a shared `settings.json` would
    point five agents at each other's games."""
    store = "/tmp/vf-prime-agent/continual-harness"
    runtime = _launch(continual_harness_dir=store)

    for name in ("settings.json", "models.json", "auth.json"):
        assert f"/tmp/vf-prime-agent/agent-trace-abc/{name}" in runtime.files
        assert f"{store}/{name}" not in runtime.files


def test_the_player_cannot_write_the_shared_store_by_default():
    """Single-writer by construction: the read-only re-bind comes after the
    read-write `install_dir` bind, so bwrap's later, narrower mount wins."""
    store = "/tmp/vf-prime-agent/continual-harness"
    runtime = _launch_sandboxed(
        continual_harness_dir=store, sandbox=True, install_dir="/tmp/vf-prime-agent"
    )
    argv = [a for a, _ in runtime.programs][0]

    assert argv[:1] == ["bwrap"]
    rw = argv.index("--bind")
    ro = argv.index("--ro-bind", argv.index("/tmp/vf-prime-agent"))
    assert argv[ro : ro + 3] == ["--ro-bind", store, store]
    assert ro > rw, "the read-only store bind must come after the read-write one"


def test_the_writable_variant_drops_the_read_only_rebind():
    store = "/tmp/vf-prime-agent/continual-harness"
    runtime = _launch_sandboxed(
        continual_harness_dir=store,
        continual_harness_writable=True,
        sandbox=True,
        install_dir="/tmp/vf-prime-agent",
    )
    argv = [a for a, _ in runtime.programs][0]
    assert ["--ro-bind", store, store] != argv[argv.index("--bind") :][:3]
    assert not any(
        argv[i : i + 3] == ["--ro-bind", store, store] for i in range(len(argv))
    )


def test_a_store_the_sandbox_would_not_bind_is_refused_loudly():
    """A store outside `install_dir` leaves the symlink dangling inside the
    sandbox: every rollout starts from an empty harness, which reads exactly like
    'the agent learned nothing'. Fail instead."""
    import asyncio
    import types

    import pytest

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(
            id="nethack-prime-agent",
            sandbox=True,
            continual_harness_dir="/root/nld/e12/continual-harness",
        )
    )
    ctx = types.SimpleNamespace(
        model="z-ai/glm-5.2", sampling=types.SimpleNamespace(reasoning_effort=None)
    )
    with pytest.raises(ValueError, match="outside install_dir"):
        asyncio.run(
            harness.launch(
                ctx,
                _build_trace(),
                _RecordingRuntimeWithWorkdir(),
                "http://127.0.0.1:9999/v1",
                "vf-secret",
                {"nethack": "http://127.0.0.1:41449"},
            )
        )


def test_a_sibling_path_does_not_pass_the_containment_check():
    """`/tmp/vf-prime-agent-other` is not inside `/tmp/vf-prime-agent`."""
    from nethack_prime_agent import _within

    assert _within("/tmp/vf-prime-agent/ch", "/tmp/vf-prime-agent")
    assert _within("/tmp/vf-prime-agent", "/tmp/vf-prime-agent")
    assert not _within("/tmp/vf-prime-agent-other/ch", "/tmp/vf-prime-agent")
