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
    assert (
        c_args["skill_set"]
        == claude["taskset"]["env_args"]["skill_set"]
        == prime["taskset"]["env_args"]["skill_set"]
        == "netplay"
    )
    assert control["num_tasks"] == claude["num_tasks"] == prime["num_tasks"] == 16
    # Budget: exactly-150 on both CLI arms, at-most-150 on the control arm.
    assert cc["max_skill_calls"] == pa["max_skill_calls"] == 150
    assert c_args["max_turns"] == 150
    # Both CLI arms must seed the workspace, or they are not capability-matched
    # to the control arm's system prompt / wiki tools / Journal.
    assert cc["seed_workspace"] is True and pa["seed_workspace"] is True
    # Distinct per-turn NDJSON directories, or one arm overwrites the other.
    assert cc["trace_dir"] != pa["trace_dir"] != control["args"]["trace_dir"]


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
    """Captures `runtime.write` / `run` instead of touching a filesystem."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.programs: list[tuple[list[str], dict[str, str]]] = []
        self.commands: list[tuple[list[str], dict[str, str]]] = []

    async def write(self, path, data):
        self.files[path] = data

    async def run(self, argv, env):
        from verifiers.v1.runtimes import ProgramResult

        self.commands.append((argv, env))
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def run_program(self, argv, env):
        from verifiers.v1.runtimes import ProgramResult

        self.programs.append((argv, env))
        return ProgramResult(exit_code=0, stdout="ok", stderr="")


def _launch(**config_overrides):
    """Drive `launch` against the recording runtime and return what it wrote."""
    import asyncio
    import types

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig
    from verifiers.v1.task import TaskData

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="nethack-prime-agent", **config_overrides)
    )
    ctx = types.SimpleNamespace(
        model="z-ai/glm-5.2",
        sampling=types.SimpleNamespace(reasoning_effort=None),
    )
    trace = types.SimpleNamespace(
        id="trace-abc",
        task=types.SimpleNamespace(
            data=TaskData(
                idx=0, prompt="Play NetHack.", system_prompt="You are a Valkyrie."
            )
        ),
    )
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
    assert argv[argv.index("--append-system-prompt") + 1] == "You are a Valkyrie."
    assert "--no-session" in argv and "--print" in argv


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
