"""The bubblewrap (`bwrap`) sandbox: proof it actually isolates, plus the wiring.

Why this exists: in a prior run the Prime Agent arm abandoned the game, probed
harness internals, printed a live MCP bearer token into its own trace, and ran
`glob('/scratch/**', recursive=True)` on this cluster's shared multi-user GPFS
filesystem — the leading suspect for a 1h53m stall that burned a rollout. Prime
Agent's only tool is `ipython`, a full Python interpreter with no denylist
(`tests/test_prime_agent_harness.py::test_disabled_tools_is_refused_rather_than_silently_ignored`
pins that `disabled_tools` cannot help here), so no config-level clamp closes
this off. A mount-namespace sandbox is the only thing that does, and this file
is the actual proof, not an assertion about it.

Two tiers, deliberately kept separate:

1. `test_base_recipe_*` -- the literal recipe from `configs/README.md` §7.2,
   with NO extra binds, run exactly as given. This is the strongest claim:
   `/scratch` doesn't exist at all.
2. `test_harness_prefix_*` -- `PrimeAgentHarness._sandbox_prefix`, the argv this
   repo's harness actually prepends to a real launch. It has to re-expose the
   nvm-under-`/scratch` install tree so `prime-agent` and `uv` resolve (the one
   subtlety configs/README.md §7.2 calls out), which makes the string
   `/scratch` walkable again as an empty stub down to that one bound leaf.
   These tests prove that stub carries nothing else: no sibling directory, no
   repo file, no other file anywhere under `/scratch`.

Every test that shells out to `bwrap` skips cleanly if it is not on PATH, so
the suite still runs on a machine without it.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "tools/cli_harness_eval/configs"

BWRAP = shutil.which("bwrap")
pytestmark = pytest.mark.skipif(BWRAP is None, reason="bwrap not found on PATH")


def _run_bwrap(*args: str, timeout: float = 15.0) -> subprocess.CompletedProcess:
    assert BWRAP is not None
    return subprocess.run(
        [BWRAP, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# -- 1. the base recipe, verbatim -------------------------------------------


def _base_recipe(workspace: str, *command: str) -> list[str]:
    """`configs/README.md` §7.2's base recipe, unmodified."""
    return [
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--bind", workspace, "/work",
        "--chdir", "/work",
        "--unshare-pid",
        "--die-with-parent",
        *command,
    ]


@pytest.fixture()
def workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    # Off /scratch on purpose: the base-recipe tests assert /scratch is fully
    # absent, so the workspace binding itself must not be the thing that
    # re-introduces it. `tmp_path` is under the pytest base temp dir (real
    # local disk on this machine, not GPFS).
    return tmp_path


def test_base_recipe_workspace_is_readable_and_writable(workspace):
    (workspace / "seed.txt").write_text("hello from the host\n")
    result = _run_bwrap(
        *_base_recipe(
            str(workspace),
            "sh", "-c",
            "cat seed.txt && echo written > out.txt && cat /work/out.txt",
        )
    )
    assert result.returncode == 0, result.stderr
    assert "hello from the host" in result.stdout
    assert "written" in result.stdout
    assert (workspace / "out.txt").read_text().strip() == "written"


def test_base_recipe_scratch_is_completely_absent(workspace):
    result = _run_bwrap(
        *_base_recipe(
            str(workspace),
            "python3", "-c",
            "import os, sys; sys.exit(0 if os.path.exists('/scratch') else 1)",
        )
    )
    # exit 1 == os.path.exists('/scratch') was False, i.e. /scratch is gone.
    assert result.returncode == 1, (
        f"/scratch was reachable inside the base-recipe sandbox: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_base_recipe_a_known_repo_file_cannot_be_opened(workspace):
    target = REPO_ROOT / "tools/cli_harness_eval/configs/README.md"
    assert target.is_file(), "fixture assumption: this file must exist on the host"
    result = _run_bwrap(
        *_base_recipe(str(workspace), "cat", str(target))
    )
    assert result.returncode != 0
    assert "No such file" in result.stderr or "no such file" in result.stderr.lower()


def test_base_recipe_network_still_works():
    """The agent has to keep reaching the MCP tool server and the model
    endpoint on 127.0.0.1 -- neither arm's config passes `--unshare-net`."""
    import http.server
    import threading

    server = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with subprocess_workspace() as workspace:
            result = _run_bwrap(
                # `/usr/bin/python3`, not `sys.executable` -- the latter is this
                # test venv's interpreter, which (like the repo) lives under
                # `/scratch` and is therefore, correctly, not visible here.
                *_base_recipe(
                    workspace,
                    "python3", "-c",
                    f"import urllib.request as u; print(u.urlopen('http://127.0.0.1:{port}/', timeout=5).status)",
                )
            )
        assert result.returncode == 0, result.stderr
        assert "200" in result.stdout
    finally:
        server.shutdown()
        thread.join(timeout=5)


def subprocess_workspace():
    import contextlib
    import tempfile

    @contextlib.contextmanager
    def _ctx():
        with tempfile.TemporaryDirectory() as d:
            yield d

    return _ctx()


# -- 2. the harness's actual, extended prefix --------------------------------


def _prime_agent_harness(**overrides):
    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig

    config = PrimeAgentHarnessConfig(id="nethack-prime-agent", sandbox=True, **overrides)
    return PrimeAgentHarness(config)


def test_harness_prefix_starts_with_the_configured_bwrap_binary():
    harness = _prime_agent_harness()
    argv = harness._sandbox_prefix("/tmp/vf-fake-workdir")
    assert argv[0] == "bwrap"
    assert "--unshare-pid" in argv
    assert "--die-with-parent" in argv


def test_harness_prefix_binds_workdir_and_install_dir_read_write():
    harness = _prime_agent_harness(install_dir="/tmp/vf-prime-agent-test")
    argv = harness._sandbox_prefix("/tmp/vf-fake-workdir")
    assert argv.count("--bind") >= 2
    binds = {argv[i + 1]: argv[i + 2] for i, a in enumerate(argv) if a == "--bind"}
    assert binds.get("/tmp/vf-fake-workdir") == "/tmp/vf-fake-workdir"
    assert binds.get("/tmp/vf-prime-agent-test") == "/tmp/vf-prime-agent-test"
    assert argv[argv.index("--chdir") + 1] == "/tmp/vf-fake-workdir"


@pytest.mark.skipif(
    not pathlib.Path(
        "/scratch/gpfs/ZHUANGL/jl0796/.xdg_config_home/nvm/versions/node/v24.13.1"
    ).is_dir(),
    reason="this cluster's nvm node install is not present",
)
def test_harness_prefix_re_exposes_path_prepend_roots_read_only():
    """The one real subtlety (module docstring / README §7.2): the agent binary
    resolves through `path_prepend`, which on this cluster lives under
    `/scratch` -- so it must be re-bound explicitly or the CLI itself vanishes.

    Same cluster-path guard as the sibling test below: `_sandbox_prefix` only
    binds paths that EXIST, so off-cluster this asserts on a bind that was
    correctly never made. It was passing here only by accident -- the whole file
    skipped while `bwrap` was absent."""
    node_root = "/scratch/gpfs/ZHUANGL/jl0796/.xdg_config_home/nvm/versions/node/v24.13.1"
    harness = _prime_agent_harness(path_prepend=f"{node_root}/bin:/home/jl0796/.local/bin")
    argv = harness._sandbox_prefix("/tmp/vf-fake-workdir")
    ro_binds = {argv[i + 1]: argv[i + 2] for i, a in enumerate(argv) if a == "--ro-bind"}
    # Bound one level above `bin/`, not `bin/` itself -- see the docstring on
    # `_sandbox_prefix` for why (an npm global's symlink target lives in a
    # sibling `lib/` directory).
    assert ro_binds.get(node_root) == node_root
    assert ro_binds.get("/home/jl0796/.local") == "/home/jl0796/.local"


@pytest.mark.skipif(
    not pathlib.Path(
        "/scratch/gpfs/ZHUANGL/jl0796/.xdg_config_home/nvm/versions/node/v24.13.1"
    ).is_dir(),
    reason="this cluster's nvm node install is not present",
)
def test_harness_prefix_stub_scratch_exposes_nothing_but_the_bound_leaf(tmp_path):
    """The documented trade-off (configs/README.md §7.2): re-exposing the node
    install under /scratch forces bwrap to materialize /scratch as an empty
    stub down to that one leaf. This proves the stub carries nothing else --
    not a sibling directory, not the repo, not another user's data."""
    node_root = "/scratch/gpfs/ZHUANGL/jl0796/.xdg_config_home/nvm/versions/node/v24.13.1"
    harness = _prime_agent_harness(
        path_prepend=f"{node_root}/bin",
        install_dir=str(tmp_path / "install"),
    )
    workdir = str(tmp_path / "workdir")
    pathlib.Path(workdir).mkdir()
    pathlib.Path(harness.config.install_dir).mkdir()
    argv = harness._sandbox_prefix(workdir)
    probe = subprocess.run(
        [*argv, "python3", "-c", _SCRATCH_STUB_PROBE],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert probe.returncode == 0, probe.stderr
    lines = dict(line.split("=", 1) for line in probe.stdout.strip().splitlines())
    assert lines["scratch_exists"] == "True"
    # Only the one path segment leading to the bound node root is listable;
    # nothing else under jl0796's scratch directory (this repo included).
    assert lines["scratch_gpfs_zhuangl_jl0796_listing"] == "['.xdg_config_home']"
    assert lines["repo_readable"] == "False"
    assert lines["nethackharness_readable"] == "False"
    assert lines["prime_agent_runs"] == "True"


_SCRATCH_STUB_PROBE = """
import os, subprocess
print("scratch_exists=%s" % os.path.exists("/scratch"))
try:
    listing = sorted(os.listdir("/scratch/gpfs/ZHUANGL/jl0796"))
except Exception:
    listing = []
print("scratch_gpfs_zhuangl_jl0796_listing=%s" % listing)
try:
    open("/scratch/gpfs/ZHUANGL/jl0796/NetHack-hub/.worktrees/cli-harness-eval/tools/cli_harness_eval/configs/README.md").read()
    repo_readable = True
except Exception:
    repo_readable = False
print("repo_readable=%s" % repo_readable)
try:
    open("/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness/README.md").read()
    nh_readable = True
except Exception:
    nh_readable = False
print("nethackharness_readable=%s" % nh_readable)
r = subprocess.run(["prime-agent", "--version"], capture_output=True, text=True)
print("prime_agent_runs=%s" % (r.returncode == 0))
"""


def test_launch_prepends_the_sandbox_prefix_and_requires_a_workdir():
    """Regression: `launch` must fail loudly (not silently run unconfined) if
    `harness.sandbox` is true and the runtime has no filesystem `workdir`."""
    import asyncio
    import types

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig
    from verifiers.v1.task import TaskData

    class _NoWorkdirRuntime:
        async def write(self, path, data):
            pass

        async def run(self, argv, env):
            from verifiers.v1.runtimes import ProgramResult

            return ProgramResult(exit_code=0, stdout="", stderr="")

        async def run_program(self, argv, env):
            from verifiers.v1.runtimes import ProgramResult

            return ProgramResult(exit_code=0, stdout="ok", stderr="")

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="nethack-prime-agent", sandbox=True)
    )
    ctx = types.SimpleNamespace(
        model="z-ai/glm-5.2",
        sampling=types.SimpleNamespace(reasoning_effort=None),
    )
    trace = types.SimpleNamespace(
        id="trace-sandbox",
        task=types.SimpleNamespace(
            data=TaskData(idx=0, prompt="Play NetHack.", system_prompt="You are a Valkyrie.")
        ),
    )
    with pytest.raises(RuntimeError, match="workdir"):
        asyncio.run(
            harness.launch(
                ctx, trace, _NoWorkdirRuntime(),
                "http://127.0.0.1:9999/v1", "vf-secret", {"nethack": "http://127.0.0.1:1"},
            )
        )


def test_launch_wires_the_prefix_in_front_of_the_real_argv():
    import asyncio
    import types

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig
    from verifiers.v1.task import TaskData

    class _RecordingRuntimeWithWorkdir:
        def __init__(self, workdir):
            self.workdir = workdir
            self.programs = []

        async def write(self, path, data):
            pass

        async def run(self, argv, env):
            from verifiers.v1.runtimes import ProgramResult

            return ProgramResult(exit_code=0, stdout="", stderr="")

        async def run_program(self, argv, env):
            from verifiers.v1.runtimes import ProgramResult

            self.programs.append((argv, env))
            return ProgramResult(exit_code=0, stdout="ok", stderr="")

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(id="nethack-prime-agent", sandbox=True)
    )
    ctx = types.SimpleNamespace(
        model="z-ai/glm-5.2",
        sampling=types.SimpleNamespace(reasoning_effort=None),
    )
    trace = types.SimpleNamespace(
        id="trace-sandbox2",
        task=types.SimpleNamespace(
            data=TaskData(idx=0, prompt="Play NetHack.", system_prompt="You are a Valkyrie.")
        ),
    )
    runtime = _RecordingRuntimeWithWorkdir("/tmp/vf-fake-workdir-2")
    asyncio.run(
        harness.launch(
            ctx, trace, runtime,
            "http://127.0.0.1:9999/v1", "vf-secret", {"nethack": "http://127.0.0.1:1"},
        )
    )
    argv, _ = runtime.programs[0]
    assert argv[0] == "bwrap"
    assert "prime-agent" in argv  # the original launch argv, unmodified, still present
    assert argv[-2:] == ["--", "Play NetHack."]  # option terminator survives the prefix


# -- config wiring ------------------------------------------------------------


def _load(name: str) -> dict:
    return tomllib.loads((CONFIGS / name).read_text())


def test_prime_agent_config_matches_the_b80_arm_on_the_sandbox():
    """Both prime_agent arms must agree, or the b80 comparison drifts on a
    dimension it is not measuring. The VALUE is an operator decision, not a
    property this suite pins: 2026-08-01 it is `false` on the Prime Intellect
    CPU sandbox by choice (no /scratch to hide, disposable single-tenant box) --
    NOT because bwrap is unavailable, which is the cluster case exp2 §2
    describes. The isolation proofs above still run whenever bwrap is present."""
    prime = _load("prime_agent.toml")
    b80 = _load("prime_agent_b80.toml")
    assert prime["harness"]["sandbox"] == b80["harness"]["sandbox"]


def test_claude_code_config_has_no_sandbox_key():
    """Claude Code cannot take a command prefix without patching the stock
    `verifiers` package (`ClaudeCodeHarnessConfig` has no `binary` field) --
    documented in `configs/README.md` §7.3. `HarnessConfig` subclasses forbid
    extra fields (`pydantic_config.BaseConfig`, `extra="forbid"`), so a stray
    `sandbox = true` here would fail at config-load time, not silently no-op --
    this test pins the absence instead of relying on that failure mode."""
    claude = _load("claude_code.toml")
    assert "sandbox" not in claude["harness"]


def test_setup_fails_loudly_when_sandbox_is_on_and_bwrap_is_missing():
    import asyncio

    from nethack_prime_agent import PrimeAgentHarness, PrimeAgentHarnessConfig

    class _ProbeRuntime:
        async def write(self, path, data):
            pass

        async def run(self, argv, env):
            from verifiers.v1.runtimes import ProgramResult

            # Simulate: prime-agent resolves fine, but the configured bwrap
            # name does not.
            joined = " ".join(argv)
            if "prime-agent" in joined and "sandbox_bwrap_does_not_exist" not in joined:
                return ProgramResult(exit_code=0, stdout="0.3.3\n", stderr="")
            return ProgramResult(exit_code=1, stdout="", stderr="not found")

    harness = PrimeAgentHarness(
        PrimeAgentHarnessConfig(
            id="nethack-prime-agent",
            sandbox=True,
            sandbox_bwrap="sandbox_bwrap_does_not_exist",
        )
    )
    with pytest.raises(RuntimeError, match="sandbox_bwrap_does_not_exist"):
        asyncio.run(harness.setup(_ProbeRuntime()))
