"""Prime Agent as an external verifiers harness, driving the NetHack toolset.

This is arm 2 of the CLI-harness comparison. Arm 1 (`claude_code`, built in to
verifiers 0.2.1) hands the agent the toolset as *agent tools* named
`mcp__nethack__<skill>`. Prime Agent cannot do that: MCP integrations are
deliberately not exposed as tools — its only built-in tool is `ipython`, and an
MCP server arrives as a Python package imported into the session's IPython
kernel (`prime-agent/docs/mcp-integrations.md`). The agent therefore writes

    import nethack
    print(await nethack.explore_and_descend())

against the same server, the same 18 tools and the same toolset-side call
budget. That difference in *kind* is a property of the scaffold under test, not
a defect of this harness; `tools/cli_harness_eval/configs/README.md` records what
it means for reading the results.

Three things the harness has to wire up, all of them per rollout:

1. **Model routing.** Prime Agent has no `<VENDOR>_BASE_URL` escape hatch. It
   reaches verifiers' interception the way the bundled `pi` harness does
   (`verifiers/v1/harnesses/pi/harness.py:186-199`): a custom provider in
   `models.json` with `baseUrl = endpoint`, `api = "openai-completions"` and the
   interception secret as its API key, selected unambiguously with
   `--provider intercept --model <ctx.model>`. Interception's chat dialect
   registers `/v1/chat/completions`, which is exactly what that API type speaks.
2. **The MCP server.** `mcpServers.nethack` in the per-rollout `settings.json`,
   carrying the OS-assigned URL from `mcp_urls`. Only remote `"http"` servers
   are supported, which is what verifiers serves.
3. **The skill package.** `setup()` materializes it at a *fixed* path, because
   Prime Agent keys its kernel venv on the set of Python-skill paths
   (`ensureKernelPythonKey`) and a per-rollout path would rebuild the venv on
   every seed.

`PRIME_AGENT_CODING_AGENT_DIR` points the whole config directory (settings,
models, auth, sessions) at a per-rollout directory, so nothing here reads or
writes the operator's `~/.prime/agent`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shlex
from importlib import resources

from pydantic import Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

__all__ = ["PrimeAgentHarness", "PrimeAgentHarnessConfig"]

PROVIDER = "intercept"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
MCP_TOKEN_VAR = "NETHACK_MCP_TOKEN"

# The prompt a relaunch gets instead of the original task prompt. The MCP tool
# server (and the NetHack game it owns) is per-ROLLOUT, not per-CLI-process, so
# it is still sitting exactly where the previous `--no-session` process left
# it; only the agent's own conversation is gone. Re-sending the original
# "Task: ... Begin." prompt after hundreds of tool calls would read as an
# instruction to start over, so this tells it plainly that it is not starting
# over -- see `PrimeAgentHarnessConfig.max_relaunches` for the measurements
# that motivate relaunching at all.
_RESUME_PROMPT = (
    "Your previous session ended, but the game did not: it is still running "
    "and your character is still alive. Call a NetHack tool now to see the "
    "current state and keep playing from there. Do not restart the task, "
    "re-describe the objective, or treat this as a new game."
)

# Files copied out of this distribution into the runtime to form the skill
# package (`skills.md#python-backed-skills` layout).
_SKILL_FILES = ("SKILL.md", "pyproject.toml", "src/nethack/__init__.py")

_NO_BATCH_RULE = (
    "- Do not batch blind sequences of calls. NetHack is turn-based and adversarial;\n"
    "  the observation after each call is what tells you whether the previous one\n"
    "  worked.\n"
)


def _strip_no_batch_rule(data: bytes) -> bytes:
    """Remove the no-batching instruction from SKILL.md (see `allow_batching`).

    Fails loudly rather than silently shipping the unmodified file: a cell that
    believed batching was enabled but ran the constrained prompt would be a
    silently invalid experiment, which is exactly the asymmetry this flag exists
    to remove.
    """
    text = data.decode()
    if _NO_BATCH_RULE not in text:
        raise RuntimeError(
            "allow_batching=True but the no-batch rule was not found verbatim in "
            "SKILL.md -- the file changed and `_NO_BATCH_RULE` is stale. Update it "
            "rather than running with the instruction still in place."
        )
    return text.replace(_NO_BATCH_RULE, "").encode()



class PrimeAgentHarnessConfig(HarnessConfig):
    binary: str = "prime-agent"
    """Executable to launch. Prime Agent is an npm global, not a downloadable
    tarball, so this harness does NOT install it — unlike `claude_code`, which
    curls a pinned release. Set an absolute path when `prime-agent` is not on the
    inherited PATH."""

    version: str = ""
    """Expected `prime-agent --version`, checked in `setup`. Empty disables the
    check. Pin it for a reproducible run; a silent CLI upgrade mid-experiment
    changes the scaffold under test."""

    install_dir: str = "/tmp/vf-prime-agent"
    """Fixed directory holding the skill package. Fixed on purpose (see module
    docstring). Verifiers' own harnesses hard-code `/tmp/vf-*` paths that are not
    per-user; on a shared node, override this to somewhere you own — see caveat 6
    in `tools/cli_harness_eval/configs/README.md`."""

    path_prepend: str = ""
    """Prepended to PATH for both `setup` and `launch`. Prime Agent needs `node`
    on PATH (its bin is a `#!/usr/bin/env node` script) and `uv` to build its
    kernel venv; the subprocess runtime inherits the host PATH, which on a module
    based cluster may not carry either."""

    thinking: str = ""
    """`--thinking` level. Empty leaves Prime Agent's default (`xhigh`)."""

    websearch: bool = False
    """Load Prime Agent's bundled `websearch` skill. Off by default: the control
    arm and the Claude Code arm both run without web access, and a live search
    tool would not be capability-matched."""

    reasoning: bool | None = Field(default=None)
    """Declare the model as reasoning-capable in `models.json`. `None` derives it
    from `ctx.sampling.reasoning_effort` the way the `pi` harness does."""

    context_window: int = 0
    """The model's context window, in tokens, declared in `models.json`.

    **Leaving this at 0 silently truncates every long rollout.** A custom
    provider's model definition that omits `contextWindow` is normalized to
    `128e3` by Prime Agent (`definition.contextWindow ?? 128e3`). Its session
    host then stops the agent loop the moment
    `contextTokens > contextWindow - reserveTokens` (reserve is 16384 by
    default) -- `_shouldStopForThresholdCompaction`, whose `true` propagates out
    of `shouldStopAfterTurn` and ends the loop. Under `--print` nothing resumes
    it: `promptAndWait` returns, the last message is a tool result rather than
    an assistant message, print mode emits nothing, and the CLI exits 0.
    Verifiers can only read a clean exit as `agent_completed`.

    That is what ended three of five run-1 rollouts mid-action, on a tool call
    that never got a follow-up, with 107-130 of 400 skill calls and 34-43 of 120
    minutes used. Reproduced against a local stub model reporting a controlled
    `usage.total_tokens` ramp: with the field omitted the run stops on the first
    response over 111,616 tokens (128000-16384); declaring `contextWindow =
    300000` moves the stop to the first response over 283,616 (300000-16384).
    Same binary, same prompt, nothing else changed.

    So: set it to the real context window of the model under test. Prime Agent's
    own bundled catalog lists `z-ai/glm-5.2` at 1048576. `0` preserves the old
    behaviour and logs a warning rather than guessing a window on your behalf."""

    capture_output: bool = True
    """Write the CLI's stdout/stderr into `install_dir/agent-<trace id>/`.

    Verifiers only surfaces a harness program's output when it exits non-zero
    (`Harness.run`), so the run-1 cutoffs -- which exited 0 -- left nothing at
    all to read. Keeping the streams costs a few KB per rollout and is the
    difference between diagnosing the next silent exit and re-running blind."""

    sandbox: bool = False
    """Prefix the launch argv with a `bwrap` (bubblewrap) mount-namespace
    sandbox that hides everything from the agent's `ipython` tool except an
    explicit allowlist -- most importantly, `/scratch` (this cluster's shared
    multi-user GPFS filesystem) and this repository. See
    `tools/cli_harness_eval/configs/README.md` §7 for why: Prime Agent's only
    built-in tool is `ipython`, a full Python interpreter with no denylist, and
    in a prior run the agent abandoned the game, ran
    `glob('/scratch/**', recursive=True)`, and printed a live MCP bearer token
    into its own trace. `disabled_tools` cannot stop a Python interpreter --
    this can, at the mount-namespace level, unconditionally.

    Defaults to False so every existing caller (including the `_RecordingRuntime`
    test fixture in `tests/test_prime_agent_harness.py`, which has no `workdir`)
    is unaffected; `prime_agent.toml` turns it on for the real arm.

    `setup` fails loudly if this is True and `bwrap` cannot be resolved, rather
    than silently running the arm unconfined. `launch` fails loudly if this is
    True and the runtime has no filesystem `workdir` (i.e. anything but the
    `subprocess` runtime) -- there is nothing to sandbox a remote/container
    runtime's own process with."""

    sandbox_bwrap: str = "bwrap"
    """`bwrap` executable. Resolved the same way as `binary` -- through PATH,
    with `path_prepend` applied first -- so an absolute override works for a
    non-PATH install."""

    allow_batching: bool = False
    """Let Prime Agent issue as many skill calls per turn as it likes.

    `skill/SKILL.md` ships an instruction telling the agent NOT to batch:

        Do not batch blind sequences of calls. NetHack is turn-based and
        adversarial; the observation after each call is what tells you whether
        the previous one worked.

    Prime obeys it -- measured 0.80-0.95 skill invocations per `ipython` call
    across m2, m3 and g2. Claude Code was never given an equivalent rule and
    batched at 2.31 tool calls per assistant turn on g2 (34 batches of 6, five
    of 10 on one seed), so it bought ~2.3x the game actions per decision on the
    same 400-call budget. The arms were therefore never on equal terms, and the
    asymmetry is ours, not the scaffolds'.

    Setting this True strips the instruction from the materialized SKILL.md, so
    the agent is free to do whatever it wants per turn. The counterpart
    experiment constrains Claude Code instead
    (`taskset.max_parallel_skill_calls = 1`). Run both, or neither -- running
    one alone re-introduces the asymmetry in the other direction.

    Default False so existing cells are unchanged.
    """

    max_relaunches: int = 5
    """Cap on auto-resume relaunches (`launch` returning `exit_code == 0` while
    the game is neither dead nor budget-exhausted -- see `PrimeAgentHarness.
    _episode_live`). `--print` mode exits the moment the model stops emitting
    tool calls, whether or not the character is alive, and the stock
    `verifiers` harness loop (`v1/harness.py:109-118`) reads that clean exit as
    `stop_condition = "agent_completed"` -- a false completion signal that
    measurably drags down every arm comparison (15-25% of rollouts stop this
    way; measured sessions ended anywhere from 0 to ~170 of a 400-call budget).
    This cannot be fixed by patching `Harness.run` (stock, read-only), so
    `launch` loops on it directly: relaunch reuses the same MCP tool server
    (game state persists there across CLI processes) and swaps in
    `_RESUME_PROMPT` so the fresh `--no-session` process picks the game back up
    instead of re-reading the original task prompt as if nothing had happened.

    5 is a deliberately modest cap, not a "guarantee the budget is reached"
    number: the worst measured session (Prime Agent, 20 of 400 calls) would
    need on the order of 20 relaunches to exhaust a 400-call budget by itself,
    and the observed 0-call session (Prime Agent, executed nothing at all)
    shows relaunching does not always help -- a session that still is not
    progressing after 5 fresh attempts is a broken rollout, not something to
    paper over with unbounded retries burning wall-clock and provider spend.
    `relaunches` is recorded on `trace.metrics["prime_agent_relaunches"]` every
    rollout (including 0) so this is never invisible in the aggregate."""


class PrimeAgentHarness(Harness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True  # `--append-system-prompt` (usage.md CLI reference)
    SUPPORTS_MCP = True
    # Prime Agent's print mode takes one prompt (plus `@file` attachments); there
    # is no multi-message input, and no user simulator.
    SUPPORTS_MESSAGE_PROMPT = False

    # -- provisioning -------------------------------------------------------

    @property
    def _skill_dir(self) -> str:
        return f"{self.config.install_dir}/skills/nethack"

    async def setup(self, runtime: Runtime) -> None:
        binary = self.config.binary
        probe = await runtime.run(
            ["sh", "-c", f'command -v {shlex.quote(binary)} && {shlex.quote(binary)} --version'],
            self._env_with_path(),
        )
        if probe.exit_code != 0:
            raise RuntimeError(
                f"prime-agent not runnable as {binary!r}: "
                f"{(probe.stderr or probe.stdout).strip()[-500:] or '<no output>'}. "
                "Install it with `npm install -g prime-agent`, or set "
                "`harness.binary` / `harness.path_prepend`."
            )
        # `--version` is one line of Node stdout followed by an immediate exit, and
        # it does not always survive being written to a pipe. So: fail on a version
        # we could read and that disagrees, warn on one we could not read — never
        # abort a paid run over an unreadable banner.
        found = re.search(
            r"^\s*v?(\d+\.\d+\.\d+\S*)\s*$", probe.stdout, flags=re.MULTILINE
        )
        version = found.group(1) if found else ""
        if self.config.version and version and version != self.config.version:
            raise RuntimeError(
                f"prime-agent version mismatch: config pins {self.config.version!r}, "
                f"the installed CLI reports {version!r}."
            )
        if self.config.version and not version:
            logger.warning(
                "could not read `%s --version` (got %r); the pin %r is unverified",
                binary,
                probe.stdout.strip()[-200:],
                self.config.version,
            )
        logger.info("prime-agent %s", version or "(version unknown)")

        # The skill package, at a path that does not vary per rollout.
        package = resources.files(__package__) / "skill"
        for name in _SKILL_FILES:
            data = (package / name).read_bytes()
            if name == "SKILL.md" and self.config.allow_batching:
                data = _strip_no_batch_rule(data)
            await runtime.write(f"{self._skill_dir}/{name}", data)

        if self.config.sandbox:
            # Same "fail loud, not silently unconfined" philosophy as the
            # prime-agent probe above: a missing `bwrap` must stop the run, not
            # quietly disable the one thing standing between the agent's
            # `ipython` and `/scratch`.
            bwrap = self.config.sandbox_bwrap
            probe = await runtime.run(
                ["sh", "-c", f"command -v {shlex.quote(bwrap)}"],
                self._env_with_path(),
            )
            if probe.exit_code != 0:
                raise RuntimeError(
                    f"harness.sandbox is true but {bwrap!r} was not found on PATH: "
                    f"{(probe.stderr or probe.stdout).strip()[-300:] or '<no output>'}. "
                    "Install bubblewrap, set `harness.sandbox_bwrap` to an absolute "
                    "path, or set `harness.sandbox = false` to run unconfined "
                    "(see configs/README.md §7)."
                )

    # -- launch -------------------------------------------------------------

    def _env_with_path(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {**self.config.resolved_env, **(extra or {})}
        if self.config.path_prepend:
            inherited = env.get("PATH") or os.environ.get("PATH", "")
            env["PATH"] = f"{self.config.path_prepend}:{inherited}"
        return env

    def _sandbox_prefix(self, workdir: str) -> list[str]:
        """Build the `bwrap` argv that hides everything from the sandboxed
        process except an explicit allowlist. Base recipe verified working on
        this machine (configs/README.md §7): base OS dirs read-only, a fresh
        `/proc`/`/dev`/`/tmp`, and the workspace bound read-write. Extended here
        with exactly what `PrimeAgentHarness` itself needs to still run --
        `path_prepend` (the agent binary, and `uv`) and `install_dir` (the skill
        package plus this rollout's `models.json`/`settings.json`/`auth.json`) --
        nothing else.

        `path_prepend` entries on this cluster resolve under `/scratch` (the npm
        global lives under an nvm install there), so re-exposing them makes the
        exact string `/scratch` walkable again as an EMPTY stub down to that one
        bound leaf (`bwrap` must materialize the intermediate directories to
        attach a mount deep inside them) -- measured, not assumed. No sibling
        directory or file anywhere under `/scratch` is reachable through that
        stub; see `tests/test_sandbox_isolation.py` for the proof, both for the
        base recipe with no extra binds (where `/scratch` is fully absent) and
        for this harness's actual extended prefix (where the stub exists but is
        empty except the one required leaf).

        NOT covered: `$HOME` is never bound, so anything Prime Agent does under
        `~/.prime` (its kernel-venv bootstrap, per configs/README.md §6.2) will
        not find it there and `/tmp` is a fresh, per-rollout tmpfs, so the
        shared daemon-socket design (`configs/README.md`, "the daemon socket is
        left shared on purpose") does not apply here -- each sandboxed rollout
        gets its own daemon. Both are recorded, not fixed, in
        `tools/cli_harness_eval/configs/README.md` §7.3.
        """
        binds: list[str] = []
        # `/run` IS REQUIRED FOR DNS and is the reason `sandbox = true` looked
        # like a networking bug. On systemd-resolved hosts (this box, Ubuntu)
        # `/etc/resolv.conf` is a SYMLINK to `../run/systemd/resolve/stub-resolv.conf`.
        # Binding `/etc` alone carries the symlink but not its target, so inside
        # the sandbox it dangles: `cat /etc/resolv.conf` -> No such file, every
        # name lookup fails, and `prime-agent` surfaces that as the generic
        # "Connection error." -- which reads as a blocked socket and sends you
        # hunting for a proxy. It is not: this prefix never passes
        # `--unshare-net`, so the network namespace is shared and localhost is
        # reachable throughout. Measured: `getent hosts api.pinference.ai`
        # returns nothing without `/run` and resolves with it.
        for base in ("/usr", "/lib", "/lib64", "/bin", "/etc", "/run"):
            if os.path.isdir(base):
                binds += ["--ro-bind", base, base]
        for entry in filter(None, self.config.path_prepend.split(":")):
            # One level above `bin/`, so an npm global's `lib/node_modules`
            # symlink target (where the real JS/venvs live) resolves too --
            # `harness.binary` under this cluster's nvm install is exactly such
            # a symlink. Measured: binding only `bin/` runs `prime-agent
            # --version` but breaks module resolution for the CLI itself.
            root = os.path.dirname(entry) if os.path.basename(entry) == "bin" else entry
            if os.path.isdir(root):
                binds += ["--ro-bind", root, root]
        # `~/.prime` -- WITHOUT THIS THE SANDBOXED ARM CANNOT RUN AT ALL.
        # Measured 2026-08-01: with `sandbox = true` every rollout died as
        # `HarnessError: harness 'nethack-prime-agent' exited 1: Connection
        # error.` at 0 turns, 3 of 3 attempts. `prime-agent --version`, `node`
        # and `uv` all resolve fine inside the prefix; what is missing is
        # `~/.prime`, which holds the credentials (`config.json`) and the
        # IPython kernel venv the agent boots into (`agent/kernel-venv`).
        # The docstring above recorded this as a known gap rather than a fix,
        # because on the Slurm box `max_user_namespaces = 0` made the sandbox
        # unreachable long before anyone hit it.
        #
        # NOT a network problem, which is the intuitive guess: this prefix
        # never passes `--unshare-net`, so the sandbox shares the host network
        # namespace and localhost is already reachable (verified: a shared-net
        # bwrap curls host loopback 200, an `--unshare-net` one gets 000). No
        # proxy is needed.
        #
        # Split by write-need so the credential file stays read-only while the
        # two paths the agent genuinely writes stay writable:
        #   config.json     read  -- API key
        #   bin/            read  -- frpc and friends
        #   agent/          WRITE -- kernel venv, __pycache__ at runtime
        #   tunnels/        WRITE -- per-rollout frpc configs
        prime_home = os.path.join(os.path.expanduser("~"), ".prime")
        if os.path.isdir(prime_home):
            for leaf, mode in (
                ("config.json", "--ro-bind"),
                ("bin", "--ro-bind"),
                ("agent", "--bind"),
                ("tunnels", "--bind"),
            ):
                p = os.path.join(prime_home, leaf)
                if os.path.exists(p):
                    binds += [mode, p, p]

        # An absolute, non-PATH `sandbox_bwrap` override might live outside
        # everything bound above (e.g. a home-directory install of bwrap
        # itself); the base-OS binds cover the stock `/usr/bin/bwrap`.
        bwrap_dir = os.path.dirname(self.config.sandbox_bwrap)
        if bwrap_dir and bwrap_dir not in ("/usr/bin", "/bin") and os.path.isdir(bwrap_dir):
            binds += ["--ro-bind", bwrap_dir, bwrap_dir]
        return [
            self.config.sandbox_bwrap,
            *binds,
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            # Read-write: the skill package plus this rollout's models/settings/
            # auth, all under `install_dir` (fixed on purpose -- module
            # docstring). Nested under `/tmp` by default, so this must come
            # after `--tmpfs /tmp` above; `bwrap` creates the intermediate dirs.
            "--bind",
            self.config.install_dir,
            self.config.install_dir,
            "--bind",
            workdir,
            workdir,
            "--chdir",
            workdir,
            "--unshare-pid",
            "--die-with-parent",
        ]

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if self.config.disabled_tools:
            # Prime Agent has exactly one built-in tool (`ipython`) and offers an
            # allowlist (`--tools`), not a denylist. Silently ignoring a denylist
            # would make the arm's tool surface differ from what the config says.
            # Checked before anything else so a misconfigured arm fails at once
            # rather than after provisioning.
            raise ValueError(
                "PrimeAgentHarness does not support `disabled_tools`: Prime Agent's "
                "only built-in tool is `ipython` (usage.md, Tool Options) and it has "
                "no per-tool deny flag. Removing it would leave the agent unable to "
                "call the game at all."
            )
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("Prime Agent requires a task prompt (it has no user simulator)")

        # Prime Agent is daemon-backed even under `--print`: the CLI attaches to a
        # supervisor whose socket lives at `<tmpdir()>/prime-agent-<uid>/daemon.sock`
        # (`defaultDaemonSocketDir`), which is per-USER, not per-rollout. That is
        # by design — one supervisor, one worker per session — so rollouts SHARE it
        # and this harness does not try to isolate them.
        #
        # Two things were tried and rejected, both measured:
        #   * a per-rollout TMPDIR (which `tmpdir()` honours) moves the socket dir
        #     into the config directory, and Prime Agent then never comes up:
        #     "Timed out waiting for daemon to start on <path>". Reproduced outside
        #     the harness, so it is the CLI's behaviour, not ours.
        #   * `prime-agent shutdown --force` in teardown would reap this rollout's
        #     daemon, but the socket is shared, so it would also kill every
        #     concurrent rollout — and the operator's own session.
        # The residual risk is a *stale* supervisor poisoning a later rollout
        # (`DaemonSocketClosedError`, seen after killing an earlier run by hand).
        # The remedy is operational, not configurable: `prime-agent doctor --fix`
        # (or `shutdown`) between runs. Recorded in configs/README.md §6.3.
        agent_dir = f"{self.config.install_dir}/agent-{trace.id}"
        mcp_token = secrets.token_hex(16)

        reasoning = self.config.reasoning
        if reasoning is None:
            reasoning = ctx.sampling.reasoning_effort not in (None, "none")
        model_def: dict = {
            "id": ctx.model,
            "reasoning": bool(reasoning),
            "input": ["text"],
        }
        if self.config.context_window > 0:
            model_def["contextWindow"] = int(self.config.context_window)
        else:
            # Not fatal -- a short rollout never reaches the threshold -- but a
            # long one dies silently, so it must not pass unremarked. See
            # `PrimeAgentHarnessConfig.context_window`.
            logger.warning(
                "harness.context_window is unset: Prime Agent will assume 128000 "
                "tokens for %r and END THE RUN (exit 0, reported as "
                "'agent_completed') once context passes ~111,616 tokens. Set it "
                "to the model's real context window.",
                ctx.model,
            )
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    # An `apiKey` that names an env var resolves to its value
                    # (models.md, Value Resolution), so the interception secret
                    # never lands on disk.
                    "apiKey": KEY_VAR,
                    "models": [model_def],
                }
            }
        }
        settings = {
            "onboardingShown": True,
            "quietStartup": True,
            # Belt and braces: `--provider/--model` already pin the route, but a
            # fallback that silently picked a built-in provider would run the
            # wrong model, which is the one failure this arm must never have.
            "defaultProvider": PROVIDER,
            "defaultModel": ctx.model,
            "skills": [f"{self.config.install_dir}/skills"],
            "bundledSkills": {"websearch": self.config.websearch},
            "mcpServers": {
                name: {
                    "type": "http",
                    "url": url,
                    "bearerTokenEnvVar": MCP_TOKEN_VAR,
                }
                for name, url in mcp_urls.items()
            },
        }
        await runtime.write(f"{agent_dir}/models.json", json.dumps(models, indent=2).encode())
        await runtime.write(f"{agent_dir}/settings.json", json.dumps(settings, indent=2).encode())
        # A missing auth.json is fine, but an empty one keeps the host from ever
        # reading (or migrating into) the operator's real credential store.
        await runtime.write(f"{agent_dir}/auth.json", b"{}\n")

        env = self._env_with_path(
            {
                KEY_VAR: secret,
                MCP_TOKEN_VAR: mcp_token,
                "PRIME_AGENT_CODING_AGENT_DIR": agent_dir,
                # No update checks, no telemetry, no package-update fetches.
                "PI_OFFLINE": "1",
                "PI_SKIP_VERSION_CHECK": "1",
                "PI_TELEMETRY": "0",
            }
        )

        argv = [
            # PYTHONPATH MUST NOT REACH THE AGENT'S KERNEL. The tool server needs
            # it (verifiers launches `python -m nethack_v1` with the parent
            # environment, so the repo root and `environments/nethack` have to be
            # importable there) and the subprocess runtime passes the host
            # environment straight through — but `environments/nethack/nethack.py`
            # then SHADOWS the skill package, because the kernel import name is
            # the `mcpServers` key and that key is fixed to `nethack` by the
            # toolset's TOOL_PREFIX. This is the name-collision caveat in
            # `mcp-integrations.md`, and it fails silently: Prime Agent reports
            # `<unavailable Python skill 'nethack': No module named 'verifiers'>`
            # and the agent, unable to reach the game, starts reading the
            # experiment's own source tree instead. Measured, in the first smoke.
            "sh",
            "-c",
            'unset PYTHONPATH; exec "$@"',
            "vf-prime-agent",
            self.config.binary,
            "--print",
            "--no-session",
            "--offline",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
        ]
        if self.config.thinking:
            argv += ["--thinking", self.config.thinking]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        # `--` ends option parsing, so a prompt starting with `-` or containing
        # `@word` is never re-read as a flag or a file attachment.
        argv += ["--", prompt]

        if self.config.sandbox:
            # `bwrap` only `exec`s into `argv`; it does not replay or inspect it,
            # so prepending it here changes nothing about what Prime Agent sees
            # or does -- only what filesystem paths exist when it looks.
            workdir = getattr(runtime, "workdir", None)
            if workdir is None:
                raise RuntimeError(
                    "harness.sandbox is true but this runtime has no filesystem "
                    "`workdir` to sandbox (only the `subprocess` runtime does). "
                    "Set `harness.runtime.type = \"subprocess\"` or "
                    "`harness.sandbox = false`."
                )
            argv = self._sandbox_prefix(str(workdir)) + argv

        # NOTE the absent teardown. `rm -rf agent_dir` is the obvious cleanup and
        # it BREAKS THE NEXT ROLLOUT: the supervisor at the shared socket keeps
        # live state under this directory (`daemon-workers/`, `session-leases/`),
        # so deleting it kills the supervisor, and the following rollout attaches
        # to a dead socket and dies with `DaemonSocketClosedError`. Measured twice.
        # The directory is small and carries no secret (models.json holds an env
        # var NAME, auth.json is empty, settings.json holds localhost URLs), so it
        # is left in `install_dir` for the operator to clear between runs.
        result = await self._run_once(runtime, argv, env, agent_dir, attempt=0)

        # Auto-resume: a clean exit only means `--print` mode ran out of tool
        # calls to make, not that the rollout is over (see
        # `PrimeAgentHarnessConfig.max_relaunches`). Relaunch on the *resume*
        # argv/prompt as long as the toolset-side referee says the character is
        # alive and the call budget isn't exhausted, up to the configured cap.
        resume_argv = argv[:-1] + [_RESUME_PROMPT]
        relaunches = 0
        while (
            result.exit_code == 0
            and relaunches < self.config.max_relaunches
            and self._episode_live(trace)
        ):
            relaunches += 1
            logger.info(
                "prime-agent (trace %s) exited 0 with the game still live "
                "(skill_calls=%s terminated=%s budget_exhausted=%s); "
                "relaunching (%s/%s)",
                getattr(trace, "id", "?"),
                getattr(trace.state, "skill_calls", "?"),
                getattr(trace.state, "terminated", "?"),
                getattr(trace.state, "budget_exhausted", "?"),
                relaunches,
                self.config.max_relaunches,
            )
            result = await self._run_once(
                runtime, resume_argv, env, agent_dir, attempt=relaunches
            )
        if trace is not None and hasattr(trace, "record_metric"):
            # Recorded every rollout, including 0, so a run that never needed
            # to resume is as visible in the aggregate as one that needed all 5.
            trace.record_metric("prime_agent_relaunches", float(relaunches))
        return result

    def _episode_live(self, trace) -> bool:
        """Whether the game is still worth relaunching into.

        `trace.state` is `NetHackState` (`nethack_v1.py`), synced live over the
        interception `/state` channel by the MCP tool server as it executes
        skill calls -- it reflects the CURRENT toolset-side truth even though
        the CLI process that was just driving it has already exited. `terminated`
        is engine-side game-over (death / ascension / step cap); `budget_exhausted`
        is the toolset's own `max_skill_calls` referee, which binds every arm
        including a CLI agent whose internal loop this harness does not control.
        Missing state (any taskset other than `nethack_v1`, or a bare test
        double) is treated as "not live" -- this harness should never spin on
        state it cannot interpret.
        """
        state = getattr(trace, "state", None) if trace is not None else None
        if state is None:
            return False
        return not bool(getattr(state, "terminated", True)) and not bool(
            getattr(state, "budget_exhausted", True)
        )

    async def _run_once(
        self,
        runtime: Runtime,
        argv: list[str],
        env: dict[str, str],
        agent_dir: str,
        *,
        attempt: int,
    ) -> ProgramResult:
        """Run one Prime Agent process and (best-effort) persist its streams.

        `attempt=0` keeps the original `program.{stdout,stderr}.txt` names (the
        contract `test_the_cli_streams_are_persisted_for_the_next_silent_exit`
        pins); a relaunch writes to `.relaunchN.txt` instead of overwriting the
        prior attempt's evidence, since diagnosing why an earlier attempt exited
        early is exactly what `capture_output` exists for.
        """
        result = await runtime.run_program(argv, env)
        if self.config.capture_output:
            suffix = "" if attempt == 0 else f".relaunch{attempt}"
            # Best effort: a diagnostics write must never fail a scored rollout.
            for name, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
                try:
                    await runtime.write(
                        f"{agent_dir}/program.{name}{suffix}.txt", (stream or "").encode()
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "could not persist prime-agent %s (attempt %s)", name, attempt
                    )
        return result
