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

# Files copied out of this distribution into the runtime to form the skill
# package (`skills.md#python-backed-skills` layout).
_SKILL_FILES = ("SKILL.md", "pyproject.toml", "src/nethack/__init__.py")


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
            await runtime.write(f"{self._skill_dir}/{name}", data)

    # -- launch -------------------------------------------------------------

    def _env_with_path(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {**self.config.resolved_env, **(extra or {})}
        if self.config.path_prepend:
            import os

            inherited = env.get("PATH") or os.environ.get("PATH", "")
            env["PATH"] = f"{self.config.path_prepend}:{inherited}"
        return env

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

        agent_dir = f"{self.config.install_dir}/agent-{trace.id}"
        mcp_token = secrets.token_hex(16)

        reasoning = self.config.reasoning
        if reasoning is None:
            reasoning = ctx.sampling.reasoning_effort not in (None, "none")
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    # An `apiKey` that names an env var resolves to its value
                    # (models.md, Value Resolution), so the interception secret
                    # never lands on disk.
                    "apiKey": KEY_VAR,
                    "models": [
                        {
                            "id": ctx.model,
                            "reasoning": bool(reasoning),
                            "input": ["text"],
                        }
                    ],
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

        try:
            return await runtime.run_program(argv, env)
        finally:
            try:
                await runtime.run(["rm", "-rf", agent_dir], {})
            except Exception:  # pragma: no cover - teardown is best-effort
                logger.warning("failed to clean up the Prime Agent config dir", exc_info=True)
