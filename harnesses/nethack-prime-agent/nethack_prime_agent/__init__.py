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

import hashlib
import json
import logging
import os
import re
import secrets
import shlex
from importlib import resources
from pathlib import Path

from pydantic import Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime

from verifiers.v1.trace import Trace

# Submodule import, not `from nethack_prime_agent import ...`: this IS that
# package's __init__, so the absolute form would re-enter a half-initialised
# module. `netplay_gate` imports nothing from the package, so this is safe.
from . import netplay_gate

logger = logging.getLogger(__name__)

__all__ = ["PrimeAgentHarness", "PrimeAgentHarnessConfig"]

PROVIDER = "intercept"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"
MCP_TOKEN_VAR = "NETHACK_MCP_TOKEN"

# Where `netplay/_base.py` writes its correlation beacons. Client-side
# bookkeeping only: the published MCP tool schemas are frozen and must never be
# changed for logging, so the correlation id rides our own record. Read back at
# teardown and reconciled against the server's own call count.
CALL_LOG_VAR = "NETPLAY_CALL_LOG"

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

# The agent-editable `netplay` layer, materialised only when
# `netplay_code_mode` is not "off" (see the field's docstring for why "off" is
# the default and what each mode is for).
#
# FROZEN: rewritten on every rollout, so agent tampering with the MCP boundary
# or the primitive floor self-heals at the start of the next episode -- the
# structural rail that holds even when every validation gate is bypassed.
_NETPLAY_FROZEN_FILES = ("src/netplay/__init__.py", "src/netplay/_base.py")

# MUTABLE: the seed generation. In "mutable" mode these are written only when
# ABSENT, so what the agent wrote last episode survives into this one. In
# "frozen" mode they are rewritten like everything else, which is what makes
# the [continual-code-frozen] control a control: same retired-composite
# surface, no accumulation.
# The branch canonical's HEAD rests on between rollouts. Named explicitly
# rather than relying on git's default, which varies with `init.defaultBranch`.
_NETPLAY_BASE_BRANCH = "netplay-canonical"

# The frozen files as bare basenames under src/netplay/, for presence checks.
_NETPLAY_FROZEN_FILES_LOCAL = ("__init__.py", "_base.py")

_NETPLAY_SEED_FILES = (
    "src/netplay/explore.py",
    "src/netplay/descend.py",
    "src/netplay/fight.py",
    "src/netplay/survive.py",
)

# The no-batching instruction, verbatim. The honesty pass (9b8d5a4) rewrote
# SKILL.md and shortened this line without updating the constant, so
# `allow_batching=True` has been raising RuntimeError ever since -- the feature
# was dead on both documents. Present in SKILL.md and SKILL.baseline.md
# identically, so one constant still covers both.
_NO_BATCH_RULE = (
    "- One call, read the observation, then decide. Never batch blind sequences.\n"
)


def _within(path: str, root: str) -> bool:
    """Whether `path` is `root` or lives under it, after normalisation.

    Used to refuse a shared continual-harness store the sandbox would not bind.
    String-prefix comparison would accept `/tmp/vf-prime-agent-other`; this does
    not.
    """
    p = os.path.normpath(os.path.abspath(path))
    r = os.path.normpath(os.path.abspath(root))
    return p == r or p.startswith(r + os.sep)


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


# The coordinate-frame note aee5c43 added to SKILL.md, and the baseline wording
# it replaced. `skill_doc_coords` OFF (the default) must serve the E10-baseline
# doc BYTE-FOR-BYTE. That is now done by serving the frozen `SKILL.baseline.md`
# wholesale (see `_skill_doc`); the sentence-substitution below is superseded
# and kept only so an old caller gets the same result. Same tool_flags family as netplay_telemetry /
# melee_hints, but consumed HERE: the doc is materialized by the harness
# process, which never imports the env-side flag registry -- so it is a harness
# config field, set by the same tier that sets the env flags.
_COORD_FRAME_NOTE = (
    "Coordinates: `x` is the column (0–78, left to right), `y` is the row (0–20,\n"
    "top to bottom) in the MAP frame: row 0 is the FIRST row of the `=== MAP ===`\n"
    "block, the same frame `Pos:` and all `VISIBLE FEATURES` coordinates use. Do\n"
    "NOT count rows from the raw terminal screen (it has extra message/status\n"
    "lines) — that yields an off-by-one that silently misses every target."
)

_COORD_FRAME_BASELINE = (
    "Coordinates: `x` is the column (0–78, left to right), `y` is the row (0–20,\n"
    "top to bottom), exactly as shown in the map."
)


def _restore_baseline_coord_note(data: bytes) -> bytes:
    """Deprecated: kept so a caller that still patches gets the same result.

    Superseded by serving `SKILL.baseline.md` wholesale (see `_skill_doc`). The
    patch approach pinned ONE paragraph, so any future edit elsewhere in
    SKILL.md would have broken byte-identity with E10 while every test still
    passed. Serving the frozen file cannot drift by construction.
    """
    text = data.decode()
    if _COORD_FRAME_NOTE not in text:
        raise RuntimeError(
            "skill_doc_coords=False but the coordinate-frame note was not found "
            "verbatim in SKILL.md -- the file changed and `_COORD_FRAME_NOTE` is "
            "stale. Update it rather than serving an unknown doc as 'baseline'."
        )
    return text.replace(_COORD_FRAME_NOTE, _COORD_FRAME_BASELINE).encode()


# The E10-baseline SKILL.md, frozen. This is `aee5c43^` verbatim -- the exact
# bytes the prime_agent arm served for every rollout in outputs/e10_baseline/.
# Pinned by hash so an accidental edit to the fixture fails the run rather than
# silently redefining what "baseline" means.
# BASELINE v2 (2026-08-24). v1 was `aee5c43^` verbatim, sha256 8585082860c7...,
# the document every rollout in outputs/e10_baseline/ was served. v2 removes the
# call-budget language (see docs/PROMPT_BUDGET_REMOVAL.md), which means it is a
# NEW baseline, not a correction: E10's numbers describe v1 and remain valid for
# it. Regenerate after editing the doc:
#   sha256sum harnesses/nethack-prime-agent/nethack_prime_agent/skill/SKILL.baseline.md
_BASELINE_SKILL_SHA256 = "39f34ad07961a27cb440ced0ff53ec3001df6172b2813dfb33e7d344af3d10aa"

# The skills-as-code tier's document, frozen and hash-pinned for the same reason
# the baseline is: it defines the surface the [continual-code] arm was run
# against, and an accidental edit would silently redefine the arm rather than
# fail it. It is a WHOLE document, not a patch of SKILL.md -- the two describe
# different tool sets (this one retires the four server-side composites), so
# there is no shared paragraph to patch. Regenerate after a deliberate edit:
#   sha256sum harnesses/nethack-prime-agent/nethack_prime_agent/skill/SKILL.code.md
_CODE_SKILL_SHA256 = "9f2b05919406955b79218af2c597ed7706aafe3b4040058bd96b4c4ce08cb058"


def _skill_doc(package, *, skill_doc_coords: bool, allow_batching: bool,
               netplay_code_mode: str = "off") -> bytes:
    """The SKILL.md bytes this tier serves.

    `skill_doc_coords=False` (the default, i.e. [base]) serves the frozen
    baseline file WHOLESALE rather than patching the live one. That is the only
    way byte-identity with E10 survives future edits to SKILL.md: a patch pins
    one paragraph, a fixture pins the document.

    `netplay_code_mode != "off"` serves `SKILL.code.md` instead, and takes
    precedence over `skill_doc_coords`: the code tiers retire four of the tools
    both other documents describe, so serving either would advertise calls that
    raise. Like the baseline it is a frozen, hash-pinned whole document.
    """
    if str(netplay_code_mode).strip().lower() != "off":
        data = (package / "SKILL.code.md").read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != _CODE_SKILL_SHA256:
            raise RuntimeError(
                "SKILL.code.md has been edited: expected sha256 "
                f"{_CODE_SKILL_SHA256}, got {got}. This file defines the surface "
                "the [continual-code] arm runs against; changing it silently "
                "redefines the arm. Update _CODE_SKILL_SHA256 deliberately."
            )
        if allow_batching:
            data = _strip_no_batch_rule(data)
        return data
    if not skill_doc_coords:
        data = (package / "SKILL.baseline.md").read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != _BASELINE_SKILL_SHA256:
            raise RuntimeError(
                "SKILL.baseline.md has been edited: expected sha256 "
                f"{_BASELINE_SKILL_SHA256}, got {got}. This file is the E10 "
                "baseline document; changing it silently redefines every [base] "
                "cell. Restore it from `git show aee5c43^`."
            )
    else:
        data = (package / "SKILL.md").read_bytes()
    if allow_batching:
        if not skill_doc_coords:
            # The hash check above verifies the FIXTURE; stripping afterwards
            # changes the SERVED bytes, so a [base] cell was serving 4753 bytes
            # where E10 served 4829 and nothing fired. The launcher refuses the
            # combination; this refuses it again for any other caller.
            raise RuntimeError(
                "allow_batching=True with skill_doc_coords=False would strip the "
                "no-batch rule out of the frozen E10 baseline document, so the "
                "bytes served would not be the bytes E10 served. A batching cell "
                "is a different experiment: run it on its own tier."
            )
        data = _strip_no_batch_rule(data)
    return data


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
    """`--thinking` level.

    CORRECTED 2026-08-24: this said "Empty leaves Prime Agent's default
    (`xhigh`)". The shipped 0.3.3 bundle says otherwise --
    `DEFAULT_THINKING_LEVEL = "medium"` -- so every cell that left this empty has
    been running at MEDIUM reasoning effort while the docstring (and
    docs/settings.md) claimed xhigh. Reasoning effort is not a free variable in a
    controlled comparison: pin it per arm rather than inheriting an undeclared
    default that a scaffold upgrade can move."""

    websearch: bool = False
    """Load Prime Agent's bundled `websearch` skill. Off by default: the control
    arm and the Claude Code arm both run without web access, and a live search
    tool would not be capability-matched."""

    continual_harness_dir: str = ""
    """Share Prime Agent's GLOBAL continual-harness store across rollouts (E13).

    Prime Agent persists prompt notes, memories, reusable skill descriptions and
    sub-agent specs in a "continual harness" state file, and renders them into
    the system prompt of every new session (`formatHarnessStateForPrompt`, called
    from the base-prompt builder with `harnessState: _loadMergedHarnessState()`).
    That is exactly the cross-episode learning channel E13 needs -- but it is
    inert under this harness, for two independent reasons:

      * LOCAL state lives in session artifacts, and this arm runs `--no-session`.
      * GLOBAL state lives at `<agentDir>/harness/`, and `agentDir` here is the
        PER-ROLLOUT `agent-<trace id>` (see `PRIME_AGENT_CODING_AGENT_DIR`
        below), so "global" is really "per game".

    Setting this to a directory makes `<agent_dir>/harness` a SYMLINK to it, so
    every rollout reads and (if `continual_harness_writable`) writes one shared
    store. Only the harness state is shared: `settings.json`, `models.json` and
    `auth.json` stay per-rollout, which they must -- seeds run CONCURRENTLY and
    each carries its own MCP URL and interception secret, so sharing the whole
    agent directory would race five rollouts onto one settings file and point
    agents at each other's games.

    Empty (the default) leaves every existing arm byte-identical: no symlink is
    created and each rollout keeps its own empty per-rollout store.

    Note what this does NOT switch on. `_autoRefineAllowedForSession()` requires
    a local (session) harness dir, so under `--no-session` automatic refinement
    never fires; and auto-refine writes LOCAL entries anyway. Entries reach the
    shared store only through an explicit `rlm.harness.create_*(global_=True)` /
    `await refine.run(..., global_=True)` call -- from the player, if the prompt
    asks for one, or from an orchestrator process pointed at the same directory
    between cells. That is a feature for an experiment: every write is deliberate
    and attributable, not a background process editing the arm mid-cell."""

    rlm_max_depth: int = 1
    """Recursion depth for `rlm(...)` sub-agents, written into the launch env.

    1 = the root player may spawn children; children may not spawn
    grandchildren. That is the scaffold default, but there is no CLI flag and no
    settings key for it, so inheriting it leaves no record of what the arm
    actually ran with."""

    auto_refine: bool = False
    """Prime Agent's automatic trajectory review (`autoRefine`).

    Default OFF and always written into settings.json explicitly, because the
    scaffold's own default is ON and only inert by accident under `--no-session`
    (see the settings block in `launch`). Turning it on is a deliberate arm
    choice with two consequences worth stating: it costs two extra model calls
    per fire on the interception endpoint, unattributed to the rollout; and its
    edits are LOCAL-scope, landing in the session artifact dir rather than the
    shared continual store, so they are discarded unless something harvests
    them."""

    continual_harness_mode: str = "shared-ro"
    """How the shared continual-harness store reaches a rollout.

    `shared-ro` (default) symlinks every rollout's harness directory at ONE
    store. Correct when a single writer (an orchestrator, between cells) owns
    it, and the store is re-bound read-only so players cannot edit it.

    `copy-merge` is for the arm where the PLAYERS write. Each rollout gets a
    private COPY of the canonical store, seeded at launch so it reads everything
    learned so far, and the runner merges the copies back afterwards.

    Why copying rather than sharing a writable store, measured: `rlm.harness`
    persists with a non-atomic whole-file rewrite -- `open("w")` + `json.dump`,
    no lock, no tmp+rename -- and its loader treats an unreadable state file as
    EMPTY rather than erroring. Five concurrent writers each adding six entries
    to one store kept 12 of 30; one of the five contributed nothing at all, and
    nothing reported the loss. A player landing mid-write would silently boot
    with no accumulated knowledge.

    Copying also buys attribution: each rollout's contribution is a separate
    file, so "what did seed 7 add?" is answerable and same-id conflicts become a
    deliberate merge decision instead of last-write-wins."""

    continual_harness_writable: bool = False
    """Let the PLAYER write to the shared continual-harness store.

    Default False: under `sandbox = true` the store is re-bound READ-ONLY, on top
    of the read-write `install_dir` bind, so a rollout can read the accumulated
    lessons and cannot edit them. That keeps the learning channel single-writer
    (an orchestrator between cells) and makes a cell reproducible from the store
    snapshot taken before it ran.

    Set True for the self-directed variant, where the player itself decides what
    to persist. Then the store is shared MUTABLE state across concurrently
    running seeds: writes are last-write-wins on one JSON file, so run such a
    cell with one seed at a time, and expect a cell to change the store that
    later cells read. With `sandbox = false` this flag cannot be enforced at all
    (nothing is bound read-only) -- it is then a declaration of intent that the
    launcher's before/after hash of the store has to police."""

    netplay_code_mode: str = "off"
    """Whether the agent-editable `netplay` code layer is materialised, and
    whether the agent's edits to it survive the episode.

    `off` (default) -- no `src/netplay` at all. The skill tree is byte-identical
    to what every arm shipped before this feature existed, which is what keeps
    `base` a valid denominator. Every existing arm runs in this mode.

    `frozen` -- the whole tree is materialised and REWRITTEN each rollout, seed
    composites included. The agent can edit `netplay/*` within an episode and
    the edits are reverted before the next one. This is the [continual-code-frozen]
    control from SPEC §7.2, and without it a gain in the code arm cannot be
    attributed: it would be indistinguishable from "our seed implementations
    beat the vendored `np_*` ones".

    `mutable` -- the treatment. `_base.py` and `__init__.py` are still rewritten
    every rollout (the boundary is not negotiable), but the four seed composites
    are written only when absent, so the agent's own code accumulates across
    episodes and rounds.

    `pinned` -- held-out evaluation. NOTHING is materialised: whatever tree the
    caller placed at `{skill_dir}/src/netplay` is what runs, untouched. This is
    the only mode that evaluates the agent's OWN final code, and it exists
    because neither of the other two can: `mutable` would let the evaluation
    write to the artifact it is measuring, and `frozen` would overwrite the
    agent's composites with our repo seeds and quietly measure those instead.
    The teardown gate still runs (a broken frozen tree is worth knowing about)
    but nothing is committed, because a held-out cell must not accumulate.
    Pair it with a read-only bind; `eval_frozen.sh` also hashes the tree before
    and after and fails the run if it changed.

    Two things this mode does NOT change, deliberately: `pyproject.toml` is
    never touched, so `pyprojectHash` stays stable and no rollout triggers the
    ~10-minute kernel-venv rebuild; and the skill directory path stays fixed,
    so the venv identity key is unaffected.

    `mutable` shares one tree across concurrently running seeds unless the
    sandbox gives each rollout its own bind -- see the launch-time refusal in
    `setup`, which is the same single-writer discipline
    `continual_harness_writable` documents for the JSON store."""

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

    skill_doc_coords: bool = False
    """Serve the SKILL.md coordinate-frame note (aee5c43). Default False =
    serve the E10-baseline doc byte-for-byte (the note is swapped back to the
    baseline wording at materialization -- `_restore_baseline_coord_note`).
    The [human] tier turns this on together with the env-side tool_flags
    (netplay_telemetry, melee_hints); see configs/tool_tiers.toml."""

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

    async def _path_exists(self, runtime: Runtime, path: str) -> bool:
        """Whether `path` exists in the runtime.

        `Runtime` has no `exists`; `read` on a missing file raises, and which
        exception depends on the runtime implementation (the subprocess one
        surfaces the OS error, remote ones wrap it). So this catches broadly on
        purpose -- an unreadable file and an absent one both mean "write the
        seed", and guessing wrong costs one overwrite of a file we shipped.
        """
        try:
            await runtime.read(path)
        except Exception:
            return False
        return True

    def _netplay_mode(self) -> str:
        """The resolved netplay_code_mode. Quote-stripped (see the materialiser
        note on the json.dumps era) and lowercased."""
        return str(self.config.netplay_code_mode).strip().strip('"\'').lower()

    @property
    def _netplay_canonical(self) -> str:
        """The one git repo the mutable arm accumulates in. SEPARATE from the
        fixed skill path, which under a sandbox is only a per-rollout bind
        mountpoint. run_e13.sh seeds this before round 1."""
        return f"{self.config.install_dir}/netplay-canonical"

    def _netplay_work(self, trace_id: str) -> str:
        """This rollout's PRIVATE netplay tree. Per-rollout like `agent-<id>`,
        so parallel rollouts never share it. Bound over the fixed skill path
        inside the sandbox (the same later-narrower-bind trick the store uses),
        which is what lets the mutable arm run in parallel."""
        return f"{self.config.install_dir}/netplay-work/{trace_id}"

    def _netplay_parallel(self) -> bool:
        """True when this rollout gets its own bound work tree. Only the mutable
        arm under a sandbox: the bind is what isolates concurrent writers, and
        without the sandbox there is no bind (sandbox=false stays serial)."""
        return self.config.sandbox and self._netplay_mode() == "mutable"

    async def _provision_netplay_worktree(self, runtime: Runtime, trace_id: str,
                                          package) -> str:
        """Clone canonical into this rollout's private tree, on its own branch.

        A CLONE, not a shared worktree: clone only READS canonical, so any
        number of rollouts can provision concurrently with no lock. Teardown
        commits into the clone; the round merge fetches each clone's branch back
        into canonical. Returns the work-tree path to bind.
        """
        canonical, work = self._netplay_canonical, self._netplay_work(trace_id)
        if not await self._path_exists(runtime, f"{canonical}/.git/HEAD"):
            raise RuntimeError(
                f"netplay canonical repo missing at {canonical}. It must be "
                "seeded once, serially, before any rollout -- run_e13.sh does "
                "this before round 1. A rollout cannot seed it safely because "
                "concurrent rollouts would race to create it."
            )
        script = (
            f"set -e; rm -rf {work}; mkdir -p {work}; "
            f"git clone -q {canonical} {work}; cd {work}; "
            f"git checkout -q -B agent-{trace_id} {_NETPLAY_BASE_BRANCH}; "
            f"git config user.email netplay@localhost; "
            f"git config user.name 'netplay rollout'"
        )
        probe = await runtime.run(["sh", "-c", script], self._env_with_path())
        if probe.exit_code != 0:
            raise RuntimeError(
                "netplay worktree clone failed: "
                f"{(probe.stderr or probe.stdout or '').strip()[-300:]}"
            )
        # Re-heal the frozen boundary in the private tree, on top of the base.
        for name in _NETPLAY_FROZEN_FILES_LOCAL:
            await runtime.write(f"{work}/{name}",
                                (package / f"src/netplay/{name}").read_bytes())
        return work

    async def _materialise_netplay(self, runtime: Runtime, package) -> None:
        """Write the agent-editable `netplay` layer, per `netplay_code_mode`.

        Frozen files are always overwritten. Seed composites are overwritten in
        `frozen` mode and written only-if-absent in `mutable` mode -- that
        single difference is the whole treatment/control split of SPEC §7.2.
        """
        # `.strip('"\'')` is defence in depth, not decoration. The tier
        # expander used to emit this value through `json.dumps`, which for a
        # STRING yields `"mutable"` WITH quotes -- and the eval CLI passes a
        # bare scalar through verbatim, so the harness received the quoted
        # form and refused every code-tier cell at setup. The expander is
        # fixed; this keeps an already-written config.toml from that era
        # loadable instead of failing on a value that is obviously intended.
        mode = str(self.config.netplay_code_mode).strip().strip('"\'').lower()
        if mode == "off":
            return
        if mode not in ("frozen", "mutable", "pinned"):
            raise ValueError(
                f"netplay_code_mode must be 'off', 'frozen', 'mutable' or "
                f"'pinned', got {self.config.netplay_code_mode!r}"
            )
        if mode == "pinned":
            # Held-out evaluation. Writing ANYTHING here would replace part of
            # the artifact under test, so this mode's whole contract is that it
            # writes nothing -- including the frozen files, which are already
            # in the tree the caller pinned.
            # Verify the WHOLE expected surface is present, not just one file:
            # a tree with _base.py but no __init__.py imports as a namespace
            # package with none of the policies, and the eval would silently
            # measure an empty netplay. Check both frozen files plus at least
            # one composite.
            missing = []
            for n in _NETPLAY_FROZEN_FILES_LOCAL:
                if not await self._path_exists(runtime, f"{self._skill_dir}/src/netplay/{n}"):
                    missing.append(f"src/netplay/{n}")
            has_any_composite = False
            for n in _NETPLAY_SEED_FILES:
                if await self._path_exists(runtime, f"{self._skill_dir}/{n}"):
                    has_any_composite = True
                    break
            if missing or not has_any_composite:
                raise ValueError(
                    "netplay_code_mode='pinned' but the tree at "
                    f"{self._skill_dir}/src/netplay is incomplete "
                    f"(missing frozen: {missing or 'none'}; any composite: "
                    f"{has_any_composite}). Pinned mode materialises nothing by "
                    "design -- eval_frozen.sh must copy the agent's full frozen "
                    "tree into place before the cell starts."
                )
            logger.info("netplay: pinned tree, materialising nothing")
            return

        tree = f"{self._skill_dir}/src/netplay"

        async def _write_frozen() -> None:
            for name in _NETPLAY_FROZEN_FILES:
                await runtime.write(f"{self._skill_dir}/{name}",
                                    (package / name).read_bytes())

        async def _write_seeds() -> None:
            for name in _NETPLAY_SEED_FILES:
                await runtime.write(f"{self._skill_dir}/{name}",
                                    (package / name).read_bytes())

        if mode == "frozen":
            # Control arm: rewrite everything every rollout, so nothing the agent
            # does accumulates. No git, no base -- frozen contributes nothing by
            # design, which is what makes it the control.
            await _write_frozen()
            await _write_seeds()
            return

        if self._netplay_parallel():
            # Parallel mutable: the accumulating git repo is canonical
            # ({install_dir}/netplay-canonical), and each rollout edits a private
            # CLONE bound over this fixed path in the sandbox (provisioned in
            # launch). Here we only need a valid tree at the fixed path so the
            # bind has a mountpoint and the sandbox=false fallback still works --
            # no git on the fixed path, since the per-rollout clones own it.
            await _write_frozen()
            await _write_seeds()
            return

        # mutable, SERIAL (sandbox=false): the git-based accumulation model runs
        # on the fixed path itself. Each rollout starts from the round base --
        # the `netplay-canonical` branch, which is the seed on
        # round 1 and the merged tree thereafter -- edits independently, and
        # teardown commits its generation on `agent-<id>` off that base. The
        # round merge (merge_netplay_code.py) combines the branches.
        #
        # RUN ONE SEED AT A TIME. Every rollout of a round shares this one tree
        # and this one repo; two concurrent teardown commits would race and the
        # round would merge a chimera. run_e13.sh pins MAX_CONCURRENT=1 for this
        # tier -- the same single-writer discipline `continual_harness_writable`
        # documents for the JSON store, and enforced the same way (by the
        # launcher, because the harness cannot see the concurrency).
        has_repo = await self._path_exists(runtime, f"{tree}/.git/HEAD")
        if not has_repo:
            # First rollout ever: write the seeds and commit them as the pristine
            # base BEFORE the agent touches anything, so every rollout of round 1
            # starts from identical code. Committing the base at teardown instead
            # -- which an earlier version did -- baked the first agent's edits
            # into what every later rollout called "base".
            await _write_frozen()
            await _write_seeds()
            init = (
                f"set -e; cd {tree}; git init -q -b {_NETPLAY_BASE_BRANCH} .; "
                f"printf '__pycache__/\\n*.pyc\\n' > .gitignore; "
                f"git config user.email netplay@localhost; "
                f"git config user.name 'netplay seed'; "
                f"git add -A; git commit -q -m 'round-0 seed'; "
                f"git tag -f round-0"
            )
            probe = await runtime.run(["sh", "-c", init], self._env_with_path())
            if probe.exit_code != 0:
                raise RuntimeError(
                    "netplay seed init failed: "
                    f"{(probe.stderr or probe.stdout or '').strip()[-300:]}"
                )
        else:
            # Later rollout: reset the working tree to the round base, discarding
            # the previous rollout's edits (already captured on its own branch),
            # so this rollout starts independent. `clean -fd` drops any file a
            # prior rollout created that base does not have.
            restore = (
                f"set -e; cd {tree}; "
                f"git checkout -q -f {_NETPLAY_BASE_BRANCH}; "
                f"git reset -q --hard {_NETPLAY_BASE_BRANCH}; "
                f"git clean -q -fd"
            )
            probe = await runtime.run(["sh", "-c", restore], self._env_with_path())
            if probe.exit_code != 0:
                raise RuntimeError(
                    "netplay base restore failed: "
                    f"{(probe.stderr or probe.stdout or '').strip()[-300:]}"
                )
            # Re-heal the frozen boundary on top of the restored base, in case a
            # merged generation ever carried a tampered copy past the gate.
            await _write_frozen()

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
            if name == "SKILL.md":
                # Tier-selected document; the baseline variant is a frozen file,
                # not a patch (see `_skill_doc`). `SKILL.baseline.md` is never
                # written into the runtime skill dir -- the agent must see
                # exactly one SKILL.md.
                data = _skill_doc(
                    package,
                    skill_doc_coords=self.config.skill_doc_coords,
                    allow_batching=self.config.allow_batching,
                    netplay_code_mode=self.config.netplay_code_mode,
                )
            else:
                data = (package / name).read_bytes()
            await runtime.write(f"{self._skill_dir}/{name}", data)

        await self._materialise_netplay(runtime, package)

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

    def _sandbox_prefix(self, workdir: str, netplay_src: str | None = None) -> list[str]:
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
            # Skills-as-code: this rollout's PRIVATE netplay tree, re-bound over
            # the fixed skill path AFTER the install_dir bind so the later,
            # narrower mount wins (the same ordering the store re-bind below
            # relies on). Inside the sandbox `src/netplay` is this rollout's own
            # tree; the venv key is the skill-package path, which is unchanged,
            # so no rebuild. This is what lets the mutable arm run in parallel.
            *(
                ["--bind", netplay_src,
                 f"{self._skill_dir}/src/netplay"]
                if netplay_src else []
            ),
            # E13: in `shared-ro` the store is re-bound READ-ONLY on top of the
            # read-write `install_dir` bind (bwrap applies binds in order, so the
            # later, narrower one wins) -- that is what keeps a single-writer arm
            # single-writer. In `copy-merge` the rollout writes its OWN private
            # copy under install_dir and canonical is never touched by a player,
            # so no re-bind applies.
            *(
                ["--ro-bind", self.config.continual_harness_dir,
                 self.config.continual_harness_dir]
                if self.config.continual_harness_dir
                and str(self.config.continual_harness_mode).strip().lower() != "copy-merge"
                and not self.config.continual_harness_writable
                else []
            ),
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
        _system_prompt, prompt = self.resolve_prompt(trace.task.data)  # prompt only; AGENTS.md carries the system prompt
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
            # PINNED, not inherited. `autoRefine` defaults to enabled (turnInterval
            # 25, 20-minute cooldown) and is undocumented in settings.md. It is
            # inert here only by ACCIDENT -- `_autoRefineAllowedForSession()`
            # needs a local session dir and `--no-session` denies one -- but the
            # moment a player calls `rlm(...)`, the host mints an ephemeral RLM
            # session dir on the PARENT, the gate opens mid-run, and auto-refine
            # starts spending two out-of-band `completeSimple` calls per fire on
            # the eval's own interception endpoint. Those are real completions
            # that never enter the session transcript, so they land in the run's
            # provider cost unattributed, and their edits are LOCAL-scope and
            # discarded. An arm must not change behaviour because the model
            # happened to spawn a sub-agent.
            "autoRefine": {"enabled": self.config.auto_refine},
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

        # E13: give this rollout its GLOBAL continual-harness directory, so
        # lessons written by an earlier game are in this game's system prompt.
        # `getGlobalHarnessStateDir()` is `join(agentDir, "harness")` with no env
        # override of its own, and the kernel is handed the same path as
        # `RLM_GLOBAL_HARNESS_STATE_DIR`, so whatever sits at that one name
        # serves both the host (refine) and the kernel (`rlm.harness.*`).
        if self.config.continual_harness_dir:
            ch = self.config.continual_harness_dir
            if self.config.sandbox and not _within(ch, self.config.install_dir):
                raise ValueError(
                    f"harness.continual_harness_dir ({ch!r}) is outside "
                    f"install_dir ({self.config.install_dir!r}) while "
                    "harness.sandbox is true, so it is not bound into the "
                    "sandbox and the shared store would be invisible to the "
                    "agent. Put the store under install_dir, or set "
                    "harness.sandbox = false."
                )
            mode = str(self.config.continual_harness_mode or "shared-ro").strip().lower()
            if mode not in ("shared-ro", "copy-merge"):
                raise ValueError(
                    f"harness.continual_harness_mode={mode!r}; expected "
                    "'shared-ro' or 'copy-merge'."
                )
            link = f"{agent_dir}/harness"
            if mode == "copy-merge":
                # A private copy, seeded from canonical. `cp -a .../.` copies the
                # CONTENTS so an absent canonical store still yields an empty
                # private one rather than a nested directory -- and an empty
                # canonical is a rc=0 no-op, so no `|| true` is needed to
                # tolerate round 1. It must NOT be suppressed: `... || true`
                # forced the whole script to exit 0, which made the exit-code
                # check below dead in this mode and let a failed copy hand the
                # rollout an empty store -- exactly the "the agent learned
                # nothing" reading the check exists to prevent.
                script = (
                    f"mkdir -p {shlex.quote(ch)} {shlex.quote(link)} && "
                    f"cp -a {shlex.quote(ch)}/. {shlex.quote(link)}/"
                )
            else:
                script = (
                    f"mkdir -p {shlex.quote(ch)} && rm -rf {shlex.quote(link)} && "
                    f"ln -sfn {shlex.quote(ch)} {shlex.quote(link)}"
                )
            probe = await runtime.run(["sh", "-c", script], self._env_with_path())
            if probe.exit_code != 0:
                raise RuntimeError(
                    f"could not provision the continual-harness store ({mode}) at "
                    f"{link}: "
                    f"{(probe.stderr or probe.stdout).strip()[-300:] or '<no output>'}. "
                    "Running on would silently give this rollout an empty store."
                )

        env = self._env_with_path(
            {
                KEY_VAR: secret,
                MCP_TOKEN_VAR: mcp_token,
                "PRIME_AGENT_CODING_AGENT_DIR": agent_dir,
                # Per-rollout, under the per-rollout agent dir, so five
                # concurrent seeds never share a log. Absent when the layer is
                # off, and `_base` degrades to not logging rather than raising.
                **(
                    {CALL_LOG_VAR: f"{agent_dir}/netplay_calls.jsonl"}
                    if str(self.config.netplay_code_mode).strip().lower() != "off"
                    else {}
                ),
                # DECLARED, not inherited. There is no CLI flag and no settings
                # key for recursion depth -- this env var is the only record. The
                # default is 1 (root may spawn children, children may not
                # recurse), which is what we want; writing it down means a
                # scaffold upgrade cannot move it silently.
                "RLM_MAX_DEPTH": str(self.config.rlm_max_depth),
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
        # De-dup (2026-08-21): the workspace AGENTS.md already carries the full
        # resolved system prompt and Prime Agent embeds it as Project Context,
        # so ALSO passing --append-system-prompt delivered the identical
        # gameplay block twice in node 0 (measured in the v3 seed-2 post-mortem
        # at chars 14616 and 21992). AGENTS.md is the single source now.
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
            netplay_src = None
            if self._netplay_parallel():
                # Private, bound per-rollout tree -> parallel-safe. Provisioned
                # here (not setup) because it needs the trace id, and cloned from
                # canonical so every rollout starts from the round base.
                netplay_src = await self._provision_netplay_worktree(
                    runtime, str(trace.id).replace("/", "-"),
                    resources.files(__package__) / "skill",
                )
            argv = self._sandbox_prefix(str(workdir), netplay_src) + argv

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
        await self._netplay_teardown(runtime, trace, agent_dir)
        return result

    async def _netplay_teardown(self, runtime: Runtime, trace, agent_dir: str) -> None:
        """Validate and commit the agent's code tree after the CLI exits.

        Three things happen here, in this order, and the order matters:

        1. **The gate runs unconditionally.** SKILL.md asks the agent to call
           `netplay.check()` after every edit; agents skip advice. This is the
           gate that actually holds, and a tree that fails it is recorded as
           failed so the round merge can refuse the branch (SPEC 6.1 rail 2).
        2. **The call log is counted**, for reconciliation against the tool
           server's own record of this rollout. A policy that produced
           observations without matching server-side calls fabricated them.
        3. **The tree is committed** on a branch named for this rollout, which
           is what makes the round-boundary three-way merge possible at all and
           what makes "what did seed 7 add?" answerable as `git log`.

        Nothing here raises. A rollout that already produced a game result must
        not be turned into a harness error by its bookkeeping -- the findings
        are recorded on the trace and the merge decides what to do with them.
        """
        mode = self._netplay_mode()
        if mode == "off":
            return

        trace_id = str(getattr(trace, "id", "unknown")).replace("/", "-")
        # Under a parallel mutable rollout the agent's edits are in its private
        # bound tree, NOT at the fixed skill path (which the bind shadowed). Read
        # and commit THAT tree, or the gate would inspect an empty mountpoint and
        # the commit would capture nothing.
        parallel = self._netplay_parallel()
        tree = self._netplay_work(trace_id) if parallel else f"{self._skill_dir}/src/netplay"

        # --- 1. the gate ---
        findings: list[dict] = []
        try:
            listing = await runtime.run(
                ["sh", "-c", f"ls -1 {tree}/*.py 2>/dev/null || true"],
                self._env_with_path(),
            )
            names = [ln.strip() for ln in (listing.stdout or "").splitlines() if ln.strip()]
            for remote_path in names:
                name = remote_path.rsplit("/", 1)[-1]
                try:
                    source = (await runtime.read(remote_path)).decode("utf-8")
                except Exception as exc:  # noqa: BLE001
                    findings.append({"file": name, "line": 0, "kind": "unreadable",
                                     "detail": str(exc)})
                    continue
                findings.extend(netplay_gate.check_source(
                    Path(name), source,
                    policy=name not in netplay_gate.FROZEN_FILES,
                ))
            for missing in sorted(netplay_gate.FROZEN_FILES - {n.rsplit("/", 1)[-1] for n in names}):
                findings.append({"file": missing, "line": 0, "kind": "missing",
                                 "detail": "frozen file deleted from the tree"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("netplay gate could not run for trace %s: %s", trace_id, exc)
            findings.append({"file": "<gate>", "line": 0, "kind": "gate-error",
                             "detail": str(exc)})

        if findings:
            logger.warning(
                "netplay gate FAILED for trace %s (%d finding(s)): %s",
                trace_id, len(findings),
                "; ".join(f"{f['file']}:{f['line']} {f['kind']}={f['detail']}"
                          for f in findings[:8]),
            )
        else:
            logger.info("netplay gate passed for trace %s", trace_id)

        # --- 2. the client-side call count ---
        calls = 0
        try:
            raw = await runtime.read(f"{agent_dir}/netplay_calls.jsonl")
            calls = sum(
                1 for line in raw.decode("utf-8").splitlines()
                if '"phase":"call"' in line.replace(" ", "")
            )
        except Exception:  # noqa: BLE001 -- an absent log means zero calls
            pass

        if trace is not None and hasattr(trace, "record_metric"):
            trace.record_metric("netplay_gate_ok", 0.0 if findings else 1.0)
            trace.record_metric("netplay_gate_findings", float(len(findings)))
            trace.record_metric("netplay_client_calls", float(calls))

        # --- 3. commit this rollout's generation ---
        # `git -C` on a tree that is not yet a repo initialises it; every later
        # rollout branches from whatever canonical left there. Committing even a
        # FAILED tree is deliberate: the branch is the evidence, and the merge
        # is where it gets refused.
        if mode != "mutable":
            # frozen and pinned contribute nothing that persists: frozen is the
            # control, pinned is held-out evaluation. Gate them (a broken tree
            # is worth knowing about) but never commit -- a commit here would
            # create a branch off the artifact under test.
            logger.info("netplay: %s mode, gate only, not committing trace %s",
                        mode, trace_id)
            return

        # Commit this rollout's generation on its own branch.
        verdict = "pass" if not findings else f"fail:{len(findings)}"
        if parallel:
            # The private clone is already on branch agent-<id> (provisioning did
            # the checkout), so just capture the working tree. The round merge
            # fetches this branch out of the clone into canonical.
            script = (
                f"set -e; cd {tree}; git add -A; "
                f"git commit -q -m 'rollout {trace_id}: gate={verdict} calls={calls}' "
                f"--allow-empty"
            )
        else:
            # Serial (sandbox=false): the fixed path IS canonical. `-B agent-<id>`
            # with no start point branches at the current HEAD (the round base
            # that setup restored) while KEEPING the working edits, then HEAD is
            # returned to the base branch for the next rollout and the merge.
            script = (
                f"set -e; cd {tree}; "
                f"git config user.email netplay@localhost; "
                f"git config user.name 'netplay rollout'; "
                f"git checkout -q -B agent-{trace_id}; "
                f"git add -A; "
                f"git commit -q -m 'rollout {trace_id}: gate={verdict} calls={calls}' "
                f"--allow-empty; "
                f"git checkout -q {_NETPLAY_BASE_BRANCH}"
            )
        try:
            probe = await runtime.run(["sh", "-c", script], self._env_with_path())
            if probe.exit_code != 0:
                logger.warning(
                    "netplay commit failed for trace %s: %s",
                    trace_id, (probe.stderr or probe.stdout or "").strip()[-300:],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("netplay commit could not run for trace %s: %s", trace_id, exc)

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
