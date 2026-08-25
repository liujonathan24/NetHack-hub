"""
nethack_v1
==========

Verifiers **v1 (0.2.x) taskset** port of the NetHack environment.

Arm split (decided in Task 12; see `.superpowers/sdd/.../task-12-report.md`)
--------------------------------------------------------------------------
The experiment runs two kinds of arm over one environment package, and 0.2.x
gives each its own first-class path:

* **Control arm (arm 0)** — the purpose-built harness loop in ``nethack.py``
  (``NetHackVerifiersEnv``, a v0 ``vf.StatefulToolEnv``). It runs through
  0.2.x's **legacy (v0) bridge** (`verifiers/v1/legacy.py`), i.e.::

      vf-eval --id nethack --args '{"task_spec": "full_nle", ...}'

  ``EnvConfig.id`` + no ``taskset.id`` selects the bridge
  (`verifiers/v1/env.py:128`), which loads the v0 env, runs the v0 rollout
  verbatim (`legacy.py:517`, ``env.run_rollout``) and maps the result into a v1
  ``Trace``. **No game logic, dispatch order, prompt compaction, or reward code
  changes for the control arm** — that is the whole point: its semantics were
  validated by a prior experiment and must not drift.

* **CLI-agent arms** — Claude Code (`verifiers.v1.harnesses.claude_code`,
  ``SUPPORTS_MCP = True``) and the external Prime Agent plugin. These drive the
  game **over MCP**, so they need a native v1 taskset whose tools execute and
  return the observation from the call itself. That is what this module builds::

      vf-eval --taskset.id nethack_v1 --harness.id claude_code ...

  A single ``EnvConfig`` is one or the other (`env.py:128`), so the two arms are
  two configs over one package — they cannot, and need not, be one taskset.

v0 -> v1 (0.2.x) mapping
------------------------
* **Toolset** (:class:`NetHackToolset`) is a *server* (`v1/mcp/server.py`). It
  owns the per-rollout engine: ``setup_task`` calls the v0 ``setup_state``
  (creating the long-lived ``NetHackCoreEnv``/``CurriculumPrimitivesEnv``), and
  a callback on ``ServerBase._exit_stack`` closes it when the server exits.
  ``_register`` publishes exactly the v0 adapter set — which is where the
  **netplay gate** lives (``move`` is withheld, so it is never advertised) —
  and every published tool routes through the v0 ``_apply_tool_call``, the one
  execution path.
* **State** (:class:`NetHackState`) is a typed ``vf.State`` synchronized over
  the interception ``/state`` channel (`v1/mcp/server.py:190-216`). The v0
  free-form state dict never leaves the tool-server process; only the scalars
  the rewards and stop conditions read cross the wire.
* **Task** (:class:`NetHackTask`) carries the rewards (``@vf.reward`` methods
  delegating to the v0 implementations verbatim) and the stop conditions
  (``@vf.stop``), and declares ``tools = (NetHackToolset,)`` so the server is
  built **per rollout**.
* **Taskset** (:class:`NetHackTaskset`) yields the curriculum rows
  (``full_nle`` / ``six_floor_primitives`` x seeds) as :class:`NetHackTaskData`.
* **Harness** (:class:`NetHackHarness`) is the taskset's *default* harness: the
  built-in MCP chat loop (``NullHarness``). It exists so that a bare
  ``--taskset.id nethack_v1`` defaults to a tool-only agent rather than
  ``bash`` (`v1/loaders.py:109-117`), which would hand the agent shell access to
  the host and confound the comparison. The CLI arms override it with
  ``--harness.id claude_code`` / the Prime Agent plugin. NOTE: this is NOT the
  0.1.14 ``NetHackHarness`` (which reproduced the v0 loop in-process);
  0.2.x harnesses launch an external program (`v1/harness.py:136-153`) and the
  v0 loop now lives on the legacy bridge instead.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
import inspect
import random
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Optional

import verifiers.v1 as vf
from verifiers.v1.harnesses.null import NullHarness
from verifiers.v1.loaders import load_harness as _vf_load_harness

# Reuse the entire v0 env: construction, game logic, encoders, curriculum.
from nethack import (
    FULL_GAME_SPEC,
    GAME_SPECS,
    load_environment as _load_v0_environment,
)

# Reuse the v0 reward implementations verbatim (do NOT reimplement).
from nethack_harness.helpers import (
    scout_reward as _v0_scout_reward,
    descent_reward as _v0_descent_reward,
    success_reward as _v0_success_reward,
    ascension_reward as _v0_ascension_reward,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.fastmcp import FastMCP


# --------------------------------------------------------------------------- #
# State: the typed, wire-synchronized per-rollout state                        #
# --------------------------------------------------------------------------- #
class NetHackState(vf.State):
    """The scalars that must cross the tool-server boundary.

    In 0.1.14 the whole v0 state dict *was* the rollout state, and cleanup had
    to strip every non-serializable game internal before the runtime's
    ``assert_serializable``. In 0.2.x the state is a declared pydantic model
    (`v1/state.py:14`) synchronized over an HTTP channel, so the engine handle,
    numpy observations, the Journal and the GameSpec simply never enter it —
    they stay on the toolset instance, inside the tool-server process. Only
    these fields are published, and they are serializable by construction.
    """

    # Cross-arm referee (see NetHackToolsetConfig.max_skill_calls).
    skill_calls: int = 0
    budget_exhausted: bool = False
    # Parallel-batch bookkeeping (see NetHackToolsetConfig.max_parallel_skill_calls).
    # `batch_started_at` is a monotonic timestamp; `batch_count` is how many calls
    # this batch has already executed. Published so the refusal is auditable in
    # the trace rather than being invisible harness behaviour.
    batch_started_at: float = 0.0
    batch_count: int = 0
    parallel_refusals: int = 0
    # A CANARY for one specific gate-leak mode, NOT independent evidence that
    # the gate holds. `tool_functions` only wraps adapters the v0 env published,
    # and under skill_set="netplay" no `move` adapter exists — so the increment
    # site is unreachable while the config is right, and this stays 0 by
    # construction. What it catches is someone changing `skill_set` (or the v0
    # adapter list) so a `move` adapter IS published: the counter then goes
    # nonzero and every rollout of the run carries the evidence.
    #
    # The gate itself is evidenced elsewhere: `move` absent from the advertised
    # MCP tool list (asserted on the wire in the cross-route tests), and
    # `tools/encoding_eval/_verify_gate.py`, which asserts a withheld `move` is
    # rejected WITHOUT stepping the engine.
    moves_executed: int = 0
    # Engine-side termination (death / ascension / step cap).
    terminated: bool = False
    # The per-turn NDJSON this rollout is writing, as `<run_id>.ndjson` under
    # the toolset's `trace_dir`. Published because the FILE is written in the
    # tool-server process while the model's assistant messages only exist on the
    # `Trace` in the driver process -- `NetHackTask.finalize` needs both to join
    # them, and `run_id` embeds a pid + epoch it cannot otherwise guess. Empty
    # until the first tool call (the writer mints the id lazily) and empty
    # forever when tracing is off.
    trace_run_id: str = ""
    # Reward-relevant scalars, mirrored out of the v0 state after each call.
    scout_reward_total: float = 0.0
    descent_count: float = 0.0
    max_dlvl_reached: int = 1
    succeeded: bool = False
    ascended: bool = False
    died: bool = False


# --------------------------------------------------------------------------- #
# Configs                                                                      #
# --------------------------------------------------------------------------- #
class NetHackToolsetConfig(vf.ToolsetConfig):
    """Everything the tool server needs to rebuild the v0 env in its own process.

    A 0.2.x toolset is launched as ``python -m <module>`` with this config
    serialized into ``VF_CONFIG`` (`v1/mcp/launch.py:188`, `:208`), so it cannot
    close over a pre-built v0 env object — every knob has to be carried here.
    """

    # Which curriculum task to run (key in nethack.GAME_SPECS).
    task_spec: str = "full_nle"
    # Obs/skill-structure variant + structured-map detail (see v0 load_environment).
    variant: str = "B0"
    map_detail: str = "full"
    # "skill" (one tool per skill) or "code" (single sandboxed `code` tool).
    interface: str = "skill"
    # Pin the character for standard tiers, e.g. "Val-hum-neu-fem".
    character: Optional[str] = None
    # Per-rollout LM-turn cap handed to the v0 env (the framework's own cap is
    # EnvConfig.max_turns; see load_v1_environment).
    max_turns: int = 200
    # Optional per-turn NDJSON trace dir (one file per rollout).
    trace_dir: Optional[str] = None
    # Every tool executes the skill and RETURNS the rendered observation, so
    # MCP-driven CLI agents get self-contained calls. False is refused: the
    # harness-driven (env_response) arm is the legacy bridge, not this taskset.
    self_dispatch: bool = True
    # "push" = every tool result carries the observation; "on_demand" = terse
    # feedback only, map gated behind an explicit look() call.
    obs_mode: str = "push"
    # Hard per-rollout budget on executed skill calls. Enforced toolset-side so
    # it binds every harness, including CLI agents whose internal loop we do not
    # control. One call == one v0 LM turn. <= 0 disables the cap.
    max_skill_calls: int = 150
    # Cap on skills executed per ASSISTANT TURN. 0 = unlimited (a client may batch
    # as many tool calls as it likes); 1 = one skill per turn.
    #
    # Why this exists. The budget above counts CALLS, not decisions, and a client
    # that batches gets fewer decisions for the same budget. Measured on identical
    # GLM-5.2/B0/seed-2 runs: the 2026-07-27 cell emitted exactly 1.00 tool calls
    # per assistant turn on all five seeds and reached the down-stair at ~decision
    # 250; the 2026-07-30 cell batched (404 tool calls in 175 turns; 34 batches of
    # 6, five of 10, almost all `press_key 's'`), exhausted the 400-call budget
    # after 175 decisions, and never left dlvl 1. Same model, same seed, same
    # dungeon -- the budget silently meant different things.
    #
    # The v0 control arm has always had this property (extra parallel tool calls
    # past the first are dropped), so `1` is what makes a CLI arm comparable to
    # the control arm and to any pre-batching run.
    #
    # Batch detection is a quiescence window, not a turn id: the toolset sees
    # individual MCP calls and has no view of assistant-turn boundaries. Calls
    # arriving within `parallel_batch_window_s` of the previous one are treated as
    # the same batch. This is safe by a wide margin -- batch members arrive within
    # milliseconds (`_with_state` serializes them and an engine step is ~1ms) while
    # a genuine next turn costs an LLM round-trip (~5s median measured). The known
    # false negative: a slow skill (`explore_level` can run seconds) may push a
    # later batch member outside the window and let it through. That errs toward
    # the permissive, pre-existing behaviour rather than silently dropping a real
    # decision.
    max_parallel_skill_calls: int = 0
    parallel_batch_window_s: float = 0.5
    # Echo the per-call correlation id (`[call#N]`) into each tool result so
    # the model transcript (`traces.jsonl`) and the turn NDJSON can be joined
    # exactly (see `nethack_harness.helpers.CALL_ID_MARKER_FORMAT`). The id is
    # always assigned server-side and stamped on the trace record; this knob
    # only controls the result-payload echo, which costs a few tokens per call
    # -- turn off for a cell that must be token-matched against pre-barrier
    # runs. The published tool schemas are identical either way: the marker
    # rides the payload, never the function definitions the model sees.
    call_id_in_results: bool = True
    # Passed through to v0 load_environment (compaction knobs, refiner, game-setup
    # overrides such as tune/modify/level_blob/skill_set, etc.). Kept opaque so
    # the v1 layer never has to track the full v0 kwarg surface.
    env_args: dict = {}


class NetHackTaskConfig(vf.TaskConfig):
    """Task-facing knobs. Holds the toolset config explicitly (see
    :meth:`NetHackTask.server_config`) rather than relying on 0.2.x's
    type-matching resolution (`v1/task.py:190-216`), which raises on ambiguity."""

    toolset: NetHackToolsetConfig = NetHackToolsetConfig()
    # Seed the CLI-agent workspace (AGENTS.md/CLAUDE.md, wiki/, memory/) into the
    # harness runtime before the agent starts. This is the filesystem counterpart
    # of the affordances the control arm gets through prompt and tools, so the
    # arms are capability-matched (tools/cli_harness_eval/workspace.py).
    seed_workspace: bool = True


class NetHackTasksetConfig(vf.TasksetConfig):
    """Env-specific knobs for the v1 taskset.

    The flat fields below are the knob surface Task 9/10 TOML configs bind to
    (``--taskset.task_spec``, ``--taskset.max_skill_calls``, ...); ``load()``
    projects them onto the per-task :class:`NetHackToolsetConfig`.
    """

    id: vf.ID = "nethack_v1"

    # --- toolset knobs (projected onto NetHackToolsetConfig by load()) -------
    task_spec: str = "full_nle"
    variant: str = "B0"
    map_detail: str = "full"
    interface: str = "skill"
    character: Optional[str] = None
    max_turns: int = 200
    trace_dir: Optional[str] = None
    self_dispatch: bool = True
    obs_mode: str = "push"
    max_skill_calls: int = 150
    max_parallel_skill_calls: int = 0
    parallel_batch_window_s: float = 0.5
    call_id_in_results: bool = True
    env_args: dict = {}
    # Where the tool server runs (colocated = share the harness's runtime).
    colocated: bool = False
    # Seed the CLI-agent workspace into the harness runtime (see NetHackTaskConfig).
    seed_workspace: bool = True

    # --- dataset shape (load-time only) --------------------------------------
    n_examples: int = 8
    seed: int = 0
    explicit_seeds: Optional[list] = None

    # --- which harness `load_harness` / `load_v1_environment` build -----------
    harness_id: str = "nethack_v1"

    def toolset_config(self) -> NetHackToolsetConfig:
        return NetHackToolsetConfig(
            colocated=self.colocated,
            task_spec=self.task_spec,
            variant=self.variant,
            map_detail=self.map_detail,
            interface=self.interface,
            character=self.character,
            max_turns=self.max_turns,
            trace_dir=self.trace_dir,
            self_dispatch=self.self_dispatch,
            obs_mode=self.obs_mode,
            max_skill_calls=self.max_skill_calls,
            max_parallel_skill_calls=self.max_parallel_skill_calls,
            parallel_batch_window_s=self.parallel_batch_window_s,
            call_id_in_results=self.call_id_in_results,
            env_args=dict(self.env_args or {}),
        )


def _resolve_config(config: object | None) -> NetHackTasksetConfig:
    if isinstance(config, NetHackTasksetConfig):
        return config
    if config is None:
        return NetHackTasksetConfig()
    if isinstance(config, dict):
        return NetHackTasksetConfig.model_validate(config)
    return NetHackTasksetConfig.model_validate(config.model_dump())


# --------------------------------------------------------------------------- #
# v0 env construction (config-only; the engine itself is created per-rollout)  #
# --------------------------------------------------------------------------- #
def _build_v0_env(cfg: NetHackToolsetConfig | NetHackTasksetConfig, *, n_examples: int = 1):
    """Construct a fully-configured v0 ``NetHackVerifiersEnv``.

    We only use it for its game logic (``setup_state`` / ``_apply_tool_call``),
    its resolved prompt ``spec``, and its tool callables. The per-rollout engine
    lives in the v0 state dict, so a single config instance is shared across all
    rollouts of one server (and two instances with identical config are
    interchangeable — nothing rollout-specific lives on the instance).
    """
    return _load_v0_environment(
        n_examples=n_examples,
        seed=0,
        max_turns=cfg.max_turns,
        interface=cfg.interface,
        task_spec=cfg.task_spec,
        variant=cfg.variant,
        map_detail=cfg.map_detail,
        character=cfg.character,
        trace_dir=cfg.trace_dir,
        # Task 18 Step 2: this v1 toolset is only ever built with
        # self_dispatch=True (NetHackToolset.__init__ raises otherwise), so
        # this always gates the JOURNAL block + HINT ladder off for the
        # CLI-agent arms. The control arm calls `nethack.load_environment`
        # directly and never reaches this function, so it never sets this.
        self_dispatch=cfg.self_dispatch,
        call_id_in_results=getattr(cfg, "call_id_in_results", True),
        **dict(cfg.env_args or {}),
    )


def _spec_for(task_spec: str):
    return GAME_SPECS.get(task_spec, FULL_GAME_SPEC)


# --------------------------------------------------------------------------- #
# Toolset: owns the per-rollout engine lifecycle + the action surface          #
# --------------------------------------------------------------------------- #
class NetHackToolset(vf.Toolset[NetHackToolsetConfig, NetHackState]):
    """The MCP tool server the CLI-agent arms drive the game through.

    Lifecycle (0.2.x, `v1/mcp/server.py:254-283`): the server process boots,
    ``setup()`` runs, ``setup_task(task)`` runs with the row fetched over the
    ``/task`` channel, then ``_register(mcp)`` publishes the tools and uvicorn
    serves. Teardown is the process exiting; the engine is closed by a callback
    on ``ServerBase._exit_stack``, which wraps the whole serve.
    """

    # Advertised MCP server name — Claude Code sees these as `mcp__nethack__<tool>`.
    TOOL_PREFIX = "nethack"

    def __init__(self, config: NetHackToolsetConfig) -> None:
        super().__init__(config)
        if not config.self_dispatch:
            raise ValueError(
                "self_dispatch=False (the harness-driven env_response loop) has no v1 "
                "path in verifiers 0.2.x: a v1 toolset is an MCP server and the only "
                "way a tool call reaches the engine is the tool call itself. Run the "
                "harness-driven control arm through the v0 legacy bridge instead "
                "(`vf-eval --id nethack --args '{...}'`), which executes the v0 rollout "
                "verbatim."
            )
        # The v0 env (game logic + tool adapters) and the v0 state dict holding
        # the live engine. Both stay inside this process; neither is serialized.
        self.v0env = None
        self.v0_state = None
        # Serializes the WHOLE pull-state / execute / push-state window; see
        # `_with_state` below for why the window and not just the engine step.
        self._call_lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------- #
    async def setup_task(self, task) -> None:
        """Create the long-lived engine + game state for this rollout (v0 logic)."""
        import verifiers as _vf0

        self.v0env = _build_v0_env(self.config)
        seed = int(getattr(task, "seed", 0))
        tier = getattr(task, "tier", None) or _spec_for(self.config.task_spec).name
        self.v0_state = _vf0.State(
            {
                "task": {"tier": tier, "seed": seed},
                "info": {"tier": tier, "seed": seed},
            }
        )
        await self.v0env.setup_state(self.v0_state)
        # Teardown: the exit stack is entered by `_serve` around the whole
        # server lifetime (`v1/mcp/server.py:259`), so this runs when the tool
        # server process shuts down at end of rollout.
        self._exit_stack.push_async_callback(self.close)

    async def close(self) -> None:
        """Release the engine and drop every non-serializable game internal."""
        state = self.v0_state
        if state is not None:
            env = state.get("env")
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        self.v0_state = None
        self.v0env = None

    # Task 18 Step 2: redundant scaffolding for a CLI agent that manages its
    # own reasoning and memory internally — the trace analyses found these
    # three never/rarely called (recall/pin_objective: 0 of 1,173 Claude Code
    # calls). Filtered ONLY here, i.e. only for this MCP-exposed toolset
    # (always self_dispatch=True); the shared `skill_set="netplay"` resolution
    # in `nethack.py`/`helpers.py` is untouched, so the control arm (which
    # gets its tools from the v0 env directly, never through this class)
    # keeps them.
    _SELF_DISPATCH_REDUNDANT_TOOLS = frozenset({"add_note", "recall", "pin_objective"})

    # -- the action surface ------------------------------------------------- #
    def tool_functions(self) -> dict[str, Callable]:
        """The executing tools, keyed by the name the model sees.

        The **netplay gate** is here: the names come from the v0 env's resolved
        adapter list, which already reflects ``skill_set`` — under
        ``skill_set="netplay"`` there is no ``move`` adapter, so ``move`` is
        never published. ``_apply_tool_call`` re-checks the same set
        (`nethack.py:754`) and rejects anything outside it *without stepping the
        engine*, so an agent that guesses a name gets a refusal, not a move.
        """
        if self.v0env is None:
            raise RuntimeError("setup_task() must run before tool_functions()")
        return {
            adapter.__name__: self._executing(adapter)
            for adapter in self.v0env.tools
            if adapter.__name__ not in self._SELF_DISPATCH_REDUNDANT_TOOLS
        }

    def _executing(self, adapter: Callable) -> Callable:
        """Bind a schema-only v0 adapter to the single execution path.

        The v0 adapters carry the JSON schema (name/signature/docstring) but do
        not touch the engine — in v0, dispatch lives in ``env_response``. CLI
        harnesses call tools over MCP and must get the result back from the call
        itself, so the adapter's identity is bound to ``_apply_tool_call``.

        The call budget is checked here, **before** any engine step: it is the
        cross-arm referee, because ``max_turns`` only binds harnesses we control
        while every arm passes through this toolset. A refused call does not
        increment ``skill_calls`` and never reaches the engine.
        ``max_skill_calls <= 0`` disables the cap.
        """
        name = adapter.__name__
        budget = self.config.max_skill_calls
        obs_mode = self.config.obs_mode
        max_parallel = self.config.max_parallel_skill_calls
        batch_window = self.config.parallel_batch_window_s

        @functools.wraps(adapter)
        async def _run(**kwargs):
            state = self.state
            # Parallel-batch cap. Refused calls consume NO budget and never reach
            # the engine, so the agent keeps the decision it would otherwise have
            # spent -- the point is to make one budget unit mean one decision, not
            # to punish batching. Checked before the budget so a refused batch
            # member cannot also trip the exhaustion terminal.
            if max_parallel > 0:
                now = time.monotonic()
                if now - state.batch_started_at <= batch_window:
                    state.batch_count += 1
                else:
                    state.batch_started_at = now
                    state.batch_count = 1
                if state.batch_count > max_parallel:
                    state.parallel_refusals += 1
                    self._publish(state)
                    return (
                        f"[Only {max_parallel} skill call per turn is executed. "
                        "This call was dropped -- issue one skill, read the result, "
                        "then decide the next one.]"
                    )
            if budget > 0 and state.skill_calls >= budget:
                state.budget_exhausted = True
                state.terminated = True
                return (
                    # The only place the call budget's SIZE ever reached a
                    # model. Post-hoc, so it cannot shape play -- but an
                    # orchestrator reading traces saw "200 skill calls used" and
                    # wrote budget advice into the continual store as a learned
                    # lesson. The episode ending is the fact; its arithmetic is
                    # ours.
                    "[The episode is over.]"
                )
            state.skill_calls += 1
            if name == "move":
                # Unreachable while `skill_set="netplay"` withholds the `move`
                # adapter — that is the point. See NetHackState.moves_executed:
                # this is a canary for a config change that republishes `move`,
                # not proof that the gate currently holds.
                state.moves_executed += 1
            content = await self.v0env._apply_tool_call(self.v0_state, name, kwargs)
            self._publish(state)
            if obs_mode == "on_demand" and name != "look":
                return _terse(content)
            return content

        # `functools.wraps` sets `__wrapped__`, so `inspect.signature(_run)`
        # resolves to the adapter's own parameters — which is exactly what
        # `ServerBase._with_state` re-advertises to FastMCP
        # (`v1/mcp/server.py:215`). No `state` parameter is (or may be) present:
        # state arrives via the `_call_state` contextvar and is read as
        # `self.state` (`server.py:132-135`), so adding one would leak a
        # framework argument into the model-visible tool schema.
        return _run

    def _with_state(self, fn: Callable) -> Callable:
        """Serialize every tool call on this rollout's server.

        Two independent reasons, both fatal if left unhandled, and both invisible
        in a rollout that happens to run serially:

        1. **The referee's count would be wrong.** ``ServerBase._with_state``
           (`v1/mcp/server.py:190-216`) is documented last-write-wins: it GETs
           the state, runs the tool, then PUTs the whole state back. Two
           concurrent calls both read ``skill_calls = N`` and both write
           ``N + 1``, so a client that batches tool calls (Claude Code emits
           several ``tool_use`` blocks in one assistant turn) can execute more
           skills than ``max_skill_calls`` allows. The budget is the cross-arm
           referee — an overrun is a silent invalidation of the comparison.
        2. **The engine is a C extension.** Two ``_apply_tool_call`` bodies
           re-entering one ``NetHackCoreEnv`` concurrently is memory corruption,
           not merely a bad count.

        Locking inside the tool body would fix neither: the state GET happens in
        the base wrapper *before* the body runs and the PUT *after* it, so the
        read-modify-write race lives outside it. The lock therefore has to wrap
        the base wrapper — the whole GET/execute/PUT window.

        The lock is per toolset instance, and ``Task.tools`` (not
        ``Taskset.tools``) means one instance per rollout, so this serializes a
        single rollout's calls and never couples two rollouts.
        """
        inner = super()._with_state(fn)

        @functools.wraps(inner)
        async def serialized(*args, **kwargs):
            async with self._call_lock:
                return await inner(*args, **kwargs)

        # Re-assert the advertised signature. The base sets it to the tool's own
        # parameters so FastMCP publishes those (`server.py:215`); keep that
        # exact contract rather than relying on `functools.wraps` copying it.
        serialized.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return serialized

    def _publish(self, state: NetHackState) -> None:
        """Mirror the reward/stop-relevant v0 scalars onto the typed state."""
        v0 = self.v0_state or {}
        state.scout_reward_total = float(v0.get("scout_reward_total", 0.0) or 0.0)
        state.descent_count = float(v0.get("descent_count", 0.0) or 0.0)
        state.max_dlvl_reached = int(v0.get("max_dlvl_reached", 1) or 1)
        state.succeeded = bool(v0.get("succeeded"))
        state.ascended = bool(v0.get("ascended"))
        state.died = bool(v0.get("died"))
        state.terminated = bool(v0.get("terminated")) or state.terminated
        state.trace_run_id = str(v0.get("_trace_run_id") or "")

    def _register(self, mcp: FastMCP) -> None:
        """Publish the gated tool set over MCP.

        Overrides the base ``@tool``-decorator scan (`v1/mcp/toolset.py:28-34`)
        because NetHack's action surface is built at runtime from the v0 skill
        registry and the configured ``skill_set``, not declared statically.
        """
        for name, fn in self.tool_functions().items():
            mcp.add_tool(
                self._with_state(fn),
                name=name,
                description=(fn.__doc__ or "").strip() or None,
            )


def _terse(content) -> str:
    """Return only the game messages plus the trailing feedback line.

    `on_demand` withholds the map until the agent asks for it. Used by the
    visibility sub-experiment (spec §5); `push` is the default and never
    calls this.
    """
    from nethack_harness.prompt.content import content_to_text

    out, in_messages = [], False
    for line in content_to_text(content).splitlines():
        if line.startswith("==="):
            in_messages = line.startswith("=== MESSAGES ===")
            continue
        if in_messages and line.strip():
            out.append(line)
        elif line.startswith("["):          # feedback from the skill call
            out.append(line)
    return "\n".join(out) or "(no message)"


# --------------------------------------------------------------------------- #
# Task: the row, the rewards, the stop conditions                              #
# --------------------------------------------------------------------------- #
class NetHackTaskData(vf.TaskData):
    """One curriculum row. ``seed`` and ``tier`` reach the tool server over the
    ``/task`` channel (`v1/mcp/server.py:171-188`), which is how the engine gets
    seeded per rollout."""

    seed: int = 0
    tier: str = "full_nle"


#: Returned when the reasoning backfill could not run at all (tracing off, the
#: NDJSON is not reachable from this process, ...). Zeroes, not absence: the
#: metric being present and zero is itself the finding.
_NO_REASONING = {"recovered": 0, "inline": 0, "unavailable": 0}


class NetHackTask(vf.Task[NetHackTaskData, NetHackState, NetHackTaskConfig]):
    """Per-rollout behavior: one tool server per rollout, the four rewards, and
    the stop conditions."""

    # Per-rollout (NOT Taskset.tools, which would share one engine across a
    # worker's rollouts) — this is the per-rollout engine lifecycle guarantee.
    tools = (NetHackToolset,)

    def server_config(self, server_cls: type):
        # Explicit pairing instead of 0.2.x's type-matching resolution, which
        # raises when several config fields match (`v1/task.py:190-216`).
        return self.config.toolset

    # -- per-rollout runtime setup ------------------------------------------ #
    async def setup(self, trace, runtime) -> None:
        """Seed the CLI-agent workspace into the harness runtime's workdir.

        `Rollout.run` calls this after `runtime.start()` and before
        `Harness.setup` / the tool servers (`v1/rollout.py:143-158`), so the
        files are in place by the time the agent's first turn runs. Uploading
        through ``runtime.write`` (rather than writing to a host path) keeps
        this correct for docker/prime runtimes too, where the workdir is not
        on the host filesystem.

        A CLI arm without its workspace is not capability-matched to the
        control arm, so a failure here is fatal rather than a warning.
        """
        if not self.config.seed_workspace:
            return
        import tempfile
        from pathlib import Path

        from tools.cli_harness_eval.workspace import build_workspace

        spec = _spec_for(self.data.tier)
        objective = (
            f"{spec.description}\n\nSuccess: {spec.success_criterion}\n\n"
            "Drive the game only through the `nethack` MCP tools."
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = build_workspace(
                Path(tmp) / "workspace",
                objective=objective,
                # The taskset already resolved the prompt against the published
                # skill set; reusing it keeps AGENTS.md from advertising tools
                # the MCP server does not serve (`tools/cli_harness_eval/
                # workspace.py`, `tests/test_prompt_tool_surface.py`).
                system_prompt=self.data.system_prompt or None,
            )
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    await runtime.write(
                        str(path.relative_to(root)), path.read_bytes()
                    )
        # `runtime.write` does not carry file modes, and the wiki being
        # read-only is load-bearing: an arm must not be able to corrupt its own
        # reference corpus mid-run and diverge from the other arms
        # (`tests/test_cli_workspace.py::test_wiki_pages_are_read_only`).
        await runtime.run(
            ["sh", "-c", "chmod 0444 wiki/*.md && chmod 0555 wiki"], {}
        )

    async def finalize(self, trace, runtime) -> None:
        """Publish the referee's bookkeeping onto the trace as unweighted metrics.

        `Trace.state` is `exclude=True` (`v1/trace.py:374`), so nothing on it
        reaches `traces.jsonl`. Without this the executed-call count — the
        cross-arm referee's own reading, and the quantity an arm comparison is
        normalized on — would be invisible in the run output, observable only
        indirectly through `stop_condition`. `move` is never published, so
        ``moves_executed`` rides along as a gate-leak canary — see
        :class:`NetHackState` for what it does and does not evidence.
        """
        state = trace.state
        reasoning = self._backfill_turn_reasoning(trace)
        trace.metrics.update(
            {
                "skill_calls": float(state.skill_calls),
                "parallel_refusals": float(state.parallel_refusals),
                "budget_exhausted": float(state.budget_exhausted),
                "max_dlvl_reached": float(state.max_dlvl_reached),
                "descent_count": float(state.descent_count),
                "scout_reward_total": float(state.scout_reward_total),
                "died": float(state.died),
                "terminated": float(state.terminated),
                "moves_executed": float(state.moves_executed),
            }
        )
        # Published so a run's own output says how much of the agent's reasoning
        # it captured, instead of that being discoverable only by reading the
        # NDJSON by hand.
        trace.metrics.update({f"reasoning_{k}": float(v) for k, v in reasoning.items()})

    def _backfill_turn_reasoning(self, trace) -> dict:
        """Write the model's own words for each turn into this rollout's NDJSON.

        WHY THIS RUNS HERE AND NOT IN THE WRITER. The trace records are written
        by the TOOL SERVER, one per `_apply_tool_call`; the model's messages
        live on the `Trace` in the DRIVER process, because a CLI agent talks to
        the interception endpoint, not to us. There is no moment inside a tool
        call at which both are in hand. `finalize` is the first moment there is:
        every call has been served (so every record exists) and the trace holds
        every sampled assistant message.

        Best-effort by construction. A failure here must never fail a scored
        rollout, and a run whose NDJSON is on another host (a container runtime)
        simply finds no file -- `tools/trace_reasoning.py` then does the same
        join offline over the collected artifacts.

        Returns `{"recovered", "inline", "unavailable"}` counts. One Task
        instance is shared across a rollout group (`v1/task.py:224`), so this
        returns per-rollout numbers rather than stashing them on `self`.
        """
        try:
            from pathlib import Path

            from tools.trace_reasoning import (
                assistant_turns_from_trace,
                backfill_records,
                _write_ndjson_atomically,
            )
            from tools.eval_metrics import read_ndjson

            trace_dir = self.config.toolset.trace_dir
            run_id = getattr(trace.state, "trace_run_id", "")
            if not trace_dir or not run_id:
                return _NO_REASONING
            path = Path(trace_dir) / f"{run_id}.ndjson"
            if not path.exists():
                return _NO_REASONING
            records = read_ndjson(path)
            stats = backfill_records(records, assistant_turns_from_trace(trace))
            _write_ndjson_atomically(path, records)
            return {
                "recovered": stats["recovered"],
                "inline": stats["already"],
                "unavailable": stats["unavailable"],
            }
        except Exception:  # pragma: no cover - diagnostics must not fail a rollout
            return _NO_REASONING

    # -- stop conditions ---------------------------------------------------- #
    @vf.stop
    async def call_budget_exhausted(self, trace) -> bool:
        """Terminal reason ``"call_budget_exhausted"``: `v1/session.py:115`
        records the firing method's ``__name__`` as the trace's stop condition."""
        return bool(trace.state.budget_exhausted)

    @vf.stop
    async def game_over(self, trace) -> bool:
        return bool(trace.state.terminated)

    # -- rewards (v0 implementations, verbatim) ----------------------------- #
    @vf.reward(weight=1.0)
    async def scout_reward(self, trace) -> float:
        return await _v0_scout_reward(
            {"scout_reward_total": trace.state.scout_reward_total}
        )

    @vf.reward(weight=10.0)
    async def descent_reward(self, trace) -> float:
        return await _v0_descent_reward({"descent_count": trace.state.descent_count})

    @vf.reward(weight=100.0)
    async def success_reward(self, trace) -> float:
        return await _v0_success_reward({"succeeded": trace.state.succeeded})

    @vf.reward(weight=1000.0)
    async def ascension_reward(self, trace) -> float:
        return await _v0_ascension_reward({"ascended": trace.state.ascended})


# The four rewards, in scoring order, with their v0 weights. Kept for
# introspection (`[f.__name__ for f in REWARDS]`) now that 0.2.x attaches
# rewards as Task methods rather than a `Taskset(rewards=[...])` list.
REWARDS = (
    NetHackTask.scout_reward,
    NetHackTask.descent_reward,
    NetHackTask.success_reward,
    NetHackTask.ascension_reward,
)


# --------------------------------------------------------------------------- #
# Taskset                                                                      #
# --------------------------------------------------------------------------- #
class NetHackTaskset(vf.Taskset[NetHackTask, NetHackTasksetConfig]):
    def load(self) -> Iterable[NetHackTask]:
        cfg = self.config
        spec = _spec_for(cfg.task_spec)
        # The system prompt is resolved by the v0 env's prompt recipe; build one
        # config-only env (no engine — that happens per rollout in the toolset).
        v0env = _build_v0_env(cfg.toolset_config())
        system_prompt = v0env.spec.system_prompt
        begin = f"Task: {spec.description}\nSuccess: {spec.success_criterion}\n\nBegin."

        task_config = NetHackTaskConfig(
            toolset=cfg.toolset_config(), seed_workspace=cfg.seed_workspace
        )
        rng = random.Random(cfg.seed)
        env_args = dict(cfg.env_args or {})
        # Seed pinning via ENV_ARGS. `env_args` is the opaque per-cell override
        # channel (launch_cell.sh's ENV_ARGS), and an operator who writes
        # `ENV_ARGS='{"explicit_seeds":[4], "resume_from":...}'` means "run
        # THESE seeds". Previously the v1 rows were seeded ONLY from the
        # taskset-level `explicit_seeds` field (the TOML's pinned [0..15]), so
        # the ENV_ARGS pin reached the v0 kwargs and did nothing to row
        # selection: `--num_tasks 1` ran seed 0 regardless. With resume_from
        # that mismatch surfaced as the tool server dying in setup_task
        # ("resume_from: no turn file for seed 0") behind a 180s opaque
        # ToolsetError, because the server's stderr lives in a runtime workdir
        # that teardown deletes. env_args wins over the taskset field: the TOML
        # field is the sweep default, ENV_ARGS the per-cell override. The
        # string form ("[4]") is how dotted CLI overrides deliver it.
        env_seed_pin = env_args.get("explicit_seeds")
        if env_seed_pin is not None:
            if isinstance(env_seed_pin, str):
                try:
                    env_seed_pin = json.loads(env_seed_pin)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        "env_args.explicit_seeds is not valid JSON: "
                        f"{env_seed_pin!r}"
                    ) from e
            if not isinstance(env_seed_pin, (list, tuple)):
                raise ValueError(
                    "env_args.explicit_seeds must be a list of seeds, got "
                    f"{type(env_seed_pin).__name__}: {env_seed_pin!r}"
                )
            seeds = [int(s) for s in env_seed_pin]
        elif cfg.explicit_seeds is not None:
            seeds = [int(s) for s in cfg.explicit_seeds]
        else:
            seeds = [rng.randint(0, 2**31 - 1) for _ in range(cfg.n_examples)]
        # Resume fail-fast, in the DRIVER. The tool server verifies the replay
        # anyway (nethack.py setup_state), but a server-side crash costs the
        # 180s port-file timeout per retry and its traceback dies with the
        # runtime workdir. A row whose seed has no source turn file can never
        # resume, so refuse to build the taskset at all -- instantly and in a
        # process whose stderr the operator actually sees.
        resume_from = env_args.get("resume_from")
        if resume_from:
            import glob as _glob
            import os as _os

            missing = [
                s
                for s in seeds
                if not (
                    _glob.glob(
                        _os.path.join(str(resume_from), "turns", f"{s}_*.ndjson")
                    )
                    or _glob.glob(_os.path.join(str(resume_from), f"{s}_*.ndjson"))
                )
            ]
            if missing:
                raise RuntimeError(
                    f"resume_from: no turn file under {resume_from} for "
                    f"seed(s) {missing} -- these rollouts could never resume. "
                    "Pin exactly the seeds that have recorded traces "
                    "(ENV_ARGS '{\"explicit_seeds\": [...]}' or "
                    "--taskset.explicit_seeds)."
                )
        for i, seed_val in enumerate(seeds):
            yield NetHackTask(
                NetHackTaskData(
                    idx=i,
                    name=f"{spec.name}:{seed_val}",
                    description=spec.description,
                    prompt=begin,
                    system_prompt=system_prompt,
                    seed=int(seed_val),
                    tier=spec.name,
                ),
                task_config,
            )


# --------------------------------------------------------------------------- #
# Harness: the taskset's default (an MCP chat loop, not a shell agent)         #
# --------------------------------------------------------------------------- #
class NetHackHarness(NullHarness):
    """Default harness for ``--taskset.id nethack_v1``.

    ``default_harness_id`` (`v1/loaders.py:109-117`) makes a taskset that
    exports a ``Harness`` subclass its own default; without one the default is
    ``bash``, which would give the agent shell access to the host running the
    engine — a real experiment-validity hazard, not just a wrong default. This
    is the built-in ``null`` program: model + MCP tools, nothing else, which is
    exactly the reference arm for the MCP toolset.

    This is NOT the 0.1.14 ``NetHackHarness``. That class reproduced the v0
    ``MultiTurnEnv`` loop in-process (``base_program`` + ``env_response`` +
    per-turn compaction); 0.2.x harnesses launch an external program
    (`v1/harness.py:136-153`), so the v0 loop now runs through the legacy
    bridge instead (see the module docstring).
    """


# --------------------------------------------------------------------------- #
# Golden v1 entrypoints                                                        #
# --------------------------------------------------------------------------- #
def load_taskset(config: NetHackTasksetConfig | dict | None = None) -> NetHackTaskset:
    return NetHackTaskset(_resolve_config(config))


def load_harness(config: NetHackTasksetConfig | dict | None = None) -> vf.Harness:
    cfg = _resolve_config(config)
    harness_config = vf.harness_config_type(cfg.harness_id).model_validate(
        {"id": cfg.harness_id}
    )
    return _vf_load_harness(harness_config)


def load_v1_environment(
    config: NetHackTasksetConfig | dict | None = None,
) -> vf.Environment:
    cfg = _resolve_config(config)
    return vf.Environment(
        vf.EnvConfig(
            taskset=cfg,
            harness=vf.harness_config_type(cfg.harness_id).model_validate(
                {"id": cfg.harness_id}
            ),
            # `max_turns` is framework-level in 0.2.x (`v1/env.py:97-100`): the
            # interception server refuses turns past it, so it binds every
            # harness rather than only the one we wrote.
            max_turns=cfg.max_turns,
        )
    )


# `__all__` is the 0.2.x plugin contract (`v1/loaders.py:58-82`): it must export
# exactly one Taskset subclass and exactly one Harness subclass.
__all__ = [
    "NetHackTasksetConfig",
    "NetHackTaskConfig",
    "NetHackToolsetConfig",
    "NetHackState",
    "NetHackTaskData",
    "NetHackToolset",
    "NetHackTask",
    "NetHackTaskset",
    "NetHackHarness",
    "load_taskset",
    "load_harness",
    "load_v1_environment",
    "REWARDS",
]


if __name__ == "__main__":  # pragma: no cover - tool-server entrypoint
    # `serve_in_runtime` launches a toolset as `python -m <module>`
    # (`v1/mcp/launch.py:208`), reading its config from VF_CONFIG.
    NetHackToolset.run()
