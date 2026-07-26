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

import functools
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
    # Engine-side termination (death / ascension / step cap).
    terminated: bool = False
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
    variant: str = "B1"
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
    # Passed through to v0 load_environment (compaction knobs, refiner, game-setup
    # overrides such as tune/modify/level_blob/skill_set, etc.). Kept opaque so
    # the v1 layer never has to track the full v0 kwarg surface.
    env_args: dict = {}


class NetHackTaskConfig(vf.TaskConfig):
    """Task-facing knobs. Holds the toolset config explicitly (see
    :meth:`NetHackTask.server_config`) rather than relying on 0.2.x's
    type-matching resolution (`v1/task.py:190-216`), which raises on ambiguity."""

    toolset: NetHackToolsetConfig = NetHackToolsetConfig()


class NetHackTasksetConfig(vf.TasksetConfig):
    """Env-specific knobs for the v1 taskset.

    The flat fields below are the knob surface Task 9/10 TOML configs bind to
    (``--taskset.task_spec``, ``--taskset.max_skill_calls``, ...); ``load()``
    projects them onto the per-task :class:`NetHackToolsetConfig`.
    """

    id: vf.ID = "nethack_v1"

    # --- toolset knobs (projected onto NetHackToolsetConfig by load()) -------
    task_spec: str = "full_nle"
    variant: str = "B1"
    map_detail: str = "full"
    interface: str = "skill"
    character: Optional[str] = None
    max_turns: int = 200
    trace_dir: Optional[str] = None
    self_dispatch: bool = True
    obs_mode: str = "push"
    max_skill_calls: int = 150
    env_args: dict = {}
    # Where the tool server runs (colocated = share the harness's runtime).
    colocated: bool = False

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
            adapter.__name__: self._executing(adapter) for adapter in self.v0env.tools
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

        @functools.wraps(adapter)
        async def _run(**kwargs):
            state = self.state
            if budget > 0 and state.skill_calls >= budget:
                state.budget_exhausted = True
                state.terminated = True
                return (
                    f"[Call budget exhausted: {budget} skill calls used. "
                    "The episode is over.]"
                )
            state.skill_calls += 1
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

        task_config = NetHackTaskConfig(toolset=cfg.toolset_config())
        rng = random.Random(cfg.seed)
        if cfg.explicit_seeds is not None:
            seeds = [int(s) for s in cfg.explicit_seeds]
        else:
            seeds = [rng.randint(0, 2**31 - 1) for _ in range(cfg.n_examples)]
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
