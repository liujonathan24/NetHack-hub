"""
nethack_v1
==========

Verifiers **v1 taskset** port of the NetHack environment.

The v0 environment (``nethack.py``, class ``NetHackVerifiersEnv`` — a
``vf.StatefulToolEnv``) is a stateful, multi-turn, tool-driven env in which each
rollout owns a long-lived NLE engine (``NetHackCoreEnv`` / the six-floor
``CurriculumPrimitivesEnv``). This module re-wraps that env onto the Verifiers v1
Taskset / Toolset / Harness API **without rewriting any game logic** — it reuses
the existing ``setup_state`` / ``env_response`` skill dispatch, the per-turn
history compaction, the prompt spec / encoders, the curriculum, and the reward
functions verbatim.

v0 -> v1 mapping
----------------
* **Toolset** (``_build_toolset``) owns the per-rollout engine lifecycle and the
  action surface:
    - ``setups=[_nethack_setup]``  -> calls the v0 ``setup_state`` (creates the
      long-lived ``NetHackCoreEnv``/``CurriculumPrimitivesEnv`` in ``state["env"]``,
      seeds it, drains the intro banner, builds the journal, etc.).
    - ``cleanups=[_nethack_cleanup]`` -> closes the engine and drops every
      non-JSON-serializable game-internal from ``state`` so the v1 runtime's
      end-of-rollout ``assert_serializable`` passes.
    - ``tools=v0env.tools`` -> the exact skill/code-mode adapters (same
      ``__name__`` / signature / docstring), so the v1 tool schemas sent to the
      model are identical to v0's. Dispatch itself is done by the v0
      ``env_response`` (each skill has a bespoke signature over the engine handle
      + structured observation), exactly as in v0.
* **Taskset** (``load_taskset``) yields the curriculum rows
  (``full_nle`` / ``six_floor_primitives`` x seeds), carries the spec
  ``system_prompt``, registers the BALROG-progression / descent / scout / success
  / ascension rewards, and attaches the Toolset.
* **Harness** (``NetHackHarness``) is a custom multi-turn harness. The v1 base
  ``Harness`` cannot express NetHack's loop because (a) the environment
  observation for turn *N* is produced by ``env_response`` from the model's turn
  *N-1* tool call rather than being the tool's return value, and (b) each prompt
  is rebuilt through the v0 per-turn history compaction (elide-all-but-last-K,
  belief-state reset, CH refiner injection, assistant-content sanitisation).
  ``NetHackHarness.base_program`` reproduces the v0 ``rollout`` loop faithfully by
  calling the v0 ``env_response`` + the v0 compaction helpers, while driving model
  requests through the v1 runtime.

The v0 ``load_environment`` / ``NetHackVerifiersEnv`` path is untouched and keeps
working; this is a dual-path add.
"""

from __future__ import annotations

import json as _json
import random
from typing import Any, Optional

import verifiers.v1 as vf
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.response_utils import parse_response_message

# Reuse the entire v0 env: construction, game logic, encoders, curriculum.
from nethack import (
    FULL_GAME_SPEC,
    GAME_SPECS,
    load_environment as _load_v0_environment,
)

# Reuse the v0 reward implementations and the pure per-turn compaction helpers
# verbatim (do NOT reimplement).
from nethack_harness.helpers import (
    scout_reward as _v0_scout_reward,
    descent_reward as _v0_descent_reward,
    success_reward as _v0_success_reward,
    ascension_reward as _v0_ascension_reward,
    _compact_chat_history,
    _drop_before_last_belief,
    _sanitize_assistant_content,
)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
class NetHackTasksetConfig(vf.TasksetConfig):
    """Env-specific knobs for the v1 taskset.

    Subclasses ``vf.TasksetConfig`` (which is ``extra="forbid"``) so these fields
    are declared and discoverable. The base Taskset only reads the framework
    fields; the extra fields below are consumed by ``load_taskset`` /
    ``load_harness`` before delegating to the base classes.
    """

    # Which curriculum task to run (key in nethack.GAME_SPECS).
    task_spec: str = "full_nle"
    # Obs/skill-structure variant + structured-map detail (see v0 load_environment).
    variant: str = "B1"
    map_detail: str = "full"
    # "skill" (one tool per skill) or "code" (single sandboxed `code` tool).
    interface: str = "skill"
    # Dataset shape.
    n_examples: int = 8
    seed: int = 0
    explicit_seeds: Optional[list] = None
    # Per-rollout LM-turn cap.
    max_turns: int = 200
    # Optional per-turn NDJSON trace dir (one file per rollout).
    trace_dir: Optional[str] = None
    # Passed through to v0 load_environment (compaction knobs, refiner, game-setup
    # overrides such as tune/modify/level_blob, etc.). Kept opaque so the v1 layer
    # never has to track the full v0 kwarg surface.
    env_args: dict = {}


def _resolve_config(config: object | None) -> NetHackTasksetConfig:
    if isinstance(config, NetHackTasksetConfig):
        return config
    return NetHackTasksetConfig.from_config(config)


# --------------------------------------------------------------------------- #
# v0 env construction (config-only; the engine itself is created per-rollout)  #
# --------------------------------------------------------------------------- #
def _build_v0_env(cfg: NetHackTasksetConfig, *, n_examples: int):
    """Construct a fully-configured v0 ``NetHackVerifiersEnv``.

    We only use it for its game logic (``setup_state`` / ``env_response``), its
    resolved prompt ``spec``, its tool callables, and its compaction knobs. The
    per-rollout engine lives in ``state``, so a single config instance is shared
    across all rollouts (and two instances with identical config are
    interchangeable — nothing rollout-specific lives on the instance).
    """
    return _load_v0_environment(
        n_examples=n_examples,
        seed=cfg.seed,
        max_turns=cfg.max_turns,
        interface=cfg.interface,
        task_spec=cfg.task_spec,
        variant=cfg.variant,
        map_detail=cfg.map_detail,
        trace_dir=cfg.trace_dir,
        explicit_seeds=cfg.explicit_seeds,
        **dict(cfg.env_args or {}),
    )


# --------------------------------------------------------------------------- #
# Source rows                                                                  #
# --------------------------------------------------------------------------- #
def _make_source(cfg: NetHackTasksetConfig):
    spec = GAME_SPECS.get(cfg.task_spec, FULL_GAME_SPEC)
    begin = f"Task: {spec.description}\nSuccess: {spec.success_criterion}\n\nBegin."

    def source():
        rng = random.Random(cfg.seed)
        if cfg.explicit_seeds is not None:
            seeds = [int(s) for s in cfg.explicit_seeds]
        else:
            seeds = [rng.randint(0, 2**31 - 1) for _ in range(cfg.n_examples)]
        for i, seed_val in enumerate(seeds):
            yield {
                # No system message here — that lives on Taskset.system_prompt.
                "prompt": [{"role": "user", "content": begin}],
                # Top-level seed so the v0 setup_state's state["task"].get("seed")
                # resolves; also mirrored into info for the fallback path.
                "seed": int(seed_val),
                "task": {"tier": spec.name, "seed": int(seed_val)},
                "info": {
                    "tier": spec.name,
                    "seed": int(seed_val),
                    "spec_description": spec.description,
                },
                "example_id": i,
                "max_turns": cfg.max_turns,
            }

    return source


# --------------------------------------------------------------------------- #
# Rewards (thin v1 (task, state) wrappers over the v0 (state) implementations) #
# --------------------------------------------------------------------------- #
@vf.reward(weight=1.0)
async def scout_reward(task, state) -> float:
    return await _v0_scout_reward(state)


@vf.reward(weight=10.0)
async def descent_reward(task, state) -> float:
    return await _v0_descent_reward(state)


@vf.reward(weight=100.0)
async def success_reward(task, state) -> float:
    return await _v0_success_reward(state)


@vf.reward(weight=1000.0)
async def ascension_reward(task, state) -> float:
    return await _v0_ascension_reward(state)


REWARDS = [scout_reward, descent_reward, success_reward, ascension_reward]


# --------------------------------------------------------------------------- #
# Toolset: owns the per-rollout engine lifecycle + the action surface          #
# --------------------------------------------------------------------------- #
# Keys the v0 setup_state / env_response stash in state that are NOT
# JSON-serializable (engine handle, numpy obs, Journal, sets, GameSpec, ...).
# Rewards read only serializable scalars (scout_reward_total, descent_count,
# ascended, succeeded, ...) which are computed during env_response, so these can
# all be dropped at cleanup before the runtime asserts serializability.
_FRAMEWORK_KEYS = frozenset(
    {
        "runtime",
        "trajectory",
        "completion",
        "prompt",
        "system_prompt",
        "metrics",
        "reward",
        "advantage",
        "timing",
        "artifacts",
        "info",
        "task",
        "example_id",
        "num_model_requests",
        "done",
        "trajectory_id",
    }
)


def _build_toolset(v0env) -> vf.Toolset:
    async def _nethack_setup(task, state) -> None:
        # Create the long-lived engine + game state for this rollout (v0 logic).
        await v0env.setup_state(state)

    async def _nethack_cleanup(task, state) -> None:
        # Free the engine.
        env = state.get("env")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        # Drop every non-serializable game-internal so the runtime's end-of-
        # rollout assert_serializable passes. Framework-managed keys are left
        # alone; INTERNAL_KEYS cannot be deleted (and are already serializable).
        for key in list(state.keys()):
            if key in _FRAMEWORK_KEYS or key in vf.State.INTERNAL_KEYS:
                continue
            try:
                _json.dumps(state[key])
            except (TypeError, ValueError):
                try:
                    state.pop(key, None)
                except Exception:
                    pass

    return vf.Toolset(
        tools=list(v0env.tools),
        setups=[_nethack_setup],
        cleanups=[_nethack_cleanup],
        scope="rollout",
    )


# --------------------------------------------------------------------------- #
# Custom multi-turn harness                                                    #
# --------------------------------------------------------------------------- #
class NetHackHarness(vf.Harness):
    """Reproduces the v0 ``MultiTurnEnv`` loop faithfully on the v1 runtime.

    Per turn:
      1. (turns > 1) call the v0 ``env_response`` on the running conversation —
         this applies the previous tool call to the engine and returns the new
         observation user-message (also mutating reward-relevant state).
      2. build the model prompt through the v0 per-turn compaction pipeline
         (``_compact_chat_history`` -> optional belief-state reset -> spec
         history transforms -> assistant-content sanitisation) — identical to the
         v0 ``get_prompt_messages``.
      3. submit the model request through the v1 runtime and append the assistant
         message to the conversation.
    Termination mirrors v0 ``is_completed`` (engine ``terminated`` or the
    per-rollout ``max_turns`` cap), without touching the framework-managed
    ``is_truncated`` key directly.
    """

    def __init__(self, v0env=None, *, max_turns: int = 200, config=None, **kwargs):
        # v0env defaults to None so the framework's teardown-handler discovery
        # (which reflects over ``harness.__class__`` and finds the inherited
        # ``teardown`` method) can construct a throwaway instance harmlessly; a
        # real harness always receives a configured v0env from ``load_harness``.
        self.v0env = v0env
        super().__init__(config=config, max_turns=max_turns, **kwargs)

    @vf.stop
    async def game_over(self, task, state) -> bool:
        return bool(state.get("terminated"))

    def _build_prompt(self, convo, state):
        v0env = self.v0env
        msgs = _compact_chat_history(
            list(convo),
            keep_full=v0env.history_keep_full,
            drop_after=v0env.history_drop_after,
        )
        if getattr(v0env, "summarize_and_reset", False):
            msgs = _drop_before_last_belief(msgs, state)
        for transform in v0env.spec.history_transforms:
            msgs = transform(v0env, msgs, state)
        msgs = _sanitize_assistant_content(msgs)
        return normalize_messages(msgs, field_name="nethack_prompt")

    async def base_program(self, task, state):
        # Runs the toolset setup -> v0 setup_state creates state["env"] etc.
        await self.runtime.setup_rollout(task, state)

        v0env = self.v0env
        system = normalize_messages(
            state.get("system_prompt", []), field_name="state.system_prompt"
        )
        prompt = normalize_messages(
            state.get("prompt", []), field_name="state.prompt"
        )
        convo = [*system, *prompt]

        max_turns = state.get_max_turns(self.config.max_turns)
        turn = 0
        first = True
        while max_turns <= 0 or turn < max_turns:
            if bool(state.get("terminated")):
                state._set_stop_condition("terminated")
                break
            if not first:
                env_resp = await v0env.env_response(convo, state)
                convo.extend(
                    normalize_messages(env_resp, field_name="env_response")
                )
                if bool(state.get("terminated")):
                    # v0 shows the terminal observation and lets the model take
                    # one final (ignored) action; we mirror that by continuing to
                    # the model request below, then stopping at the loop top.
                    pass
            first = False

            model_prompt = self._build_prompt(convo, state)
            response = await self.runtime.submit_model_request(
                model_prompt,
                task,
                state,
                tool_defs=self.runtime.tool_defs(state),
            )
            turn += 1
            convo.extend(await parse_response_message(response))

            if max_turns > 0 and turn >= max_turns:
                state._set_truncated(True)
                state._set_stop_condition("max_turns_reached", overwrite=True)
                break
        return state


# --------------------------------------------------------------------------- #
# Golden v1 entrypoints                                                        #
# --------------------------------------------------------------------------- #
def load_taskset(config: NetHackTasksetConfig | dict | None = None) -> vf.Taskset:
    cfg = _resolve_config(config)
    v0env = _build_v0_env(cfg, n_examples=1)
    spec = GAME_SPECS.get(cfg.task_spec, FULL_GAME_SPEC)
    return vf.Taskset(
        source=_make_source(cfg),
        system_prompt=v0env.spec.system_prompt,
        rewards=REWARDS,
        toolsets=[_build_toolset(v0env)],
        taskset_id=f"nethack:{spec.name}",
    )


def load_harness(config: NetHackTasksetConfig | dict | None = None) -> vf.Harness:
    cfg = _resolve_config(config)
    v0env = _build_v0_env(cfg, n_examples=1)
    return NetHackHarness(v0env, max_turns=cfg.max_turns)


def load_v1_environment(config: NetHackTasksetConfig | dict | None = None) -> vf.Env:
    cfg = _resolve_config(config)
    return vf.Env(taskset=load_taskset(cfg), harness=load_harness(cfg))


__all__ = [
    "NetHackTasksetConfig",
    "NetHackHarness",
    "load_taskset",
    "load_harness",
    "load_v1_environment",
    "scout_reward",
    "descent_reward",
    "success_reward",
    "ascension_reward",
    "REWARDS",
]
