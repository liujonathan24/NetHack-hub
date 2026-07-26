"""NetPlay's real skill layer, bound to our engine and exposed as `netplay_true`.

This registers the *vendored* NetPlay skills (see environments/nethack/vendor/,
upstream github.com/CommanderCero/NetPlay @ 6acb90d) into our skill registry,
alongside -- not replacing -- the existing hand-written `netplay` skill set.

The action surface reproduced here is upstream `netplay/__init__.py:9-18`:

    SkillRepository([
        *skills.ALL_COMMAND_SKILLS,
        skills.set_avoid_monster_flag,
        skills.melee_attack,
        skills.explore_level,
        skills.move_to,
        skills.go_to,
        skills.press_key,
        skills.type_text,
    ])

Everything in this file is ours: it is the engine seam (NLE -> nethack_core) and
the adapter between NetPlay's generator-of-Steps skill protocol and our
SkillResult protocol. No NetPlay behaviour is reimplemented here.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Optional

import numpy as np

# The vendored tree is a sys.path *root* so that upstream's own absolute imports
# (netplay.*, nle.*, nle_language_wrapper, gradio) resolve without editing a
# single vendored line. Appended, never prepended, so a real `nle`/`gradio`
# install would still win if one were ever added to the venv.
_VENDOR = pathlib.Path(__file__).resolve().parents[2] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.append(str(_VENDOR))

from netplay.core.skill import Skill  # noqa: E402
from netplay.core.skill_repository import SkillRepository  # noqa: E402
import netplay.nethack_agent.skills as netplay_skills  # noqa: E402
from netplay.nethack_agent.agent import NetHackAgent  # noqa: E402

from nethack_harness.tools.skills import SkillResult, registry  # noqa: E402


# Upstream netplay/__init__.py:9-18, verbatim in content and order.
NETPLAY_SKILL_REPOSITORY = SkillRepository([
    *netplay_skills.ALL_COMMAND_SKILLS,
    netplay_skills.set_avoid_monster_flag,
    netplay_skills.melee_attack,
    netplay_skills.explore_level,
    netplay_skills.move_to,
    netplay_skills.go_to,
    netplay_skills.press_key,
    netplay_skills.type_text,
])

#: Tool names our harness exposes for skill_set="netplay_true".
NETPLAY_TRUE_SKILL_NAMES = tuple(NETPLAY_SKILL_REPOSITORY.skills.keys())

# Our registry namespaces these so they cannot collide with the existing
# hand-written skills of the same name (`move_to`, `eat`, `pray`, ...).
TOOL_PREFIX = "np_"


# ---------------------------------------------------------------------------
# Observation adapter: CoreObservation (dataclass) -> NLE-style mapping.
# ---------------------------------------------------------------------------

_OBS_FIELDS = (
    "glyphs", "chars", "colors", "blstats", "message",
    "inv_strs", "inv_letters", "inv_glyphs", "tty_chars", "tty_colors",
    "tty_cursor", "misc",
)


class _ObsMapping(dict):
    """NLE observations are dicts; ours is a dataclass. Bridge them.

    NetPlay reads observation["glyphs"|"chars"|"blstats"|"message"|"tty_chars"|
    "inv_strs"|"inv_letters"]. We expose every CoreObservation field, and keep
    the raw dataclass on `.raw` so the harness can hand it back unchanged.
    """

    def __init__(self, core_obs):
        super().__init__(
            {f: getattr(core_obs, f) for f in _OBS_FIELDS if hasattr(core_obs, f)}
        )
        self.raw = core_obs


class NetPlayEngineEnv:
    """The gym-like surface `NethackBaseAgent` expects, over `NetHackCoreEnv`.

    Two deliberate deviations from a real gym env, both because our harness --
    not the NetPlay agent -- owns the episode:

    * `reset()` does NOT reset the engine. It returns the current observation.
      `NethackBaseAgent.init()` calls `reset()` to seed its level tracker, and
      it is also called after a terminal step; resetting for real would destroy
      the rollout the harness is running.
    * rewards/termination are accumulated here so the adapter can report them
      back through SkillResult's pre_executed fields.
    """

    def __init__(self, core_env):
        self._env = core_env
        self.reward_acc = 0.0
        self.terminated = False
        self.truncated = False
        self.last_raw = None
        self.steps = 0

    # -- gym-ish surface ---------------------------------------------------
    @property
    def unwrapped(self):
        return self

    def _current_core_obs(self):
        # Prefer the env's own unfiltered observation. NetHackCoreEnv.last_observation
        # is a *view* filtered to observation_keys, and "misc" -- the (in_yn_function,
        # in_getlin, xwaitforspace) popup flags our waiting_for_* properties below read
        # -- is NOT in the default key set (nethack_core/env.py:123-129). Rebuilding a
        # CoreObservation from that filtered view (the old code path) silently stamped
        # misc=None into every fresh reset() (called on every get_agent(), see
        # get_agent() below), which made waiting_for_yn/line/space read False even
        # with a prompt open -- so `fail_on_popup`'s entry-time guard never fired, and
        # a skill's first keystroke landed in the open prompt instead. The engine
        # always computes `misc` regardless of observation_keys
        # (nethack_core/_engine.py:1046); only the harness-level filtering drops it,
        # so the unfiltered attribute carries it through unconditionally.
        raw = getattr(self._env, "_last_observation", None)
        if raw is not None:
            return raw
        keys = self._env.observation_keys
        last = self._env.last_observation
        if last is None:
            raise RuntimeError("NetPlay skills require an env that has been reset.")
        if not isinstance(last, (list, tuple)):
            return last  # already a CoreObservation
        # NetHackCoreEnv.last_observation is a list ordered by observation_keys.
        from nethack_core.env import CoreObservation
        return CoreObservation(**{k: last[i] for i, k in enumerate(keys)})

    def reset(self, *args, **kwargs):
        core = self._current_core_obs()
        self.last_raw = core
        return _ObsMapping(core), {}

    def step(self, action):
        core, reward, terminated, truncated, info = self._env.step(int(action))
        self.reward_acc += float(reward)
        self.terminated = self.terminated or bool(terminated)
        self.truncated = self.truncated or bool(truncated)
        self.last_raw = core
        self.steps += 1
        return _ObsMapping(core), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass

    # -- popup detection ---------------------------------------------------
    # NLE exposes these through its `internal` observation; our engine exposes
    # the same three flags as `misc` = (in_yn_function, in_getlin, xwaitforspace).
    def _misc(self, idx: int) -> bool:
        raw = self.last_raw if self.last_raw is not None else self._current_core_obs()
        misc = getattr(raw, "misc", None)
        if misc is None:
            return False
        try:
            return bool(np.asarray(misc).reshape(-1)[idx])
        except (IndexError, ValueError):
            return False

    @property
    def waiting_for_yn(self) -> bool:
        return self._misc(0)

    @property
    def waiting_for_line(self) -> bool:
        return self._misc(1)

    @property
    def waiting_for_space(self) -> bool:
        return self._misc(2)


# ---------------------------------------------------------------------------
# Per-env agent cache. NetPlay's agent carries persistent state across skill
# calls (the per-level `has_seen` / `search_count` maps that make explore_level
# terminate, plus the room graph), so it must survive between tool calls and be
# rebuilt only when the underlying episode changes.
# ---------------------------------------------------------------------------

_AGENTS: "dict[int, tuple[Any, NetPlayEngineEnv, NetHackAgent]]" = {}


def _episode_key(core_env) -> Any:
    try:
        return tuple(core_env.current_seeds)
    except Exception:
        return None


def get_agent(core_env) -> NetHackAgent:
    """Return the NetPlay agent bound to `core_env`, creating it if needed."""
    cache_id = id(core_env)
    entry = _AGENTS.get(cache_id)
    key = _episode_key(core_env)
    if entry is not None and entry[0] == key:
        agent = entry[2]
        # Re-sync: the harness may have stepped the env between skill calls
        # (other tools, --More-- dismissal). Refresh the tracker's view without
        # discarding the persistent per-level exploration memory.
        agent.last_observation = entry[1].reset()[0]
        return agent

    wrapped = NetPlayEngineEnv(core_env)
    agent = NetHackAgent(env=wrapped, log_folder="", render=False)
    agent.init()
    _AGENTS[cache_id] = (key, wrapped, agent)
    return agent


def reset_agent_cache() -> None:
    """Drop all cached agents (used by tests)."""
    _AGENTS.clear()


# ---------------------------------------------------------------------------
# Adapter: NetPlay's Iterator[Step] protocol -> our SkillResult protocol.
# ---------------------------------------------------------------------------

def run_netplay_skill(core_env, skill: Skill, kwargs: Dict[str, Any]) -> SkillResult:
    """Drive one NetPlay skill to completion against the live engine.

    Mirrors upstream's own execution pipeline (`agent.py::_solve_task`, lines
    165-168): `_execute_skill` -> `_skip_more_messages` -> `_update_objects`.
    All three are vendored verbatim, so the bounding behaviour (100 in-game
    turns per skill, interrupt on level change / teleport / new glyph / low HP)
    is NetPlay's, not ours.
    """
    agent = get_agent(core_env)
    wrapped: NetPlayEngineEnv = agent.env
    wrapped.reward_acc = 0.0
    wrapped.terminated = False
    wrapped.truncated = False
    wrapped.steps = 0

    # pre_executed skills report action_indices=[] to env_response (nethack.py),
    # so its normal step loop -- the thing that populates state["scout_tiles_seen"]
    # / state["_visited_tiles"] -- never runs for them. Record every observation
    # this skill's own internal loop produces so env_response can back-fill that
    # bookkeeping from `pre_visible_obs` instead. Scoped to this call only (the
    # instance patch is removed in `finally`); `wrapped.step` is `NetPlayEngineEnv.
    # step`, which `agent.step()` (agent_base.py:173) calls for every real engine
    # step a skill takes.
    step_observations: list = []
    original_step = wrapped.step

    def _tracking_step(action):
        result = original_step(action)
        step_observations.append(wrapped.last_raw)
        return result

    wrapped.step = _tracking_step
    thoughts: list[str] = []
    try:
        strategy = agent._execute_skill(skill, kwargs)
        strategy = agent._skip_more_messages(strategy)
        strategy = agent._update_objects(strategy)
        for step in strategy:
            if step.has_thoughts() and step.thoughts:
                thoughts.append(str(step.thoughts))
            if step.executed_action() and step.step_data.done:
                break
    except Exception as e:  # a skill raising must not kill the rollout
        thoughts.append(f"Skill raised {type(e).__name__}: {e}")
    finally:
        wrapped.step = original_step

    feedback = " ".join(t for t in thoughts if t).strip() or "No effect."
    return SkillResult(
        actions=[],
        feedback=feedback,
        pre_executed=True,
        pre_reward=wrapped.reward_acc,
        final_obs=wrapped.last_raw,
        pre_terminated=wrapped.terminated,
        pre_truncated=wrapped.truncated,
        pre_visible_obs=step_observations,
    )


_TYPE_NAMES = {"string": "string", "integer": "integer", "bool": "boolean"}


def _schema_for(skill: Skill) -> dict:
    params = {}
    for p in skill.parameters:
        entry = {
            "type": _TYPE_NAMES.get(p.type.name, "string"),
            "description": f"{p.name}" + (" (optional)" if p.optional else ""),
        }
        if p.optional:
            # `_make_skill_adapter` (helpers.py) treats a missing "default" as
            # `inspect.Parameter.empty` -- i.e. required. Upstream's own optional
            # params (SkillParameter(..., optional=True)) all default to None in
            # the skill function itself (create_position_command's x/y,
            # create_inventory_command's item_letter, rest's count), and
            # `_make_adapter` below already drops a None arg before calling the
            # skill, so publishing "default": None here reproduces upstream's
            # real, callable-with-no-args surface (e.g. `down()` = "descend
            # where I stand") instead of silently making it required.
            entry["default"] = None
        params[p.name] = entry
    return {"description": skill.description, "parameters": params}


def _make_adapter(skill: Skill):
    param_names = [p.name for p in skill.parameters]

    def _fn(env, obs, **kwargs):
        call_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
        # NetPlay skills declare optional params via Python defaults; passing
        # None explicitly would override those defaults, so drop empty values.
        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}
        return run_netplay_skill(env, skill, call_kwargs)

    _fn.__name__ = f"netplay_{skill.name}"
    # The registry filters kwargs by signature, so advertise the real params.
    import inspect
    _fn.__signature__ = inspect.Signature(
        [
            inspect.Parameter("env", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("obs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
        + [
            inspect.Parameter(
                n, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            )
            for n in param_names
        ]
    )
    return _fn


def register_all() -> list[str]:
    """Register every NetPlay skill under the `np_` prefix. Idempotent."""
    names = []
    for skill in NETPLAY_SKILL_REPOSITORY.skills.values():
        tool_name = f"{TOOL_PREFIX}{skill.name}"
        registry.register(tool_name, schema=_schema_for(skill))(_make_adapter(skill))
        names.append(tool_name)
    return names


NETPLAY_TRUE_TOOL_NAMES = tuple(register_all())
