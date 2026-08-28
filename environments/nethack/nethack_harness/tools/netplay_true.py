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

import logging
import os
import pathlib
import sys
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

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
from nethack_harness import tool_flags as _flags
from nethack_harness.helpers import (  # noqa: E402
    CARRIAGE_RETURN,
    _cr_would_be_unknown_command,
)

# ---------------------------------------------------------------------------
# Swallowed exceptions must not be silent.
#
# This module is full of `except Exception: return []`-shaped guards, and they
# are right to exist: a bug in a guard must never abort a skill that was
# working, mid-rollout, on a paid run. But "total" was implemented as
# "invisible", and that cost us the spoiling-food interrupt -- `_spoiling_now`
# raised `AttributeError: message` on EVERY call for months and reported the
# same `[]` a healthy game with no corpses reports (docs/HARNESS_DEFECTS.md
# 3.2). A defensive except that hides a programming error is not defensive.
#
# So: still swallow, but count and log. `swallowed_exceptions()` is readable
# from a sweep at the end of a batch; the first occurrence of each
# (site, exception type) also logs at WARNING with a traceback, and repeats
# drop to DEBUG so a per-step guard cannot flood the log.
#
# `NETHACK_STRICT_SKILL_ERRORS=1` re-raises instead of swallowing. Tests set it
# so a regression of this exact class fails loudly instead of degrading into a
# quiet no-op.
# ---------------------------------------------------------------------------

#: "<site>:<ExceptionType>" -> count, for the life of the process.
_SWALLOWED: dict[str, int] = {}


def _strict_errors() -> bool:
    return os.environ.get("NETHACK_STRICT_SKILL_ERRORS", "") in ("1", "true", "True")


def _swallowed(where: str, exc: BaseException) -> None:
    """Record (and optionally re-raise) an exception a guard is about to eat."""
    if _strict_errors():
        raise exc
    key = f"{where}:{type(exc).__name__}"
    seen = _SWALLOWED.get(key, 0)
    _SWALLOWED[key] = seen + 1
    if seen == 0:
        logger.warning("swallowed %s in %s: %s", type(exc).__name__, where, exc,
                       exc_info=True)
    else:
        logger.debug("swallowed %s in %s (x%d): %s",
                     type(exc).__name__, where, seen + 1, exc)


def swallowed_exceptions() -> dict:
    """Snapshot of `"<site>:<ExceptionType>" -> count`. Empty is the good case."""
    return dict(_SWALLOWED)


def reset_swallowed_exceptions() -> None:
    """Zero the swallowed-exception counters (per-batch sweeps, tests)."""
    _SWALLOWED.clear()


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

#: Consecutive executed actions a netplay skill may take without the game
#: clock advancing or the player moving before `run_netplay_skill` breaks the
#: macro. This is the ONLY bound on a skill that keeps re-issuing an action the
#: game refuses: upstream's loop counts in-game turns, and a refused action
#: costs none, so without this the loop is infinite (see the livelock interrupt
#: in `run_netplay_skill` for the measured case).
#:
#: Sized against what legitimately takes engine steps at a frozen clock: menu
#: paging, `--More--` acknowledgement, and typing an engraving are all tens of
#: keystrokes. 200 is far above every one of those and ~2 orders of magnitude
#: below the 18,000+ steps the measured livelock reached before its tool call
#: was abandoned.
NO_PROGRESS_STEP_LIMIT = 200


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
        # Same stray-CR swallow as the harness funnel (nethack.py env_response /
        # interface.py TypedNetHackInterface.step). Upstream's `press_key`/
        # `type_text` skills can pass a bare CR (RawKeyPress.KEYPRESS_ENTER = 13,
        # nethack_utils/nle_wrapper.py) straight to this method -- e.g.
        # `np_press_key(key="enter")` outside a prompt -- and every netplay_true
        # skill routes its engine steps through here (see `run_netplay_skill`'s
        # `_tracking_step`), so this is the one place that has to catch it. A CR
        # that reaches command context is only ever `Unknown command '^M'.`; a
        # CR a prompt is waiting for is untouched (checked live, same as the
        # other two funnels).
        if int(action) == CARRIAGE_RETURN and _cr_would_be_unknown_command(self.last_raw):
            core = self.last_raw
            return _ObsMapping(core), 0.0, bool(self.terminated), bool(self.truncated), {}
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


def _spoiling_now(core_env):
    """Carried corpses within the urgent window, or [] if none/unknown.

    Deliberately cheap and total: any failure means "nothing urgent", because a
    bug here must never be able to abort a skill that was working.

    THE BUG THIS USED TO HAVE (docs/HARNESS_DEFECTS.md 3.2). It passed a
    three-field `_RawView` (chars/glyphs/blstats) into `shape()`, which reads
    `.message` first -- so every call raised `AttributeError`, the bare
    `except` below turned that into `[]`, and the interrupt NEVER fired in any
    rollout. "Nothing urgent" and "this function is broken" were the same
    return value, which is why it took months to notice. Now the swallow is
    counted and logged (`_swallowed`), and `_raw_view` hands over a complete
    observation.

    Relationship to `prompt/corpse_age.py`: that module owns all the rot
    arithmetic and the kill log; it decides what is urgent
    (`urgent_corpses`, URGENT_WINDOW = 5 turns of edibility left) and renders
    the inventory annotation. This function only asks it, mid-macro, and does
    not re-derive any threshold of its own.
    """
    try:
        from nethack_core.observations import shape as _shape
        from nethack_harness.prompt.corpse_age import urgent_corpses
        raw = _raw_view(core_env)
        if raw is None:
            return []
        # Fast path. This runs after EVERY engine step of a macro (up to 100
        # per agent turn) and `shape()` costs ~220 us, so skip it outright when
        # the raw inventory bytes contain no corpse at all -- which is the vast
        # majority of steps. `urgent_corpses` reads nothing but the inventory,
        # so this cannot change the answer.
        inv = getattr(raw, "inv_strs", None)
        if isinstance(inv, np.ndarray) and inv.dtype == np.uint8:
            if b"corpse" not in inv.tobytes().lower():
                return []
        so = _shape(raw, getattr(core_env, "_character", None))
        gt = (so.status or {}).get("time")
        return urgent_corpses(so.inventory, core_env, gt)
    except Exception as exc:
        _swallowed("_spoiling_now", exc)
        return []


#: Every field of a `CoreObservation`, in the order `NetHackCoreEnv` declares
#: them. `shape()` reads message / inv_strs / inv_letters / inv_glyphs /
#: tty_chars / chars / glyphs / blstats -- a view carrying only the first three
#: of those is not a substitute for an observation, it is a trap.
_VIEW_FIELDS = (
    "tty_chars", "tty_colors", "tty_cursor", "glyphs", "chars", "colors",
    "message", "inv_strs", "inv_letters", "inv_glyphs", "blstats", "misc",
)


class _RawView:
    """Adapter over `core_env.last_observation`, which is a LIST.

    `NetHackCoreEnv.last_observation` returns the raw observation TUPLE indexed
    by `observation_keys`, not a CoreObservation. Anything expecting `.chars` /
    `.glyphs` / `.blstats` silently sees `None` and returns empty -- which is
    how the pet guard came to no-op while reporting success, letting
    `melee_attack` kill the pet it was written to protect.

    It now carries EVERY key the env publishes, not just those three: the
    three-field version made `shape(view)` raise `AttributeError: message`,
    which is defect 3.2 (the inert spoiling-food interrupt). Fields the env
    does not publish are simply absent, so `getattr(view, f, None)` still works
    and anything that genuinely requires them still fails loudly.
    """

    __slots__ = _VIEW_FIELDS

    def __init__(self, chars=None, glyphs=None, blstats=None, **fields):
        self.chars, self.glyphs, self.blstats = chars, glyphs, blstats
        for name, value in fields.items():
            setattr(self, name, value)

    def __getattr__(self, name):  # only called for unset __slots__ entries
        raise AttributeError(
            f"{type(self).__name__} has no {name!r}: the env did not publish it "
            f"(observation_keys). Do not paper over this with a default -- "
            f"whatever needs it needs a real observation.")


def _raw_view(core_env):
    """Best-effort CoreObservation-shaped view; None if unavailable.

    Prefers the env's own `_last_observation`, which IS a `CoreObservation`
    (`nethack_core/env.py:140`) and therefore complete. The list form is only
    the public `last_observation` property re-projecting that same object
    through `observation_keys`, so rebuilding a partial view from it threw away
    fields for no reason.
    """
    obs = getattr(core_env, "_last_observation", None)
    if obs is not None and not isinstance(obs, (list, tuple)):
        return obs
    raw = getattr(core_env, "last_observation", None)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return raw                      # already a CoreObservation
    try:
        ks = list(core_env.observation_keys)
        fields = {k: raw[i] for i, k in enumerate(ks)
                  if k in _VIEW_FIELDS and i < len(raw)}
        if not {"chars", "glyphs", "blstats"} <= set(fields):
            raise KeyError(f"observation_keys lacks chars/glyphs/blstats: {ks}")
        return _RawView(**fields)
    except Exception as exc:
        _swallowed("_raw_view", exc)
        return None


#: Non-hostiles that `melee_attack` must never be pointed at. Pets are detected
#: from the glyph; the rest are recognised by name, because NLE exposes no
#: peaceful flag and NetHack only reveals peacefulness through the "Really
#: attack?" prompt -- which is exactly the prompt `melee_attack` bypasses.
_NEVER_ATTACK_NAMES = (
    "shopkeeper", "watchman", "watch captain", "guard", "priest", "priestess",
    "aligned priest", "nurse", "oracle", "vault guard", "quest",
)


def _refuse_attack(core_env, kwargs):
    """Return a refusal SkillResult if the target must not be force-fought.

    WHY. `vendor/.../skills.py:389` issues `Command.FIGHT` then a direction.
    FIGHT deliberately skips NetHack's `Really attack? [yn](n)` confirmation --
    and that confirmation is also where OUR peaceful safety net lives
    (`observations.py::_YN_NO_PATTERNS` matches "really attack" precisely to
    protect pets). So the skill punches straight through both guards.

    Measured live: it killed the pet the observation itself labels
    `[PET - don't attack]`, and killed a shopkeeper, whose retaliation killed
    the character outright. Blocking here is the only place we can, short of
    editing vendored code.
    """
    try:
        # Not counted: a skill called without x/y is a normal control path,
        # not a bug (`kwargs.get` -> None -> TypeError).
        tx, ty = int(kwargs.get("x")), int(kwargs.get("y"))
    except (TypeError, ValueError):
        return None
    try:
        from nethack_harness.prompt.features import visible_monsters
        for m in visible_monsters(_raw_view(core_env)):
            if (m.x, m.y) != (tx, ty):
                continue
            if m.is_pet:
                return SkillResult(actions=[], feedback=(
                    f"REFUSED: ({tx},{ty}) is your pet {m.name}. Attacking it "
                    f"would kill it and anger your god. Walk around it instead."))
            low = (m.name or "").lower()
            if any(n in low for n in _NEVER_ATTACK_NAMES):
                return SkillResult(actions=[], feedback=(
                    f"REFUSED: {m.name} at ({tx},{ty}) is peaceful. Attacking it "
                    f"makes it and its allies hostile and is usually fatal. "
                    f"Leave it alone."))
    except Exception as exc:
        # Counted: this guard is the ONLY thing standing between `melee_attack`
        # and the pet/shopkeeper it force-fights (1.2). If it starts raising it
        # must not degrade quietly into "nothing to refuse" -- that is the same
        # failure shape as 3.2.
        _swallowed("_refuse_attack", exc)
        return None
    return None


def _interrupt_worthy(agent, ev) -> bool:
    """Severity filter for mid-skill interruptions (E10/E11 audit, 2026-08-21).

    The vendored `_execute_skill` interrupts on ANY `NewGlyphEvent` -- every
    newly seen monster, OBJECT, or feature glyph. Measured across 25 games:
    that turned 67% of the call budget into consecutive same-tool re-issues
    ("Interrupting skill to rethink because 'A gold piece appeared at
    (2,13)'"), with `np_move_to` alone losing 468 calls to en-route trivia.

    Keep: level change / teleport / low HP (always), a MONSTER newly appearing
    within Chebyshev radius 3 of the player (close enough to matter this
    turn), and newly seen stairs/ladders (the objective; matters under fog).
    Drop: items, corpses, distant monsters, decorative features. The harness's
    own HP-drop and spoiling-food interrupts (below, in the drive loop) are
    unaffected and backstop anything dangerous this filter skips.
    """
    from netplay.nethack_agent import tracking as _tr
    if isinstance(ev, (_tr.DungeonLevelChangeEvent, _tr.TeleportEvent,
                       _tr.LowHealthEvent)):
        return True
    if isinstance(ev, _tr.NewGlyphEvent):
        try:
            from nle.nethack import glyph_is_monster
            if glyph_is_monster(ev.glyph):
                px, py = ev.position
                return max(abs(int(px) - int(agent.blstats.x)),
                           abs(int(py) - int(agent.blstats.y))) <= 3
        except Exception:
            return True   # fail OPEN: better a spurious interrupt than a hidden monster
        try:
            d = ev.describe().lower()
        except Exception:
            return False
        return ("stair" in d) or ("ladder" in d)
    return False


def _execute_skill_filtered(agent, skill, skill_kwargs):
    """`NetHackAgent._execute_skill` (vendored agent.py:112-135) with the
    severity filter above in place of the blanket `NewGlyphEvent` interrupt,
    plus a hunger-state interrupt (the vendored tracker has no hunger event).
    The vendored file stays byte-for-byte; this replicates its loop verbatim
    except for the two marked lines.
    """
    from netplay.core.agent_base import Step
    kwargs_str = [str(x) for x in skill_kwargs.values()]
    skill_description = " ".join([skill.name, *kwargs_str])
    yield Step.think(f"Executing skill '{skill_description}'.")

    start_ingame_time = agent.blstats.time
    prev_hunger = getattr(agent.blstats, "hunger_state", None)
    for step in skill(agent, **skill_kwargs):
        if step.is_done():
            thoughts = f"Skill '{skill_description}' {step.status}"
            thoughts += f": {step.thoughts}" if step.has_thoughts() else ""
            yield Step(step.status, thoughts, step.thought_type, step.step_data)
            return
        yield step

        if (agent.blstats.time - start_ingame_time) >= agent.max_skill_gamesteps:
            yield Step.think(f"Skill has been running for {(agent.blstats.time - start_ingame_time)} timesteps without interruption. Rethinking.")
            return

        if step.executed_action():
            # -- changed line 1: hunger interrupt (no vendored event exists) --
            hz = getattr(agent.blstats, "hunger_state", None)
            if prev_hunger is not None and hz is not None and hz > prev_hunger:
                yield Step.think("Interrupting skill to rethink because 'Your hunger worsened.'.")
                return
            prev_hunger = hz if hz is not None else prev_hunger
            # -- changed line 2: severity filter instead of blanket isinstance --
            interrupt_events = [ev for ev in step.step_data.events
                                if _interrupt_worthy(agent, ev)]
            if len(interrupt_events) != 0:
                yield Step.think(f"Interrupting skill to rethink because '{interrupt_events[0].describe()}'.")
                return


def run_netplay_skill(core_env, skill: Skill, kwargs: Dict[str, Any]) -> SkillResult:
    """Drive one NetPlay skill to completion against the live engine.

    Mirrors upstream's own execution pipeline (`agent.py::_solve_task`, lines
    165-168): `_execute_skill` -> `_skip_more_messages` -> `_update_objects`.
    The first stage is `_execute_skill_filtered` (above): upstream's loop with
    a severity filter on interruptions -- see the E10/E11 audit. The bounding
    behaviour (100 in-game turns per skill, interrupt on level change /
    teleport / low HP) is still NetPlay's.
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
    if getattr(skill, "name", "") == "melee_attack":
        _refusal = _refuse_attack(core_env, kwargs)
        if _refusal is not None:
            return _refusal

    step_observations: list = []
    original_step = wrapped.step

    def _tracking_step(action):
        result = original_step(action)
        step_observations.append(wrapped.last_raw)
        return result

    wrapped.step = _tracking_step
    thoughts: list[str] = []

    def _hp_now():
        """`(hp, max_hp)` from the engine's latest observation; (None, None)
        when unreadable -- the interrupt must never break a skill."""
        try:
            bl = getattr(wrapped.last_raw, "blstats", None)
            if bl is None:
                return None, None
            return int(bl[10]), int(bl[11])  # NLE blstats: 10=hp, 11=max_hp
        except Exception:
            return None, None

    def _progress_key():
        """`(game turn, x, y)` -- the three numbers a real action must move.

        `None` when unreadable, which the livelock guard below treats as
        "cannot tell", never as "stalled".
        """
        try:
            bl = getattr(wrapped.last_raw, "blstats", None)
            if bl is None:
                return None
            return int(bl[20]), int(bl[0]), int(bl[1])  # time, x, y
        except Exception:
            return None

    def _last_message():
        try:
            msg = getattr(wrapped.last_raw, "message", None)
            if msg is None:
                return ""
            return "".join(chr(int(c)) for c in msg if int(c)).strip()
        except Exception:
            return ""

    hp_start, _mx = _hp_now()
    # THE LIVELOCK GUARD. See NO_PROGRESS_STEP_LIMIT: a netplay skill's only
    # bound is in-game turns, and a refused action costs zero of them.
    _stall_key = _progress_key()
    _stall_n = 0
    try:
        # netplay_telemetry gates the severity filter (c1a0bec). Off = the
        # vendored blanket NewGlyphEvent interrupt the baseline was measured
        # with; see nethack_harness.tool_flags.
        if _flags.enabled("netplay_telemetry"):
            strategy = _execute_skill_filtered(agent, skill, kwargs)
        else:
            strategy = agent._execute_skill(skill, kwargs)
        strategy = agent._skip_more_messages(strategy)
        strategy = agent._update_objects(strategy)
        for step in strategy:
            if step.has_thoughts() and step.thoughts:
                thoughts.append(str(step.thoughts))
            if step.executed_action() and step.step_data.done:
                break
            # --- livelock interrupt (E16) --------------------------------
            # THE ONLY BOUND A NETPLAY SKILL HAS IS IN-GAME TURNS:
            # `_execute_skill_filtered` (and upstream's own loop) stop when
            # `blstats.time - start_ingame_time >= max_skill_gamesteps`. An
            # action the GAME REFUSES costs zero in-game turns, so a skill that
            # keeps re-issuing one never terminates -- the delta stays 0
            # forever and no other counter is consulted. The engine env accepts
            # a `no_progress_timeout` but never reads it, so there is no
            # backstop underneath this either.
            #
            # MEASURED, not hypothetical. Resuming E16 checkpoint `c2` puts the
            # player on an intact doorway at (26,10); `explore_level`'s first
            # pathfinding step out of it is diagonal, NetHack answers "You
            # can't move diagonally out of an intact doorway", and the position
            # and clock never change. The rollout burned 18,000+ engine steps
            # at 100% CPU inside ONE tool call and was still spinning when the
            # agent's MCP client gave up at 300s -- which reached the operator
            # as a stalled rollout with no error anywhere, because nothing had
            # failed: a loop was simply running forever.
            #
            # So: bound the loop by something a refused action cannot fake.
            # `(time, x, y)` is exactly that -- the three numbers a real action
            # has to move. Only executed actions count, and the game's own
            # refusal is quoted back so the agent can choose differently
            # instead of re-issuing the same call.
            if step.executed_action():
                _k = _progress_key()
                if _k is not None and _k == _stall_key:
                    _stall_n += 1
                    if _stall_n >= NO_PROGRESS_STEP_LIMIT:
                        _why = _last_message()
                        thoughts.append(
                            f"[INTERRUPTED: this skill took "
                            f"{NO_PROGRESS_STEP_LIMIT} actions in a row without "
                            f"the game clock advancing or your position "
                            f"changing (still at {_k[1]},{_k[2]} on turn "
                            f"{_k[0]}). The game is refusing the move it keeps "
                            f"retrying"
                            + (f": {_why!r}" if _why else "")
                            + ". Do something different -- a different "
                            f"direction, or a different tool.]")
                        break
                else:
                    _stall_key = _k
                    _stall_n = 0
            # --- spoiling-food interrupt ---------------------------------
            # Closed-loop skills run to completion regardless of what happens
            # inside them; `np_explore_level` alone burns up to 100 game turns
            # in ONE agent turn. A corpse is only safe for ~30 turns after the
            # kill, so a single explore call can outlast the entire edible
            # window -- the agent leaves carrying food and returns carrying
            # poison. That is the measured pathway for ~18% of deaths, and no
            # amount of labelling fixes it if the agent is never given the turn
            # in which to act. So break the macro while the food is still food.
            _urgent = _spoiling_now(core_env)
            if _urgent:
                thoughts.append(
                    "[INTERRUPTED: " + "; ".join(_urgent) +
                    ". Stopped early so you can eat before it spoils.]")
                break
            # --- HP-drop interrupt (exp4 fix 2) --------------------------
            # The exp3b death autopsy's clearest artifact: HP tails like
            # 16->16->16->0 -- a rollout at full health on one observation and
            # dead on the next, because a closed-loop skill gave a monster
            # dozens of free attacks between decisions. NetPlay's own low-HP
            # interrupt exists but fires too late and not on every path. Break
            # the macro the moment HP crosses 50% of max OR falls >=25% of max
            # within this one skill, so the agent gets a decision BEFORE the
            # death, not a report after it.
            _hp, _hpmax = _hp_now()
            if _hp is not None and _hpmax:
                # "Crossed" means crossed DURING this skill: a skill that
                # STARTS below half (the agent knowingly acting while hurt --
                # fleeing, praying, fighting back) must not be re-interrupted
                # on its first step or low HP becomes unactionable.
                _crossed_half = (_hp * 2 < _hpmax
                                 and hp_start is not None and hp_start * 2 >= _hpmax)
                _big_drop = hp_start is not None and (hp_start - _hp) * 4 >= _hpmax
                if _hp > 0 and (_crossed_half or _big_drop):
                    # Name the THREAT in the interrupt itself. Measured on exp4:
                    # after this interrupt Claude Code's next call was
                    # melee_attack/pray 12/22 times, while Prime Agent's was
                    # `reveal` 10/11 -- under BBOX_MIN it cannot see what is
                    # hitting it without buying the entity feed, so it spent a
                    # call re-orienting at every danger moment. Embedding
                    # ADJACENT + VISIBLE MONSTERS here removes that forced
                    # extra call for the minimal encodings and is a no-op
                    # burden for the full ones (a dozen tokens of redundancy).
                    _threat = ""
                    try:
                        from nethack_core.observations import shape as _shape2
                        from nethack_harness.prompt.features import monsters_in_sight as _mis
                        _so2 = _shape2(wrapped.last_raw, getattr(core_env, "_character", None))
                        _adj = getattr(_so2, "adjacent", None) or {}
                        if _adj:
                            _order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                            _threat += " ADJACENT: " + " ".join(
                                f"{d}={_adj.get(d, '?')}" for d in _order) + "."
                        _mons = _mis(wrapped.last_raw)
                        if _mons:
                            _threat += " VISIBLE MONSTERS: " + "; ".join(_mons[:4]) + "."
                    except Exception:
                        pass  # the interrupt must fire even if the threat scan fails
                    thoughts.append(
                        f"[INTERRUPTED: HP {_hp}/{_hpmax} -- "
                        f"{'below half' if _crossed_half else 'dropped fast'} "
                        f"during this skill. Something is hitting you.{_threat} "
                        f"Decide now: fight back, retreat, pray, or engrave "
                        f"Elbereth.]")
                    break
    except Exception as e:  # a skill raising must not kill the rollout
        thoughts.append(f"Skill raised {type(e).__name__}: {e}")
    finally:
        wrapped.step = original_step

    # Harvest kills from the INTERMEDIATE observations. `shape()` keeps only the
    # last message, so a 100-step macro surfaces one line and `You kill the X!`
    # is usually lost -- which is why the corpse-age kill log missed 25% of
    # kills. These observations already exist (`step_observations`); we read
    # them for the log ONLY. Nothing here reaches the model: surfacing a
    # macro's whole message stream would flood the observation, which is the
    # opposite of what the encoding work is for.
    try:
        from nethack_core.observations import shape as _shape
        from nethack_harness.prompt.corpse_age import note_kills
        _log_state = core_env
        for _o in step_observations:
            try:
                _so = _shape(_o, getattr(core_env, "_character", None))
                note_kills({"env": core_env}, _so.messages or [],
                           (_so.status or {}).get("time"))
            except Exception as exc:
                _swallowed("run_netplay_skill.kill_log.step", exc)
                continue
    except Exception as exc:
        _swallowed("run_netplay_skill.kill_log", exc)

    feedback = " ".join(t for t in thoughts if t).strip() or "No effect."

    # move_to progress telemetry (E10/E11 audit): 468 of 1068 move_to calls
    # stopped en route and reported only the interrupt reason -- not how far
    # they got or how far remains, so the model re-issued blind. Append both.
    # Prose-only: rides the result payload, tool schema untouched.
    if getattr(skill, "name", "") == "move_to" and _flags.enabled("netplay_telemetry"):
        try:
            tx, ty = int(kwargs.get("x")), int(kwargs.get("y"))
            bx, by = int(agent.blstats.x), int(agent.blstats.y)
            if (bx, by) == (tx, ty):
                where = "target reached"
            else:
                rem = agent.distance_to(tx, ty)
                where = (f"~{int(rem)} steps remaining" if rem not in (None, float("inf"))
                         else "no current path from here")
            feedback += f" [walked {wrapped.steps} steps, now at ({bx},{by}); {where}]"
        except Exception as exc:
            _swallowed("run_netplay_skill.move_to_telemetry", exc)

    # Report what the GAME said, not just that the skill returned.
    # `create_position_command` / `create_inventory_command` yield
    # `Step.completed()` unconditionally without reading the result, so a kick
    # that bounced and a kick that smashed the door open produce the identical
    # string `Skill 'kick 43 5' completed`. NetHack's own message is the only
    # thing that distinguishes them, and it is right there in the final
    # observation -- appending it costs nothing and turns an unfalsifiable
    # "completed" into a fact the agent can act on.
    try:
        from nethack_core.observations import shape as _shp
        _rv = _raw_view(core_env)
        _last = None
        if step_observations:
            _last = step_observations[-1]
        elif _rv is not None:
            _last = getattr(core_env, "_last_observation", None)
        if _last is not None:
            _msgs = [m for m in (_shp(_last, getattr(core_env, "_character", None)).messages or []) if m]
            if _msgs and _msgs[-1] not in feedback:
                feedback = f"{feedback} GAME: {_msgs[-1]}".strip()
    except Exception as exc:
        _swallowed("run_netplay_skill.game_message", exc)

    # Two cleanups on the vendored event text before the agent sees it.
    #
    # (a) numpy scalars repr as `np.int64(2)` under numpy 2.x, and the event
    #     formats tuples of blstats values straight into the message -- so the
    #     agent read `Changed dungeon level from (np.int64(0), np.int64(2))`.
    #
    # (b) that tuple is `(dnum, dlevel)` -- the DUNGEON BRANCH id and the level
    #     within it (winrl.cc:1088-1089, from `u.uz`). The branch is not a seed;
    #     it distinguishes the Dungeons of Doom from the Gnomish Mines, Sokoban
    #     and so on (src/dat/dungeon.def). Almost always it is unchanged, and
    #     printing it as half of an unexplained pair just obscured the number
    #     that matters. Render the level plainly, and mention the branch only on
    #     the rare turn it actually changes.
    try:
        import re as _re
        feedback = _re.sub(r"np\.(?:int|uint|float)\d+\(([-\d.]+)\)", r"\1", feedback)

        def _lvl(m):
            d0, l0, d1, l1 = (int(m.group(i)) for i in (1, 2, 3, 4))
            if d0 == d1:
                return f"Changed dungeon level from {l0} to {l1}"
            return (f"Changed dungeon level from {l0} to {l1} "
                    f"(and entered a different dungeon branch, {d0} -> {d1})")

        feedback = _re.sub(
            r"Changed dungeon level from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)",
            _lvl, feedback)
    except Exception as exc:
        _swallowed("run_netplay_skill.feedback_cleanup", exc)

    # NOTE: a `_fight_fallback` used to sit here, silently converting the
    # "There is no monster at (x,y)" failure into an `F<dir>` attack. It is
    # DELIBERATELY REMOVED. It fired 205 times in a 399-turn rollout, which
    # meant the traces no longer showed the underlying defect at all -- the
    # renderer drawing a monster glyph the tracker does not hold. Masking a bug
    # in the layer above the one that causes it makes the traces undiagnosable.
    # The failure is left raw; the fix belongs in the renderer.

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
