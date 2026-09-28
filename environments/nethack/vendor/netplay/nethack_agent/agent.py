"""ADAPTED from NetPlay @ 6acb90d -- netplay/nethack_agent/agent.py.

Upstream this module is the *LLM agent*: an `AgentMemory` rolling chat buffer,
a `_solve_task` loop that calls an LLM skill-selector, gradio/minihack renderers,
and a `NetHackAgent` that owns all of it. It imports langchain, openai,
nle_language_wrapper, gradio and minihack.

We reuse NetPlay's *skill layer*, not its agent loop -- our harness supplies the
policy (an LLM scaffold under evaluation) and calls skills one at a time. So
this module keeps only what the skill layer actually touches:

  * `finish_task_skill`     -- reproduced BYTE-FOR-BYTE (upstream lines 38-44).
  * `_execute_skill`        -- ADAPTED from upstream lines 215-238: the
    interrupt policy changed, see LOCAL CHANGE below.
  * `_skip_more_messages`   -- BYTE-FOR-BYTE (upstream lines 189-196).
  * `_update_objects`       -- BYTE-FOR-BYTE (upstream lines 198-213).
  * `NetHackAgent.__init__` -- REPLACED (dropped the LLM/descriptor/selector
    plumbing; kept the two knobs the retained methods read).

The three `_`-prefixed methods are kept because they are not LLM plumbing --
they are the *skill execution contract*, and they materially bound behaviour:
`_execute_skill` stops a skill after `max_skill_gamesteps` (100) in-game turns
and interrupts it on a level change, teleport, low health or (upstream) a
newly-seen glyph -- locally, a nearby hostile monster instead (see below).
`explore_level` runs until exploration is provably exhausted, so without this
cap it would not be the same action NetPlay's agent actually took.

This matters for fidelity and is worth being precise about: every attribute
`skills.py` uses on the agent -- `blstats`, `current_level`, `step`,
`get_path_to`, `distance_to`, `get_distance_map`, `get_walkable_mask`,
`waiting_for_popup`, `current_game_message`, `set_current_room`,
`avoid_monsters` -- is defined by the *vendored, byte-verbatim*
`netplay/core/agent_base.py`, not here. None of the skill-relevant behaviour
lives in upstream's `NetHackAgent`, so dropping the LLM plumbing changes no
skill semantics.

Everything else from upstream's agent.py (AgentMemory, _solve_task,
_skip_more_messages, _update_objects, _execute_skill, the renderers) belongs to
the agent loop, not the skills, and is not reproduced. See vendor/PROVENANCE.md.

LOCAL CHANGE (interrupt policy) -- `_execute_skill` is NO LONGER byte-verbatim.
Upstream interrupts on every `NewGlyphEvent`: any monster, object, down-stair,
altar or fountain glyph that was not on screen a moment ago, anywhere on the
level. Play-tests showed that makes the closed-loop skills unusable (a kick
interrupted by a lichen 60 columns away, every shop item and the corpse of the
monster just killed "appearing", the same werejackal stopping two calls in a
row with no progress) and that the event text leaked TRUE object identities
("worthless piece of green glass" for what the game calls a "green gem"),
because `describe_glyph` names the glyph's actual object type. The level-change
text also printed the level index within the branch, not the Dlvl the status
line shows. What changed, and nothing else:

  * `NewGlyphEvent` no longer interrupts. Items and dungeon features never do.
    Instead `_hostile_monster_interrupt` interrupts for a HOSTILE monster that
    comes within `HOSTILE_NEAR_RADIUS` squares, or that keeps closing in from
    within `APPROACH_RADIUS`, once per monster (and not again on the next call
    while it stays put). Pets and peaceful monsters never interrupt; see that
    method for how peacefulness is decided. It reads observations only.
  * Level changes are reported in player-visible terms (`describe_level_change`).
  * Low HP, teleport and level change still interrupt exactly as upstream.
  * `last_interrupt` records why the last skill was interrupted (for the
    harness's "no progress" note).
"""

from netplay.core.agent_base import NethackBaseAgent, Step
from netplay.core.skill import Skill, skill
from netplay.nethack_utils.nle_wrapper import RawKeyPress
import netplay.nethack_agent.tracking as tracking

from nle.nethack import actions
import nle.nethack as nh

from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np


@skill(
    "finish_task",
    "Use this skill when the task has been fulfilled. DO NOT CONTINUE playing without an task.",
    parameters=[]
)
def finish_task_skill(agent: "NetHackAgent"):
    assert False


class NetHackAgent(NethackBaseAgent):
    """NetPlay's agent-side skill context, bound to our engine.

    All pathfinding, level tracking and stepping behaviour is inherited from the
    verbatim `NethackBaseAgent`. `env` must present the small gym-like surface
    that `NethackBaseAgent` uses: `reset()`, `step(action)` returning a 5-tuple,
    dict-style observations, and `.unwrapped.waiting_for_{yn,line,space}` --
    which `nethack_harness.tools.netplay_true.NetPlayEngineEnv` provides on top
    of `nethack_core`.
    """

    def __init__(
        self,
        env,
        log_folder: str = "",
        render: bool = False,
        max_skill_gamesteps=100,
        update_hidden_objects=False,
    ):
        super().__init__(env=env, log_folder=log_folder, render=render)
        self.max_skill_gamesteps = max_skill_gamesteps
        self.update_hidden_objects = update_hidden_objects
        # ADDED (fix/netcore-input): monster glyphs that answered "Really
        # attack?" when walked into, so pathing treats them as obstacles
        # (skills._known_peaceful). Lives as long as this agent (one episode).
        self.known_peaceful_glyphs = set()

        # LOCAL (interrupt policy): see the module docstring.
        # `describe_tile(x, y)` -> what NetHack's farlook (`;`) prints for that
        # map square, or None when the harness cannot supply it. Set by
        # nethack_harness.tools.netplay_true.get_agent.
        self.describe_tile: Optional[Callable[[int, int], Optional[str]]] = None
        self.last_interrupt: Optional[str] = None
        self._tracked_monsters: List[dict] = []
        # Monsters an interrupt already named during the previous skill call.
        self._reported_last_call: List[dict] = []

    # ADDED (fix/netcore-input): agent_base.get_path_to / distance_to compute
    # `avoid_monsters` from their argument and then pass `self.avoid_monsters`
    # instead, so an explicit `avoid_monsters=True` (melee_attack's approach
    # walk) was silently ignored and the walk went through other monsters,
    # peaceful ones included. These overrides pass the resolved value on.
    # (get_distance_map in agent_base already does this correctly.)
    def get_path_to(self, x, y, bump_into_unwalkables=True, avoid_monsters=None):
        if not avoid_monsters:
            avoid_monsters = self.avoid_monsters
        return self.last_bfs_results.get_path_to(x, y, bump_into_unwalkables, avoid_monsters=avoid_monsters)

    def distance_to(self, x, y, bump_into_unwalkables=True, avoid_monsters=None):
        if not avoid_monsters:
            avoid_monsters = self.avoid_monsters
        return self.last_bfs_results.distance_to(x, y, bump_into_unwalkables, avoid_monsters=avoid_monsters)

    # ----------------------------------------------------------------------
    # Below: upstream netplay/nethack_agent/agent.py, reproduced verbatim,
    # except `_execute_skill` (LOCAL CHANGE, see the module docstring).
    # ----------------------------------------------------------------------

    def _skip_more_messages(self, generator: Iterator[Step]) -> Iterator[Step]:
        for step in generator:
            yield step
            if step.is_done():
                return

            # LOCAL: bounded. Upstream pressed SPACE for as long as the screen
            # showed --More--; a screen that SPACE never clears (reached by applying
            # a nonsense inventory letter) looped forever and grew until the kernel
            # killed the game server. Give up after MORE_PRESS_LIMIT presses, try
            # ESC once, and otherwise leave the screen for the agent to answer.
            presses = 0
            while self.showing_more_message and presses < self.MORE_PRESS_LIMIT:
                yield self.step(RawKeyPress.KEYPRESS_SPACE)
                presses += 1
            if self.showing_more_message:
                yield self.step(RawKeyPress.KEYPRESS_ESC)
                if self.showing_more_message:
                    yield Step.failed(f"A --More-- screen did not clear after {presses} presses and ESC; "
                                      "it is left open for you to answer.")
                    return

    def _update_objects(self, generator: Iterator[Step]) -> Iterator[Step]:
        for step in generator:
            yield step

        # There was an issue with hidden objects still showing up in the environment description
        # This fixes it by asking the game to hide monsters
        # This updates our tracked data
        # Not sure if not yielding the steps will cause any issues, so far it hasn't
        if self.update_hidden_objects and not self.waiting_for_popup():
            self.step(actions.Command.EXTCMD)
            self.step(RawKeyPress.KEYPRESS_t)
            self.step(RawKeyPress.KEYPRESS_e)
            self.step(RawKeyPress.KEYPRESS_ENTER)
            self.step(RawKeyPress.KEYPRESS_c)
            self.step(RawKeyPress.KEYPRESS_ESC)
            self.step(RawKeyPress.KEYPRESS_ESC)

    def _execute_skill(self, skill: Skill, skill_kwargs: Dict[str, Any]) -> Iterator[Step]:
        kwargs_str = [str(x) for x in skill_kwargs.values()]
        skill_description = " ".join([skill.name, *kwargs_str])
        yield Step.think(f"Executing skill '{skill_description}'.")

        # LOCAL (interrupt policy): seed monster tracking from what is on
        # screen as the call starts; the melee target is never an interrupt.
        self.last_interrupt = None
        exempt = None
        if skill.name == "melee_attack":
            try:
                exempt = (int(skill_kwargs.get("x")), int(skill_kwargs.get("y")))
            except (TypeError, ValueError):
                exempt = None
        self._begin_monster_tracking(exempt)

        start_ingame_time = self.blstats.time
        prev_blstats = self.blstats
        # LOCAL (no-progress guard): the cap below counts in-game turns, so a skill
        # that keeps sending keys the game rejects without spending a turn (a move
        # into a doorway diagonal, a wall) looped forever and wedged the game
        # server. Stop once this many keys in a row leave the clock where it was.
        # press_key is exempt: its key count is bounded by what the agent typed.
        idle_keys, idle_since = 0, self.blstats.time
        # A skill can also loop while yielding only thoughts, sending no key at all;
        # count every step, and stop that the same way.
        idle_steps = 0
        try:
            for step in skill(self, **skill_kwargs):
                if step.is_done():
                    thoughts = f"Skill '{skill_description}' {step.status}"
                    thoughts += f": {step.thoughts}" if step.has_thoughts() else ""
                    yield Step(step.status, thoughts, step.thought_type, step.step_data)
                    return
                yield step

                idle_steps = idle_steps + 1 if self.blstats.time == idle_since else 0
                if idle_steps >= self.MAX_IDLE_STEPS and skill.name != "press_key":
                    yield Step.failed(f"Skill '{skill_description}' stopped: {idle_steps} steps in a row "
                                      "without the game clock advancing, so it was making no progress.")
                    return

                if step.executed_action() and skill.name != "press_key":
                    if self.blstats.time == idle_since:
                        idle_keys += 1
                    else:
                        idle_keys, idle_since = 0, self.blstats.time
                    if idle_keys >= self.MAX_IDLE_KEYS:
                        message = self.current_game_message.strip()
                        yield Step.failed(
                            f"Skill '{skill_description}' stopped: {idle_keys} keys in a row did not advance "
                            f"the game clock, so it was making no progress"
                            + (f" (the game said: {message!r})" if message else "") + ".")
                        return

                if (self.blstats.time - start_ingame_time) >= self.max_skill_gamesteps:
                    yield Step.think(f"Skill has been running for {(self.blstats.time - start_ingame_time)} timesteps without interruption. Rethinking.")
                    return

                if step.executed_action():
                    # LOCAL: NewGlyphEvent removed from this tuple (see docstring).
                    interrupt_event_types = (tracking.DungeonLevelChangeEvent, tracking.TeleportEvent, tracking.LowHealthEvent)
                    interrupt_events = [event for event in step.step_data.events if isinstance(event, interrupt_event_types)]
                    reason = None
                    if len(interrupt_events) != 0:
                        event = interrupt_events[0]
                        if isinstance(event, tracking.DungeonLevelChangeEvent):
                            reason = describe_level_change(
                                prev_blstats.depth, prev_blstats.dungeon_number,
                                self.blstats.depth, self.blstats.dungeon_number)
                            self._tracked_monsters = []
                        else:
                            reason = event.describe()
                    else:
                        reason = self._hostile_monster_interrupt()
                    if reason:
                        self.last_interrupt = reason
                        yield Step.think(f"Interrupting skill to rethink because '{reason}'.")
                        return
                prev_blstats = self.blstats
        finally:
            self._end_monster_tracking()

    # ----------------------------------------------------------------------
    # LOCAL (interrupt policy). Not upstream code.
    # ----------------------------------------------------------------------

    #: A hostile monster this close (Chebyshev squares, i.e. moves) interrupts.
    #: Why 7: most early monsters move at speed <= 12, the hero's speed, so one
    #: that is 7 squares away needs ~6 turns to reach melee -- enough for one
    #: considered decision (throw, retreat, Elbereth, change route) before it
    #: can attack. It also roughly covers an ordinary lit room around the hero,
    #: so "a monster entered my room" interrupts while monsters wandering in
    #: other rooms (the 60-column-away lichen of the play-tests) do not.
    HOSTILE_NEAR_RADIUS = 7
    #: Keys in a row that may leave the game clock unchanged before a skill is
    #: stopped as making no progress (see `_execute_skill`). Menus, prompts and
    #: `--More--` screens legitimately take a few keys without a turn passing;
    #: fifty is far above that and far below "forever".
    MAX_IDLE_KEYS = 50
    #: Steps of any kind (thoughts included) in a row without the clock moving.
    MAX_IDLE_STEPS = 200
    #: SPACE presses spent dismissing one run of --More-- screens.
    MORE_PRESS_LIMIT = 20
    #: A hostile monster further away than HOSTILE_NEAR_RADIUS but within this
    #: range interrupts only if it is visibly closing in: it moved toward the
    #: hero on APPROACH_STREAK consecutive moves. (Two was too easy for a
    #: wandering monster to satisfy by chance in live tests; three is not.)
    APPROACH_RADIUS = 10
    APPROACH_STREAK = 3
    #: How many observations a tracked monster may be out of sight and still be
    #: recognised as the same monster when it reappears.
    FORGET_AFTER = 10
    #: Species that are always generated peaceful (M2_PEACEFUL). Used ONLY
    #: when there is no farlook description of the monster itself.
    ALWAYS_PEACEFUL = ("shopkeeper", "guard", "prisoner", "Oracle",
                       "aligned priest", "high priest", "watchman",
                       "watch captain")

    def _tile_description(self, x: int, y: int) -> Optional[str]:
        fn = self.describe_tile
        if fn is None:
            return None
        try:
            text = fn(int(x), int(y))
        except Exception:
            return None
        return text or None

    def _visible_hostiles(self) -> List[dict]:
        """Hostile monsters on screen now, from the observation only.

        Pets are recognised from the glyph. Peaceful monsters are recognised
        from the farlook description of the square ("peaceful gnome",
        "tame kitten") -- the text a human gets for free with `;`, taken from
        the engine's per-square screen descriptions without sending a key.
        Without descriptions, species that are always peaceful are skipped by
        name. The name reported is the one the player sees: the displayed
        monster (so a shapeshifter is named by its current form), never
        anything the glyph does not show.
        """
        obs = self.last_observation
        if obs is None:
            return []
        glyphs = np.asarray(obs["glyphs"])
        hx, hy = int(self.blstats.x), int(self.blstats.y)
        ys, xs = np.nonzero(nh.glyph_is_monster(glyphs))
        out = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            if (x, y) == (hx, hy):
                continue
            g = int(glyphs[y, x])
            if nh.glyph_is_pet(g):
                continue
            mon = int(nh.glyph_to_mon(g))
            species = nh.permonst(mon).mname or "monster"
            descr = self._tile_description(x, y)
            low = (descr or "").lower()
            if low.startswith(("peaceful ", "tame ")):
                continue
            # Is the farlook text about this monster? Either it names the
            # species ("gnome lord", "invisible stalker") or it is a proper
            # name (an angry shopkeeper is just "Asidonhopo").
            if species.lower() in low or (descr and descr[:1].isupper()):
                name = descr.split(",")[0].strip() or species
            else:
                # No description, or one that is about the square rather than
                # the monster (the map can show a monster the hero cannot look
                # at, e.g. one in a dark area): name the species on the map.
                if species in self.ALWAYS_PEACEFUL:
                    continue
                name = species
            out.append({"mon": mon, "x": x, "y": y, "name": name,
                        "d": max(abs(x - hx), abs(y - hy))})
        return out

    @staticmethod
    def _match(m: dict, candidates: List[dict], max_dist: int, consume: bool = True) -> Optional[dict]:
        best, best_d = None, None
        for c in candidates:
            if c.get("_taken") or c["mon"] != m["mon"]:
                continue
            d = max(abs(c["x"] - m["x"]), abs(c["y"] - m["y"]))
            if d <= max_dist + c.get("missing", 0) and (best is None or d < best_d):
                best, best_d = c, d
        if best is not None and consume:
            best["_taken"] = True
        return best

    def _begin_monster_tracking(self, exempt=None):
        try:
            carried = [dict(c) for c in self._reported_last_call]
            tracked = []
            for m in self._visible_hostiles():
                # Not consumed: with two of a species close together the
                # identities are ambiguous, and repeating is the worse error.
                prev = self._match(m, carried, max_dist=2, consume=False)
                m.update(reported=prev is not None,
                         exempt=exempt is not None and (m["x"], m["y"]) == tuple(exempt),
                         inside=m["d"] <= self.HOSTILE_NEAR_RADIUS,
                         adjacent=m["d"] <= 1, streak=0, missing=0)
                tracked.append(m)
            self._tracked_monsters = tracked
        except Exception:
            self._tracked_monsters = []

    def _end_monster_tracking(self):
        self._reported_last_call = [
            {"mon": m["mon"], "x": m["x"], "y": m["y"]}
            for m in self._tracked_monsters if m.get("reported") and not m.get("missing")
        ]

    def _hostile_monster_interrupt(self) -> Optional[str]:
        """Update monster tracking after one engine step; return the interrupt
        reason, or None. Never raises (a bug here must not break a skill)."""
        try:
            return self._hostile_monster_interrupt_unsafe()
        except Exception:
            return None

    def _hostile_monster_interrupt_unsafe(self) -> Optional[str]:
        hx, hy = int(self.blstats.x), int(self.blstats.y)
        prev_list = self._tracked_monsters
        for p in prev_list:
            p.pop("_taken", None)
        carried = [dict(c) for c in self._reported_last_call]
        current = self._visible_hostiles()
        # Match the closest current monsters first, so a monster next to the
        # hero keeps its identity when two of a species are on screen.
        current.sort(key=lambda m: m["d"])
        reason = None
        for m in current:
            d = m["d"]
            p = self._match(m, prev_list, max_dist=2)
            if p is None:
                c = self._match(m, carried, max_dist=2, consume=False)
                m.update(reported=c is not None, exempt=False, streak=0,
                         inside=False, adjacent=False)
                if reason is None and not m["reported"] and d <= self.HOSTILE_NEAR_RADIUS:
                    reason = f"A hostile {m['name']} came into view at ({m['x']},{m['y']}), {d} squares away"
                    m["reported"] = True
            else:
                moved = (m["x"], m["y"]) != (p["x"], p["y"])
                was_d = max(abs(p["x"] - hx), abs(p["y"] - hy))
                if moved and d < was_d:
                    streak = p.get("streak", 0) + 1
                elif moved:
                    streak = 0
                else:
                    streak = p.get("streak", 0)
                m.update(reported=p.get("reported", False), exempt=p.get("exempt", False),
                         streak=streak, inside=p.get("inside", False),
                         adjacent=p.get("adjacent", False))
                if reason is None and not m["exempt"]:
                    if m["reported"]:
                        if d <= 1 and not m["adjacent"]:
                            reason = f"The hostile {m['name']} is now next to you at ({m['x']},{m['y']})"
                    elif d <= self.HOSTILE_NEAR_RADIUS and not m["inside"]:
                        reason = f"A hostile {m['name']} is {d} squares away at ({m['x']},{m['y']})"
                    elif (d <= self.APPROACH_RADIUS and d > self.HOSTILE_NEAR_RADIUS
                          and streak >= self.APPROACH_STREAK):
                        reason = f"A hostile {m['name']} at ({m['x']},{m['y']}) is approaching you, {d} squares away"
                    if reason is not None:
                        m["reported"] = True
            m["inside"] = d <= self.HOSTILE_NEAR_RADIUS
            m["adjacent"] = d <= 1
            m["missing"] = 0
        # Keep monsters that just went out of sight for a while, so one that
        # steps behind a wall and back is not "new" again.
        for p in prev_list:
            if not p.get("_taken") and p.get("missing", 0) < self.FORGET_AFTER:
                q = dict(p)
                q["missing"] = p.get("missing", 0) + 1
                current.append(q)
        for p in prev_list:
            p.pop("_taken", None)
        for q in current:
            q.pop("_taken", None)
        self._tracked_monsters = current
        return reason


#: Dungeon branch names, indexed by `u.uz.dnum` (blstats dungeon_number), in
#: the order of NetHack 3.6's dat/dungeon.def. These are the names the game's
#: own dungeon overview (Ctrl-O) shows the player.
BRANCH_NAMES = {
    0: "the Dungeons of Doom",
    1: "Gehennom",
    2: "the Gnomish Mines",
    3: "the Quest",
    4: "Sokoban",
    5: "Fort Ludios",
    6: "Vlad's Tower",
    7: "the Elemental Planes",
}


def describe_level_change(old_depth, old_dnum, new_depth, new_dnum) -> str:
    """LOCAL: a level change in the terms the player sees.

    Uses the depth the status line shows as `Dlvl:` (blstats depth), not the
    level index within the branch, and names the branch only when it changes.
    """
    old_depth, new_depth = int(old_depth), int(new_depth)
    old_dnum, new_dnum = int(old_dnum), int(new_dnum)
    if new_depth > old_depth:
        text = f"You went down from Dlvl {old_depth} to Dlvl {new_depth}"
    elif new_depth < old_depth:
        text = f"You went up from Dlvl {old_depth} to Dlvl {new_depth}"
    else:
        text = f"You are now on a different level at Dlvl {new_depth}"
    if new_dnum != old_dnum:
        branch = BRANCH_NAMES.get(new_dnum)
        text += f" and entered {branch}" if branch else " and entered a different dungeon branch"
    return text
