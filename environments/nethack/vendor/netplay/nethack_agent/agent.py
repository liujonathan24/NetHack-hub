"""ADAPTED from NetPlay @ 6acb90d -- netplay/nethack_agent/agent.py.

Upstream this module is the *LLM agent*: an `AgentMemory` rolling chat buffer,
a `_solve_task` loop that calls an LLM skill-selector, gradio/minihack renderers,
and a `NetHackAgent` that owns all of it. It imports langchain, openai,
nle_language_wrapper, gradio and minihack.

We reuse NetPlay's *skill layer*, not its agent loop -- our harness supplies the
policy (an LLM scaffold under evaluation) and calls skills one at a time. So
this module keeps only what the skill layer actually touches:

  * `finish_task_skill`     -- reproduced BYTE-FOR-BYTE (upstream lines 38-44).
  * `_execute_skill`        -- BYTE-FOR-BYTE (upstream lines 215-238).
  * `_skip_more_messages`   -- BYTE-FOR-BYTE (upstream lines 189-196).
  * `_update_objects`       -- BYTE-FOR-BYTE (upstream lines 198-213).
  * `NetHackAgent.__init__` -- REPLACED (dropped the LLM/descriptor/selector
    plumbing; kept the two knobs the retained methods read).

The three `_`-prefixed methods are kept because they are not LLM plumbing --
they are the *skill execution contract*, and they materially bound behaviour:
`_execute_skill` stops a skill after `max_skill_gamesteps` (100) in-game turns
and interrupts it on a level change, teleport, newly-seen glyph or low health.
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
"""

from netplay.core.agent_base import NethackBaseAgent, Step
from netplay.core.skill import Skill, skill
from netplay.nethack_utils.nle_wrapper import RawKeyPress
import netplay.nethack_agent.tracking as tracking

from nle.nethack import actions

from typing import Any, Dict, Iterator


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

    # ----------------------------------------------------------------------
    # Below: upstream netplay/nethack_agent/agent.py, reproduced verbatim.
    # ----------------------------------------------------------------------

    def _skip_more_messages(self, generator: Iterator[Step]) -> Iterator[Step]:
        for step in generator:
            yield step
            if step.is_done():
                return

            while self.showing_more_message:
                yield self.step(RawKeyPress.KEYPRESS_SPACE)

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

        start_ingame_time = self.blstats.time
        for step in skill(self, **skill_kwargs):
            if step.is_done():
                thoughts = f"Skill '{skill_description}' {step.status}"
                thoughts += f": {step.thoughts}" if step.has_thoughts() else ""
                yield Step(step.status, thoughts, step.thought_type, step.step_data)
                return
            yield step

            if (self.blstats.time - start_ingame_time) >= self.max_skill_gamesteps:
                yield Step.think(f"Skill has been running for {(self.blstats.time - start_ingame_time)} timesteps without interruption. Rethinking.")
                return

            if step.executed_action():
                interrupt_event_types = (tracking.DungeonLevelChangeEvent, tracking.TeleportEvent, tracking.NewGlyphEvent, tracking.LowHealthEvent)
                interrupt_events = [event for event in step.step_data.events if isinstance(event, interrupt_event_types)]
                if len(interrupt_events) != 0:
                    yield Step.think(f"Interrupting skill to rethink because '{interrupt_events[0].describe()}'.")
                    return
