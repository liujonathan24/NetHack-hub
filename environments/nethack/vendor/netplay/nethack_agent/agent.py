"""ADAPTED from NetPlay @ 6acb90d -- netplay/nethack_agent/agent.py.

Upstream this module is the *LLM agent*: an `AgentMemory` rolling chat buffer,
a `_solve_task` loop that calls an LLM skill-selector, gradio/minihack renderers,
and a `NetHackAgent` that owns all of it. It imports langchain, openai,
nle_language_wrapper, gradio and minihack.

We reuse NetPlay's *skill layer*, not its agent loop -- our harness supplies the
policy (an LLM scaffold under evaluation) and calls skills one at a time. So
this module keeps only what the skill layer actually touches:

  * `finish_task_skill`  -- reproduced BYTE-FOR-BYTE (upstream lines 38-44).
  * `NetHackAgent`       -- REPLACED. Upstream's class is
    `NethackBaseAgent` + LLM plumbing; ours is `NethackBaseAgent` + our engine
    and nothing else.

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

from netplay.core.agent_base import NethackBaseAgent
from netplay.core.skill import skill


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

    def __init__(self, env, log_folder: str = "", render: bool = False):
        super().__init__(env=env, log_folder=log_folder, render=render)
