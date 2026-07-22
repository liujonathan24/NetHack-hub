"""nethack_harness.curriculum

The six-floor "primitives" curriculum — an honest-navigation task.

The former 13-tier named curriculum ladder (curriculum.py + milestones.py +
subgoals.py) was removed in the NetHack-Hub extraction; the default env runs the
standard full ascension game (see nethack.GameSpec / FULL_GAME_SPEC).

This package now implements :class:`CurriculumPrimitivesEnv`, the compressed
6-floor down / 6-floor up tour (DoD 1-2-3 <-> Gehennom 48-49-50) reached
**entirely by primitive stair navigation** — the faithful, no-cheat sibling of
``CurriculumEnv`` (which lets the descend/ascend mega-skills teleport across
floors). The agent must navigate onto a real stair and press ``>``/``<``; the
only internal redirection is the two cross-branch boundary jumps, each firing
only when the hero genuinely stands on the boundary stair:

    * DoD level 3's DOWN stair -> Gehennom deep_lo (with a stats-only upgrade);
    * Gehennom deep_lo's UP stair -> back to DoD level 3.

At the DoD3->Gehennom boundary the env grants the invocation ritual kit
(``EngineEnv.grant_invocation_kit``). On the Invocation level (Gehennom's
``num_dunlevs-1`` maze, which has no down-staircase by design) it stages the
hero adjacent to the vibrating square (``EngineEnv.seat_on_invocation_square``);
``invocation_square`` surfaces the square's coords (``EngineEnv.invocation_pos``)
so the agent can complete the ritual itself. On-stair checks use the raw
engine's ``hero_on_stair`` via the public ``EngineEnv.engine`` property.

Depends on the engine's public invocation hooks, which live on engine PR #37
(``feat/4b-six-floor-hooks``); until that reaches engine main, the Hub's
``nethack-core`` dependency must point at that branch to expose the wrappers.
"""
from .curriculum_primitives_env import CurriculumPrimitivesEnv

__all__ = ["CurriculumPrimitivesEnv"]
