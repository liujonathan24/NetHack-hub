=== EXPERIMENT-SPECIFIC REFLECTION INSTRUCTIONS (level_then_descend.md) ===

You are optimizing for BALROG_min, the MINIMUM of two achievement axes:
dungeon depth and experience level. A game that dives to depth 9 at experience
level 1 scores ZERO, and so does a game that reaches experience level 5 without
ever leaving dungeon level 1. Only a character that BOTH descends AND levels
scores at all. In the round report, BALmax rising while BALmin stays at zero
means the run bought one axis by abandoning the other. That is not progress,
and an entry of yours that produced it should be deleted.

Previous runs of this experiment failed in one direction, repeatedly. The
in-run reward is descent-dominated -- about ten points per new depth against
about four in total for fully scouting a level -- so round after round the
store filled with entries like "don't explore extra rooms once you find
stairs," and the agent became a descend bot: depth 7-9 at experience level 1,
BALmin zero. Do not write that store again.

=== THE ORDERING TO PREFER ===

Err on the side of gaining experience levels FIRST and descending second. The
reference is AutoAscend, the symbolic bot that won the NeurIPS 2021 NetHack
Challenge: it pins the character to dungeon level 1 and farms until an explicit
experience-level threshold, and only then begins its descent chain. It reached
the highest median final experience level of any entrant.

Two corrections matter more than the rule itself:

1. ITS THRESHOLD IS UNREACHABLE HERE. AutoAscend plays about 20,000 game turns
   per episode. This agent's games run 150-700 turns and have never exceeded
   experience level 3. A "reach experience level 8, then descend" entry is one
   the player can never satisfy: it would grind the first level until it starved
   or died and never descend at all -- the mirror of the descend bot, and worth
   exactly as little.

2. AUTOASCEND ITSELF PAYS FOR THIS. Its gate sits above its own median final
   experience level, so its MEDIAN GAME ENDS ON DUNGEON LEVEL 1 -- only about
   half its episodes reach depth 2 -- and its most common death is starvation.
   The competition organisers named the pattern: "the intentional restriction of
   the agent to the top-level dungeons," teams that "camp in the early stages of
   the dungeon, grinding out a high score instead of progressing."

So take the ORDERING, not the constant. What you want is PACING, in both
directions: the character's experience level should keep up with its depth.
As a starting point you may revise with evidence, prefer levelling before
descending when experience level has fallen behind dungeon level, and prefer
descending when it is ahead. Calibrate the actual numbers from these traces --
what this agent reaches and what each step costs it -- not from AutoAscend's
numbers and not from mine. If the evidence supports a different pacing rule,
write that one and say so in your rationale.

Remember that experience from the first dungeon level is very small: a previous
reflection in this experiment established that twenty-three kills on dungeon
level 1 still leave a character at experience level 1. "Farm where you are" is
therefore not automatically levelling. Depth at which the character can still
win its fights is what levels it.

Levelling requires fighting winnable fights, so watch whether the player is
fighting at all, and with what. In previous rounds the combat policy went
uncalled for entire rounds while combat happened through raw keypresses. An
entry that makes the player fight the right monsters, or retreat from the wrong
ones, is worth more than another entry about stairs.

=== SUB-AGENTS ===

You may create sub-agent specs (`rlm.harness.create_subagent(...)`); they are
rendered into the next player's prompt and the player can invoke one from its
kernel. A sub-agent acts on THE SAME live character -- it is a second controller
of one avatar sharing one call budget, not a second game. So specs must hand
control over SEQUENTIALLY, each with a clear entering and exit condition, in the
style of AutoAscend's priority ordering: emergency healing preempts combat,
combat preempts exploration, exploration preempts descent. Never spec two
sub-agents that would hold the character at the same time. Spec one only where
the traces show a whole category of play going missing -- combat is the standing
example -- and cite the rollouts and turns that motivated it.

=== EVERYTHING ELSE ===

The contract above still governs: 6 entries per kind at 180 characters, evidence
from these rollouts, deletion when the evidence contradicts an entry, and a
rationale file. This file changes what you optimize toward, not how you work.
