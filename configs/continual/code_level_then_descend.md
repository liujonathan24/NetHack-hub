You are optimizing for BALROG_min, which is the MINIMUM of two achievement
axes: dungeon depth and experience level. A game that dives to depth 9 at
experience level 1 scores ZERO on this metric, and so does a game that reaches
experience level 5 without ever leaving dungeon level 1. Only a character that
BOTH descends AND levels scores at all. Read every round's report with that in
mind: BALROG_max rising while BALROG_min stays at zero means the run bought one
axis by abandoning the other, and is not progress.

The previous runs of this experiment failed in one specific direction. Round
after round, reflection made the agent descend faster -- because the in-run
reward is descent-dominated (about ten points per new depth against about four
in total for fully scouting a level) -- until the policies were a descend bot:
it reached depth 7-9 at experience level 1, and BALROG_min sat at zero. The
fighting policy was written and then never called; combat happened, when it
happened at all, through raw keypresses. Do not rebuild that agent.

=== THE ORDERING TO PREFER ===

Err on the side of gaining experience levels FIRST, and descending second.
The reference point is AutoAscend, the symbolic bot that won the NeurIPS 2021
NetHack Challenge: its global strategy pins the character to dungeon level 1
and farms there until an explicit experience-level threshold is met, and only
then does the descent milestone chain begin
(`global_logic.py`: `condition = lambda: self.agent.blstats.experience_level >= 8`).
It reached the highest median final experience level of any entrant in the
competition.

Two corrections before you copy it, both of which matter more than the rule:

1. ITS THRESHOLD IS NOT REACHABLE HERE. AutoAscend plays roughly 20,000 game
   turns per episode. This agent's games run 150-700 game turns and have never
   exceeded experience level 3. An "XL >= 8 then descend" gate in this regime
   is a gate that never opens: the agent would grind the first level until it
   starved or died, never descend, and score zero on the depth axis -- the
   mirror image of the descend bot, and just as worthless.

2. AUTOASCEND ITSELF PAYS FOR THIS. Because its gate is set above its own
   median final experience level, its MEDIAN GAME ENDS ON DUNGEON LEVEL 1 --
   only about half its episodes ever reach depth 2. Its most common recorded
   death is starvation while fainted from lack of food. The competition
   organisers named this behaviour directly, as "the intentional restriction of
   the agent to the top-level dungeons" and teams that "camp in the early
   stages of the dungeon, grinding out a high score instead of progressing."
   A hard level-gate trades one degenerate policy for another.

So take the ORDERING from AutoAscend, not the constant. What you want is
PACING: the character's experience level should keep up with its depth, in
both directions. Concretely, as a starting point you may revise with evidence:
prefer to gain a level before descending when the character's experience level
has fallen behind its dungeon level, and prefer to descend when it is ahead.
Calibrate the actual numbers from the traces you have -- what depth and what
experience level this agent actually reaches, and how many turns each costs it
-- not from AutoAscend's numbers and not from mine. If the evidence says a
different rule paces better, write that one and say so in your rationale.

Levelling means fighting winnable fights, which means the combat policy has to
be worth calling: a policy that fails on a monster that stepped, or that the
player routes around instead of invoking, will not level anything. Watch for
that in the traces. Note also that experience from the first dungeon level's
monsters is very small -- a previous reflection in this experiment established
that twenty-three kills on dungeon level 1 still leave a character at
experience level 1 -- so "farm where you are" is not automatically levelling
either. Depth where the character can still win fights is what levels.

=== SUB-AGENTS: DELEGATE SEMANTIC TASKS ===

You may also create sub-agent specs in the harness store
(`rlm.harness.create_subagent(...)`), which are rendered into the next player's
prompt. The player can invoke one with `asyncio.create_task(rlm('...'))` or
`await rlm('...')` from its kernel. A sub-agent inherits this rollout's game
connection and acts on THE SAME live character -- it is a second controller of
one avatar, not a second game.

That single-character constraint is the whole design problem, and AutoAscend
solves it in a way worth copying: it does not run its behaviours in parallel.
It keeps ONE priority-ordered list, and whichever behaviour's entering
condition fires first preempts the running one, mid-execution. Its order,
highest priority first, is: emergency healing (potion, then prayer, at hard HP
fractions) > combat while swallowed > ordinary combat (fires when a hostile is
within about seven steps) > vault-guard handling > eating > curing disease >
waiting out blindness/confusion > altar sacrifice > Sokoban > and only at the
bottom, the base milestone chain of go-to-level, explore, descend. Exploration
is interrupted by combat; combat is interrupted by emergency healing.

Therefore: give each sub-agent ONE semantic task with a clear entering
condition and a clear exit condition, and hand control over sequentially. Good
candidates, mirroring that decomposition, are combat and threat handling,
level exploration, survival (HP, nutrition, prayer timing), and descent.
Do NOT spec two sub-agents that would hold the controller at the same time --
they would fight over one character and spend one shared call budget. Say in
each spec when the sub-agent should be started and when it must return control.

Sub-agents are a means, not the goal: spec them where the traces show the
player losing a whole category of play (combat is the standing example -- the
fighting policy went uncalled for entire rounds), and leave them unspec'd where
they would only add indirection. Every spec you write must cite the rollouts
and turns that motivated it, exactly like a code edit.

=== EVERYTHING ELSE ===

The contract above still governs: differential evidence per change, reverting
is first-class, cite seeds and turns, gate and commit. This file changes what
you are optimizing toward, not how you are allowed to work.
