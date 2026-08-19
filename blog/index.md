# Revisiting NetHack with Agents.
<!--A strong model, NetPlay's own tools, full vision — and it still doesn't beat the game -->

*Draft. Prose here; the live gameplay archive is `e7_viewer.html`.*

Games have been a popular method for benchmarking language models in recent years due to their ability to detect usability errors in models (Jeurissen et al., 2024). Although knowledge-based benchmarks for language models have steadily improved over time, perceived LLM strengths continue to lag what their benchmarks report. Games, which are, at their core, long-horizon tasks that require agents to plan and act over extended periods, offer us a better evaluation on the capabilities of language models since game environments are arguably simpler than many of the real-world tasks we ultimately hope autonomous agents can solve. If they can’t do well on games, then they are not yet ready for real-world long-running tasks. 


As a result, substantial research has focused on optimizing language models for game playing, with particularly rapid progress over the past year.
Language models have achieved human-level or superhuman performance on games such as Pokémon Red and Blue, Baba Is You, and several games in the BALROG benchmark suite. However, one notable exception remains NetHack. Originally released in the 1980s, NetHack appears to be orders of magnitude more difficult than the other games that have been successfully tackled. Even notoriously challenging environments such as Montezuma's Revenge have now seen strong performance, yet NetHack remains largely unsolved (e.g., zero neural agents were able to complete the NetHack Learning Environment during the NeurIPS 2021 competition, and symbolic approaches continue to outperform neural networks; Onuki et al., 2025). In the BALROG paper on LLMs and games, the strongest language models solve only about 7% of NetHack, leaving it as an open challenge. 

A year later, we revisit NetHack with agentic harnesses and the newest generation of models. We find that models still fail spectacularly to beat the game, raising some alarm bells: What makes NetHack fundamentally different from other game environments? Are current evaluation methods adequately capturing the capabilities of language models, or is the game presented in a way that is particularly inaccessible for agents? Or are there limitations in today's language models that prevent them from succeeding in such long-horizon, complex environments? 

In this blog, we investigate the failures of the current training stack and summarize meta patterns across research on this topic. We present a modernized harness for the game through Prime Intellect’s Verifiers package and an evaluation method with a refactored game engine to bring it up to standards with modern reinforcement learning and LLMs for games. 



# NetHack Introduction

NetHack—a 1987 fork of Hack, which was initially inspired by Rogue—is one of the oldest roguelike games. In the game, a player navigates a procedurally generated dungeon with a single life to retrieve the Amulet of Yendor and escape, battling the hundreds of unique monsters and navigating intricate traps along the way. With over 200 distinct items and 300 entities providing complex interactions, the human win-rate is approximately 1%, with the successes taking a median of 50 thousand moves to complete.

The game is played in an 80x21 character terminal interface, where the entire world is rendered in ASCII characters. Players interact using modified vim keybinds to move, attack, quaff potions, read scrolls, and manage an inventory system. Because the environment is procedurally generated, heavily relies on hidden state (e.g., unidentified items, hidden doors, and unseen monsters), and has unintuitive traps, NetHack requires an incredible ability to plan over long horizons, generalize to new settings, and navigate the text-based dungeon. 

LLMs have been most famously introduced to NetHack in the BALROG paper. BALROG introduced a new metric that estimated game-progress using two key metrics: experience level and dungeon level percentiles. Specifically, each metric measured the fraction of human-played games that beat NetHack given that they hit a certain experience or dungeon level. Then, the BALROG metric takes the maximum over these two percentiles. We use that as our primary metric for progress.

# Initial Agent Harness 
BALROG used fixed horizon and full action and full observation descriptions, with no active memory, as well as NetHack actions at a single action granularity. However, much progress over in RL for both games and for agents has been made by changing this basic setup. Encoding manipulations, full horizon with compaction, and action chunking via skills are now de facto settings for the most capable agents. As a result, we design a basic setup based on a previous paper, NetPlay, that retains the expressiveness of the original keys while also saving time up to 50 actions per LLM tool call. 

The reduced surface we call **np_core** exposes eight skills — the narrow, individually-debugged core of NetPlay's action layer. Each skill is a small program over the raw NetHack keys: `np_move_to`, for instance, runs A\* to a target tile and emits the whole run of movement keys in a single tool call. Anything not covered by a dedicated skill (eating, quaffing, wielding, answering a menu) is reachable through `np_press_key`, which answers NetHack's own prompts.

**NetPlay reduced skills**

| Skill | Arguments | What it does |
|---|---|---|
| `np_explore_level` | *(none)* | Auto-explores the level to reveal new rooms, corridors, and hidden doors; opens up to four doors it encounters. |
| `np_move_to` | `x`, `y` | Pathfinds (A\*) to tile `(x, y)`, emitting the full sequence of movement keys. |
| `np_melee_attack` | `x`, `y` | Pursues the monster at `(x, y)` and attacks it in melee until it is dead. |
| `np_kick` | `x`, `y` | Steps adjacent to `(x, y)` and kicks toward it (e.g. to force a locked door). |
| `np_press_key` | `key` | Presses a single key — answers NetHack's own prompts and menus (any letter, plus ESC/SPACE/ENTER). |
| `np_apply` | `item_letter` *(optional)* | Applies (uses) a tool from the inventory. |
| `np_rest` | `count` *(optional, default 5)* | Waits in place for `count` turns, or until something happens. |
| `np_pray` | *(none)* | Prays to your god for help, auto-confirming the "Are you sure?" prompt. |

Under the skills sit the raw NetHack action classes they compose. A single `np_move_to` might expand into a dozen movement keys; `np_kick` into a kick command plus a direction; `np_pray` into the extended `#pray` command plus a `y`.

**Raw NetHack actions**

| Raw action | Keys | What it does |
|---|---|---|
| Move | `h j k l` / `y u b n` | One step W/S/N/E, and the four diagonals; skills chain these into paths. |
| Melee | move into a monster | Walking into an adjacent hostile swings your wielded weapon. |
| Kick | `Ctrl-D` + direction | Kick in a direction — break down doors, etc. |
| Open | `o` + direction | Open an adjacent door. |
| Search | `s` | Search adjacent tiles for hidden doors and traps. |
| Apply | `a` + item | Use a tool from the inventory. |
| Pray | `#pray` | Pray to your god (a last-ditch rescue). |
| Wait | `.` | Do nothing for one turn (rest / heal). |
| Key press | any key | Answer a prompt/menu or issue any raw NetHack command. |


Below, we give an example demonstration in terms of LLM actions and number of in-game NetHack actions required to complete a navigation task.

<div class="game-embed" data-demo="opening"></div>

| # | Skill call (what the model emits) | Raw NetHack actions (what the engine runs) | Keys |
|---|---|---|---|
| 1 | `np_move_to(x=3, y=5)` | `k` | 1 |
| 2 | `np_move_to(x=15, y=9)` | `llllllllnnnn` | 12 |
| 3 | `np_move_to(x=27, y=10)` | `hbnllllulullllln` | 16 |
| | **3 skill calls** | | **29 keys** |

(Keys are vi-style: `h j k l` step W/S/N/E and `y u b n` the diagonals, so `np_move_to` is turning a target tile into the walk that reaches it.)

Specifically, the attribute that we deem important are expressivity. One common case is doing an action will prompt the game to ask the user for a confirmation, for example, whether to attack an enemy or not. We find that these are essential to the game due to innate mechanics based on navigation. For example, oftentimes, if a user is running past a monster, they will automatically try to attack it. However, this may not be the model's intention, so if this ever occurs in one of our skills, we break from the skill and the confirmation message is displayed to the model.

# Evaluations
With this setup, we create a custom port of NetHack 3.6.7 that enables flexibility in evaluations and game modes[^1]. During experimentation, we find that current models cannot beat the game even without partial observability when we provide full vision over the grid at any given time, so we dedicate our time to exploring what patterns and behaviors cause our models to fail with full observability. The default metric that we report is BALROG Score at a Fixed Expenditure of 200 LLM turns. 

TODO (Seth): Do I need justification here for why we don't have results without full visibility?

On our initial harness, we see high variance across five seeds, with a mean BALROG score of **3.06** and every game ending in death. Every score reported here is the real BALROG progression metric (the max over the dungeon-level and experience-level percentiles), reported ×100 as a 0–100%.

| Seed | BALROG | Max Dlvl | Max XL | Outcome |
|---|---|---|---|---|
| Seed 0 | 1.75 | 3 | 1 | died |
| Seed 1 | 4.85 | 7 | 3 | died |
| Seed 2 | 1.75 | 3 | 1 | died |
| Seed 3 | 2.12 | 4 | 2 | died |
| Seed 4 | 4.85 | 7 | 1 | died |
| **Mean** | **3.06** | **4.8** | **1.6** | **5/5 died** |

<div class="game-embed" data-demo="harness"></div>

Despite the poor performance, we see that the model reasons fairly reasonably about many aspects of the game, identifying monsters, whether monsters are appropriate to attack, and also what different objects in the game are. However, we see that all of them suffer from similar issues:

1. All seeds reach higher dungeon levels than experience levels — descent outpaces leveling in every game (e.g. seed 4 reached Dlvl 7 still at XL 1). The model dives faster than it grows strong enough to survive down there. This motivated the **E8a descent gate**, which surfaces the human-norm XP-for-depth line and asks the model to consolidate before diving.
2. **Reasoning is accurate but inert (PLANNING).** The model narrates the game well — it names monsters, judges which are safe to fight, identifies items, and even times prayer correctly (3 of 4 control prayers fired at critical HP) — yet this reasoning rarely changes the policy it then executes. That gap between good narration and unchanged behavior is the central question of the **E8 "does telling the model help?"** experiments (E8a descent gate, E8b prayer hint).
3. **Pathfinding stalls on dense maps (CONTROL/PERCEPTION).** When `np_move_to` cannot find a route it returns having moved nothing, and the model re-issues the same target instead of re-planning — seed 0 alone aimed `np_move_to(57,13)` at the downstairs nine times, three of them advancing zero squares. This route-failure mode motivated the **E8d density sweep**, which thins the map to separate route-finding from decision-making.
4. **Locked doors drain the run (CONTROL of execution).** With no dedicated unlock skill, the model falls back to `np_kick` and simply repeats it — seed 0 kicked the same door four times in a row. Door/kick friction motivated **E8c**, which hands the model a seed-matched dungeon with the doors already unlocked.


# Fixing Failure modes

## Short-term Rewards
We've observed that our LLM over-indexes into descending in the dungeon, but we haven't yet measured its pace against a human. Below, we extract human gameplay data from the NetHack Learning Dataset to plot our models against human gameplay. Notably, we actually descend with the same speed as the top 1-5% of humans! 

<!-- TODO: claude. add plots like in https://claude.ai/code/artifact/772d0d6c-3cfe-4886-8769-5f84876d3b40?org=e8e04b44-e81a-4556-9290-f8103fec7728, comparing the nethack reduced seeds to human gameplay. -->

To mitigate these short-term tendencies, we try to add scaffolding for the model to understand what typical, successful gameplay looks like. Each time the model tries to descend to the next level, we add a new confirmation panel that alerts the model to the average experience level a human would descend at. 

<!-- TODO: claude. Add widget to show confirmation insertion-->

<!-- TODO: claude. Add results and pairwise comparisons on each seed (to original). Also add widget to view these games-->

## Simplifying Path-finding 
Similarly, X% of tool calls in our initial set of evaluations resulted in no-ops due to navigation errors.<!-- TODO: claude. figure out what x% is --> As a result, we try two simplifications. First, we unlock all the doors, allowing navigation tool calls to go uninterrupted for longer durations; and 2) we rewire the dungeon map generation to create fewer dungeon rooms. 

<!-- TODO: claude. Add results and pairwise comparisons on each seed (to original). Also add widget to view these games-->

## 

# Harness Optimization
So far, we have only tested the ability to manually customize a harness. However, how well do models do in optimizing for their errors? In other words, is a model able to continually improve its performance by leveraging previous gameplay?



[^1] https://github.com/liujonathan24/NetHack. 





<!--GLM-5.2 driving NetPlay's published action surface, Prime Agent scaffold, the
whole level revealed. Across three tool-surface variants and 15 games it never
clears the early dungeon... -->

## How they die
<!-- taxonomy: underleveled dive → swarmed → mistimed prayer; reward is pure depth -->

## The games
See `e7_viewer.html` (open in a browser; no server needed).

## E8 — does telling the model help? (2026-08-19)

Four experiments against the NPCORE_v3 baseline (median 2.12, 5/5 died).
Viewers: `e8a_viewer.html` (descent gate), `e8b_viewer.html` (prayer hint),
`e8c_viewer.html` (unlocked doors, seed-matched), `e8d_viewer.html` (density
sweep). All E8 games replay **move-by-move** (per-step frames).

| Cell | Median | Deaths | Readout |
|---|---|---|---|
| E8a norm gate | 2.12 (=control) | 5/5 | 20 gates shown, 0 followed |
| E8b prayer hint | 6.96 (variance) | 5/5 | all prayers at full HP |
| E8c doors unlocked | 3.54 | 4/5 | kicks 18→3; only survivor of the round |
| E8d density 0.25/0.10/0.025 | 3.54/3.54/2.65 | 5/5 | route-fails 49→30→19→6, bal flat |

**The story:** verbal guidance is read and ignored — the model re-descends
straight through the human-norm line and prays at full HP despite being told
the heal band. Removing friction (doors, rooms) removes errors but not deaths.
24/25 rollouts died. Advice doesn't transfer to policy; the next lever is
scaffold-enforced constraints.

### E8 correction (traces re-read)

An earlier draft said the prayer hint was "read and ignored — all prayers at
full HP." That was a measurement bug: the pray-turn record's HP is *post*-heal.
The true pre-prayer HP was 2/16, 3/43, 5/16 — **all critical, all correctly
timed** — and the reasoning shows the model did the math ("43/7 ≈ 6.1, I'm at 3,
pray!"). Corrected story: prayer was already well-timed without the hint (3/4 in
control), so the hint had little to fix; the score gap is seed variance. And the
descent gate was *engaged with*, not ignored — on 7/20 gates the model reasoned
about the XP tradeoff and chose to dive anyway with a defensible argument. The
theme survives but softens: guidance is read and reasoned about, yet rarely
changes policy — advice, even correct advice the model understands, doesn't
reliably transfer to behavior.
