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

<!-- TODO: claude. Show the table of NetPlay reduced actions v3 and how each skill works. Also add a table of nethack actions and what they do.-->


Below, we give an example demonstration in terms of LLM actions and number of in-game NetHack actions required to complete a navigation task.} <!-- TODO: claude. Navigate from 1 room to another with netplay skills and show the number of moves on the left side. Should be one viewer with two rows of actions below one w/ skills, other with nethack letters-->

Specifically, the attribute that we deem important are expressivity. One common case is doing an action will prompt the game to ask the user for a confirmation, for example, whether to attack an enemy or not. We find that these are essential to the game due to innate mechanics based on navigation. For example, oftentimes, if a user is running past a monster, they will automatically try to attack it. However, this may not be the model's intention, so if this ever occurs in one of our skills, we break from the skill and the confirmation message is displayed to the model.

# Evaluations
With this setup, we create a custom port of NetHack 3.6.7 that enables flexibility in evaluations and game modes[^1]. During experimentation, we find that current models cannot beat the game even without partial observability when we provide full vision over the grid at any given time, so we dedicate our time to exploring what patterns and behaviors cause our models to fail with full observability. The default metric that we report is BALROG Score at a Fixed Expenditure of 200 LLM turns. 

TODO (Seth): Do I need justification here for why we don't have results without full visibility?

On our initial harness, we see high variance in results with an overall median of X and an average of Y BALROG score. 
<!-- TODO: Claude. Insert results (X and Y) and the table of results and the html widget to view games with the NetHack-->

Despite the poor performance, we see that the model reasons fairly reasonably about the game, identifying monsters, whether monsters are appropriate to attack, and also what different objects in the game are. However, we see that all of them suffer from similar issues: 
1. All seeds reach higher dungeon levels than experience levels
2. <!-- TODO: Claude. Insert our observations that led to new experiments -->









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
