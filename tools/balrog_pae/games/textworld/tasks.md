# TextWorld in BALROG — what the benchmark actually runs

Everything here is read off the installed BALROG checkout
(`/root/nld/gen-pae/BALROG`, commit `b7afe79`) and measured in
`/root/nld/gen-pae/.venv-balrog` (textworld 1.6.2rc1, jericho 3.3.1), not from
the paper.

## 1. The three games

`balrog/config/config.yaml: tasks.textworld_tasks`

| task | game files | format | backend | `max_score` | progression | frontier best (BALROG naive) |
|---|---|---|---|--:|---|--:|
| `treasure_hunter` | 25 × `tw_games/treasure_hunter/seed_*.ulx` | Glulx | `GitGlulxEnv` | **1** | `score/1` → **binary**: found the treasure or not | 40.0 % (gpt-5.6-sol-max, gpt-6-astra-max) |
| `the_cooking_game` | 25 × `tw_games/the_cooking_game/cooking_item5_seed_*.z8` | Z-machine | `JerichoEnv` | **17** | `score/17` → **dense**, 17 graded sub-goals | 50.6 % (gpt-5.6-sol-max) |
| `coin_collector` | 25 × `tw_games/coin_collector/level_220_seed_*.ulx` | Glulx | `GitGlulxEnv` | **1** | `score/1` → **binary**: picked the coin up or not | 100 % (gpt-5.6-sol-max, gpt-6-astra-max) |

Count: **3 tasks**, 25 pre-generated games each (BALROG's default is 10 episodes
per task, so the default sweep touches games 1..10 of each).

Difficulty as generated (BALROG `docs/envs/textworld.md`): treasure_hunter =
20-room maze at difficulty 30 with locked doors/containers and no solution
description, filtered so the optimum is > 20 steps; the_cooking_game = 13 rooms,
up to 5 ingredients, all challenge options on; coin_collector = 40 rooms with
distractors, optimal path 20 steps, no solution description.

## 2. Episode step cap

**80 steps for all three games.** `envs.textworld_kwargs.max_episode_steps: 80`
is the only cap that is used: `TextWorldFactory.initialize` passes it to
`textworld.gym.register_game`, and `EnvWrapper.max_steps` → `TextWorldWrapper.max_steps`
returns it, which is what BALROG's evaluator uses as the episode horizon.

Caveat worth stating in the paper: the *instruction prompts* in
`balrog/environments/textworld/__init__.py` tell the agent "You have 40 steps"
(treasure_hunter), "80 steps" (the_cooking_game) and "25 steps"
(coin_collector). Those numbers are prompt text only; the real cap is 80 for all
three. We leave the prompts untouched (they are part of stock BALROG).

Truncation at the cap arrives as `terminated=True, truncated=False` (the
gym-v21 compatibility wrapper folds `done` into `terminated`), and
`textworld.envs.wrappers.Limit` is the wrapper that raises it.

## 3. How `get_stats()["progression"]` is defined

`balrog/environments/textworld/base.py`:

```python
def step(self, action):
    obs, reward, done, info = self.env.step(action)
    ...
    if done:
        self.progression = max(info["score"] / info["max_score"], 1.0 if info["won"] else 0.0)
    return ...

def get_stats(self):
    return {"progression": self.progression}
```

So for every game progression is **the score fraction at the end of the
episode**, floored by the win flag. Two consequences that matter for us:

1. **It is 0.0 for the whole episode and is written exactly once, when `done`
   first becomes true.** A mid-episode read (which is what a checkpointed loop
   does) always returns 0.0. The PAE loop keeps reporting `progression()` as
   the metric, and the TextWorld adapter exposes the live `score/max_score` as
   the dense `aux` signal for the orchestrator's ledger only — it drives
   nothing (not the checkpoint trigger, not the plateau guard, not the reported
   number).
2. For `treasure_hunter` and `coin_collector`, `max_score == 1`, so progression
   is binary; only `the_cooking_game` gives partial credit (`k/17`). The
   published per-task means confirm it: coin_collector comes out at exactly
   0 % or 100 %, treasure_hunter at multiples of 10 % over 10 episodes, while
   the_cooking_game lands on 23.53 % = 4/17 and 34.71 % ≈ 5.9/17.

## 4. Frontier headroom

From `/root/overleaf/balrog-experiments/submissions/LLM/*/textworld/textworld_summary.json`
(BALROG naive agent, 10 episodes/task):

| task | gpt-5.6-luna-max | gpt-5.6-sol-max | gpt-6-astra-max | headroom to 100 % |
|---|--:|--:|--:|--:|
| treasure_hunter | 20.0 ± 12.6 | 40.0 ± 15.5 | 40.0 ± 15.5 | **60 pts** |
| the_cooking_game | 34.7 ± 4.9 | 50.6 ± 10.4 | 23.5 ± 0.0 | **49 pts** |
| coin_collector | 0.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | **0 pts at the top, 100 for weaker models** |
| textworld overall | 18.2 ± 5.2 | 63.5 ± 7.8 | 54.5 ± 7.9 | 36 pts |
| average steps/episode | 63.8 | 51.2 | 51.6 | |

`claude-opus-4.5` and `gemini-3.1-pro-thinking` submissions in that directory
contain NLE only, so there are no TextWorld numbers for them.

Read-out for the panel: **the_cooking_game is the game to lead with** — dense
17-point progression, ~50 pts of headroom even for the best frontier model, and
it is the one game with true save/restore. **treasure_hunter** has the most
headroom but a binary metric, so a run of 5 seeds moves in 20-pt steps.
**coin_collector is saturated at the frontier** (100 % for two of three models):
it is only informative for a weaker player such as GLM-5.2, where luna-max's
0 % shows the task can still fail badly.

## 5. Seeding — and the trap

`TextWorldFactory.get_textworld_env(task, seed=...)` selects the *game file*:

```python
if seed is not None:
    env_id = self.env_ids[task][seed % len(self.env_ids[task])]   # 25 games
else:
    self.count[task] += 1
    env_id = self.env_ids[task][self.count[task] % len(self.env_ids[task])]
```

BALROG's shipped config has `envs.env_kwargs.seed: null`, so the evaluator takes
the **second** branch: a per-process counter. In a fresh process that makes
"seed 0" and "seed 1" the *same game* — the seed the evaluator passes to
`env.reset(seed=...)` only reaches `TextworldGymEnv.seed()`, which seeds an RNG
these deterministic games never read.

`TextWorldAdapter` therefore binds the game file to the episode seed: it builds
(and, if the seed changes, rebuilds) the env with `envs.env_kwargs.seed = seed`,
which is the documented, supported first branch above. Seeds 0–4 map to game
files:

| seed | treasure_hunter | the_cooking_game | coin_collector |
|--:|---|---|---|
| 0 | `seed_10033.ulx` | `cooking_item5_seed_10980.z8` | `level_220_seed_100.ulx` |
| 1 | `seed_10915.ulx` | `cooking_item5_seed_11996.z8` | `level_220_seed_1171.ulx` |
| 2 | `seed_14115.ulx` | `cooking_item5_seed_12274.z8` | `level_220_seed_12089.ulx` |
| 3 | `seed_16404.ulx` | `cooking_item5_seed_12896.z8` | `level_220_seed_15858.ulx` |
| 4 | `seed_18762.ulx` | `cooking_item5_seed_13009.z8` | `level_220_seed_16706.ulx` |

The games carry **no run-time randomness**: given a game file, the command
sequence determines the state. That is what makes the replay restore exact.

## 6. Action space and invalid actions

`TextWorldWrapper.language_action_space` is `AlwaysTrue()` — `x in space` is
`True` for anything. So `EnvWrapper.check_action_validity` never rewrites the
model's completion, `failed_candidates` stays empty, and BALROG's
"Your previous output did not contain a valid action" feedback **never fires
for TextWorld**. Whatever the model emits is passed verbatim to the TextWorld
parser, and the *parser's* error message is the feedback the agent sees:

* unknown verb → `That's not a verb I recognise.`
* unknown noun → `You can't see any such thing.`

Both are part of the restored state and are asserted identical across restore in
`verify_restore.py` (`restore_verification.json`). The adapter keeps the
invalid-action rewrite in its replay path anyway, so it would stay correct if
BALROG ever constrained the TextWorld action space.

## 7. Env stack

```
EnvWrapper > GymV21CompatibilityV0 > TextWorldWrapper > TextworldGymEnv
  > SyncBatchEnv > Filter > Limit > GenericEnvironment > TWInform7
  > StateTracking > Inform7Data > GameData > { JerichoEnv | GitGlulxEnv }
```

Observation is `{"text": {"long_term_context": <parser output>,
"short_term_context": ""}, "image": None}`. `TextWorldWrapper.filter_objective`
strips the objective paragraph out of the text. `info` keys on `step`:
`score`, `max_score`, `won`, `objective`, `description`. `reset` returns an
**empty** info dict (the gym-v21 shim returns `obs, {}`), so `max_score` is only
known after the first step.

## 8. Restore method per game

| game | method | why |
|---|---|---|
| `the_cooking_game` | native: `jericho.FrotzEnv.get_state()/set_state()` + deepcopy of every Python wrapper's mutable state (`Limit.nb_steps`, `StateTracking`, `Inform7Data`, `GameData`, `SyncBatchEnv.last`, `TextworldGymEnv.last_commands/obs`, `TextWorldWrapper.progression`) | `JerichoEnv` exports Z-machine state |
| `treasure_hunter`, `coin_collector` | replay: reset the seed-bound game + re-issue the recorded command prefix | `GitGlulxEnv` is a `git-glulx-ml` subprocess over a socket: `copy()` → `NotImplementedError`, in-game `save` prompts for a filename and kills the interpreter, `undo` is one turn only |

`SyncBatchEnv.last` is a done-latch that short-circuits `step()` once the
episode finished; omitting it from the snapshot silently truncates every
restored branch to one step.

Measured cost (seed 0, `restore_verification.json`): snapshot 36 ms / 875 KB
for the Jericho path; replay restore 13–55 ms for a 10–40 step prefix
(reset 43–266 ms amortised, ~0.5 ms/command). Both are negligible next to one
LLM call.
