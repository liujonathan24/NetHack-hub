# NetHack-Hub

The **verifiers / Prime Intellect environment + LLM harness + research/eval
tooling** for training and evaluating language-model agents on NetHack.


## Running experiments

Experiments on Prime run the verifiers `nethack` env from
[residency-environments](https://github.com/RL-Residency/residency-environments/tree/adffc89/environments/nethack)
(verifiers 0.4). There are two ways to run it. Both play the same game in
the same Prime boxes, so their results are comparable.

```
nethack-plus explore -n 5 ...        vf-eval nethack --env.taskset.* ...
        │  writes one vf-eval config         │
        │  per skill library, runs them      │
        └───────────────┬────────────────────┘
                        ▼
         verifiers `nethack` env (residency-environments)
           mode = single | explore
                        ▼
         Prime inference  +  Prime boxes running the nethack-play image
```

Both need Prime credentials: `prime login` or `$PRIME_API_KEY`.

### Option 1: the `nethack-plus` CLI

Use this for experiments. It pins the env and the game image. It runs
the skill libraries side by side and prints the estimated cost and the
wallet balance before it starts. Afterwards it reports per-seed and overall
results and the cost. See [`tools/nethack_plus/README.md`](tools/nethack_plus/README.md).

```bash
uv tool install "nethack-plus @ git+https://github.com/liujonathan24/NetHack-hub.git#subdirectory=tools/nethack_plus"

# 5 seeds, 1 run each, up to 15 explore attempts per run (3 at a time), both libraries, map revealed
nethack-plus explore -n 5 -r 1 -a 15 -p 3 --skills netcore,codehack --full-vision

# the same games played once each, 10 runs per seed
nethack-plus single -n 5 -r 10 --skills netcore,codehack --full-vision

nethack-plus status  runs/<name>     # live games
nethack-plus summary runs/<name>     # per run, per seed, overall, cost
```

The default model is `deepseek/deepseek-v4.1-flash-1017`; change it with `-m`.

### Option 2: verifiers directly

Use this for one-off runs, for other harnesses, or for settings the CLI has
no flag for. Every taskset field is documented in the env's README
(`num_seeds`, `max_calls`, `tune`, `skill_library`, `mode`, `attempts`,
`parallel`, `chooser`, ...).

```bash
uv pip install "nethack @ git+https://github.com/RL-Residency/residency-environments.git@adffc89#subdirectory=environments/nethack"

# single games
uv run vf-eval nethack \
  --env.agent.harness.id nethack --env.agent.runtime.type prime \
  -m deepseek/deepseek-v4.1-flash-1017 -n 5 -r 1 \
  --env.taskset.max-calls 500 --env.taskset.skill-library netcore

# explore: up to 15 attempts per run, 3 at a time
uv run vf-eval nethack \
  --env.agent.harness.id nethack --env.agent.runtime.type prime \
  -m deepseek/deepseek-v4.1-flash-1017 -n 5 -r 1 \
  --env.taskset.mode explore --env.taskset.attempts 15 --env.taskset.parallel 3 \
  --env.max-concurrent-agents 3
```

The vision defaults are opposite. The env's default `tune` is
`{"reveal_map": 1.0}`, which reveals the whole map, and
`--env.taskset.tune '{}'` turns fog of war back on. The CLI's default is
fog of war, and `--full-vision` reveals the map. Any CLI run can be repeated
with plain vf-eval: each run directory holds the config it ran,
`runs/<name>/<library>/config.json`, and
`vf-eval @ runs/<name>/<library>/config.json` runs it again.

## Dependencies

This repo depends on the NetHack engine, the repo
[`NetHack-engine`](https://github.com/liujonathan24/NetHack-engine), and
imports it under the `nethack_core` namespace. 

```
NetHack-engine  (nethack_core)  <-- external dependency (the fork engine + ctypes binding)
        ▲
        │ imports
NetHack-Hub     (this repo: environments/nethack, approaches/, tools/, docs/)
```

The env sources `nethack_core`, the default set of skills, from the engine package as well.

## The default task

The env runs the **standard full NetHack ascension game** (`nethack.GameSpec` /
`FULL_GAME_SPEC`). `nethack_harness/curriculum/` is kept as an empty
seam for a future six-floor "primitives" curriculum that depends on unmerged
fork hooks.

## Prime Hub deploy

The env deploys to the Prime Hub as a wheel that carries the engine `.so` + data
files (`hatch_build.py` build hook + the `artifacts` / `force-include` config in
`environments/nethack/pyproject.toml`).
