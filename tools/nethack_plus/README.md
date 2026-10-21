# nethack-plus

One command per NetHack LLM experiment on Prime. It runs the verifiers `nethack` env from
[residency-environments](https://github.com/RL-Residency/residency-environments) (pinned in
`pyproject.toml`) through `vf-eval`, one run per skill library, side by side.

```bash
uv tool install "nethack-plus @ git+https://github.com/liujonathan24/NetHack-hub.git@<commit>#subdirectory=tools/nethack_plus"
prime login            # or export PRIME_API_KEY

# 5 seeds, 1 run each, up to 15 explore attempts per run, both libraries, map revealed
nethack-plus explore -n 5 -r 1 -a 15 --skills netcore,codehack --full-vision

# the same games played once each, 15 times per seed
nethack-plus single -n 5 -r 15 --skills netcore,codehack --full-vision

nethack-plus status  runs/<name>     # live games: seed, attempt, calls, Dlvl, HP, idle time
nethack-plus summary runs/<name>     # per run, per seed (mean of reps), overall (mean of seed means), cost
nethack-plus resume  runs/<name>     # re-run missing or failed runs
```

It needs its own environment (`uv tool install`, or `uv sync` here): the Hub's own
`environments/nethack` package is also named `nethack` and pins an older verifiers.

## Modes

- `single`: one game per run.
- `explore`: checkpointed exploration (Prime-Agent-Explore). A run is up to `-a` attempts at one
  game. The game saves checkpoints as it plays (every `--every` calls, each new Dlvl, each XL
  gain). After each attempt the player's own model reads the checkpoint leaderboard and picks the
  checkpoint the next attempt resumes, with a short strategy directive, or ends the run. A run's
  result is its attempt with the highest `balrog_min`. `--chooser rule` (the previous attempt's
  last checkpoint, no directive) and `--chooser blind` (the model picks, the directive is replaced
  by "Continue playing.") are the control arms. The env's README describes the algorithm and where
  it differs from the Hub's machine.

## Flags

| Flag | Default | |
| --- | --- | --- |
| `-n`, `--seeds` | 5 | seeds from `--seed-base` (0) |
| `-r`, `--reps` | 1 | independent runs per seed |
| `-a`, `--attempts` | 10 | explore only: most attempts per run |
| `-m`, `--model` | `deepseek/deepseek-v4.1-flash-1017` | a Prime inference model; in explore it also chooses |
| `--skills` | `netcore` | `netcore`, `codehack`, or `netcore,codehack` |
| `--full-vision` | off | reveal the whole map (off: fog of war) |
| `--max-calls` | 500 | skill calls per game; in explore, per attempt |
| `--timeout` | 14400 | seconds per game; in explore, per attempt |
| `-c`, `--concurrency` | 20 | runs at once, split across libraries; each live game holds 2 Prime sandboxes |
| `--max-tokens` | 32768 | output cap per model turn |
| `--damage X`, `--xp X`, `--tune KNOB=X`, `--modify FIELD=N` | | game modifications; the prompt tells the model |
| `--character` | `Val-hum-neu-fem` | or `random` |
| `--every`, `--chooser` | 20, `model` | explore only |
| `-o`, `--out`, `--name` | `runs/`, generated | where the run goes |
| `--push` | off | upload to Prime Evals |
| `--estimate`, `--dry-run` | | print the estimate; write the configs without running |

Before it starts, a run prints the estimated inference cost (a token profile measured on
deepseek-v4.1-flash-1017 games, scaled by the model's listed prices; in explore an upper bound,
every attempt used) and the wallet balance. Nothing is gated on them. `summary` prices each run
from its own tokens, chooser calls included, and prints the wallet change beside it; the wallet is
the billed truth.

## A run directory

```
runs/<name>/
  manifest.json            the command, model, seeds, estimate, balance before, start time
  <library>/config.json    the vf-eval config
  <library>/vf/run/        the vf-eval run: traces.jsonl, configs/resolved/eval.json, logs/
  <library>/progress/      one line per skill call, per tool server (what `status` reads)
  <library>/archive/       explore: each run's checkpoints and ledger.jsonl
  <library>/vf-eval.log
```

## Versions

The env pin, the env's `DEFAULT_IMAGE` and the game image move together: the image refuses a
tool server whose game module differs from its own. A new image means a new env commit, then a
new pin here.
