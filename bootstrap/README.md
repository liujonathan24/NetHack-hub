# Moving to a new box

This box is disposable and has no persistent volume, so bring-up is a script,
not a remembered sequence.

```
bootstrap/bootstrap_new_box.sh
```

It clones both repos, runs `setup_sandbox.sh` (system deps, uv + Python 3.12,
engine submodule at the pinned SHA, the venv, prime-agent with its version
assertion), installs the Claude Code settings and memory files, then runs eight
checks and tells you what is still missing.

## Carry these two by hand

Everything else rebuilds. These do not.

**1. Prime credentials** — `~/.prime/config.json`, 494 bytes. Either copy it or
run `prime login --headless --plain`. Without it there is no inference.

**2. Prior run data** — not in git, and the evidence behind every published
number:

| directory | why it matters | size |
|---|---|---|
| `outputs/e10_baseline` | the v1 reference numbers (median 3.54 / mean 5.34 / 14 of 15 died / ceiling dl11) | 80 MB |
| `outputs/e11_gates` | the descent-gate comparison | 37 MB |
| `outputs/e13/` | continual-harness rounds | 86 MB |
| `outputs/e14_baseline/` | the v2 baseline | 16 MB |

`rsync` them into the new checkout's `outputs/`.

## Do not carry

`/tmp/vf-prime-agent-*` — the continual-harness stores. Every experiment is
designed to begin from an empty store, and `run_e13.sh` refuses to start against
a non-empty one. Copying one across machines would silently break the guarantee
that a run started from the base set of skills.

## What the checks catch

| check | why it is here |
|---|---|
| `prime-agent 0.3.3` | the harness **refuses to launch** on a version mismatch; if the installer has moved on, this is where you find out rather than mid-cell |
| `bubblewrap present` | `sandbox = true` fails loudly without it, and the arm will not run unconfined |
| `tomllib` | the tier resolver needs Python 3.11+; `PY_BIN` silently falls back to the system 3.10 and `TOOL_TIER` dies with a bare `ModuleNotFoundError` |
| engine `.so` fresh | the preflight refuses a stale or off-pin shared object — a source checkout does **not** rebuild it, and a mismatch silently corrupts a whole sweep |
| tier registry resolves | catches a malformed `tool_tiers.toml` before a cell inherits the wrong contract |
| harness importable | `nethack_prime_agent` is an **editable install pointing at the main checkout** — a worktree's harness changes are invisible without `PYTHONPATH`, and the eval CLI rejects new `--harness.*` flags as "Extra inputs are not permitted", which reads like a typo rather than a stale package |

## Sizing the new box

The current one is 62 GB / 16 CPUs. Concurrency is bounded by the NetHack tool
server's memory, one process per rollout: ~0.3 GB early in a game, and a single
server was observed at 7.4 GB about 24 minutes into a 200-call E9 rollout. Five
concurrent rollouts is comfortable; ten would peak near 74 GB on that
observation and OOM.

What actually grows has **not** been established — an early hypothesis
(`record_step_frames`) was wrong: `TurnRecorder` is per-turn and its frames are
written straight into `turns/*.ndjson`, so they never accumulate. Before sizing
on the 7.4 GB figure, get the growth curve; if it turns out to be a leak rather
than legitimate state, fixing it buys the parallelism without paying for RAM.

## Claude Code config

`claude-config/` carries the portable parts only: `settings.json`,
`settings.local.json`, and the memory files under `projects/-root/memory/`.

Deliberately excluded: `.credentials.json` (a secret), and `projects/`,
`file-history/`, `cache/`, `plugins/` (~100 MB of session transcripts and caches
that mean nothing on a new machine).

The memory files are the accumulated operational gotchas — the daemon-wedge
recipe, the editable-install trap, git authorship, the worktree rule. They cost
more to rediscover than they look.
