# Task 13 acceptance artifact: a Claude Code agent playing NetHack over MCP

These three files are the retained evidence for the acceptance gate of the whole CLI-harness
experiment — the first rollout in which an external CLI coding agent drove the NetHack engine
through the MCP toolset. Committed because a gate that only exists in a scratch directory is
not a gate.

| file | what it is |
|---|---|
| `task13_claude_code_seed0.config.toml` | the fully resolved run config, exactly as `eval` wrote it |
| `task13_claude_code_seed0.traces.jsonl` | the verifiers v1 `Trace` (one line): message graph, advertised tools, rewards, metrics, stop condition |
| `task13_claude_code_seed0.turns.ndjson` | the env's own per-turn record (20 lines): `raw_grid`, `status`, the rendered user message, `tool_calls`, `reward`, `dlvl`, `hp` |

## Provenance

`eval` on verifiers 0.2.1, `--taskset.id nethack_v1 --harness.id claude_code`, Claude Code
**2.1.214** (the harness's own pinned install), `z-ai/glm-5.2` via Prime Inference, subprocess
runtime, seed 0, `Val-hum-neu-fem`, `task_spec="full_nle"`, `skill_set="netplay"`,
`max_skill_calls=20`. Run 2026-07-26.

## What it shows

* **21 `mcp__nethack__*` tool calls, 0 non-MCP tool calls** — `explore_and_descend` ×8,
  `move_to` ×5, `search` ×3, `eat` ×2, `pickup` ×1, `attack` ×1, `pray` ×1.
* `trace.tools` lists 18 `mcp__nethack__*` tools; **`mcp__nethack__move` is absent** (netplay
  gate) and `mcp__nethack__explore_and_descend` is present.
* `metrics`: `skill_calls = 20`, `budget_exhausted = 1`, `moves_executed = 0`,
  `max_dlvl_reached = 3`, `descent_count = 2`.
* `stop_condition = "call_budget_exhausted"` — the toolset-side referee ended the episode, and
  it did so by counting over the HTTP `/state` channel.
* `rewards`: `descent_reward = 20.0`, `scout_reward = 0.426` (**CLI-arm weights** — see the
  cross-arm weight caveat in `../configs/README.md`; do not compare this to a control-arm
  `reward`).
* Tool results are the full rendered observation (`=== JOURNAL ===` / `=== MAP ===` /
  `=== STATUS ===` / `=== INVENTORY ===` / `=== ADJACENT ===` / `=== HINT ===`).

## Redactions

Only three strings were changed, all in the config file: the funded Prime team id, and two
absolute scratch paths (`output_dir`, `trace_dir`). The two data files are **verbatim** — they
were scanned for the API key, `pit_` prefixes, bearer tokens, the team id and user paths, and
contain none. The only path in either is the ephemeral runtime workdir (`/tmp/<trace-id>`).

## Regenerating

```bash
ENG=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness
PYTHONPATH="$ENG:$PWD:$PWD/environments/nethack" \
  .venv-cli-eval/bin/eval @ tools/cli_harness_eval/configs/claude_code.toml \
    --taskset.max-skill-calls 20 --num-tasks 1 -o outputs/acceptance-rerun
```

A rerun will not be byte-identical: the model is sampled, so the agent's choices differ. What
must reproduce is the structure — MCP-only tool calls, `move` absent, `skill_calls` reaching
the cap, `moves_executed = 0`.

---

# Task 10 acceptance artifact: a Prime Agent agent playing NetHack over MCP

The arm-2 counterpart. Same game, same tool server, same referee — but the agent reaches it as
a **Python skill inside its IPython kernel**, not as agent tools.

| file | what it is |
|---|---|
| `task10_prime_agent_seed0.config.toml` | the fully resolved run config, exactly as `eval` wrote it |
| `task10_prime_agent_seed0.traces.jsonl` | the verifiers v1 `Trace` (one line) |
| `task10_prime_agent_seed0.turns.ndjson` | the env's per-turn record (12 lines — one per executed skill) |

## Provenance

`eval` on verifiers 0.2.1, `--harness.id nethack-prime-agent` (the external plugin in
`harnesses/nethack-prime-agent/`), **prime-agent 0.3.3** (an npm global; this harness verifies
it rather than installing it), `z-ai/glm-5.2` through verifiers' interception, subprocess
runtime, seed 0, `Val-hum-neu-fem`, `task_spec="full_nle"`, `skill_set="netplay"`,
`max_skill_calls=12`. Run 2026-07-26.

## What it shows

* **`trace.tools` is a single tool, `ipython`** — the structural difference from arm 1, whose
  trace lists 18 `mcp__nethack__*` tools. Every NetHack call happens *inside* Python, so the
  per-skill breakdown is not in the trace; `metrics.skill_calls` and the tool server's own log
  are where it lives. **Task 11 must not count arm-2 skills from the node graph.**
* `metrics`: `skill_calls = 12` (the cap), `budget_exhausted = 1`, `moves_executed = 0`,
  `max_dlvl_reached = 2`, `descent_count = 1`.
* `stop_condition = "call_budget_exhausted"`, `errors: []` — the toolset-side referee ended the
  episode, counting over the HTTP `/state` channel.
* **18 model turns for 12 executed skills**, versus arm 1's ~1 skill per turn: the agent spent
  turns on discovery (`await nethack.list_tools()`, reading `SKILL.md`) that arm 1 gets free.
* The netplay gate held in a new place. The agent tried `await nethack.move(direction='N')` and
  `McpIntegration.__getattr__` refused it client-side — `'nethack' has no tool 'move'.
  Available: ...` — so the call never reached the wire, let alone the engine
  (`moves_executed = 0`).
* Tool results are the full rendered observation (`=== JOURNAL ===` / `=== MAP ===` /
  `=== STATUS ===` / `=== INVENTORY ===` / `=== ADJACENT ===`), printed by the agent's own code.

## Redactions

The funded Prime team id, and every occurrence of the two user path prefixes
(`/scratch/gpfs/<user>`, `/home/<user>`) → `<REDACTED-PATH>`. Unlike the Task 13 artifact, the
data files needed path redaction too: Prime Agent's system prompt lists the absolute
`.../node_modules/prime-agent/dist/skills/*/SKILL.md` location of every bundled skill. Both
files still parse as JSON/NDJSON. No API key, `pit_` prefix or bearer token appears in any of
the three (checked before writing).

## Regenerating

```bash
ENG=/path/to/NetHackHarness
PATH="<node bin>:<uv bin>:$PATH" \
PYTHONPATH="$ENG:$PWD:$PWD/environments/nethack" \
  .venv-cli-eval/bin/eval @ tools/cli_harness_eval/configs/prime_agent.toml \
    --taskset.max-skill-calls 12 --num-tasks 1 \
    -o outputs/acceptance-rerun \
    --taskset.trace-dir "$PWD/outputs/acceptance-rerun/turns"
```

**`--taskset.trace-dir` must be ABSOLUTE.** A relative `trace_dir` is resolved in the *tool
server's* runtime workdir (`/tmp/vf-<id>`), which is deleted at teardown, so the per-turn NDJSON
silently vanishes. Measured: identical runs with a relative and an absolute `trace_dir` produced
no file and 12 lines respectively. `control.toml`, `claude_code.toml` and `prime_agent.toml` all
carry *relative* `trace_dir` values, so this affects every arm.

Also clear `/tmp/vf-prime-agent/agent-*` between runs, and make sure no stale Prime Agent daemon
is listening (`prime-agent doctor`); a dead supervisor at the shared socket fails a fresh rollout
with `DaemonSocketClosedError`.
