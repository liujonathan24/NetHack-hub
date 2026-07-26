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
