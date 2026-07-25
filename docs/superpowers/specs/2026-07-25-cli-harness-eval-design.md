# CLI-agent harness comparison — design

**Date:** 2026-07-25. **Author:** Jonathan Liu. **Repo:** NetHack-hub, branch `exp/cli-harness-eval`.
**Status:** design, pre-implementation.

---

## 1 · Question

Exp1 held the harness fixed and varied the observation encoding. This experiment inverts it:
**holding the game, the action surface, the seeds, and the model fixed, does a general-purpose
coding-agent scaffold (Codex, Prime Agent) play NetHack better than the purpose-built harness?**

The unit under test is the *scaffold*: its loop, its context management, its planning, its use of
a filesystem workspace. Not the model — that is pinned.

**Primary metric:** Depth Score (`max_dlvl_reached`). **Secondary:** real BALROG progression,
alive-at-cap, death %, tokens/turn, `turns_used`.

## 2 · What is held fixed

| factor | value | mechanism |
|---|---|---|
| model | `z-ai/glm-5.2` | verifiers v1 intercepted endpoint, identical for every arm |
| taskset | `nethack_v1.load_taskset` | one taskset, all arms |
| seeds | 0–15, pinned, identical per arm | paired comparison |
| character | `Val-hum-neu-fem` | taskset config |
| tier | `full_nle`, uncapped | ends on death or the call cap |
| action surface | `skill_set="netplay"` (15 skills) | same registry, same gate |
| budget | 150 skill calls | toolset-side counter (§5) |
| affordances | strategy primer, wiki, memory | prompt+tools (arm 0) vs files (CLI arms), §6 |

Only the **harness** varies.

## 3 · Arms

| arm | harness | status |
|---|---|---|
| 0 — control | `nethack_v1.NetHackHarness` | exists |
| 1 | Codex (`CLIHarness` built-in) | needs verifiers > 0.1.14 |
| 2 | Prime Agent (external plugin) | needs PR #1985 loader, vendored |

Claude Code and Cursor are deferred; they slot in as arms 3–4 with no redesign.

## 4 · Architecture

```
                  ┌─ Harness: NetHackHarness (native tools)   ← arm 0, control
Taskset(nethack)  ├─ Harness: Codex        (CLIHarness)
  seeds 0–15      └─ Harness: PrimeAgent   (external plugin)
  Val-hum-neu-fem              │
                               ├── toolset: NetHack Toolset ─► MCP (HTTP) ─► engine
                               ├── endpoint: intercepted ────► GLM 5.2
                               └── sandbox: seeded workspace  (AGENTS.md, wiki/, memory/)
```

Everything shared is shared *by construction* in one `EvalConfig`, rather than by discipline in a
bash wrapper — the improvement over `encoding_eval/launch_cell.sh`.

MCP transport is **streamable HTTP**, not stdio: `docs/MCP_INTEGRATIONS.md:38` states Prime Agent
supports only `"http"` servers. Codex supports HTTP (`codex mcp add --url --bearer-token-env-var`),
which matches Prime Agent's `bearerTokenEnvVar`. One server, one bearer token, both arms.
The engine runs **outside** the agent sandbox; the sandbox reaches it over the network.

## 5 · The dispatch split (the one substantial change)

`_build_toolset(v0env)` currently returns `vf.Toolset(tools=list(v0env.tools), …)`, but
`v0env.tools` are **schema-only stubs** — dispatch lives in `env_response`, which is why
`update_tool_args` is a no-op (`nethack.py:1157`). `NetHackHarness` calls `env_response` each turn
to apply the *previous* tool call.

That is fine when we own the harness and fatal when we do not: an MCP tool call from Codex must be
self-contained — execute the skill, step the engine, return the observation.

**Change:** `_build_toolset(v0env, *, self_dispatch: bool = False)`.

- `self_dispatch=False` — today's behavior, byte-identical. Arm 0.
- `self_dispatch=True` — each of the 15 netplay tools runs the dispatch half of `env_response`
  (gate → `skill_registry.call` → engine steps → menu/`--More--` drain → outcome detection →
  bookkeeping → trace write) and returns the rendered observation as its result. Arms 1–2.

The render half is unchanged and shared. One code path, one flag, one gate, one trace writer.

`obs_mode: "push" | "on_demand"` rides alongside: `push` returns the observation in every result
(default); `on_demand` returns terse feedback and gates the map behind an explicit `look()`. This
reserves the visibility sub-experiment without a later rewrite.

## 6 · Seeded workspace (CLI arms)

Delivered via `CLIHarness(files=…, dirs=…)`. Each item is the filesystem counterpart of an
affordance arm 0 gets through prompt and tools, so the arms are **capability-matched**:

| arm 0 | CLI arms |
|---|---|
| `SYSTEM_PROMPT` (`rendering.py:14`) | `AGENTS.md` — verbatim port |
| `wiki_lookup` / `wiki_search` | `wiki/` — 105 pages exploded from `snapshot.json`, read-only |
| `Journal` (objective + notes) | `memory/objective.md` + writable `memory/` |
| per-turn rendered observation | MCP tool results |

**Rules.** `memory/` is **wiped per rollout** — arm 0's `Journal` is fresh per rollout
(`nethack.py:495`), so persistence would hand the CLI arms cross-seed learning and read as a
scaffold win. No `map/` is seeded: it duplicates the observation channel, collides with
`obs_mode=on_demand`, and *whether a scaffold builds its own map notes* is a finding we would
destroy by pre-seeding one. The workspace must not contain the repo, engine source, or level files.

Per-CLI filename differences (`AGENTS.md` vs `CLAUDE.md` vs `.cursor/rules`) are written from one
source by the workspace builder.

## 7 · Referee and termination

The 150-call cap lives in the **toolset**, not `max_turns` — `max_turns` only binds harnesses we
control. The toolset counts skill calls, refuses past 150, and ends the episode. Uniform across
arms and un-gameable regardless of a CLI's internal loop.

**Early stop counts as terminal** (decided): if an agent quits at call 40, that is its result.
`turns_used` is reported so a loss by quitting is visible as such rather than silent.

## 8 · Scoring

Reuses `tools/encoding_eval/aggregate_run.py`: depth from traces (`max_dlvl_reached`), real BALROG
progression (`nethack_harness/prompt/balrog.py`, validated against the DL10/XL6 = 12.56% anchor),
alive@cap, death %, tokens/turn — plus `turns_used`. Paired per-seed deltas and an exact sign test
across the 16 seeds, as in exp1 §3.1; marginal SEs at n=16 are wide and must not be quoted alone.

## 9 · Risks

1. **The dispatch split touches the path exp1 depends on.** Mitigated by the golden test in §10.
2. **Prime Agent's tool surface differs in kind.** Per `docs/MCP_INTEGRATIONS.md`, MCP tools are not
   agent tools — they are a Python skill imported into an IPython kernel, called as
   `await nethack.explore_and_descend(...)`. Prime Agent writes *Python* to drive the game while
   Codex emits tool calls. That is a real scaffold difference and the writeup must say so rather
   than presenting the arms as identical-but-for-the-loop.
3. **GLM 5.2 billing.** `z-ai/glm-5.2` sits under `pinference-glm` in `configs/endpoints.toml`,
   which carries **no** `X-Prime-Team-ID`; only `prime-team` does, and it lists Gemini only. As
   written GLM bills the personal balance ($0) → `insufficient_funds`, which surfaces as a *hang*
   (exp1 confound #1). The team header must be set once, centrally, before any run.
4. **Out-of-band state reads.** CLI arms keep native shell as a thinking aid but must not reach the
   engine or level files. Enforced by sandbox scope; verified by trace audit.
5. **PR #1985 is OPEN.** Loader patch vendored (decided), so the Prime Agent arm does not wait on
   upstream merge. Revisit when it lands.
6. **No comparability to run2.** run2 was Gemini 3 Flash; this series is GLM 5.2. Arm 0 is the only
   baseline, and it must be run — there is no borrowing exp1's numbers.

## 10 · Testing

- **Golden parity:** arm 0 with `self_dispatch=False` reproduces a recorded run2 rollout
  turn-for-turn.
- **Dispatch equivalence:** the same skill sequence through `self_dispatch=True` and `False`
  yields identical engine state and identical trace fields.
- **Gate:** `tools/encoding_eval/_verify_gate.py` passes on both paths — `move` executed = 0.
- **Smoke per arm:** n=1, ~20 calls, verifying the arm reached the engine, the workspace seeded,
  the observation rendered, and the counter incremented — the exp1 discipline that caught
  silently-wasted runs.

## 11 · Layout

```
NetHack-hub/
  environments/nethack/nethack_v1.py        # _build_toolset gains self_dispatch, obs_mode
  harnesses/nethack-prime-agent/            # installable distribution, NOT a loose module:
    pyproject.toml                          #   name = "nethack-prime-agent"
    nethack_prime_agent/__init__.py         #   __all__ exports exactly one Harness subclass
  tools/cli_harness_eval/
    launch_cell.sh          # one arm, all factors pinned
    configs/{control,codex,prime_agent}.toml
    workspace.py            # AGENTS.md + wiki/ + memory/ builder
    aggregate_run.py        # reuses encoding_eval aggregation
  vendor/verifiers_1985/    # vendored external-harness loader patch
  outputs/cli_harness_eval/<run>/<arm>/
```

## 12 · Open items

- Verify Prime Agent's `mcpServers` + skill-package path end-to-end against a live HTTP MCP server.
- Confirm the exact verifiers version that ships the Codex harness, and pin it.
- Resolve the GLM 5.2 team-billing route before the first run (§9.3).
