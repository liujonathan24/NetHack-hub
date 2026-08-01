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
| budget | 150 skill calls (**CLI arms exactly; control arm at most** — see §7) | toolset-side counter (§5) |
| affordances | strategy primer, wiki, memory | prompt+tools (arm 0) vs files (CLI arms), §6 |

Only the **harness** varies.

## 3 · Arms

| arm | harness | status |
|---|---|---|
| 0 — control | `nethack_v1.NetHackHarness` | exists |
| 1 | Codex (`CLIHarness` built-in) | needs verifiers > 0.1.14 |
| 2 | Prime Agent (external plugin) | resolves on stock 0.2.1 (Task 12) |

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

Each arm connects through **its own natively available MCP registry** — we add the server to
whatever mechanism the CLI already ships (`codex mcp add` for Codex, `mcpServers` +
skill package for Prime Agent) rather than building a bespoke integration per arm.

## 5 · The dispatch split (the one substantial change)

> **Amended by Task 12 — read with §10.** This section is written against verifiers 0.1.14,
> where `self_dispatch` was a flag on one in-process toolset. On 0.2.1 the two values became
> two *routes*: `self_dispatch=True` is the native v1 taskset whose `NetHackToolset` is an MCP
> server, and `self_dispatch=False` — arm 0's `env_response` loop — is 0.2.x's v0 **legacy
> bridge** (`vf-eval --id nethack`), which runs the v0 rollout verbatim. Passing
> `self_dispatch=False` to the v1 toolset now raises, pointing at the bridge, rather than
> silently running arm 0's name with MCP semantics. Everything below about the *gate*, the
> *shared render half*, the *single dispatch path* (`_apply_tool_call`) and `obs_mode` still
> holds unchanged; only "one flag" became "two routes over one execution path", and §10's
> equivalence criterion is what holds them together.

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

### 7.1 · Amendment (Task 13): the 150-call budget is NOT exactly equal across the arms

Measured while writing the arm configs. The toolset-side referee gives a CLI arm **exactly 150
executed skills**: its counter advances only when a call reaches the engine, refused calls do not
increment it, and concurrent calls are serialized (`NetHackToolset._with_state`) so a batched
assistant turn cannot overrun the cap.

The control arm has no such counter. Its nearest equivalent is the v0 `max_turns = 150`, which
caps **LM turns**, and a turn can pass without executing a skill:

* `nethack.py:676-680` — a turn with no tool call returns a "you must call a tool" nudge and
  burns the turn. Measured: the engine is not stepped, `terminated` stays `False`, and
  `is_completed` (`nethack.py:1284-1297`) returns `False`.
* `nethack.py:690` — only the **first** tool call of a turn is applied. Measured with three
  `search` calls in one turn: in-game time advanced by 1 and the observation carried
  `[multi-tool warning: only the first of 3 tool calls was applied.]`. The
  `_dropped_extra_tool_calls` counter is consumed and zeroed when that warning renders
  (`nethack.py:1263-1270`), so it cannot be read back afterwards — the warning line in the
  rendered observation is the durable artifact.

So the control arm executes **at most 150** skills and typically fewer. The asymmetry runs in the
CLI arm's favour on the experiment's controlled variable, and it widens exactly where CLI agents
differ most: a batched assistant turn has *every* tool call dispatched over MCP, where the control
arm would have executed only the first.

This is **not** corrected by config. Raising the control arm's `max_turns` to compensate would
change arm-0 semantics that Task 12 preserved deliberately, and the size of the gap is a
per-rollout property no single value can fix.

**Requirement on Task 11 (aggregation).** Normalize on the *measured* action count, never on the
nominal 150:

* control arm — the v0 rubric already emits **`total_tool_calls`** per rollout (plus a
  `<skill>_calls` breakdown), carried into `Trace.metrics` by the legacy bridge;
* CLI arms — **`skill_calls`**, written onto `Trace.metrics` by `NetHackTask.finalize`.

These two are the same quantity and are the denominator for any per-action comparison. The
aggregator must also report the realized counts per arm, so a reader can see the gap rather than
assume parity; a paired per-seed comparison at unequal action counts must be labelled as such.

## 8 · Scoring

Reuses `tools/encoding_eval/aggregate_run.py`: depth from traces (`max_dlvl_reached`), real BALROG
progression (`nethack_harness/prompt/balrog.py`, validated against the DL10/XL6 = 12.56% anchor),
alive@cap, death %, tokens/turn — plus `turns_used`. Paired per-seed deltas and an exact sign test
across the 16 seeds, as in exp1 §3.1; marginal SEs at n=16 are wide and must not be quoted alone.

## 9 · Risks

1. **The dispatch split touches the path exp1 depends on.** Mitigated by the golden test in §10.
2. **Each arm consumes the toolset through its own native MCP registry** — Codex through its MCP
   client, Prime Agent through its kernel-side MCP skill (`docs/MCP_INTEGRATIONS.md`). Same server,
   same 15 tools; the integration path is whatever each CLI ships. This is a stated design fact,
   not something the writeup needs to dwell on.
3. **GLM 5.2 billing.** `z-ai/glm-5.2` sits under `pinference-glm` in `configs/endpoints.toml`,
   which carries **no** `X-Prime-Team-ID`; only `prime-team` does, and it lists Gemini only. As
   written GLM bills the personal balance ($0) → `insufficient_funds`, which surfaces as a *hang*
   (exp1 confound #1). The team header must be set once, centrally, before any run.
4. **Out-of-band state reads.** CLI arms keep native shell as a thinking aid but must not reach the
   engine or level files. Enforced by sandbox scope; verified by trace audit.
5. **PR #1985 is OPEN — and not needed.** *Amended by Task 12:* stock verifiers 0.2.1 already
   resolves an external plugin by probing `find_spec("verifiers.v1.harnesses.<id>")` and falling
   back to the top-level module (`verifiers/v1/loaders.py:32-45`); the PR's residual delta is
   error-message quality only. The vendored patch has been **reverted** — a patched
   site-packages is a reproducibility liability for a benchmark — and `tests/test_vendored_loader.py`
   now pins both stock external-plugin resolution and a wheel-RECORD integrity check, so a future
   patch fails the suite. The Prime Agent arm never waited on the merge.
6. **No comparability to run2.** run2 was Gemini 3 Flash; this series is GLM 5.2. Arm 0 is the only
   baseline, and it must be run — there is no borrowing exp1's numbers.

## 10 · Testing

> **Amended by Task 12.** The original two criteria were written against a `self_dispatch`
> flag with two live values. Under verifiers 0.2.x that flag no longer has two values: a v1
> toolset *is* an MCP server, so the only way a tool call reaches the engine is the call
> itself, and the harness-driven (`self_dispatch=False`) loop moved to 0.2.x's v0 **legacy
> bridge** — which is now arm 0's execution path. `self_dispatch=False` raises rather than
> silently running arm 0's name with MCP semantics. The criteria below are re-expressed
> against the two routes that actually exist; the property being checked is unchanged.

- **Arm-0 fidelity:** arm 0 runs through 0.2.x's v0 legacy bridge (`vf-eval --id nethack`),
  which executes the v0 rollout **verbatim** — `run_legacy_eval` calls `env.run_rollout`
  (`verifiers/v1/legacy.py:517`), so `env_response`, the per-turn compaction pipeline, the
  skill dispatch order and the v0 rubric are the same code exp1 validated. Fidelity is
  therefore established by construction (no port in arm 0's path), not by replaying a
  recorded run2 rollout — run2 was Gemini 3 Flash and this series is GLM 5.2, so a
  turn-for-turn replay was never comparable anyway (see §9.6). Covered by
  `environments/nethack/tests/test_v1_taskset.py::test_control_arm_runs_through_the_v0_legacy_bridge`,
  which drives the bridge end-to-end with a keyless mock model.
- **Cross-route dispatch equivalence:** the same skill sequence, from the same seed, must
  yield identical engine state and identical trace fields whether dispatched by the **control
  route** (v0 `env_response`, what the bridge runs) or the **native route** (the callables
  `NetHackToolset._register` publishes over MCP). This is the check that catches
  arm-0-vs-CLI-arm drift; if it fails the two arms are not playing the same game and no
  comparison between them means anything.
  - *Engine-state half — DONE (Task 12):*
    `environments/nethack/tests/test_cross_route_equivalence.py` asserts identical engine
    status, identical reward/stop scalars, and byte-identical rendered observations after
    every step of a fixed 4-skill sequence, plus that the typed v1 state agrees with the v0
    state the control arm's rewards read. Mutation-checked (a seed change on one route and a
    disabled state mirror are both caught).
  - *Trace-field half — TODO, Task 9 must implement:* needs a booted MCP tool server
    (`python -m nethack_v1`) and a launched harness program, so it cannot run in process.
    Task 9 must run one rollout per route over the same seed and the same forced skill
    sequence and assert the resulting `Trace` fields agree — at minimum `reward`, the four
    reward metrics, `num_turns`, and `stop_condition`.
- **Gate:** `tools/encoding_eval/_verify_gate.py` passes on both routes — `move` executed = 0
  (the script now asserts this explicitly: a rejected `move` leaves `structured_obs`
  object-identical and `max_dlvl_reached` unchanged). On the native route the same gate is
  covered by
  `environments/nethack/tests/test_toolset_self_dispatch.py::test_netplay_gate_holds_at_the_mcp_registration_point`
  (`move` is never published to MCP) and
  `::test_calls_outside_the_exposed_set_are_refused_without_stepping`.
- **Smoke per arm:** n=1, ~20 calls, verifying the arm reached the engine, the workspace seeded,
  the observation rendered, and the counter incremented — the exp1 discipline that caught
  silently-wasted runs.

## 11 · Layout

```
NetHack-hub/
  environments/nethack/nethack_v1.py        # 0.2.1 taskset: NetHackToolset (MCP server, netplay
                                            #   gate + call budget), NetHackTask (rewards/@stop),
                                            #   NetHackTaskset, NetHackHarness (default harness)
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
