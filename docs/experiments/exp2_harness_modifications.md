# Experiment 2 — Agent Harness Modifications

Status: **plan + scaffolding** (this branch). No paid evals run yet.
Owner: TBD.  Env: `environments/nethack` (Prime Hub `jonathanliu/nethack`).

---

## (a) Goal & why we care

> **Experiment 2: Agent Harness Modifications.** Test and validate our custom
> harness and architectural fixes across a diverse set of frontier models,
> specifically Gemini, GLM, and GPT-5.5. **Detail:** Iterate rapidly using
> Gemini to implement fixes, and then evaluate the final harness structure
> across the other models. **Why we care:** It is crucial to prove that
> improvements in progression are robust and fundamentally derived from the
> harness architecture itself, rather than being an anomaly or artifact specific
> to one model's internal representations.

In one line: **hold the harness fixed, vary the model — the progression gain
should survive.** If it does, the gain is architectural; if it evaporates when
we swap Gemini for GLM or GPT-5.5, it was a model-specific artifact.

---

## (b) Infrastructure — technical

### The harness architecture under test

The "harness" is `environments/nethack/nethack_harness/` — everything the LLM
sees and every lever it can pull, sitting between the raw NetHack engine
(`nethack_core`, layer 1) and the verifiers rollout loop (`nethack.py`). Its
robustness across models is the whole claim, so it is worth naming the parts:

| Component | File | What it is |
|---|---|---|
| **Typed interface** | `nethack_harness/interface.py` | `TypedNetHackInterface` subclasses the engine's `NetHackInterface` and adds an `Action(name, args)` branch to `step`. `action_spec()` is sourced from the live skill registry, so the typed action set never drifts from the skills. |
| **Skills / action surface** | `nethack_harness/tools/skills.py` | NetPlay-style skill registry. Each skill is a callable `(env, args) -> (primitive NLE actions, feedback)`. The registry auto-generates the OpenAI tool schemas the model calls. This is the agent's API to the world. |
| **Code mode** | `nethack_harness/tools/code_mode.py` | Alternative action surface: a single `code` tool that runs sandboxed Python against a curated `nh` namespace (AST validator + SIGALRM cap). Selected by `interface="code"`. |
| **Navigation** | `nethack_harness/navigation/pathfinding.py` | A* over the glyph grid + frontier autoexplore. Powers `move_to` / `autoexplore` so the model issues goals, not keystrokes. |
| **Memory** | `nethack_harness/memory/journal.py` | Keyed note store + pinned objective. Long-horizon belief state that survives history compaction (the "Gemini/Claude Plays Pokemon" lesson: the scratchpad is load-bearing). |
| **Prompt** | `nethack_harness/prompt/` | `prompt_spec.py` (the `PromptSpec` = obs form + system prompt + per-turn template + tool set, dispatched via `VARIANT_REGISTRY`), `rendering.py`, `map_encoders.py`, and `balrog.py` (the progression metric, below). |
| **Refiner** | `nethack_harness/refiner.py` | Continual-Harness refiner: every N turns a teacher LLM emits CRUD edits over prompt/sub-agents/skills/memory. Optional (`variant="CH"` / `refine=True`). |

The rollout loop (`nethack.py::NetHackVerifiersEnv`) holds one resolved
`PromptSpec` and dispatches through it, instead of scattering `variant ==`
checks. `load_environment(...)` is the single constructor; its knobs
(`interface`, `variant`, `compact_obs`, `history_*`, `belief_state_interval`,
`refine*`, `continual*`, `tune`/`modify`/`level_blob`) are the coarse harness
config surface.

### What the "architectural fixes" surface is

There are **two** places a harness change lands:

1. **In-source** — edit the modules above. This is what you iterate on with
   Gemini (fast, local; see the loop below).

2. **Overlay (the ablation toggle)** — `environments/nethack/harness_overlay.py`
   is a runtime seam gated by the `NETHACK_HARNESS=<name>` env var. When set,
   `load_environment` calls `apply_overlay(...)`, which mutates four
   well-defined harness surfaces **without a code edit**:
   - `SYSTEM_PROMPT` (replace / append / patch),
   - the per-step formatter (`_VARIANT_FORMATTERS` selection),
   - the enabled/disabled skill set (`filter_tool_callables`), and
   - reward weights (`apply_reward_weights`).

   With `NETHACK_HARNESS` unset it is a **strict no-op** — the default path is
   bit-identical. This is the seam that lets one rollout run "baseline" and the
   next run "fixes" against the *same model*. The intended overlay schema is
   documented in `configs/harness/` (`baseline.toml`, `fixes.toml`).

   **Caveat (wiring gap):** `apply_overlay` loads a named version via
   `tools.launchpad.core.harness.load_harness`, and the `tools/launchpad`
   package is **not yet vendored into this repo**. Until it lands, an unknown
   `NETHACK_HARNESS` name is a *logged no-op*. See "What remains".

### The Gemini fix-iterate loop (rapid, local)

Gemini is the iteration model because `GEMINI_API_KEY` is set on the dev box
(the only frontier key currently set here). One turn of the loop:

```
edit nethack_harness/**            # implement a fix (skills / prompt / nav / memory)
py_compile + pytest tests/ -q      # local sanity, no API keys
prime eval run nethack \           # cheap local rollout against Gemini
  -m gemini-2.5-flash --provider gemini \
  -a '{"tier":"corridor_explore","max_turns":200,"trace_dir":"outputs/iter/<n>"}' \
  --num-examples 1 --rollouts-per-example 1
read balrog_progression / descent  # did the fix move the needle?
```

The env runs **locally**; the model is called **remotely**. Traces land under
the cell's `trace_dir` as NDJSON (`rendered_user_content` per turn), replayable
with `tools/encoding_eval/replay.py`. Iterate here until the fix holds on
Gemini, then **freeze the harness** and validate on the other models.

### Running the SAME harness across Gemini / GLM / GPT-5.5

Same env, same env-args, same harness version — only `-m/--provider` changes.
Provider→endpoint mapping (mirrors
`approaches/voyager/reverse_curriculum_sweep.py::_endpoint_for`, now also in
`configs/endpoints.toml`):

| Model family | id form | provider | endpoint | key |
|---|---|---|---|---|
| Gemini | `gemini-*` | `gemini` | `generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` ✅ set here |
| GLM | `z-ai/glm-*` | `prime` | `api.pinference.ai/api/v1` | `PI_API_KEY` ⚠️ **unset here** |
| GPT-5.5 | `gpt-5.5` | `openai` | `api.openai.com/v1` | `OPENAI_API_KEY` ⚠️ **unset here** |

Per-model eval configs live at `environments/nethack/configs/eval/{gemini,glm,gpt-5-5}.toml`.

### The progression metric that proves robustness

`nethack_harness/prompt/balrog.py::progression_score(max_dlvl, xp_level)` — a
BALROG-style (Paglieri et al., ICLR 2025) smooth proxy for **P(ascend)** from the
deepest `(dungeon level, experience level)` a rollout reached:
`(DL/50)^1.3 · (XL/30)^0.6`, clipped to `[0,1]`, bucketed by `progression_tier`
into `spawn → early → past_mines → midgame → endgame`. It is written to
`state["balrog_progression"]` every step (informational; **not** a rubric
reward, so it never touches training gradients). Alongside it, the rubric's
**`descent_reward`** (reached DL≥2) gives a binary success rate with a Wilson CI
via `tools/eval_instrument.py::summarize_eval`.

**Why this metric for this claim:** progression is a smooth, model-agnostic
"how far did the agent get" that is comparable across models with different
token economies. Robustness = the progression curve **moves with the harness
version and clusters across models**, not with the model identity.

---

## (c) The claim, in blog-sense

Frontier models have wildly different internal representations, tool-call
dialects, and failure modes. A naive reading of "our agent got deeper in NetHack"
is: *maybe this model just happens to be good at NetHack.* Experiment 2 refutes
that by turning the harness into the independent variable and the model into a
control. We iterate the harness against Gemini until progression improves, then
**freeze it** and drop in GLM and GPT-5.5 unchanged.

Two outcomes, both informative:

- **Gains hold across all three** → the progression comes from *structure* — the
  skills, the pathfinder, the journal, the typed action layer — scaffolding that
  compensates for what any single model lacks. That is the result we want to
  publish: a harness, not a lucky model.
- **Gains hold only on Gemini** → we overfit the harness to one model's quirks,
  and the honest write-up says so. The cross-model **spread** of the progression
  metric is exactly the number that distinguishes these two worlds.

The design already leans this way on purpose: the journal exists because
*models re-derive world state every turn without it*; the pathfinder exists so
*models issue goals, not fragile keystrokes*; code-mode exists because *weaker
policies emit malformed multi-call tool sequences*. Each is a bet that the
bottleneck is scaffolding, not raw model IQ. Experiment 2 is the bet's payoff
test.

---

## (d) How to run

### Quick single-model smoke (Gemini — reachable here)

```bash
cd environments/nethack
prime eval run nethack -m gemini-2.5-flash --provider gemini \
  --num-examples 1 --rollouts-per-example 1 \
  -a '{"tier":"corridor_explore","max_turns":200}'
```

### Cross-model comparison (the experiment) — `tools/harness_sweep.py`

The sweep runner drives the **model × harness-version** matrix, sets
`NETHACK_HARNESS=<version>` per cell, resolves each model's provider/endpoint,
and reports per-cell progression (mean/stdev) + descent-rate CI plus a
**per-harness cross-model spread** (the robustness statistic).

```bash
# 0) Inspect the plan + credential status WITHOUT spending (recommended first):
python tools/harness_sweep.py --dry-run \
  --models gemini-2.5-flash z-ai/glm-5 gpt-5.5 \
  --harness-versions baseline fixes

# 1) Prove the aggregation with a synthetic runner (no model calls, no engine):
python tools/harness_sweep.py --demo

# 2) Real sweep (spends tokens; needs the three keys). Start tiny:
python tools/harness_sweep.py \
  --models gemini-2.5-flash z-ai/glm-5 gpt-5.5 \
  --harness-versions baseline fixes \
  --tier corridor_explore --num-examples 2 --rollouts-per-example 3 \
  --run-dir outputs/harness_sweep/run1
```

Outputs land in `--run-dir` (default `outputs/harness_sweep/`): `table.json`
(full stats) + `table.md` (the two tables below). Per-cell traces land under
`<run-dir>/<harness>__<model>/` (NDJSON, replayable).

### Config matrix (model × harness-version)

| model \\ harness | `baseline` (no overlay) | `fixes` (candidate) |
|---|---|---|
| `gemini-2.5-flash` | iterate + baseline anchor | primary iteration target |
| `z-ai/glm-5` | cross-model control | validation |
| `gpt-5.5` | cross-model control | validation |

Robustness reads off the **per-harness** table: `fixes` should show a higher
`mean(progression)` than `baseline` **and a small `spread(stdev)` across
models**. A large spread = the gain is model-specific.

### Where results land

- `--run-dir/table.{json,md}` — aggregated matrix (this is the headline artifact).
- `--run-dir/<harness>__<model>/*.ndjson` — per-rollout traces.
- `--run-dir/<harness>__<model>/eval_out/**/results.jsonl` — prime's scalar
  rubric rewards (`descent_reward`, `scout_reward`, ...) when the run completes.
- Hosted runs additionally surface at
  `app.primeintellect.ai/dashboard/evaluations/<id>` (URL printed by prime).

---

## (e) What remains

Infra **present** (this branch / already in repo):

- ✅ Harness architecture (skills, prompt, navigation, memory, typed interface, refiner).
- ✅ Overlay seam `harness_overlay.py` wired into `load_environment` (no-op-safe).
- ✅ Progression metric `balrog.progression_score` + `descent_reward` rubric +
  `summarize_eval` CI/taxonomy.
- ✅ Per-model eval configs `configs/eval/{gemini,glm,gpt-5-5}.toml` (this branch).
- ✅ Google/Gemini endpoint + GPT-5.5 ids added to `configs/endpoints.toml` (this branch).
- ✅ Cross-model × harness sweep runner `tools/harness_sweep.py` with
  progression + variance reporting, provider mapping, `--dry-run` + `--demo`
  (this branch; verified without spend).
- ✅ Example overlay configs `configs/harness/{baseline,fixes}.toml` (this branch).

**Remaining** work to make the experiment fully executable:

1. **Overlay config loader (the one real gap).** `harness_overlay.apply_overlay`
   imports `tools.launchpad.core.harness.load_harness`, which is not vendored
   here. Until it (or a small TOML loader with the same `HarnessConfig` shape:
   `system_prompt{mode,text}`, `per_step_prompt{template}`, `tools{enabled,disabled}`,
   `rewards{...}`) is added, `baseline` vs `fixes` differ only in label. This is
   the blocker for a *real* harness ablation. `configs/harness/*.toml` are ready
   to feed it.
2. **Missing credentials on this box.** Only `GEMINI_API_KEY` is set. `PI_API_KEY`
   (GLM, team-billed) and `OPENAI_API_KEY` (GPT-5.5) must be exported before the
   GLM/GPT cells run. `--dry-run` flags each missing key per cell.
3. **Confirm live model ids.** `gemini-2.5-*`, `z-ai/glm-5*`, `gpt-5.5` are
   plausible but must be checked against the live providers before a paid run
   (the configs list alternates as comments; do not hardcode a dead id — cf. the
   `Qwen3.5-VL-7B` id that never existed on Prime, noted in encoding_eval).
4. **Provider strings.** `--provider gemini|prime|openai` in the sweep is a
   best-effort map; confirm the exact provider tokens `prime eval run` accepts on
   the current CLI (the repo elsewhere uses `--provider prime`).
5. **Budget + n.** Token cost is punishing (`docs/EVAL_RECIPES.md`: a single
   10-min Qwen rollout burned 4.3M input tokens). Pick `n`/`rollouts` for a
   meaningful CI without blowing budget; consider `docs/PROMPTING_SURVEY.md`
   token-reduction first.

---

## (f) Risks & open questions

- **Overlay is currently label-only.** Without the loader (remaining #1), a
  "baseline vs fixes" sweep today measures *nothing* about the fix. Either land
  the loader or iterate in-source and compare git revisions of the harness
  instead (run the same sweep on two checkouts).
- **Progression is a proxy, not ground truth.** `balrog.progression_score` is an
  *analytic approximation* of the BALROG P(ascend) table, not the published
  table. It is monotone and fine for relative comparison, but don't report it as
  an absolute ascension probability.
- **Cross-model confounds.** Different models have different tool-call dialects,
  context windows, and token costs. `corridor_explore` (DL≥2) keeps rollouts
  short and cheap, but at that shallow depth progression differences are tiny
  (see the demo: 0.003 vs 0.03) — the *relative* lift and the cross-model spread
  matter, not the absolute magnitude. Deeper tiers give more signal but cost more.
- **Small-n variance.** At `n≤5` the Wilson CIs are wide; a "robust across
  models" claim needs enough rollouts that the per-harness spread is
  distinguishable from rollout noise. Budget accordingly.
- **Fairness of the freeze.** "Iterate on Gemini, freeze, validate elsewhere" is
  the honest protocol only if we do **not** then tweak the harness after seeing
  GLM/GPT numbers. Pre-register the frozen harness (a git tag) before the
  validation runs.
- **GPT-5.5 / GLM-5 availability.** Both ids are forward-looking; if a provider
  serves a different current id, update the config and the endpoint models list
  rather than silently falling back.
