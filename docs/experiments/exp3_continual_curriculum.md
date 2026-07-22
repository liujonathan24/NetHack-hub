# Experiment 3 — Continual Harness & Curriculum Learning Analysis

*Status: executable plan + cheap analysis scaffolding landed. Depends on the
six-floor primitives curriculum (Step 4C, branch `feat/4c-six-floor-curriculum`)
for the reduced-scale environment — do not reimplement it here.*

---

## (a) Goal & why we care

> **Experiment 3: Continual Harness & Curriculum Learning Analysis.** Evaluate
> our continual harness by leveraging established agentic exploration
> architectures, specifically Voyager and Go-Explore, as an extension of the
> agent harness tests. Analyze the emergent behavior where models consistently
> attempt to build a navigation tool or component to solve the curriculum.
>
> **Detail:** Document that this navigation component inevitably gets stuck, even
> when tested on reduced-scale curriculum environments (e.g., limited to 6
> floors), resulting in extremely high variance across different environment
> seeds.
>
> **Why we care:** This highlights the fundamental difficulty of achieving high
> coverage on gameplay traces. Understanding these failure modes — where models
> struggle to create a consistently good run despite attempting to build
> navigation primitives — is essential for diagnosing why current agentic LLMs
> fail to generalize in NetHack's procedural generation.

---

## (b) Infrastructure (technical)

Three exploration architectures share one immutable NetHack engine and one
reduced-scale curriculum. Nothing below touches the game engine; every knob lives
in the harness.

### The continual-harness loop — `approaches/continuous_harness/`

An automated outer loop that **tunes the harness around an immutable engine**.
`python -m approaches.continuous_harness.loop --iterations N [...]`. Each
iteration:

1. **Worktree isolation.** `make_iteration_worktree` runs
   `git worktree add .claude/worktrees/harness-iter/iter<N> -b harness-iter-<runid>-<N> HEAD`,
   then **symlinks** (never copies) `third_party/NetHack` and the shared `.venv`
   into it. The engine is a shared pointer back to the source tree so an
   iteration physically cannot patch it.
2. **Immutable-game guard.** `assert_engine_untouched` runs before and after the
   eval: the engine dir must be a symlink, and `git status --porcelain` over
   `third_party/NetHack`, `third_party/nethack`, `nethack_core` must be clean.
   Any dirty engine path fails the iteration. (The code-editing sibling
   `auto_improve.py` extends this to a champion/challenger loop that lets an LLM
   rewrite ONE whitelisted harness file per iteration — engine still frozen —
   gated by pytest + an eval-margin accept/reject.)
3. **Bootstrap writing.** `write_bootstrap` emits one `seed<N>.json` per seed in
   the shape of `nethack_harness.refiner.snapshot_components(state)`
   (`prompt_addendum`, `subagents`, `skills`, `notes`, `objective`). This is the
   only channel by which the proposer's prompt/macros/sub-agents reach the
   rollout; `variant="CH"` consumes it at rollout start.
4. **Eval.** `run_eval` sources Prime Inference creds from `--env-sh` then runs
   `uv run vf-eval nethack --provider prime -m <policy> -n <n_seeds> -r 1 -a <env_args>`.
   In `--dry-run` it instead synthesizes deterministic NDJSON traces — **no API,
   no budget** — so the whole pipeline (parse → leaderboard → run-log) exercises
   real code paths for free.
5. **Depth parsing.** `parse_iteration_depth` reads every `*.ndjson` trace under
   `trace_dir`; per-rollout depth = max `max_dlvl_reached` (fallback max `dlvl`);
   iteration score = **mean** per-rollout depth across seeds. `parse_mean_reward`
   best-effort pulls a mean reward from vf-eval's saved `metadata.json`.
6. **Proposer.** `FallbackProposer` (deterministic) or `LLMProposer` (the teacher
   model, e.g. GLM-5) reads the compact history (config + depth + a trajectory
   excerpt) and proposes the next `HarnessConfig` — it may switch the observation
   `variant`, the `skill_set`, or rewrite the prompt addendum / macros /
   sub-agents. `HarnessConfig.validate()` clamps any bad proposal so a bad LLM
   suggestion can never crash the loop. Policy and teacher **must differ** (Prime
   Inference serves both).
7. The best config sits on a leaderboard ranked by mean depth; a `run_log_<runid>.json`
   records every iteration.

**Three mutable surfaces** (all map onto existing `load_environment(...)` kwargs):
observation format (`variant`), tools (`skill_set`), and prompt+macros+sub-agents
(the `seed<N>.json` bootstrap).

### Go-Explore — `approaches/go_explore/`

"First return, then explore" (Ecoffet et al. 2021) over the engine's **byte-exact
in-memory snapshot/restore/branch** API (`nethack_core/engine_env.py`).

- **Archive** cells keyed by `(progress, dungeon-number, x//grid, y//grid)`; each
  cell holds a `snapshot()` handle, the trajectory that first reached it, depth,
  and a visit count.
- **Return** deterministically via `restore(handle)` (no replay), then `reseed`
  after restore so random chance diverges across returns.
- **Explore** K weighted primitive actions (8 compass moves, run-macros, `search`,
  real `>`/`<`) from the returned state; archive newly reached cells.
- **`--branch-probe`** (landed): a cell's *frontier promise* is measured with the
  engine's `EngineEnv.branch(n, reseed=True, horizon, action)` primitive —
  snapshot once, restore `n` times, roll each branch forward `horizon` steps of a
  fixed action, and score the fraction of **distinct** char-traces (`0..1`).
  Selection then biases returns toward unsettled frontiers. `--branch-n` /
  `--branch-horizon` tune the probe. The report records `mean_branch_divergence`
  and `n_branch_probes`. This is **keyless** (no API).
- `curriculum_go_explore.py` is the six-floor curriculum variant: it scores cells
  by **tour progress** (descend 1..6, then `6 + (6-floor)` on the way up) and
  drives `CurriculumEngineEnv` with real primitives only. `run_curriculum_go_explore(iterations, explore_steps, seed)`
  returns `n_cells`, `deepest_floor`, `climbed_back`, `reached_bottom`, and a
  per-iteration timeseries. This is the driver the seed-sweep's live backend calls.

### Voyager — `approaches/voyager/`

Automatic-curriculum, skill-library agent (Wang et al. 2023): (1) an LLM proposes
the next objective biased toward novelty; (2) it synthesizes/composes a **skill**
(an ordered macro over primitive skills — the same `K` component the CH refiner
edits, persisted via `bootstrap_dir`); (3) self-verifies by running it and keeps
the skill iff its success predicate holds. `reverse_curriculum_sweep.py` is the
sharpest probe: it constructs the hero at curriculum floor `s` (via the internal
`goto_abs` cheat, used ONLY to build the start state — never exposed to the agent)
and asks it to **climb back to floor 1 using only real `<` stairs it navigates to
itself**. Conditions `climb_from_2 .. climb_from_6` + `full_tour`. It ships a
door-aware BFS navigator (`nav_to` / `_bfs_path`) — i.e. the very "navigation
primitive" this experiment studies.

### Reduced-scale env — the six-floor curriculum (Step 4C)

The compressed tour (see `docs/CURRICULUM.md`): a fixed female-neutral Valkyrie
with full vision descends `DoD 1→2→3`, cross-branch **jumps** to `Gehennom 48→49→50`
(with a realistic stat upgrade), then climbs back — six floors down, six up.
`nethack_core/curriculum_engine_env.py` intercepts the `>`/`<` keystrokes at the
cross-branch boundary so agents run unmodified. **Step 4C** (parallel work, branch
`feat/4c-six-floor-curriculum`) builds the *primitives* six-floor curriculum this
experiment runs on; treat it as the reduced-scale environment and depend on it.

### Seeds — how they are set and swept

The engine seeds the core + display RNG via `EngineEnv.seed(core, disp)` (staged
for the next `reset()`) or `reset(seeds=(core, disp))`; the curriculum drivers use
`env.reset(seeds=(seed, seed))`. Go-Explore additionally reseeds **after each
restore** (`engine.reseed(core, disp)`) so branches diverge. A seed sweep just
runs the same driver across a list of seeds (default `19 2 9`, plus `7 42` for the
mock) and compares outcomes; because layout is procedurally generated per seed,
the sweep is exactly what surfaces cross-seed variance.

### Coverage — how it is measured

Two complementary proxies, both computed by the scaffolding
(`approaches/analysis/seed_variance.py`):

- **State-space coverage** = distinct Go-Explore archive cells (`n_cells`): how
  much of the `(floor, position, progress)` space a run touched. This is the
  direct gameplay-trace-coverage proxy.
- **Tour coverage** = `progress / 11`, layout-agnostic: how far along the 6-down /
  6-up curriculum a seed got (`deepest_floor`, then `+climbed_back` after the
  bottom).

---

## (c) The emergent failure mode (blog-sense)

Point any of the three architectures at the six-floor curriculum and the same
thing happens: **the model tries to build a navigation tool.** Voyager's very
first synthesized skills are "walk to the stairs"; Go-Explore's whole return step
is navigation; the continual-harness proposer's most common edit is to
`navigation/pathfinding.py` / a `move_to` macro. Everyone independently converges
on "if I could just reliably path to the next staircase, the curriculum falls."

**And the navigation component inevitably gets stuck.** Not at the end of the
game — a couple of floors in. On the compressed tour, the scripted greedy-nav
baseline reaches the top from a **1-floor** climb only 30% of the time, and from
any longer climb **0%** — the per-segment "reach just the next floor" probability
never approaches the 0.90 advancement gate (scripted 0.16–0.26; GLM-5.2 0.67 at
floor 2 then 0.00 at floors 3–4). See `outputs/curriculum_experiments/reverse_curriculum/REPORT.md`.
The nav primitive wedges on a closed/locked door it never kicks, a hidden passage
it never searches, a monster it won't fight through, or a level with no visible
route to the stair — and then it sits there.

**Where it stalls swings wildly from seed to seed**, because the dungeon is
procedurally generated. The keyless Go-Explore curriculum runs
(`outputs/curriculum_experiments/go_explore/`) all cap at floor 2, but *when* they
stall varies enormously — last depth improvement at iteration 22 on one seed vs.
~1050 and ~1795 on others — and cell coverage swings 92↔101 with the same ceiling.
The reverse-curriculum sweep shows the harsher axis: p(reach top) is 0.30 on one
start and 0.00 two floors deeper. **Same harness, same model, wildly different
outcome per seed = extremely high variance.**

**The consequence is low coverage of the gameplay-trace space.** No single run
gets close to touching the whole tour; the mean tour-coverage across seeds sits
near ~0.18 (2 of 6 floors) and reached-bottom is 0%. You cannot build a
consistently good run, so you cannot cover the trace distribution — which is
exactly why current agentic LLMs fail to generalize under NetHack's procedural
generation. The bottleneck is not the curriculum length; it is a navigation
primitive that cannot be made reliable across seeds.

---

## (d) How to run

All commands are **free/keyless unless marked PAID**. From the hub root, with the
engine importable:

```bash
export NLE_LIB_PATH=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness/third_party/NetHack/src/build/libnethack.so
export PYTHONPATH=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness:$PWD/environments/nethack
PY=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness/.venv/bin/python   # prebuilt engine venv
```

### Continual-harness loop on the six-floor curriculum

```bash
# Orchestration smoke — NO API, NO budget (synthesized depths):
$PY -m approaches.continuous_harness.loop --iterations 2 --dry-run \
    --out /tmp/harness_loop_dryrun

# PAID: real 3-iteration LLM-proposer loop on the curriculum tier.
#   /tmp/ch_env.sh must export PI_API_KEY / REFINER_API_KEY / REFINER_BASE_URL.
source /tmp/ch_env.sh && $PY -m approaches.continuous_harness.loop \
    --iterations 3 --proposer llm \
    --policy z-ai/glm-4.6 --teacher z-ai/glm-5 \
    --tier curriculum --seed 19 --n-seeds 3 \
    --max-turns 200 --refine-interval 20 --out /tmp/harness_loop_run
```

### Go-Explore (keyless) and the branch probe

```bash
# Six-floor Go-Explore across seeds (no API):
$PY approaches/go_explore/curriculum_go_explore.py \
    --iterations 400 --explore-steps 40 --seeds 19 2 9 \
    --out outputs/curriculum_experiments/go_explore

# Frontier-divergence branch probe (exercises EngineEnv.branch()):
$PY -m approaches.go_explore.go_explore --iterations 400 \
    --branch-probe --branch-n 4 --branch-horizon 20
```

### Voyager reverse-curriculum climb sweep (PAID — LLM per turn)

```bash
PI_API_KEY=... $PY approaches/voyager/reverse_curriculum_sweep.py --launch \
    --seeds 19 2 9 --reps 4 --workers 6 --model z-ai/glm-5.2 \
    --out outputs/curriculum_experiments/reverse_curriculum
```

### Seed sweep + cross-seed variance report (the quantifier)

```bash
# FREE smoke — mock backend, NO engine, NO API (reproducible):
$PY -m approaches.analysis.seed_variance --dry-run --seeds 19 2 9 7 42

# FREE — aggregate whatever real runs already exist on disk:
$PY -m approaches.analysis.seed_variance --source aggregate \
    --results-dir outputs/curriculum_experiments/go_explore

# FREE — live keyless six-floor Go-Explore sweep (engine on CPU, no API):
$PY -m approaches.analysis.seed_variance --source go-explore \
    --seeds 19 2 9 --iterations 400 --explore-steps 40 \
    --out outputs/curriculum_experiments/seed_variance/report.json
```

---

## (e) Metrics

Emitted by `approaches/analysis/seed_variance.py` (per-seed table + rolled-up
`VarianceReport`):

| Metric | Meaning |
|---|---|
| `deepest_floor` / `climbed_back` / `reached_bottom` | per-seed tour outcome (of 6 down / 6 up) |
| `depth_mean` / `depth_std` / `depth_max` / `depth_min` / `depth_range` | cross-seed depth distribution |
| `depth_cv` = std/mean | the headline **cross-seed variance** number |
| `coverage_cells_{mean,std,max,min}` | gameplay-trace **coverage** (distinct archive cells) and its spread |
| `tour_coverage` = progress/11 | layout-agnostic fraction of the tour reached |
| `frac_reached_bottom` | how often the run completes the descent (expected ≈ 0) |
| `stall_iter` (per seed) + `mean_stall_fraction` | **where/when the nav component stalls** — the last iteration depth improved, and how early (as a fraction of the run) the plateau sets in |
| `modal_stall_floor` | the floor the nav primitive most often wedges at |

The `stuck_signal` string composes these into a one-line diagnosis. The nav-stuck
fingerprint = **high `depth_cv` and/or high `coverage_cells_std`, `frac_reached_bottom`≈0,
and an early `mean_stall_fraction`.** Note the two variance axes: sometimes the
*depth ceiling* itself varies across seeds (mock/reverse-curriculum data), and
sometimes the ceiling is uniform but the *stall timing + cell coverage* vary
hugely (real Go-Explore data) — both are the same failure, and the report surfaces
both.

---

## (f) What remains + risks

**Landed (this PR):**
- `approaches/analysis/seed_variance.py` + `__init__.py` — seed-sweep + variance
  report, three backends (`mock` keyless/no-API default, `aggregate` from disk,
  `go-explore` live keyless), verified in all three modes.
- This plan doc.

**Remaining (needs Step 4C and/or paid runs):**
- Wire the sweep to the Step-4C **primitives** six-floor curriculum tier once it
  lands (the live backend currently drives `CurriculumEngineEnv` via
  `curriculum_go_explore`; point it at the 4C tier when available).
- A continual-harness `--tier curriculum` backend for `seed_variance` (parse the
  loop's `run_log_<runid>.json` per seed) so the loop feeds the same report.
- Real paid sweeps: the LLM continual-harness loop and the Voyager reverse
  curriculum across ≥3 seeds × ≥4 reps, then `--source aggregate` over the
  outputs for the headline variance table.
- Plots (reuse `approaches/voyager/analyze_reverse_curriculum.py` /
  `plot_final.py` style): depth-vs-seed scatter, coverage CDF, stall-iteration
  histogram.

**Risks / caveats:**
- **Cost.** Voyager and the LLM loop call Prime Inference every turn; a single
  rollout can burn millions of tokens (see `docs/EVAL_RECIPES.md`). Keep the
  default analysis keyless (mock/aggregate/go-explore) and gate paid runs.
- **Engine is process-global.** The NetHack C engine is a singleton per process;
  the reverse-curriculum sweep already isolates each episode in a subprocess.
  A multi-seed *live* `go-explore` backend runs seeds sequentially in one process
  (safe: each `reset` reseeds), but do NOT parallelize live seeds in-process.
- **`goto_abs` construction.** Some seeds' `goto_abs` silently lands on floor 1
  (observed seed 22), faking a "win"; `construct_start` already rejects these —
  keep that guard when extending the sweep.
- **Dependency on 4C.** The reduced-scale claim rests on Step 4C; until it merges,
  the live backend uses the existing compressed curriculum, which is close but not
  identical to the 4C primitives tier.
- **Variance vs. small N.** `depth_cv` over 3 seeds is noisy; treat <5-seed
  numbers as directional and prefer ≥8-seed sweeps for headline claims.
```
