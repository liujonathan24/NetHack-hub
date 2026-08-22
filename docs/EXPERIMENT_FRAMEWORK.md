# Experiment framework — what to run, in what order, and why

Status: **plan**. Written 2026-08-16. Consolidates the three-part study structure
agreed with the advisor into a concrete, ordered experiment list, grounded in
what this repo can already do.

Companions: `docs/HARNESS_DEFECTS.md` (validity traps — read §4 before designing
any cell), `docs/experiments/exp_ablations_level_mod.md` (the knob sweep, already
scaffolded), `RUNBOOK.md` (how to launch), `docs/NLD_HUMAN_DATA.md` (human corpus).

---

## 0. The thesis, in one paragraph

Models cannot play NetHack. The interesting question is not *that* they fail but
*which layer* fails, and whether that layer is a property of the model or of the
harness we wrapped around it. This program answers that with **paired ablation**:
for each hypothesised bottleneck we can either remove the difficulty from the
**world** (engine difficulty knob) or hand the agent a perfect solution from the
**harness** (oracle substitution). Running both, against a **policy bracket**
(scripted floor, oracle ceiling), turns "the agent got deeper" into an
attribution. The study ends at the strongest available negative result: give the
agent the answer key distilled from human play and ask whether it can convert
knowledge into a successful run.

Three parts, in dependency order:

| Part | Question | Status |
|---|---|---|
| **1. Metrics** | What does "progress" mean, and what is BALROG missing? | mostly assembly — the code exists |
| **2. Attribution** | Which layer fails: perception, planning, control, or survival? | the negative result; needs one logging fix first |
| **3. Test-time learning** | Can more test-time compute fix it, or is it a model limit? | the "answer key" ladder |

---

## 1. Part 0 — the bracket (run this before anything else)

Nothing in Parts 1–3 is interpretable without it, and it is nearly free.

### The problem it solves

`docs/netplay-vs-our-harness.md` concluded that our plateau was **the exploration
layer, not the LLM**. `docs/HARNESS_DEFECTS.md` then measured that **34% of every
rollout** went to loops, failed calls and zero-clock no-ops against **2.0%** spent
descending, and that fixing the harness alone moved the identical config from
**2.45 ± 0.34 → 3.71 ± 0.49** BALROG (mean Dlvl 3.0 → 6.0, deaths 5/5 → 1/5).
`docs/CURRICULUM_RL_RESULTS.md` has the counter-example: on seed 19 the stairs
were reachable, `move_to`/A* were correct, `open` verifiably worked, and GLM 5.2
still failed to produce a 19-step sequence in 45 turns — an *agent* limitation.

Both stories are true for different cells. Without a reference policy you cannot
tell which one a given number is.

### B1 — scripted policy floor (no LLM, $0)

Run the skill layer in a loop with no model: `explore_and_descend` /
`np_explore_level` driven by a fixed controller, same seeds, same budget.

Precedent exists and is validated in the ascent setting:
`approaches/voyager/scripted_nav_reachability.py` — *"the best a legal-primitives
agent could do if its reasoning were perfect… a HARNESS property, not an
agent-reasoning property."* The reverse-curriculum sweep already reports scripted
greedy-nav at 0.30 / 0.00 / 0.00 for climb-from-2/3/4 against GLM's 0.667 / 0.0 / 0.0.

**Work:** generalise that script from ascent to descent on `full_nle`. Half a day,
no API spend, covers every seed the model budget cannot.

**New metric it unlocks — model lift:**

```
lift = (agent_score − scripted_score) / (ceiling_score − scripted_score)
```

If lift ≈ 0, the model contributed nothing and every model comparison in that
cell is measuring the scripted layer. This is the honest answer to "what is
BALROG missing" and it is the single most important number in the program.

### B2 — replay floor (harness fidelity, ~$0)

Replay a known-good action stream through the *full* harness — rendering,
prompt auto-dismissal, macro execution — and confirm it reaches depth.
`tools/record_demo.py` + `legacy.replay::TrajectoryRecorder` record it;
`resume_from` (in `NetHackVerifiersEnv.__init__`) already replays a prior cell's
turns NDJSON into a fresh engine and continues.

Any depth the harness cannot replay is not a model failure. This also guards the
exp2 defect class where 402 stubs were scored as shallow rollouts.

### B3 — random policy floor

Uniform-random over the same skill surface. Cheap, and it calibrates how much of
BALROG's low end is reachable by noise. Relevant because BALROG's `Dlvl:2` is
worth 1.54% and several published cells sit near there.

---

## 2. Part 1 — metrics

Most of this is **already implemented**; the deliverable is a spec plus a
rescoring of existing runs, at $0.

### What BALROG is missing (each point has evidence in this repo)

1. **`max` over axes inflates stuck runs.** `balrog_progress` takes
   `max(Dlvl_value, Xp_value)`. Rescoring BALROG's own published episodes with
   `min`: Gemini 3 Pro **6.77 ± 3.59 → 1.84 ± 0.61**; Gemini 3 Flash
   **3.96 ± 0.87 → 0.37 ± 0.37** (4 of 5 zeroed). Flash's headline was carried by
   Dlvl 8 at XP 1, and by **Dlvl 1 held for 12,527 steps at XP 5**. Our BBOX cell
   held **3.71 → 2.35 ± 0.07, 0 of 5 zeroed**. `balrog_progress_min` and the
   `xp_carried` flag already exist in `nethack_harness/prompt/balrog.py` and
   `tools/eval_metrics.py::balrog_columns`. **Always report max and min together.**
2. **No policy floor.** No scripted or random baseline, so model contribution is
   unidentifiable. → B1 above.
3. **One budget axis.** A harness that does more game-turns per LLM call looks
   better per call and worse per turn. `pace_columns` already emits
   `depth_per_game_turn`, `depth_per_llm_call`, `balrog_pct_per_game_turn`,
   `balrog_pct_per_llm_call`. Report the pair **and their ratio**
   (game-turns/call = harness macro-ness). Measured spread is large:
   `netplay_true` runs 7.0–10.5 game turns/call, `balrog80` runs 1.1–1.4 — the
   latter needs **3.8× the LLM calls to play half as much NetHack**. Numbers from
   different action surfaces are not comparable without this.
4. **Max-over-episode, no terminal-state taxonomy.** Report the distribution over
   `died / starved / stuck / budget_exhausted / harness_error`. The vocabulary
   exists (`tools/eval_instrument.py::FAILURE_MODES`,
   `trace_schema.TOOL_STATUSES`) and **has never been populated in a committed
   run output**. Current `stop` distribution over 121 rollouts: `game_over` 63,
   `agent_completed` 20, `None` 13, `call_budget_exhausted` 12,
   `harness_timeout` 7, `error` 3, `max_output_tokens` 3 — i.e. ~1 in 5 rollouts
   ends for a reason that is not the game.
5. **No human calibration.** We have it and nobody else does — see below.
6. **Action surface is not declared.** Any reported NetHack number should carry
   its action surface (`balrog80` = raw keystrokes vs `netplay_true` = 31 macros)
   and its game-turns/call. Otherwise the comparison is meaningless.
7. **n is below the variance floor.** `B0`, `B1` and `N` are **byte-identical
   encodings** and scored **2.31 and 4.48 in the same sweep**. That 2.2-point
   spread is the benchmark's noise floor. n=5 cannot separate anything;
   `BBOX_JSON` went 4.05 ± 1.33 (n=5) → 3.25 ± 0.58 (n=10). Prefer **more seeds
   over longer caps**: doubling the turn cap moved BBOX 3.71 → 3.78 while 0/10
   seeds reached the 800 cap.

### The headline human-calibration result (already computed, in `/root/nld`)

This reframes "models fall off over time" and should lead the paper.

| turn | NAO human median BALROG (n alive) | top-10 human (n alive) | agent, GLM-5.2 vision (n alive) |
|---|---|---|---|
| 515 | 1.54 (20,273) | 1.54 (4,205) | **1.85 (19)** |
| 1,002 | 1.85 (16,378) | 1.85 (3,910) | **2.65 (1)** |
| 1,950 | 2.42 (10,450) | 2.65 (3,318) | — (0) |
| 10,306 | 3.54 (1,736) | 7.45 (1,808) | — |

**Agents do not fall off on rate. They fall off on duration.** Through turn
~1,000 GLM-5.2 with vision is at or above the human median. The median agent game
ends at **turn 333**; 1 of 56 rollouts is alive at turn 1,000 and **zero** past
1,347. The human median game runs to 1,675 turns, the top-10 median to 5,865, and
every point of human separation is earned after turn 2,000 — where the agent
population is empty. Ascensions take a median of ~42,000 turns.

Human pace for reference (top-10 players, first turn reaching Dlvl N, median):
Dlvl 2 at 194, Dlvl 3 at 461, Dlvl 5 at 1,411, Dlvl 10 at 6,481, Dlvl 15 at 12,100.
So **agents die before the human median even leaves Dlvl 2–3**.

Caveat to nail down first, at $0: how much of "ends at turn 333" is death versus
our own budget cap. Death rates in the cells are 80–100%, and exp3's plan notes
median death at ~110–150 calls with 2/10 rollouts reaching the 200 cap — so it is
mostly death, but this must be reported per-rollout, not asserted.

### Metric spec to adopt

Always report, per cell: `balrog_max`, `balrog_min`, `xp_carried` count,
`max_dlvl`, terminal-state distribution, `game_turns`, `llm_calls`,
game-turns/call, both pace rates, `$`/rollout, **model lift vs scripted**, and
turns-to-milestone at BALROG 1/2/5/10/20% (`tools/progress_rate.py::MILESTONES`).
Compare **turns-to-milestone**, never raw %/turn — the curve is concave and a
rate measured over the first 600 turns is not comparable to an ascender's.

Everything above except lift is already implemented in `tools/eval_metrics.py`
and `tools/progress_rate.py`.

---

## 3. Part 2 — attribution (the negative result)

### 3.1 Blocking prerequisite: reasoning is not being logged

Across **all 202 NDJSON files / 15,759 turn records** in `outputs/`:
`rendered_user_content` is present on **100%**, `raw_grid` on **100%**,
`assistant_message` on **1 record (0.006%)**, `reasoning` on **0**,
`tool_results` on **0**. No `traces.jsonl` survives anywhere under `outputs/`,
so the v3 backfill in `tools/trace_reasoning.py` has nothing to read.

**The reasoning for every existing run is unrecoverable.** A perception/planning
taxonomy needs the agent's own words. Fix before spending anything on Part 2:
run at schema v3, retain `traces.jsonl`, verify `reasoning.available` and
`tool_results` are non-empty on a smoke cell. Everything else in Part 2 assumes
this is done.

### 3.2 The decomposition

Treat the agent as a pipeline and give each stage a measurable failure signature.
Note the **fourth** category, which the usual perception/planning/control split
omits and which this harness demonstrably needs:

| Layer | Definition | Automatic signature |
|---|---|---|
| **Observability** | the harness never showed it | diff `gt_obs` (engine planes) against `rendered_user_content` |
| **Perception** | shown but misread | model's stated belief contradicts `raw_grid`/`gt_obs`; proxy: call targets a coordinate whose content contradicts the call (`move_to` into a wall, `attack` on an empty tile) |
| **Planning** | read correctly, wrong subgoal | no perception error, skills succeed, but no progress — e.g. stairs known and reachable and the agent does something else |
| **Control** | right subgoal, execution failed | skill **postcondition** not met |

Observability is not hypothetical here: `HARNESS_DEFECTS` §1.6 found the MAP was
not byte-faithful to the engine grid and **rendered real walkable doors as wall**
across the agent's only early exit, under the encoding whose whole point is
"lights on".

**Control needs postconditions, not statuses.** `TOOL_STATUSES` documents that
`completed` "says nothing about progress" — the vendored NetPlay skills return
`Step.completed()` without inspecting the result. Define per-skill postconditions
(`explore_and_descend` → dlvl increased; `move_to(x,y)` → player at (x,y);
`attack` → target gone or HP dropped) and evaluate them from `gt_obs` before/after.
That converts control failure from prose into a number. `trace_diagnostics.py`
already has the raw material: zero-clock calls per skill, the game's own message
on a wasted call, thrash (identical name+args repeated), all-null arguments, and
"% of published coordinates that never steered a call".

Validate the classifier against ~200 hand-labelled turns and report agreement.
Do not ship an unvalidated taxonomy.

### 3.3 The knob grid — world-side difficulty removal

The fork carries a **17-knob continuous difficulty catalog**
(`third_party/NetHack/src/include/nle.h`, `NLE_TUNE_FIELDS`), threaded through
`load_environment(tune=...)`. I verified every knob has real engine read sites:

| knob | default | read site | layer it probes |
|---|---|---|---|
| `reveal_map` | 0.0 | `win/rl/winrl.cc` | perception / exploration |
| `vision_radius` | 0.0 | `vision.c` (6 sites) | perception |
| `room_density` | 1.0 | `mklev.c` | perception / search |
| `room_size` | 1.0 | `sp_lev.c` | perception / navigation |
| `corridor_connectivity` | 1.0 | `mklev.c` | navigation / planning |
| `locked_door` | 1.0 | `mklev.c` | control (kick/unlock sequences) |
| `trap_density` | 1.0 | `mklev.c` | control / hazard handling |
| `monster_speed_scale` | 1.0 | `mon.c` | control (timing) |
| `hunger_rate_scale` | 1.0 | `eat.c` | planning (resource management) |
| `xp_gain_scale` | 1.0 | `exper.c` | planning (the neglected BALROG axis) |
| `ongoing_spawn_scale` | 1.0 | `allmain.c` | survival |
| `mob_spawn` | 1.0 | `mklev.c` | survival |
| `monster_difficulty_scale` | 1.0 | `dungeon.c` | survival |
| `dmg_to_player_scale` | 1.0 | `mhitu.c` | **survival — the duration knob** |
| `dmg_by_player_scale` | 1.0 | `uhitm.c` | survival / combat |
| `player_hp_scale` | 1.0 | `attrib.c` | survival |
| `hp_regen_scale` | 1.0 | `allmain.c` | survival |

`tools/ablation_sweep.py` already implements four named cells — `baseline`,
`full_vision`, `infinite_health`, `unlocked_doors` — with free local
verification (`--verify`), dry-run env-args, and `--emit-cmds`. The design doc is
`docs/experiments/exp_ablations_level_mod.md`. **It has never been run against a
model.** This is the shortest path from here to a result.

#### Measured: which knobs preserve map layout at a fixed seed

Paired-seed comparison is only valid if the knob does not reshape the dungeon.
I tested this (terrain footprint hash under `reveal_map=1`, seeds 42/7/101):

| knob | terrain identical to baseline? | note |
|---|---|---|
| `locked_door=0.0` | **yes** (3/3 seeds) | same map, doors just unlocked — clean paired comparison |
| `trap_density=0.0` | **yes** (3/3) | terrain footprint identical; only contents change |
| `mob_spawn=0.0` | **yes** (3/3) | same |
| `monster_difficulty_scale`, `hunger_rate_scale`, `dmg_to_player_scale` | **yes** (3/3) | live knobs, no generation effect |
| `room_density=0.5` | **yes** — but it is a **no-op** at this value | natural room count is already under the cap |
| `room_density=0.1` | no | open cells 614 → 357; the knob only binds well below 0.5 |
| `corridor_connectivity=2.0 / 0.0` | no | 614 → 643 / 586 |
| `room_size=2.0 / 0.5` | no | 614 → 529 |

**Consequences.** Layout-preserving knobs (including the whole survival group and
`locked_door`) support tight within-seed paired comparisons at small n.
Layout-changing knobs (`room_size`, `corridor_connectivity`, `room_density` at
values that bind) generate a *different dungeon* and must be treated as
between-subjects with n ≥ 16 against the 2.2-point variance floor. And
`room_density` in the range you would naively try does nothing — check that a
knob binds before spending on a cell.

#### The inference rule, stated carefully

"Flip a knob; if performance jumps, the model could not handle that difficulty"
is *almost* right, and wrong in a way that matters: removing a difficulty helps
**any** policy. The identifying quantity is the interaction, not the main effect.
Every knob cell must therefore also run against the **scripted policy** (B1).

- Model gain ≫ scripted gain → the difficulty was binding **on the model**.
- Model gain ≈ scripted gain → the world just got easier; says nothing about the model.
- Neither gains → the difficulty was not binding at all.

Because the knobs are continuous doubles, prefer a **dose-response curve**
(e.g. `dmg_to_player_scale` ∈ {0, 0.25, 0.5, 1.0}) over binary on/off. It is
strictly more informative and it defuses the saturation risk already flagged in
the design doc: `dmg_to_player_scale=0` plus 9999 HP can make every cell hit the
depth cap, at which point the metric cannot rank anything.

Two knob-specific traps: generation knobs (`locked_door`, `room_*`, `mob_spawn`,
`trap_density`) are read **only while a level is built**, so they must be passed
to `reset(tune=...)` — setting them live is a silent no-op. And `blstats` HP is
display-capped at `min(hp, 9999)`, so an agent reasoning about HP fraction sees a
capped number in the infinite-health cell.

### 3.4 The oracle ladder — agent-side capability grants

The world-side twin of each knob. Same difficulty, opposite intervention.

| rung | intervention | already available? |
|---|---|---|
| A0 | baseline (best config: `BBOX`, `netplay_true,reveal`, 400 calls) | yes — `HARNESS_DEFECTS` §5 |
| A1 | **perception oracle** — `reveal_map=1`, JSON entity map, explicit stair/frontier annotations | yes (`tune`, `variant=JSON`, `E1`/`E2` frontier variants) |
| A2 | **control oracle** — provably-complete explore + reliable combat | **needs work**: persistent `has_seen`/`search_count` + prioritised search mask, per `netplay-vs-our-harness.md` §5 |
| A3a | **generic plan** — static expert checklist from `wiki/snapshot.json` | yes (`system_prompt` overlay in `configs/harness/`) |
| A3b | **human-derived level-indexed plan** — "by Dlvl 3 a strong human has done X" | needs the extractor in §4 |
| A4 | **seed-specific answer key** — this seed's layout, stair locations, level-by-level plan | needs a prior successful run of that seed |
| A5 | **exact action replay** — upper bound / sanity only, not a real arm | yes (B2) |

**The decisive cell is A1+A2.** If the model still fails with perfect perception
*and* perfect control, the deficit is planning — a genuine model-level claim. If
it succeeds, then NetHack-for-LLMs is a harness-engineering problem and we say
so. Pre-commit which result means which, because both are publishable and the
temptation to spin is real.

The 2×2 of world-knob × agent-oracle for the same difficulty is the sharpest
instrument in the program:

- world removal helps, capability grant does not → the model **cannot deploy** a
  solution it has been handed → planning/perception failure, not a missing tool.
- both help about equally → the model simply lacked the capability → ship the tool.
- both needed → interacting bottlenecks, and the gains will be sub-additive.

### 3.5 Depth-conditioned evaluation — the best single experiment for "why do they fall off"

Two competing explanations for degradation with depth:

- **H-difficulty**: the game gets harder (monsters scale with depth).
- **H-drift**: the agent degrades with context length and accumulated error.

**Discriminator:** compare an agent *teleported* to Dlvl k against an agent that
*walked* to Dlvl k. Same game state, different history. If fresh-at-depth
performs markedly better, it is drift, not difficulty.

Fully supported today: `EngineEnv.modify(goto_depth=n)` jumps the level **and
seats the hero on the downstair**; `level_up=n` grants real HP/stat gains;
`nle_goto_abs(dnum, dlevel)` does cross-branch jumps; `curriculum_upgrade.py`
carries a Valkyrie stat-by-depth model. Grid: start at Dlvl {1, 3, 5, 8, 12} with
human-calibrated stats, fixed budget, measure marginal progress and survival.

Caveat: `tools/build_valkyrie_model.py` is **blocked** (needs `nle.dataset` /
`_pyconverter`, and no `valkyrie_model.json` artifact exists anywhere). Either
unblock it or derive the stat prior from `/root/nld` status-line parsing, which
`nethack_core.nld_parse.parse_status` already extracts (str/dex/con/int/wis/cha).

Prior evidence that this will be informative: the reverse-curriculum sweep shows
**climb-from-2 at P=0.667 but climb-from-3 and climb-from-4 at exactly 0.0**
(n=12/12/11). Something falls off a cliff between floors, and it is not gradual.

---

## 4. Part 3 — test-time learning and the answer key

The pivot from "why they fail" to "can it be fixed". The answer key **upper-bounds
what any skill-induction method could deliver**: if handing over the answer does
not produce a win, no amount of self-generated skill discovery will, and the
limit is execution rather than knowledge.

### The ladder

| rung | what the agent gets | measures |
|---|---|---|
| R0 | **i.i.d. repeats, no memory** | the control: pass@k from resampling alone |
| R1 | file-backed memory across episodes | does carried state help? |
| R2 | self-authored skill library, curated (add useful, delete useless) | Voyager-style induction |
| R3 | **skills distilled from human traces**, then run *without* the traces | knowledge → procedure conversion |
| R4 | GEPA-style reflective evolution of prompt/skill set across runs | ([arXiv 2507.19457](https://arxiv.org/abs/2507.19457)) |

**R0 is not optional.** Sequential-with-memory must be compared against
pass@k at *equal total budget*, or a "learning" result is just resampling luck.
This is the control most commonly botched in the literature.

**Held-out seed transfer is not optional either.** Train on seeds A, test on
held-out seeds B. Without it you have shown memorisation, not skill. Add an
explicit memorisation check: does the induced library contain seed-specific
coordinates, and does it transfer to a different seed?

Read-outs: score vs episode index; **cost to reach Dlvl k vs episode index** (the
"will it be cheaper next time" question); transfer gap.

### What already exists

Substantial. `nethack_harness/refiner.py` implements a full teacher-refiner with
`RefinerEdits{prompt_addendum, subagents_set/delete, skills_set/delete,
notes_set/delete, objective}`, `SubagentSpec` with a trigger DSL (`hp_pct<0.4`,
`depth>=4`, `hostile_count>0`), `MacroStep`, and a `run_macro(name=...)` tool;
`bootstrap_dir` persists all four components across rollouts, which is the skill
library. `approaches/continuous_harness/` is a working champion-vs-challenger
loop. `approaches/voyager/` is the skill-library scaffold. `continual=True` with
`continual_lives=5` auto-resets on death while preserving chat, journal and
belief state. Policy ≠ teacher is enforced (`allow_same_teacher`).

Budget note for R4: GEPA needs many rollouts (fewer than RL, still many). Run it
on short-horizon tasks — the `six_floor_primitives` task spec or curriculum
tiers — not on full 400-call `full_nle` rollouts.

### Building the answer key from human data (§ feasibility confirmed)

`/root/nld` has 28,040 decoded NAO games (106.5M game turns, 433 ascensions,
96.5% agreement with the xlogfile on max Dlvl) and 12,154 top-10 games (1,703
ascensions). **Screens are recoverable exactly; keystrokes are not** — NAO ttyrecs
record terminal output only, so actions must be inferred from message lines and
`@`-position deltas. That is fine, and arguably better: it forces the answer key
to be *semantic* rather than a memorisable action list, which is exactly the
distinction the advisor drew.

The extractor is a small addition, not a project: `endgame_parse.py` already does
frame iteration → VT emulation → row extraction → `scan_messages()` regex bank
with turn stamps, validated at **precision 1.000 / recall 0.9977** against the
xlogfile on ascension detection. Adding a semantic event is appending a regex to
`ASC_PATTERNS` plus a byte prefilter. One measured session yielded 15,085
turn-stamped message states — 437 kills, 530 "You see here", 266 altar, 105 trap,
80 wishes — and ~10% of frames render a full inventory listing. Bucket events by
current Dlvl to get "by Dlvl N the player had done X, Y, Z". The xlogfile
`conduct`/`achieve` bitmasks give strategy labels for all 28,040 games with zero
parsing — the cheapest win available.

Cost: the full 126k-file NAO shard decodes in ~62 min on 12 procs; the endgame
scan over 10,089 files took 418 s on 14 procs. An afternoon of CPU for the whole
corpus, minutes for a few hundred exemplar games.

---

## 5. Ordered run list

| # | Experiment | Spend | Status | Gates |
|---|---|---|---|---|
| 1 | Terminal-state taxonomy over existing 202 traces — death vs cap vs harness error | $0 | `eval_instrument.FAILURE_MODES` exists, never populated | everything |
| 2 | Rescore all existing runs under the Part-1 metric spec | $0 | code exists | the metrics paper section |
| 3 | **B1 scripted floor** + B3 random floor, descent setting | $0 | port `scripted_nav_reachability.py` | all attribution |
| 4 | Fix reasoning logging (schema v3 + retain `traces.jsonl`); smoke-verify | ~$0 | §3.1 | all of Part 2 |
| 5 | B2 replay fidelity | ~$0 | `resume_from` exists | interpretation of every depth number |
| 6 | **Knob sweep, dose-response, each cell × {model, scripted}** — lead with `dmg_to_player_scale` | low | `ablation_sweep.py` ready | the bottleneck ranking |
| 7 | Failure classifier + 200 hand-labelled validation turns | low | needs #4 | the taxonomy figure |
| 8 | Depth-conditioned eval: teleported vs walked to Dlvl k | low | `modify(goto_depth=)` ready | difficulty vs drift |
| 9 | Oracle ladder A1, A2, **A1+A2** | medium | A2 needs the complete-explore rewrite | the model-level claim |
| 10 | Answer key A3b / A4 + human-trace extractor | medium | extractor is a small addition | the negative result's punchline |
| 11 | Part 3 R0 vs R1/R2/R3 on one seed family + held-out transfer | high | refiner/bootstrap exist | the test-time-compute claim |

**Start with #1, #3 and #6-with-`dmg_to_player_scale`.** Given the
duration-not-rate finding, the single most decisive cheap experiment is: *if the
agent cannot die, does it keep progressing at the human median rate?* If yes, the
whole negative result becomes one crisp sentence — LLM agents play NetHack at
roughly human pace and die five times too early. If it plateaus anyway, the
failure is exploration/planning and the plateau depth names the bottleneck.
Either outcome is a result, and it costs one knob.

---

## 6. Design rules to pre-commit

- **Pair everything by seed.** Use paired sign tests; the unpaired variance floor
  is 2.2 BALROG points. n=5 is exploratory only; n ≥ 16 for any unpaired claim.
  Existing significance levels are honest and weak: the best scaffold result is
  depth p=0.107, BALROG p=0.227, paired sign p=0.070 — "suggestive and
  replicating, not established".
- **Never mix model comparison with attribution.** Attribution runs one model
  family; a second model only checks the conclusion is not model-specific.
- **Every arm ships with its scripted counterpart.** No exceptions — that is the
  whole identification strategy.
- **Budget-match in both currencies** and say which one was matched. Control's
  `max_turns` caps LM turns while the CLI arms' referee grants exactly N executed
  skills; the existing `run1` table is explicitly **not** budget-matched.
- **Declare the action surface and game-turns/call** on every reported number.
- **Freeze the observation format for the duration of a sweep.** `HARNESS_DEFECTS`
  §4.1: numbers from before 2026-07-28 04:07 are not comparable to numbers after,
  because two of the repairs changed observation content. Re-bless the golden
  snapshots (`environments/nethack/tests/golden/`) and run them with `--check`
  immediately before cutting a sweep.
- **Verify a knob binds before buying a cell** (`ablation_sweep.py --verify`).
  `room_density=0.5` is a no-op; `BBOX` "withheld" nothing for an entire study
  because VISIBLE FEATURES published stair coordinates on ~100% of turns, so
  `reveal` fired on 2.1% of turns and 40% of rollouts never called it.
- **Arm the stall watchdog** (`STALL_WATCHDOG=1`). Two unbounded hangs are open
  and neither reaches `env.step`, so `no_progress_timeout` cannot fire.
- **Move `turns/<seed>_*.ndjson` aside before re-running a seed** — grading groups
  by seed prefix and a killed run's partial NDJSON merges with the relaunch.

---

## 7. Open questions

- The metrics-standardisation reference (`x.com/testingham/status/2082119317530030320`)
  could not be retrieved here (HTTP 402). Someone with access should fold its
  claims into §2 before that section is called complete.
- `room_size` reads in `sp_lev.c` (special-level generation) rather than
  `mklev.c`. It visibly changes ordinary levels in the test above, but the
  mechanism is worth confirming before the knob is used in a headline cell.
- `better_items_luck` still has no realization — no `luck` knob, no `luck` field
  in `_MODIFY_BOUNDS`. Options are ranked in `exp_ablations_level_mod.md` §e;
  the cheapest is adding `luck` to the `nle_set_state` whitelist.
