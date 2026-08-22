# Baseline and tool-tier policy for the continual-harness experiment

*For the agent running continual harness optimization (Prime Agent with
persistent session state). Defines what to measure against, what counts as a
"manual fix", and how tool inclusion must be recorded for a publishable,
reproducible run.*

## The baseline is E10 — nothing more

The baseline for all continual-harness comparisons is **Experiment 10**: the
np_core surface on the honest harness, exactly as it ran on 2026-08-21
(10:44–14:39 UTC). Concretely: branch `exp/e8-planning-guidance` at the
honesty-pass state (commits `fd8aa13` + `9b8d5a4`, launcher `0458275`), cell
config `VARIANT=BBOX_MIN`, `ENV_ARGS = {"skill_set": "np_core,request_map,
search", "auto_dismiss": "false", "tune": {"reveal_map": 1.0}}`, GLM-5.2,
200 skill calls, seeds 0–4 (`tools/cli_harness_eval/run_e10_parallel.sh` is
the exact launcher). Reference numbers (15 games): **median BALROG 3.54, mean
5.34, 14/15 died, depth ceiling Dlvl 11**. Raw data:
`outputs/e10_baseline/NPCORE_v3H_r{1,2,3}__prime_agent/`.

## E11 and E12 are MANUAL fixes — the human-augmented tier, not the baseline

Everything added after E10 is a human optimization and must be reported as
such, never silently folded into the baseline:

- **E11 (gates)**: `descent_gate = "norm" | "enforce"` env flags. The code
  ships in the baseline branch but is **off by default**, so the baseline is
  byte-identical with it present; turning a gate on is a flagged, recorded
  intervention. Measured effect: changes leveling *policy* (XL/Dlvl 0.47→0.74
  under enforce), not outcomes.
- **E12 (NetPlay telemetry/affordance fixes)**: branch
  `fix/netplay-telemetry` (PR #29; commits `c1a0bec`, `aee5c43`, `ea8cc15`) —
  rest rewrite, interrupt severity filter + hunger interrupt, move_to/melee
  failure telemetry, seven missing keys (the Elbereth path), coordinate-frame
  documentation. Measured effect: efficiency (repeat-cluster budget 67%→35%),
  not outcomes (median unchanged).

Today the E10-vs-E12 distinction is a **branch checkout**, not a flag: E10 =
`exp/e8-planning-guidance` (pre-fix tool code), E12 = `fix/netplay-telemetry`.
The gates are already proper config flags. Every cell writes its resolved
`config.toml` and `engine_provenance.json` into its output dir — cite those in
anything published.

## Tool inclusion must move to a config file (three tiers)

For publication the run definition must be data, not a branch pointer.
Partition the tool surface in a versioned config (e.g.
`tools/cli_harness_eval/configs/tool_tiers.toml`) with three sections:

1. **`[base]`** — np_core exactly as it existed in E10: the 8 np_* skills +
   `request_map` + `search`, with the E10-era behavior. This tier is frozen;
   it is the published baseline.
2. **`[human]`** — the manually engineered fixes on top of base (the PR #29
   set, plus any gate flag used). `base + human` is the E12/E11 surface.
   Additions to this tier require a commit + a measured cell before they may
   appear in any pooled number.
3. **`[continual]`** — tools, prompts, or edits produced by the continual
   harness itself. This section is expected to MUTATE over time, so it gets
   different bookkeeping: every run must pin the exact contents (hash of the
   tier file + git commit of any generated code) into its output
   `config.toml`, and continual-tier runs are their own arm — never pooled
   with base or human tiers, and always compared against BOTH (base = what
   the model earns over the raw harness; base+human = whether the continual
   harness rediscovers or beats hand engineering).

A cell then declares `tool_tier = "base" | "human" | "continual"` in
`ENV_ARGS`, the launcher resolves it against the pinned tiers file, and the
resolved list + hashes land in the per-run `config.toml`. That makes any
published row regenerable from its output dir alone: checkout the recorded
commit, replay the recorded env_args.

## Continual-harness-specific notes

- Session-state persistence is a **scaffold change** (Prime Agent currently
  runs `--print --no-session`; the continual arm relaxes that and/or persists
  the `memory/` workspace across rollouts). It changes the scaffold under
  test, so it is part of the `[continual]` arm definition, never mixed into
  base/human cells.
- Known no-op to ignore: `set_avoid_monster_flag` / the `avoid_monsters`
  pathfinding parameter is silently dead upstream (`agent_base.py:230`), is
  not published on np_core, and stays unpublished. Do not spend continual
  optimization on it.
- Verification protocol (mandatory, learned the hard way): before any paid
  cell, read back the RESOLVED `config.toml` from the output dir and
  mock-play one seed in-process; a launch script's intent is not evidence.
  E9/early-E10 lost a day to silently dropped `reveal_map`/`auto_dismiss`.
