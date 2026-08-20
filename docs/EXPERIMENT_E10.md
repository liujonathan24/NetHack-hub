# E10 — Does past play make future play cheaper?

**Status:** mechanism built and smoke-tested end to end (2026-08-20). E9 has
finished. The experiment itself has not been run.

**Question.** After the agent has played NetHack *X* times, is playing time
*X+1* cheaper? Concretely: can an orchestrator read prior traces, edit what the
player is told (add the right guidance, delete the wrong guidance), and move the
cost-per-unit-progress curve — on seeds it has never played.

Everything below is the NPCORE_v3 control config (GLM-5.2, `prime_agent`,
`full_nle`, `BBOX_MIN`, Val-hum-neu-fem, `np_core,request_map,search`, 200
turns / 200 skill calls, `record_step_frames=true`), varied by one knob. See
RUNBOOK / docs/EXPERIMENT_E9.md.

---

## 1. What the scaffold actually permits

The design is forced by facts about Prime Agent 0.3.3 and this harness. All of
these were read out of the installed CLI and `harnesses/nethack-prime-agent/`,
not assumed.

### 1.1 Skill refresh in a live process

| Change | Visible to a running Prime Agent? |
|---|---|
| Edit the **body** of an already-registered `SKILL.md` | **Yes.** Only `name`+`description` go into the system prompt (`<available_skills>`); the body is read off disk by the agent when it decides the skill is relevant (`docs/skills.md` §How Skills Work). |
| Add a **new** skill, or change a name/description | **No.** The roster is scanned at process start. `/reload` re-scans, but it is a TUI/slash command — there is no `skills.reload` host request (the kernel's typed host-request surface is `goal.*`, `mcp.*`, `refine.*`, `rlm.*`), so an unattended `--print` agent cannot trigger it. |
| Add or change a **Python-backed** skill | **No.** Needs a fresh session for kernel setup to install and import it; a `pyproject.toml` change rebuilds the shared kernel venv. |
| Edit the **env / harness code** in the worktree | Irrelevant to a running agent; picked up by the *next* rollout, because `launch_cell.sh` resolves `REPO`/`PYTHONPATH` at launch. |

**Sub-agents do not help.** `rlm(...)` children "inherit the parent model,
provider configuration, skills, tools, retry policy, and resource loader"
(`docs/rlm.md` §2) — the parent's *startup* snapshot. A child cannot see a skill
created after its parent booted. Default `RLM_MAX_DEPTH = 1`, so children cannot
recurse further either.

**Design consequence:** do not hot-reload anything. Make every actor a fresh
process and keep the learning in files. Each verifiers rollout already launches
its own `prime-agent` process (`__init__.py:596`), so a player picks up whatever
is on disk at launch, for free.

### 1.2 One rollout is one game

The tool server publishes no reset / new-game tool: a rollout is a single
episode with a single toolset-side `max_skill_calls` referee. An orchestrator
*inside* a rollout could spawn `rlm()` children, but they would all drive the
same game against the same shared budget. **Players cannot be sub-agents of an
in-rollout orchestrator.** The orchestrator has to sit above verifiers and
launch rollouts.

### 1.3 What persists across rollouts, and what does not

- `install_dir = /tmp/vf-prime-agent` is **fixed on purpose** (the kernel venv is
  keyed on Python-skill paths) and has **no teardown** (`__init__.py:626`). It is
  bind-mounted **read-write** into the bwrap sandbox (`__init__.py:449`).
- `settings.skills = ["{install_dir}/skills"]` (`__init__.py:549`), and directory
  skill roots are discovered recursively for `SKILL.md`.
  → **`/tmp/vf-prime-agent/skills/<name>/SKILL.md` is read by every subsequent
  rollout, with zero code changes.** This is the one durable, shared channel.
- Everything else is per-rollout: `agent_dir = {install_dir}/agent-{trace.id}`
  (`:504`) and `PRIME_AGENT_CODING_AGENT_DIR` points settings, models, auth and
  sessions at it (`:570`), so "nothing here reads or writes the operator's
  `~/.prime/agent`" (module docstring).

### 1.4 The native continual harness — inert by default, and how it is switched on

Prime Agent ships exactly the mechanism E10 wants: `rlm.harness.create_skill /
update_skill / delete_skill / create_memory / create_prompt_note /
create_subagent`, persisting prompt notes, memories, reusable skill descriptions
and sub-agent specs, and rendering them into the system prompt of every new
session — `buildSystemPrompt` is called with `harnessState:
_loadMergedHarnessState()`, which is
`mergeHarnessStates(loadHarnessState(getGlobalHarnessStateDir(), "global"), local)`.
So a global entry written by one game is in the next game's system prompt.

Under this harness it is inert, for two independent reasons:

- *Local* state lives in session artifacts, and the arm runs `--no-session`
  (`__init__.py:597`) — so there are none. `rlm.harness` says so itself: "Use
  get_harness_state(global_=True) for global state."
- *Global* state lives at `<agentDir>/harness/`, and `agentDir` is the
  **per-rollout** `agent-{trace.id}`. "Global" is really "per game".

**The fix is one symlink.** `getGlobalHarnessStateDir()` is
`join(agentDir, "harness")` with no env override of its own, and the kernel is
handed that same path as `RLM_GLOBAL_HARNESS_STATE_DIR` (`rlm/harness.py:80`).
So linking `agent-<id>/harness` → a shared directory redirects both the host and
the kernel, and **only** the harness state: `settings.json`, `models.json` and
`auth.json` stay per-rollout, which they must — seeds run concurrently and each
carries its own MCP URL and interception secret, so sharing the whole agent
directory would race five rollouts onto one settings file and point agents at
each other's games. That is `harness.continual_harness_dir` (default `""`, which
leaves every existing arm byte-identical).

The store must live under `install_dir`: the sandbox binds that path and little
else, so a store outside it would leave the symlink dangling and every rollout
would start empty — which reads exactly like "the agent learned nothing". The
harness refuses to launch in that case rather than producing that result.

**What this does not switch on.** `_autoRefineAllowedForSession()` returns
`this._rlmDepth === 0 && this._localHarnessStateDir() !== undefined`, so under
`--no-session` automatic refinement never fires — and auto-refine writes *local*
entries anyway ("Only create/update/delete local harness entries"). Entries reach
the shared store only through an explicit `global_=True` call. For an experiment
that is a feature: every write is deliberate and attributable, not a background
process editing the arm mid-cell.

**Verified round-trip** (kernel venv, no prime-agent process): `create_memory`,
`create_skill`, `create_prompt_note` and `delete_memory` against
`RLM_GLOBAL_HARNESS_STATE_DIR` all persist to `harness_state.json` and are read
back by a second process. `create_skill` is validated — it requires
`reference={"type": "python", "import": ..., "callable"|"call_pattern": ...}` and
raises otherwise.

**The render budget is the real cap.** `formatHarnessStateForPrompt` shows at
most `DEFAULT_OVERVIEW_ENTRY_LIMIT = 6` entries **per kind**, each compacted to
`DEFAULT_OVERVIEW_CONTENT_LIMIT = 180` characters, plus the 5 most recent
refinement events. Everything past that is invisible to the player. So the
"delete something before you add something" pressure that makes §3.2 an actual
research question is enforced by the scaffold, not by us — we only have to hold
the orchestrator to it.

### 1.5 The orchestrator must not be resident during a cell

The wedge recipe is `pkill -9 -f 'prime-agent'`, which kills *every* prime-agent
process regardless of its agent dir. A daemon-backed resident orchestrator would
be killed by its own per-cell reset. So the orchestrator runs as **one
`prime-agent --print` invocation per round, strictly between cells** — never
concurrently with a rollout. Its state is the files it writes; a fresh process
each round is a feature, not a limitation.

---

### 1.6 Measured, not inferred (smoke test, 2026-08-20)

Every link in the chain was run:

1. **Delivery.** A store seeded by a separate process, read by
   `prime-agent --print --no-session --offline` against a stub OpenAI endpoint
   that logs what the CLI sends: both entries appear **verbatim in the system
   prompt** (19,722 chars), rendered as
   `- [global:<id>] <title> (<path>, v1): <content>`. Neither `traces.jsonl`
   (call metadata only) nor `turns/*.ndjson` records the agent's system prompt,
   so this is the only way to see it.
2. **The link under bwrap.** A store bound read-only on top of the read-write
   `install_dir` bind is readable and unwritable, the rest of `install_dir` stays
   writable, and `agent-<id>/harness -> <store>` resolves inside the sandbox.
3. **A real cell.** `launch_cell.sh prime_agent ... 10 1` with
   `CONTINUAL_HARNESS` set: completed, 10 skill calls, no errors, the resolved
   `config.toml` records both knobs, the symlink was created, and the store's
   sha256 was unchanged afterwards.
4. **End to end, behaviourally.** A store carrying one prompt note — "your very
   first tool call this episode must be search with times=1" — produced, in a
   real rollout through verifiers and bwrap:
   `turn 1: search(times=1)`, then `request_map`, then `np_move_to`. The control
   opens with `request_map`. The store reaches the player and changes what it
   does.
5. **The orchestrator.** `e10_orchestrate.sh` against E9's `NPCORE_v3_r3` (5
   rollouts): 9.5 minutes, exit 0, wrote **exactly 6 entries per kind, every one
   under 180 characters**, plus an 18-edit `orchestrator_rationale.json` whose
   evidence fields cite rollouts and turn numbers ("R2 T65: searched at 15HP →
   goblin hit for 8"). It found the explore/request_map thrash costing ~40% of a
   budget, gas spores exploding on melee kills, and prayer-anger deaths.

### 1.7 Two traps that cost a run each

- **The harness package does not come from your worktree.** `nethack_prime_agent`
  is an editable install in the shared venv pointing at
  `/root/NetHack-hub/harnesses/nethack-prime-agent` — the *main* checkout. The
  env code follows the worktree you launch from (RUNBOOK); the harness does not.
  A cell launched without
  `PYTHONPATH=$REPO/harnesses/nethack-prime-agent` validates against the old
  config class and dies with `--continual-harness-dir  Extra inputs are not
  permitted`, which reads like a bad flag rather than a stale package.
  `run_e10.sh` exports it.
- **`--model z-ai/glm-5.2` alone is a model *pattern*.** For the orchestrator it
  matched openrouter's catalog entry and died with "No API key found for
  openrouter" despite `defaultProvider = prime-inference` in settings. Pin
  `--provider prime-inference`. Credentials come from `~/.prime/config.json`,
  which is outside the agent dir, so a private orchestrator agent dir keeps them.

### 1.8 Players already write to this store, unprompted

An E9 control rollout (`NPCORE_v3_r3`, trace `dbc6a4df…`) called
`rlm.harness.create_memory(..., global_=True)` **twice** — `source: "agent"`,
version 2 — writing:

> "The harness auto-presses ESC for all multi-step command prompts. Eating,
> wearing, wielding, reading, quaffing, zapping and throwing are ALL impossible…
> Character will starve without ability to eat."

Nobody asked it to. It went into that rollout's per-rollout directory and was
discarded, which is precisely the waste E10 exists to stop. Two consequences:

- the self-directed arm (`CONTINUAL_HARNESS_WRITABLE=1`) is not hypothetical —
  the write behaviour is already there, only the destination is wrong;
- the orchestrator independently reached the same conclusion from the traces
  ("5/5 rollouts confirmed"), so this is corroborated from two directions and is
  worth checking as a possible **harness defect**, separately from E10. A `eat`
  skill *is* registered (`skills.py:470`), so either the `np_core` surface does
  not publish it or the multi-step prompt handling cancels it. If the agents are
  right, every E7–E9 character has been unable to eat.

## 2. Architecture

```
round r:
  [daemon reset]
  [orchestrator]  one `prime-agent --print` process, private agent dir whose
                  `harness` symlinks to the shared store
       reads   outputs/e10/round<r-1>/train__prime_agent/{turns/*.ndjson,traces.jsonl}
       writes  the store, via rlm.harness.{create,update,delete}_*(global_=True)
       writes  outputs/e10/round<r>/orchestrator_rationale.json
  [daemon reset]
  [train cell]    CONTINUAL_HARNESS=<store> launch_cell.sh prime_agent .../train  (seeds 0-4)
  [daemon reset]
  [eval  cell]    CONTINUAL_HARNESS=<store> launch_cell.sh prime_agent .../eval   (seeds 5-9)
```

`run_e10.sh` drives this; `e10_orchestrate.sh` is the orchestrator step. Each
cell snapshots the store before and after itself (contents + sha256) and warns
when a store we believe is read-only changed under it.


Cells sequential, seeds concurrent within a cell, daemon reset before each —
the E8/E9 pattern. `run_e10.sh` is modelled on `run_e9.sh`.

The orchestrator gets the run directory and its own IPython kernel; it writes
its own summarisation code rather than being fed a fixed digest. That is the
point of using an RLM as the orchestrator. It may fan out with `rlm()` children
to read seeds in parallel — that use *is* sound, because analysis children need
no post-boot skill refresh.

**Variation 1 (JEPA / offline) is the same loop with a scripted orchestrator.**
Build one loop and swap the driver; that also gives a free arm: LLM orchestrator
vs. a fixed script, on identical rails.

---

## 3. The learning carrier

### 3.1 Primary: the shared continual-harness store

`CONTINUAL_HARNESS=/tmp/vf-prime-agent/continual-harness/e10` on
`launch_cell.sh` (prime_agent only). Every rollout in the cell boots with the
accumulated entries rendered into its system prompt; the orchestrator edits them
between cells. **No change to `nethack.py`, no new env knob, no published tool
schema touched** — the learning lives entirely in Prime Agent's own store.

Preferring this over a hand-written playbook file is not just economy. The store
is typed (memory / skill / prompt_note / subagent), versioned per entry, carries
an id the orchestrator can update or delete by name, and is rendered by the
scaffold in a fixed budget — so "add the right skills, remove the wrong ones" is
a first-class operation with an audit trail, not prose editing.

### 3.2 Single-writer by default

`continual_harness_writable = false` (the default) re-binds the store read-only
inside the sandbox, on top of the read-write `install_dir` bind. Players read;
only the orchestrator writes. That keeps each cell reproducible from the snapshot
taken before it ran, and stops five concurrent seeds racing one JSON file
(writes are last-write-wins).

`CONTINUAL_HARNESS_WRITABLE=1` is the **self-directed variant**: the player
itself decides what to persist mid-game, via `rlm.harness.create_memory(...,
global_=True)`. That is a genuinely different and more interesting experiment —
online, in-episode learning rather than between-round curation — but it is
shared mutable state across concurrent seeds, so run it with `MAX_CONCURRENT=1`
and expect the cell to change what later cells read. Treat it as a follow-up
arm, after the curated version establishes there is signal.

### 3.3 Alternative carrier, kept in reserve

`/tmp/vf-prime-agent/skills/<name>/SKILL.md` is discovered by every later rollout
with zero code change (§1.3), and tests something harder: whether the player
*retrieves* the right lesson, at zero context cost when unused. The
continual-harness store is always in context; a skill file is only read if the
agent decides to. Run this after the primary carrier, as a retrieval arm.

## 4. Protocol

### 4.1 Seeds: train and held-out must be disjoint

The game seed is fixed, so re-running a seed re-plays the same dungeon;
improvement there is memorisation, which is a legitimate but *different* result.
Report both, never merge them.

- **Train pool:** seeds 0–4 (the existing control seeds).
- **Held-out pool:** seeds 5–9. Never analysed by the orchestrator. Its traces
  are not shown to it — enforce by path, the orchestrator is only given
  `round<r>/train`.

Held-out seeds have no baseline yet: **round 0 must run a no-harness control on
seeds 5–9** (≥3 reps, for the null band) before any entry is written.

### 4.2 Rounds and arms

| Cell | Playbook | Seeds | Purpose |
|---|---|---|---|
| `R0_train_ctl` | none | 0–4 | trace corpus for v1 (reuse E9 control reps where byte-identical) |
| `R0_eval_ctl` | none | 5–9, 3 reps | **held-out baseline + null band** |
| `R<r>_train` | store after round r | 0–4 | next corpus, and the memorisation curve |
| `R<r>_eval` | store after round r | 5–9 | **the headline curve** |
| `LEN_ctl` | 6 length-matched non-actionable entries | 5–9 | controls for "any extra prompt text helps" |
| `FROZEN_v1` | store after round 1, frozen | 5–9 | does *iteration* buy anything past round 1? |

`LEN_ctl` filler = the same entry count and content length of on-topic but
non-actionable text (e.g. wiki prose), so the render budget is identically full. This is the control that most cheap "the agent learned!" results
fail.

Optional `ANTI` arm (deliberately inverted lessons) if `LEN_ctl` comes out
ambiguous.

### 4.3 Staging

- **Stage 0 (pilot, ~3 cells):** R0_eval_ctl (3 reps) + one round of
  train→v1→eval. Purpose: does the plumbing work, is the orchestrator's edit
  non-degenerate, is the effect anywhere near the null band.
- **Stage 1 (full):** rounds 1–4 with LEN_ctl and FROZEN_v1.

Do not commit Stage 1 budget before Stage 0's null band is measured.

---

## 5. Metrics and decision rules

The headline is **cost per unit progress**, not raw cost. The store adds prompt
tokens on every turn, so `$/rollout` can rise while the agent gets strictly
better; and `$/rollout` can fall simply because the agent died sooner.

Primary endpoints, all already produced by `tools/cli_harness_eval/aggregate.py`:

1. `skill_calls` **to first reach dlvl 2** (and dlvl 3) — the direct "cheaper"
   measure. Right-censor at the budget; report as a survival curve, not a mean
   over completers only.
2. `max_dlvl` at fixed 200 calls — did the ceiling move.
3. `rollout_cost` (`PRICE_TABLE_GLM_5_2`) per rollout **and** per dlvl reached.
4. Death rate and `post_death_drain` — a store that trades survival for depth
   should be visible, not hidden inside a mean.

**Null band:** within-seed variance across ≥3 reps of the *same* cell (game seed
fixed, model sampling varies) — the existing E8/E9 rule. A round-over-round move
inside that band is not a result.

**Ratchet (this is what "remove the wrong skills" means operationally):** keep
`v<r+1>` only if held-out performance does not regress beyond the null band;
otherwise restore the previous round's snapshot and require the orchestrator to
propose a different edit. Every edit is already recorded twice — in the store's
per-entry version and in `orchestrator_rationale.json` — so the final store is
auditable entry-by-entry against the round and the evidence that introduced it.

---

## 6. Hazards

- **Cross-cell contamination.** The store lives under `install_dir`, which
  persists across cells *and across experiments* and has no teardown. A control
  cell run after an E10 cell would silently inherit the learned entries — and
  because `CONTINUAL_HARNESS` is opt-in, that only happens if the operator sets
  it, which is the point of making it an explicit variable rather than a default
  path. Every cell still snapshots the store before and after itself, so a cell
  that ran against a store it should not have is detectable after the fact.
- **Player self-writes.** Enforced away by default: the store is re-bound
  read-only inside the sandbox (later, narrower bwrap bind wins over the
  read-write `install_dir` bind), so a player physically cannot edit what later
  rollouts read. `run_e10.sh` still hashes before/after and warns on a change,
  because with `sandbox = false` nothing is enforced and the flag becomes a
  declaration of intent. This is the same class of failure as the zombie-seed
  incident, so it gets a mount, not a promise.
- **Orchestrator reward hacking.** It reads traces and writes the prompt; the
  cheapest "improvement" is an entry that games the metric (e.g. dive
  recklessly to raise `max_dlvl` while dying at turn 40). The death-rate and
  cost-per-dlvl endpoints exist to catch that; do not drop them.
- **Daemon.** Reset before every cell; never run the orchestrator while a cell is
  live (§1.5). `prime-agent status` lies.
- **E9 in flight.** Nothing in E10 runs until the E9a/control batch finishes.

---

## 7. Build order

1. **Done.** `harness.continual_harness_dir` / `continual_harness_writable` on
   `PrimeAgentHarness`: symlink `agent-<id>/harness` at the shared store, refuse
   a store the sandbox would not bind, re-bind read-only unless the writable flag
   is set. 7 tests in `tests/test_prime_agent_harness.py` (35 pass), covering the
   byte-identical default, the link name, per-rollout file isolation, both bind
   modes, the refusal, and the containment check.
2. **Done.** `CONTINUAL_HARNESS` / `CONTINUAL_HARNESS_WRITABLE` passthrough in
   `launch_cell.sh`, arm-guarded to `prime_agent`.
3. **Done.** `tools/cli_harness_eval/run_e10.sh` — round loop, daemon reset per
   cell, before/after store snapshots with hashes and a warning when a supposedly
   read-only store changed, held-out seeds pinned through `ENV_ARGS`.
4. **Done.** `tools/cli_harness_eval/e10_orchestrate.sh` — one `prime-agent
   --print` per round against a private agent dir whose `harness` links to the
   shared store, with the render budget and the skill-reference contract in its
   prompt.
5. **Done.** Smoke test, all five links — see §1.6.
6. **Next.** Stage 0 pilot (§4.3): round-0 baselines on both seed halves, then
   one full round. Nothing about the mechanism is unverified; what is unknown is
   whether the entries change the *outcome*.

## 8. Open decisions

- Held-out pool costs roughly double. Confirm the budget before Stage 1.
- Whether the orchestrator may edit *only* the harness store, or also propose
  env-knob settings (e.g. turn `reflect` on). The latter is a bigger claim ("it
  tunes its own harness") and a much larger search space; recommend
  store-only for E10.
- Whether to run the self-directed (`CONTINUAL_HARNESS_WRITABLE=1`) arm at all in
  E10, or hold it for E11. It answers a different question — in-episode vs.
  between-round learning — and it forfeits single-writer reproducibility.
- Whether to also run the seed-matched (memorisation) arm as a headline result
  rather than a diagnostic.
