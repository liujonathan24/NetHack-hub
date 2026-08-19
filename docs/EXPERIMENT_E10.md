# E10 — Does past play make future play cheaper?

**Status:** design. No code, no runs yet. E9a/control is in flight (owned by the
main agent); nothing here may touch the prime-agent daemon until that batch is
done.

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

### 1.4 The native continual harness does not carry across episodes *as configured*

Prime Agent ships exactly the mechanism E10 wants: `rlm.harness.create_skill /
update_skill / delete_skill / create_memory / create_prompt_note /
create_subagent`, plus `refine.run()`, persisting prompt notes, memories,
reusable skill descriptions and sub-agent specs, rendered back into the system
prompt at session start (`docs/rlm-runtime.md` §Continual Harness State). Under
this harness it is inert across episodes:

- *Local* harness state lives in session artifacts — and the arm runs
  `--no-session` (`:597`), so there are none. ("Local harness refinement requires
  a persisted session.")
- *Global* harness state lives at `<agentDir>/harness/harness_state.json`, and
  `agentDir` is the **per-rollout** `agent-{trace.id}`. So "global" is per-game.

Pointing the agent dir at a shared path would fix that and simultaneously share
the daemon supervisor state across rollouts — which is precisely the wedge that
`run_e9.sh`'s reset recipe exists to clear. **Do not do this for E10.** Use the
explicit file carrier instead; revisit `refine` only as a later variant, with
its own control.

### 1.5 The orchestrator must not be resident during a cell

The wedge recipe is `pkill -9 -f 'prime-agent'`, which kills *every* prime-agent
process regardless of its agent dir. A daemon-backed resident orchestrator would
be killed by its own per-cell reset. So the orchestrator runs as **one
`prime-agent --print` invocation per round, strictly between cells** — never
concurrently with a rollout. Its state is the files it writes; a fresh process
each round is a feature, not a limitation.

---

## 2. Architecture

```
round r:
  [orchestrator]  one `prime-agent --print` process, own kernel
       reads   outputs/e10/round<r-1>/**/turns/*.ndjson + traces.jsonl
       writes  playbooks/v<r>.md   (token-capped, see §3.2)
       writes  playbooks/v<r>.rationale.json  (what changed and why)
  [daemon reset]  the run_e9.sh recipe
  [train cell]    launch_cell.sh prime_agent outputs/e10/round<r>/train ... playbook=v<r>
  [daemon reset]
  [eval cell]     launch_cell.sh prime_agent outputs/e10/round<r>/eval  ... playbook=v<r>
```

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

### 3.1 Primary: `playbook` env knob (guaranteed delivery)

A one-knob ctor param on `NetHackEnv`, mirroring `descent_gate` / `reflect` /
`mechanic_hints`:

```python
playbook: str = ""          # path to a markdown file; "" = off, byte-identical
```

parsed with the string-coercion pattern (env_args arrive as strings), a no-op
when empty, with the block appended to the system prompt through a new
`nethack_harness/prompt/playbook.py` (like `reflection.py` / `human_norms.py`).
Passed as `ENV_ARGS={"...","playbook":"/root/nld/e10/playbooks/v2.md"}`. **No
published tool schema changes** — prompt-side only, per the frozen-tool-surface
rule.

Why primary: delivery is guaranteed, so the experiment measures *whether the
lessons are good*, not whether the player happened to retrieve them.

### 3.2 Token cap — the part that makes "remove the wrong skills" real

The playbook is capped (proposal: **1500 tokens**, enforced by the launcher,
which refuses to launch an over-budget playbook). Without a cap, "learning" is
indistinguishable from "the prompt got longer every round", and the orchestrator
is never forced to delete anything. With a cap, adding a lesson *requires*
evicting one — which is the actual research question.

### 3.3 Secondary: `SKILL.md` in the shared skills dir (agentic retrieval)

Write `/tmp/vf-prime-agent/skills/e10_playbook/SKILL.md`; every later rollout
discovers it at process start, no code change (§1.3). This tests something
different and harder: whether the player *retrieves* the right lesson at the
right time, at zero context cost when unused. Run it as a later arm, after the
primary carrier establishes there is any signal to retrieve.

---

## 4. Protocol

### 4.1 Seeds: train and held-out must be disjoint

The game seed is fixed, so re-running a seed re-plays the same dungeon;
improvement there is memorisation, which is a legitimate but *different* result.
Report both, never merge them.

- **Train pool:** seeds 0–4 (the existing control seeds).
- **Held-out pool:** seeds 5–9. Never analysed by the orchestrator. Its traces
  are not shown to it — enforce by path, the orchestrator is only given
  `round<r>/train`.

Held-out seeds have no baseline yet: **round 0 must run a no-playbook control on
seeds 5–9** (≥3 reps, for the null band) before any playbook is applied there.

### 4.2 Rounds and arms

| Cell | Playbook | Seeds | Purpose |
|---|---|---|---|
| `R0_train_ctl` | none | 0–4 | trace corpus for v1 (reuse E9 control reps where byte-identical) |
| `R0_eval_ctl` | none | 5–9, 3 reps | **held-out baseline + null band** |
| `R<r>_train` | `v<r>` | 0–4 | next corpus, and the memorisation curve |
| `R<r>_eval` | `v<r>` | 5–9 | **the headline curve** |
| `LEN_ctl` | length-matched filler | 5–9 | controls for "any extra prompt text helps" |
| `FROZEN_v1` | `v1`, never updated | 5–9 | does *iteration* buy anything past round 1? |

`LEN_ctl` filler = same token count of on-topic but non-actionable text (e.g.
wiki prose). This is the control that most cheap "the agent learned!" results
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

The headline is **cost per unit progress**, not raw cost. A playbook adds prompt
tokens on every turn, so `$/rollout` can rise while the agent gets strictly
better; and `$/rollout` can fall simply because the agent died sooner.

Primary endpoints, all already produced by `tools/cli_harness_eval/aggregate.py`:

1. `skill_calls` **to first reach dlvl 2** (and dlvl 3) — the direct "cheaper"
   measure. Right-censor at the budget; report as a survival curve, not a mean
   over completers only.
2. `max_dlvl` at fixed 200 calls — did the ceiling move.
3. `rollout_cost` (`PRICE_TABLE_GLM_5_2`) per rollout **and** per dlvl reached.
4. Death rate and `post_death_drain` — a playbook that trades survival for depth
   should be visible, not hidden inside a mean.

**Null band:** within-seed variance across ≥3 reps of the *same* cell (game seed
fixed, model sampling varies) — the existing E8/E9 rule. A round-over-round move
inside that band is not a result.

**Ratchet (this is what "remove the wrong skills" means operationally):** keep
`v<r+1>` only if held-out performance does not regress beyond the null band;
otherwise roll back to `v<r>` and require the orchestrator to propose a different
edit. Record every accepted/rejected edit in `v<r>.rationale.json` so the final
playbook is auditable line-by-line against the round that introduced each line.

---

## 6. Hazards

- **Cross-cell contamination.** `/tmp/vf-prime-agent/skills` persists across
  cells *and across experiments*. A control cell run after an E10 cell would
  silently inherit a learned skill. `run_e10.sh` must wipe/restore that directory
  around every cell, and every cell manifest must record a hash of the skills
  dir and the playbook file.
- **Player self-writes.** The sandbox binds `install_dir` **read-write**, so a
  *player* can write into the skills dir that later rollouts read — an
  uncontrolled learning channel, and the same class of failure as the zombie-seed
  incident. Hash the dir before and after every rollout and fail the cell on an
  unexpected change.
- **Orchestrator reward hacking.** It reads traces and writes the prompt; the
  cheapest "improvement" is a playbook that games the metric (e.g. dive
  recklessly to raise `max_dlvl` while dying at turn 40). The death-rate and
  cost-per-dlvl endpoints exist to catch that; do not drop them.
- **Daemon.** Reset before every cell; never run the orchestrator while a cell is
  live (§1.5). `prime-agent status` lies.
- **E9 in flight.** Nothing in E10 runs until the E9a/control batch finishes.

---

## 7. Build order

1. `playbook` knob + `nethack_harness/prompt/playbook.py` (+ test: off ⇒
   byte-identical prompt; on ⇒ block appended; over-cap ⇒ refuses).
2. `tools/cli_harness_eval/run_e10.sh` — modelled on `run_e9.sh`, plus the
   skills-dir wipe/hash and the playbook token-cap check.
3. Orchestrator prompt + `tools/e10_orchestrate.sh` (one `prime-agent --print`
   per round, given only `round<r>/train`).
4. Stage 0 pilot.
5. Commit as `Jonathan Liu <jl0796@princeton.edu>`.

## 8. Open decisions

- Held-out pool costs roughly double. Confirm the budget before Stage 1.
- Whether the orchestrator may edit *only* the playbook, or also propose env-knob
  settings (e.g. turn `reflect` on). The latter is a bigger claim ("it tunes its
  own harness") and a much larger search space; recommend playbook-only for E10.
- Whether to also run the seed-matched (memorisation) arm as a headline result
  rather than a diagnostic.
