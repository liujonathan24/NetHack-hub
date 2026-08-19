# E8 — Planning injection vs. mechanic guidance (control / perception / planning)

Date: 2026-08-19. Baseline: `outputs/e7_seed/NPCORE_v3__prime_agent` (np_core, raw
prompts, full vision, per-move frames; median bal_max 2.12, 5/5 died).

## Observed behaviors driving this round (from the 20-rollout death analysis)

1. **Prayer is a mechanic the model does not understand.** 5 of 9 dying runs
   prayed at 25–45% HP (above the divine-heal band, wasting the once-per-game
   button); v3 seed 1 prayed twice at FULL HP; two runs prayed on the death turn.
2. **It dives without regard for XL.** Every corpse was underleveled for its
   depth (XL1@Dlvl7 at the extreme). Both budget survivors kept XL ≈ Dlvl.

Mapped onto control / perception / planning: the failures are **planning**
(when to descend) and **control-policy knowledge** (how prayer works), not
perception — the HP-drop interrupt fired before all 20 deaths.

## The human norm table (new data, `nethack_harness/prompt/human_norms.json`)

Derived from the 433 NAO **ascended** runs (first arrival at each depth,
XL inverted from the BALROG min-axis; quartiles tight):

| Dlvl | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|------|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
| median XL | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 8 | 9 | 10 | 11 | 11 | 11 |

Note what it says: winners ALSO reach Dlvl 3 at XL1 — the early dive is
human-normal. The divergence is Dlvl 5–7, where winners arrive XL3–5 and our
agents arrived XL1–3 and died. The norm line should therefore bite exactly
where the deaths happen.

## E8a — planning injection (descent gate)

**Question:** if we insert the plan into the scaffold, does the model improve
automatically?

**Mechanism (new env knob `descent_gate`, default `"off"` — no existing arm
changes):** intercept `np_down` in `_apply_tool_call_inner`. The FIRST
`np_down` on a given dungeon level returns `interrupted` with zero engine
steps and a gate line; a second `np_down` on the same level proceeds
unconditionally (soft gate — agency preserved, game never blocked). State:
`state["_descent_gate_ack"] = set of dlvls acknowledged`.

Two prompt variants (the only difference between the arms):

- `descent_gate = "norm"` (descriptive, human-anchored):
  `[descent check: you are XL {xl} on Dlvl {d}. Typical successful human runs
  reach XL {norm} before leaving this depth. Call np_down again to descend.]`
- `descent_gate = "directive"` (imperative, no rationale):
  `[descent check: level to XL {norm} before moving on. Call np_down again to
  descend.]`

Norm values from `human_norms.json` (missing depth -> nearest lower).

**Cells** (5 seeds each, everything else byte-identical to NPCORE_v3 incl.
`record_step_frames`):
- `outputs/e8_plan/NORM__prime_agent`      — `descent_gate=norm`
- `outputs/e8_plan/DIR__prime_agent`       — `descent_gate=directive`
- control — reuse `outputs/e7_seed/NPCORE_v3__prime_agent` (no rerun).

**Readouts:** deaths/5, median bal_max/min, XL-vs-Dlvl at death (the target
variable), gate-acknowledgement behavior (how often it descends anyway —
measurable from the two np_down calls), turns spent grinding per level.

## E8b — mechanic guidance (prayer, then the guidance stack)

**Question:** do targeted hints for identified issues help — or does guidance
hurt, as NetPlay's own ablation found (depth 2.60 -> 2.00 when tactical goal
text was added)? This is the direct comparison point.

**Mechanism (new env knob `mechanic_hints`, default `""`):** appends a block to
the SYSTEM prompt only (published tool schemas untouched — frozen surface).

- `mechanic_hints = "prayer"` adds:
  ```
  === PRAYER (game mechanic) ===
  Praying heals ONLY when HP is critical (below ~1/7 of max, or under 6).
  Above that the god is "pleased" but restores nothing and the prayer is
  wasted. Prayer works roughly once per ~1000 turns; a second prayer too
  soon angers your god. Save it: pray only at critical HP.
  ```
- `mechanic_hints = "prayer,descend_pacing"` (the stack — tests the
  more-guidance-worse curve) additionally adds:
  ```
  === PACING ===
  Fight monsters on your current level until your experience level is close
  to the dungeon level before taking stairs down.
  ```

**Cells** (5 seeds each):
- `outputs/e8_hint/PRAYER__prime_agent`  — `mechanic_hints=prayer`
- `outputs/e8_hint/STACK__prime_agent`   — `mechanic_hints=prayer,descend_pacing`
- control — NPCORE_v3 again.

**Readouts:** prayer timing distribution (HP at each np_pray — the audit
already extracts this), wasted-prayer count, deaths/5, bal_max — and the
E8b-vs-E8a comparison: prompt-block guidance vs in-loop planning injection.

## Indexing / no-overwrite guarantees

- All new outputs land under `outputs/e8_plan/` and `outputs/e8_hint/` —
  no existing directory is touched; e6/e7 results stay as committed.
- Both knobs are NEW env_args defaulting off; a pinning test asserts the
  default renders byte-identical observations (same discipline as
  `auto_dismiss` / `record_step_frames`).
- Baseline reuse: NPCORE_v3 is the shared control for both sub-experiments
  (same code revision; the knobs default off there by construction).
- Launch pattern per cell (from repo root, healthy daemon):
  `VARIANT=BBOX_MIN STALL_WATCHDOG=1 ENV_ARGS='{"skill_set":"np_core,request_map,search","auto_dismiss":false,"record_step_frames":true,"descent_gate":"norm","tune":{"reveal_map":1.0}}' tools/cli_harness_eval/launch_cell.sh prime_agent outputs/e8_plan/NORM__prime_agent 200 5`

## Cost estimate

4 new cells x 5 seeds ≈ $100 at e7 prices (~$25/cell). Wallet ~$1,500.

## Order

1. E8a NORM (the headline experiment: planning injection).
2. E8b PRAYER (cheapest targeted fix, direct NetPlay comparison).
3. E8a DIR and E8b STACK (the framing/stacking contrasts).

---

# E8c / E8d — control (doors) and perception (density). Deferred: infinite life.

Run AFTER E8a/E8b. Infinite life (deathless planning probe) is deferred to a
later round (E8e): the descent gate already probes the planning axis, and a
deathless cell needs a much larger call budget to reach winning depth (~1,400
game turns at 200 calls vs ~42,000 for a human win).

## How the two knobs actually work (verified in the engine)

Both are **generation-time**, applied automatically in `mklev.c` every time a
level is created — set the tune value once in the cell config and every level
the agent descends into is generated with it. No per-level or per-step work.

- **`locked_door`** (`mklev.c:427`) scales the door-lock chance. The `rn2()`
  draw is preserved, so the RNG stream is unchanged and **layouts stay
  byte-identical to the baseline for the same seed** — only whether each door
  ends up locked vs closed changes. `locked_door <= 0` -> lock modulus 100000
  -> doors are effectively **never locked**. => E8c is seed-matched to
  NPCORE_v3; a clean paired A/B.
- **`room_density`** (`mklev.c:233`) caps room count at
  `round(room_density * MAXNROFROOMS)`, MAXNROFROOMS = 40. Fewer rooms consume
  different generation RNG, so the level **regenerates** — NOT seed-matched.
  Note the cap only bites BELOW NetHack's natural ~6-9 rooms/level, so the
  meaningful sweep is LOW values: 1.0 (baseline, cap 40, non-binding) ->
  0.25 (cap 10, ~vanilla) -> 0.1 (cap 4) -> 0.025 (cap 1, one big room).

## E8c — unlock every door (CONTROL)

**Question:** doors are an execution tax (kick sequences, direction prompts,
approach pathing, the `#`-extended-command trap that killed two NPCORE v1
seeds). Remove them and does progress improve?

**Cell:** `outputs/e8c_doors/UNLOCKED__prime_agent` — `tune.locked_door=0`,
everything else byte-identical to NPCORE_v3, seed-matched. Control = NPCORE_v3.

**Readouts:** deaths/5, median bal_max, and specifically the control-error
budget: `np_kick` calls, door-related failed `np_move_to`, freeze turns from
kick/direction prompts, `np_press_key('#')` occurrences (the trap). A control
win shows up as fewer wasted calls per depth, not necessarily a higher ceiling.

## E8d — minimum room density sweep (PERCEPTION, control-mixed)

**Question:** simpler topology = less map to read/model (perception), and also
mechanically fewer route-execution failures (49 of 58 baseline failures are
`np_move_to` no-route). With one big room, how far do we get?

**Cells** (5 seeds each, self-controlled within the sweep — NOT seed-matched to
the baseline; keep out of the baseline table):
- `outputs/e8d_density/D100__prime_agent` — `room_density=1.0` (in-sweep anchor)
- `outputs/e8d_density/D025__prime_agent` — `room_density=0.25`
- `outputs/e8d_density/D010__prime_agent` — `room_density=0.1`
- `outputs/e8d_density/D0025__prime_agent` — `room_density=0.025` (~one room)

**Readouts:** bal_max vs density (the dose-response), `np_move_to` failure rate
vs density (isolates the control component — if failures fall but bal_max
doesn't, the win was mechanical not perceptual), turns-to-first-descent.

## Indexing / no-overwrite

- New dirs `outputs/e8c_doors/` and `outputs/e8d_density/` — nothing touched.
- E8c seed-matched to NPCORE_v3 (RNG-preserved). E8d self-controlled.
- Both use only existing engine tune knobs (no new code), so no pins needed
  beyond the E8 knobs already merged. Full logging (frames + glyph ids) is on
  by the launcher default.
- Launch (per cell, with the per-cell daemon clean the E8 driver already does):
  `ENV_ARGS='{"skill_set":"np_core,request_map,search","auto_dismiss":false,"tune":{"reveal_map":1.0,"locked_door":0}}'`
  (E8c); swap `locked_door` for `room_density` per E8d cell.

## Category summary (control / perception / planning matrix)

- **Control:** E8c doors; (later) the pursue-until-dead vs single-swing melee A/B.
- **Planning:** E8a descent gate; (later) E8e infinite life.
- **Perception:** E8d density (control-mixed — read with the np_move_to-failure
  covariate); the observation-encoding comparisons (BBOX_MIN / reveal_map) are
  the cleaner perception axis. A pure perception probe (oracle stairs hint)
  remains the one open cell if we want the matrix fully square.
