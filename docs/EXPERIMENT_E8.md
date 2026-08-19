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
