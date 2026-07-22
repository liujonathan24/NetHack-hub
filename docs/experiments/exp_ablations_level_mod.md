# Experiment: Ablations with Level Modification

Status: **plan + scaffolding landed** (runnable sweep + free local verification;
paid eval not yet run). Owner: TBD. Engine knobs verified 2026-07-22.

## a) Goal & why we care

> Ablations with Level Modification. Evaluate our strategies with different
> difficulties removed: Full vision provided; Infinite health; Better
> items/luck; Unlocked doors. Analyze bottlenecks to agent performance.

NetHack failure is *multi-causal*: an agent can stall because it cannot see the
level, because it dies in combat, because it is under-equipped/unlucky, or
because it cannot get through a locked door. When we only observe the final
depth reached, we cannot tell **which** difficulty is the binding constraint.
This study removes one difficulty at a time (holding everything else at
vanilla) and measures how much progression recovers. The difficulty whose
removal moves the metric most is the current bottleneck — that is where harness
/ prompt / tool effort pays off.

## b) Infrastructure — the exact knob/modify API per ablation

Every ablation is realized through the fork engine's parametric difficulty
knobs (`env.tune`, catalog of 17) and secure-state pokes (`env.modify`,
whitelisted + bounds-checked). No game-content fork is required for the three
supported ablations.

### Engine surface

- **`env.tune`** (`EngineEnv.tune` / `RawEngine.get_tune`/`set_tune`) — the knob
  catalog discovered generically from the C `NLE_TUNE_FIELDS` X-macro
  (`third_party/NetHack/src/include/nle.h`). Full catalog (defaults in parens):

  `dmg_to_player_scale(1)`, `dmg_by_player_scale(1)`, `player_hp_scale(1)`,
  `hp_regen_scale(1)`, `vision_radius(0)`, `reveal_map(0)`, `hunger_rate_scale(1)`,
  `ongoing_spawn_scale(1)`, `monster_difficulty_scale(1)`, `monster_speed_scale(1)`,
  `xp_gain_scale(1)`, `room_density(1)`, `mob_spawn(1)`, `trap_density(1)`,
  `locked_door(1)`, `corridor_connectivity(1)`, `room_size(1)`.

  Two kinds of knob: **live** knobs (read every step / at render — e.g.
  `reveal_map`, `dmg_to_player_scale`, `hp_regen_scale`) can be set any time via
  `env.set_tune(...)`; **generation** knobs (read only while a level is built —
  e.g. `locked_door`, `room_density`, `mob_spawn`) must be present at
  `reset(tune=...)` so they reshape the floor as it is generated.

- **`env.modify(**changes)`** (`EngineEnv._MODIFY_BOUNDS`) — post-start secure
  state pokes: `hp(0..30000)`, `max_hp(1..30000)`, `gold(0..10_000_000)`,
  `xp_level(1..30)`, `hunger(0..2000)`, `level_up(1..29)`,
  `str/dex/con/int/wis/cha`, plus `goto_depth`. **Note:** internal `u.uhp` can
  reach 30000, but the emitted `blstats` HP is display-capped at `min(hp,9999)`
  in `botl.c` — the agent *sees* 9999 even when the hero has more.

### Per-ablation realization (empirically verified 2026-07-22)

| Ablation | Difficulty removed | How to realize | Status | C hook |
|---|---|---|---|---|
| **Full vision** | fog of war / exploration | `tune reveal_map=1.0` (live render overlay; whole-level terrain + live monsters filled into every obs; reversible, side-effect-free) | **Supported** | `win/rl/winrl.cc:699` (`fill_obs`) |
| **Infinite health** | combat lethality / survival | `tune dmg_to_player_scale=0.0` (hero takes **no** monster damage) — primary lever. Optional belt-and-suspenders: `hp_regen_scale=8`, `modify hp=9999,max_hp=9999` | **Supported** | `src/mhitu.c:2367`; regen `src/allmain.c:542` |
| **Unlocked doors** | locked-door / access barriers | `tune locked_door=0.0` at `reset(tune=...)` (scales the 1-in-6 lock roll; `<=0` ⇒ doors never rolled locked). Generation-time — cannot be toggled live. | **Supported** | `src/mklev.c:435` |
| **Better items / luck** | item scarcity / bad luck | **No knob and no modify field exists.** `modify gold=N` is only a weak buying-power proxy (no shops early, no items materialize). | **Needs engine/env work** — see §e | — |

How env-args thread the knobs in (unchanged from existing interface):
`load_environment(tune=..., modify=...)` (`environments/nethack/nethack.py:1290`)
→ stored as `setup_tune` / `setup_modify` → `NetHackCoreEnv(tune=, modify=)` →
`EngineEnv.reset(tune=...)` (which calls `engine.start(..., tune=...)`) and then
`EngineEnv.modify(**modify)` after start. At the CLI these are the `-a` JSON:

```
vf-eval nethack -m <model> -n 20 -r 1 \
  -a '{"tune": {"reveal_map": 1.0}}'
```

**Snapshot preservation.** Knobs live on the engine ctx, so `snapshot()`
captures them and `restore()` reverts to the snapshotted values (verified:
`hunger_rate_scale` 2.0 survives a 5.0→restore cycle; see
`environments/nethack/tests/test_tune.py::test_tune_is_captured_by_snapshot`).
This matters because the go-explore / branch paths snapshot mid-episode — the
ablation knob stays in force across restore, so a branched rollout keeps the
same difficulty removed. `modify` pokes are ordinary game state and are likewise
captured by the byte-exact snapshot.

## c) Narrative — remove one difficulty at a time, watch the bottleneck move

We run a **baseline** (vanilla) plus one cell per removed difficulty, all on the
**same seed set** so map layouts are identical and the only variable is the
ablated difficulty. Read it as a bottleneck ranking:

- If **full vision** alone recovers most of the lost progression, the agent is
  *exploration/navigation*-bound — it is dying to not knowing where to go, not
  to the game being lethal.
- If **infinite health** dominates, the agent is *combat/survival*-bound — it
  navigates fine but cannot survive fights.
- If **unlocked doors** moves the needle, the agent is *puzzle/tool*-bound —
  losing turns (or getting stuck) on kick/#force/unlock micro-decisions.
- Removing several and seeing *sub-additive* gains means the bottlenecks
  interact (e.g. vision helps only once you also survive long enough to use it).

The cell whose removal recovers the most progression is the difficulty to invest
against next.

## d) How to run

Scaffolding: `tools/ablation_sweep.py` (config set + runner) and
`environments/nethack/tests/test_tune.py` / `test_vision_overlay.py` (knob
effects, already green).

```bash
# 0. env for LOCAL (free) engine use
export NLE_LIB_PATH=$HARNESS/third_party/NetHack/src/build/libnethack.so
export PYTHONPATH=$HARNESS            # NetHackHarness root (provides nethack_core)

# 1. see the sweep cells and their support status
python tools/ablation_sweep.py --list

# 2. see the exact env-args each cell passes to load_environment (no engine, no API)
python tools/ablation_sweep.py --dry-run --base-env-args '{"explicit_seeds":[42,101,202]}'

# 3. FREE local check that every supported knob really takes effect (no model)
python tools/ablation_sweep.py --verify

# 4. print (do NOT auto-run) the paid vf-eval commands for the sweep
python tools/ablation_sweep.py --emit-cmds -m <model> -n 20 -r 1
```

**Sweep cells:** `baseline`, `full_vision`, `infinite_health`, `unlocked_doors`
(the unsupported `better_items_luck` is listed but excluded from the paid run).
Pin the **same** `explicit_seeds` across all cells so the comparison is
controlled.

**Metric — progression / depth bottleneck analysis.** Primary: **max dungeon
level reached** (`max_dlvl_reached`, tracked per rollout in `nethack.py`).
Secondary: depth-over-turns curve, turns-survived, and the existing scout /
tiles-seen reward. Per cell report mean max-depth and its delta vs baseline; the
**largest positive delta names the dominant bottleneck**. Because seeds are
pinned and rollouts are cheap-ish, 20 examples × 1 rollout per cell is a
reasonable first pass; widen `-r` for variance once the ranking is directional.

## e) What remains — not yet supported by a knob

**Better items / luck** has no realization today. Empirically confirmed
(`ablation_sweep --verify` `better_items_luck` cell): the tune catalog has no
`luck`/`item`/`loot` knob, and `_MODIFY_BOUNDS` has no `luck` field. `xp_gain_scale`
speeds leveling but is not items/luck; `modify gold=N` gives buying power but
early NetHack has no shop, so it does not put better gear in hand.

To support it, pick one (in rough effort order):

1. **Luck poke via `modify`.** Add `luck` to `EngineEnv._MODIFY_BOUNDS` and a
   `nle_set_state("luck", n)` case writing `u.uluck` (bounded, e.g. −13..13).
   Smallest change; realizes the "remove bad luck" half directly (luck feeds
   to-hit, prayer outcome, theft, etc.). C touch-point: the `nle_set_state`
   whitelist in `src/nle.c` alongside the existing `hp`/`gold` cases.
2. **Starting-inventory / gear injection.** A generation-time knob or a
   `modify` that injects a blessed kit (e.g. extra healing, better weapon/armor)
   at start — realizes the "better items" half. Larger: needs an object-creation
   hook and a curated kit spec; touches `src/u_init.c` / an `mksobj` helper.
3. **`item_quality` / `bones_luck` generation knob** in `NLE_TUNE_FIELDS`
   scaling drop quality or the blessed/cursed roll — the most "knob-shaped"
   option but the deepest engine change.

Until one lands, the sweep runs the three supported cells; `better_items_luck`
stays flagged `supported=False` and is excluded from paid evals.

## f) Risks

- **Over-powered cells mask signal.** `dmg_to_player_scale=0` + 9999 HP makes the
  hero effectively immortal; if *every* ablation cell reaches the depth cap the
  metric saturates and cannot rank bottlenecks. Mitigation: also report
  turns-to-depth / efficiency, not just max depth; consider softer settings
  (e.g. `dmg_to_player_scale=0.25`) if saturated.
- **HP display cap confuses the agent.** blstats shows `min(hp,9999)`; an agent
  reasoning about HP fraction sees a capped number. Usually harmless for a
  survival ablation but note it when interpreting HP-based agent behavior.
- **Generation vs live knob mistakes.** `locked_door` (and other generation
  knobs) only take effect at `reset(tune=...)`; setting them live via
  `set_tune` after the level exists is a silent no-op. The runner always routes
  them through the reset env-args to avoid this.
- **reveal_map ≠ knowledge, only observation.** Full vision reveals terrain +
  live monsters in the obs, but the agent must still *use* it; a weak
  navigation policy may not convert vision into depth, which is itself a finding
  (exploration is not the bottleneck) but must not be misread as "the knob
  didn't work".
- **Confound from single seed set.** Depth is high-variance in NetHack. Pin
  seeds across cells (done) and use enough examples that the baseline↔cell delta
  exceeds seed noise before ranking bottlenecks.
- **Cost.** Only `--emit-cmds` output spends money (remote model). `--verify`
  and `--dry-run` are free/local; never wire the paid path to run automatically.
