# Harness defects: found, fixed, and still open

Written 2026-07-31, at the end of an investigation that started as an
observation-encoding sweep and turned into a harness audit. Persisted here
because the findings are worth more than the numbers they invalidated.

**The headline.** Trace profiling over 171 rollouts / 8,934 LLM turns found the
agent was not playing badly — the harness was spending the run. **34% of every
rollout** went to loops, failed calls, zero-clock no-ops and recovery, against
**2.0%** on the descend action. Fixing that moved the same config from
2.45 ± 0.34 to 3.71 ± 0.49, mean depth Dlvl 3.0 → 6.0, deaths 5/5 → 1/5.

---

## 1. FIXED — wrong information reaching the agent

### 1.1 The HINT block recommended ~3,058 tools that do not exist
`=== HINT ===` was written against an older skill set and never reconciled with
what `netplay_true` publishes. Measured across 8 cells:

| hint says | times | actually bound |
|---|---|---|
| `search` | 1,722 | **nothing** |
| `descend` | 592 | `np_down` |
| `attack(direction=)` | 484 | `np_melee_attack(x, y)` |
| `kick(direction=)` | 371 | `np_kick(x, y)` |
| `engrave_elbereth` | 260 | **nothing** |

Fix: `rendering.py::_fix_hint_vocabulary`. Safe-renames where the signature
matches; **deletes the whole sentence** where the capability is unbound or takes
different arguments. Renaming `attack` → `np_melee_attack` was tried first and
was worse — it kept the `direction=` argument and produced a call that failed on
every invocation.

### 1.2 `melee_attack` force-fought pets and peacefuls, killing runs outright
`vendor/.../skills.py:389` issues `Command.FIGHT`, which deliberately skips
NetHack's `Really attack? [yn]` — and that prompt is also where our own peaceful
guard lives (`observations.py::_YN_NO_PATTERNS`). So the skill punched through
both. Observed live: it killed the pet the observation itself labels
`[PET - don't attack]`, and killed a shopkeeper, whose retaliation killed the
character.

Fix: `netplay_true.py::_refuse_attack`, a pre-flight guard on pets (from the
glyph) and 11 peaceful classes (by name — NLE exposes no peaceful flag).

### 1.3 Statues and the `I` marker were invisible
Both draw a **monster letter** but are not monsters: excluded from
`glyph_is_monster`, from the NetPlay tracker, **and** from the feature scan. So
the agent attacked them, got `There is no monster at (x,y)` at **zero
game-turn cost**, saw an unchanged observation, and repeated — an absorbing
loop. One rollout burned turns 36–400 (365 identical calls, 91% of the run).

Frequency: statues visible on Dlvl 1 in **4 of 7 seeds**. Full-map cells hit
this **810 times in 3,600 turns**; `BBOX`, which hides the grid, hit it **once
in 693**.

Fix: statues named in `=== VISIBLE FEATURES ===` (`features.py::statues`).
**The MAP is deliberately NOT rewritten** — the grid must stay a faithful render
of the engine. Derived knowledge belongs in the derived sections.

### 1.4 The NetPlay tracker desynced whenever the harness stepped the engine
`data.update()` runs only inside `agent.step()`, but the menu auto-dismiss loop
(up to 8 steps) and `rollback` both call `env.step` directly. Result: map,
`VISIBLE MONSTERS` and the tracker disagreeing three ways.

Fix: `reset_agent_cache()` after the auto-dismiss loop and after `rollback`.

### 1.5 Dungeon-level changes rendered as numpy reprs
`blstats` is a numpy array, so `(dnum, dlevel)` formatted as
`(np.int64(0), np.int64(2))` under numpy 2.x. Note `dnum` is the dungeon
**branch** (Dungeons of Doom / Gnomish Mines / Sokoban…, `dat/dungeon.def`), not
a seed. Now renders `Changed dungeon level from 1 to 2`, mentioning the branch
only when it actually changes.

### 1.6 The MAP was not byte-faithful to the engine grid, and nothing checked
§1.3 says the grid "must stay a faithful render of the engine". Nothing tested
it, and it was not true. Diffing the trace's `raw_grid` (from `tty_chars`)
against the rendered `=== MAP ===` (from `chars`) on seed 0 under
`tune={"reveal_map":1.0}` shows two cells differing from turn 2 on — (16,5) and
(11,10), `+` on the tty, `|`/`-` on the map — and both are absent from VISIBLE
FEATURES. Reproduced in the committed trace
`outputs/trace_probe/B0_reveal/turns/0_28800_1785554140.ndjson`.

They are **real, walkable doors**: `np_move_to(16,5)` takes the hero through one
and out of the starting room. The agent was shown a wall across its only early
exit, under the one encoding whose whole point is "lights on".

Cause — **not** the NetPlay tracker (the MAP never touches it). `reveal_map` is
a render-time overlay in the engine (`win/rl/winrl.cc :: fill_obs`). It runs
`cvt_sdoor_to_door()` on secret doors, which permanently converts them in
`levl[][]` so the revealed level is genuinely connected — but it only *paints* a
cell that is still unknown or was secret **on that frame**. So the frame that
converts the door paints it into `chars`, `glyphs` and `tty_chars`; on every
later frame the cell is neither unknown nor secret, the guard skips it, and
`chars` falls back to the wall face the hero remembers. `tty_chars` keeps the
door because the overlay writes straight into the exported buffer and NLE
re-copies only *dirty* terminal lines. The engine emitted the truth once and
stopped repeating it.

Fix: `prompt/engine_grid.py` reconciles the two planes back into one grid, in
the direction of `levl[][]` — a cell `chars` calls a wall face that the engine's
own tty still draws as `+` is restored. Deliberately narrow, because tty is the
plane that can carry menu prose (the 10.6% bleed that moved the MAP off tty in
the first place): wall faces only, `+` only, and never on a tty row containing a
4+ letter run. No harness memory is involved — drop the tty plane and it
degrades to plain `chars`. `prompt/features.py` reads the same grid, so the MAP
and VISIBLE FEATURES can no longer disagree about a door.
`tests/test_map_engine_fidelity.py` diffs MAP against `raw_grid` across the
first 6 turns under both fog and reveal and asserts they agree, and pins that
fog does **not** gain the (still-secret) doors.

Residual, engine-side, not copied into the MAP: a monster that moved can linger
on `tty_chars` for a frame or two (dirty-line refresh) while `chars` is already
correct — staleness in the other direction.

---

## 2. FIXED — information the agent could not have

### 2.1 Corpse rot was invisible and caused ~18% of deaths
NetHack computes rot from `otmp->age` (when the monster *died*), but **the
corpse's name never changes** — `a newt corpse` reads identically at 5 turns and
500. `otmp->age` is not exposed through the RL interface. 30 rollouts show the
taint message; **all 30 died**, 29 within 9–18 game turns, at a median of 100% HP.

Fix: `corpse_age.py` reconstructs age from a kill log (`You kill the X` + game
turn), rendering `[~12 turns dead - ROTS IN 18 MOVES.]`. Threshold derived, not
guessed: `eat.c:1673` sickens at `rotted > 3`, `rotted = (moves - age)/(10 +
rn2(20))`, **+2 if cursed** ⇒ worst case `moves - age > 30`.

Corpses the agent did not kill report `age UNKNOWN`, never a guess — those are
the dangerous ones (median 48 game turns between kill and pickup, p75 280).

### 2.2 Feature lists truncated with `+N more`
`DISPLAY_CAP = 3` silently dropped the coordinates the block exists to deliver;
a closed door vanished and the agent could not tell whether it had opened or been
truncated. Under `SPARSE` the feature list **is** the map. Now uncapped.

### 2.3 Skills reported success regardless of outcome
`create_position_command` / `create_inventory_command` yield `Step.completed()`
without reading the result, so `Skill 'kick 43 5' completed` was emitted
identically for two kicks that bounced and one that smashed the door open.
Fix: append the game's own message — `GAME: As you kick the door, it crashes
open!`.

### 2.4 `rollback` was published but unadvertised
The tool worked; it had no `_SKILL_BLURBS` entry, so it appeared in the tool list
with no prompt guidance and was called **0 times in 5 rollouts**. Adding a blurb
took it to 8+. Same failure mode as the journal tools (unused across 15
rollouts). **Publishing a tool is not the same as offering it.**

---

## 3. STILL OPEN — read before running unattended

### 3.1 Two unbounded hangs, neither catchable by `no_progress_timeout`
- `np_explore_level` spins in vendored pathfinding (`nethack_bfs`) — **no return
  in 600 s**, versus 0.0–0.5 s normally. Intermittent, no death or rollback
  involved.
- Post-death `rollback` then any NetPlay skill deadlocks the adapter (~60 leaked
  threads). Reproduced 3/3.

**Neither reaches `env.step`**, so the engine's own `no_progress_timeout` can
never fire. **A stall watchdog is mandatory**: kill any rollout whose newest
`turns/<seed>_*.ndjson` has not been written in ~300 s. Note the watchdog exits
when the eval queue drains, so it must be re-armed for each batch.

**Mitigation shipped** (the hangs themselves are still unfixed):
`tools/stall_watchdog.py`, armed per batch with `STALL_WATCHDOG=1` on either
launcher or via `tools/with_stall_watchdog.sh`. It kills the PID named in the
turn filename plus its descendants, quarantines that PID's turn files to
`turns.stalled/` so the retry cannot merge with them (§4.6), and logs the seed,
PID, idle time and last recorded turn. Silence is measured per PID rather than
per seed — one process owns every seed of a cell, so a seed that merely
*finished* would otherwise look stalled. See `RUNBOOK.md`.

### 3.2 The spoiling-food interrupt and the standing-on-stairs fallback were inert — **BOTH FIXED**
- **FIXED.** `_spoiling_now` passed a `_RawView` (chars/glyphs/blstats only) into
  `shape()`, which needs `.message` → `AttributeError` swallowed by a bare
  `except` → always `[]`. The interrupt had never fired. `_raw_view` now returns
  the env's own `_last_observation` (a complete `CoreObservation`), and the list
  fallback builds a view carrying **every** published key rather than three.
  Verified firing end-to-end inside a live `explore_level` macro
  (`tests/test_spoiling_interrupt.py`), on `corpse_age`'s own thresholds
  (RISKY_AGE 30, URGENT_WINDOW 5) — no rot arithmetic is duplicated here.
  A `b"corpse" in inv_strs` fast path keeps the per-engine-step cost near zero
  (`shape()` is ~220 µs and this runs after every step of a ≤100-step macro).
  **The swallow itself was the real defect**: `[]` meant both "no spoiling food"
  and "this function is broken". Guards in `netplay_true.py` now go through
  `_swallowed(site, exc)` — counted in `swallowed_exceptions()`, logged at
  WARNING with a traceback on first occurrence — and
  `NETHACK_STRICT_SKILL_ERRORS=1` re-raises instead of swallowing.
- **FIXED.** `rendering._remember_stairs_down` stores `(depth,x,y)`;
  `sparse_map.py` tested `(x,y)` membership, so the fallback could not fire on
  any input. **Verified live before the fix** (seed 0, `reveal_map`): one real
  render populated the memo as `{(1, 57, 13)}`, i.e. the test the sparse map
  performs for a hero on those stairs is `(57,13) in {(1,57,13)}` → False.
  It matters because under SPARSE the entity list *is* the map and `@` covers
  its own tile, so this was the only channel that could say "you are standing on
  `>`". `sparse_map.py` now reads the memo through the same depth-aware
  `rendering._remembered_stairs_down` helper the HINT ladder uses, so producer
  and consumer cannot disagree about the key shape again; legacy 2-tuple entries
  still resolve. `tests/test_sparse_map_under_player.py` wires the real producer
  to the real consumer and also pins the cross-floor case the depth key exists
  for.

### 3.3 `reveal` silently clamped to 41 of 79 columns — **FIXED**
`_REVEAL_MAX_W = 40`, undocumented, applied to the exclusive difference
`x2 - x1`, so `x1=0,x2=78` returned `(x0-40, y0-20)` and said nothing. Worst
under `BBOX`/`SPARSE_ONDEMAND`, where `reveal` is the only map access.

The caps are now `_REVEAL_MAX_COLS = 79` / `_REVEAL_MAX_ROWS = 21` — the whole
map — because the request is clamped to the grid **before** the size cap, so the
worst case a caller can reach is the map itself. Measured: a full 79×21 reveal
is **1,807 characters**, against 1,328 for the `=== MAP ===` section a full-map
variant sends *every* turn and 2,613 for a whole observation. The blowup the cap
was written against does not exist. The clamping machinery is kept for
experiments that want it, but every clamp now (a) appends a
`NOTE: partial view — …` sentence naming the columns/rows withheld, and (b)
increments `skills.reveal_clamp_counts()` (`calls` / `out_of_range` / `too_wide`
/ `too_tall` / `truncated`, mirrored per-episode on `env._reveal_clamps`), so a
sweep can measure the rate instead of nobody finding out for months.

### 3.4 Intermediate messages are dropped
`shape()` keeps only the last message, so a 100-step macro surfaces one line —
`You kill the little dog!` never reached the agent. `pre_visible_obs` already
holds them; the kill log now scans them, but nothing else does.

### 3.5 No reasoning is recorded — FIXED (schema version 3)
Was: `assistant_message` empty on every turn of every CLI-arm trace (0 of 12 and
0 of 20 on the two committed reference artifacts; 9 of 29 on the control arm),
and `tool_calls` empty on 100% of CLI-arm turns. We could see *what* it decided,
never *why*.

Two different causes, two different fixes:

* **`tool_calls`** was empty for no good reason. The dispatched name and
  arguments are the arguments to `_apply_tool_call` — `tool_results[i].name` was
  already built from them — but the field was filled only from a *parsed*
  assistant message, which exists only inside the in-process v0 rollout loop.
  It is now synthesized on the MCP route, and every record carries
  `dispatch_route` (`"harness"` / `"mcp"`) saying which route wrote it.
* **The reasoning** genuinely is not visible to the tool server: a CLI agent
  talks to the interception endpoint in another process. So the record now says
  so — `reasoning.available = false` with a `reason`, never a silent `""` — and
  the words are recovered afterwards from `traces.jsonl`'s sampled assistant
  nodes, which is where every arm's model calls are recorded.
  `NetHackTask.finalize` does that join live; `python -m tools.trace_reasoning
  <run_dir>` does it over any completed run, including old ones.

The join is verified, not assumed: it must reproduce the `assistant_message` the
control arm already wrote inline, or it refuses to write anything. Measured
396/396 exact on `outputs/pilot_reveal`. Where a scaffold makes a 1:1 mapping
impossible — Prime Agent runs skills inside `ipython`, so its model turns do not
correspond to skill dispatches — the records say `available: false` with that as
the reason rather than attaching a plausible paragraph to the wrong move.

Still missing, and honestly labelled rather than guessed: reasoning for any arm
whose model calls do not pass through the interception endpoint, and per-turn
attribution for `ipython`-mediated arms.

### 3.6 Death is not announced on the turn it happens — **FIXED**
The death turn rendered `HP: 0/N` with no death text and a stale HINT (observed:
"Hostile adjacent (SE). Call `attack(...)` — your HP is healthy." on a corpse);
the agent learned on the *next* call, when every tool is already gated and the
one action that still works has never been mentioned.

Fix: `rendering._game_over_block`, emitted **first**, above JOURNAL, and it
suppresses the HINT ladder. Detected from the observation — `hitpoints <= 0`, or
NetHack's own death-screen phrases in the messages or on the tty — *not* from
`state["died"]`, which `env_response` sets after this render; that ordering is
exactly what made the announcement a turn late. It names HP/Dlvl/turn, quotes
the game's own cause line, says every further call is refused, and (only when
`rollback` is in this rollout's published set) says rollback still works.
`tests/test_death_announcement.py` drives a real `modify={"hp":0}` death and
asserts the *death turn's own* content says so.

### 3.7 `rendering._PUBLISHED_TOOLS` is a process-global that is never restored — **FIXED**
`load_environment` -> `render_system_prompt(published_tools=...)` wrote a
module-level set, and `_fix_hint_vocabulary` read it to delete HINT sentences
naming unbound tools (correct in production: `search` is not bound under
`netplay_true`). Nothing ever put it back, so *booting an environment changed
how every later render in the same process behaved*. Harmless in a real rollout,
but it made render-level tests order-dependent: a test that boots an env
truncated the exit hint asserted by `test_hint_actionability`, which passes in
isolation.

Fix: the global is gone. `render_system_prompt` is now pure, and the resolved
adapter names are written once per rollout into
`state[rendering.PUBLISHED_TOOLS_STATE_KEY]` by `setup_state`;
`_fix_hint_vocabulary(hint, published_tools)` takes the set explicitly and
rewrites nothing when it is empty ("unknown publisher → leave the text alone",
the same default the global had at import). The save/restore workaround in
`tests/golden/obs_configs.py` is deleted. `tests/test_published_tools_scope.py`
pins both ends, including the original order-dependence; the five golden
snapshots are byte-identical before and after, so production behaviour is
unchanged.

---

## 4. Experiment-design traps (cost real results)

### 4.1 The baseline moved mid-investigation
§1–2 are **harness-wide**, not axis flags — and two of them (statue naming,
feature un-truncation) are literally observation content. **Numbers from before
2026-07-28 04:07 are not comparable to numbers after.** The same `BBOX` config
scored 3.58 ± 1.55, then 2.45 ± 0.34 pre-repair, then 3.71 ± 0.49 post-repair.

Nothing pins the observation format. **Add a golden-file test** so the next fix
fails loudly instead of silently rebasing every result.

### 4.2 `B0`, `B1` and `N` are the same encoding
All three are bare `canonical(name)` — same template, same obs spec, same system
prompt. Verified byte-identical on a live game. They scored 2.31 and 4.48 in the
same sweep; that 2.2-point spread is **pure seed noise between identical
configs**, and is the best available measure of the benchmark's variance floor.

### 4.3 `BBOX` never actually withheld anything
It hides the ASCII grid, but `VISIBLE FEATURES` publishes stair/door/item
coordinates on ~100% of turns — so `reveal` fires on **2.1% of turns** and 40% of
rollouts never call it. The delivery-timing axis was void until `SPARSE_ONDEMAND`
folded every entity under `MAP`.

### 4.4 `max(dlvl, xp)` rewards single-axis play
BALROG progression is a max, so a run that dies on Dlvl 2 still scores ~2% off
its experience level. Rescoring BALROG's own published per-episode traces
(`github.com/balrog-ai/experiments`, `submissions/LLM/<model>/nle/`) with
`min()` instead:

| model | official (max) | min | zero-scoring episodes |
|---|---|---|---|
| Gemini 3 Pro | 6.77 ± 3.59 | 1.84 ± 0.61 | 1/5 |
| Gemini 3 Flash | 3.96 ± 0.87 | **0.37 ± 0.37** | **4/5** |
| our `BBOX` 400 | 3.71 ± 0.49 | **2.35 ± 0.07** | **0/5** |

Flash's headline is carried by degenerate runs — Dlvl 8 at XP 1, and Dlvl 1 held
for 12,527 steps at XP 5. Always report the xp-carried count alongside the mean.

### 4.5 n=5 cannot separate these configs
Every cell that gained seeds moved ~1 point. `BBOX_JSON` went 4.05 ± 1.33 (n=5)
→ 3.25 ± 0.58 (n=10). The same seed under the same config produced Dlvl 5 alive
and Dlvl 3 dead on different runs — the seed fixes the dungeon, LLM sampling
diverges the policy. **Prefer more seeds over longer caps**: doubling the turn
cap moved `BBOX` from 3.71 → 3.78 while 0/10 seeds reached the 800 cap.

### 4.6 Operational: killed rollouts corrupt the retry
`turns/` is shared per cell and grading groups by seed prefix, so a dead run's
partial NDJSON **merges with the relaunch**. Always move
`turns/<seed>_*.ndjson` aside before re-running a seed.

Also: ~30% of rollouts died to a login-node CPU ceiling at concurrency ≥6 and
needed relaunching. Cores were not the constraint — Slurm compute nodes have
**no outbound network** (`HTTP 000` to the inference endpoint), so they cannot
run LLM evals at all.

---

## 5. Reproduction: the best config

`BBOX` at a 400-turn cap — 3.71 ± 0.49 (n=5), mean Dlvl 6.0, 4/5 alive at cap,
**0/5 xp-carried** (the only cell where every score came from descent).

```
base:      tools/encoding_eval/configs/encoding_base.toml
model:     google/gemini-3-flash-preview
task_spec: full_nle      character: Val-hum-neu-fem      compact_obs: false
overrides: {"variant":"BBOX", "skill_set":"netplay_true,reveal,rollback",
            "belief_state_interval":25, "max_turns":400,
            "explicit_seeds":[N], "n_examples":1}
launch:    tools/encoding_eval/launch_encoding_cell.sh BBOX <outdir> 400 1
grade:     nethack_harness.prompt.balrog.balrog_progress(max_dlvl, max_xp)
           over each rollout's turns NDJSON; count a rollout only if
           s<N>/traces.jsonl is NON-EMPTY (empty files are created at start).
```
