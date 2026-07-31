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

### 3.2 The spoiling-food interrupt and the standing-on-stairs fallback are inert
Both are present and both silently do nothing:
- `_spoiling_now` passes a `_RawView` (chars/glyphs/blstats only) into `shape()`,
  which needs `.message` → `AttributeError` swallowed by a bare `except` → always `[]`.
- `rendering.py` stores remembered stairs as `(depth,x,y)`; `sparse_map.py` tests
  `(x,y)` membership — can never match.

### 3.3 `reveal` silently clamps to 41 of 79 columns
`_REVEAL_MAX_W = 40`, undocumented. Requesting `x1=0,x2=78` returns
`(x0-40, y0-20)`. Worst under `BBOX`/`SPARSE_ONDEMAND`, where `reveal` is the
only map access.

### 3.4 Intermediate messages are dropped
`shape()` keeps only the last message, so a 100-step macro surfaces one line —
`You kill the little dog!` never reached the agent. `pre_visible_obs` already
holds them; the kill log now scans them, but nothing else does.

### 3.5 No reasoning is recorded
`assistant_message` is empty on every turn of every trace — the model replies
with a bare tool call. We can see *what* it decided, never *why*. This caps how
far trace debugging can go.

### 3.6 Death is not announced on the turn it happens
The death turn renders `HP: 0/N` with no death text and a stale HINT; the agent
learns on the *next* call, when every tool is already gated.

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
