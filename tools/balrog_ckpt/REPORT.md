# BALROG snapshot/restore feasibility: Crafter, MiniHack, TextWorld

Goal: can a Go-Explore-style layer (save a state, later restore it and branch)
sit on top of BALROG's naive agent for Crafter, MiniHack and TextWorld?
Everything below was measured on this box (Linux, CPU, python 3.12.13,
numpy 2.5.3, gym 0.23.0) in a fresh venv with BALROG `b7afe79`, balrog-nle
0.9.0, minihack 0.1.6+3ecb6da, textworld 1.6.2rc1, crafter 1.8.3, jericho 3.3.1,
and the NetHack-engine fork at `/root/NetHack-engine` (`18ad13d`, submodule
`7a872c4`, `libnethack.so` built Aug 28). No LLM calls.

Scripts: `tools/balrog_ckpt/` (`verify_crafter.py`, `crafter_ckpt.py`,
`verify_minihack_replay.py`, `verify_minihack_engine.py`, `verify_textworld.py`,
`common.py`, `run_all.sh`). Raw outputs: `tools/balrog_ckpt/results/*.json`,
engine crash logs in `results/crashlogs/`.

"Identical branches" below always means: play K prefix actions, snapshot, then
3 x [restore, play the same 50 fixed actions], with the live env deliberately
mutated (20 extra actions) between restores; every per-step record
(observation text, image/array hashes, reward, done, info) must match.
"No alias" means the saved snapshot's bytes are unchanged after the branch
mutated the live env and the state right after a restore equals the state at
snapshot time.

## Summary table

| env | install ok | native snapshot | approach that works | identical branches verified | snapshot ms | restore ms | replay cost / step | notes |
|---|---|---|---|---|---|---|---|---|
| Crafter (`default`) | yes | no API | **pickle of inner `crafter.Env` (minus gym `_np_random`) restored in place** (B) — also deepcopy-inner (A) and explicit `_world`/player/RNG copy (C) — **all require the 3-line determinism patch** (`crafter_ckpt.apply_determinism_patch`) | **yes**, 5/5 seeds for A, B, C with patch; without patch only 2/5, 1/5, 1/5 | 0.4 (B) / 0.5 (C) / 0.8 (A) | 0.5 (B) / 0.6 (C) / 0.8 (A) | 3.8 ms (a step) | blob 120 KB (A/B), 36 KB (C). `copy.deepcopy(env)` of the BALROG stack fails (numpy-2 vs gym 0.23 `_np_random`). Crafter's despawn logic iterates a `set` of objects -> restored copies diverge unless sorted. |
| MiniHack, path 2: balrog-nle replay (`Quest-Easy`, `Boxoban-Medium`) | yes | no (`_pynethack` API: `get_seeds/set_seeds/set_initial_seeds` only) | reset with the *effective* seeds (`get_seeds()` after the first reset) + replay the action prefix; clear BALROG's `NLELanguageWrapper.done` latch on reset | **yes**: 3/3 replays identical on Quest-Easy (13 steps to death), Quest-Easy lava-safe actions (30+50), Boxoban (30+50) | n/a (store seeds + actions) | 4.3 ms / 13 steps, 7.3 ms / 30 steps (Quest), 13.1 ms / 30 steps (Boxoban) | 0.21-0.30 ms per replayed step (+1.4-5.4 ms reset) | `env.reset(seed=s)` draws `disp` from `SystemRandom`; in our tests that did not change any observation, but store `get_seeds()` anyway. Obs keys recorded below. |
| MiniHack, path 1: NetHack-engine fork + MiniHack `.des` | yes (engine already built) | **yes** (`EngineEnv.snapshot()/restore()`) | fork `lev_comp` compiles the task's `.des`, fork `dlb` packs it with MiniHack's compiled `dungeon` into a patched `nhdat`, `RawEngine._build_dat_path` -> patched datadir. Needs auto-`--More--` and `^R` after restore | **partially**: Boxoban seed 7 and Corridor-R3: yes (all obs incl. tty). MazeWalk-9x9, CorridorBattle-Dark, Boxoban seed 11: game state (glyphs/chars/colors/blstats/message/inventory) identical, `tty_chars` differ. **Quest-Easy / Quest-Medium: engine aborts in `nle_fr_restore` -> `NetHackRL::load_mirror`** | 0.05-0.18 | 0.03-0.2 (incl. `^R`) | 0.03 ms (a step) | The fork *can* load MiniHack levels (first frame = the des map). Two engine issues: tty mirror not reliably resynced after restore; heap corruption on restore after >=39-41 post-snapshot steps on Quest levels (repro below). |
| TextWorld `the_cooking_game` (.z8) | yes | **yes**: `JerichoEnv` -> `jericho.FrotzEnv.get_state()/set_state()/copy()` | `get_state()` + deepcopy of every textworld wrapper's mutable attrs (Limit.nb_steps, StateTracking, Inform7Data, GameData, GameState) + `SyncBatchEnv.last` + gym wrapper fields, restored in place | **yes** (3 branches identical, no alias, and equal to the replay-based branch) | 26 | 45 | replay: 1.8 ms/step + 96 ms reset (132 ms for 20 steps) | 844 KB pickled; cost dominated by deepcopy of textworld's Python-side state tracker, not Jericho. |
| TextWorld `treasure_hunter`, `coin_collector` (.ulx) | yes | **no**: `GitGlulxEnv` (git-glulx-ml subprocess over a UNIX socket), `Environment.copy()` -> `NotImplementedError`, no state export; in-game `save` kills the interpreter, `undo` is one turn only | replay from reset (games are deterministic) | **yes**: 3/3 replays identical (20 + 50 commands) | n/a | replay: 64 ms (treasure_hunter, 20 steps), 28 ms (coin_collector) | 0.68 / 0.54 ms per step + 50 / 17 ms reset | Only 1 of the 3 BALROG TextWorld tasks has native checkpointing. |

## Details per environment

### Crafter

BALROG stack: `EnvWrapper -> GymV21CompatibilityV0 -> CrafterLanguageWrapper -> crafter.Env`.

1. `copy.deepcopy(env)` of the full stack (and `pickle` of `crafter.Env`) fail with
   `ValueError: <numpy.random._pcg64.PCG64 object at 0x...> is not a known BitGenerator module.`
   The culprit is not Crafter's RNG (`crafter.Env._world.random` is a
   `np.random.RandomState`, which copies fine) but `crafter.Env._np_random`, a
   gym-0.23 `RandomNumberGenerator` that `GymV21CompatibilityV0.reset(seed=)`
   installs via `gym.Env.seed()`. Crafter never reads it, so it is dropped from
   the snapshot (`crafter_ckpt._SKIP`).
2. With `_np_random` stripped, deepcopy/pickle/explicit copies all *restore*
   correctly, but branches still diverged on 3-4 of 5 seeds
   (`[C_explicit_world_player_rng b1-vs-b2] DIVERGENCE at step 28: key='semantic'`).
   Root cause is in Crafter: `Env._balance_object` builds
   `creatures = [obj for obj in objs if isinstance(obj, cls)]` from a `set`
   (`World._chunks` is `defaultdict(set)`) and picks `creatures[random.randint(...)]`
   for despawn; set order follows `id()`-hashes, so any restored copy orders
   creatures differently from the original and from other restores.
   `crafter_ckpt.apply_determinism_patch()` wraps `Env._balance_chunk` to sort
   `objs` by `(x, y, class)` first; it does not touch the RNG stream. With it,
   A/B/C are identical on 5/5 seeds and `no_alias` holds (canonical state digest).
3. Recommended: approach B (`snapshot_pickle`/`restore_pickle`), ~0.4 ms / 0.5 ms,
   120 KB, plus the patch; or C for a 36 KB blob. Wrapper-side fields that must
   be part of the snapshot: `CrafterLanguageWrapper.score_tracker`,
   `.achievements`, `EnvWrapper.failed_candidates`.

### MiniHack, path 2: balrog-nle replay

BALROG stack: `EnvWrapper -> GymV21CompatibilityV0 -> NLETimeLimit -> balrog NLELanguageWrapper -> AutoMore -> MiniHack`.
`nle._pynethack.Nethack` exposes only `close, done, get_seeds, how_done, in_normal_game, reset, set_buffers, set_initial_seeds, set_seeds, set_wizkit, step` — no state export.

Verified: same seeds + same actions => identical BALROG observations, 3/3 replays,
for Quest-Easy (all movement actions; the fixed sequence walks into lava at step
13 and every replay dies at the same step with the same observations), Quest-Easy
with lava-safe actions (30 prefix + 50 branch) and Boxoban-Medium (30 + 50).
Replay cost: 0.21-0.30 ms/step, reset 1.4 ms (Quest) / 5.4 ms (Boxoban, re-runs
`lev_comp`+`dlb` every reset).

Things a replay-based Go-Explore layer must do (all found the hard way):
- `balrog.environments.nle.base.NLELanguageWrapper.step` does
  `self.done = done if not self.done else self.done` and `reset()` never clears
  it: after one death every later step reports `done=True`
  (`[replay0 prefix] LENGTH MISMATCH 14 vs 2`). BALROG's evaluator never sees
  this because it builds a fresh env per episode. Clear it (`verify_minihack_replay.full_reset`).
- `env.reset(seed=s)` -> `NLE.seed(core=s, disp=None)` -> `disp = SystemRandom()`.
  Store `inner.get_seeds()` after the first reset and re-apply with
  `inner.seed(core, disp, reseed)` before each replay reset. (With only the core
  seed fixed, all our traces were still identical, but that is not guaranteed.)
- Seed python `random` and `np.random` before each reset like the evaluator does:
  Boxoban picks its level with `random.choice` (`minihack/envs/boxohack.py:82`).

BALROG MiniHack observation (for the later parity check): top-level keys
`['text', 'image', 'obs']`; `text = {'long_term_context', 'short_term_context'}`
(hybrid prompt: `message`, `language observation` (text_glyphs), `cursor`, and
the **map rendered from `tty_chars` rows 1..23**; short term = `inventory`);
`image = None` (non-VLM); `obs` = raw NLE dict with keys
`['blstats', 'glyphs', 'inv_letters', 'inv_strs', 'text_message', 'tty_chars', 'tty_colors', 'tty_cursor']`.
Available BALROG actions on Quest-Easy: the 8 compass moves, far-moves, up/down,
wait, more, apply, close, open, eat, force, kick, loot, pickup, pray, puton,
quaff, search, zap; on Boxoban: `north, east, south, west` only.

### MiniHack, path 1: NetHack-engine fork

Can the fork load MiniHack `.des` files? **Yes.** `nethack_core` itself has no
des support left (`env.py` refuses any `"MiniHack"` task: "MiniHack tasks
removed; use native tasks or load_level blobs"), but the fork's build tree ships
`util/lev_comp`, `util/dgn_comp`, `util/dlb`, and `RawEngine` has a `datadir`
setting for the read-only data tree. `verify_minihack_engine.py` does exactly
what `minihack/scripts/mh_patch_nhdat.sh` does: `lev_comp mylevel.des` with the
fork's compiler, `dlb cf nhdat` over the fork's unpacked `nhdat` + MiniHack's
compiled `dungeon` (`minihack/lib/dungeon`, first level "mylevel") + `mylevel.lev`,
and points `RawEngine._build_dat_path` at the result. `quest_easy.des` and
MiniHack's generated Boxoban/MazeWalk/Corridor des text all compile and the
first frame is the des map (recorded in the result JSONs, e.g. the Quest-Easy
lava column and Boxoban box room).

Native `EngineEnv.snapshot()/restore()` results (30 prefix + 3 x 50 branch, seed 7):

| task | branches identical (all obs) | game-state only (glyphs/chars/colors/blstats/message/inv) | snapshot | restore + `^R` | step |
|---|---|---|---|---|---|
| Boxoban-Medium seed 7 | yes | yes | 0.05 ms | 0.19 ms | 0.03 ms |
| Boxoban-Medium seed 7, no `^R` | **no** (`tty_chars` at step 0) | yes | 0.05 | 0.03 | 0.03 |
| Boxoban-Medium seed 11 | **no** (`tty_chars`) | yes | 0.05 | 0.22 | 0.03 |
| Corridor-R3 | yes | yes | 0.14 | 0.14 | 0.03 |
| MazeWalk-9x9 | **no** (`tty_chars`) | yes | 0.14 | 0.07 | 0.03 |
| CorridorBattle-Dark | **no** (`tty_chars`) | yes | 0.14 | 0.10 | 0.03 |
| Quest-Easy (seeds 7, 11, 13), Quest-Medium | **engine abort** | — | 0.05 | — | — |

Engine issues found (both reproducible with the script):

1. *Stale tty mirror after restore.* Game state is restored byte-for-byte but
   `tty_chars` can keep rows from the branch played before the restore, e.g.
   `row 13: '#.###.\`@##'` vs `'#.###\`@@##'` (Boxoban), or the pet `d`/`@`
   drawn on the previous branch's cells (MazeWalk). A `^R` (18) after
   `restore()` fixed Boxoban seed 7 completely but not MazeWalk / CorridorBattle /
   Boxoban seed 11. This matters for parity because BALROG's MiniHack prompt
   renders the map from `tty_chars`; an engine-based layer should build the map
   from `chars`/`glyphs` (identical after restore) or the mirror resync needs fixing.
2. *Heap corruption on restore (Quest levels).* Every restore made after
   >=39-41 post-snapshot steps aborts (38 steps: OK, 41: abort; the immediate
   restore and short branches are fine). Seeds 7/11/13, Quest-Easy and Quest-Medium:
   ```
   double free or corruption (out)            (or: corrupted double-linked list (not small) / SIGSEGV)
   === NLE SENTINEL: SIGABRT ===
   libnethack.so(_ZN10nethack_rl9NetHackRL11load_mirrorEPKv+0x1c4)
   libnethack.so(nle_fr_restore+0xc7)
   ```
   Full logs: `results/crashlogs/engine_MiniHack-Quest-*.log`. Repro:
   `PYTHONPATH=/root/NetHack-engine .venv/bin/python tools/balrog_ckpt/verify_minihack_engine.py --task MiniHack-Quest-Easy-v0 --prefix 0`.
   The crash is inside `winrl.cc:516 NetHackRL::load_mirror` (`src/win/rl/winrl.cc`),
   which does `inventory_.clear()/push_back` and `windows_[WIN_MESSAGE]->last_msg.assign()`;
   a `free()` of a bad pointer there means one of those libc-heap containers is
   being overwritten/aliased by the arena restore. Not diagnosed further —
   Corridor/Boxoban/MazeWalk never trigger it, Quest levels (start square holds
   a wand + horn, a kitten, monsters) always do. Removing the `OBJECT:`/`MONSTER:`
   lines from `quest_easy.des` did not change the immediate-restore result (OK),
   the delayed-restore case was not re-tested on the stripped level.
3. `--More--` swallows keystrokes (a prefix of 30 direction keys on Quest-Easy
   advanced 0 turns at 0.012 ms/step until an AutoMore loop was added); the
   engine equivalent of BALROG's `AutoMore` is needed.

Bottom line for path 1: feasible and 10x faster per step than balrog-nle, but
not usable for a BALROG-parity Go-Explore until the two engine bugs are fixed;
for Quest tasks it currently cannot be used at all.

### TextWorld

BALROG stack: `EnvWrapper -> GymV21CompatibilityV0 -> TextWorldWrapper -> textworld.gym TextworldGymEnv -> SyncBatchEnv -> Filter > Limit > GenericEnvironment > TWInform7 > StateTracking > Inform7Data > GameData > {GitGlulxEnv (.ulx) | JerichoEnv (.z8)}`.
Games come from `balrog-post-install` (`tw_games/`): treasure_hunter 25 x .ulx,
coin_collector 25 x .ulx, the_cooking_game 25 x .z8.

- `.ulx`: `GitGlulxEnv` runs `git-glulx-ml` as a subprocess and talks over a
  socket; `copy()` -> `NotImplementedError` at both the chain top and the base.
  In-game `save` prints `Enter saved game to store:` and the interpreter exits
  (`GameNotRunningError`); `undo` works but only one turn (`[Previous turn undone.]`).
  Replay determinism: 3/3 identical (20 + 50 commands), 0.5-0.7 ms/step,
  17-50 ms reset.
- `.z8`: `JerichoEnv.copy()` works; `_jericho` has `get_state/set_state/copy`
  (`undo` in-game: `Your interpreter does not provide "undo". Sorry!`). The
  BALROG chain above it has no `copy()` (`Filter.copy()` exists but `Limit`,
  `GenericEnvironment` do not), so `verify_textworld.snapshot_native` restores
  in place: `set_state()` + deepcopy of every wrapper's mutable `__dict__`
  entries (skipping `_game`, `_inform7`, `_jericho`, request infos) +
  `SyncBatchEnv.last` (a done-latch: forgetting it gave
  `[the_cooking_game native b1-vs-b2] LENGTH MISMATCH 50 vs 1`) +
  `TextworldGymEnv.last_commands/obs` + `TextWorldWrapper.progression` +
  `EnvWrapper.failed_candidates`. Verified identical 3 branches, no alias, and
  equal to the replay-based branch. 26 ms snapshot / 45 ms restore (844 KB).
- BALROG TextWorld observation keys: `['text', 'image']` (image None),
  info keys `['max_score', 'won', 'description', 'objective', 'score']`.

## Every failure hit (verbatim, short)

- `import nle` in a bare 3.12 venv: `ModuleNotFoundError: No module named 'pkg_resources'` -> `uv pip install "setuptools<70"`.
- Crafter `copy.deepcopy(env)` / `pickle.loads` of `crafter.Env`: `ValueError: <numpy.random._pcg64.PCG64 object at 0x7f0c4a9eba00> is not a known BitGenerator module.` (gym 0.23 `_np_random`; stripped).
- Crafter unpatched, e.g. seed 1: `[B_pickle_inner_inplace b2-vs-b3] DIVERGENCE at step 28` / `[C_explicit_world_player_rng b1-vs-b2] DIVERGENCE at step 28: key='semantic'` (unpatched pass rate over 5 seeds: A 2/5, B 1/5, C 1/5; patched 5/5).
- MiniHack replay, first version: `RuntimeError: Called step on finished NetHack` (script kept stepping after a lava death) and then `[replay0 prefix] LENGTH MISMATCH 14 vs 2` (BALROG `NLELanguageWrapper.done` latch survives `reset()`).
- TextWorld `.ulx`: `copy() FAIL: NotImplementedError`; `save` -> `Enter saved game to store:` then `textworld.core.GameNotRunningError: Game is not running at the moment.`
- TextWorld `.z8` native, first version: `[the_cooking_game native b1-vs-b2] LENGTH MISMATCH 50 vs 1` (missing `SyncBatchEnv.last`).
- Engine, Quest-Easy/Medium restore: `double free or corruption (out)`, `corrupted double-linked list (not small)`, `=== NLE SENTINEL: SIGSEGV ===` at `NetHackRL::load_mirror` <- `nle_fr_restore` (rc 134/139).
- Engine, no `^R` after restore: `[engine b1-vs-b2] DIVERGENCE at step 0: key='tty_chars'`; with `^R` still on MazeWalk-9x9, CorridorBattle-Dark, Boxoban seed 11 (`b2-vs-b3`).
- Engine, Quest-Easy before AutoMore: welcome `--More--` swallowed all 30 prefix keystrokes (frame unchanged, 0.012 ms/step) — the earlier "identical" result from that run was meaningless and was discarded.

## Unverified / caveats

- Only BALROG's default tasks were exercised (Crafter default; MiniHack Quest-Easy
  and Boxoban-Medium for the replay path, plus Corridor-R3 / MazeWalk-9x9 /
  CorridorBattle-Dark / Quest-Medium on the engine path; the three TextWorld tasks).
  Prefix 30 (20 for TextWorld), branch 50, 3 branches; Crafter 5 seeds, others 1 seed.
- Crafter determinism patch changes which creature gets despawned relative to
  stock Crafter for a given seed (stock Crafter is not reproducible across
  processes anyway); rewards/achievement logic are untouched.
- MiniHack replay: `disp`-seed independence of observations was observed, not proven.
- Engine path: parity of the fork's MiniHack episodes with balrog-nle (same seed
  => same level/monsters) was **not** checked; only self-consistency of snapshot/restore.
- TextWorld `.z8`: the 844 KB / 26-45 ms snapshot is unoptimised (deepcopy of
  the Inform7 state tracker); Jericho's own state is ~100x smaller.

## Reproduce

```bash
# one-shot (install + everything): ~10 min
SCRATCH=/tmp/claude-0/-root-NetHack-hub/5557e37f-5844-469d-83a8-5ac4ec77b191/scratchpad/balrog_ckpt \
  bash /root/NetHack-hub/tools/balrog_ckpt/run_all.sh
# or re-run individual checks in the existing venv
S=/tmp/claude-0/-root-NetHack-hub/5557e37f-5844-469d-83a8-5ac4ec77b191/scratchpad/balrog_ckpt
PY=$S/.venv/bin/python; T=/root/NetHack-hub/tools/balrog_ckpt
$PY $T/verify_crafter.py --seeds 1,2,3,4,5
$PY $T/verify_minihack_replay.py --task MiniHack-Quest-Easy-v0 --actions north,south,west,search
$PY $T/verify_minihack_replay.py --task MiniHack-Boxoban-Medium-v0
PYTHONPATH=/root/NetHack-engine $PY $T/verify_minihack_engine.py --task MiniHack-Boxoban-Medium-v0 --workdir $S/mh_engine_MiniHack-Boxoban-Medium-v0
PYTHONPATH=/root/NetHack-engine $PY $T/verify_minihack_engine.py --task MiniHack-Quest-Easy-v0 --prefix 0   # aborts in load_mirror
$PY $T/verify_textworld.py
```

Manual install steps (what `run_all.sh` does): `git clone --depth 1 https://github.com/balrog-ai/BALROG.git`;
`uv venv --python <python3.12> .venv`; `uv pip install --python .venv/bin/python -e ./BALROG`
(pulls `balrog-nle`, the balrog-ai `minihack`/`TextWorld` forks, `crafter`, `gym==0.23`,
`minigrid`, `baba`); `uv pip install "setuptools<70"`; `cd BALROG && balrog-post-install`
(Boxoban levels into `minihack/dat`, TextWorld games into `BALROG/tw_games`).
All installs succeeded on the first attempt.
