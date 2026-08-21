# NLD-NAO human data: can we derive BALROG progression over time?

**Yes.** The human recordings carry the bottom status line in the clear, so
`Dlvl` and `Xp` — the only two inputs `balrog_progress` / `balrog_progress_min`
take — are recoverable at *every frame*, giving a full progression curve against
game turns, not just a final score. Verified end-to-end against ground truth.

Measured on `nld-nao-dir-aa` (1 of 41 shards: 126,199 ttyrec files, 3,586
players) plus the full `nld-nao_xlogfiles` release (3,585,691 game records), and
on the DeepMind `nao_top10` corpus (all 10 players, 16,482 sessions). **28,040
NAO games and 12,154 top-10 games decoded end to end**, August 2026.

Tooling written for this, copied into `tools/` (originals and the decoded data
live in `/root/nld/`; the scripts still `sys.path`-insert absolute local paths,
so they need a small path fix to run from a clean checkout):
`ttyrec_probe.py` (decode + parse + BALROG curve), `sample_study.py` (sampling +
xlogfile join), `human_pace.py` (game reconstruction + pace-to-milestone),
`join_diag.py` (join diagnostics), `ui_probe.py` (see `NETHACK_UI_SURFACE.md`).
Requires `pyte`; no `nle` install needed.

## What the data actually is

The dataset is NLD, from Hambro et al., *Dungeons and Data: A Large-Scale
NetHack Dataset* (NeurIPS 2022 D&B, [arXiv:2211.00539](https://arxiv.org/pdf/2211.00539)).
Two products, and the distinction matters:

| | NLD-NAO | NLD-AA |
|---|---|---|
| source | 1.5M human games, nethack.alt.org 2009–2020 | 100k AutoAscend (bot) games |
| actions | **no** | yes |
| content | ttyrec terminal recordings | NLE observations + actions |
| size | 41 × ~5.6 GB shards + 130 MB xlogfiles | 16 shards (+1.6 GB taster) |

So the human half is *terminal output only* — which is exactly why the status
line is the whole story: there is no state vector, only what the player's screen
showed.

Download (`DATASET.md` in facebookresearch/nle):
`https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-{aa..bn}.zip` and
`nld-nao_xlogfiles.zip`.

## Decoding

A NAO ttyrec is a plain frame stream — 12-byte little-endian header
(`sec`, `usec`, `len`) then `len` bytes of ANSI — bz2-compressed. NLE's own
`nle.dataset` loader needs the compiled `_pyconverter`, which this repo dropped
in the nle migration (`docs/CURRICULUM.md` notes this). It isn't needed: a
40-line reader plus `pyte` as the VT emulator reproduces the 24×80 grid, ~0.3 ms
per frame single-threaded.

The first frame carries a NAO header with `Player:` / `Game: NetHack 3.6.0` /
`Server:`, which is how you filter by game version without touching xlogfiles.

Rendered frame from a real game (`4hodmt`, 2009-12-27), status block intact:

```
4hodmt the Thaumaturge    St:13 Dx:18 Co:14 In:14 Wi:10 Ch:9  Chaotic S:4198
Dlvl:6  $:1212 HP:27(32) Pw:69(69) AC:-11 Xp:7/684 T:5373 Satiated
```

→ `{"depth": 6, "xp_level": 7, "turns": 5373, "hp": 27, "max_hp": 32}` →
`balrog_max 3.69%`, `balrog_min 3.54%`.

**88.7%** of frames across a 300-game random sample carry a parseable status
line (the rest are menus/overlays/redraws mid-write, where the status block is
covered or not yet drawn). Since we only need the *distinct-state* curve, that
is far more sampling than the curve has resolution for.

## Validation against ground truth

The xlogfile is the join target, but note what it does **not** have: NAO's
xlogfile has no `xplevel` field (`version points deathdnum deathlev maxlvl hp
maxhp deaths dates uid role race gender align name death conduct turns achieve
realtime starttime endtime flags`). So **the XP axis of BALROG exists only in
the ttyrecs** — the xlogfile alone can score the Dlvl axis and nothing else.

First-pass agreement between ttyrec-derived `max_dlvl` and xlog `maxlvl` was
only 79%, and chasing that produced the single most important structural fact:

> **A ttyrec is a login session, not a game.** One game spans however many
> sessions the player used (save/resume, disconnects); one player directory
> holds many games.

Chaining a player's session files in time order fixes it exactly:

| player | sessions | derived | xlog truth |
|---|---|---|---|
| `4hodmt` | 3 (T:1→2069, 2069→4894, 4894→7335) | maxlvl 10, T 7335 | maxlvl 10, turns 7335 |
| `Abeixa` | 2 | maxlvl 9, T 2659 | maxlvl 9, turns 2659 |
| `36ji` | 2 | maxlvl 47, T 27156 | maxlvl 47, turns 27156 |

Exact on both axes, 3/3. Per-session curves for `4hodmt` compose into one
monotone game curve ending at `balrog_max 12.56% / balrog_min 7.45%`.

Two further join notes: ttyrec filenames are UTC (78 of 80 nearest-record deltas
were 0 h — no server-local offset to correct), and only ~85% of xlog records
carry `starttime` at all (older records use the `:`-separated pre-2008 format),
which caps the join rate, not the decode rate.

## Caveats that bite

1. **Session chaining is mandatory** (above). Scoring per file understates deep
   games badly (`36ji`: 29 vs 47).
2. **A session file belongs to exactly ONE game.** The obvious rule -- give a
   file to every game whose `starttime`..`endtime` window contains it -- is
   wrong, because a saved character resumed later has a window spanning
   everything played in between (the longest measured spans **4.7 years**).
   13.4% of files were multiply claimed that way. Assigning each file to the
   most recently started game still open at that moment moved validation from
   95.8% to 98.4% on a pilot slice, and 96.5% across all 28,040 games.
3. **Shards are partitioned by player, not by file** — verified: 3,586 players
   in shard `aa`, 2,577 in `ab`, **0 overlap**. So one shard gives complete
   session chains for the players it contains, and a 1/41 download is a valid
   ~2.4% player sample rather than a corpus of fragments. (Good news; I assumed
   the opposite before checking.)
4. **The turn counter is optional and lies mid-redraw.** `T:` only appears if
   the player has `time` on (275 of 277 sampled games had it), and NetHack
   repaints the status field-by-field, so a frame caught mid-write can read
   `T:41` while `T:41148` is being drawn — and can *hold* that truncated value
   for several consecutive frames. Taking it at face value as a "new game" split
   single games into dozens and produced nonsense pace numbers (BALROG 50%
   "reached" at turn 44). The rule that works: a game boundary requires the
   depth axis too — `Dlvl == 1 and T <= 3`. A turn-only rule, even with 5
   confirming samples, still admitted mis-cuts.
5. **The level field is not always `Dlvl:N`** (`botl.c describe_level`): the
   Quest shows `Home N`, Ft. Ludios its dungeon name, the endgame the plane
   name. 1,608 of 162,819 sampled state transitions were `Home N`. A bare
   `Dlvl:` regex silently drops the deepest, most interesting frames.
6. **Polymorph replaces `Xp:N/n` with `HD:n`** — the XP axis is unreadable while
   polymorphed (rare: 2 of 277 games).
7. **Terminal size varies.** Scan all rows for the status block rather than
   assuming rows 22–23; NAO players used taller terminals.
8. **Every game in this shard is NetHack 3.4.3** (49,234/49,234, including all
   438 ascensions), which prints `End Game` for the Elemental Planes -- the
   per-plane names only arrived in 3.6.2. A plane matcher written against
   modern NetHack silently misses every Elemental-Plane frame here.
9. **Actions are absent but not entirely.** The recording is the server→client
   stream, and NetHack echoes yn answers, getlin text and menu letters — so
   *prompt-answering* keystrokes are partially recoverable. Movement is not.

## Human reference numbers (new, usable as a baseline)

**Whole-corpus results** (games of >= 100 turns; Dlvl capped at 50, so 80.68%
is the ceiling):

| population | games | median max | mean max | p90 | median min | median turns | ascended |
|---|---|---|---|---|---|---|---|
| NAO population | 23,860 | 3.54% | 8.16% | 17.91% | 2.12% | 1,675 | 433 (1.8%) |
| NAO top 10 | 4,480 | 12.56% | 25.34% | 68.10% | 6.96% | 5,865 | 298 (6.7%) |
| our GLM-5.2, vision | 56 | 2.38% | 5.05% | 12.56% | 0.00% | 333 | 0 |

The sampled pilot (96 games) got the median about right (3.69 vs 3.54) but
overstated p90 badly (25.48 vs 17.91) -- it happened to catch too many deep
games. Sample with care at this scale, or not at all.

**Final-score distribution from the xlogfile alone, all 3.59M games** (Dlvl axis
only -- NAO's xlogfile has no `xplevel`, so these are lower bounds):

| cohort | n | median maxlvl | p75 | p90 | p99 |
|---|---|---|---|---|---|
| all games | 3,585,691 | 1 (0.00%) | 3 (1.75%) | 7 (4.85%) | 27 (55.4%) |
| turns >= 100 | 906,172 | 5 (**2.65%**) | 7 (4.85%) | 10 (12.6%) | 49 (80.2%) |
| turns >= 1000 | 616,320 | 6 (**3.54%**) | 8 (6.96%) | 12 (20.6%) | 50 (80.7%) |

67% of all NAO games end on Dlvl 1 and the median game lasts 1 turn -- the
corpus is mostly abandoned starts, so any human baseline must filter on turns.
Overall ascension rate 0.62%; Valkyrie games with >= 100 turns ascend at 1.6%.

**Pace to milestone**, measured over every decoded game rather than a sample.
Turn at which each BALROG-max level is first reached:

| BALROG % | population: reached | median turn | top 10: reached | median turn |
|---|---|---|---|---|
| 1% | 93.5% | 469 | 98.3% | 451 |
| 2% | 77.8% | 924 | 91.8% | 1,078 |
| 5% | 36.0% | 2,693 | 67.9% | 3,617 |
| 10% | 16.2% | 5,491 | 54.0% | 6,617 |
| 15% | 11.6% | 7,438 | 48.6% | 9,146 |
| 20% | 8.7% | 9,841 | 43.2% | 11,892 |
| 30% | 6.2% | 14,020 | 35.5% | 15,892 |
| 50% | 3.7% | 22,752 | 20.3% | 19,860 |
| 80% | 2.0% | 41,818 | 6.2% | 33,656 |

Note what separates the experts: not speed but **completion rate**. The top 10
reach each milestone at roughly the same turn as anyone else -- they are
slightly *slower* to 5% and 10% -- but 54% of their games reach 10% against
16% of the population's, and 6.2% reach 80% against 2.0%. Expertise here buys
survival, not pace.

This is the extrapolated-human-speed curve the progress-rate work wanted, now
measured rather than extrapolated: **~900 game turns to 2%, ~2,500 to 5%,
~3,700 to 10%, ~23,000 to 50%.** Against a 200-call budget that is the whole
problem statement in one line — matching median human pace to 10% means ~3,700
game turns, i.e. ~18 turns of real progress per LLM call, sustained.

Milestone counts fall off fast (only 13 of 173 games ever reach 10%), so the
deep rows are directional; widening the player sample is cheap if we want tighter
bounds there.

## If we want this at corpus scale

Full NAO is ~225 GB zipped. Decoding all 1.5M games with pyte is ~10⁹ frames —
feasible but a day-scale job; the practical route is to keep the xlogfile for
final-score statistics (free, already downloaded) and decode ttyrecs only for
the ~1% of games that matter (deep runs, Valkyrie runs, ascensions), selected
*from* the xlogfile first. Note that selection requires the session-chaining fix
above, since the xlogfile identifies games, not files.


## Beating the game

Ascension is the win condition and is rare enough that it has to be detected
rather than assumed. `tools/human_baselines/endgame_parse.py` finds it from the
status line (the level field reads `Astral Plane` / `End Game` / `Plane of X`)
plus the message text from `pray.c`. Validation against the xlogfile's
`death=ascended`: **precision 1.000, recall 0.998** (437 of 438), zero false
positives, zero containment violations. Full detail in
`NLD_ENDGAME_VALIDATION.md`.

| corpus | games scanned | reached the Planes | ascended | median ascension turn |
|---|---|---|---|---|
| NLD-NAO | 638 (all 438 xlog wins + 200 deep controls) | 466 | 437 | 43,040 |
| nao_top10 | 11,590 | 1,780 | 1,703 | 41,671 |

Two detection traps worth knowing:

- `You offer the Amulet of Yendor to` is **not** proof of a win. Offering to the
  wrong deity prints the same line and ends the game as *escaped in celestial
  disgrace*; there are exactly two such games in the scanned set and both are
  correctly excluded.
- The farewell line is role-dependent (`role.c`): *Farvel* for a Valkyrie,
  *Sayonara* for a Samurai, *Aloha* for a Tourist. Matching only "Goodbye" costs
  130 detections.

**A win takes ~42,000 game turns.** Our agents reach a median of 333. That gap
is roughly two orders of magnitude, and it is a turns-per-call problem before it
is a reasoning problem.

## Which BALROG variant models "how close am I to finishing"?

Use **min**. Measured on the 2,570 complete top-10 games available at the time
(131 of which reached Dlvl 50): at turn 2,000, **758 of the 1,677 games still in
play score above zero on max but exactly zero on min** -- all progress on one
axis. Only **3.3%** of those ever reach the end, against **11.5%** of the rest.

Two honest qualifications:

- Max is marginally the more *linear* in elapsed progress (pooled R^2 0.41 vs
  0.38; median per-game 0.74 vs 0.71). Min is the truer one; max is the smoother.
- **Neither predicts finishing mid-game.** AUC for separating eventual finishers
  sits near 0.5 at every checkpoint, and at turns 5k-10k the median *non*-finisher
  is the deeper one (16.13% vs 12.56% at turn 10,000). Among experts, being
  further along at a given turn does not make you more likely to finish -- the
  fast divers are the ones who die. Treat a fitted line as a description of pace,
  never as a completion forecast.

Reproduce with `tools/human_baselines/metric_test.py`.
