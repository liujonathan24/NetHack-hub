# Human baselines: BALROG progression from human NetHack recordings

Decodes human play into the same BALROG progression curve our agents produce, so
agent and human trajectories can be compared on one axis. Findings and numbers
live in `docs/NLD_HUMAN_DATA.md`; endgame-detection validation in
`docs/NLD_ENDGAME_VALIDATION.md`; the rendered result in
`docs/artifacts/nethack_progression.html`.

## Heads-up before running any of this

These scripts were written against a working data directory and **carry
absolute paths** (`/root/nld/...` for the corpora, `/root/NetHack-engine` and
`/root/NetHack-hub/environments/nethack` for imports). They are committed as the
record of how the numbers were produced, not as a turn-key pipeline — expect to
fix the paths at the top of each file before re-running elsewhere. The decoded
outputs (`all_games.json` 12 MB, `endgame_states*.json` ~9 MB each,
`top10_curves.json` 6 MB) are deliberately **not** committed; they rebuild from
the raw corpora.

Requires `pyte` and `numpy` (no `nle` install needed — that is the point of the
standalone decoder). Scoring imports `nethack_harness.prompt.balrog`.

## Data

| corpus | what | where |
|---|---|---|
| NLD-NAO | 1.5M human games as ttyrec terminal recordings | `dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-{aa..bn}.zip` + `nld-nao_xlogfiles.zip` |
| nao_top10 | top 10 NAO players by Z-score, pre-decoded to `(T,24,80)` tensors | `storage.googleapis.com/dm_nethack/nao_top10/{user}.tar` |

Note `nao_top10`'s published `metadata.pkl` is served as 10,240 null bytes — it
is useless, and the session `stamp` it carries has to be replaced by chaining on
in-frame turn and experience-point continuity.

## The scripts

Decoding and reconstruction:

| file | role |
|---|---|
| `ttyrec_probe.py` | reference ttyrec decoder: frames → `pyte` → status line → curve |
| `fast_scan.py` | same, ~3x faster; reads only the status rows (whole-shard runs) |
| `build_games.py` | reconstructs whole games from session files via xlogfile time windows |
| `build_all.py` | whole-shard decode; **exclusive** file→game assignment (see below) |
| `top10_curves.py` | the `nao_top10` equivalent, chaining on turn + experience points |
| `endgame_parse.py` | detects the Planes and ascension; validated against the xlogfile |
| `endgame_audit.py` | prints every distinct matched string so flags can be checked |

Analysis and presentation:

| file | role |
|---|---|
| `agent_curves.py` | extracts the same curve from our own rollout traces |
| `prep_compare.py` | merges the three populations onto one turn grid for the artifact |
| `prep_plots.py` | per-game curves + percentile fans |
| `metric_test.py` | the max-vs-min evidence (linearity, AUC, one-axis-only runs) |
| `human_pace.py`, `sample_study.py`, `join_diag.py` | earlier sampled passes and join diagnostics |
| `ui_probe.py` | unrelated to the corpora: probes NetHack's blocking UI states against the live engine (see `docs/NETHACK_UI_SURFACE.md`) |

## The two mistakes this code exists to not repeat

**A recording is a login session, not a game.** One game spans as many sessions
as the player used. Scoring per file understates deep games badly (one game
reads maxlvl 29 per-file, 47 chained) — and it fails silently, because a
truncated game still looks like a complete short one.

**A session file belongs to exactly one game.** Assigning a file to every game
whose `starttime`..`endtime` window contains it is wrong: a saved character
resumed later has a window spanning everything played in between (longest
measured: 4.7 years), and 13.4% of files end up multiply claimed. Assigning each
file to the most recently started game still open at that moment moved
validation against the xlogfile from 95.8% to 98.4%.
