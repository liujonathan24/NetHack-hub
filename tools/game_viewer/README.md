# game_viewer — NetHack run → self-contained HTML widget

Turns an eval run directory into one HTML file that replays every game with
aligned model reasoning. No server, no network: open it locally or push it to a
blog branch and edit the prose around it.

```bash
# from the repo root, with the env importable (ENG + PYTHONPATH as in launch_cell.sh)
python -m tools.game_viewer.export outputs/e7_seed/NPCORE_v2__prime_agent \
    --label "outputs/e7_seed/NPCORE_v2__prime_agent=NPCORE v2" \
    -o viewer.html
```

- Pass several run dirs to put them all in one widget (game selector + stats table).
- `--label DIR=Name` sets the display name per dir.
- Edit `template.html` (the layout) — never the generated output; re-run to regenerate.

## What it reads
- `turns/*.ndjson` (game side): per-LM-turn final screen, keystrokes, status,
  tool call/result, rendered observation; and **per-move screens** under
  `step_frames` when the run set `record_step_frames=true`.
- `traces.jsonl` (LLM side): reasoning, aligned to game turns by ordered
  `nethack.*` call matching (best-effort until the call-id barrier lands here).

## Widget features
Wide 80-col engine screen · dual scrubbers (LLM turn / game turn — or game *move*
when step_frames exist) · per-turn keystrokes + step messages · collapsible full
reasoning · function-call log with per-call game-clock deltas · per-game
annotation box · sortable stats table.

## The blog page itself
- `build_viewers.py` — the standing set of standalone viewers (`blog/all_games.html`,
  `blog/e7_viewer.html`, …). `--data-root` points at a *backfilled* copy of the
  outputs (`python -m tools.trace_reasoning <run_dir>` first); `--with-e6` is the
  blog variant, and the default omits that cell to stay under the 16 MB artifact cap.
- `build_demo.py` — the small **in-article** embed. Writes a fragment to
  `blog/embeds/<name>.html` (scoped CSS/JS, no `<html>`), which `build_index.py`
  splices in wherever `blog/index.md` carries
  `<div class="game-embed" data-demo="<name>"></div>`. The fragment is committed,
  so `index.html` rebuilds from checked-in files alone.
- `build_index.py` — `blog/index.md` → `blog/index.html`. **Re-run after every
  Markdown edit** so the rendered page never goes stale.

## Drafting a blog around it
The exporter runs where the data is (remote); the output is a standalone file you
edit anywhere. Recommended flow: generate on the remote, commit to a `blog/*`
branch, `git pull` locally, and edit `template.html` + a prose wrapper, previewing
in a browser (the file needs no server). See the repo's blog branch.
