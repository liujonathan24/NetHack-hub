---
name: blog-nethack-e7-handoff
description: NetHack E7 blog lives on blog/nethack-e7; index.md is source of truth and index.html must be regenerated with build_index.py on every md change
metadata:
  type: project
---

The NetHack E7 blog is on branch `blog/nethack-e7` in `github.com/liujonathan24/NetHack-hub`.
`blog/index.md` is the source of truth; `blog/index.html` is generated from it by
`python -m tools.game_viewer.build_index` and is committed alongside. The gameplay
viewers (`blog/*_viewer.html`, `blog/all_games.html`) come from
`tools/game_viewer/build_viewers.py`, which needs run data already backfilled by
`tools/trace_reasoning.py` — the backfilled copy is at `/root/nld/outputs-copy`
(raw `outputs/**/turns/` + `traces.jsonl` are gitignored/ephemeral). The claude.ai
16 MB artifact cap is handled by a real flag, not an ad-hoc snippet:
`build_viewers.py` omits the e6 cell by default (artifact) and `--with-e6` adds it
back for the blog. Interpreter `/root/NetHack-hub/.venv-cli-eval/bin/python` with
`PYTHONPATH` including `/root/NetHack-engine`; `markdown` + `pymdown-extensions`
must be installed into that venv.

**Why:** index.html is a build product — committing an md edit without re-running
build_index.py silently ships a stale rendered page, and origin moves under you
(rebase, then regenerate, then amend).

**How to apply:** any time `index.md` changes — yours or pulled from upstream —
re-run `build_index.py` before committing. Commit as Jonathan
([[commits-authored-as-jonathan]]). Work in your own worktree
([[work-in-own-worktree]]) since `/root/nld/blog-wt` is shared with other agents.
