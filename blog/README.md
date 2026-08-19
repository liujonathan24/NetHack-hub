# blog/nethack-e7 — drafting workspace

The self-contained gameplay archive plus prose. Nothing here needs a server.

## Files
- `e7_viewer.html` — generated widget (all games + reasoning). **Do not hand-edit**; regenerate.
- `template.html`  — the widget layout. Edit this to change how games render, then regenerate.
- `index.md`       — the blog prose. Edit freely.

## Local workflow
```bash
git clone -b blog/nethack-e7 <repo-url> nethack-blog && cd nethack-blog
open blog/e7_viewer.html          # or drag into a browser — renders offline
# edit blog/index.md and blog/template.html locally, commit, push
```

## Regenerating the viewer (needs the run data — do this on the machine with outputs/)
```bash
python -m tools.game_viewer.export \
  outputs/e7_seed/NPCORE_v2__prime_agent \
  --label "outputs/e7_seed/NPCORE_v2__prime_agent=NPCORE · 8 tools" \
  -o blog/e7_viewer.html
git add blog/e7_viewer.html && git commit -m "regen viewer" && git push
```
Then `git pull` locally to get the refreshed HTML.
