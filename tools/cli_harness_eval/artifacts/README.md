# E16 artifact refresh

Regenerates the two published E16 pages from the live run directories.

    bash refresh_e16.sh                 # writes the HTML next to this README
    E16_ART_DIR=/some/dir bash refresh_e16.sh

It rebuilds three datasets (`e16_data.json`, `traces_data.json`, `curves.json`)
and two pages, prints a status block, and **publishes nothing** — publishing
needs the Artifact tool, which only a Claude session can call.

## Recovering the published artifacts from a new session

The two pages are already published. Their URLs are bound to the file path they
were first published from, which was a **session-scoped scratchpad directory**
that no longer exists in a new session. To update them from anywhere, pass the
URL explicitly instead of relying on the path:

| page | URL |
|---|---|
| Run explorer | https://claude.ai/code/artifact/9941775c-36f3-4a33-990e-f5f8f4e65b34 |
| Reasoning traces | https://claude.ai/code/artifact/b3a8c036-7313-44f8-bc9e-9ae19be7e50e |

Publishing without the `url` creates a SEPARATE artifact rather than updating
these, which is the whole reason the URLs are written down here.

## Why this directory exists

These scripts originally lived in `/tmp/claude-0/-root/<session-id>/scratchpad`.
The experiments and their data are durable — the orchestrators are detached
processes and everything lands in `/root/nld/e16_runs/` — but the code that
turns that data into the pages was keyed to one session's temp path and would
have been lost with it.

## Files

- `refresh_e16.sh` — the entry point; regenerates everything and rebuilds both pages
- `gen_e16_data.py` — explorer dataset: one record per CLOSED attempt
- `gen_traces_data.py` — traces dataset: every orchestrator reply verbatim + each player's opening
- `build_explorer.py` / `build_traces.py` — the two page builders

`e16_progress_curve.py` and `e16_level_balance.py` live one directory up and are
imported by these.

## What the runs are

`RUNS` at the top of `refresh_e16.sh` names them. A run with no closed attempt
yet is skipped by the explorer generator but still plotted, because the curve
script reads the turn stream directly — that asymmetry is deliberate.
