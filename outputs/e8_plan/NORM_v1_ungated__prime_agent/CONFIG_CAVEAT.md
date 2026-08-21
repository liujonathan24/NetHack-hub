# RETRACTED (2026-08-21, same day): the fog caveat was a false alarm

An earlier version of this file claimed the cell ran without full vision.
That conclusion came from grepping a config.toml that DOES NOT EXIST in this
dir (the launcher of that era wrote no config.toml; grep of a missing file
returns nothing). The authoritative resolved config in eval.log shows
`tune.reveal_map: "1.0"` AND `auto_dismiss: "false"` -- this cell ran with
FULL VISION, matching its E7 control. Verified twice: config in eval.log, and
the first request_map observation shows the entire level.

Remaining true caveat: this cell predates the harness-honesty pass (fd8aa13),
so it ran with the old misleading docs (empty schemas, stale SKILL.md, wiki,
duplicated prompt) -- exactly like the E7 control it was compared against.
Within-E8 comparisons are therefore internally consistent; only comparisons
against post-fd8aa13 (E10+) cells must account for the doc change.
