"""The `=== MAP ===` block, rendered from the glyph plane (`chars`) — not tty.

Mirrors what `prompt/features.py` already does for VISIBLE FEATURES.
`nethack_core.observations.render_map_view` builds the map block from
`tty_chars` and masks an open menu by truncating every row at an inferred
left column — a heuristic that only holds when the overlay is a clean
right-hand column. A `--More--` prompt, an `#attributes` screen, or (the
committed repro) the pickup menu NetHack's `,` command opens lands mid-row
and survives the mask:

    Ve------------0 Unix post-`+.....@.....             |............|
    S#.......<....###etails. |#|..........|#------  ----------....<..|

`Unix post-`, `etails.`, `Ve`, `S#` are menu prose sitting where dungeon
glyphs belong. Measured across every run committed under
`outputs/cli_harness_eval/*/*/turns/*.ndjson`: 723 of 6,802 MAP blocks
(10.6%) contained this class of bleed (detector: a 4+ consecutive-letter run
inside the MAP block — no legitimate NetHack map produces one).

`chars` is NetHack's actual dungeon state — the same array `a_star`,
`move_to`, `descend` and `prompt/features.py` already path over — and has no
representation for menu text at all, so reading it instead of `tty_chars`
deletes the defect class rather than papering over it, exactly as Task 17
did for feature coordinates (see `features.py`'s module docstring).

Frame: `chars` carries no tty message row (tty row 0 is the message line;
`chars` starts at map row 0), so this module needs no offset. Row N of the
string this renders is map row N — the same frame `Pos:(x,y)` and every
skill argument already use. Do not reintroduce a `-1`/`+1` here.
"""
from __future__ import annotations


def render_map_from_chars(chars) -> str:
    """Render the dungeon area as ASCII straight from the glyph-char plane.

    Each row is right-trimmed (matching `nethack_core.render_map_view`'s own
    `r.rstrip()`), so unseen trailing columns don't pad every row out to the
    full map width and inflate the token bill.

    Returns "" if `chars` is unavailable, mirroring `features.py`'s fallback
    so callers can degrade gracefully instead of raising.
    """
    if chars is None:
        return ""
    rows = []
    for row in chars:
        rows.append("".join(chr(int(c)) for c in row).rstrip())
    return "\n".join(rows)


def render_map(raw_obs) -> str:
    """The `=== MAP ===` body for a live observation.

    Identical to `render_map_from_chars(raw_obs.chars)` except that it reads the
    grid through `prompt/engine_grid.py`, which reconciles `chars` against the
    engine's own tty plane so terrain the reveal overlay emitted once and then
    stopped repeating (converted secret doors) is not silently dropped from the
    map. See that module for the measurement and the (narrow) rule.
    """
    from nethack_harness.prompt.engine_grid import engine_map_rows

    rows = engine_map_rows(raw_obs)
    if not rows:
        return ""
    return "\n".join(r.rstrip() for r in rows)
