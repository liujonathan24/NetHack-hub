"""The one authoritative dungeon-char grid, reconciled across the engine's two
map planes.

WHY THIS EXISTS
---------------
NLE emits the same dungeon twice: `chars` (the glyph plane, a full snapshot
rebuilt every step) and `tty_chars` (the terminal, a PERSISTENT buffer that
`nle.c`'s TMT callback refreshes only for lines the terminal marked *dirty*).
The harness renders the `=== MAP ===` block and `=== VISIBLE FEATURES ===` from
`chars`; per-turn traces record `raw_grid` from `tty_chars`. They are supposed to
be two views of one dungeon, and a diff of the two is the cheapest possible
check that the MAP is a faithful render of the engine
(docs/HARNESS_DEFECTS.md 1.3: "the MAP is deliberately NOT rewritten — the grid
must stay a faithful render of the engine").

MEASURED DIVERGENCE (seed 0, `Val-hum-neu-fem`, `tune={"reveal_map": 1.0}`)
--------------------------------------------------------------------------
Turn 1 renders `+` at (16,5) and (11,10) in BOTH planes. From turn 2 onwards
`chars` (and `glyphs`, cmap 1/2 = S_vwall/S_hwall) says `|` / `-` while
`tty_chars` still says `+`, and the two doors vanish from VISIBLE FEATURES.
They are not decoration: driving `np_move_to` straight at (16,5) walks the hero
THROUGH it and out of the starting room. The agent was being shown a wall where
a real, walkable exit was — under an encoding whose entire purpose is "lights
on".

ROOT CAUSE (it is NOT the NetPlay tracker; the MAP never touches it)
-------------------------------------------------------------------
`reveal_map` is a render-time overlay in the engine's rl window port
(`third_party/NetHack/src/win/rl/winrl.cc :: NetHackRL::fill_obs`). Two things
in it interact:

  1. Navigation-isolation. For every cell it runs
     `cvt_sdoor_to_door(&levl[x][y])` on secret doors, so a fully-revealed level
     is genuinely connected instead of merely visible. That poke is PERMANENT —
     `levl[][]` keeps the real door forever after.
  2. The overlay only paints a cell when it is still unknown
     (`obs->glyphs[offset] == nul_glyph`) **or** when it was secret on THIS
     frame (`was_secret`).

On the frame that converts the secret door, `was_secret` is true, so the door is
painted into `chars`, `glyphs` and `tty_chars`. On every later frame the cell is
no longer secret and the hero's remembered glyph is a wall, so the guard skips
it: `chars` reverts to the wall face the hero remembers. `tty_chars` does not
revert, because the overlay wrote it directly into the exported buffer and NLE
only re-copies dirty terminal lines — so the exported tty keeps the last
correct value.

So the engine emitted the truth once and then stopped repeating it, and the two
planes disagree for exactly the class of cell where `tty_chars` is right.

WHAT THIS MODULE DOES
---------------------
Reconciles the two planes back into one grid, in the direction of the engine's
own `levl[][]`: a cell that `chars` calls solid but the engine's tty still draws
as a closed door is restored to `+`. Everything else is `chars` verbatim.

The rule is deliberately narrow, because tty_chars is the plane that CAN carry
menu prose (`ascii_map.py`: 723 of 6,802 committed MAP blocks, 10.6%, were
polluted by it — that defect is why the MAP moved off tty in the first place).
A cell is restored only when ALL of:

  * `chars` says a WALL FACE there (`|` or `-`) — never over floor, items,
    monsters, the hero, or even unseen rock, so a stale tty monster can never be
    resurrected onto the map and nothing can appear in territory the hero has
    not been shown. (Unseen rock needs no help: it is `nul_glyph`, the case the
    overlay's own guard already paints into `chars` every frame.)
  * the tty says `+`, the closed-door glyph the reveal overlay produces for a
    converted secret door.  A `+` is also a spellbook lying on the floor, but a
    floor tile is not a cell `chars` calls solid, so over a wall the reading is
    unambiguous;
  * the tty row carries no 4+ consecutive-letter run — the exact prose detector
    `tests/test_map_frame_bleed.py` uses. Menu/banner text is words; dungeon
    rows never are. A row with prose on it is not trusted at all.

This is not "rewriting the grid to match a tracker": every character here came
out of the engine, in the same frame, from the engine's own render of
`levl[][]`. No harness memory is involved and nothing is remembered across
turns — drop the tty plane and the result degrades to plain `chars`.
"""
from __future__ import annotations

import re

#: Prose detector, identical to `tests/test_map_frame_bleed.py::_LETTER_RUN`.
#: No legitimate dungeon row produces a 4+ consecutive-letter run: terrain is
#: punctuation and creatures are single letters separated by floor/wall glyphs.
_LETTER_RUN = re.compile(r"[A-Za-z]{4,}")

#: What `chars` may say for a cell we are willing to restore: a wall face, and
#: only a wall face. Anything else is real content and is left exactly as
#: emitted -- including unseen rock, which must never gain terrain from a plane
#: the hero's own vision did not put there.
_SOLID = frozenset("|-")

#: What the tty may say for us to believe it: the closed-door glyph.
_RESTORABLE = frozenset("+")

#: tty row 0 is NetHack's message line; tty row N+1 is map row N. Columns are
#: 1:1 (verified against a live frame: `chars[5]` and `tty_chars[6]` are the
#: same string). Do not reintroduce a column offset here.
_TTY_MAP_ROW_OFFSET = 1


def _rows(plane) -> list[str]:
    """A 2-D char plane as a list of full-width (un-stripped) strings."""
    return ["".join(chr(int(c)) for c in row) for row in plane]


def reconcile_rows(char_rows: list[str], tty_rows: list[str]) -> list[str]:
    """Pure form of the reconciliation, over already-decoded rows.

    `char_rows[y]` is map row y; `tty_rows[y + 1]` is the same row on the tty.
    Returns new rows; inputs are not mutated.
    """
    out: list[str] = []
    for y, row in enumerate(char_rows):
        t = y + _TTY_MAP_ROW_OFFSET
        if t >= len(tty_rows):
            out.append(row)
            continue
        tty_row = tty_rows[t]
        # A row carrying menu/banner prose is not a dungeon row; trust none of it.
        if _LETTER_RUN.search(tty_row):
            out.append(row)
            continue
        cells = None
        for x, ch in enumerate(row):
            if ch not in _SOLID or x >= len(tty_row):
                continue
            tc = tty_row[x]
            if tc not in _RESTORABLE:
                continue
            if cells is None:
                cells = list(row)
            cells[x] = tc
        out.append(row if cells is None else "".join(cells))
    return out


def engine_map_rows(raw_obs) -> list[str]:
    """The dungeon as rows of characters, reconciled across both engine planes.

    Falls back to `chars` alone (and then to `[]`) whenever the tty plane is
    missing, so every caller degrades exactly the way it did before this module
    existed.
    """
    chars = getattr(raw_obs, "chars", None) if raw_obs is not None else None
    if chars is None:
        return []
    char_rows = _rows(chars)
    tty = getattr(raw_obs, "tty_chars", None)
    if tty is None:
        return char_rows
    try:
        return reconcile_rows(char_rows, _rows(tty))
    except Exception:
        return char_rows


def engine_map_chars(raw_obs):
    """The reconciled grid as a numpy uint8 plane, drop-in for `raw_obs.chars`.

    Used by `prompt/features.py` so VISIBLE FEATURES and the MAP are read off
    ONE grid — the split between them is how the two restored doors could be on
    the tty, off the map, and out of the feature list all at once.
    """
    chars = getattr(raw_obs, "chars", None) if raw_obs is not None else None
    if chars is None:
        return None
    rows = engine_map_rows(raw_obs)
    if not rows:
        return chars
    try:
        import numpy as np

        out = np.array([[ord(c) for c in r] for r in rows], dtype=chars.dtype)
        if out.shape != chars.shape:
            return chars
        return out
    except Exception:
        return chars
