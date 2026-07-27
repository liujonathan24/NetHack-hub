"""Regression test: the `=== MAP ===` block must never carry menu/tty prose.

Real repro (see `outputs/cli_harness_eval/*/*/turns/*.ndjson`; measured 723 of
6,802 committed MAP blocks = 10.6%): NetHack's pickup command (`,`) opens a
menu window that NetHack draws directly on top of the dungeon in `tty_chars`.
`nethack_core.observations.render_map_view` masks menus by inferring a single
left-hand column to truncate every row at — a heuristic that only holds when
the overlay is a clean right-hand column. It does nothing for an overlay that
starts mid-row or isn't a bottom-anchored `(end)` list (a boxed pop-up, the
startup banner, an `#attributes` screen), so prose like

    Ve------------0 Unix post-`+.....@.....             |............|
    S#.......<....###etails. |#|..........|#------  ----------....<..|

lands where dungeon glyphs belong. This test drives a REAL seeded engine and
opens the inventory menu (`i`) — same window system, same masking code path,
and readily reproduces the exact same class of bleed (confirmed manually: the
startup-banner lines and the menu's item lines both survive the column mask
and land inside the rendered MAP block).

Fix: `nethack_harness.prompt.ascii_map.render_map_from_chars` renders the MAP
block straight from `chars` — the glyph plane, which has no representation
for tty text at all, so it cannot carry menu/banner prose by construction —
instead of `structured.map_view` (tty-derived). Mirrors what
`nethack_harness/prompt/features.py` already does for VISIBLE FEATURES
(Task 17); `test_map_frame_preserved_hero_at_pos` below pins the same
coordinate-frame property `test_observation_frame.py::
test_rendered_map_row_index_equals_map_y` pins for that fix.
"""
from __future__ import annotations

import re

from nethack_core.env import NetHackCoreEnv
from nethack_core.observations import shape
from nethack_harness.prompt.rendering import format_observation_as_chat
from nethack_harness.tools.skills import bootstrap_character

CHARACTER = "Val-hum-neu-fem"

# No legitimate NetHack map glyph row produces a 4+ consecutive-letter run:
# terrain is punctuation and creatures are single letters separated by floor/
# wall/space glyphs. Any such run is prose that leaked in from an overlay.
_LETTER_RUN = re.compile(r"[A-Za-z]{4,}")


def _menu_open_frame(seed: int = 0):
    """A real seeded engine with the inventory menu open, mid-dungeon.

    Returns (raw_obs, structured) exactly like a live turn would build them.
    The inventory command (`i`) is deterministic (every character starts with
    inventory) and opens the same NetHack menu window the pickup command
    (`,`) does, so it exercises the identical extraction/masking code path.
    """
    env = NetHackCoreEnv(task_name="NetHackChallenge-v0")
    try:
        env.seed(core=seed, disp=seed)
        env.reset(character=CHARACTER)
        character = bootstrap_character(env)
        co, _reward, _done, _trunc, _info = env.step(ord("i"))
        return co, shape(co, character)
    finally:
        env.close()


def _map_block(rendered: str) -> str:
    """The MAP body only. Splits on the next `===` header, not on the first
    blank line — a freshly-started game has unexplored (blank) rows near the
    top of the map, so splitting on "\n\n" truncates the block prematurely."""
    body = rendered.split("=== MAP ===\n", 1)[1]
    return body.split("\n=== STATUS ===", 1)[0].rstrip("\n")


def test_map_block_has_no_menu_bleed_with_a_menu_open():
    """With a menu open, the MAP block must be pure dungeon glyphs."""
    raw, structured = _menu_open_frame()
    assert structured.menu is not None, "fixture did not actually open a menu"
    rendered = format_observation_as_chat(
        structured, None, state={"raw_obs": raw}, compact=False
    )
    block = _map_block(rendered)
    m = _LETTER_RUN.search(block)
    assert m is None, f"menu/tty prose leaked into MAP block: {m.group(0)!r}\n{block}"


def test_map_frame_preserved_hero_at_pos():
    """`map[y][x]` for `Pos:(x,y)` must be `@` — the coordinate frame Task 17
    pinned for VISIBLE FEATURES must survive this fix unchanged."""
    raw, structured = _menu_open_frame()
    rendered = format_observation_as_chat(
        structured, None, state={"raw_obs": raw}, compact=False
    )
    block = _map_block(rendered)
    rows = block.split("\n")
    px = int(structured.status["x"])
    py = int(structured.status["y"])
    assert len(rows) > py, "map block is shorter than the player's row"
    assert len(rows[py]) > px and rows[py][px] == "@", (
        f"map row {py} col {px} is {rows[py][px:px + 1]!r}, expected '@'\n"
        f"row={rows[py]!r}"
    )
