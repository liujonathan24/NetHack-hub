"""The `=== MAP ===` block must be byte-faithful to the engine's own grid.

docs/HARNESS_DEFECTS.md 1.3: "the MAP is deliberately NOT rewritten -- the grid
must stay a faithful render of the engine. Derived knowledge belongs in the
derived sections." Nothing checked it, and it was not true.

MEASURED (seed 0, Val-hum-neu-fem, `tune={"reveal_map": 1.0}`). Diffing the
per-turn trace's `raw_grid` (built from `tty_chars`) against the rendered MAP
(built from `chars`) showed two cells differing from turn 2 on: (16,5) and
(11,10), both `+` on the tty and `|` / `-` on the map, and both missing from
VISIBLE FEATURES. They are real doors -- `np_move_to(16, 5)` walks the hero
through one and out of the starting room -- so the agent was shown a wall across
its only early exit. Reproduced in the committed trace
`outputs/trace_probe/B0_reveal/turns/0_28800_1785554140.ndjson` (turn 1:
`[(16,5,'+','|'), (11,10,'+','-')]`).

CAUSE, and why it is not the tracker: see `prompt/engine_grid.py`. Short version
-- `reveal_map`'s overlay permanently converts secret doors to real doors in
`levl[][]` but only paints a cell into the emitted obs while it is still
*unknown or secret*, so `chars` reverts to the wall the hero remembers, while
the exported `tty_chars` (a persistent buffer refreshed per dirty line) keeps
the door.

WHAT THIS FILE PINS
-------------------
1. The MAP equals the engine's tty map rows, byte for byte, across the first N
   turns, under both fog and `reveal_map` -- the diff the defect report asked
   for, as a standing invariant.
2. The reconciliation rule itself, in isolation, including that it cannot
   reopen the menu-bleed defect `test_map_frame_bleed.py` pins.

NOTE ON MONSTERS. The tty plane is refreshed per *dirty line*, so a monster that
moved can linger on it for a frame or two while `chars` (a full snapshot) is
already correct -- the opposite direction of staleness, engine-side, and NOT
something the MAP should copy. The turn loop below therefore drives `np_look`,
which does not move the party, and asserts on every cell; `_terrain_diff` exists
so a future motion-driven case can assert on terrain only.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

#: How many consecutive turns to compare. Long enough to cover the whole
#: window the defect was measured in (it healed by turn 5 as the hero gained
#: line of sight), short enough to stay a unit test.
TURNS = 6

#: Glyphs that are a creature rather than terrain -- see the module note.
_MONSTERISH = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@'&;:")


def _tty_map_rows(raw_obs) -> list[str]:
    """The engine's own map rows, exactly as the trace's `raw_grid` records
    them: tty row 0 is the message line, so map row N is tty row N+1."""
    rows = ["".join(chr(int(c)) for c in row).rstrip() for row in raw_obs.tty_chars]
    return rows[1:22]


def _map_block(rendered: str) -> list[str]:
    body = rendered.split("=== MAP ===\n", 1)[1]
    body = body.split("\n=== ", 1)[0]
    rows = [r.rstrip() for r in body.split("\n")]
    while rows and not rows[-1]:
        rows.pop()
    return rows


def _diff(map_rows, tty_rows, terrain_only=False):
    out = []
    for y in range(21):
        a = map_rows[y] if y < len(map_rows) else ""
        b = tty_rows[y] if y < len(tty_rows) else ""
        for x in range(max(len(a), len(b))):
            ca = a[x] if x < len(a) else " "
            cb = b[x] if x < len(b) else " "
            if ca == cb:
                continue
            if terrain_only and (ca in _MONSTERISH or cb in _MONSTERISH):
                continue
            out.append((x, y, cb, ca))   # (x, y, engine, rendered)
    return out


async def _turn_diffs(tune):
    import nethack

    env = nethack.load_environment(
        variant="B0", skill_set="netplay_true,reveal,rollback",
        max_turns=TURNS + 5, explicit_seeds=[0], n_examples=1,
        character="Val-hum-neu-fem", compact_obs=False, tune=tune,
    )
    ex = env.dataset[0]
    state = {
        "task": {"seed": 0}, "info": ex["info"], "prompt": ex["prompt"],
        "responses": [], "turn": 0, "id": "map_fidelity", "model": "test",
    }
    state = await env.setup_state(state)

    per_turn = []
    for _ in range(TURNS):
        rendered = env._render_obs_text(state)
        per_turn.append((
            _diff(_map_block(rendered), _tty_map_rows(state["raw_obs"])),
            rendered,
        ))
        # `np_look` is a no-op observation skill: it keeps the engine stepping
        # (so the reveal overlay re-runs, which is what exposed the defect)
        # without moving the hero or the pet.
        await env._apply_tool_call(state, "np_look", {})
    return per_turn


@pytest.mark.parametrize("label,tune", [
    ("fog", None),
    ("reveal", {"reveal_map": 1.0}),
])
def test_rendered_map_matches_the_engine_grid(label, tune):
    per_turn = asyncio.run(_turn_diffs(tune))
    for turn, (diffs, rendered) in enumerate(per_turn, start=1):
        assert not diffs, (
            f"[{label}] turn {turn}: the rendered MAP disagrees with the "
            f"engine's own grid at {len(diffs)} cell(s): "
            f"{diffs[:8]} (x, y, engine, rendered).\n"
            f"The MAP must be a faithful render of the engine "
            f"(docs/HARNESS_DEFECTS.md 1.3); if the engine and the map really "
            f"do disagree, fix it toward the engine -- never by rewriting the "
            f"grid.\n{rendered[:2000]}"
        )


def test_the_two_seed0_doors_stay_on_the_map_and_in_the_features():
    """The specific regression, named.

    Under `reveal_map` the engine converts the starting room's two secret doors
    into real ones on the first frame. They must not disappear again.
    """
    per_turn = asyncio.run(_turn_diffs({"reveal_map": 1.0}))
    for turn, (_diffs, rendered) in enumerate(per_turn, start=1):
        rows = _map_block(rendered)
        assert rows[5][16] == "+", (
            f"turn {turn}: (16,5) rendered {rows[5][16]!r}, expected the "
            f"converted secret door '+' -- np_move_to(16,5) walks through it")
        assert rows[10][11] == "+", (
            f"turn {turn}: (11,10) rendered {rows[10][11]!r}, expected '+'")
        feats = [ln for ln in rendered.splitlines()
                 if ln.startswith("=== VISIBLE FEATURES ===")]
        assert feats, f"turn {turn}: no VISIBLE FEATURES block"
        assert "(16,5)" in feats[0] and "(11,10)" in feats[0], (
            f"turn {turn}: doors missing from VISIBLE FEATURES: {feats[0]}")


def test_fog_does_not_invent_the_secret_doors():
    """Without `reveal_map` those doors are still SECRET and must stay hidden.

    The reconciliation restores what the engine emitted; it must never
    manufacture terrain the hero has not been given.
    """
    per_turn = asyncio.run(_turn_diffs(None))
    for turn, (_diffs, rendered) in enumerate(per_turn, start=1):
        rows = _map_block(rendered)
        assert rows[5][16] in "|-", f"turn {turn}: (16,5) leaked as {rows[5][16]!r}"
        for ln in rendered.splitlines():
            if ln.startswith("=== VISIBLE FEATURES ==="):
                assert "(16,5)" not in ln, f"turn {turn}: {ln}"


# --------------------------------------------------------------------------- #
# The reconciliation rule, in isolation                                        #
# --------------------------------------------------------------------------- #

def test_rule_restores_a_door_the_char_plane_dropped():
    from nethack_harness.prompt.engine_grid import reconcile_rows

    chars = ["  |$....|", "  |@....|"]
    tty = ["a message line", "  |$....+", "  |@....|"]
    assert reconcile_rows(chars, tty) == ["  |$....+", "  |@....|"]


def test_rule_only_ever_touches_a_wall_face():
    """Floor, items, monsters, the hero AND unseen rock are all off limits.

    Every tty cell here says `+`; only the two wall faces may change. In
    particular the two leading blanks (unseen rock) must stay blank -- restoring
    there would show the agent terrain its own vision never revealed.
    """
    from nethack_harness.prompt.engine_grid import reconcile_rows

    chars = ["  |.d..>|"]
    tty = ["msg", "+++++++++"]
    assert reconcile_rows(chars, tty) == ["  +.d..>+"]


def test_rule_ignores_rows_carrying_menu_prose():
    """The menu-bleed defect must not come back through this door.

    `ascii_map.py`: 723 of 6,802 committed MAP blocks (10.6%) carried menu text
    because the map was read off the tty. A tty row with prose on it is not a
    dungeon row and none of it is trusted.
    """
    from nethack_harness.prompt.engine_grid import reconcile_rows

    chars = ["  |$....|"]
    tty = ["msg", "  |$..+ Unix post-details."]
    assert reconcile_rows(chars, tty) == ["  |$....|"]


def test_rule_degrades_to_chars_without_a_tty_plane():
    import types

    import numpy as np

    from nethack_harness.prompt.engine_grid import engine_map_rows

    rows = ["  |$....|", "  |@....|"]
    raw = types.SimpleNamespace(
        chars=np.array([[ord(c) for c in r] for r in rows], dtype=np.uint8))
    assert engine_map_rows(raw) == rows
