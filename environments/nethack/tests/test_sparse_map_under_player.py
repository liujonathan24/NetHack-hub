"""SPARSE's "you are standing on the down-staircase" fallback must actually fire.

DEFECT (docs/HARNESS_DEFECTS.md 3.2). `rendering._remember_stairs_down` memoises
`(depth, x, y)` -- deliberately, since (x,y) means a different tile on every
floor and the memo is never cleared on descent. `sparse_map.sparse_entity_map`
tested `(x, y) in state["_seen_stairs_down"]`: a 2-tuple against a set that only
ever holds 3-tuples. It could not match on any input.

VERIFIED LIVE before the fix (seed 0, `tune={"reveal_map": 1.0}`): one real
render populated the memo as `{(1, 57, 13)}`, so the membership test the sparse
map performs for a hero standing on those stairs is `(57, 13) in {(1, 57, 13)}`
-- False, every time.

Why it matters: under SPARSE the entity list IS the map (terrain is dropped
entirely), and `@` covers its own tile, so this fallback is the ONLY channel
that can tell the agent it is standing on `>` -- the single tile the score
depends on.

The tests below wire the REAL producer to the REAL consumer, so a future change
to the key shape breaks here rather than silently disabling the fallback again.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nethack_core.observations import StructuredObservation
from nethack_harness.prompt.features import visible_features_from_chars
from nethack_harness.prompt.rendering import _remember_stairs_down
from nethack_harness.prompt.sparse_map import sparse_entity_map

STAIRS_X, STAIRS_Y = 5, 2


def _chars():
    """A small room with a `>` at (5,2)."""
    rows = [
        "-" * 12,
        "|" + "." * 10 + "|",
        "|" + "." * 4 + ">" + "." * 5 + "|",
        "|" + "." * 10 + "|",
        "-" * 12,
    ]
    return np.array([[ord(c) for c in r] for r in rows], dtype=np.int16)


def _structured(x, y, depth=1):
    return StructuredObservation(
        map_view="", messages=[], inventory=[],
        status={"x": x, "y": y, "hitpoints": 16, "max_hitpoints": 16,
                "armor_class": 6, "depth": depth, "time": 40,
                "experience_level": 1, "gold": 0, "hunger_state": 1},
        character={}, adjacent={},
    )


def _state_with_memo(depth=1):
    """A state whose stairs memo was filled by the production writer."""
    import types

    state = {"_seen_stairs_down": set(),
             "raw_obs": types.SimpleNamespace(chars=_chars())}
    feats = visible_features_from_chars(_chars())
    _remember_stairs_down(state, _structured(1, 1, depth=depth), feats)
    return state


def test_the_memo_really_is_depth_keyed():
    """Pins the shape the consumer has to speak: 3-tuples, never 2-tuples."""
    memo = _state_with_memo()["_seen_stairs_down"]
    assert memo == {(1, STAIRS_X, STAIRS_Y)}, memo
    assert (STAIRS_X, STAIRS_Y) not in memo, (
        "the 2-tuple lookup the sparse map used to do can never hit this set"
    )


def test_standing_on_remembered_stairs_is_announced():
    state = _state_with_memo()
    body = sparse_entity_map(_structured(STAIRS_X, STAIRS_Y), state)
    line = [ln for ln in body.splitlines() if ln.startswith("standing on:")]
    assert line, body
    assert "stairs down" in line[0], line[0]
    assert "DESCEND FROM HERE" in line[0], line[0]


def test_standing_elsewhere_still_admits_ignorance():
    state = _state_with_memo()
    body = sparse_entity_map(_structured(3, 3), state)
    line = [ln for ln in body.splitlines() if ln.startswith("standing on:")]
    assert "unknown" in line[0], line[0]


def test_the_memo_does_not_carry_across_floors():
    """The reason the memo is depth-keyed in the first place.

    Dlvl 1's staircase coordinate must not claim to be a staircase on Dlvl 2 --
    `descend` would then fail from a tile the observation swore was `>`.
    """
    state = _state_with_memo(depth=1)
    body = sparse_entity_map(_structured(STAIRS_X, STAIRS_Y, depth=2), state)
    line = [ln for ln in body.splitlines() if ln.startswith("standing on:")]
    assert "DESCEND FROM HERE" not in line[0], line[0]


def test_legacy_unkeyed_entries_still_work():
    """`_remembered_stairs_down` accepts pre-Task-17 2-tuples; so must this."""
    import types

    state = {"_seen_stairs_down": {(STAIRS_X, STAIRS_Y)},
             "raw_obs": types.SimpleNamespace(chars=_chars())}
    body = sparse_entity_map(_structured(STAIRS_X, STAIRS_Y), state)
    line = [ln for ln in body.splitlines() if ln.startswith("standing on:")]
    assert "DESCEND FROM HERE" in line[0], line[0]
