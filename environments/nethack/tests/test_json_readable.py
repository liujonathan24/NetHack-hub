"""The JSON encoding must be readable: per-tile x/y records with rendered
characters, plus whichever cell_schema attributes are enabled.

Before this, the JSON body carried `grid` — an RLE of raw glyph ids
(`"2359x79,2362,2361x7,..."`). Nothing in the prompt tells a model that 2362 is
a wall, so the structure was legible while the content was not.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nethack_harness.prompt.map_encoders import (  # noqa: E402
    build_cell_masks, build_cells, json_encode,
)


def _chars(rows):
    """Build a char grid from ASCII rows, space-padded to a rectangle."""
    w = max(len(r) for r in rows)
    return np.array(
        [[ord(c) for c in r.ljust(w)] for r in rows], dtype=np.uint8
    )


class _Model:
    """Minimal stand-in for the canonical MapModel."""

    def __init__(self):
        self.player = (2, 1)
        self.entities = []
        self.grid = "2359x5,2362"  # the opaque form, to prove it is dropped


def test_cells_carry_xy_and_the_rendered_character():
    chars = _chars(["  ", ".>"])
    cells = build_cells(chars)
    assert {"x": 0, "y": 1, "c": "."} in cells
    assert {"x": 1, "y": 1, "c": ">"} in cells


def test_unrevealed_tiles_are_omitted_not_blank():
    chars = _chars(["  ", ".."])
    cells = build_cells(chars)
    # Row 0 is all spaces: absent entirely, not emitted as {"c": " "}.
    assert all(c["y"] == 1 for c in cells)
    assert len(cells) == 2


def test_enabled_attributes_ride_on_each_cell():
    chars = _chars(["..", ".."])
    masks = build_cell_masks(
        chars, player=(0, 0), visited_xy={(1, 1)}, cell_schema={"seen", "visited"},
    )
    cells = build_cells(chars, masks)
    by_xy = {(c["x"], c["y"]): c for c in cells}
    assert by_xy[(1, 1)]["visited"] == 1
    assert by_xy[(0, 0)]["visited"] == 0
    assert by_xy[(0, 0)]["seen"] == 1
    # A disabled attribute is absent entirely, not emitted as 0.
    assert "reach" not in by_xy[(0, 0)]


def test_json_body_drops_the_glyph_id_grid_when_chars_are_given():
    body = json.loads(json_encode(_Model(), detail="full", chars=_chars([".>"])))
    assert "grid" not in body
    assert body["cells"] == [
        {"x": 0, "y": 0, "c": "."},
        {"x": 1, "y": 0, "c": ">"},
    ]


def test_legacy_grid_survives_when_chars_are_not_given():
    body = json.loads(json_encode(_Model(), detail="full"))
    assert body["grid"] == "2359x5,2362"
    assert "cells" not in body


@pytest.mark.parametrize("detail", ["minimal"])
def test_minimal_detail_emits_no_map_body(detail):
    body = json.loads(json_encode(_Model(), detail=detail, chars=_chars([".>"])))
    assert "cells" not in body and "grid" not in body
