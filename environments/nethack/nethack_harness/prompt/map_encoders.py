# environments/nethack/nethack_harness/prompt/map_encoders.py
"""Serialize the canonical MapModel as JSON or TOON, at a selectable detail.

`full`  -> rich entity attributes + the RLE grid.
`minimal` -> entity kind/coord/description only; no grid, no rich attrs.
Both project the SAME model, so JSON and TOON cannot diverge.
"""
from __future__ import annotations

import json
from typing import Any

_RICH_FIELDS = ("species", "is_pet", "obj_class", "detail")


def _entity_dict(e: Any, detail: str) -> dict:
    d = {"kind": e.kind, "x": e.x, "y": e.y, "desc": e.description}
    if detail == "full":
        for f in _RICH_FIELDS:
            v = getattr(e, f, None)
            if v is not None:
                d[f] = v
    return d


def _model_dict(model: Any, detail: str) -> dict:
    d = {
        "player": list(model.player) if model.player else None,
        "entities": [_entity_dict(e, detail) for e in model.entities],
    }
    if detail == "full":
        d["grid"] = model.grid
    return d


def build_cells(chars, cell_masks: dict | None = None) -> list:
    """Per-tile records for every REVEALED tile: ``{x, y, c, <enabled attrs>}``.

    Readable by construction. The previous JSON body carried ``grid`` — an RLE
    of raw *glyph ids* (``"2359x79,2362,2361x7,..."``) — which is unreadable to a
    model: nothing in the prompt says 2362 is a wall and 2359 is unseen rock.
    Here ``c`` is the rendered character (``.`` floor, ``|``/``-`` wall, ``>``
    stairs down, ``#`` corridor), the same alphabet the ASCII encodings use and
    the one the system prompt's glyph key already explains.

    Unrevealed tiles (space) are omitted rather than emitted as blanks, so the
    payload stays proportional to what the hero has actually seen instead of
    always paying for 21x79.
    """
    import numpy as np

    chars = np.asarray(chars)
    h, w = chars.shape
    masks = cell_masks or {}
    out: list = []
    for y in range(h):
        row = chars[y]
        for x in range(w):
            ch = int(row[x])
            if ch == ord(" "):
                continue
            rec = {"x": x, "y": y, "c": chr(ch)}
            for attr in _CELL_ATTRS:
                m = masks.get(attr)
                if m is not None:
                    rec[attr] = int(m[y, x])
            out.append(rec)
    return out


# ---------- sub-experiment 1b: per-cell SPATIAL / EXPLORATION layers ----------
#
# The identification-rich JSON base (kind/coord/desc/species/door/stair + the
# RLE terrain grid) is enriched with optional, strictly-gated 0/1 mask layers:
#   seen_grid    -- 1 = tile revealed (non-space char), 0 = unseen
#   visited_grid -- 1 = hero has stood on this tile this level
#   reach_grid   -- 1 = walkable-reachable from the hero's tile right now
# Each mask reuses the terrain RLE format (`_rle_grid`) so a 21x79 layer stays
# bounded (a single run "0x79" per empty row). Only the enabled attributes are
# emitted; a disabled one is ABSENT from the JSON entirely.

_CELL_ATTRS = ("seen", "visited", "reach")


def build_cell_masks(chars, player, visited_xy, cell_schema) -> dict:
    """Return ``{attr: 0/1 ndarray}`` for each enabled cell_schema attribute.

    ``chars`` is the 21x79 char grid, ``player`` is (x, y), ``visited_xy`` is the
    set of (x, y) tiles the hero has stood on this level, ``cell_schema`` is a
    collection drawn from ``_CELL_ATTRS``. Consumed by :func:`build_cells`, which
    folds each enabled attribute into the per-tile record.
    """
    import numpy as np

    schema = set(cell_schema or ())
    chars = np.asarray(chars)
    h, w = chars.shape
    masks: dict = {}

    if "seen" in schema:
        # Revealed iff the rendered char is not the unseen/rock sentinel (space).
        masks["seen"] = (chars != ord(" ")).astype(int)

    if "visited" in schema:
        vis = np.zeros((h, w), dtype=int)
        for (x, y) in visited_xy or ():
            if 0 <= y < h and 0 <= x < w:
                vis[y, x] = 1
        masks["visited"] = vis

    if "reach" in schema:
        from nethack_harness.navigation.pathfinding import reachable_set
        reach = np.zeros((h, w), dtype=int)
        for (x, y) in reachable_set(chars, tuple(player)):
            if 0 <= y < h and 0 <= x < w:
                reach[y, x] = 1
        masks["reach"] = reach

    return masks


def build_cell_layers(chars, player, visited_xy, cell_schema) -> dict:
    """RLE 0/1 mask layers (``seen_grid`` / ``visited_grid`` / ``reach_grid``).

    The original sub-experiment-1b form, kept because `docs/experiments/
    exp1b-json-cellcontent-arms.md` documents these key names. The rendered JSON
    now uses :func:`build_cells` instead, which attaches the same attributes to
    per-tile ``{x, y, c}`` records rather than to a separate opaque mask.
    """
    from nethack_core.map_model import _rle_grid

    masks = build_cell_masks(chars, player, visited_xy, cell_schema)
    return {f"{attr}_grid": _rle_grid(m) for attr, m in masks.items()}


def json_encode(
    model: Any,
    *,
    detail: str = "full",
    chars=None,
    cell_masks: dict | None = None,
) -> str:
    """Serialize the map model as JSON.

    At ``detail="full"``, passing ``chars`` emits readable per-tile ``cells``
    (``{x, y, c, ...}``) in place of the raw-glyph-id ``grid`` RLE. Without
    ``chars`` the legacy ``grid`` form is kept, so callers that only have a
    model still work.
    """
    d = _model_dict(model, detail)
    if detail == "full" and chars is not None:
        d.pop("grid", None)
        d["cells"] = build_cells(chars, cell_masks)
    return json.dumps(d, separators=(",", ":"))


def toon_encode(model: Any, *, detail: str = "full") -> str:
    """Token-frugal line-oriented encoding of the same model.

    Format (deterministic):
        @ x,y
        <kind> x,y desc[ k=v ...]
        ...
        grid: <rle>            # full detail only
    """
    lines = []
    if model.player:
        lines.append(f"@ {model.player[0]},{model.player[1]}")
    for e in model.entities:
        parts = [e.kind, f"{e.x},{e.y}", e.description]
        if detail == "full":
            for f in _RICH_FIELDS:
                v = getattr(e, f, None)
                if v is not None:
                    parts.append(f"{f}={v}")
        lines.append(" ".join(str(p) for p in parts))
    if detail == "full":
        lines.append(f"grid: {model.grid}")
    return "\n".join(lines)
