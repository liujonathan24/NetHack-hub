"""Why did pathfinding fail? Turn "No valid path found" into a reason.

Exp3b's inefficiency audit found the bare failure string was the single
largest waste bucket in BOTH arms (~25% of all `move_to` calls): the agent is
told the path failed but not why, so it re-issues the identical coordinates a
few calls later. This module names the reason -- a closed door on the route, a
monster in the way, an unwalkable target, or no explored route at all -- plus
the most useful next action for each case.

Read-only over `obs.chars`; never raises (the caller treats any exception as
"no diagnosis").
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

#: Tiles the player can step onto freely.
_FREE = frozenset(ord(c) for c in ".>#<_{\\'@")
#: Floor items are walkable too.
_ITEMS = frozenset(ord(c) for c in "()[*$/=\"?!%")
#: A closed door: passable in principle, after opening/kicking.
_DOOR = ord("+")


def _walk(ch: int, doors_ok: bool) -> bool:
    return ch in _FREE or ch in _ITEMS or (doors_ok and ch == _DOOR)


def _bfs(chars: np.ndarray, sx: int, sy: int, doors_ok: bool):
    """`(dist, prev)` maps over the 8-connected walkable grid from (sx, sy)."""
    h, w = chars.shape
    dist = {(sx, sy): 0}
    prev: dict = {}
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in dist:
                    continue
                if not _walk(int(chars[ny, nx]), doors_ok):
                    continue
                dist[(nx, ny)] = dist[(x, y)] + 1
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return dist, prev


def explain_path_failure(raw_obs, skill_args: dict) -> Optional[str]:
    """One sentence of WHY the route to `skill_args`'s (x, y) failed, or None.

    The grid is the map-frame `chars` plane (tty row offset already absent).
    Coordinates arrive in map frame from the skill call itself.
    """
    if raw_obs is None or not isinstance(skill_args, dict):
        return None
    try:
        tx, ty = int(skill_args.get("x")), int(skill_args.get("y"))
    except (TypeError, ValueError):
        return None
    chars = getattr(raw_obs, "chars", None)
    bl = getattr(raw_obs, "blstats", None)
    if chars is None or bl is None:
        return None
    chars = np.asarray(chars)
    h, w = chars.shape
    if not (0 <= tx < w and 0 <= ty < h):
        return f"[why: ({tx},{ty}) is outside the map]"
    sx, sy = int(bl[0]), int(bl[1])

    tch = int(chars[ty, tx])
    tglyph = chr(tch) if tch else " "
    # A letter at the target is a monster occupying the tile.
    if chr(tch).isalpha() and tch != ord("@"):
        return (f"[why: a monster ('{tglyph}') is standing on ({tx},{ty}) -- "
                f"attack it or path next to it instead]")
    if not _walk(tch, doors_ok=True) and tch not in (0, ord(" ")):
        return (f"[why: ({tx},{ty}) is '{tglyph}' -- not a walkable tile; "
                f"pick an adjacent floor tile]")

    # Route WITH closed doors treated passable: if one exists, the blocker is
    # the first closed door along it.
    dist_doors, prev_doors = _bfs(chars, sx, sy, doors_ok=True)
    if (tx, ty) in dist_doors:
        node = (tx, ty)
        doors_on_route = []
        while node in prev_doors:
            x, y = node
            if int(chars[y, x]) == _DOOR:
                doors_on_route.append((x, y))
            node = prev_doors[node]
        if doors_on_route:
            dx, dy = doors_on_route[-1]  # nearest to the player
            return (f"[why: the route passes a CLOSED door at ({dx},{dy}) -- "
                    f"move adjacent and open or kick it first]")
        # Walkable-per-us but the skill still failed: usually a pet/peaceful
        # monster shuffling on the route, which is transient.
        return ("[why: a route exists on the map -- a monster or pet is "
                "probably blocking a corridor; try again or clear it]")

    # No route even through doors: the connection is unexplored or hidden.
    dist_free, _ = _bfs(chars, sx, sy, doors_ok=True)
    if dist_free:
        near = min(dist_free, key=lambda p: max(abs(p[0] - tx), abs(p[1] - ty)))
        gap = max(abs(near[0] - tx), abs(near[1] - ty))
        return (f"[why: NO explored route connects you to ({tx},{ty}). The "
                f"nearest reachable tile is ({near[0]},{near[1]}), {gap} away "
                f"-- the gap is unexplored or hidden; `reveal` the area "
                f"between, or `search` near dead ends]")
    return None
