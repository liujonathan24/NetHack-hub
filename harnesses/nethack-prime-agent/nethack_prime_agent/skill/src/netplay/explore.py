"""explore: walk to the nearest unexplored edge.

The code twin of the retired `np_explore_level` tool. Finds frontier tiles
(known-walkable tiles touching unrendered ones), routes to the nearest, and
repeats up to `max_moves` legs. It only explores; anything else is a
separate decision that stays yours.

This file is YOURS TO EDIT: improve it, and what you change persists.
"""
from __future__ import annotations

from netplay import _base
from netplay.move import (_DIRS, WALKABLE, _extra_walkable, _game_grid,
                          _interrupted, move_to, route)

__all__ = ["explore", "frontiers"]


def frontiers(obs: str) -> list[tuple[int, int]]:
    """Known-walkable tiles that touch an unrendered one -- the explore edge."""
    rows = _game_grid(obs)
    extra = _extra_walkable(obs)
    out = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in WALKABLE and (x, y) not in extra:
                continue
            if ch == "@":
                continue
            for (dx, dy) in _DIRS:
                nx, ny = x + dx, y + dy
                if not (0 <= ny < len(rows)):
                    continue
                nrow = rows[ny]
                if not (0 <= nx < len(nrow)) or nrow[nx] == " ":
                    out.append((x, y))
                    break
    return out


async def explore(max_moves: int = 3) -> str:
    """Walk to the nearest unexplored edge, up to `max_moves` times.

    Stops early when there is no reachable frontier left -- which usually
    means the level's remaining exits are behind a closed door or a hidden
    passage. What to do about that (search, kick, leave) is your decision.
    """
    obs = await _base.screen()
    for _ in range(max_moves):
        here = _base.position(obs)
        if here is None:
            return obs
        targets = [f for f in frontiers(obs) if f != here]
        if not targets:
            return (f"netplay.explore: no reachable unexplored edge left. "
                    f"Remaining exits may be behind closed doors or hidden "
                    f"passages.\n{obs}")
        best = None
        for t in sorted(targets,
                        key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1])):
            if route(obs, t) is not None:
                best = t
                break
        if best is None:
            return obs
        obs = await move_to(*best)
        if _interrupted(obs):
            return obs
        obs = await _base.screen()
    return obs
