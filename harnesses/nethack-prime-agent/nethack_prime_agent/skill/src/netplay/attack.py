"""attack: walk adjacent to a monster and melee it.

The code twin of the retired `np_melee_attack` tool: verify a monster is at
(x, y), path next to it, and bump-attack until it is gone or unreachable.
This file is YOURS TO EDIT: improve it, and what you change persists.
"""
from __future__ import annotations

from netplay import _base
from netplay.move import (_extra_walkable, _game_grid, _passable, move_to,
                          route, step_key)

__all__ = ["attack"]


def _monster_at(obs: str, x: int, y: int):
    for m in _base.monsters(obs):
        if tuple(m.get("pos", ())) == (x, y):
            return m
    return None


async def attack(x: int, y: int, max_swings: int = 10) -> str:
    """Melee the monster at (x, y): approach, then bump-attack.

    Fails with a message when no monster is there (it moved or died), when no
    adjacent tile is reachable, or after `max_swings` swings. Pets are attacked
    like anything else if you aim at them -- check `_base.monsters()` first.
    """
    obs = await _base.screen()
    for _ in range(max_swings):
        if _monster_at(obs, x, y) is None:
            return f"netplay.attack: there is no monster at ({x}, {y}).\n{obs}"
        here = _base.position(obs)
        if here is None:
            return obs
        dx, dy = x - here[0], y - here[1]
        if max(abs(dx), abs(dy)) <= 1 and (dx, dy) != (0, 0):
            key = step_key((dx > 0) - (dx < 0), (dy > 0) - (dy < 0))
            if key is None:
                return obs
            obs = await _base.press(key)
            continue
        # Not adjacent: walk to the closest passable neighbor of the target.
        rows = _game_grid(obs)
        extra = _extra_walkable(obs)
        best = None
        for ax in (-1, 0, 1):
            for ay in (-1, 0, 1):
                if ax == ay == 0:
                    continue
                nx, ny = x + ax, y + ay
                if not _passable(rows, nx, ny, extra=extra):
                    continue
                if route(obs, (nx, ny)) is None:
                    continue
                d = max(abs(nx - here[0]), abs(ny - here[1]))
                if best is None or d < best[0]:
                    best = (d, nx, ny)
        if best is None:
            return (f"netplay.attack: no reachable tile adjacent to the "
                    f"monster at ({x}, {y}).\n{obs}")
        obs = await move_to(best[1], best[2])
    return obs
