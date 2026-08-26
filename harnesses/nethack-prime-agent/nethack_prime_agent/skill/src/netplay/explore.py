"""Movement and exploration. YOURS TO EDIT.

This is the seed generation: a breadth-first route over the rendered map, then
a straight-line walk that stops the moment the game says something happened.
It is deliberately plain. Obvious things it does NOT do, if you want somewhere
to start:

  - it re-plans only on interruption, never mid-path;
  - it treats every unknown glyph as a wall, so it under-uses doorways;
  - `explore()` walks to the nearest frontier, not the most informative one;
  - it has no memory between calls -- every route is planned from scratch.
"""

from __future__ import annotations

from netplay import _base

__all__ = ["move_to", "explore", "route", "step_key", "WALKABLE"]

# Glyphs you can stand on. Items (`%$*?!/=("[)`) are walkable -- they sit on
# floor. `+` is BOTH a closed door and a spellbook on the floor; it is included
# because walking into a closed door is how you open it, and the wasted turn
# when it is a spellbook is cheaper than routing around every one.
WALKABLE = set(".#<>_`{}\\%$*?!/=(\"[)+^|-")

# Directions as (dx, dy) -> the key that moves you that way. NetHack's vi keys.
_DIRS: dict[tuple[int, int], str] = {
    (0, -1): "k", (0, 1): "j", (-1, 0): "h", (1, 0): "l",
    (-1, -1): "y", (1, -1): "u", (-1, 1): "b", (1, 1): "n",
}


def step_key(dx: int, dy: int) -> str | None:
    """The movement key for a single step, or None if it is not one step."""
    return _DIRS.get((dx, dy))


def _passable(rows: list[str], x: int, y: int) -> bool:
    if not (0 <= y < len(rows)):
        return False
    row = rows[y]
    if not (0 <= x < len(row)):
        return False
    ch = row[x]
    # `@` is you; monsters are letters. Route THROUGH neither: a letter is a
    # monster to be fought or avoided, not a tile, and pathing into a pet
    # swaps places at best.
    return ch in WALKABLE or ch == "@"


def route(obs: str, target: tuple[int, int],
          start: tuple[int, int] | None = None) -> list[tuple[int, int]] | None:
    """Breadth-first route from `start` (default: you) to `target`.

    Returns the list of tiles to step onto, excluding the start and including
    the target, or None when no route over KNOWN tiles exists. "No route" is
    usually "explore more first", not "impossible".
    """
    rows = _base.grid(obs)
    if not rows:
        return None
    origin = start or _base.position(obs)
    if origin is None or not _passable(rows, *target):
        return None

    seen = {origin}
    frontier = [(origin, [])]
    while frontier:
        nxt = []
        for (cx, cy), path in frontier:
            for (dx, dy) in _DIRS:
                step = (cx + dx, cy + dy)
                if step in seen or not _passable(rows, *step):
                    continue
                if step == target:
                    return path + [step]
                seen.add(step)
                nxt.append((step, path + [step]))
        frontier = nxt
    return None


def _interrupted(obs: str) -> bool:
    """True when the observation says something happened worth stopping for.

    Deliberately blunt: ANY message stops the walk. That costs extra `screen()`
    calls on harmless messages, and it is the single easiest thing here to
    improve -- but a walk that ignores "You are hit!" is worse.
    """
    return bool(_base.messages(obs))


async def move_to(x: int, y: int, max_steps: int = 30,
                  max_replans: int = 2) -> str:
    """Walk to (x, y) in the MAP frame. Plan, walk, re-plan on drift, stop on
    interruption.

    Returns the last observation. Read it -- arriving and being stopped en route
    look the same from the return type alone; compare `_base.position()` against
    the target if you need to know which happened.

    `max_replans` is a HARD bound on re-planning. The first version of this
    recursed on every drifted step, which -- if the target was never reachable
    or the position kept drifting -- looped without end (a seed bug that pressed
    200k keys in one game). Re-planning is now a bounded loop, and the frozen
    floor raises `EpisodeOver` past the call budget regardless.
    """
    obs = await _base.screen()
    for _ in range(max_replans):
        here = _base.position(obs)
        if here == (x, y):
            return obs
        path = route(obs, (x, y))
        if path is None:
            return (f"netplay.move_to: no known route to ({x}, {y}). Explore "
                    f"toward it first, or open a door on the way.\n{obs}")
        drifted = False
        for step in path[:max_steps]:
            cur = _base.position(obs) or here
            key = step_key(step[0] - cur[0], step[1] - cur[1])
            if key is None:
                # Position drifted from the plan (a pet swap, a trapdoor).
                # Break out to re-plan -- ONCE per outer iteration, so the
                # re-plan budget actually bounds the work.
                drifted = True
                break
            obs = await _base.press(key)
            if _interrupted(obs):
                return obs
            here = step
        if not drifted:
            return obs
    return (f"netplay.move_to: gave up reaching ({x}, {y}) after {max_replans} "
            f"re-plans -- position keeps drifting.\n{obs}")


def frontiers(obs: str) -> list[tuple[int, int]]:
    """Known-walkable tiles that touch an unrendered one -- the explore edge."""
    rows = _base.grid(obs)
    out = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in WALKABLE:
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

    Stops early when there is no reachable frontier left -- which usually means
    the level's remaining exits are behind a closed door or a hidden passage,
    so try `_base.search()` or a door from `_base.features()`.
    """
    obs = await _base.screen()
    for _ in range(max_moves):
        here = _base.position(obs)
        if here is None:
            return obs
        targets = [f for f in frontiers(obs) if f != here]
        if not targets:
            return (f"netplay.explore: no reachable unexplored edge left. Try "
                    f"search() for a hidden passage, or a closed door from "
                    f"features().\n{obs}")
        # Nearest by route length, not by straight-line distance: on this map a
        # tile two rows away can be a corridor's whole length in walking terms.
        best, best_path = None, None
        for t in sorted(targets, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1])):
            p = route(obs, t)
            if p is not None and (best_path is None or len(p) < len(best_path)):
                best, best_path = t, p
                break
        if best is None:
            return obs
        obs = await move_to(*best)
        if _interrupted(obs):
            return obs
        obs = await _base.screen()
    return obs
