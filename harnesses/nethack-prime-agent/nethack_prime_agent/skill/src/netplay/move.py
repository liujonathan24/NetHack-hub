"""move_to: walk to a coordinate.

The one policy the round-0 seed ships -- the code twin of the retired np_move_to
tool. Everything else (exploring, descending, fighting, resting) is YOURS to
write: add new .py files beside this one and they load on import.

This file is YOURS TO EDIT: improve it, and what you change persists.
"""

from __future__ import annotations

import re as _re

from netplay import _base

__all__ = ["move_to", "route", "step_key", "WALKABLE"]


def _game_grid(obs: str) -> list[str]:
    """Map rows aligned so grid[y][x] matches the game coordinate frame.

    ``_base.grid()`` strips leading blank lines from the MAP section (via
    ``str.strip('\\n')`` in ``_sections``).  When the map has one or more
    blank leading rows, every grid Y is shifted from the status / features /
    monsters frame.  We detect the exact offset by locating ``@`` in the raw
    grid and comparing it to the status position, then prepend the correct
    number of dummy rows.
    """
    rows = _base.grid(obs)
    if not rows:
        return rows
    pos = _base.position(obs)
    if pos is None:
        return rows
    px, py = pos
    if 0 <= py < len(rows) and 0 <= px < len(rows[py]) and rows[py][px] == "@":
        return rows  # No offset
    # Find @ in the raw grid to compute the exact row offset.
    at_row = None
    for ry, row in enumerate(rows):
        if 0 <= px < len(row) and row[px] == "@":
            at_row = ry
            break
    if at_row is not None:
        offset = py - at_row
        if offset > 0:
            return [""] * offset + rows
    return [""] + rows  # Fallback: offset 1


def _extra_walkable(obs: str) -> set[tuple[int, int]]:
    """Tiles from VISIBLE FEATURES that are walkable despite a wall glyph.

    Open doors and gaps render as ``|`` or ``-`` (the same glyphs as walls)
    but are confirmed walkable by the harness feature list.
    """
    out: set[tuple[int, int]] = set()
    for label, coords in _base.features(obs).items():
        if "door" in label.lower():
            out.update(coords)
    return out


# Glyphs you can stand on. Items (`%$*?!/=("[)`) are walkable -- they sit on
# floor. `+` is BOTH a closed door and a spellbook on the floor; it is included
# because walking into a closed door is how you open it, and the wasted turn
# when it is a spellbook is cheaper than routing around every one.
# ``|`` and ``-`` are WALL glyphs and are NOT in this set.  Open doors that
# share those glyphs are handled via ``_extra_walkable()`` and the context
# check in ``_passable()``.
WALKABLE = set(".#<>_{\\%$*?!/=(\"[)+'")

# Directions as (dx, dy) -> the key that moves you that way. NetHack's vi keys.
_DIRS: dict[tuple[int, int], str] = {
    (0, -1): "k", (0, 1): "j", (-1, 0): "h", (1, 0): "l",
    (-1, -1): "y", (1, -1): "u", (-1, 1): "b", (1, 1): "n",
}


def step_key(dx: int, dy: int) -> str | None:
    """The movement key for a single step, or None if it is not one step."""
    return _DIRS.get((dx, dy))


def _passable(rows: list[str], x: int, y: int,
          extra: set[tuple[int, int]] | None = None) -> bool:
    if not (0 <= y < len(rows)):
        return False
    row = rows[y]
    if not (0 <= x < len(row)):
        return False
    ch = row[x]
    if ch == "@" or ch in WALKABLE:
        return True
    # Open door / gap confirmed by the harness feature list.
    if extra and (x, y) in extra:
        return True
    # ``|`` inside a horizontal wall row (``-`` on both sides) is an open
    # door gap -- walkable.  Requiring both sides avoids false positives at
    # wall corners.
    if ch == "|":
        left = row[x - 1] if x > 0 else " "
        right = row[x + 1] if x + 1 < len(row) else " "
        if left == "-" and right == "-":
            return True
    # ``-`` inside a vertical wall column (``|`` above and below) is a
    # broken wall -- walkable.
    if ch == "-":
        above = rows[y - 1][x] if y > 0 and x < len(rows[y - 1]) else " "
        below = rows[y + 1][x] if y + 1 < len(rows) and x < len(rows[y + 1]) else " "
        if above == "|" and below == "|":
            return True
    # `@` is you; monsters are letters. Route THROUGH neither: a letter is a
    # monster to be fought or avoided, not a tile, and pathing into a pet
    # swaps places at best.
    return False


def route(obs: str, target: tuple[int, int],
          start: tuple[int, int] | None = None,
          blocked: set[tuple[int, int]] | None = None) -> list[tuple[int, int]] | None:
    """Breadth-first route from `start` (default: you) to `target`.

    Returns the list of tiles to step onto, excluding the start and including
    the target, or None when no route over KNOWN tiles exists. "No route" is
    usually "explore more first", not "impossible".

    ``blocked`` tiles are treated as impassable -- used to avoid tiles the
    game said were walls or otherwise blocked on a previous attempt.
    """
    rows = _game_grid(obs)
    if not rows:
        return None
    extra = _extra_walkable(obs)
    origin = start or _base.position(obs)
    if origin is None or not _passable(rows, *target, extra=extra):
        return None
    if blocked and target in blocked:
        return None

    seen = {origin}
    if blocked:
        seen |= blocked
    frontier = [(origin, [])]
    while frontier:
        nxt = []
        for (cx, cy), path in frontier:
            for (dx, dy) in _DIRS:
                step = (cx + dx, cy + dy)
                if step in seen or not _passable(rows, *step, extra=extra):
                    continue
                # NetHack forbids diagonal moves that cut through a wall
                # corner: both orthogonal neighbours must be passable too.
                if dx != 0 and dy != 0:
                    if (not _passable(rows, cx + dx, cy, extra=extra) or
                            not _passable(rows, cx, cy + dy, extra=extra)):
                        continue
                if step == target:
                    return path + [step]
                seen.add(step)
                nxt.append((step, path + [step]))
        frontier = nxt
    return None


_last_msg: str = ""

def _interrupted(obs: str) -> bool:
    """True when a NEW, non-benign game message appears.

    The NetHack message line persists: if nothing new happens, the old message
    stays.  The original version interrupted on ANY message (including the
    harness ``[call#N]`` beacon that is appended to every observation), which
    meant ``move_to`` stopped after every single keypress.

    We strip the ``[call#N]`` beacon and only interrupt when the remaining
    message text changes from the last one we saw AND is not a known benign
    message (door opening, pet displacement, distant sounds, stat gains).
    This lets ``move_to`` walk through doors and past pets without stopping
    on every harmless event, while still halting on combat, damage, or
    prompts.  ``move_to`` primes this tracker with the current observation
    before walking so that the pre-walk message does not count as "new".
    """
    global _last_msg
    msg = _base.messages(obs)
    msg = _re.sub(r"\s*\[call#\d+\]\s*", "", msg).strip()
    if not msg:
        return False
    if msg == _last_msg:
        return False
    _last_msg = msg
    low = msg.lower()
    # Benign messages that should NOT interrupt a walk.  These are events
    # that never indicate danger: opening a door, a pet swapping places,
    # distant sounds, stat gains, or environmental flavour text.
    benign = (
        "the door opens", "you open", "door opens",
        "swap places", "you displaced", "you swap",
        "you hear water", "you hear noise", "you hear a sound",
        "you hear someone counting", "you hear some noises",
        "you hear footsteps", "you hear a slow",
        "you hear bubbling", "you hear water falling",
        "you hear a door open", "you hear a e note",
        "you hear the splashing", "you hear a crunching",
        "you hear a clank", "you hear a chinking",
        "you hear crashing", "for sale", "zorkmids",
        "you feel tough", "you feel strong", "you feel agile",
        "you feel wise", "you feel charismatic",
        "you feel more confident", "you feel tougher",
        "you return to your normal self",
        "you can see again",
        "the kitten picks up", "the kitten drops",
        "the little dog picks up", "the little dog drops",
        "the dog picks up", "the dog drops",
        "the cat picks up", "the cat drops",
        "picks up a", "drops a",
    )
    if any(b in low for b in benign):
        return False
    return True




def _nearest_reachable(rows: list[str], extra: set[tuple[int, int]],
                       start: tuple[int, int], target: tuple[int, int],
                       blocked: set[tuple[int, int]]) -> tuple[int, int] | None:
    """The reachable tile that minimizes Chebyshev distance to `target`.

    BFS from `start` over passable tiles (same rules as `route`), ties broken
    by closeness to the player -- the incremental step np_move_to takes when no
    full path exists yet."""
    from collections import deque
    if start is None:
        return None
    seen = {start}
    q = deque([start])
    best = None
    best_key = None
    tx, ty = target
    sx, sy = start
    while q:
        cx, cy = q.popleft()
        if (cx, cy) != start:
            key = (max(abs(cx - tx), abs(cy - ty)),
                   max(abs(cx - sx), abs(cy - sy)))
            if best_key is None or key < best_key:
                best_key = key
                best = (cx, cy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen or (nx, ny) in blocked:
                    continue
                if not _passable(rows, nx, ny, extra=extra):
                    continue
                if dx and dy and not (_passable(rows, cx + dx, cy, extra=extra)
                                      and _passable(rows, cx, cy + dy, extra=extra)):
                    continue
                seen.add((nx, ny))
                q.append((nx, ny))
    return best


async def move_to(x: int, y: int, max_steps: int = 30,
                  max_replans: int = 5) -> str:
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
    blocked: set[tuple[int, int]] = set()
    obs = await _base.screen()
    # Prime the message tracker so the current (old) message does not look new.
    _interrupted(obs)
    for _ in range(max_replans):
        here = _base.position(obs)
        if here == (x, y):
            return obs
        path = route(obs, (x, y), blocked=blocked)
        if path is None:
            rows = _game_grid(obs)
            # Wall / out-of-bounds targets are refused for good -- exploring
            # cannot make a wall walkable. Mirrors np_move_to's messages.
            if not (0 <= y < len(rows) and 0 <= x < len(rows[y])):
                return (f"netplay.move_to: ({x}, {y}) is out of bounds -- the "
                        f"map is 0..{max((len(r) for r in rows), default=0)-1}"
                        f" x 0..{len(rows)-1}. Pick a tile on the map.\n{obs}")
            ch = rows[y][x]
            if ch in "|-" and not _passable(rows, x, y, extra=_extra_walkable(obs)):
                return (f"netplay.move_to: ({x}, {y}) is a wall (`{ch}`) -- "
                        f"not a tile you can stand on, so there is no route to "
                        f"it and never will be. Target a floor `.`, corridor "
                        f"`#`, doorway or `>`/`<` instead.\n{obs}")
            # Best-effort: no full route yet, so take ONE bounded leg toward
            # the reachable tile closest to the target (np_move_to's signature
            # behavior) -- walking there reveals map and can open a real route.
            near = _nearest_reachable(rows, _extra_walkable(obs),
                                      _base.position(obs) or here,
                                      (x, y), blocked)
            if near is None or near == here:
                return (f"netplay.move_to: no known route to ({x}, {y}) and no "
                        f"closer reachable tile. Explore elsewhere, or search "
                        f"for a hidden passage.\n{obs}")
            path = route(obs, near, blocked=blocked)
            if path is None:
                return (f"netplay.move_to: no known route to ({x}, {y}).\n{obs}")
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
            msg = _base.messages(obs)
            msg_clean = _re.sub(r"\s*\[call#\d+\]\s*", "", msg).strip()
            # Locked door on the path: NOT ours to force. Mark it blocked,
            # stop, and let the agent decide (kick, unlock, or route around).
            if msg_clean and "door" in msg_clean.lower() and (
                    "locked" in msg_clean.lower()
                    or "resists" in msg_clean.lower()):
                blocked.add(step)
                return (f"netplay.move_to: a locked door at {step} blocks the "
                        f"path to ({x}, {y}). Decide how to handle it -- kick "
                        f"it, look for another route, or pick a new target.\n{obs}")
            # The router thought this tile was passable but the game says
            # it is a wall or diagonally blocked.  Block it and re-plan.
            if msg_clean and ("it's a wall" in msg_clean.lower()
                              or "it is a wall" in msg_clean.lower()
                              or "can't move diagonally" in msg_clean.lower()
                              or "solid stone" in msg_clean.lower()):
                blocked.add(step)
                drifted = True
                break
            if _interrupted(obs):
                return obs
            # Verify we actually moved to the planned step.  If the key
            # triggered an attack or was otherwise blocked, do not advance
            # ``here`` -- retry the same step next iteration.
            actual = _base.position(obs)
            if actual is not None and actual != step:
                here = cur
            else:
                here = step
        if not drifted:
            return obs
        obs = await _base.screen()
    return (f"netplay.move_to: gave up reaching ({x}, {y}) after {max_replans} "
            f"re-plans -- position keeps drifting.\n{obs}")


