"""Melee. YOURS TO EDIT.

The seed policy walks adjacent and attacks until the tile is clear. What it
does badly, in rough order of how much it costs you:

  - it never retreats, so it will happily trade to death;
  - it picks the nearest monster, not the most dangerous or the weakest;
  - it has no idea what any monster actually is -- no threat table;
  - it cannot use a ranged attack, because the primitive floor has no `throw`
    (you would answer the game's own prompts with `_base.press`).
"""

from __future__ import annotations

from netplay import _base
from netplay.explore import move_to, step_key

__all__ = ["threats", "attack", "clear_threats"]


def threats(obs: str) -> list[dict]:
    """Visible non-pet monsters, nearest first by Chebyshev distance.

    Pets are excluded, not merely flagged: attacking one is a real mistake the
    harness will not stop you making.
    """
    here = _base.position(obs)
    out = [m for m in _base.monsters(obs) if not m["pet"]]
    if here is None:
        return out
    return sorted(out, key=lambda m: max(abs(m["pos"][0] - here[0]),
                                         abs(m["pos"][1] - here[1])))


async def attack(x: int, y: int, max_swings: int = 12) -> str:
    """Pursue the monster at (x, y) and attack until the tile is clear.

    Attacking IS moving into the tile, so this presses a direction key rather
    than any attack command. Returns the last observation.
    """
    obs = await _base.screen()
    here = _base.position(obs)
    if here is None:
        return obs

    if max(abs(x - here[0]), abs(y - here[1])) > 1:
        # Walk to a tile adjacent to the target, not onto it.
        adjacent = [(x + dx, y + dy)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx, dy) != (0, 0)]
        for spot in sorted(adjacent, key=lambda p: max(abs(p[0] - here[0]),
                                                       abs(p[1] - here[1]))):
            obs = await move_to(*spot)
            if _base.position(obs) == spot:
                break
        else:
            return (f"netplay.attack: could not get adjacent to ({x}, {y}).\n{obs}")

    for _ in range(max_swings):
        here = _base.position(obs)
        if here is None:
            return obs
        key = step_key(x - here[0], y - here[1])
        if key is None:
            return (f"netplay.attack: ({x}, {y}) is not adjacent any more -- "
                    f"it moved. Re-read the map.\n{obs}")
        obs = await _base.press(key)
        if _base.prompt_open(obs):
            # The game is asking something (a #force, a "Really attack?").
            # Answering blind is how a pet gets killed; hand it back instead.
            return (f"netplay.attack: the game is asking you something. "
                    f"Answer it with _base.press().\n{obs}")
        # No monster left on the tile is the only reliable "it died" signal we
        # have: the message text varies by monster and by killing blow. This
        # costs one `screen()` per swing, which is the obvious thing to
        # optimise if you rewrite this loop.
        after = await _base.screen()
        if not any(m["pos"] == (x, y) for m in _base.monsters(after)):
            return after
        obs = after
    return obs


async def clear_threats(max_fights: int = 4) -> str:
    """Attack visible non-pet monsters, nearest first, up to `max_fights`."""
    obs = await _base.screen()
    for _ in range(max_fights):
        found = threats(obs)
        if not found:
            return obs
        obs = await attack(*found[0]["pos"])
        obs = await _base.screen()
    return obs
