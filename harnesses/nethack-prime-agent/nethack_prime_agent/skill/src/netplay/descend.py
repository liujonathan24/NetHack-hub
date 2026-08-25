"""Getting deeper. YOURS TO EDIT.

Depth is the scored quantity that moves most, so this is the policy worth the
most attention. The seed version is the three-step recipe from SKILL.md --
find `>`, walk to it, press `>` -- with no cleverness at all:

  - it never considers a trapdoor, a hole, or digging;
  - it gives up on a level rather than searching for a hidden staircase;
  - it does not check whether descending now is survivable.
"""

from __future__ import annotations

from netplay import _base
from netplay.explore import explore, move_to

__all__ = ["find_stairs", "descend", "dive"]


def find_stairs(obs: str, direction: str = "down") -> tuple[int, int] | None:
    """The (x, y) of a visible up/down staircase, or None if none is rendered."""
    want = "DOWN" if direction.lower().startswith("d") else "UP"
    for label, coords in _base.features(obs).items():
        if "stairs" in label.lower() and want in label.upper() and coords:
            return coords[0]
    return None


async def descend(search_rounds: int = 3) -> str:
    """Descend one level: find the down staircase, walk to it, press `>`.

    Explores up to `search_rounds` times when no staircase is visible yet.
    Returns the observation after the descent attempt -- check `Dlvl:` in
    `_base.status()` to confirm it actually worked.
    """
    obs = await _base.screen()
    start_dlvl = _base.status(obs).get("dlvl")

    for _ in range(search_rounds + 1):
        stairs = find_stairs(obs, "down")
        if stairs is not None:
            break
        obs = await explore()
        obs = await _base.screen()
    else:
        return (f"netplay.descend: no down staircase found after "
                f"{search_rounds} explore rounds.\n{obs}")

    if _base.position(obs) != stairs:
        obs = await move_to(*stairs)
        if _base.position(obs) != stairs:
            return (f"netplay.descend: could not reach the staircase at "
                    f"{stairs} -- something interrupted the walk.\n{obs}")

    obs = await _base.press(">")
    now = _base.status(obs).get("dlvl")
    if now is not None and start_dlvl is not None and now <= start_dlvl:
        # Pressing `>` off-staircase prints a message and costs nothing. Say so
        # rather than returning a success-shaped result.
        return (f"netplay.descend: still on Dlvl {now} after pressing '>'. "
                f"Read the message -- you may not be on the stairs.\n{obs}")
    return obs


async def dive(levels: int = 3) -> str:
    """Descend `levels` times, stopping at the first level that fails."""
    obs = await _base.screen()
    for n in range(levels):
        obs = await descend()
        if "netplay.descend:" in obs.split("\n", 1)[0]:
            return f"netplay.dive: stopped after {n} of {levels} levels.\n{obs}"
    return obs
