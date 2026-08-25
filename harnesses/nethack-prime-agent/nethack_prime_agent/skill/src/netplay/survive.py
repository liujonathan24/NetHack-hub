"""Staying alive. YOURS TO EDIT.

Dying ends the episode, so everything else is worth zero if this is wrong. The
seed policy is the conservative reading of SKILL.md's advice and nothing more:

  - the prayer threshold is a flat HP fraction, ignoring prayer timeout, which
    is the single most common way to waste the one prayer you get;
  - it cannot eat, because it has no inventory parsing -- when you are Weak
    you must answer the game's own prompts with `_base.press`;
  - it never flees, because fleeing needs a destination and this module has no
    map opinion of its own.
"""

from __future__ import annotations

from netplay import _base

__all__ = ["hp_fraction", "in_danger", "pray_safely", "recover"]

# NetHack's own "low hitpoint warning" fires around 1/7. Praying above this is
# usually wasted; praying below it is the intended use of your one prayer.
PRAYER_THRESHOLD = 1.0 / 7.0


def hp_fraction(obs: str) -> float | None:
    """Current HP as a fraction of max, or None when the status line is absent."""
    st = _base.status(obs)
    hp, maxhp = st.get("hp"), st.get("maxhp")
    if hp is None or not maxhp:
        return None
    return hp / maxhp


def in_danger(obs: str, threshold: float = 0.34) -> bool:
    """True when HP is below `threshold` -- the "disengage now" signal."""
    frac = hp_fraction(obs)
    return frac is not None and frac < threshold


async def pray_safely() -> str:
    """Pray only when HP is genuinely critical. Refuses otherwise.

    You get roughly one prayer per 1000 turns. Spending it at half health is
    how the next emergency kills you, so this refuses rather than asking.
    """
    obs = await _base.screen()
    frac = hp_fraction(obs)
    if frac is None:
        return f"netplay.pray_safely: cannot read HP; not praying.\n{obs}"
    if frac > PRAYER_THRESHOLD:
        return (f"netplay.pray_safely: HP is {frac:.0%} of max, above the "
                f"{PRAYER_THRESHOLD:.0%} threshold -- NOT praying. Rest or "
                f"retreat instead.\n{obs}")
    return await _base.pray()


async def recover(target: float = 0.75, rounds: int = 6) -> str:
    """Rest until HP reaches `target` of max, or something interrupts.

    Resting with a monster visible is how you die while resting, so this stops
    as soon as one appears rather than burning the whole budget.
    """
    obs = await _base.screen()
    for _ in range(rounds):
        frac = hp_fraction(obs)
        if frac is None or frac >= target:
            return obs
        hostile = [m for m in _base.monsters(obs) if not m["pet"]]
        if hostile:
            return (f"netplay.recover: {hostile[0]['name']} is visible -- "
                    f"not resting. Fight it or leave.\n{obs}")
        await _base.rest(count=10)
        obs = await _base.screen()
    return obs
