"""Human pacing norms + mechanic-hint blocks for experiment E8.

`human_norms.json` is derived from the 433 NAO ascended runs: median XL on
FIRST ARRIVAL at each depth (XL inverted from the BALROG min axis; see
docs/EXPERIMENT_E8.md). The norm for LEAVING depth d is therefore the arrival
norm of d+1 (nearest lower entry when d+1 is beyond the table).
"""
from __future__ import annotations
import json, os
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def _table() -> dict[int, int]:
    raw = json.load(open(os.path.join(_HERE, "human_norms.json")))
    return {int(k): int(v) for k, v in raw["xl_by_dlvl"].items()}


def norm_xl_for_leaving(dlvl: int) -> int:
    """Median XL human winners have when ARRIVING at dlvl+1 (nearest lower)."""
    t = _table()
    target = dlvl + 1
    best = 1
    for d in sorted(t):
        if d <= target:
            best = t[d]
    return best


def norm_xl_for_arriving(dlvl: int) -> int:
    """Median XL human winners have on FIRST ARRIVAL at dlvl (nearest lower).

    The pacing directive compares against THIS on arrival. Using the leaving
    norm there (one depth ahead) made the directive fire on 100% of arrivals
    at depth >= 3 in E15 P2 r1 — an agent exactly on the human arrival pace
    was still told it was lagging.
    """
    t = _table()
    best = 1
    for d in sorted(t):
        if d <= dlvl:
            best = t[d]
    return best


MECHANIC_HINT_BLOCKS = {
    "prayer": (
        "=== PRAYER (game mechanic) ===\n"
        "Praying heals ONLY when HP is critical (below ~1/7 of max, or under 6).\n"
        "Above that the god is \"pleased\" but restores nothing and the prayer is\n"
        "wasted. Prayer works roughly once per ~1000 turns; a second prayer too\n"
        "soon angers your god. Save it: pray only at critical HP."
    ),
    "descend_pacing": (
        "=== PACING ===\n"
        "Fight monsters on your current level until your experience level is\n"
        "close to the dungeon level before taking stairs down."
    ),
}
