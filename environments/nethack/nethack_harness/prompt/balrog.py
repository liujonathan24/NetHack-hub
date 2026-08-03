"""BALROG progression score for NetHack.

Two implementations live here:

1. `balrog_progress` — the **real** BALROG metric (Paglieri et al., ICLR 2025),
   reproduced faithfully from the official scorer
   (balrog-ai/BALROG `balrog/environments/nle/progress.py` + `achievements.json`).
   BALROG tracks the `Dlvl:{depth}` and `Xp:{experience_level}` "achievements" a
   rollout reaches and sets progression to the **max normalized value** over all
   reached achievements (plus specials: the Elemental/Astral Planes and ascension).
   The value table is vendored in `balrog_achievements.json`. Report `×100` as the
   0–100 "BALROG Progress (%)". USE THIS for anything quoted against BALROG.

2. `progression_score` — a DEPRECATED analytic proxy `(DL/50)^1.3·(XL/30)^0.6`
   from before we had the real table. It reads ~2× off the real metric (e.g. at
   DL=10/XL=6 the real metric gives Dlvl:10 = 12.56%, the proxy 6.4%). Kept only
   for back-compat with older writeups; do not quote it as "BALROG".
"""
from __future__ import annotations

import functools
import json
import os
import re

_ACHIEVEMENTS_PATH = os.path.join(os.path.dirname(__file__), "balrog_achievements.json")


@functools.lru_cache(maxsize=1)
def _achievements() -> dict:
    """The vendored BALROG achievement→value table (0..1)."""
    with open(_ACHIEVEMENTS_PATH) as f:
        return json.load(f)


def balrog_progress(
    max_dlvl: int,
    xp_level: int,
    *,
    reached_planes: bool = False,
    ascended: bool = False,
) -> float:
    """The real BALROG NetHack progression in [0, 1] (×100 for the published %).

    = max normalized value over the achievements this rollout reached:
    `Dlvl:{max_dlvl}`, `Xp:{xp_level}`, and — if applicable — the Astral Plane
    ("Astral Plane") and ascension ("You ascend t"). Missing exact keys fall
    back to the nearest lower reached level (the table is monotonic per axis).
    """
    ach = _achievements()

    def _lookup(prefix: str, n: int) -> float:
        n = int(max(1, n))
        if f"{prefix}{n}" in ach:
            return ach[f"{prefix}{n}"]
        # nearest lower defined level (monotone table)
        cands = [
            int(m.group(1))
            for k in ach
            if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", k)) and int(m.group(1)) <= n
        ]
        return ach[f"{prefix}{max(cands)}"] if cands else 0.0

    vals = [_lookup("Dlvl:", max_dlvl), _lookup("Xp:", xp_level)]
    if reached_planes:
        vals.append(ach.get("Astral Plane", 0.0))
    if ascended:
        vals.append(ach.get("You ascend t", 1.0))
    return max(0.0, min(1.0, max(vals)))


def balrog_progress_min(
    max_dlvl: int,
    xp_level: int,
    *,
    reached_planes: bool = False,
    ascended: bool = False,
) -> float:
    """The same table scored with min() over the two progress axes.

    BALROG's published metric is a `max`, which lets a run that only ever
    advanced on ONE axis keep most of its headline score -- a rollout that died
    on Dlvl 2 still scores off its experience level. Rescoring BALROG's own
    published per-episode traces with `min` collapsed Gemini 3 Flash from
    3.96 to 0.37 (4 of 5 episodes to zero), while our BBOX cell held 2.35 with
    0 of 5 zeroed. So `min` is the discriminator between "descended" and
    "levelled up while stuck", and both numbers must always be reported --
    never this one alone, and never the max alone.

    Astral Plane / ascension stay maxed in: they are terminal achievements, not
    a third axis, and gating them behind a min would make a genuine ascension
    score less than a shallow one.
    """
    ach = _achievements()
    floor = min(_lookup_pair(ach, max_dlvl, xp_level))
    if ascended:
        return max(0.0, min(1.0, ach.get("You ascend t", 1.0)))
    if reached_planes:
        return max(0.0, min(1.0, max(floor, ach.get("Astral Plane", 0.0))))
    return max(0.0, min(1.0, floor))


def _lookup_pair(ach: dict, max_dlvl: int, xp_level: int) -> tuple[float, float]:
    """`(dlvl_value, xp_value)` from the achievement table, same lookup rule as
    `balrog_progress` (exact key, else nearest lower defined level)."""

    def _lookup(prefix: str, n: int) -> float:
        n = int(max(1, n))
        if f"{prefix}{n}" in ach:
            return ach[f"{prefix}{n}"]
        cands = [
            int(m.group(1))
            for k in ach
            if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", k)) and int(m.group(1)) <= n
        ]
        return ach[f"{prefix}{max(cands)}"] if cands else 0.0

    return _lookup("Dlvl:", max_dlvl), _lookup("Xp:", xp_level)


def balrog_both(
    max_dlvl: int,
    xp_level: int,
    *,
    reached_planes: bool = False,
    ascended: bool = False,
) -> tuple[float, float]:
    """`(max, min)` in one call -- the pair every report must carry.

    Also the cheapest way to spot an xp-carried rollout: `max > 0 and min == 0`
    means the score came entirely from levelling, not descent.
    """
    kw = {"reached_planes": reached_planes, "ascended": ascended}
    return balrog_progress(max_dlvl, xp_level, **kw), balrog_progress_min(
        max_dlvl, xp_level, **kw
    )


# ---- DEPRECATED analytic proxy (pre-real-table). Do not quote as "BALROG". ----
# Calibrated against four headline points from the BALROG paper.
_DL_EXP = 1.3
_XL_EXP = 0.6
_DL_NORM = 50.0
_XL_NORM = 30.0


def progression_score(max_dlvl: int, xp_level: int) -> float:
    """Empirical-ish P(ascend) given (max DL reached, current XL).

    Returns a float in [0, 1]. Pass `max_dlvl` (not current dlvl) so we
    capture the deepest level the agent has touched, mirroring BALROG.
    """
    dl = max(0.0, float(max_dlvl)) / _DL_NORM
    xl = max(0.0, float(xp_level)) / _XL_NORM
    raw = (dl ** _DL_EXP) * (xl ** _XL_EXP)
    return max(0.0, min(1.0, raw))


def progression_tier(score: float) -> str:
    """Human-readable bucket from a progression score."""
    if score >= 0.5:
        return "endgame"
    if score >= 0.1:
        return "midgame"
    if score >= 0.01:
        return "past_mines"
    if score > 0:
        return "early"
    return "spawn"
