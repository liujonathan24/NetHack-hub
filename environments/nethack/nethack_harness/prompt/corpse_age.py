"""Reconstruct corpse age, because NetHack refuses to tell you.

THE PROBLEM
-----------
Eating a rotten corpse is ~18% of all agent deaths in our corpus (30 rollouts
show the taint message; all 30 died, 29 within 9-18 game turns, at a median of
100% HP). NetHack decides rot with

    rotted = (monstermoves - otmp->age) / (10 + rn2(20))      # eat.c:1623

where `age` is the game turn the monster DIED. Above ~5 the meal is tainted, so
the danger threshold is roughly 50-150 turns since death.

But that number is invisible. A corpse's inventory line never changes as it
rots -- the `"rotted "` prefix in objnam.c:905 is for corroded equipment, not
food -- and `otmp->age` is not exposed through the RL interface (our
`InventoryItem` has letter/description/glyph and no age). So a fresh newt corpse
and a 500-turn-old one are the same string, and the agent cannot tell them
apart. Neither could a human without keeping the count in their head.

THE RECONSTRUCTION
------------------
NetHack does announce kills, and the kill turn IS the `age` the formula uses. So
we log every "You kill the X" with the game turn and match corpses in inventory
back to the most recent kill of that species.

LIMITS, stated plainly:
  * A corpse picked up off the floor that the agent did not kill has no known
    age. It is reported as UNKNOWN, not guessed -- floor corpses are the
    dangerous ones (median 48 game turns between kill and pickup, p75 280).
  * Two kills of the same species bind to the most recent, which under-estimates
    age. Under-estimating is the unsafe direction, so the caution threshold is
    deliberately set well below NetHack's own.
  * Lichen and lizard corpses never rot in NetHack; they are exempted rather
    than flagged, or the warning becomes noise the agent learns to ignore.
"""
from __future__ import annotations

import re

#: Corpses that never rot (NetHack special-cases these).
_NEVER_ROTS = ("lichen", "lizard")

#: Game turns since death beyond which eating can make you sick. Derived, not
#: guessed: eat.c:1673 sickens on `rotted > 3`, eat.c:1623 sets
#: `rotted = (monstermoves - age) / (10 + rn2(20))`, and a CURSED corpse adds 2.
#: Worst case (cursed, rn2(20) == 0) that is `moves - age > 30`. So 30 is the
#: earliest a corpse can ever hurt you -- the only threshold safe to act on,
#: since we cannot see curse status or the RNG.
RISKY_AGE = 30

#: Show a countdown once a corpse is within this many turns of RISKY_AGE.
COUNTDOWN_WINDOW = 20

#: Within this many turns, eating becomes urgent enough to abandon whatever
#: multi-step skill is running. See `urgent_corpses`.
URGENT_WINDOW = 5


def rots_in(age: int) -> int:
    """Turns of edibility left. Negative means already past the safe window."""
    return RISKY_AGE - int(age)


def urgent_corpses(inventory, state_or_env, game_turn) -> list[str]:
    """Carried corpses that will spoil within `URGENT_WINDOW` turns.

    Used to break out of closed-loop skills. `np_explore_level` runs up to 100
    game turns inside ONE agent turn, which is more than the entire safe window
    of a corpse -- so without an interrupt the agent reliably returns from
    exploring to find the food it was carrying has become the thing that kills
    it. That is the measured pathway for ~18% of deaths.
    """
    out = []
    for item in inventory or ():
        desc = getattr(item, "description", "") or ""
        if "corpse" not in desc.lower():
            continue
        m = _CORPSE_RE.search(desc)
        if not m:
            continue
        sp = _species(m.group(1))
        if any(n in sp for n in _NEVER_ROTS):
            continue
        killed = kill_log(state_or_env).get(sp)
        if killed is None:
            continue                       # unknown age: cannot time it
        left = rots_in(int(game_turn or 0) - int(killed))
        if 0 < left <= URGENT_WINDOW:
            out.append(f"{sp} corpse ({getattr(item, 'letter', '?')}) rots in {left} move(s)")
    return out

_KILL_RE = re.compile(
    r"You (?:kill|destroy|smite)(?: the)?\s+(?:poor\s+)?([a-z][a-z '-]+?)[!.]", re.I)
_CORPSE_RE = re.compile(r"([a-z][a-z '-]+?)\s+corpse", re.I)

#: Articles and buff/curse/condition words NetHack prefixes onto an item name.
#: Without stripping these, "a newt corpse" yields the species "a newt" and
#: every kill-log lookup misses -- the annotation then claims UNKNOWN for a
#: monster the agent killed thirty seconds ago.
_NOISE = {
    "a", "an", "the", "uncursed", "blessed", "cursed", "partly", "eaten",
    "rotted", "greased", "diluted", "empty", "unlabeled", "your",
}


def _species(raw: str) -> str:
    toks = [t for t in raw.lower().replace("+", " ").split()
            if t and not t.isdigit() and t not in _NOISE]
    return " ".join(toks)


def kill_log(state_or_env) -> dict:
    """The shared species -> kill-turn map.

    Stored on the ENV, not the render state, because both layers need it and
    only one of them can see `state`: the renderer writes the log while drawing
    the inventory, and the skill layer reads it mid-macro to decide whether to
    abort. One game, one env, one log.
    """
    env = None
    if isinstance(state_or_env, dict):
        env = state_or_env.get("env")
    else:
        env = state_or_env
    if env is None:
        # No env (unit tests / synthetic state): fall back to the dict itself.
        if isinstance(state_or_env, dict):
            return state_or_env.setdefault("_kill_log", {})
        return {}
    log = getattr(env, "_kill_log", None)
    if log is None:
        log = {}
        try:
            env._kill_log = log
        except Exception:
            return {}
    return log


def note_kills(state, messages, game_turn) -> None:
    """Record `species -> game turn` for every kill announced this turn."""
    if state is None or not messages:
        return
    log = kill_log(state)
    for m in messages:
        for hit in _KILL_RE.finditer(m or ""):
            log[hit.group(1).strip().lower()] = int(game_turn or 0)


def annotate_corpse(description: str, state, game_turn) -> str:
    """Append an age/risk note to a corpse's inventory line."""
    if state is None or "corpse" not in (description or "").lower():
        return description
    m = _CORPSE_RE.search(description)
    if not m:
        return description
    species = _species(m.group(1))
    if any(s in species for s in _NEVER_ROTS):
        return f"{description}  [never rots - safe to eat]"
    killed = kill_log(state).get(species)
    if killed is None:
        return (f"{description}  [age UNKNOWN - you did not kill this; "
                f"it may already be rotten. Eating it can be fatal.]")
    age = int(game_turn or 0) - int(killed)
    left = rots_in(age)
    if left <= 0:
        return f"{description}  [~{age} turns dead - RISKY, likely tainted; do not eat]"
    if left <= COUNTDOWN_WINDOW:
        urgent = " EAT IT NOW" if left <= URGENT_WINDOW else ""
        return f"{description}  [~{age} turns dead - ROTS IN {left} MOVES.{urgent}]"
    return f"{description}  [~{age} turns dead - fresh]"
