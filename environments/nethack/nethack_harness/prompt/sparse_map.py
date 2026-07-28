"""Sparse entity map: everything that is NOT wall, floor, or unexplored rock.

WHY
---
The dungeon is overwhelmingly terrain. A 21x79 ASCII grid spends nearly all of
its tokens on `|`, `-`, `.` and blank rock, and the measured result is that the
grid is not what the agent acts on: across 17 cells the format of the map moved
BALROG progression less than chance (cell identity explained 13.7% of outcome
variance against a ~19% random-grouping baseline), while only 14-50% of
`move_to` targets ever matched a coordinate the map had shown.

What the agent demonstrably does use is a short list of *things with
coordinates*. So this encoding drops terrain entirely and emits only the
entities: the hero, monsters (live and remembered), items, and the features
that matter (stairs, doors, fountains, altars).

IT ALSO CONSOLIDATES
--------------------
`=== ADJACENT ===`, `=== VISIBLE FEATURES ===`, `=== VISIBLE MONSTERS ===` and
`=== REMEMBERED MONSTERS ===` are all folded in here and suppressed at their own
sites. That is not tidying -- it repairs the delivery-timing axis. `BBOX` was
supposed to withhold the map until `reveal` was called, but VISIBLE FEATURES
kept publishing stair coordinates on ~100% of turns, so nothing was ever
actually withheld and `reveal` fired on 2.1% of turns. With every entity living
under MAP, withholding MAP withholds something real for the first time.

STALENESS IS EXPLICIT
---------------------
Monsters carry an age. NetHack reverts the glyph plane when a monster leaves
line of sight but leaves the character on the char plane, so the map and the
monster list disagree by construction -- the source of `There is no monster at
(x,y)` (22.5% of turns in the full-map `best_fixed` cell). Here a sighting is
either current or explicitly aged, and never silently either.
"""
from __future__ import annotations

from nethack_harness.prompt.features import visible_monsters

#: Cap per category so a treasure room cannot flood the observation.
_CAP = 12


def _features(structured, state):
    """Stairs / doors / fountains / altars, via the existing feature scanner."""
    try:
        from nethack_harness.prompt.features import visible_features, format_features
        feats = visible_features(state["raw_obs"])
        return format_features(feats)[:_CAP]
    except Exception:
        return []


def sparse_entity_map(structured, state) -> str:
    """Render the entity-only map body.

    Lines are `kind: detail` so the block stays greppable by the model and
    cheap to diff turn over turn.
    """
    if state is None:
        return "(no state; entity map unavailable)"
    raw = state.get("raw_obs")
    if raw is None:
        return "(no observation)"

    out: list[str] = []
    st = getattr(structured, "status", None) or {}
    px, py = st.get("x"), st.get("y")
    if px is not None:
        out.append(f"you: ({px},{py})  HP {st.get('hitpoints')}/{st.get('max_hitpoints')}  Dlvl {st.get('depth')}")

    # What the hero is STANDING ON. Without terrain there is no other way to
    # learn this: the `@` covers its own tile, so a down-staircase underfoot is
    # invisible in an entity-only view. Observed live -- the agent began on the
    # up-stair and on arriving at Dlvl 2 stood on stairs both times without the
    # observation ever saying so. The dangerous mirror case is standing on a `>`
    # and not knowing to descend, which is the single action the whole task is
    # scored on.
    under = getattr(structured, "under_player", None)
    if under:
        u = str(under).strip()
        low = u.lower()
        flag = ""
        if "stairs down" in low or "staircase down" in low or u.strip() == ">":
            flag = "  <- DESCEND FROM HERE"
        elif "stairs up" in low or "staircase up" in low:
            flag = "  <- up-staircase"
        out.append(f"standing on: {u}{flag}")
    else:
        # DO NOT claim "floor". `@` covers its own tile, so the char plane cannot
        # answer this, and `under_player` is only populated when NetHack happens
        # to have said so. Asserting floor here would be inventing information --
        # and the tile we would most be lying about is a down-staircase, the one
        # thing the score depends on. Fall back to the remembered stair coords
        # (the same memo the descend hint uses), then admit ignorance.
        pos = (px, py)
        known = state.get("_seen_stairs_down") or set()
        if pos in known:
            out.append("standing on: stairs down  <- DESCEND FROM HERE")
        else:
            out.append("standing on: unknown (the @ hides its own tile; "
                       "call np_look to identify it)")

    # --- monsters: live first, then aged sightings -------------------------
    live, pets = [], []
    for m in visible_monsters(raw):
        entry = f"{m.name} ({m.x},{m.y}) {m.steps} step(s) {m.bearing}"
        (pets if m.is_pet else live).append(
            entry + (" [PET - do not attack]" if m.is_pet else "")
        )
    if live:
        out.append("monsters (visible now): " + "; ".join(live[:_CAP]))
    if pets:
        out.append("pets: " + "; ".join(pets[:4]))

    if state.get("_remember_monsters"):
        try:
            from nethack_harness.prompt.features import remembered_monsters
            _rm = remembered_monsters(state, raw, st.get("time"), st.get("depth"))
            if _rm:
                out.append("monsters (SEEN EARLIER, not visible now): " + "; ".join(_rm))
        except Exception:
            pass

    # --- features and items -------------------------------------------------
    feats = _features(structured, state)
    if feats:
        out.append("features: " + "; ".join(feats))

    # Statues draw a monster letter but are not monsters. Naming them is the
    # only signal the agent can get; see features.statues.
    try:
        from nethack_harness.prompt.features import statues as _statues
        st_list = _statues(raw)
        if st_list:
            out.append("statues: " + "; ".join(st_list[:_CAP]))
    except Exception:
        pass

    if not out:
        out.append("(nothing but terrain in view)")
    out.append(
        "NOTE: terrain (walls, floor, corridors) is omitted. Coordinates above "
        "are exact; explore to discover what is not listed."
    )
    return "\n".join(out)
