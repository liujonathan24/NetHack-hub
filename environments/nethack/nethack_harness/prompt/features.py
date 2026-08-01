"""The harness's ONE source of visible-feature / visible-monster extraction.

Everything here is in the **NLE map frame** — the same frame `blstats[0..1]`
(`Pos:`), `chars`, `a_star`, `move_to` and `descend` use. There is deliberately
no second frame anywhere in the harness.

Why this module exists
----------------------
`nethack_core.observations.extract_visible_features` reads `tty_chars`, whose
row 0 is the message line, so every coordinate it returns is one row below the
map frame the tools consume. On the committed artifact
`tools/cli_harness_eval/acceptance/task13_claude_code_seed0.turns.ndjson`
(turn 2) the renderer emitted `stairs UP at (50,18)` for a `<` that the skills
index at map row 17: an agent copying a feature coordinate into `move_to`
aimed one row past it, every single time.

Reading `chars` instead of `tty_chars` removes the conversion entirely rather
than papering over it with a `-1`, and it drops two incidental defects that came
with the tty source: the startup copyright banner (tty rows 1-4 on reset) was
being scanned for monster glyphs, and an open menu drawn over the dungeon was
scanned for features. `chars` carries neither.

`environments/nethack/tests/test_observation_frame.py` asserts on a real seeded
engine that `chars[y][x]` is the right glyph for every `<label> at (x,y)` the
renderer emits, and that no other harness module imports the tty-frame
extractor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# glyph -> label. Object classes follow NetHack's own class letters:
# `)` is the WEAPON class and `(` the TOOL class (objclass.c / `def_oc_syms`).
# The pair used to be swapped, so an agent told to arm itself walked to the
# wrong pile.
FEATURE_LABELS: dict[str, str] = {
    ">": "stairs DOWN",
    "<": "stairs UP",
    "_": "altar",
    "{": "fountain",
    "\\": "throne",
    "$": "gold",
    "%": "food/corpse",
    "[": "armor",
    ")": "weapon",
    "(": "tool",
    "?": "scroll",
    "!": "potion",
    "/": "wand",
    "=": "ring",
    '"': "amulet",
    "*": "gem/rock",
}

DOOR_CLOSED = "door (closed)"
DOOR_OPEN = "door (open/gap)"
SPELLBOOK = "spellbook"

_WALL = "-|"
# Order matters: a clipped feature list must still carry the essentials.
LABEL_ORDER = (
    "stairs DOWN", "stairs UP", DOOR_OPEN, DOOR_CLOSED,
    "altar", "fountain", "throne", "food/corpse", "potion", "scroll",
    SPELLBOOK, "wand", "ring", "amulet", "armor", "weapon", "tool",
    "gem/rock", "gold",
)
_LABEL_RANK = {label: i for i, label in enumerate(LABEL_ORDER)}

#: Coordinates per label to render. Was 3-with-"+N more", which silently
#: deleted the very thing this block exists to deliver: a closed door at
#: (22,12) fell off the list and the agent could not tell whether it had opened
#: or been truncated. Under SPARSE the feature list IS the map, so truncating it
#: removes the encoding's entire content. Show them all.
DISPLAY_CAP = 10_000


@dataclass(frozen=True)
class Feature:
    label: str
    x: int
    y: int
    glyph: str

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.label} at ({self.x},{self.y})"


def _ch(chars, y: int, x: int) -> str:
    return chr(int(chars[y, x]))


def _is_wallish(c: str) -> bool:
    return c in _WALL or c == "+"


def visible_features(raw_obs) -> list[Feature]:
    """Every navigable/notable tile on the currently-drawn map, in map coords.

    Reads the RECONCILED grid (`prompt/engine_grid.py`), the same one the
    `=== MAP ===` block renders, rather than `raw_obs.chars` directly. The two
    used to be read off different planes, which is how the seed-0 starting room's
    two converted secret doors managed to be visible on the engine's tty, absent
    from the map, and absent from this list, all at once. One grid, one answer.
    """
    if raw_obs is None:
        return []
    from nethack_harness.prompt.engine_grid import engine_map_chars

    chars = engine_map_chars(raw_obs)
    if chars is None:
        return []
    return visible_features_from_chars(chars)


def visible_features_from_chars(chars) -> list[Feature]:
    """Same, for callers that already hold the `chars` grid (the skills do).

    Skills path with `a_star(chars, ...)`, so they need features in the frame
    of the very array they path over — which is exactly what this returns.
    Before the fix, `autoexplore` and `find_and_descend` fed tty-frame door
    coordinates straight into `a_star(chars, ...)`, i.e. they pathed to the
    tile one row BELOW every door they picked.
    """
    if chars is None:
        return []
    h, w = chars.shape
    out: list[Feature] = []

    for y in range(h):
        for x in range(w):
            c = _ch(chars, y, x)
            if c == "+":
                # `+` is BOTH the closed-door glyph and the spellbook glyph.
                # A door sits inside a wall run; a spellbook lies on floor.
                # Getting this wrong sent the agent off to `kick` a spellbook.
                in_wall = (
                    (x > 0 and _is_wallish(_ch(chars, y, x - 1)))
                    or (x + 1 < w and _is_wallish(_ch(chars, y, x + 1)))
                    or (y > 0 and _is_wallish(_ch(chars, y - 1, x)))
                    or (y + 1 < h and _is_wallish(_ch(chars, y + 1, x)))
                )
                out.append(Feature(DOOR_CLOSED if in_wall else SPELLBOOK, x, y, c))
                continue
            label = FEATURE_LABELS.get(c)
            if label:
                out.append(Feature(label, x, y, c))

    out.extend(_wall_gaps(chars))
    out.sort(key=lambda f: (_LABEL_RANK.get(f.label, 99), f.y, f.x))
    return out


def _wall_gaps(chars) -> list[Feature]:
    """Open doorways and broken-wall gaps: the tiles you can leave a room by.

    NetHack draws an open door as the wall glyph rotated 90 degrees (`|` inside
    a horizontal wall run, `-` inside a vertical one) and a doorless doorway as
    `.`, so a gap is "a non-wall-ish tile with wall on both sides".

    The vertical pass used to demand `|` both above AND below, which misses
    every gap that touches a room corner — where one neighbour is the corner
    `-`. Artifact turn 2 had exactly that: an open gap at map (61,18) whose
    lower neighbour is the bottom-wall corner, so VISIBLE FEATURES listed one
    exit for that room when there were two, which is what let the "only exit"
    hint be confidently wrong.
    """
    h, w = chars.shape
    seen: set[tuple[int, int]] = set()
    out: list[Feature] = []

    def add(x: int, y: int, c: str) -> None:
        if (x, y) not in seen:
            seen.add((x, y))
            out.append(Feature(DOOR_OPEN, x, y, c))

    for y in range(h):
        for x in range(1, w - 1):
            c = _ch(chars, y, x)
            if c not in ("|", "."):
                continue
            if _ch(chars, y, x - 1) == "-" and _ch(chars, y, x + 1) == "-":
                add(x, y, c)
    for y in range(1, h - 1):
        for x in range(w):
            c = _ch(chars, y, x)
            if c not in ("-", "."):
                continue
            up, down = _ch(chars, y - 1, x), _ch(chars, y + 1, x)
            if up in _WALL and down in _WALL and "|" in (up, down):
                add(x, y, c)
    return out


def format_features(feats: Iterable[Feature]) -> list[str]:
    """Render as `["stairs DOWN at (47,6)", "door (closed) at (22,7)"]`.

    Grouped by label and capped at DISPLAY_CAP coordinates each; the ranking
    helpers below read the *structured* list, so nothing downstream has to
    re-parse a capped string (which is how the door hint used to pick the
    wrong door — `re.search` saw only the first of three coordinate pairs).
    """
    by_label: dict[str, list[Feature]] = {}
    for f in feats:
        by_label.setdefault(f.label, []).append(f)
    out: list[str] = []
    for label in LABEL_ORDER:
        group = by_label.get(label)
        if not group:
            continue
        coords = ", ".join(f"({f.x},{f.y})" for f in group[:DISPLAY_CAP])
        more = "" if len(group) <= DISPLAY_CAP else f" +{len(group)-DISPLAY_CAP} more"
        out.append(f"{label} at {coords}{more}")
    return out


def visible_feature_strings(raw_obs) -> list[str]:
    return format_features(visible_features(raw_obs))


def stairs_down(feats: Iterable[Feature]) -> list[Feature]:
    return [f for f in feats if f.label == "stairs DOWN"]


def exits(feats: Iterable[Feature]) -> list[Feature]:
    """Every way out of the current room: open doorways and closed doors."""
    return [f for f in feats if f.label in (DOOR_OPEN, DOOR_CLOSED)]


def nearest_exit(feats: Iterable[Feature], px: int, py: int) -> Optional[Feature]:
    """Closest exit by Chebyshev distance, open doorways winning ties.

    Walking through an open doorway costs nothing; a closed door may need a
    `kick`, so prefer the free one when they are equally far.
    """
    cands = exits(feats)
    if not cands:
        return None
    return min(
        cands,
        key=lambda f: (
            max(abs(f.x - px), abs(f.y - py)),
            0 if f.label == DOOR_OPEN else 1,
            f.y,
            f.x,
        ),
    )


# --------------------------------------------------------------------------- #
# Monsters                                                                     #
# --------------------------------------------------------------------------- #

_BEARINGS = (
    ("N", 0, -1), ("NE", 1, -1), ("E", 1, 0), ("SE", 1, 1),
    ("S", 0, 1), ("SW", -1, 1), ("W", -1, 0), ("NW", -1, -1),
)

#: Monsters farther than this are listed under "distant" (NetPlay's split).
CLOSE_STEPS = 10
#: Cap on how many monsters we name, nearest first.
MONSTER_CAP = 8


def bearing(dx: int, dy: int) -> str:
    """Quantize an offset to one of 8 compass points."""
    if dx == 0 and dy == 0:
        return "here"
    import math

    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    best, best_dot = "?", -1e9
    for name, bx, by in _BEARINGS:
        bnorm = math.hypot(bx, by) or 1.0
        dot = ux * (bx / bnorm) + uy * (by / bnorm)
        if dot > best_dot:
            best_dot, best = dot, name
    return best


@dataclass(frozen=True)
class Monster:
    name: str
    glyph: str
    x: int
    y: int
    steps: int
    bearing: str
    is_pet: bool


def visible_monsters(raw_obs) -> list[Monster]:
    """Named, located, ranged monsters — what NetPlay and BALROG both emit.

    We used to hand the agent a bare `B`, which covers giant bats *and* the far
    deadlier stalker family; `a` covers a giant ant and a soldier ant. The glyph
    id already in `raw_obs.glyphs` resolves the species exactly, so there is no
    reason to make the model guess.
    """
    chars = getattr(raw_obs, "chars", None)
    glyphs = getattr(raw_obs, "glyphs", None)
    blstats = getattr(raw_obs, "blstats", None)
    if chars is None or glyphs is None or blstats is None:
        return []
    from nethack_core.glyphs import glyph_is_monster, glyph_is_pet, glyph_to_mon, monster_name

    px, py = int(blstats[0]), int(blstats[1])
    out: list[Monster] = []
    h, w = chars.shape
    for y in range(h):
        for x in range(w):
            if (x, y) == (px, py):
                continue
            c = _ch(chars, y, x)
            if not c.isalpha() and c not in ("@", "&", "'", ";", ":"):
                continue
            g = int(glyphs[y, x])
            if not glyph_is_monster(g):
                continue
            name = monster_name(glyph_to_mon(g)) or c
            dx, dy = x - px, y - py
            out.append(
                Monster(
                    name=name,
                    glyph=c,
                    x=x,
                    y=y,
                    steps=max(abs(dx), abs(dy)),
                    bearing=bearing(dx, dy),
                    is_pet=bool(glyph_is_pet(g)),
                )
            )
    out.sort(key=lambda m: (m.steps, m.y, m.x))
    return out


def format_monsters(mons: Iterable[Monster]) -> list[str]:
    out: list[str] = []
    for m in list(mons)[:MONSTER_CAP]:
        step = "step" if m.steps == 1 else "steps"
        tag = " [PET - don't attack]" if m.is_pet else ""
        out.append(f"{m.name} at ({m.x},{m.y}), {m.steps} {step} {m.bearing}{tag}")
    return out


def monsters_in_sight(raw_obs) -> list[str]:
    return format_monsters(visible_monsters(raw_obs))


def statues(raw_obs) -> list[str]:
    """Statue positions, as text.

    Statues draw the MONSTER LETTER on the map but are not monsters: they are
    excluded from `glyph_is_monster`, from the NetPlay tracker, and (until now)
    from the feature list -- so nothing anywhere told the agent they existed. It
    would attack one, get "There is no monster at (x,y)" at zero game-turn cost,
    see an unchanged observation, and repeat. Statues were present on Dlvl 1 in
    4 of 7 seeds.

    We deliberately do NOT rewrite the map: the grid is the game's own output
    and must stay faithful. This adds a line of text instead, which is the only
    honest fix available from outside the engine.
    """
    chars = getattr(raw_obs, "chars", None)
    glyphs = getattr(raw_obs, "glyphs", None)
    blstats = getattr(raw_obs, "blstats", None)
    if chars is None or glyphs is None or blstats is None:
        return []
    from nethack_core.glyphs import glyph_is_statue, glyph_to_mon, monster_name
    px, py = int(blstats[0]), int(blstats[1])
    out = []
    h, w = chars.shape
    for y in range(h):
        for x in range(w):
            g = int(glyphs[y, x])
            if not glyph_is_statue(g) or (x, y) == (px, py):
                continue
            try:
                nm = monster_name(glyph_to_mon(g)) or "creature"
            except Exception:
                nm = "creature"
            out.append(f"statue of a {nm} at ({x},{y}) [NOT a monster - do not attack]")
    return out


#: Game turns a lapsed sighting stays worth reporting. A lichen has not moved;
#: a jackal has crossed the level. Beyond this the coordinate misleads.
MONSTER_MEMORY_TURNS = 60


def remembered_monsters(state, raw_obs, game_turn, dlvl) -> list[str]:
    """Monsters seen earlier on THIS floor that are not currently visible.

    This is the "unseen" half of the seen-vs-remembered axis. An earlier version
    of this block was removed for being actively misleading, and the four
    defects that killed it are each corrected here:

      1. **Dead monsters were still listed.** It advertised a kobold 22 turns
         after the agent watched it die. Now a sighting is dropped when the kill
         log records that species dying after the sighting was taken.
      2. **It crossed floors.** On Dlvl 2 it still listed jackal/fox/newt at
         Dlvl 1 coordinates with no floor qualifier. The memory is now keyed by
         depth and cleared on descent.
      3. **Wrong clock.** Ages were counted in LLM turns while STATUS reports
         game turns -- the same word meaning two things ~15x apart. Now game
         turns throughout.
      4. **Ghost trails.** Keyed by position, one jackal walking two tiles
         became two remembered jackals. Now keyed by species, most recent
         sighting only.

    Returns [] rather than raising on any inconsistency: a memory aid must never
    be able to break a rollout.
    """
    if state is None:
        return []
    mem = state.setdefault("_mon_mem", {})
    if state.get("_mon_mem_dlvl") != dlvl:      # (2) new floor, new memory
        mem.clear()
        state["_mon_mem_dlvl"] = dlvl
    gt = int(game_turn or 0)

    live = {}
    for m in visible_monsters(raw_obs):
        if m.is_pet:
            continue
        live[m.name] = m
        mem[m.name] = (m.x, m.y, gt)           # (4) species-keyed, latest only

    try:
        from nethack_harness.prompt.corpse_age import kill_log
        kills = kill_log(state)
    except Exception:
        kills = {}

    out: list[str] = []
    for name in list(mem):
        x, y, seen = mem[name]
        if name in live:
            continue
        killed_at = kills.get((name or "").lower())
        if killed_at is not None and int(killed_at) >= int(seen):
            del mem[name]                      # (1) we killed it; it is gone
            continue
        age = gt - int(seen)                   # (3) game turns
        if age > MONSTER_MEMORY_TURNS:
            del mem[name]
            continue
        out.append(f"{name} last seen at ({x},{y}), {age} game turns ago "
                   f"- NOT visible now, may have moved")
    return out[:MONSTER_CAP]
