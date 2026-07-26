"""Regression tests for the observation layer the agents actually read.

Every assertion here pins a defect that was verified on a committed rollout
artifact (`tools/cli_harness_eval/acceptance/task13_claude_code_seed0.turns.ndjson`)
and written up in `.superpowers/sdd/2026-07-25-cli-harness-eval/research-prompt-quality.md`.

The load-bearing one is `test_every_emitted_feature_coordinate_indexes_its_glyph`:
it drives a REAL seeded engine and asserts that for every `<name> at (x,y)` the
renderer emits, `chars[y][x]` really is that feature's glyph. `chars` is the map
frame every skill (`move_to`, `a_star`, `descend`) consumes, so that single
assertion makes the whole class of coordinate-frame bug unreintroducible.

Run with the engine on the path:
    PYTHONPATH=<NetHackHarness>:.:environments/nethack \
        pytest environments/nethack/tests/test_observation_frame.py
"""
from __future__ import annotations

import re

import pytest

from nethack_core.env import CoreObservation, NetHackCoreEnv
from nethack_core.observations import shape
from nethack_harness.prompt.rendering import format_observation_as_chat
from nethack_harness.tools.skills import (
    bootstrap_character,
    explore_and_descend,
    parse_character_from_welcome,
)

CHARACTER = "Val-hum-neu-fem"


def _core_obs(env: NetHackCoreEnv) -> CoreObservation:
    """Rebuild a CoreObservation from the env's public last_observation view."""
    return CoreObservation(
        **dict(zip(env.observation_keys, env.last_observation))
    )


def _frames(seed: int = 0, rounds: int = 5, steps: int = 60):
    """Yield (raw_obs, structured) pairs from a real seeded game.

    We advance with `explore_and_descend` because it reveals rooms, corridors,
    doors and stairs quickly — which is exactly the feature population the
    coordinate assertions need to be non-vacuous.
    """
    env = NetHackCoreEnv(task_name="NetHackChallenge-v0")
    try:
        env.seed(core=seed, disp=seed)
        obs, _ = env.reset(character=CHARACTER)
        character = bootstrap_character(env)
        yield obs, shape(obs, character)
        for _ in range(rounds):
            structured = shape(_core_obs(env), character)
            explore_and_descend(env, structured, max_floors=1, max_game_steps=steps)
            obs = _core_obs(env)
            yield obs, shape(obs, character)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Bug 1 — the coordinate frames must agree                                     #
# --------------------------------------------------------------------------- #

# label -> the glyph(s) that legitimately produce it, read off `chars`.
LABEL_GLYPHS = {
    "stairs DOWN": ">",
    "stairs UP": "<",
    "altar": "_",
    "fountain": "{",
    "throne": "\\",
    "gold": "$",
    "door (closed)": "+",
    "spellbook": "+",
    "door (open/gap)": "|-.",
    "food/corpse": "%",
    "armor": "[",
    "weapon": ")",
    "tool": "(",
    "scroll": "?",
    "potion": "!",
    "wand": "/",
    "ring": "=",
    "amulet": '"',
    "gem/rock": "*",
}

_FEATURES_LINE = re.compile(r"^=== VISIBLE FEATURES === (.*)$", re.MULTILINE)
_COORD = re.compile(r"\((\d+),\s*(\d+)\)")


def parse_rendered_features(rendered: str) -> list[tuple[str, int, int]]:
    """Pull every `<label> at (x,y)[, (x,y)...]` the renderer emitted."""
    out: list[tuple[str, int, int]] = []
    m = _FEATURES_LINE.search(rendered)
    if not m:
        return out
    for entry in m.group(1).split("; "):
        label, sep, coords = entry.partition(" at ")
        if not sep:
            continue
        for c in _COORD.finditer(coords):
            out.append((label.strip(), int(c.group(1)), int(c.group(2))))
    return out


def test_every_emitted_feature_coordinate_indexes_its_glyph():
    """For every `<name> at (x,y)` the renderer emits, chars[y][x] is that glyph.

    This is the anti-regression for the headline defect: VISIBLE FEATURES used
    the tty frame (row 0 = message line) while `Pos:` and every skill use the
    NLE map frame, so every coordinate handed to the agent was one row low.
    """
    checked = 0
    for raw, structured in _frames():
        rendered = format_observation_as_chat(
            structured, None, state={"raw_obs": raw}, compact=False
        )
        chars = raw.chars
        for label, x, y in parse_rendered_features(rendered):
            assert label in LABEL_GLYPHS, f"unlabelled feature {label!r}"
            assert 0 <= y < chars.shape[0] and 0 <= x < chars.shape[1], (
                f"{label} at ({x},{y}) is outside the map frame "
                f"{chars.shape[1]}x{chars.shape[0]}"
            )
            got = chr(int(chars[y, x]))
            assert got in LABEL_GLYPHS[label], (
                f"{label} at ({x},{y}): chars[{y}][{x}] == {got!r}, expected one "
                f"of {LABEL_GLYPHS[label]!r}"
            )
            checked += 1
    assert checked >= 10, f"only {checked} features checked — test is vacuous"


def test_status_position_indexes_the_player_glyph():
    """`Pos: (x,y)` must index the `@` in the same map frame as the features."""
    for raw, structured in _frames(rounds=3):
        px = int(structured.status["x"])
        py = int(structured.status["y"])
        assert chr(int(raw.chars[py, px])) == "@", (
            f"Pos ({px},{py}) does not index the player: "
            f"chars[{py}][{px}] == {chr(int(raw.chars[py, px]))!r}"
        )


def test_rendered_map_row_index_equals_map_y():
    """Row N of the `=== MAP ===` block must be map row N.

    BALROG strips the tty message row for exactly this reason
    (`nle/base.py:184`); we did not, so a model counting rows landed one off.
    """
    for raw, structured in _frames(rounds=3):
        rendered = format_observation_as_chat(
            structured, None, state={"raw_obs": raw}, compact=False
        )
        block = rendered.split("=== MAP ===\n", 1)[1].split("\n\n", 1)[0]
        rows = block.split("\n")
        px = int(structured.status["x"])
        py = int(structured.status["y"])
        assert len(rows) > py, "map block is shorter than the player's row"
        assert len(rows[py]) > px and rows[py][px] == "@", (
            f"map row {py} col {px} is "
            f"{rows[py][px:px+1]!r}, expected '@'\nrow={rows[py]!r}"
        )


def test_harness_never_calls_the_tty_frame_extractor_directly():
    """One source of truth: only prompt/features.py may touch the raw extractor.

    The engine's `extract_visible_features` returns TTY-frame coordinates. If a
    second caller reappears, the frames diverge again — so the import is fenced
    to the single module that converts to the map frame.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "nethack_harness"
    allowed = {root / "prompt" / "features.py"}
    offenders = [
        str(p)
        for p in root.rglob("*.py")
        if p not in allowed and "extract_visible_features" in p.read_text()
    ]
    assert not offenders, (
        "these modules bypass prompt/features.py and get tty-frame coords: "
        + ", ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# Bug 2 — object-class labels must match NetHack's glyph classes               #
# --------------------------------------------------------------------------- #

def test_object_class_glyphs_map_to_their_nethack_meaning():
    """`)` is the weapon class and `(` the tool class — they were swapped."""
    from nethack_harness.prompt.features import FEATURE_LABELS

    assert FEATURE_LABELS[")"] == "weapon"
    assert FEATURE_LABELS["("] == "tool"


def test_a_weapon_on_the_floor_is_reported_as_a_weapon():
    """End-to-end: a `)` tile renders as `weapon at (x,y)`, never `tool`."""
    seen = {"weapon": 0, "tool": 0}
    for raw, structured in _frames(rounds=5):
        rendered = format_observation_as_chat(
            structured, None, state={"raw_obs": raw}, compact=False
        )
        for label, x, y in parse_rendered_features(rendered):
            if label in seen:
                seen[label] += 1
                assert chr(int(raw.chars[y, x])) == LABEL_GLYPHS[label]
    assert sum(seen.values()) > 0, "no weapon/tool seen — test is vacuous"


# --------------------------------------------------------------------------- #
# Bug 5 — a pinned character must surface in the observation                   #
# --------------------------------------------------------------------------- #

def test_welcome_line_without_a_gender_word_still_yields_the_role():
    """NetHack omits gender for fixed-gender roles: 'a neutral human Valkyrie'.

    The welcome regex demanded male|female|neuter between alignment and race,
    so every Valkyrie rollout reported `unknown (unknown, unknown)`.
    """
    got = parse_character_from_welcome(
        "Velkommen Agent, welcome to NetHack!  You are a neutral human Valkyrie."
    )
    assert got["role"] == "valkyrie"
    assert got["race"] == "human"
    assert got["alignment"] == "neutral"


def test_pinned_character_surfaces_in_the_rendered_observation():
    raw, structured = next(_frames(rounds=0))
    rendered = format_observation_as_chat(
        structured, None, state={"raw_obs": raw}, compact=False
    )
    assert "unknown (unknown, unknown)" not in rendered
    assert "valkyrie" in rendered.lower()


# --------------------------------------------------------------------------- #
# Bug 6 — monsters need species names and distance/bearing                     #
# --------------------------------------------------------------------------- #

def test_visible_monsters_render_with_species_name_and_bearing():
    """A known monster must render as e.g. `little dog at (5,8), 1 step SE`.

    Seed 0 starts the Valkyrie next to her pet little dog (`d`), so this is a
    deterministic fixture. We previously emitted the bare glyph `d (pet)`.
    """
    raw, structured = next(_frames(rounds=0))
    rendered = format_observation_as_chat(
        structured, None, state={"raw_obs": raw}, compact=False
    )
    block = [ln for ln in rendered.splitlines() if ln.startswith("=== VISIBLE MONSTERS")]
    assert block, "no visible-monster block rendered"
    line = block[0]
    assert "little dog" in line, line
    assert re.search(r"\(\d+,\d+\)", line), f"no coordinates in {line!r}"
    assert re.search(r"\d+ steps? [NSEW]", line), f"no distance/bearing in {line!r}"


# --------------------------------------------------------------------------- #
# Bug 3 — the "only exit is a door" hint is routinely false                    #
# --------------------------------------------------------------------------- #

def _synthetic_room_with_three_exits():
    """A 21x79 `chars` map: one room, a closed door and two open doorways.

        col:    10 ......... 20
        row 5   -----------
        row 6   |.........|
        row 7   |....@....+     <- closed door, east wall
        row 8   ..........|     <- open doorway, west wall (col 10)
        row 9   |....@....|
        row 10  -----.-----     <- open gap in the south wall
    """
    import numpy as np

    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    x0, x1, y0, y1 = 10, 20, 5, 10
    for x in range(x0, x1 + 1):
        chars[y0, x] = ord("-")
        chars[y1, x] = ord("-")
    for y in range(y0 + 1, y1):
        chars[y, x0] = ord("|")
        chars[y, x1] = ord("|")
        for x in range(x0 + 1, x1):
            chars[y, x] = ord(".")
    chars[7, x1] = ord("+")     # closed door in the east wall
    chars[8, x0] = ord(".")     # open doorway in the west wall
    chars[y1, 15] = ord(".")    # gap in the south wall
    chars[7, 15] = ord("@")
    return chars


class _FakeObs:
    def __init__(self, chars):
        import numpy as np

        self.chars = chars
        self.glyphs = np.zeros_like(chars, dtype="int16")
        tty = np.full((24, 80), ord(" "), dtype=np.uint8)
        tty[1:22, :79] = chars
        self.tty_chars = tty


def test_multiple_exits_do_not_produce_the_single_exit_hint():
    """The hint said "only exit is a door at ..." while five exits were listed.

    Root cause: the door ranking used `re.search` over each feature *string*,
    which packs up to three coordinate pairs, so every door but the first was
    invisible to the ranking.
    """
    from nethack_core.observations import StructuredObservation

    chars = _synthetic_room_with_three_exits()
    raw = _FakeObs(chars)
    structured = StructuredObservation(
        map_view="",
        messages=[],
        inventory=[],
        status={"x": 15, "y": 7, "hitpoints": 16, "max_hitpoints": 16,
                "armor_class": 6, "depth": 1, "time": 1, "experience_level": 1,
                "gold": 0, "hunger_state": 1},
        character={"role": "valkyrie", "race": "human", "alignment": "neutral"},
        adjacent={},
    )
    rendered = format_observation_as_chat(
        structured, None, state={"raw_obs": raw}, compact=False,
        include_map=False,
    )
    assert "only exit" not in rendered, rendered
    feats = parse_rendered_features(rendered)
    exits = {(x, y) for label, x, y in feats if label.startswith("door")}
    assert (20, 7) in exits, f"closed east door missing from {sorted(exits)}"
    assert (10, 8) in exits, f"west doorway missing from {sorted(exits)}"
    assert (15, 10) in exits, f"south wall gap missing from {sorted(exits)}"


def test_door_hint_ranks_over_every_parsed_coordinate_not_just_the_first():
    """The nearest exit must win, even when it is not first in its label group."""
    from nethack_harness.prompt.features import nearest_exit, visible_features

    chars = _synthetic_room_with_three_exits()
    feats = visible_features(_FakeObs(chars))
    best = nearest_exit(feats, 15, 7)
    assert best is not None
    # The closed door at (20,7) is 5 tiles east; the south gap at (15,10) is 3
    # tiles south. The south gap must win.
    assert (best.x, best.y) == (15, 10), f"ranked {best} instead of the south gap"


def test_remembered_stairs_do_not_leak_across_dungeon_levels():
    """The `>` memo is per-floor; (x,y) on Dlvl 2 is a different tile than Dlvl 1.

    `state["_seen_stairs_down"]` accumulates every `>` coordinate ever seen and
    is never cleared on descent, so standing on Dlvl 2 at a coordinate that held
    the stairs on Dlvl 1 asserted "You are standing on stairs DOWN — call
    `descend` now", and `descend` then failed. Before the frame fix the memo was
    tty-frame and compared against map-frame `Pos`, so it almost never matched
    and the falsehood stayed hidden; making the frames agree makes this live.
    """
    from nethack_core.observations import StructuredObservation

    chars = _synthetic_room_with_three_exits()
    raw = _FakeObs(chars)
    state = {"raw_obs": raw, "_seen_stairs_down": set()}

    def render(depth):
        structured = StructuredObservation(
            map_view="", messages=[], inventory=[],
            status={"x": 15, "y": 7, "hitpoints": 16, "max_hitpoints": 16,
                    "armor_class": 6, "depth": depth, "time": 1,
                    "experience_level": 1, "gold": 0, "hunger_state": 1},
            character={}, adjacent={},
        )
        return format_observation_as_chat(
            structured, None, state=state, compact=False, include_map=False,
        )

    # Pretend we saw `>` here on Dlvl 1.
    chars[7, 15] = ord(">")
    render(1)
    assert state["_seen_stairs_down"], "stairs were not memoised at all"
    # Now we are on Dlvl 2, same (x,y), and the tile is plain floor.
    chars[7, 15] = ord("@")
    out = render(2)
    assert "standing on stairs DOWN" not in out, out


@pytest.mark.parametrize("seed", [0, 1])
def test_hint_does_not_repeat_verbatim_forever(seed):
    """A hint repeated identically turn after turn is evidence it is not working.

    Artifact turns 13-20 carried one byte-identical hint eight turns running.
    """
    state = {"raw_obs": None, "_seen_stairs_down": set()}
    hints = []
    for raw, structured in _frames(seed=seed, rounds=4, steps=20):
        state["raw_obs"] = raw
        rendered = format_observation_as_chat(
            structured, None, state=state, compact=False
        )
        for line in rendered.splitlines():
            if line.startswith("=== HINT ==="):
                hints.append(line)
    for i in range(2, len(hints)):
        assert not (hints[i] == hints[i - 1] == hints[i - 2]), (
            f"identical hint emitted 3 turns running: {hints[i]!r}"
        )
