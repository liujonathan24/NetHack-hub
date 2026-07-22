"""CurriculumPrimitivesEnv: the six-floor honest-navigation curriculum.

Pure-engine tests (no LLM). They exercise the engine's public invocation hooks
(grant_invocation_kit / invocation_pos / seat_on_invocation_square) surfaced on
EngineEnv by engine PR #37 (feat/4b-six-floor-hooks), plus the on-stair boundary
jump (goto_abs + raw hero_on_stair). Seed 19 (the env default) is used because
its Gehennom reaches absolute depth 50, so the deep segment (48/49/50) and the
Invocation level (num_dunlevs-1) are entirely real.

Run with a nethack-core that carries the invocation hooks on PYTHONPATH and the
matching prebuilt libnethack.so via NLE_LIB_PATH, e.g.::

    PYTHONPATH=<engine feat/4b clone> \\
    NLE_LIB_PATH=.../third_party/NetHack/src/build/libnethack.so \\
    pytest tests/test_curriculum_primitives_env.py
"""
import pathlib
import sys

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[2] / "environments" / "nethack"),
)

import numpy as np
import pytest

from nethack_core.actions import MiscDirection
from nethack_core.observations import BLSTATS_IDX
from nethack_harness.curriculum import CurriculumPrimitivesEnv

DOWN = int(MiscDirection.DOWN)


def _b(obs, name):
    return int(obs.blstats[BLSTATS_IDX[name]])


def _inv_text(obs):
    return "\n".join(
        s.tobytes().decode(errors="replace").rstrip("\x00")
        for s in np.array(obs.inv_strs)
    )


@pytest.fixture
def env():
    e = CurriculumPrimitivesEnv()
    yield e
    e.close()


def _branch_dnums(e):
    """(dod_dnum, geh_dnum, geh_depth_start, geh_num_dunlevs) from the live table."""
    table = e._engine.dungeon_table()
    dod = next(d for d in table if "Dungeons of Doom" in d["name"])
    geh = next(d for d in table if "Gehennom" in d["name"])
    return dod["dnum"], geh["dnum"], geh["depth_start"], geh["num_dunlevs"]


def test_starts_valkyrie_boosted_on_floor_one(env):
    obs, _ = env.reset()
    inv = _inv_text(obs)
    assert "long sword" in inv  # female Valkyrie signature kit
    assert env._tune.get("reveal_map") == 1.0  # "lights on"
    # Starts on Dungeons of Doom level 1 (curriculum floor 1).
    assert _b(obs, "depth") == 1
    assert env.curriculum_floor(obs) == 1
    # Survivability/attack boost so the curriculum tests navigation, not RNG.
    assert _b(obs, "hitpoints") >= 250
    assert _b(obs, "strength") >= 118  # NetHack-encoded 25 (>= 18/**)
    assert _b(obs, "experience_level") >= 10


def test_boundary_jump_grants_kit_and_upgrade(env):
    """A real '>' on DoD level 3's downstair jumps to the Gehennom deep segment,
    granting the invocation kit and applying the stats-only upgrade."""
    env.reset()
    dod_dnum, _, _, _ = _branch_dnums(env)
    # Navigate onto DoD level 3's downstair (incremental goto so the deferred
    # cross-branch goto inside the jump step stays in sync), then take the real
    # '>' the curriculum intercepts only on a genuine stair.
    env._engine.goto_abs(dod_dnum, 2)
    env._engine.goto_abs(dod_dnum, 3, seat=True)
    assert env._engine.engine.hero_on_stair() == 1  # standing on a down stair

    obs, _reward, _done, _trunc, info = env.step(DOWN)
    assert info["curriculum"] == "jump_down"
    assert _b(obs, "depth") == 48                 # deep-segment low (Gehennom)
    assert _b(obs, "dungeon_number") == 1          # Gehennom branch
    assert env.curriculum_floor(obs) == 4          # first deep floor
    assert "upgrade" in info                       # stats-only deep-jump upgrade
    # The invocation ritual kit was injected at the boundary.
    inv = _inv_text(obs)
    assert "Candelabrum" in inv
    assert "Bell" in inv
    assert "Book of the Dead" in inv


def test_invocation_level_pos_seat_and_progression(env):
    """Grant the kit, navigate to Gehennom's Invocation level (num_dunlevs-1),
    and verify invocation_pos/seat_on_invocation_square + measured progression."""
    obs0, _ = env.reset()
    start_floor = env.curriculum_floor(obs0)
    assert start_floor == 1

    _, geh_dnum, _geh_start, geh_num = _branch_dnums(env)
    # grant_invocation_kit is a ctrl-R (no deferred goto), so a single goto to
    # the Invocation level afterwards stays in sync.
    env._engine.grant_invocation_kit()
    inv_dlevel = geh_num - 1  # Invocation level = num_dunlevs - 1
    obs = env._engine.goto_abs(geh_dnum, inv_dlevel)
    env._last_observation = obs

    # We are on the Invocation level (the maze above Moloch's Sanctum).
    assert env.on_invocation_level(obs)

    # invocation_pos surfaces the vibrating square's coordinates.
    square = env.invocation_square(obs)
    assert square is not None
    x, y = square
    assert isinstance(x, int) and isinstance(y, int)
    assert x > 0 and y > 0

    # seat_on_invocation_square stages the hero at the ritual site.
    seated = env._engine.seat_on_invocation_square(adjacent=True)
    env._last_observation = seated
    # The square is unchanged and still reported after seating.
    assert env.invocation_square(seated) == square

    # Progression is measured by curriculum_floor: the Invocation level is a
    # deep floor (>= 4) and strictly deeper than the start.
    inv_floor = env.curriculum_floor(obs)
    assert inv_floor >= 4
    assert inv_floor > start_floor


def test_off_invocation_level_reports_no_square(env):
    """invocation_square is game-knowledge, not a global locator: it returns
    None anywhere but the Invocation level."""
    obs, _ = env.reset()  # DoD level 1
    assert not env.on_invocation_level(obs)
    assert env.invocation_square(obs) is None
