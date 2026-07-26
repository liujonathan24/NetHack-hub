"""Bounds on the closed-loop skills.

`explore_and_descend` shipped with a 400 in-game-step budget per call and the
system prompt told the agent to call it every turn; the committed artifact shows
three calls returning `descended 0 floor(s)` and the hero Hungry on Dlvl 3 at
in-game turn 1611. `move_to` toward a tile it can never reach kept probing one
step into a wall (`It's a wall.` on three consecutive turns of the smoke2
prime_agent rollout) instead of saying so.

Both are budget bugs: a call must be bounded, and a hopeless call must fail fast.
"""
from __future__ import annotations

import time

from nethack_core.env import CoreObservation, NetHackCoreEnv
from nethack_core.observations import shape
from nethack_harness.prompt import features
from nethack_harness.tools import skills
from nethack_harness.tools.skills import (
    bootstrap_character,
    explore_and_descend,
    move_to,
)

CHARACTER = "Val-hum-neu-fem"


def _env(seed: int = 0):
    env = NetHackCoreEnv(task_name="NetHackChallenge-v0")
    env.seed(core=seed, disp=seed)
    obs, _ = env.reset(character=CHARACTER)
    return env, obs, bootstrap_character(env)


def _core(env) -> CoreObservation:
    return CoreObservation(**dict(zip(env.observation_keys, env.last_observation)))


# --------------------------------------------------------------------------- #
# move_to must fail fast on a target it can never reach                        #
# --------------------------------------------------------------------------- #

def _open_door_tiles(core):
    """Tiles NetHack draws with a WALL glyph that are actually open doors.

    NetHack renders an open door as the wall glyph rotated 90 degrees, so `-`
    and `|` in `chars` are ambiguous. The GLYPH layer is not: `_glyph_clean_chars`
    maps an open door to `.` and a real wall to `|`.
    """
    import numpy as np

    clean = skills._glyph_clean_chars(core.glyphs)
    mask = (clean == ord(".")) & (
        (core.chars == ord("-")) | (core.chars == ord("|"))
    )
    ys, xs = np.nonzero(mask)
    return {(int(x), int(y)) for y, x in zip(ys, xs)}


def test_move_to_a_wall_fails_immediately_instead_of_probing_into_it():
    env, obs, character = _env()
    try:
        core = _core(env)
        chars = core.chars
        doors = _open_door_tiles(core)
        # A REAL wall -- explicitly NOT an open door, which NetHack draws with
        # the same glyph. Taking the first `-`/`|` on the map (as this test
        # originally did) can land on an open door, which would enshrine the
        # false-refusal bug this suite is supposed to catch.
        wall = None
        for y in range(chars.shape[0]):
            for x in range(chars.shape[1]):
                if chr(int(chars[y, x])) in "-|" and (x, y) not in doors:
                    wall = (x, y)
                    break
            if wall:
                break
        assert wall is not None, "no real wall glyph on the starting map"
        t0 = time.monotonic()
        res = move_to(env, shape(_core(env), character), x=wall[0], y=wall[1])
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"move_to took {elapsed:.1f}s on an unreachable tile"
        assert res.actions == [], (
            f"move_to stepped toward a wall: {res.actions} / {res.feedback}"
        )
        assert res.interrupted, f"failure not reported: {res.feedback}"
        assert "wall" in res.feedback.lower() or "not walkable" in res.feedback.lower()
    finally:
        env.close()


def test_every_recommended_exit_is_a_legal_move_to_target():
    """The prompt and the tool must not contradict each other.

    `nearest_exit` ranks over `door (open/gap)` and `door (closed)` and the HINT
    tells the agent to `move_to` the winner. NetHack draws an OPEN door as `-`
    or `|`, and `move_to`'s fast unreachable-target exit refuses exactly those
    glyphs with "there is no route to it and never will be" -- a false statement
    about a tile the same observation just recommended. It fired on two of three
    arms of the committed smoke3 run (control turns 6 and 13, prime_agent turn
    9) and helped wedge the control arm in its starting room until it died.

    So: every tile `nearest_exit` could name must be a tile `move_to` accepts.
    """
    checked = 0
    wall_drawn_doors = 0
    for seed in (0, 1, 2):
        env, obs, character = _env(seed)
        try:
            for _ in range(3):
                structured = shape(_core(env), character)
                explore_and_descend(env, structured, max_floors=1, max_game_steps=60)
                core = _core(env)
                feats = features.visible_features(core)
                doors = _open_door_tiles(core)
                for f in features.exits(feats):
                    res = move_to(
                        env, shape(core, character), x=f.x, y=f.y
                    )
                    if "diverted to" in res.feedback:
                        continue  # `>` short-circuit; never reaches the check
                    assert "never will be" not in res.feedback, (
                        f"seed {seed}: hint can recommend {f.label} at "
                        f"({f.x},{f.y}) but move_to refuses it: {res.feedback}"
                    )
                    checked += 1
                    if (f.x, f.y) in doors:
                        wall_drawn_doors += 1
                        # The specific failure: an open door drawn as a wall
                        # glyph must be routable, not merely un-refused.
                        assert res.actions, (
                            f"seed {seed}: no path to open door ({f.x},{f.y}) "
                            f"drawn as {chr(int(core.chars[f.y, f.x]))!r}: "
                            f"{res.feedback}"
                        )
        finally:
            env.close()
    assert checked >= 10, f"only {checked} exits checked — test is vacuous"
    assert wall_drawn_doors >= 1, (
        "no open door drawn with a wall glyph was seen — the regression this "
        "test exists for would not be exercised"
    )


def test_move_to_out_of_bounds_fails_immediately():
    env, obs, character = _env()
    try:
        t0 = time.monotonic()
        res = move_to(env, shape(_core(env), character), x=500, y=500)
        assert time.monotonic() - t0 < 1.0
        assert res.actions == []
        assert res.interrupted
    finally:
        env.close()


def test_move_to_stops_churning_after_repeated_no_progress_on_one_target():
    """Repeated best-effort calls to the same hopeless target must give up.

    Best-effort probing is useful once (it reveals map); repeating it forever is
    the `It's a wall.` loop from the smoke2 rollout.
    """
    env, obs, character = _env()
    try:
        chars = _core(env).chars
        # A tile that is walkable-looking but sealed off: unexplored rock far
        # from the starting room. Best-effort will probe toward it.
        target = (78, 20)
        results = [
            move_to(env, shape(_core(env), character), x=target[0], y=target[1])
            for _ in range(6)
        ]
        assert any(r.interrupted and not r.actions for r in results), (
            "move_to never gave up: "
            + " | ".join(r.feedback for r in results)
        )
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# explore_and_descend must be bounded in game steps AND wall clock             #
# --------------------------------------------------------------------------- #

def test_explore_and_descend_default_step_budget_is_bounded():
    """400 in-game steps per call is a starvation engine (research D6).

    NetPlay caps every skill at 100 in-game turns and interrupts on level
    change / teleport / new glyph / low health; ours only interrupts on
    HP/hunger, so its budget must be at least as tight.
    """
    schema = skills.registry.all_schemas()["explore_and_descend"]
    default = schema["parameters"]["max_game_steps"]["default"]
    assert default <= 150, f"default max_game_steps={default} is unbounded in practice"

    import inspect

    sig = inspect.signature(explore_and_descend)
    assert sig.parameters["max_game_steps"].default == default


def test_explore_and_descend_budget_is_a_cap_not_just_a_default():
    """A default the model can override to 100000 is not a bound.

    Observed live in smoke3: the agent called it with `max_game_steps=200`,
    stepping straight past the default. NetPlay's 100-turn cap is not
    agent-overridable either.
    """
    env, obs, character = _env()
    try:
        res = explore_and_descend(
            env, shape(_core(env), character),
            max_floors=1, max_game_steps=100000, max_seconds=30.0,
        )
        turn = int(_core(env).blstats[20])
        assert turn <= skills._EAD_MAX_GAME_STEPS + 50, (
            f"burned {turn} in-game turns in one call: {res.feedback}"
        )
    finally:
        env.close()


def test_explore_and_descend_honours_a_wall_clock_deadline():
    env, obs, character = _env()
    try:
        t0 = time.monotonic()
        res = explore_and_descend(
            env, shape(_core(env), character),
            max_floors=1, max_game_steps=100000, max_seconds=0.25,
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"deadline ignored: {elapsed:.1f}s"
        assert "wall-clock time budget" in res.feedback, (
            f"deadline exit not reported: {res.feedback}"
        )
    finally:
        env.close()


def test_explore_and_descend_schema_publishes_the_deadline():
    schema = skills.registry.all_schemas()["explore_and_descend"]
    assert "max_seconds" in schema["parameters"]
