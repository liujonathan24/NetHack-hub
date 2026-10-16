"""
Tests for the reveal_map knob: the hero remembers the whole level, as after
magic mapping.

With the knob on, the engine maps every square the hero cannot see into his
memory each time his view is recomputed (``reveal_map_memory`` in vision.c),
secret doors opened; the rl window port overlays the live monsters. What he has
remembered stays when the knob is turned off, as it would after magic mapping.

``step(18)`` is ctrl-R (redraw): it re-emits the observation without advancing
game time.
"""

from __future__ import annotations

from nethack_core.engine_env import EngineEnv


def _visible(env) -> int:
    return int((env.engine.chars != ord(" ")).sum())


def test_reveal_map_shows_the_level_at_once():
    env = EngineEnv()
    env.seed(42, 42)
    env.reset()
    base = _visible(env)
    env.set_tune(reveal_map=1.0)
    env.step(18)  # ctrl-R redraw: the overlay fills what the hero has not seen
    on = _visible(env)
    env.close()
    assert on > base + 50, f"reveal_map=1 should reveal much more (base={base}, on={on})"


def _wall_count(env) -> int:
    """Count wall glyphs ('-' and '|') in the emitted map obs."""
    chars = env.engine.chars
    return int(((chars == ord("-")) | (chars == ord("|"))).sum())


def test_reveal_map_reveals_walls():
    """reveal_map=1 reveals the level's walls (the bug we fixed: reveal_map is
    now the single 'show whole map incl. walls + live monsters' knob).

    The base obs after a few steps shows only the walls the hero has explored;
    reveal_map=1 + a ctrl-R redraw fills in the rest of the floor's walls, so the
    '-'/'|' count jumps substantially (observed ~32 -> ~201).
    """
    env = EngineEnv()
    env.seed(42, 42)
    env.reset()
    for _ in range(3):
        env.step(ord("l"))  # move east to explore a little
    base_walls = _wall_count(env)

    env.set_tune(reveal_map=1.0)
    env.step(18)  # ctrl-R redraw: re-emit obs with the overlay applied
    on_walls = _wall_count(env)
    env.close()

    assert on_walls > base_walls * 2, (
        f"reveal_map=1 should reveal substantially more walls "
        f"(base={base_walls}, on={on_walls})"
    )


def test_the_revealed_level_stays_in_memory():
    """Like magic mapping: once his view is recomputed under reveal_map, the hero
    remembers the level after the knob is turned off."""
    move = ord("l")
    control = EngineEnv()
    control.seed(42, 42)
    control.reset()
    control.step(move)
    control.step(move)
    control_count = _visible(control)
    control.close()

    revealed = EngineEnv()
    revealed.seed(42, 42)
    revealed.reset()
    revealed.set_tune(reveal_map=1.0)
    revealed.step(move)
    revealed.set_tune(reveal_map=0.0)
    revealed.step(move)
    revealed_count = _visible(revealed)
    revealed.close()

    assert revealed_count > control_count + 50, (revealed_count, control_count)
