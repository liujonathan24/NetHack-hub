"""Pins for the fix1 rollback semantics: out-of-range n CLAMPS to the
retained history instead of erroring, and the feedback quotes the restored
game turn.

E15 P3 r1 motivated both: seed 3 issued rollback(20) at 1 HP (meaning ~20
GAME turns, not tool calls), got a range error, and wasted its crisis call;
after a deep rollback truncated the ring, a later rollback(6) failed the same
way at 4 HP. A best-effort restore is strictly better than a refusal in a
crisis, and agents demonstrably reason in game turns, so the feedback names
the restored clock.
"""

import pathlib
import sys

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.tools.skills import (  # noqa: E402
    push_rollback_snapshot,
    rollback,
)


class _StubEngine:
    def __init__(self):
        self.snapshots = 0
        self.restored = []
        self.freed = []

    def snapshot(self):
        self.snapshots += 1
        return f"h{self.snapshots}"

    def restore(self, handle):
        self.restored.append(handle)

    def free_snapshot(self, handle):
        self.freed.append(handle)


class _StubEnv:
    def __init__(self):
        self._engine = _StubEngine()


def _fill(env, n):
    for turn in range(1, n + 1):
        push_rollback_snapshot(env, turn, game_time=turn * 10)


def test_rollback_out_of_range_clamps_instead_of_erroring():
    env = _StubEnv()
    _fill(env, 5)  # ring: turns 1..5; deepest legal n is 4
    res = rollback(env, obs=None, n=20)
    assert res.actions == [27], "clamped rollback must still restore (ESC)"
    assert env._engine.restored == ["h1"], "clamp lands on the oldest retained state"
    assert "rolled back 4" in res.feedback
    assert "asked for 20" in res.feedback


def test_rollback_feedback_quotes_game_turn():
    env = _StubEnv()
    _fill(env, 5)
    res = rollback(env, obs=None, n=2)
    # n=2 from ring[1..5] lands on the turn-3 snapshot (game time 30).
    assert env._engine.restored == ["h3"]
    assert "game turn 30" in res.feedback


def test_rollback_in_range_unchanged():
    env = _StubEnv()
    _fill(env, 5)
    res = rollback(env, obs=None, n=1)
    assert env._engine.restored == ["h4"]
    assert "asked for" not in res.feedback  # no clamp note when in range


def test_rollback_single_snapshot_still_refuses():
    # With only the current state retained there is nothing to restore INTO;
    # that refusal (not a clamp to n=0) must survive.
    env = _StubEnv()
    _fill(env, 1)
    res = rollback(env, obs=None, n=1)
    assert res.actions == []
    assert "no earlier turn" in res.feedback
