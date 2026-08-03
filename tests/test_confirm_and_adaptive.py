"""Pins for the relocated [ynq] confirm policy and the adaptive stall timeout.

The confirm policy's first implementation was structurally unreachable (it
lived in the dismissal loop's else-branch, but any "[ynq]" message is parsed
into yn_prompt and takes the yn branch) -- caught in pre-launch review, not by
any test. These are the tests that would have caught it, applied to the pure
helpers the fixed version routes through.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "environments" / "nethack"))
sys.path.insert(0, str(REPO / "tools"))

from nethack import _confirm_yes_for  # noqa: E402
import stall_watchdog as sw  # noqa: E402


def test_loot_confirm_is_answered_yes():
    assert _confirm_yes_for("np_loot", ["There is a chest here, loot it? [ynq] (q)"])


def test_attack_confirms_are_never_auto_answered():
    # A `y` here murders the Minetown watch captain and ends the run.
    assert not _confirm_yes_for("np_move_to", ["Really attack the watch captain? [yn] (n)"])
    assert not _confirm_yes_for("np_loot", ["Really attack the watch captain? [yn] (n)"])


def test_unrelated_skill_does_not_claim_the_prompt():
    assert not _confirm_yes_for("np_move_to", ["There is a chest here, loot it? [ynq] (q)"])
    assert not _confirm_yes_for(None, ["There is a chest here, loot it? [ynq] (q)"])
    assert not _confirm_yes_for("np_loot", [])


def test_newest_yn_message_wins():
    msgs = ["There is a chest here, loot it? [ynq] (q)",
            "Really attack the shopkeeper? [yn] (n)"]
    assert not _confirm_yes_for("np_loot", msgs)  # newest is the attack prompt


def test_adaptive_timeout_small_samples_protect_not_bias():
    # 2 gaps: p95 index would pick the MIN (30) -> no protection. Must use MAX.
    assert sw.adaptive_timeout(300, [30.0, 400.0], factor=4, ceiling=1800) == 1600


def test_adaptive_timeout_bounds():
    assert sw.adaptive_timeout(300, [], factor=4, ceiling=1800) == 300
    assert sw.adaptive_timeout(300, [10.0] * 24, factor=4, ceiling=1800) == 300  # fast model: base
    assert sw.adaptive_timeout(300, [500.0] * 24, factor=4, ceiling=1800) == 1800  # ceiling


def test_recent_call_gaps_survives_garbage(tmp_path):
    p = tmp_path / "0_1_2.ndjson"
    p.write_text('{"t_wall": 100.0}\n{"t_wall": "oops"}\n{"t_wall": 130.0}\nnot json\n')
    assert sw.recent_call_gaps(str(p)) == [30.0]
