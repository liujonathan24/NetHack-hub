"""Pins for the E14 crisis-directive heuristics (crisis_directive="on").

Pure logic only -- the decision thresholds and the exact directive text --
imported straight from prompt/crisis_directive.py, which is engine-free by
design: nothing here needs nethack_core or a booted engine (the same recipe
test_np_core_surface.py uses to pin norm_xl_for_leaving). The stateful
plumbing (edge-trigger re-arm, per-level ack-set, trace stamping) lives in
nethack.py's _apply_tool_call_inner and is exercised by live runs, not here.
"""

import pathlib
import sys

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.prompt.crisis_directive import (  # noqa: E402
    hp_crisis_active,
    hp_crisis_line,
    pacing_lagging,
    pacing_line,
)
from nethack_harness.prompt.human_norms import norm_xl_for_leaving  # noqa: E402


# ---- HP-CRISIS threshold: HP strictly below max/3 AND a hostile adjacent ---


def test_hp_crisis_fires_below_third_with_hostile():
    assert hp_crisis_active(5, 16, True)   # 5*3 = 15 < 16
    assert hp_crisis_active(1, 4, True)


def test_hp_crisis_boundary_is_strict():
    # Exactly a third is NOT a crisis (5*3 == 15).
    assert not hp_crisis_active(5, 15, True)
    assert not hp_crisis_active(6, 16, True)


def test_hp_crisis_needs_a_hostile():
    assert not hp_crisis_active(1, 16, False)


def test_hp_crisis_never_fires_on_unreadable_stats():
    assert not hp_crisis_active(None, None, True)
    assert not hp_crisis_active("?", 16, True)
    assert not hp_crisis_active(3, 0, True)   # max_hp 0 (unreadable blstats)


def test_hp_crisis_line_text():
    assert hp_crisis_line(4, 16) == (
        "[crisis directive: HP 4/16 with a hostile adjacent. "
        "Retreat, pray, or engrave Elbereth NOW -- do not melee.]"
    )


# ---- PACING threshold: XL < human-winner norm for leaving this depth -------


def test_pacing_lags_when_below_norm():
    # norm_xl_for_leaving(4) == 3 (pinned in test_np_core_surface).
    assert pacing_lagging(2, 4)
    assert not pacing_lagging(3, 4)   # at the norm -> no directive
    assert not pacing_lagging(9, 4)


def test_pacing_dlvl1_at_xl1_is_on_pace():
    assert norm_xl_for_leaving(1) == 1
    assert not pacing_lagging(1, 1)


def test_pacing_never_fires_on_unreadable_stats():
    assert not pacing_lagging(None, 4)
    assert not pacing_lagging("?", 4)


def test_pacing_line_text():
    norm = norm_xl_for_leaving(6)
    assert norm == 5
    assert pacing_line(2, 6, norm) == (
        "[pacing directive: you are XL 2 on Dlvl 6; typical successful "
        "human runs reach XL 5 before leaving this depth. Level here "
        "or retreat to a higher dungeon level before descending.]"
    )
