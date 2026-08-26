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
from nethack_harness.prompt.human_norms import (  # noqa: E402
    norm_xl_for_arriving,
    norm_xl_for_leaving,
)


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
    # Fix1: retreat-only. The r1 wording recommended prayer (cooldown death)
    # and Elbereth (dust-engraving impossible: '-' key gated off).
    assert hp_crisis_line(4, 16) == (
        "[crisis directive: HP 4/16 with a hostile adjacent. "
        "Retreat NOW -- move away from the monster toward explored "
        "territory or upstairs; do not melee.]"
    )
    for banned in ("pray", "Elbereth", "engrave"):
        assert banned not in hp_crisis_line(4, 16)


# ---- PACING threshold: XL < human-winner norm for ARRIVING at this depth ---


def test_pacing_uses_arrival_norm():
    # Fix1: r1 compared against the LEAVING norm (arrival norm one depth
    # ahead), so an agent exactly on the human arrival pace was still told
    # it lagged — the directive fired on 100% of arrivals at depth >= 3.
    for d in range(1, 12):
        at_arrival_norm = norm_xl_for_arriving(d)
        assert not pacing_lagging(at_arrival_norm, d)
        if at_arrival_norm > 1:
            assert pacing_lagging(at_arrival_norm - 1, d)


def test_pacing_lags_when_below_norm():
    # norm_xl_for_arriving(4) <= norm_xl_for_leaving(4) == 3 (pinned in
    # test_np_core_surface); XL 9 is comfortably above either.
    assert not pacing_lagging(9, 4)
    assert not pacing_lagging(norm_xl_for_arriving(4), 4)  # at the norm -> no directive


def test_arrival_norm_lags_leaving_norm():
    # The arrival norm for d is the leaving norm for d-1; leaving(d) is the
    # arrival norm of d+1, so arriving(d) <= leaving(d) everywhere.
    for d in range(1, 15):
        assert norm_xl_for_arriving(d) <= norm_xl_for_leaving(d)
        assert norm_xl_for_arriving(d + 1) == norm_xl_for_leaving(d)


def test_pacing_dlvl1_at_xl1_is_on_pace():
    assert norm_xl_for_arriving(1) == 1
    assert not pacing_lagging(1, 1)


def test_pacing_never_fires_on_unreadable_stats():
    assert not pacing_lagging(None, 4)
    assert not pacing_lagging("?", 4)


def test_pacing_line_text():
    norm = norm_xl_for_arriving(6)
    assert pacing_line(2, 6, norm) == (
        f"[pacing directive: you are XL 2 on Dlvl 6; typical successful "
        f"human runs arrive here at XL {norm}. Level here "
        f"or retreat to a higher dungeon level before descending.]"
    )
    # The user-specified action sentence survives verbatim.
    assert ("Level here or retreat to a higher dungeon level before "
            "descending.]") in pacing_line(2, 6, norm)
