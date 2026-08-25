"""E14 crisis directive: pure heuristics + directive-line formatting.

The two naive triggers for the `crisis_directive="on"` arm (see
NetHackVerifiersEnv.__init__ in nethack.py, where the stateful plumbing --
edge-trigger re-arm, per-level ack-set, trace stamping -- lives):

  * HP-CRISIS -- HP strictly below a third of max with a hostile in melee
                 range;
  * PACING    -- XL below the human-winner arrival norm on first sight of a
                 dungeon level (norms: human_norms.norm_xl_for_leaving).

Deliberately engine-free (nothing on this import path touches nethack_core)
so the decision thresholds and the exact directive text are unit-testable
without a built engine -- tests/test_crisis_directive.py, the same recipe
test_np_core_surface.py uses to pin norm_xl_for_leaving.
"""
from __future__ import annotations


def hp_crisis_active(hp, max_hp, hostile_adjacent) -> bool:
    """True iff the HP-crisis heuristic holds: HP strictly below max_hp/3
    AND a hostile in melee range. Unreadable stats never fire."""
    try:
        hp, max_hp = int(hp), int(max_hp)
    except (TypeError, ValueError):
        return False
    return bool(hostile_adjacent) and max_hp > 0 and hp * 3 < max_hp


def hp_crisis_line(hp, max_hp) -> str:
    """The bracketed HP-CRISIS directive, exactly as the model sees it."""
    return (f"[crisis directive: HP {hp}/{max_hp} with a hostile adjacent. "
            f"Retreat, pray, or engrave Elbereth NOW -- do not melee.]")


def pacing_lagging(xl, dlvl) -> bool:
    """True iff XL lags the human-winner norm for leaving dlvl."""
    from nethack_harness.prompt.human_norms import norm_xl_for_leaving
    try:
        return int(xl) < int(norm_xl_for_leaving(int(dlvl)))
    except (TypeError, ValueError):
        return False


def pacing_line(xl, dlvl, norm) -> str:
    """The bracketed PACING directive, exactly as the model sees it."""
    return (f"[pacing directive: you are XL {xl} on Dlvl {dlvl}; typical "
            f"successful human runs reach XL {norm} before leaving this "
            f"depth. Level here or retreat to a higher dungeon level "
            f"before descending.]")
