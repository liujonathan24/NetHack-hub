"""E14/E15 crisis directive: pure heuristics + directive-line formatting.

The two naive triggers for the `crisis_directive="on"` arm (see
NetHackVerifiersEnv.__init__ in nethack.py, where the stateful plumbing --
edge-trigger re-arm, per-level ack-set, trace stamping -- lives):

  * HP-CRISIS -- HP strictly below a third of max with a hostile in melee
                 range;
  * PACING    -- XL below the human-winner ARRIVAL norm on first sight of a
                 dungeon level (norms: human_norms.norm_xl_for_arriving).

E15 P2 r1 revisions (fix1):
  * The HP-crisis line is retreat-only. The r1 wording recommended prayer
    (re-fired straight into the once-per-~1000-turn cooldown death) and
    engraving Elbereth (impossible: the '-' fingers key is gated off in this
    tool surface; two seeds died typing a weapon engraving letter-by-letter).
    4 of 5 r1 deaths happened during or just after directive-compliant
    remedies. Retreat is the one recommendation this action surface can
    actually execute.
  * Pacing compares against the ARRIVAL norm of the level just entered, as
    this docstring always claimed; r1 used norm_xl_for_leaving (the arrival
    norm one depth ahead), which made the directive fire on 100% of arrivals
    at depth >= 3 and pushed agents into lethal low-XL grinding.

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
            f"Retreat NOW -- move away from the monster toward explored "
            f"territory or upstairs; do not melee.]")


def pacing_lagging(xl, dlvl) -> bool:
    """True iff XL lags the human-winner norm for ARRIVING at dlvl."""
    from nethack_harness.prompt.human_norms import norm_xl_for_arriving
    try:
        return int(xl) < int(norm_xl_for_arriving(int(dlvl)))
    except (TypeError, ValueError):
        return False


def pacing_line(xl, dlvl, norm) -> str:
    """The bracketed PACING directive, exactly as the model sees it."""
    return (f"[pacing directive: you are XL {xl} on Dlvl {dlvl}; typical "
            f"successful human runs arrive here at XL {norm}. Level here "
            f"or retreat to a higher dungeon level before descending.]")
