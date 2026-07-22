"""nethack_harness.curriculum

The former 13-tier named curriculum ladder (curriculum.py + milestones.py +
subgoals.py) was removed in the NetHack-Hub extraction; the env now runs the
standard full ascension game (see nethack.GameSpec / FULL_GAME_SPEC).

This package is intentionally kept as an empty seam for the deferred six-floor
"primitives" curriculum, which depends on NetHack-fork C hooks
(nle_hero_on_stair / nle_grant_invocation_kit / nle_invocation_pos /
nle_seat_on_invocation_square) living on the unmerged fork branch
feature/invocation-ritual. See hub_notes.md.
"""
