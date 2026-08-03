"""Adaptive objective hint, computed from BALROG's own scoring table.

BALROG progression is ``max(value(Dlvl:D), value(Xp:X))`` -- a MAX, not a sum.
That single fact drives everything here, and it is counter-intuitive enough to
state plainly: **effort spent on the lower-scoring axis earns nothing until it
overtakes the higher one.** An agent at Dlvl 6 / XP 4 scores 3.543 (the depth
value); grinding to XP 5 moves the xp term 2.416 -> 2.911 and the score not at
all.

That argues for pushing the axis already AHEAD (``mode="lead"``, exploit). But
the axes are not independent: experience is survival currency, and a hero who
dies stops accumulating on BOTH axes forever. So pushing the axis BEHIND
(``mode="lag"``, explore) may raise the eventual max even though it never raises
the current one. Which wins is an empirical question, so both modes ship and are
run head-to-head rather than argued about.

Two further facts are handled explicitly:

* ``Xp:n > Dlvl:n`` at every n (Xp:7 = 5.076 vs Dlvl:7 = 4.846), so experience
  is worth marginally more *per level*. It is also far slower to gain early, so
  depth is what binds in practice for our runs.
* The leading axis can jam -- no reachable down-stair, or a floor too dangerous
  to cross. Pushing a jammed axis forever is worse than overtaking on the other
  one. Hence the stall fallback: if the binding axis has not moved in
  ``STALL_TURNS``, recommend the other axis, because overtaking is then the only
  route to a higher max.

The block is emitted as plain text at the top of the observation so it costs
~60 tokens/turn and is trivially ablatable (drop the variant, keep everything
else identical).
"""
from __future__ import annotations

from nethack_harness.prompt.balrog import _achievements

#: Turns the binding axis may stall before we recommend the other one.
STALL_TURNS = 150


def _val(prefix: str, n: int) -> float:
    """Achievement value for `Dlvl:n` / `Xp:n`, falling back to nearest lower."""
    ach = _achievements()
    n = int(max(1, n))
    while n >= 1:
        v = ach.get(f"{prefix}{n}")
        if v is not None:
            return v * 100.0
        n -= 1
    return 0.0


def objective_hint(structured, state, *, mode: str = "lead") -> str:
    """One short block naming which axis to pursue, and why.

    ``mode="lead"`` (EXPLOIT) pushes the axis that is currently AHEAD -- the one
    binding the max(), where a checkpoint converts straight into score.

    ``mode="lag"`` (EXPLORE) pushes the axis that is BEHIND. This scores nothing
    immediately, and is the interesting arm precisely because of that: the two
    axes are not independent. Experience is survival currency -- an
    under-levelled hero dies on deeper floors, and death ends accumulation
    permanently -- so raising the trailing axis can lift the eventual max even
    though it never lifts the current one. Whether that long-run effect beats
    the immediate conversion of `lead` is an empirical question, which is the
    whole point of running both.

    Both modes keep the stall fallback: an axis that has not moved in
    ``STALL_TURNS`` is jammed, and hammering it further is worse than switching.
    """
    status = getattr(structured, "status", None) or {}
    depth = int(status.get("depth") or 1)
    xp = int(status.get("experience_level") or 1)

    # Stall bookkeeping. Keyed in `state` so it is per-rollout, not global.
    turn = int(state.get("turn_count") or 0)
    if state.get("_hint_last_depth") != depth:
        state["_hint_last_depth"] = depth
        state["_hint_depth_turn"] = turn
    if state.get("_hint_last_xp") != xp:
        state["_hint_last_xp"] = xp
        state["_hint_xp_turn"] = turn
    depth_stall = turn - int(state.get("_hint_depth_turn") or 0)
    xp_stall = turn - int(state.get("_hint_xp_turn") or 0)

    d_val, x_val = _val("Dlvl:", depth), _val("Xp:", xp)
    cur = max(d_val, x_val)
    d_next = _val("Dlvl:", depth + 1) - cur   # gain if we descend one level
    x_next = _val("Xp:", xp + 1) - cur        # gain if we level up once

    depth_binds = d_val >= x_val
    # Which axis does this mode want? lead -> the one ahead; lag -> the one behind.
    want_depth = depth_binds if mode == "lead" else not depth_binds
    if mode == "lead":
        why = "depth is ahead" if want_depth else "experience is ahead"
    else:
        why = "depth trails" if want_depth else "experience trails"
    # Stall override, in both modes: a jammed axis cannot be pushed further, so
    # switch regardless of which one the mode nominally prefers.
    if want_depth and depth_stall >= STALL_TURNS:
        want_depth, why = False, f"depth stuck {depth_stall} turns"
    elif (not want_depth) and xp_stall >= STALL_TURNS:
        want_depth, why = True, f"experience stuck {xp_stall} turns"
    focus = "DESCENT" if want_depth else "COMBAT"

    if focus == "DESCENT":
        advice = ("Find the down-staircase `>` and descend. Explore unvisited "
                  "corridors and rooms to locate it; stand ON the `>` then go down.")
    else:
        advice = ("Gain experience: attack weak adjacent monsters by moving into "
                  "them. Avoid fights you can lose -- dying ends the run.")

    return (
        "=== OBJECTIVE ===\n"
        f"Score = max(depth_value, experience_value). Now: Dlvl {depth} = "
        f"{d_val:.2f} | XP {xp} = {x_val:.2f} -> score {cur:.2f}.\n"
        f"Next: Dlvl {depth + 1} would add {max(0.0, d_next):.2f}; "
        f"XP {xp + 1} would add {max(0.0, x_next):.2f}.\n"
        f"FOCUS: {focus} ({why}). {advice}"
    )
