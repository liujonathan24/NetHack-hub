"""fig4_depth_xp, with the Go-Explore arms overlaid at a 30-attempt cut.

Same plane, same contours, same human norm as `fig4_depth_xp` -- this module
imports that figure's `draw()` rather than restating it, so the background can
only ever be identical to the one already in the paper.

WHAT IS ADDED. One marker per Go-Explore ATTEMPT, which is the like-for-like
unit against the base arm's rollouts: both are single lives that end when the
character dies. Every arm is cut at its first 30 attempts so the overlay is
read at equal budget rather than at whatever each run happened to reach; the
seed-1 arms ran to 61-95 and those later attempts are deliberately not here.

WHAT IS EXCLUDED, and why it matters for this plot specifically. Attempts
censored as `env_crash` are dropped (23 of 230). An engine SIGSEGV empties
traces.jsonl, so those attempts' (Dlvl, XL) come from a progress-file estimate
rather than an engine reading -- fine for a spend total, not for a point whose
whole job is to sit at exact coordinates on this plane.

The points are per-attempt maxima, so each marker is a state the run actually
occupied. That is why the overlay is NOT the same as the run's headline
BALROG: the headline is the best single attempt, which is one marker here, not
the corner of the cloud.
"""
import json
import matplotlib.pyplot as plt
from style import AGENT, legendsize, MS, ALPHA, style_axes, save
from plots.fig4_depth_xp import draw as draw_plane

NAME = "fig4_depth_xp_ge"
GE = "/root/overleaf/figs/e16_ge30.json"

#: ONE colour for the whole overlay, and it is AGENT -- the same green the base
#: rollouts already use, because these are also our rollouts, just orchestrated.
#: An earlier draft coloured the points by seed, which silently reassigned three
#: role colours (green already means "our rollouts", red means "deaths", blue
#: means "NAO population"), and style.py exists so a colour means the same thing
#: in every figure. Base vs Go-Explore is carried by MARKER instead: filled
#: triangle for a single uninterrupted life, open circle for an orchestrated
#: attempt. Per-seed detail belongs in the results table, not in five new hues.


def render():
    d = json.load(open(GE))
    fig, ax = plt.subplots(figsize=(7, 4.8))

    # The published plane, verbatim. legend=False so the base arm's entries can
    # be folded into one legend with the overlay instead of two stacked boxes.
    draw_plane(ax, legend=False)

    xs = [p[1] for p in d["points"]]
    ys = [p[2] for p in d["points"]]
    nseed = len({p[0] for p in d["points"]})
    ax.plot(xs, ys, marker="o", markersize=MS - 4, linestyle="none",
            markerfacecolor="none", markeredgewidth=1.4,
            color=AGENT, alpha=ALPHA, zorder=6,
            label=(rf"Go-Explore attempts, {nseed} seeds "
                   rf"($n{{=}}{len(xs)}$, first 30 each)"))

    ax.set_xlim(1, 50)
    ax.set_ylim(1, 30)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    style_axes(ax, "Max Dungeon Level", "Max Experience Level")
    ax.grid(False)
    ax.legend(fontsize=legendsize - 6, loc="upper left", framealpha=.9)
    save(fig, NAME)


if __name__ == "__main__":
    render()
