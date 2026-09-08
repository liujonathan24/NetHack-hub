"""P1: one plane, one panel. Base, corrections (as a band), Explore, humans.

Every line is the median experience level a series holds on first arrival at
each dungeon level, ending at the deepest level half its runs reached -- the
same statistic as the human winning pace, so all lines are directly comparable.

The six correction arms are one light-green band: at each depth, the range of
the six arms' lines. Same hue family as the base line because it is the same
model on a tweaked harness. The `_revive` variant pulls forced revival out of
the band and draws it as its own line, since it is the one correction that
moves right as well as up.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import AGENT, HUMAN_TOP, blue, GREY, LW, legendsize, style_axes, save
from common import winners_by_depth
from plots.fig4_plane_story import pace_of, min_bands, PLANE, PROBES, ARMS, CONTINUAL
from plots.fig2_trajectories_ge import ge_rollouts, GE30, GE100

NAME = "p1_plane"
OUTDIR = "/root/overleaf/images/proposed"
XLIM, YLIM = (0.5, 30.5), (0.5, 18.5)
BANDS = [2, 5, 10, 20]
GOEX = blue
CORR = "#77B25D"        # base green, used at low alpha for the band
REVIVE = "#B07A2A"


def envelope(lines):
    """(depths, lo, hi) across several pace lines, where at least one exists."""
    by = {}
    for ds, ys in lines:
        for d, y in zip(ds, ys):
            by.setdefault(d, []).append(y)
    ds = sorted(by)
    return ds, [min(by[d]) for d in ds], [max(by[d]) for d in ds]


def render(revive_line=False):
    doc = json.load(open(PLANE))
    probes = json.load(open(PROBES))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    min_bands(ax, levels=BANDS, alpha=0.045, fs=legendsize - 7)

    # humans
    by, n_win = winners_by_depth()
    ds = [d for d in sorted(by) if d <= XLIM[1] and len(by[d]) >= 10]
    q = lambda p: np.array([np.percentile(by[d], p) for d in ds])
    ax.fill_between(ds, q(25), q(75), step="post", color=HUMAN_TOP, alpha=.18,
                    lw=0, zorder=2)
    ax.step(ds, q(50), where="post", color=GREY, lw=LW - 0.5, ls="--", zorder=5)

    # corrections band
    arm_keys = [k for k, *_ in ARMS] + [CONTINUAL[0]]
    lines = {}
    for k in arm_keys:
        rolls = (probes["arms"][k]["rollouts"] if k in probes["arms"]
                 else doc["models"][k]["rollouts"])
        lines[k] = pace_of(rolls)
    band_keys = [k for k in arm_keys if not (revive_line and k == "revive")]
    ed, lo, hi = envelope([lines[k] for k in band_keys])
    ax.fill_between(ed, lo, hi, step="post", color=CORR, alpha=.28, lw=0,
                    zorder=3)

    handles = []
    def line(rolls_or_line, col, ls, label, lw=2.6, z=4):
        ds_, ys_ = rolls_or_line if isinstance(rolls_or_line, tuple) else pace_of(rolls_or_line)
        ax.step(ds_, ys_, where="post", color=col, lw=lw, ls=ls, zorder=z,
                solid_capstyle="butt")
        handles.append(Line2D([], [], color=col, lw=lw, ls=ls, label=label))

    line(doc["models"]["glm52"]["rollouts"], AGENT, "-",
         r"GLM-5.2 single life ($n{=}15$)")
    handles.append(Patch(color=CORR, alpha=.28,
                         label=rf"harness corrections, {len(band_keys)} arms (range)"))
    if revive_line:
        line(lines["revive"], REVIVE, "-", "forced revive on death")
    line(ge_rollouts(GE30), GOEX, "-", r"Go-Explore, 30 attempts ($n{=}7$)")
    line(ge_rollouts(GE100), GOEX, "--", r"Go-Explore, 100 attempts ($n{=}3$)")
    handles += [
        Line2D([], [], color=GREY, lw=LW - 0.5, ls="--",
               label=rf"NAO ascensions, median ($n{{=}}{n_win}$)"),
        Patch(color=HUMAN_TOP, alpha=.25, label="NAO ascensions, interquartile"),
    ]

    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([1, 5, 10, 15])
    style_axes(ax, "Max Dungeon Level", "Max Experience Level")
    ax.grid(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.11),
               ncol=2, fontsize=legendsize - 6, frameon=False,
               handlelength=2.2, handletextpad=.7, columnspacing=2.0)
    fig.subplots_adjust(bottom=0.3)
    save(fig, NAME + ("_revive" if revive_line else ""), outdir=OUTDIR)


if __name__ == "__main__":
    render(); render(True)
