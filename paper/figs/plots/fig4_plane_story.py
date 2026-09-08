"""The (Dlvl, XL) plane, one line per series: models on the left, corrections
on the right.

ONE STATISTIC FOR EVERYTHING ON THE PLANE. Each line is the median experience
level a series held on first arrival at each dungeon level, and it stops at the
deepest level that at least half of the series' rollouts reached. The human
"winning pace" the paper already draws is exactly that statistic over the 824
NAO ascensions, so agents and humans are the same kind of line and can be read
against each other directly: at every depth, how much experience does this
player have, and how deep does it get. Nothing is drawn per rollout.

WHY NOT THE ROLLOUTS THEMSELVES. Seventy-five staircases with crosses on their
ends made the plane unreadable and said nothing a median does not: the finding
is that the lines are flat, and one flat line per series says it.

THE SHADING is the BALROG-min level set, min(depth pct, XL pct) >= L, read off
the achievement table. It is L-shaped because a run enters it only when both
axes rise, which is the reason the paper reports the min. Each band is named
once, at the right edge of its horizontal arm, where no line ever goes.

Two panels, identical grammar: (a) five models on the base harness, (b) the
base harness against the interventions and the continual arm.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import (AGENT, HUMAN, HUMAN_TOP, OTHER, BAD, GREY, LIGHT, LW,
                   legendsize, labelsize, style_axes, save)
from common import balrog_table, winners_by_depth

NAME = "fig4_plane_models_corrections"
OUTDIR = "/root/overleaf/images/proposed"

PLANE = "/root/overleaf/figs/e14_plane.json"
PROBES = "/root/overleaf/figs/e15_probes_plane.json"

#: Every single life in the paper dies inside Dlvl 15 / XL 7.
XLIM, YLIM = (0.5, 15.5), (0.5, 8.0)
MIN_BANDS = [2, 5]

MODELS = [("glm52", AGENT), ("sol", HUMAN), ("luna", HUMAN_TOP),
          ("gemini37", OTHER), ("qwen38", BAD)]

TEAL = "#1F7A72"          # continual, the same teal it has in fig_plane_models
BROWN = "#8C5E2A"
DEEP_ORANGE = "#E39A2C"   # the palette orange is too pale for a 2pt line
ARMS = [("gate", DEEP_ORANGE, "Advisory descent gate"),
        ("revive", HUMAN_TOP, "Forced revive on death"),
        ("doors", BAD, "Locked doors removed"),
        ("ascii", HUMAN, "ASCII map every turn"),
        ("json", BROWN, "JSON map every turn")]
CONTINUAL = ("continual_wiki", TEAL, "Continual $+$ wiki")


def _elbow(L):
    """(first depth, first XL) whose percentiles both reach level L."""
    dl, xp = balrog_table()
    ds = [k for k in sorted(dl) if dl[k] >= L]
    xs = [k for k in sorted(xp) if xp[k] >= L]
    return (ds[0], xs[0]) if ds and xs else None


def min_bands(ax, levels=MIN_BANDS, alpha=0.06, fs=None, label=True):
    """Shade the BALROG-min level sets as nested L-shaped quadrants, each
    named at the right edge just above its horizontal arm."""
    x1, y1 = ax.get_xlim()[1], ax.get_ylim()[1]
    for L in levels:
        e = _elbow(L)
        if e and e[0] <= x1 and e[1] < y1:
            ax.fill_between([e[0], x1], e[1], y1, color=GREY, alpha=alpha,
                            lw=0, zorder=0)
            if not label:
                continue
            ax.annotate(rf"BALROG-min $\geq$ {L}\%", xy=(x1, e[1]),
                        xytext=(-4, 3), textcoords="offset points",
                        ha="right", va="bottom", color=GREY,
                        fontsize=fs or legendsize - 8, zorder=1)


def pace_of(rolls):
    """Median XL on first arrival at each depth; stops where half have died.

    Arrival rather than departure, so experience earned after a dive is not
    backfilled onto the shallow levels. The cut at half the rollouts means the
    line ends at the series' median reach, which is the second thing the plane
    is for.
    """
    by = {}
    for r in rolls:
        a = np.asarray(r["pts"], float)
        d = np.maximum.accumulate(a[:, 0])
        x = np.maximum.accumulate(a[:, 1])
        first = {}
        for dd, xx in zip(d, x):
            first.setdefault(int(dd), xx)
        for dd, xx in first.items():
            by.setdefault(dd, []).append(xx)
    minn = max(2, -(-len(rolls) // 2))
    ds = [k for k in sorted(by) if len(by[k]) >= minn]
    ys = [float(np.median(by[k])) for k in ds]
    # hold the last value one level to the right, so a line whose final point
    # is a jump ends in a flat segment and not in a bare vertical tick
    return ds + [ds[-1] + 1], ys + [ys[-1]]


#: Vertical dodge between series, in XL. Every agent series sits on exactly
#: XL 1 for its first few levels, so without a dodge the lines lie on top of
#: one another and only the last one drawn exists. 0.07 XL is well below what
#: a reader can mistake for a value and enough to show a line is there.
DODGE = 0.07


def draw_series(ax, series):
    """series: [(label, colour, rollouts)] -> legend handles."""
    handles = []
    n = len(series)
    for i, (label, col, rolls) in enumerate(series):
        ds, xs = pace_of(rolls)
        off = (i - (n - 1) / 2) * DODGE
        ax.step(ds, np.asarray(xs) + off, where="post", color=col, lw=2.2,
                zorder=4, solid_capstyle="butt")
        handles.append(Line2D([], [], color=col, lw=2.2, label=label))
    return handles


def draw_pace(ax, dmax=60):
    """The winning pace, in the same statistic and step convention as the
    agent lines: median XL on arrival at each depth over the ascensions."""
    by, _ = winners_by_depth()
    ds = [d for d in sorted(by) if d <= dmax and len(by[d]) >= 10]
    ys = [float(np.median(by[d])) for d in ds]
    ax.step(ds + [ds[-1] + 1], ys + [ys[-1]], where="post", color=GREY,
            lw=LW - 0.5, ls="--", zorder=5)
    return Line2D([], [], color=GREY, lw=LW - 0.5, ls="--",
                  label="NAO winning pace")


def frame(ax, title, ylabel=None):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_title(title, loc="left", pad=8, fontsize=labelsize - 2)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
    style_axes(ax, "Max Dungeon Level", ylabel)
    ax.grid(False)


def legend(ax, handles, ncol):
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=ncol, fontsize=legendsize - 8, frameon=False,
              handlelength=1.8, handletextpad=.6, columnspacing=1.4,
              labelspacing=.35)


def render():
    doc = json.load(open(PLANE))
    probes = json.load(open(PROBES))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.3))

    frame(a, "(a) Models", "Max Experience Level")
    frame(b, "(b) Corrections")
    for ax in (a, b):
        min_bands(ax)
    pace_h = draw_pace(a)
    draw_pace(b)

    models = [(doc["models"][k]["label"], c, doc["models"][k]["rollouts"])
              for k, c in MODELS]
    ha = draw_series(a, models)

    base = doc["models"]["glm52"]["rollouts"]
    arms = [("Base harness", AGENT, base)]
    arms += [(l, c, probes["arms"][k]["rollouts"]) for k, c, l in ARMS]
    ck, cc, cl = CONTINUAL
    arms.append((cl, cc, doc["models"][ck]["rollouts"]))
    hb = draw_series(b, arms)

    legend(a, ha + [pace_h], ncol=2)
    legend(b, hb + [pace_h], ncol=2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.3,
                        wspace=0.18)
    save(fig, NAME, outdir=OUTDIR)


if __name__ == "__main__":
    render()
