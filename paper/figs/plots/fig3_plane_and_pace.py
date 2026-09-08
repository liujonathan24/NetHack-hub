"""Where the runs end, and how fast they got there -- one figure, two views.

Merges the two result figures the paper used to carry. Left is the (depth,
experience) plane, drawn by fig4_depth_xp.draw so the two stay identical: the
faint L-shaped contours are the BALROG-min field, the dashed grey line is the
human experience-for-depth norm, and each run is one point. Right is the same
runs against game turns, on BALROG-min, with the NAO population and the NAO
games that won as bands. The two panels answer "how far" and "how fast" with
one legend and one colour vocabulary.

The alive-fraction strip that used to sit under the progression panel is gone:
it duplicated what the truncated agent curves already say -- every one of our
rollouts ends in death before turn 2k -- and it cost a third of the figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from style import (HUMAN, HUMAN_TOP, AGENT, BAD, GREY, LW, MS, ALPHA,
                   legendsize, style_axes, save)
from common import compare, e14_on_grid_min
from plots.fig4_depth_xp import draw as draw_plane
from plots.fig3_progression import balrog_on_grid_min, FLOOR, XT, XTL

NAME = "fig3_plane_and_pace"

# Proposals land beside the paper's figures without replacing them; point this
# at style.save's default (images/) to adopt the figure.
OUTDIR = "/root/overleaf/images/proposed"


def _median(M, keep_min):
    """Column median of a run family, masked where too few runs are still alive."""
    alive = np.sum(~np.isnan(M), axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(M, axis=0)
    return med, alive >= keep_min


def draw_pace(ax):
    """BALROG-min against game turns, log-log; the top panel of the old fig3."""
    d = compare()
    g = np.array(d["grid"], float)
    nao = d["pop"]["nao"]
    wins = nao["wins"]

    def clip(a):
        return np.maximum(np.array(a, float), FLOOR)

    for src, col in ((nao["series"]["min"], HUMAN), (wins["series"]["min"], HUMAN_TOP)):
        ax.fill_between(g, clip(src["25"]), clip(src["75"]), color=col,
                        alpha=.20, lw=0)
        ax.plot(g, clip(src["50"]), color=col, lw=LW, alpha=ALPHA)

    M = e14_on_grid_min(g)
    for row in M:
        ax.plot(g, clip(row), color=AGENT, lw=1.0, alpha=.28)
    med, keep = _median(M, 5)
    ax.plot(g[keep], clip(med[keep]), color=AGENT, lw=LW + 1.2)

    B, bmodel = balrog_on_grid_min(g)
    for row in B:
        ax.plot(g, clip(row), color=BAD, lw=1.0, alpha=.28)
    bmed, bkeep = _median(B, 3)
    ax.plot(g[bkeep], clip(bmed[bkeep]), color=BAD, lw=LW + 1.2)

    ax.set_xscale("log")
    ax.set_xlim(10, 50000)
    ax.set_xticks(XT)
    ax.set_xticklabels(XTL)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 90)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axes(ax, "Game Turns", "BALROG-min (\\%)")
    return nao["stats"]["n"], wins["n"], M.shape[0], B.shape[0]


def render():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.5))
    draw_plane(ax, legend=False)
    n_pop, n_win, n_ours, n_blr = draw_pace(bx)

    # One legend for both panels: line handles carry the plane's marker, so a
    # single row means the same thing on the left and on the right.
    handles = [
        Line2D([], [], color=HUMAN, lw=LW, label=_n("NAO population", n_pop)),
        Line2D([], [], color=HUMAN_TOP, lw=LW, label=_n("NAO successes", n_win)),
        Line2D([], [], color=GREY, lw=LW, ls="--", label="human norm"),
        Line2D([], [], color=AGENT, lw=LW, marker="^", ms=MS - 2,
               label=_n("ours", n_ours)),
        # no marker: the leaderboard squares are gone from the plane, so this
        # row now describes the right-hand panel's curves alone.
        Line2D([], [], color=BAD, lw=LW, label=_n("BALROG agent", n_blr)),
    ]
    ax.legend(handles=handles, fontsize=legendsize - 6, loc="upper left",
              framealpha=.92, borderpad=.5, labelspacing=.38,
              handlelength=1.8, handletextpad=.6)

    fig.subplots_adjust(wspace=0.24)
    save(fig, NAME, outdir=OUTDIR)


def _n(label, n):
    return rf"{label} ($n{{=}}{n:,}$)".replace(",", "{,}")
