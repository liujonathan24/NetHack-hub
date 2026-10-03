"""Where a game goes, and where the agents stop going.

The (depth, experience) plane with the *paths* drawn rather than the endpoints:
every winning NAO game is a staircase climbing from the bottom-left corner to the
top-right, and the agents' staircases are the short scribbles that never leave
that corner. The heavy dashed line is what a winning pace looks like -- the
median experience level a winner held the first time it stood on each depth.

Axes are fig4's: max dungeon level 1..50 (the percentile table's ceiling) against
max experience level 1..30. Human trajectories come from the v2 NLD-NAO decode
(824 ascensions of 44,008 games), reconstructed on the raw axes by inverting the
stored BALROG percentile pair -- the depth and experience tables are disjoint
apart from 0.0, so the inversion is exact rather than approximate.

Set SHOW_AUTOASCEND to add the local AutoAscend HEAD build as a fifth series.
It is off by default: that build stalls on Dlvl 1 in 48% of its 504 episodes,
so its plotted spread describes our build, not the published bot.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import (HUMAN_TOP, AGENT, BAD, blue, GREY, LW, ALPHA,
                   legendsize, style_axes, save)
from common import arm, balrog_table
from paths import ALL_GAMES

NAME = "fig2_trajectories"

# Proposals land beside the paper's figures without replacing them; point this
# at style.save's default (images/) to adopt the figure.
OUTDIR = "/root/overleaf/images/proposed"

BALROG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "balrog_leaderboard.json")
AUTOASCEND_JSON = ("/tmp/claude-0/-root-overleaf/"
                   "a060b531-2a62-4781-865b-9cc9e074663c/scratchpad/"
                   "autoascend_head.json")
SHOW_AUTOASCEND = False

DMAX, XMAX = 50, 30
N_WINNERS = 60          # sampled staircases; the full 824 would be a purple wash
MIN_N = 20              # depths with fewer winners than this leave the norm undrawn
SEED = 43


# ---- raw-axis reconstruction ------------------------------------------------
def _inverses():
    """percentile -> level, one dict per BALROG axis (injective after rounding)."""
    dl, xp = balrog_table()
    return ({round(v, 3): k for k, v in dl.items()},
            {round(v, 3): k for k, v in xp.items()})


def winner_staircases():
    """(depth, experience) staircase per NAO game that ascended, plus the count."""
    D_INV, X_INV = _inverses()
    out = []
    for g in json.load(open(ALL_GAMES)):
        if g["turns"] < 100 or not str(g.get("death", "")).startswith("ascended"):
            continue
        d = x = 1
        ds, xs = [], []
        for pt in g["c"]:
            for v in (round(pt[1], 3), round(pt[2], 3)):
                if v == 0.0:
                    continue
                if v in D_INV:
                    d = D_INV[v]
                elif v in X_INV:
                    x = X_INV[v]
            ds.append(d)
            xs.append(x)
        if ds:
            out.append((np.array(ds), np.array(xs)))
    return out


def winners_norm(stairs):
    """Median XL at the first record standing on each depth, where n >= MIN_N."""
    ds, ys = [], []
    for d in range(1, DMAX + 1):
        v = [x[np.argmax(dd >= d)] for dd, x in stairs if dd.max() >= d]
        if len(v) >= MIN_N:
            ds.append(d)
            ys.append(float(np.median(v)))
    return np.array(ds), np.array(ys)


def our_staircases():
    out = []
    for r in arm("base")["rollouts"]:
        pts = sorted((t["gturn"], t["dlvl"], t["xp"])
                     for t in r["turns"] if "gturn" in t)
        if pts:
            out.append((np.maximum.accumulate([p[1] for p in pts]),
                        np.maximum.accumulate([p[2] for p in pts])))
    return out


def balrog_staircases():
    out = []
    for e in json.load(open(BALROG_JSON))["episodes"]:
        s = sorted(e["steps"])
        out.append((np.maximum.accumulate([r[1] for r in s]),
                    np.maximum.accumulate([r[2] for r in s])))
    return out


def autoascend_staircases():
    eps = json.load(open(AUTOASCEND_JSON))["episodes"]
    out = []
    for e in eps:
        c = e.get("curve") or []
        if len(c) < 2 or e["max_depth"] <= 1:      # never left Dlvl 1: build stall
            continue
        out.append((np.maximum.accumulate([r[1] for r in c]),
                    np.maximum.accumulate([r[2] for r in c])))
    return out, len(eps)


# ---- figure -----------------------------------------------------------------
def render():
    rng = np.random.default_rng(SEED)
    wins = winner_staircases()
    nx, ny = winners_norm(wins)
    ours = our_staircases()
    blr = balrog_staircases()

    fig, ax = plt.subplots(figsize=(9, 5.4))

    # 0 -- the corner every agent stays inside, drawn from the agents' own extent
    ad = max(max(d[-1] for d, _ in ours), max(d[-1] for d, _ in blr))
    axl = max(max(x[-1] for _, x in ours), max(x[-1] for _, x in blr))
    ax.add_patch(plt.Rectangle((0.5, 0.5), ad + .5, axl + .5, facecolor=AGENT,
                               alpha=.07, lw=0, zorder=0))

    def stair(d, x, jitter=0.0, **kw):
        j = rng.uniform(-jitter, jitter, 2) if jitter else (0.0, 0.0)
        ax.step(np.minimum(d, DMAX) + j[0], np.minimum(x, XMAX) + j[1],
                where="post", **kw)
        return j

    # 1 -- the winning journeys, as a texture rather than as individual curves
    for i in rng.choice(len(wins), min(N_WINNERS, len(wins)), replace=False):
        stair(*wins[i], jitter=.12, color=HUMAN_TOP, lw=0.9, alpha=.32, zorder=1)

    # 2 -- the agents, each ending in the death that stopped it
    for fam, col, z in ((blr, BAD, 3), (ours, AGENT, 4)):
        for d, x in fam:
            j = stair(d, x, jitter=.14, color=col, lw=1.7, alpha=.85, zorder=z)
            ax.plot(min(d[-1], DMAX) + j[0], min(x[-1], XMAX) + j[1], "x",
                    color=col, ms=7, mew=1.8, zorder=z + 1)

    if SHOW_AUTOASCEND:
        aa, n_aa = autoascend_staircases()
        for i in rng.choice(len(aa), min(N_WINNERS, len(aa)), replace=False):
            stair(*aa[i], jitter=.12, color=blue, lw=1.0, alpha=.45, zorder=2)

    # 3 -- the pace that wins
    ax.step(nx, ny, where="mid", color=GREY, lw=LW, ls="--", alpha=ALPHA, zorder=6)

    handles = [
        Line2D([], [], color=HUMAN_TOP, lw=1.6, alpha=.7,
               label=rf"NAO ascensions ($n{{=}}{N_WINNERS}$ of {len(wins)})"),
        Line2D([], [], color=GREY, lw=LW, ls="--", label="winning pace (median)"),
        Line2D([], [], color=AGENT, lw=1.9, marker="x", ms=7, mew=1.8,
               label=rf"our rollouts ($n{{=}}{len(ours)}$)"),
        Line2D([], [], color=BAD, lw=1.9, marker="x", ms=7, mew=1.8,
               label=rf"BALROG agent ($n{{=}}{len(blr)}$)"),
    ]
    if SHOW_AUTOASCEND:
        handles.append(Line2D([], [], color=blue, lw=1.6,
                              label=rf"AutoAscend ($n{{=}}{len(aa)}$ of {n_aa})"))

    ax.set_xlim(0.5, DMAX + 0.5)
    ax.set_ylim(0.5, XMAX + 0.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    style_axes(ax, "Max Dungeon Level", "Max Experience Level")
    ax.grid(False)
    ax.legend(handles=handles, fontsize=legendsize - 5, loc="upper left",
              framealpha=.95, borderpad=.55, labelspacing=.42,
              handlelength=1.9, handletextpad=.7)

    save(fig, NAME, outdir=OUTDIR)
