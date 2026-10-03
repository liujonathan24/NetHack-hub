"""Progress and survival against game turns, on one shared axis.

Merges what were two figures (progression and survival). They share an x-axis and
tell one story: our rollouts track the human population through the early game and
then stop, because the games end. The progress metric is BALROG-min -- the weaker
of the two BALROG axes (depth, experience) -- which rewards balanced progression
rather than a bare dive. The human comparison is the full NAO population and the
subset of games that actually won.

The keystroke-level BALROG-style harness arm (encoding__balrog_parity__bp_hybrid)
is configured in e14/results/configs but its rollout traces live on the cluster
scratch volume and are not present locally, so that series is omitted here.
"""
import json
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from style import (HUMAN, HUMAN_TOP, AGENT, BAD, labelsize, legendsize, ticksize,
                   LW, ALPHA, style_axes, save)
from common import compare, e14_on_grid_min, balrog_table, _pct

NAME = "fig3_progression"
XT = [10, 100, 1000, 10000, 50000]
XTL = ["10", "100", "1k", "10k", "50k"]

# Cached BALROG-leaderboard trajectories (parsed once from
# github.com/balrog-ai/experiments); the figure regenerates offline from this.
BALROG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "balrog_leaderboard.json")

# Log-y floor for the top panel: BALROG-min is exactly 0 until Dlvl 2 or the
# relevant XL is reached, and 0 cannot render on a log axis, so every plotted
# series is clipped up to this smallest displayed value.
FLOOR = 0.5


def balrog_on_grid_min(g):
    """Leaderboard episodes' running BALROG-min on grid g; NaN once each ends.

    Same construction as e14_on_grid_min: per-axis running max first, then the
    min across the depth and experience axes, stepped onto the shared grid.
    """
    dl, xp = balrog_table()
    d = json.load(open(BALROG_JSON))
    rows = []
    for e in d["episodes"]:
        pts = sorted((s[0], _pct(dl, s[1]), _pct(xp, s[2])) for s in e["steps"])
        gt = np.array([p[0] for p in pts], float)
        bv = np.minimum(np.maximum.accumulate([p[1] for p in pts]),
                        np.maximum.accumulate([p[2] for p in pts]))
        idx = np.searchsorted(gt, g, side="right") - 1
        vals = np.where(idx >= 0, bv[np.clip(idx, 0, len(bv) - 1)], 0.0)
        rows.append(np.where(g <= gt[-1], vals, np.nan))
    return np.array(rows), d["model"]


def render():
    d = compare()
    g = np.array(d["grid"], float)
    nao = d["pop"]["nao"]
    wins = nao["wins"]

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(12, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.10})

    # ---- top: BALROG-min progression (log y; zeros clipped to FLOOR) ------
    def clip(a):
        return np.maximum(np.array(a, float), FLOOR)

    for src, col, lab, n in [(nao["series"]["min"], HUMAN, "NAO population",
                              nao["stats"]["n"]),
                             (wins["series"]["min"], HUMAN_TOP, "NAO successes",
                              wins["n"])]:
        ax.fill_between(g, clip(src["25"]), clip(src["75"]), color=col,
                        alpha=.20, lw=0)
        ax.plot(g, clip(src["50"]), color=col, lw=LW, alpha=ALPHA,
                label=rf"{lab} ($n{{=}}{n:,}$)".replace(",", "{,}"))

    M = e14_on_grid_min(g)
    for row in M:
        ax.plot(g, clip(row), color=AGENT, lw=1.0, alpha=.28)
    alive = np.sum(~np.isnan(M), axis=0)
    keep = alive >= 5
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(M, axis=0)
    ax.plot(g[keep], clip(med[keep]), color=AGENT, lw=LW + 1.2,
            label=rf"ours ($n{{=}}{M.shape[0]}$)")

    # BALROG-leaderboard naive agent, regraded with the same table.
    B, bmodel = balrog_on_grid_min(g)
    for row in B:
        ax.plot(g, clip(row), color=BAD, lw=1.0, alpha=.28)
    balive = np.sum(~np.isnan(B), axis=0)
    bkeep = balive >= 3
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bmed = np.nanmedian(B, axis=0)
    ax.plot(g[bkeep], clip(bmed[bkeep]), color=BAD, lw=LW + 1.2,
            label=rf"BALROG ({bmodel}, $n{{=}}{B.shape[0]}$)")

    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 90)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axes(ax, None, "BALROG-min (\\%)")
    ax.legend(fontsize=legendsize, loc="upper left")

    # ---- bottom: fraction still alive -------------------------------------
    for src, col in [(nao["series"]["alive"], HUMAN),
                     (wins["series"]["alive"], HUMAN_TOP)]:
        a = np.array(src, float)
        bx.plot(g, 100 * a / a[0], color=col, lw=LW, alpha=ALPHA)
    bx.plot(g, 100 * alive / M.shape[0], color=AGENT, lw=LW + 1.2)
    bx.plot(g, 100 * balive / B.shape[0], color=BAD, lw=LW + 1.2)

    bx.set_ylim(0, 105)
    bx.set_yticks([0, 50, 100])
    bx.set_xscale("log")
    bx.set_xlim(10, 50000)
    bx.set_xticks(XT)
    bx.set_xticklabels(XTL)
    style_axes(bx, "Game Turns", "Alive (\\%)")

    save(fig, NAME)
