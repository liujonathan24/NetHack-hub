"""The headline figure. Left: BALROG-min against game turns, one median curve
per series. Right: the distribution of each series' final BALROG-min, one box
per series, on the same y-scale.

WHAT IS ON IT. Nine agent series and two human references.
  * The BALROG leaderboard's best NetHack entry, Gemini 3 Pro,
    regraded from its published per-episode steps onto the same table (n=5).
  * Four frontier models on the Prime-Agent + NetPlay Core harness (n=15 each).
  * GLM-5.2 on that harness, and three variants of it: the best hand-written
    harness correction that does not roll back the game (the ASCII map on
    every turn; forced revive scores higher but is a rollback), the continual
    + wiki harness, and Prime-Agent-Explore. Named "GLM-5.2 (method)" so the reader sees that
    every one of them is the same model.

CURVES are median running-maximum BALROG-min over the series' rollouts, dead
runs carried forward at their final value, so every curve is monotone and its
last value is the median of the box beside it. Solid while any run is alive,
faint once the last one has died.

BOXES show quartiles, median and full range of the per-rollout final score,
with the rollouts themselves drawn over the box. Zeros -- runs that never left
XL 1 -- sit on the log-axis floor at FLOOR; a point on the floor is a zero.

COLOUR. The four GLM-5.2 series are the subject and take saturated colours;
the four other frontier models are context and take muted ones; the
leaderboard entry is red. Boxes and curves share the palette, so the box
labels double as the legend for the agent series.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from style import (AGENT, blue, GREY, LIGHT, legendsize, LW, ALPHA, labelsize,
                   ticksize, style_axes, save)
from common import compare
from plots.fig3_progression_ge import FLOOR
from plots.fig3_progression_pair import _carry, _line, XT, XTL

NAME = "fig_main_min"
OUTDIR = "/root/overleaf/images/proposed"
SRC = "/root/overleaf/figs/e_progression_main.json"

#: Drawing order is also box order, left to right: them, the field, us.
ORDER = ["human", "gemini3pro", "sol", "luna", "gemini37", "qwen38",
         "glm52", "ascii", "continual", "explore"]
COL = {
    "human": "black",
    "gemini3pro": "#E0645C",
    "sol": "#8FA3BF", "luna": "#A990C4", "gemini37": "#C9B27A", "qwen38": "#B88E8E",
    "glm52": AGENT, "ascii": "#E39A2C", "continual": "#1F7A72", "explore": blue,
}
SUBJECT = {"human", "glm52", "ascii", "continual", "explore", "gemini3pro"}
SHORT = {"human": "Human\npopulation", "gemini3pro": "Gemini 3 Pro\n(BALROG)", "sol": "GPT-5.6\nSol",
         "luna": "GPT-5.6\nLuna", "gemini37": "Gemini-3.7\nFlash",
         "qwen38": "Qwen-3.8\nMax", "glm52": "GLM-5.2",
         "ascii": "GLM-5.2\n(ASCII map)", "continual": "GLM-5.2\n(continual)",
         "explore": "GLM-5.2\n(explore)"}


def matrix(rolls, g):
    cv = [_carry(r["pts"], r["end"], g, 1) for r in rolls]
    return np.array([c[0] for c in cv]), np.array([c[1] for c in cv])


def box(ax, vals, pos, color, rng, strong):
    """Quartiles, median, and whiskers over the full range -- except for the
    human population, whose 44k games are summarised at the 5th/95th
    percentile, since its single best game (80%) would set the panel's scale
    on its own. Every series small enough to show as dots is shown as dots."""
    v = np.maximum(np.asarray(vals, float), FLOOR)
    ax.boxplot([v], positions=[pos], widths=0.6, patch_artist=True,
               showfliers=False, whis=(0, 100) if len(v) <= 50 else (5, 95),
               zorder=3,
               medianprops=dict(color=color, lw=2.6 if strong else 2.0),
               boxprops=dict(facecolor=to_rgba(color, .25 if strong else .18),
                             edgecolor=color, lw=1.5 if strong else 1.1),
               whiskerprops=dict(color=color, lw=1.5 if strong else 1.1),
               capprops=dict(color=color, lw=1.5 if strong else 1.1))
    if len(v) <= 50:          # the human population is 44k games: box only
        ax.scatter(pos + rng.uniform(-.18, .18, len(v)), v, s=16, color=color,
                   alpha=.9, lw=0, zorder=4)


#: The right panel in the "key results" variant: the leaderboard entry and
#: the GLM-5.2 family only. The left panel is unchanged, so the four other
#: models stay visible as curves and in the legend.
KEY = ["human", "gemini3pro", "glm52", "ascii", "continual", "explore"]


def render(boxes=None):
    boxes = boxes or ORDER
    d = compare()
    g = np.array(d["grid"], float)
    nao, wins = d["pop"]["nao"], d["pop"]["nao"]["wins"]
    doc = json.load(open(SRC))
    series = doc["series"]
    finals_of = lambda k: (doc["human_population"]["finals"] if k == "human"
                           else [r["pts"][-1][1] for r in series[k]["rollouts"]])
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(14.5 if len(boxes) > 6 else 13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5 if len(boxes) > 6 else 0.95],
                          wspace=0.05,
                          left=0.06, right=0.995, top=0.94, bottom=0.30)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1], sharey=ax)

    # ---- left: progression --------------------------------------------------
    for src, col, ls, lab, n in [
            (nao, "black", "-", "Human population", nao["stats"]["n"]),
            (wins, GREY, "--", "Human successes", wins["n"])]:
        ax.plot(g, np.maximum(np.array(src["series"]["min"]["50"], float), FLOOR),
                color=col, lw=LW, ls=ls, alpha=1.0,
                label=rf"{lab} ($n{{=}}{n:,}$)".replace(",", "{,}"))
    for key in ORDER:
        if key == "human":
            continue                          # drawn above, from compare()
        M, A = matrix(series[key]["rollouts"], g)
        strong = key in SUBJECT
        _line(ax, g, M, A, COL[key],
              rf"{series[key]['label']} ($n{{=}}{M.shape[0]}$)",
              lw=LW if strong else LW - 1.2)

    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 90)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xscale("log")
    ax.set_xlim(100, 50000)
    ax.set_xticks(XT)
    ax.set_xticklabels(XTL)
    style_axes(ax, "In-Game Turns", r"BAL$_{\min}$ (\%)")
    ax.set_title("(a) Progression over time", loc="left", pad=8, fontsize=labelsize - 3)

    # ---- right: final-score distributions ------------------------------------
    for i, key in enumerate(boxes):
        box(bx, finals_of(key), i, COL[key], rng, key in SUBJECT)
    bx.set_xlim(-0.6, len(boxes) - 0.4)
    bx.set_xticks(range(len(boxes)))
    bx.set_xticklabels([SHORT[k] for k in boxes],
                       fontsize=ticksize - (5.5 if len(boxes) > 6 else 3.5),
                       linespacing=1.15)
    for lab, key in zip(bx.get_xticklabels(), boxes):
        lab.set_color(COL[key] if key in SUBJECT else GREY)
    bx.tick_params(axis="y", which="both", left=False, labelleft=False)
    bx.tick_params(axis="x", length=0, pad=6)
    bx.grid(True, axis="y", linestyle="--")
    bx.grid(False, axis="x")
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.set_title(r"(b) Distribution of BAL$_{\min}$ scores", loc="left", pad=8,
                 fontsize=labelsize - 3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.12),
               ncol=4, fontsize=legendsize - 8, frameon=False,
               columnspacing=1.4, handlelength=2.2)
    save(fig, NAME + ("" if boxes is ORDER else "_key"), outdir=OUTDIR)


if __name__ == "__main__":
    render()
    render(KEY)
