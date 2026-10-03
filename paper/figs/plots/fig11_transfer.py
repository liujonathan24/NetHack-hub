"""What the store actually transfers.

Held-out experience level and held-out BALROG-min over ten iterations, both runs.
Iteration 0 is the empty store. Depth is flat (see the companion figure), but both
runs lift the experience axis and hold it, and the wiki run closes above the base
on BALROG-min --- the metric that penalises single-axis progress.
"""
import numpy as np
import matplotlib.pyplot as plt
from style import (HUMAN, purple, GREY, legendsize, ticksize, LW, MS, ALPHA,
                   style_axes, save)
from common import CURVE, BASE_XL, BASE_BMIN

NAME = "fig11_transfer"
RUNS = [("mem", "Memory-only", HUMAN), ("wiki", "$+$ wiki", purple)]


def _r(vals, base):
    return np.arange(0, len(vals) + 1), np.array([base] + list(vals), float)


def render():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             gridspec_kw={"wspace": .22})
    panels = [("test_xl", BASE_XL, "Held-out Experience Level", (0, 5)),
              ("test_bmin", BASE_BMIN, "Held-out BALROG-min (\\%)", (0, 3.0))]

    for ax, (series, base, ylab, ylim) in zip(axes, panels):
        ax.axhline(base, color=GREY, lw=1.8, ls="--", zorder=2)
        ax.text(-0.2, base, "base", fontsize=ticksize, color=GREY,
                ha="left", va="bottom")
        for (key, name, col), dy in zip(RUNS, (-16, 10)):
            x, y = _r(CURVE[key][series], base)
            ax.plot(x, y, color=col, lw=LW + .6, marker="^", markersize=MS - 1,
                    alpha=ALPHA, label=name)
            ax.plot(x[-1], y[-1], marker="o", markersize=MS + 2, color=col,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=6)
            ax.annotate(f"{y[-1]:.2f}", xy=(x[-1], y[-1]),
                        xytext=(-8, dy), textcoords="offset points",
                        fontsize=ticksize, color=col, ha="right")
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlim(-.4, 10.6)
        ax.set_ylim(*ylim)
        style_axes(ax, "Iteration", ylab)
    axes[0].legend(fontsize=legendsize, loc="lower right")
    save(fig, NAME)
