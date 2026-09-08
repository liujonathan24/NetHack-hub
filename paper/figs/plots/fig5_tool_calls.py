"""Which tools the model actually reaches for."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import (HUMAN, OTHER, legendsize, ticksize, ALPHA, SIZE, mono,
                   style_axes, save)
from common import figdata

NAME = "fig5_tool_calls"
PERCEPTION = {"request_map", "reveal"}


def render():
    d = figdata()
    items = list(d["tool_counts"].items())[:9][::-1]
    total = sum(d["tool_counts"].values())
    fig, ax = plt.subplots(figsize=SIZE)
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], height=.66, alpha=ALPHA,
            color=[OTHER if k in PERCEPTION else HUMAN for k, _ in items])
    for yy, (k, v) in zip(y, items):
        ax.text(v + total * .012, yy, f"{100 * v / total:.0f}\\%",
                va="center", fontsize=ticksize)
    ax.set_yticks(y)
    ax.set_yticklabels([mono(k) for k, _ in items])
    ax.set_xlim(0, max(v for _, v in items) * 1.22)
    style_axes(ax, "Calls Issued")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=OTHER, alpha=ALPHA,
                             label="perception, not progress")],
              fontsize=legendsize, loc="lower right")
    save(fig, NAME)
