"""Knob validation: each knob against the mechanism it is supposed to move."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import (BAD, HUMAN, AGENT, GREY, legendsize, ticksize, LW, MS,
                   ALPHA, SIZE, style_axes, save)

NAME = "figA4_knob_validation"
ROWS = [("locked\\_door $1.0 \\to 0.0$\nkicks per run", 18, 3, BAD),
        ("room\\_density $1.0 \\to 0.025$\nnav no-op rate (\\%)", 22.0, 7.1, HUMAN),
        ("reveal\\_map $0.0 \\to 1.0$\nmap present (\\% turns)", 12.4, 100.0, AGENT)]


def render():
    fig, ax = plt.subplots(figsize=SIZE)
    y = np.arange(len(ROWS))[::-1]
    for yy, (lab, a, b, col) in zip(y, ROWS):
        m = max(a, b)
        ax.plot([100 * a / m, 100 * b / m], [yy, yy], color=col, lw=LW, alpha=ALPHA)
        ax.plot(100 * a / m, yy, marker="o", markersize=MS, mfc="white",
                mec=col, mew=3.0)
        ax.plot(100 * b / m, yy, marker="o", markersize=MS, color=col)
        ax.text(104, yy, f"{a:g} $\\to$ {b:g}", fontsize=ticksize, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.set_xlim(0, 150)
    ax.set_ylim(-1.1, len(ROWS) - 0.5)
    ax.set_xticks([])
    style_axes(ax)
    ax.grid(visible=False)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, mew=3.0,
               markersize=MS, label="off"),
        Line2D([], [], marker="o", ls="", color=GREY, markersize=MS, label="on")],
        fontsize=legendsize, ncol=2, loc="lower right")
    save(fig, NAME)
