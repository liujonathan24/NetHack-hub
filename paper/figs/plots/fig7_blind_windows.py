"""Spatial content per LM turn, one strip per rollout, death marked."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from style import HUMAN, BAD, LIGHT, legendsize, SIZE_TALL, style_axes, save
from common import figdata

NAME = "fig7_blind_windows"


def render():
    rs = figdata()["rollouts"]
    fig, ax = plt.subplots(figsize=(12, 6.0))
    for i, r in enumerate(rs):
        y = len(rs) - 1 - i
        n = len(r["turns"])
        if not n:
            continue
        ax.add_patch(Rectangle((0, y - .36), n, .72, fc=LIGHT, ec="none"))
        for j, t in enumerate(r["turns"]):
            if t["has_map"]:
                ax.add_patch(Rectangle((j, y - .36), 1, .72, fc=HUMAN, ec="none"))
        ax.plot(n, y, marker="|", markersize=16, markeredgewidth=3.5, color=BAD)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([f"s{r['seed']}r{r['rep']}" for r in rs][::-1])
    ax.set_xlim(0, max(len(r["turns"]) for r in rs) + 4)
    ax.set_ylim(-.8, len(rs) - .2)
    style_axes(ax, "LM Turn Within Rollout")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([], [], color=HUMAN, lw=10, label="map present"),
        Line2D([], [], color=LIGHT, lw=10, label="no spatial content"),
        Line2D([], [], color=BAD, lw=3, marker="|", ls="", markersize=14,
               label="death")],
        fontsize=legendsize, ncol=3, loc="upper center",
        bbox_to_anchor=(.5, 1.14))
    save(fig, NAME)
