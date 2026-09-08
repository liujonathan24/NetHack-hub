"""P2: BALROG-max against BALROG-min, one dot per rollout.

The metric argument with no geometry to explain: every single life in the paper
(five models, six correction arms) lies along the x-axis -- max spans 0 to 35
while min stays under 3 -- because these runs go deep without levelling and
max follows whichever axis leads. Go-Explore lineages leave the axis.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import AGENT, blue, GREY, LIGHT, legendsize, style_axes, save
from common import balrog_table, _pct
from plots.fig4_plane_story import PLANE, PROBES, ARMS, CONTINUAL
from plots.fig2_trajectories_ge import ge_rollouts, GE30, GE100

NAME = "p2_metrics"
OUTDIR = "/root/overleaf/images/proposed"


def render():
    doc = json.load(open(PLANE))
    probes = json.load(open(PROBES))
    dl, xp = balrog_table()

    single = []
    for m in doc["models"].values():
        single += [(r["bal"], r["bal_min"]) for r in m["rollouts"]]
    for k, *_ in ARMS:
        single += [(r["bal"], r["bal_min"]) for r in probes["arms"][k]["rollouts"]]
    ge = []
    for path in (GE30, GE100):
        for r in ge_rollouts(path):
            d, x = np.maximum.accumulate(np.asarray(r["pts"]), axis=0)[-1]
            a, b = _pct(dl, int(d)), _pct(xp, int(x))
            ge.append((max(a, b), min(a, b)))
    single, ge = np.array(single), np.array(ge)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 60], [0, 60], color=LIGHT, lw=1, zorder=1)
    rng = np.random.default_rng(3)
    j = rng.uniform(-.35, .35, single.shape)          # dodge the integer grid
    ax.scatter(single[:, 0] + j[:, 0], single[:, 1] + j[:, 1], s=22,
               color=AGENT, alpha=.55, lw=0, zorder=3)
    ax.scatter(ge[:, 0], ge[:, 1], s=46, color=blue, lw=0, zorder=4)
    ax.set_xlim(-1, 56)
    ax.set_ylim(-1, 56)
    ax.set_aspect("equal")
    style_axes(ax, r"BALROG-max (\%)", r"BALROG-min (\%)")
    ax.grid(False)
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", color=AGENT, ms=6,
               label=rf"single life, all models and corrections ($n{{=}}{len(single)}$)"),
        Line2D([], [], ls="none", marker="o", color=blue, ms=7,
               label=rf"Go-Explore lineage ($n{{=}}{len(ge)}$)"),
        Line2D([], [], color=LIGHT, lw=1, label="min $=$ max"),
    ], loc="upper left", fontsize=legendsize - 8, frameon=False)
    save(fig, NAME, outdir=OUTDIR)


if __name__ == "__main__":
    render()
