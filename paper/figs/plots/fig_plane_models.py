"""Depth against experience: ten human deaths, ten ascensions, five agent models.

WHAT IS PLOTTED. The (Dlvl, XL) plane with paths, not endpoints. Three families:

  * 10 NAO games sampled from the 43,184 that did not ascend, in light grey.
  * 10 NAO ascensions sampled from 824, in dark grey dashed.
  * Five base-tier agent models in colour, one path per selected rollout.
  * GLM-5.2 with the continual harness and wiki, last three iterations, in dark
    teal. These are its HELD-OUT evaluations, which run on seeds 0-4 like every
    other series; its TRAINING rounds run on seeds 6-10 and are not comparable,
    so they are not here. Its three iterations take the place of the other
    models' three replicas, so the selection rules below apply unchanged.
    17 scored rollouts, not 15: seed 0 was retried twice in iteration 7.

Human paths are reconstructed by inverting the stored BALROG percentile pair;
the depth and experience tables are disjoint apart from 0.0, so the inversion is
exact. Agent paths come from the turn streams directly and need no inversion.

WHICH AGENT ROLLOUT TO SHOW is a real choice, so it is a parameter rather than a
decision baked in. Each model ran 5 seeds x 3 replicas, and MODE picks per
(model, seed):

  all         every replica -- 75 paths, honest about spread, hard to read
  bal_max     the replica with the highest BALROG-max (depth-driven)
  bal_median  the median replica by BALROG-max, of three
  balmin_max  the replica with the highest BALROG-min (needs both axes to rise)

`bal_max` flatters depth and `balmin_max` flatters balance; they select different
replicas often enough that the choice changes the picture, which is why all four
are rendered rather than one being declared correct.

INCOMPLETE CONTINUAL ATTEMPTS ARE EXCLUDED. Eight turn streams in the continual
heldout dirs have no scored trace record. Every one of them was cut off rather
than ending in a death, so they are abandoned attempts and not results -- but
three of them had reached Dlvl 10, 11 and 15, deeper than any scored rollout in
that arm, so the arm's plotted reach is conservative.

TEN AND TEN IS NOT A RATE. The human samples are drawn from pools of 43,184 and
824, so their 1:1 appearance here says nothing about how often either outcome
happens -- ascension is 1.4% of games. This plane answers "what does each kind
of play look like", not "how likely is it".
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import (AGENT, HUMAN, HUMAN_TOP, OTHER, BAD, GREY, LIGHT, LW,
                   legendsize, style_axes, save)

NAME = "fig_plane_models"
SRC = "/root/overleaf/figs/e14_plane.json"
OUTDIR = "/root/overleaf/images/proposed"

#: Full plane, matching fig2: ascensions reach the bottom of the dungeon, and
#: clipping them at 30 put every ascension endpoint on the same false vertical.
#: The agents are compressed into the left fifth as a result -- that IS the
#: finding, not a layout problem.
DMAX, XMAX = 50, 30
SEEDJ = 7          # jitter rng, so overlapping staircases stay distinguishable

#: The continual arm needs a sixth hue and style.py has exactly five, all
#: taken. Defined locally rather than added to the shared palette, because no
#: other figure needs it and every figure that imports style would inherit it.
#: Dark teal: separates from the five pastels by value as well as hue.
CONTINUAL = "#1F7A72"

#: model key -> colour, matching fig3/fig4 so a model reads the same everywhere
COL = {"glm52": AGENT, "sol": HUMAN, "luna": HUMAN_TOP,
       "gemini37": OTHER, "qwen38": BAD, "continual_wiki": CONTINUAL}
ORDER = ["glm52", "sol", "luna", "gemini37", "qwen38", "continual_wiki"]

MODES = {
    "all": "every replica",
    "bal_max": "best BALROG-max per seed",
    "bal_median": "median BALROG-max per seed",
    "balmin_max": "best BALROG-min per seed",
}


def select(rolls, mode):
    """Rollouts to draw, per the selection rule."""
    if mode == "all":
        return rolls
    out = []
    for seed in sorted({r["seed"] for r in rolls}):
        grp = [r for r in rolls if r["seed"] == seed]
        if not grp:
            continue
        if mode == "bal_max":
            out.append(max(grp, key=lambda r: r["bal"]))
        elif mode == "balmin_max":
            out.append(max(grp, key=lambda r: (r["bal_min"], r["bal"])))
        elif mode == "bal_median":
            out.append(sorted(grp, key=lambda r: r["bal"])[len(grp) // 2])
    return out


def _render(mode, doc):
    rng = np.random.default_rng(SEEDJ)
    fig, ax = plt.subplots(figsize=(9.8, 6.2))

    def stair(pts, jitter=0.0, **kw):
        d = np.minimum([p[0] for p in pts], DMAX).astype(float)
        x = np.minimum([p[1] for p in pts], XMAX).astype(float)
        j = rng.uniform(-jitter, jitter, 2) if jitter else (0.0, 0.0)
        ax.step(d + j[0], x + j[1], where="post", **kw)
        return d[-1] + j[0], x[-1] + j[1]

    # humans first: the field the agents are read against
    for h in doc["human_dead"]:
        e = stair(h["pts"], .10, color=LIGHT, lw=1.0, alpha=.75, zorder=1)
        ax.plot(*e, "x", color=LIGHT, ms=5, mew=1.3, zorder=2)
    for h in doc["human_won"]:
        e = stair(h["pts"], .10, color=GREY, lw=1.0, ls="--", alpha=.60,
                  zorder=2)
        ax.plot(*e, "x", color=GREY, ms=5, mew=1.3, zorder=3)

    counts = {}
    for key in ORDER:
        m = doc["models"][key]
        sel = select(m["rollouts"], mode)
        counts[key] = len(sel)
        for r in sel:
            e = stair(r["pts"], .14, color=COL[key], lw=1.5, alpha=.85,
                      zorder=4)
            ax.plot(*e, "x", color=COL[key], ms=7, mew=1.8, zorder=5)

    handles = [
        Line2D([], [], color=LIGHT, lw=1.4, marker="x", ms=5, mew=1.3,
               label=rf"NAO deaths ($n{{=}}{len(doc['human_dead'])}$ of "
                     rf"{doc['n_dead_pool']:,})".replace(",", "{,}")),
        Line2D([], [], color=GREY, lw=1.4, ls="--", marker="x", ms=5, mew=1.3,
               label=rf"NAO ascensions ($n{{=}}{len(doc['human_won'])}$ of "
                     rf"{doc['n_won_pool']:,})".replace(",", "{,}")),
    ] + [
        Line2D([], [], color=COL[k], lw=1.7, marker="x", ms=7, mew=1.8,
               label=rf"{doc['models'][k]['label']} ($n{{=}}{counts[k]}$)")
        for k in ORDER
    ]

    ax.set_xlim(0.5, DMAX + 0.5)
    ax.set_ylim(0.5, XMAX + 0.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    style_axes(ax, "Dungeon Level", "Experience Level")
    ax.grid(False)
    ax.set_title(MODES[mode], loc="left", pad=8)   # no \textbf: see style.py
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.115),
               ncol=4, fontsize=legendsize - 6, frameon=False,
               handletextpad=.6, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.27)
    save(fig, f"{NAME}_{mode}", outdir=OUTDIR)


def render():
    doc = json.load(open(SRC))
    for mode in MODES:
        _render(mode, doc)


if __name__ == "__main__":
    render()
