"""The headline figure: BALROG-min | BALROG-max against game turns, with the
final-score distribution of GLM-5.2 and Go-Explore as box plots beside each
panel.

The curves are fig3_progression_pair's (median of running maxima, dead runs
carried forward, faint after the last death). What is new is the marginal axis
to the right of each panel: it shares the y-scale and holds one box per series
of the per-rollout FINAL score, with the rollouts themselves drawn over the box
so that n=15 and n=7 are visible as what they are. The curve's last value and
the box's median line are the same number, so the two views join up.

Zeros. Six of the fifteen GLM-5.2 single lives never leave XL 1 and score
exactly 0 on BALROG-min. On the log axis they sit on the floor (FLOOR = 0.5);
the box's lower whisker and quartile reach the floor for the same reason. The
caption has to say that a point on the floor is a zero, not a small number.

Go-Explore is blue here, not GLM's green: the pair's "same player, different
mode" hue only worked while the two were never side by side, and two boxes in
one colour would be unreadable. Blue is what Explore is in the plane figures.

VARIANTS. `render(all_models=True)` keeps the four comparison models' curves;
`all_models=False` draws only the humans, GLM-5.2 and Go-Explore, which is the
version with nothing on it that the box plots do not also explain.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from style import (AGENT, blue, GREY, LIGHT, legendsize, LW, ALPHA, labelsize,
                   style_axes, save)
from common import compare, arm, balrog_table, _pct
from plots.fig3_progression_ge import GE, DIV, EXCLUDE, DIVC, FLOOR
from plots.fig3_progression_pair import (_carry, _base_carry, _line, PANELS,
                                         XT, XTL)

NAME = "fig3_progression_main"
OUTDIR = "/root/overleaf/images/proposed"
GOEX = blue
OTHERS = "#A9B7C6"     # the four comparison models, as context


def _final(M):
    """Per-rollout final score from a carried-forward matrix."""
    return M[:, -1]


def _box(ax, vals, pos, color, rng):
    v = np.maximum(np.asarray(vals, float), FLOOR)
    bp = ax.boxplot([v], positions=[pos], widths=0.55, patch_artist=True,
                    showfliers=False, whis=(0, 100), zorder=3,
                    medianprops=dict(color=color, lw=2.4),
                    boxprops=dict(facecolor=to_rgba(color, .22), edgecolor=color,
                                  lw=1.4),
                    whiskerprops=dict(color=color, lw=1.4),
                    capprops=dict(color=color, lw=1.4))
    j = rng.uniform(-.16, .16, len(v))
    ax.scatter(pos + j, v, s=18, color=color, alpha=.85, lw=0, zorder=4)
    return bp


def render(all_models=True):
    d = compare()
    g = np.array(d["grid"], float)
    nao, wins = d["pop"]["nao"], d["pop"]["nao"]["wins"]
    div = json.load(open(DIV))
    lin = [x for x in json.load(open(GE))["lineages"] if x["run"] not in EXCLUDE]
    rng = np.random.default_rng(5)

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.15, 1, 0.15],
                          wspace=0.06, left=0.07, right=0.99, top=0.93,
                          bottom=0.30)
    mains, boxes = [], []
    for i, (metric, title) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, 2 * i])
        bx = fig.add_subplot(gs[0, 2 * i + 1], sharey=ax)
        mains.append(ax); boxes.append(bx)
        ci = 1 if metric == "min" else 2

        for src, col, ls, lab, n in [
                (nao, LIGHT, "-", "NAO population", nao["stats"]["n"]),
                (wins, GREY, "--", "NAO successes", wins["n"])]:
            hs = src["series"][metric]["50"]
            ax.plot(g, np.maximum(np.array(hs, float), FLOOR), color=col,
                    lw=LW, ls=ls, alpha=ALPHA,
                    label=rf"{lab} ($n{{=}}{n:,}$)".replace(",", "{,}"))

        B, BA = _base_carry(g, metric)
        _line(ax, g, B, BA, AGENT, rf"GLM-5.2 ($n{{=}}{B.shape[0]}$)")

        if all_models:
            # Context, not subject: the four comparison models in one muted
            # tone and one legend entry. Giving them their own colours put
            # GPT-5.6 sol in Go-Explore's blue and pulled the eye four ways.
            for j, key in enumerate(("sol", "luna", "gemini37", "qwen38")):
                v = div[key]
                cv = [_carry(r["pts"], r["end"], g, ci) for r in v["rollouts"]]
                M = np.array([c[0] for c in cv]); A = np.array([c[1] for c in cv])
                _line(ax, g, M, A, OTHERS,
                      r"other frontier models ($4 \times n{=}15$)" if j == 0 else None,
                      lw=LW - 1.4)

        cl = [_carry(x["pts"], x["pts"][-1][0], g, ci) for x in lin]
        L = np.array([c[0] for c in cl]); LA = np.array([c[1] for c in cl])
        _line(ax, g, L, LA, GOEX, rf"Go-Explore ($n{{=}}{L.shape[0]}$)")

        # -- marginal: final-score distributions -------------------------------
        _box(bx, _final(B), 0, AGENT, rng)
        _box(bx, _final(L), 1, GOEX, rng)
        bx.set_xlim(-0.6, 1.6)
        bx.set_xticks([])
        bx.tick_params(axis="y", which="both", left=False, right=False,
                       labelleft=False, labelright=False)
        for sp in ("top", "right", "left", "bottom"):
            bx.spines[sp].set_visible(False)
        bx.grid(False)
        bx.set_title("final", loc="center", pad=8, fontsize=labelsize - 6,
                     color=GREY)

        ax.set_yscale("log")
        ax.set_ylim(FLOOR, 90)
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xscale("log")
        ax.set_xlim(100, 50000)
        ax.set_xticks(XT)
        ax.set_xticklabels(XTL)
        style_axes(ax, "In-Game Turns",
                   r"Progression (\%)" if metric == "min" else None)
        ax.set_title(title, loc="left", pad=8)

    handles, labels = mains[0].get_legend_handles_labels()
    keep = [i for i, l in enumerate(labels) if l and not l.startswith("_")]
    handles = [handles[i] for i in keep]; labels = [labels[i] for i in keep]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.15),
               ncol=4 if all_models else 4, fontsize=legendsize - 6,
               frameon=False, columnspacing=1.6)
    save(fig, NAME + ("" if all_models else "_two"), outdir=OUTDIR)


if __name__ == "__main__":
    render(True); render(False)
