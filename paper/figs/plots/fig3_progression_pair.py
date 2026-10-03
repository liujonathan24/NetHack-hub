"""BALROG against game turns, as a min|max pair. The headline progression figure.

WHY TWO PANELS AND NOT ONE. The four comparison models sit at or near zero on
BALROG-min for their whole life. That is not a plotting problem to be worked
around -- it is the result. BALROG-min only rises when depth AND experience both
rise, so a model that descends while staying at XL 1 scores nothing on it. Put
the two metrics side by side and that becomes legible in one look: on the right
panel the weak models are clearly moving (they are finding stairs), on the left
they are flat (the character behind the descent never develops). One panel would
force a choice between showing the differences and showing the finding.

WHAT IS PLOTTED. Seven series per panel, every one a MEAN over the rollouts
still alive at that game turn.

  * Two human references from NAO: the whole population and the subset that won.
  * Five single-life agent arms on the same uncapped base harness surface:
    GLM-5.2 (n=15) plus GPT-5.6 sol, GPT-5.6 luna, Gemini 3.7 Flash and
    Qwen3.8-Max (E14 diversity, 5 seeds x 3 repeats each).
  * One orchestrated arm: Go-Explore best paths, seven runs, cut at 30 attempts.

NO ALIVE PANEL. The earlier version carried a survival strip under the curves.
It is dropped: it doubled the figure's height for a quantity that is context
rather than result, and its meaning was not even constant across series
(character survival for the single-life arms, "still finding better states" for
the lineages). What it carried now lives in the fade described below.

MONOTONE BY CONSTRUCTION: DEAD RUNS ARE CARRIED FORWARD. Each rollout is a
running maximum, so no single run can go down. An earlier version summarised
only over the runs still ALIVE at t, and that curve fell whenever an
above-median run died -- Gemini 3.7 Flash lost its 6.96 and 12.56 rollouts at
turn 401 and its median dropped from 3.54 to 2.12. Nothing had regressed; the
best games had stopped existing. Since the summary is taken over a set that
changes, neither the median nor the mean is monotone, and the MEAN is worse:
3-8 downward steps per arm against 1-3 for the median.

Carrying a dead run forward at its final value fixes this at the root. The set
being summarised is then constant, so a median of running maxima is itself a
running maximum: zero downward steps for every arm on both metrics, verified.
The curve now answers "run this agent and wait N turns -- what is the median
best it will have achieved", which is the question a reader actually has, and it
does not condition on survival at all.

Survivorship still has to be visible, or a plateau reads as an agent that is
alive and stuck rather than one that is dead. It is encoded in the line itself:
solid until the arm's LAST run dies, faint and flat after. Every solid point is
supported by at least one game still being played; the faint tail is the
carried-forward plateau and nothing else. This is what the removed alive panel
used to carry, at no vertical cost.

MEDIAN, NOT MEAN. Chosen for legibility: BALROG is a discrete metric, so a
median steps between achieved states while a mean drifts between them. The
curves read as a sequence of things that actually happened rather than as
squiggles. This is affordable ONLY because of the min|max pair -- a median
reports exactly 0 for a zero-inflated arm, and GPT-5.6 sol and luna are at 0 for
their whole life on the min panel. The max panel is where those two are legible,
which is the pair paying for itself. Series flattened to zero are labelled as
such in the legend so a floored line is not mistaken for a missing one.

The alternative, kept one edit away as SUMMARY = "mean", matches the rest of the paper (the tier contract's reference
numbers are means with standard error) and avoids reporting a flat zero for arms
that are not at zero: GPT-5.6 sol and luna both have a median of exactly 0 for
their whole life against means of 0.76 and 0.49. The cost is that at n=15 on a
heavy-tailed metric the mean is pulled by one or two lucky rollouts, so the
ordering AMONG the four comparison models is not to be read as meaningful. Set
SUMMARY = "median" to get the old behaviour.

WHY GO-EXPLORE IS A LINEAGE. A Go-Explore attempt resumes a checkpoint, so it
starts partway along the game clock and cannot be plotted against turns as if it
were a game. The archive is a tree: take the checkpoint with the best
BALROG-min, walk `parent` links back to the seed, and that chain is one
continuous history from turn 1 in which every edge was really played. It is a
SURVIVING path by selection -- the branches that died are not in it -- so it
belongs beside NAO successes rather than beside the population mean.

COLOUR. The five model arms take fig4_depth_xp's assignment verbatim so a model
reads the same across figures. Go-Explore shares GLM-5.2's green, dashed at the
same weight as every other arm: Go-Explore IS GLM-5.2 with an archive, and one hue says "same player,
different mode". Only legible while GLM-5.2 is the sole other green series.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from style import (HUMAN, HUMAN_TOP, AGENT, BAD, OTHER, GREY, LIGHT,
                   legendsize, LW, ALPHA, style_axes, save)
from common import compare, arm, balrog_table, _pct
from plots.fig3_progression_ge import (GE, DIV, EXCLUDE, DIVC, GOEX, FLOOR,
                                       SUMMARY)

NAME = "fig3_progression_pair"
XT = [100, 1000, 10000, 50000]
XTL = ["100", "1k", "10k", "50k"]

#: Panel order. Left is the conservative joint metric and carries the claim;
#: right is effectively depth and is what the field usually reports. The gloss
#: that used to sit beside each title lives in the caption now -- what the two
#: metrics mean is worth a sentence, not three words squeezed into an axes.
PANELS = [("min", "BALROG-min"), ("max", "BALROG-max")]


def _carry(pts, end, g, ci):
    """Step series with dead runs CARRIED FORWARD at their final value.

    Returns (values, alive_mask). The values never go NaN, so a median across
    runs is monotone by construction: every run is a running maximum and the
    set being summarised never changes. The mask records where the run was
    still alive, which is used only for line styling.
    """
    t = np.array([p[0] for p in pts], float)
    v = np.maximum.accumulate([p[ci] for p in pts])
    idx = np.searchsorted(t, g, side="right") - 1
    out = np.where(idx >= 0, v[np.clip(idx, 0, len(v) - 1)], 0.0)
    return out, g <= end


def _base_carry(g, metric):
    """_base_on_grid, carried forward. Same construction, no NaN tail."""
    dl, xp = balrog_table()
    pick = np.minimum if metric == "min" else np.maximum
    vals, alive = [], []
    for r in arm("base")["rollouts"]:
        pts = sorted((t["gturn"], _pct(dl, t["dlvl"]), _pct(xp, t["xp"]))
                     for t in r["turns"] if "gturn" in t)
        if not pts:
            continue
        gt = np.array([p[0] for p in pts], float)
        bv = pick(np.maximum.accumulate([p[1] for p in pts]),
                  np.maximum.accumulate([p[2] for p in pts]))
        idx = np.searchsorted(gt, g, side="right") - 1
        vals.append(np.where(idx >= 0, bv[np.clip(idx, 0, len(bv) - 1)], 0.0))
        alive.append(g <= r["end_gturn"])
    return np.array(vals), np.array(alive)


def _summary(M):
    return (np.nanmean if SUMMARY == "mean" else np.nanmedian)(M, axis=0)


def _line(ax, g, M, A, color, label, ls="-", lw=None):
    """One monotone summary line, solid for as long as ANY run is still alive.

    The cut is the last death in the arm, not a half-alive threshold. Before it,
    every point is supported by at least one game still being played. After it
    the curve is flat by construction -- nothing is left to raise it -- so the
    tail is drawn as a faint straight line to the right edge. It carries one
    piece of information, the level the arm finished at, and it should not look
    like play.
    """
    y = np.maximum(_summary(M), FLOOR)
    any_alive = A.any(axis=0)
    cut = int(np.argmax(~any_alive)) if (~any_alive).any() else len(g)
    lw = lw or LW
    ax.plot(g[:cut], y[:cut], color=color, lw=lw, ls=ls, label=label,
            solid_capstyle="round", dash_capstyle="round", zorder=3)
    if cut < len(g):
        # Flat by construction, so two points rather than a resampled series.
        ax.plot([g[cut - 1], g[-1]], [y[cut - 1], y[cut - 1]], color=color,
                lw=lw, ls=ls, alpha=0.22, zorder=2)


def render():
    d = compare()
    g = np.array(d["grid"], float)
    nao, wins = d["pop"]["nao"], d["pop"]["nao"]["wins"]
    div = json.load(open(DIV))
    lin = [x for x in json.load(open(GE))["lineages"] if x["run"] not in EXCLUDE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)

    for ax, (metric, title) in zip(axes, PANELS):
        ci = 1 if metric == "min" else 2

        # Humans first so the agent arms draw over them: context, not subject.
        for src, col, ls, lab, n in [
                (nao, LIGHT, "-", "NAO population", nao["stats"]["n"]),
                (wins, GREY, "--", "NAO successes", wins["n"])]:
            hs = (src["series"][f"mean_{metric}"] if SUMMARY == "mean"
                  else src["series"][metric]["50"])
            ax.plot(g, np.maximum(np.array(hs, float), FLOOR), color=col,
                    lw=LW, ls=ls, alpha=ALPHA,
                    label=rf"{lab} ($n{{=}}{n:,}$)".replace(",", "{,}"))

        B, BA = _base_carry(g, metric)
        _line(ax, g, B, BA, AGENT, rf"GLM-5.2 ($n{{=}}{B.shape[0]}$)")

        for key in ("sol", "luna", "gemini37", "qwen38"):
            v = div[key]
            cv = [_carry(r["pts"], r["end"], g, ci) for r in v["rollouts"]]
            M = np.array([c[0] for c in cv]); A = np.array([c[1] for c in cv])
            # A zero-inflated arm summarises to exactly 0 under a median and is
            # drawn on the axis floor. Say so in the legend, or the reader
            # cannot tell a floored series from one that was left out.
            flat = np.nanmax(_summary(M)) < FLOOR
            lab = rf"{v['label']} ($n{{=}}{v['n']}$" + \
                  (rf", {SUMMARY} $\equiv 0$)" if flat else ")")
            _line(ax, g, M, A, DIVC[key], lab)

        cl = [_carry(x["pts"], x["pts"][-1][0], g, ci) for x in lin]
        L = np.array([c[0] for c in cl]); LA = np.array([c[1] for c in cl])
        _line(ax, g, L, LA, GOEX, rf"Go-Explore ($n{{=}}{L.shape[0]}$)",
              ls="--")

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
        # Panel identity sits inside the axes: a suptitle per column would
        # collide with the shared legend and read as a figure title.
        ax.set_title(title, loc="left", pad=8)   # no \textbf: see style.py

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.145),
               ncol=4, fontsize=legendsize - 6, frameon=False)
    fig.subplots_adjust(bottom=0.26, wspace=0.16)
    save(fig, NAME)


if __name__ == "__main__":
    render()
