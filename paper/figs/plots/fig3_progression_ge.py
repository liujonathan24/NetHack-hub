"""BALROG-min against game turns: five models, one orchestrated arm, two human bands.

WHAT IS PLOTTED. Eight series, every one a MEDIAN across its rollouts at each
game turn, with no individual traces and no shaded intervals -- see the two
notes below, both of which are deliberate and both of which changed this figure
from its earlier form.

  * Two human references from NAO: the whole population and the subset that won.
  * Five single-life agent arms, all on the same uncapped base harness surface:
    GLM-5.2 (the reference, n=15) plus GPT-5.6 sol, GPT-5.6 luna,
    Gemini 3.7 Flash and Qwen3.8-Max (E14 diversity, 5 seeds x 3 repeats each).
  * One orchestrated arm: Go-Explore best paths, seven runs, cut at 30 attempts.

NO SHADED INTERVALS, ANYWHERE. The earlier version drew interquartile bands for
the two human series and nothing for the agents, which was already inconsistent;
adding five more agent arms at n=15 would have made it indefensible. An IQR band
at n=15 on a heavy-tailed metric is wide enough to swallow the differences the
figure exists to show, and drawing bands for the n=44,008 humans while omitting
them for the agents implies the agent medians are the better-determined numbers,
which is backwards. So every series is a bare median and the n's are in the
legend. Spread belongs in a table, not here.

NO INDIVIDUAL TRACES. The earlier version drew every rollout as a thin line.
At two arms that was texture; at six it is a haze that hides the medians. The
per-rollout endpoints are in fig4_depth_xp, which is the figure for dispersion.

WHY GO-EXPLORE IS A LINEAGE, NOT AN ATTEMPT. A Go-Explore attempt resumes a
checkpoint, so it starts partway along the game clock and cannot be plotted
against turns as if it were a game. The archive is a tree: take the checkpoint
with the best BALROG-min, walk `parent` links back to the seed, and that chain
is one continuous history from turn 1 in which every edge was really played.
It is a SURVIVING path by selection -- the branches that died are not in it --
so it is a best-of-search trajectory and belongs beside NAO successes rather
than beside the population median.

THE BOTTOM PANEL MEANS TWO THINGS, and the caption must say so. For the humans
and the five single-life arms it is the fraction of games whose character is
still alive. A lineage never dies: it ends where the SEARCH stopped, at the
attempt cap or the plateau guard. So the Go-Explore curve there reads "fraction
of runs still finding better states". Same construction (share of series still
defined at t), different meaning.

COLOUR. The five model arms take fig4_depth_xp's assignment verbatim so a model
is the same colour in both figures, which uses up every house colour and pushes
the two human series onto greys -- they are context here, the agents are the
subject. Go-Explore shares GLM-5.2's green, dashed: the pairing is the point,
because Go-Explore IS GLM-5.2 with an archive, and putting them in one hue says
"same player, different mode" in a way a separate colour would hide. It only
works while GLM-5.2 is also green; if that ever changes, this must change too.

MEAN, NOT MEDIAN. Every series is a MEAN over the rollouts still alive at that
turn. The figure previously used medians, inherited from the human panel where
the population is summarised by percentiles. Mean is the better choice here for
two reasons. It matches the rest of the paper, which reports mean BALROG with a
standard error (the tier contract's reference numbers are means). And BALROG-min
is zero-inflated -- a rollout scores 0 until BOTH axes move, and the XL
percentile is 0 below XL 2 -- so the median of an arm can sit at exactly 0 while
the arm is plainly making progress: GPT-5.6 sol and luna both have a median of 0
for their whole life against means of 1.65 and 1.50. A summary that reports 0
for an arm that is not at 0 is worse than a noisy one.

The cost is real and worth stating in the caption: at n=15 on a heavy-tailed
metric the mean is pulled by one or two lucky rollouts, so differences between
the comparison models should not be read as ordered. Set SUMMARY = "median"
below to get the old behaviour.

BOTH SUMMARIES CONDITION ON SURVIVAL. At turn t only the rollouts still alive
contribute, so every curve rises partly because weak games have dropped out.
That is inherent to a turn-indexed view and applies to the human series too.

TWO METRICS. render() writes both panels-pairs: BALROG-min (the weaker axis,
which rewards balanced progress) and BALROG-max (the stronger axis, effectively
depth). They matter differently: on min, three of the four comparison models
have a median of identically zero for their whole life, because the XL
percentile is 0 below XL 2 and at least half their rollouts sit at XL 1. On max
all four are visible. The min figure is the honest headline; the max figure is
what the field usually reports.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from style import (HUMAN, HUMAN_TOP, AGENT, BAD, OTHER, GREY, LIGHT,
                   legendsize, LW, ALPHA, style_axes, save)
from common import compare, arm, balrog_table, _pct

NAME = "fig3_progression_ge"
NAME_MAX = "fig3_progression_ge_max"
GE = "/root/overleaf/figs/e16_ge30_lineage.json"
DIV = "/root/overleaf/figs/e14_diversity_turns.json"
XT = [10, 100, 1000, 10000, 50000]
XTL = ["10", "100", "1k", "10k", "50k"]
FLOOR = 0.5

#: "median" or "mean". MEDIAN is the default, chosen for legibility: on a
#: discrete metric it steps rather than drifts, so each curve reads as a
#: sequence of achieved states instead of a squiggle. The cost is that a
#: zero-inflated arm summarises to exactly 0 -- see the note in
#: fig3_progression_pair, where the min|max pair is what makes that affordable.
SUMMARY = "median"

#: Model -> colour, copied from fig4_depth_xp.SERIES so a model reads the same
#: in both figures. GLM-5.2 keeps AGENT; the comparison models take the rest.
DIVC = {"sol": HUMAN, "luna": HUMAN_TOP, "gemini37": OTHER, "qwen38": BAD}

#: Go-Explore is GLM-5.2 with an archive, so it shares AGENT green and carries
#: the difference in the dash. This is only legible because no other series is
#: green; it is tied to GLM-5.2 keeping AGENT above.
GOEX = AGENT

#: Go-Explore is excluded for treesmoke8: N=20, stall_attempts=8, and its first
#: eight rounds had no winners'-norm signal, so it is a before/after inside one
#: run rather than an arm of this experiment.
EXCLUDE = {"treesmoke8"}


def _step_on_grid(pts, end, g, ci=1):
    """A step series sampled onto g; NaN past `end`. `ci` picks the value column
    (1 = BALROG-min, 2 = BALROG-max) of the [turn, min, max] records."""
    t = np.array([p[0] for p in pts], float)
    v = np.maximum.accumulate([p[ci] for p in pts])
    idx = np.searchsorted(t, g, side="right") - 1
    out = np.where(idx >= 0, v[np.clip(idx, 0, len(v) - 1)], 0.0)
    return np.where(g <= end, out, np.nan)


def _base_on_grid(g, metric):
    """The base arm's running BALROG-min or -max per rollout, stepped onto g.

    common.e14_on_grid_min only offers the min, and the max figure needs the
    same construction on the other axis: running max per axis first, then min OR
    max across the two. Written here rather than in common.py so the shared
    module keeps exactly the surface the published figures already rely on.
    """
    dl, xp = balrog_table()
    pick = np.minimum if metric == "min" else np.maximum
    rows = []
    for r in arm("base")["rollouts"]:
        pts = sorted((t["gturn"], _pct(dl, t["dlvl"]), _pct(xp, t["xp"]))
                     for t in r["turns"] if "gturn" in t)
        if not pts:
            continue
        gt = np.array([p[0] for p in pts], float)
        bv = pick(np.maximum.accumulate([p[1] for p in pts]),
                  np.maximum.accumulate([p[2] for p in pts]))
        idx = np.searchsorted(gt, g, side="right") - 1
        vals = np.where(idx >= 0, bv[np.clip(idx, 0, len(bv) - 1)], 0.0)
        rows.append(np.where(g <= r["end_gturn"], vals, np.nan))
    return np.array(rows)


def _summary_line(ax, bx, g, M, color, label, ls="-", lw=None, minalive=5,
                  bottom=True):
    """One summary line on the top panel and its alive fraction on the bottom."""
    alive = np.sum(~np.isnan(M), axis=0)
    keep = alive >= min(minalive, M.shape[0])
    med = (np.nanmean if SUMMARY == "mean" else np.nanmedian)(M, axis=0)
    ax.plot(g[keep], np.maximum(med[keep], FLOOR), color=color, ls=ls,
            lw=lw or (LW + 1.0), alpha=ALPHA, label=label)
    if bottom:
        bx.plot(g, 100 * alive / M.shape[0], color=color, ls=ls,
                lw=lw or (LW + 1.0), alpha=ALPHA)


def _render(metric, name):
    """One figure for one metric. `metric` is "min" or "max"."""
    ci = 1 if metric == "min" else 2
    d = compare()
    g = np.array(d["grid"], float)
    nao = d["pop"]["nao"]
    wins = nao["wins"]

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(12, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.10})

    # ---- human references, as greys: context, not subject -------------------
    for src, col, ls, lab, n in [
            (nao, LIGHT, "-", "NAO population", nao["stats"]["n"]),
            (wins, GREY, "--", "NAO successes", wins["n"])]:
        hs = (src["series"][f"mean_{metric}"] if SUMMARY == "mean"
              else src["series"][metric]["50"])
        ax.plot(g, np.maximum(np.array(hs, float), FLOOR),
                color=col, lw=LW, ls=ls, alpha=ALPHA,
                label=rf"{lab} ($n{{=}}{n:,}$)".replace(",", "{,}"))
        a = np.array(src["series"]["alive"], float)
        bx.plot(g, 100 * a / a[0], color=col, lw=LW, ls=ls, alpha=ALPHA)

    # ---- five single-life agent arms ----------------------------------------
    B = _base_on_grid(g, metric)
    _summary_line(ax, bx, g, B, AGENT, rf"GLM-5.2 ($n{{=}}{B.shape[0]}$)")

    div = json.load(open(DIV))
    for key in ("sol", "luna", "gemini37", "qwen38"):
        v = div[key]
        M = np.array([_step_on_grid(r["pts"], r["end"], g, ci)
                      for r in v["rollouts"]])
        # On BALROG-min three of these four sit at a median of identically zero
        # for their whole life (XL percentile is 0 below XL 2), so the label
        # says so rather than leaving a reader to wonder why the line is flat.
        flat = (SUMMARY == "median"
                and np.nanmax(np.nanmedian(M, axis=0)) < FLOOR)
        lab = rf"{v['label']} ($n{{=}}{v['n']}$" + \
              (r", median $\equiv 0$)" if flat else ")")
        _summary_line(ax, bx, g, M, DIVC[key], lab)

    # ---- the orchestrated arm -----------------------------------------------
    lin = [x for x in json.load(open(GE))["lineages"] if x["run"] not in EXCLUDE]
    L = np.array([_step_on_grid(x["pts"], x["pts"][-1][0], g, ci) for x in lin])
    _summary_line(ax, bx, g, L, GOEX,
                 rf"Go-Explore best path ($n{{=}}{L.shape[0]}$)",
                 ls="--", lw=LW + 2.0, minalive=3)

    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 90)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axes(ax, None, rf"BALROG-{metric}, {SUMMARY} of alive (\%)")

    bx.set_ylim(0, 105)
    bx.set_yticks([0, 50, 100])
    bx.set_xscale("log")
    bx.set_xlim(10, 50000)
    bx.set_xticks(XT)
    bx.set_xticklabels(XTL)
    style_axes(bx, "Game Turns", "Alive (\\%)")

    # Legend below the x-axis label: eight series will not fit inside the axes
    # without covering the early-game rise, which is the part being compared.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=4, fontsize=legendsize - 6, frameon=False)
    fig.subplots_adjust(bottom=0.20)
    save(fig, name)


def render():
    _render("min", NAME)
    _render("max", NAME_MAX)


if __name__ == "__main__":
    render()
