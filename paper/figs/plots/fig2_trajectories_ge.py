"""The depth-experience plane, for the same series as the main figure.

Same comparison, different view. fig_main_min shows BALROG-min against game
turns and its final distribution; this shows WHERE on the (Dlvl, XL) plane each
series gets to, which is what BALROG-min is made of. The series, names and
colours are the main figure's key set -- the leaderboard entry, GLM-5.2 and its
three variants -- plus the 100-attempt continuation of Explore, so a reader
carries one palette between the two figures.

ONE STATISTIC FOR EVERY LINE: the median experience level a series holds on
first arrival at each dungeon level, ending at the deepest level half of its
runs reached. The human population and the ascensions are drawn as the same
statistic (median; the ascensions also get their interquartile band), so
agents and humans are directly comparable at every depth.

THE SHADING is the BALROG-min level set: min(depth pct, XL pct) >= L. It is
L-shaped because a run enters it only when both axes rise, and it is what
separates the families -- every single life stays under 5%, and only Explore
reaches 10% and 20%. Labelled at the right edge, where no line goes.

Data: e_progression_main.json, built by gen_progression_main.py, which checks
every series against the paper's tables before writing. The continual arm is
one rollout per seed per iteration (n=15).
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import GREY, HUMAN_TOP, LW, legendsize, labelsize, ticksize, style_axes, save
from common import winners_by_depth
from plots.fig4_plane_story import min_bands, _elbow
from plots.fig_main_min import COL, SRC

NAME = "fig2_trajectories_ge"
OUTDIR = "/root/overleaf/images/proposed"
POP = "/root/nld/all_games_v2.json"

DMAX, XMAX = 26, 15.5
BANDS = [2, 5, 10, 20]
#: The LLMs as single lives, then Explore. The four comparison models have no
#: (Dlvl, XL) path in the main data file, so theirs come from e14_plane.json.
SERIES = ["sol", "luna", "gemini37", "qwen38", "glm52", "explore"]
PLANE = "/root/overleaf/figs/e14_plane.json"


def pace_of(paths, minn=None):
    """Median XL on first arrival at each depth over a list of (Dlvl, XL)
    staircases, held one level past the last depth half the runs reached."""
    by = {}
    for path in paths:
        first = {}
        for d, x in path:
            first.setdefault(int(d), x)
        for d, x in first.items():
            by.setdefault(d, []).append(x)
    minn = minn if minn is not None else max(2, -(-len(paths) // 2))
    ds = [k for k in sorted(by) if len(by[k]) >= minn]
    ys = [float(np.median(by[k])) for k in ds]
    return ds + [ds[-1] + 1], ys + [ys[-1]]


def population_paths():
    """(Dlvl, XL) staircases of every decoded NAO game with more than one
    turn (n=46,796), the population the paper's tables and the main figure's
    black line use (turns > 1, as in prep_compare_v2.py)."""
    from common import _invert_tables, _depth_xp_track
    D, X = _invert_tables()
    out = []
    for g in json.load(open(POP)):
        if int(g.get("turns") or 0) <= 1:
            continue
        seen = _depth_xp_track(json.loads(g["c"]) if isinstance(g["c"], str) else g["c"], D, X)
        # a game that never left Dlvl 1 at XL 1 has no percentile steps to
        # invert, but it is still a game at (1, 1) and belongs in the median
        out.append(sorted(seen.items()) if seen else [(1, 1)])
    return out


def render(inset=False):
    doc = json.load(open(SRC))
    series = doc["series"]
    e14 = json.load(open(PLANE))["models"]
    def paths_of(key):
        rs = series[key]["rollouts"]
        if "plane" in rs[0]:
            return [r["plane"] for r in rs]
        return [r["pts"] for r in e14[key]["rollouts"]]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_xlim(0.5, DMAX + 0.5)
    ax.set_ylim(0.5, XMAX)
    min_bands(ax, levels=BANDS, alpha=0.045, label=False)
    # the gradient as an axis: every band's corner is at Dlvl 4-12, so across
    # the drawn plane the experience level is the binding term of BALROG-min
    # and the band boundaries are horizontal. A right-hand axis at those
    # heights reads the metric off the XL directly.
    rx = ax.twinx()
    rx.set_ylim(*ax.get_ylim())
    elbows = [(L, _elbow(L)[1]) for L in BANDS]
    rx.set_yticks([x for _, x in elbows])
    rx.set_yticklabels([rf"{L}\%" for L, _ in elbows])
    rx.set_ylabel(r"BAL$_{\min}$", fontsize=labelsize, rotation=270, labelpad=26)
    rx.tick_params(axis="y", labelsize=ticksize, length=4, colors=GREY)
    rx.grid(False)

    # humans: population median (black), ascensions median + IQR (grey)
    pop = population_paths()
    pd_, py_ = pace_of(pop)
    ax.step(pd_, py_, where="post", color="black", lw=LW - 0.4, zorder=5,
            solid_capstyle="butt")
    by, n_win = winners_by_depth()
    ds = [d for d in sorted(by) if d <= DMAX and len(by[d]) >= 10]
    q = lambda p: np.array([np.percentile(by[d], p) for d in ds])
    # the band is purple, not grey: grey on the grey BALROG-min shading was
    # invisible, and purple is the house colour for the NAO ascensions
    ax.fill_between(ds, q(25), q(75), step="post", color=HUMAN_TOP, alpha=.28,
                    lw=0, zorder=2)
    ax.step(ds, q(50), where="post", color=GREY, lw=LW - 0.5, ls="--", zorder=5)

    handles = [
        Line2D([], [], color="black", lw=LW,
               label=rf"Human population ($n{{=}}{len(pop):,}$)".replace(",", "{,}")),
        Line2D([], [], color=GREY, lw=LW - 0.5, ls="--",
               label=rf"Human successes ($n{{=}}{n_win}$)"),
        Patch(color=HUMAN_TOP, alpha=.35, label="Human successes, interquartile"),
    ]
    # the single lives all sit on XL 1-2 for their first levels; a small fixed
    # vertical dodge keeps each visible as its own line (fig4 does the same)
    # upward only, so no line ever sits below the XL 1 floor
    lives = [k for k in SERIES if k != "explore"]

    def draw_lives(target, dodge, scale=1.0):
        for key in SERIES:
            xs, ys = pace_of(paths_of(key))
            off = (lives.index(key) + 1) * dodge if key in lives else 0
            strong = key in ("glm52", "explore")
            lw = (3.2 if strong else 1.6) * scale
            # thin lines above thick ones: a 3pt line is ~0.2 XL tall here and
            # would otherwise swallow the neighbours it is dodged past
            target.step(xs, np.asarray(ys) + off, where="post", color=COL[key],
                        lw=lw, zorder=8 - 2 * strong, solid_capstyle="butt")

    draw_lives(ax, 0.11)
    for key in SERIES:
        strong = key in ("glm52", "explore")
        handles.append(Line2D([], [], color=COL[key], lw=3.2 if strong else 1.6,
                              label=rf"{series[key]['label']} ($n{{=}}{len(paths_of(key))}$)"))

    if inset:
        # the single lives, magnified: the empty upper-left corner holds a
        # zoom of Dlvl 1-11, XL 1-3, where all of them live
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        ins = inset_axes(ax, width="36%", height="30%", loc="upper left",
                         borderpad=1.2)
        ins.set_xlim(0.8, 11.2)
        ins.set_ylim(0.85, 2.9)
        draw_lives(ins, 0.05, scale=0.8)
        ins.step(pd_, py_, where="post", color="black", lw=2.0, zorder=5,
                 solid_capstyle="butt")
        ins.set_xticks([1, 5, 10])
        ins.set_yticks([1, 2, 3])
        ins.tick_params(labelsize=ticksize - 5, length=3)
        ins.set_facecolor("white")
        for sp in ins.spines.values():
            sp.set_color(GREY)
        mark_inset(ax, ins, loc1=3, loc2=4, fc="none", ec=GREY, lw=0.9,
                   ls=(0, (3, 2)))

    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_yticks([1, 5, 10, 15])
    style_axes(ax, "Max Dungeon Level", "Max Experience Level")
    ax.grid(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.175),
               ncol=3, fontsize=legendsize - 8, frameon=False,
               handlelength=2.2, handletextpad=.6, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.31)
    save(fig, NAME + ("_inset" if inset else ""), outdir=OUTDIR)


if __name__ == "__main__":
    render()
    render(inset=True)
