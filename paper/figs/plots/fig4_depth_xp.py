"""The (depth, experience) plane, regraded under BALROG-min.

Background: a small set of thin light-grey labelled contours of the min field
min(depth percentile, experience percentile) at every (max dungeon level, max
experience level) cell, straight from the vendored achievement table -- a
nearest-lower-key lookup, not an interpolation, so the steps are honest. The
min rises only when BOTH axes rise, so its contours are L-shapes opening
toward the top-right; a reader can read each rollout's regraded BALROG-min
off the plane. The dashed grey curve is the NAO ascension experience-for-depth
norm.

WHAT CHANGED. The plane used to carry the public BALROG leaderboard as red
squares. Those are gone: the leaderboard is a different harness on a different
observation surface, so it could only ever be read as a rough backdrop, and the
regrading argument it supported now lives in the text. In its place the figure
carries the model-diversity arm -- four frontier models run through the SAME
uncapped base harness as our GLM-5.2 reference, 5 seeds x 3 repeats each, so
every point on the plane is now one of our own lives and the comparison across
series is like-for-like.

READING THE POINTS. A small marker is one rollout's (max Dlvl, max XL); the
large ringed marker of the same colour is that model's mean, with standard
errors on both axes. Rollouts land on integer cells and collide heavily (nine
of the 75 sit on Dlvl 5, XL 1 alone), so the small markers are dodged onto a
ring inside their cell. The dodge is cosmetic and deterministic; the mean
markers and every number in the text are computed on the undodged values.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from style import (AGENT, HUMAN, HUMAN_TOP, OTHER, BAD, GREY, LIGHT,
                   legendsize, LW, MS, ALPHA, style_axes, save)
from common import arm, balrog_table, human_norm, _pct

NAME = "fig4_depth_xp"

DIVERSITY = "/root/overleaf/figs/model_diversity.json"

LEVELS = [2, 5, 10, 30, 50]

#: The window the standalone figure uses. Every rollout we have ever run on the
#: base harness dies inside Dlvl 13 / XL 5, so the old 50x30 window spent nine
#: tenths of the canvas on emptiness -- unreadable the moment there are five
#: series instead of two. Callers that need the full plane (the Go-Explore
#: overlay, the merged plane-and-pace figure) pass their own.
XLIM, YLIM = (0.3, 20), (0.3, 13.6)

#: Colour AND marker per series, because five series have to survive greyscale
#: printing. Green stays AGENT -- these are all our rollouts -- and the four
#: comparison models take the remaining house colours in a fixed order so the
#: legend order and the plotting order never drift apart.
SERIES = [
    ("base",     AGENT,     "^", r"GLM-5.2 (ours)"),
    ("sol",      HUMAN,     "o", r"GPT-5.6 sol"),
    ("luna",     HUMAN_TOP, "D", r"GPT-5.6 luna"),
    ("gemini37", OTHER,     "s", r"Gemini 3.7 Flash"),
    ("qwen38",   BAD,       "v", r"Qwen3.8-Max"),
]


def _series_points():
    """(key, colour, marker, label, points) for each series, base arm first."""
    d = json.load(open(DIVERSITY))
    base = [(int(r["max_dlvl"]), int(r["max_xp"])) for r in arm("base")["rollouts"]]
    out = []
    for key, col, mk, lab in SERIES:
        pts = base if key == "base" else [tuple(p) for p in d[key]["points"]]
        out.append((key, col, mk, lab, pts))
    return out


def _dodge(series):
    """Fan colliding rollouts out onto a ring inside their integer cell.

    Returns, per series, the drawn (x, y) arrays. Points are grouped by cell
    across ALL series and ordered by series index, so a cell always reads in
    the legend's order and the layout is stable between renders.
    """
    cells = {}
    for si, (_, _, _, _, pts) in enumerate(series):
        for p in pts:
            cells.setdefault(p, []).append(si)
    drawn = [[] for _ in series]
    for (cx, cy), members in cells.items():
        members.sort()
        k = len(members)
        if k == 1:
            offs = [(0.0, 0.0)]
        elif k <= 6:
            r, ang = 0.36, np.pi / 2
            offs = [(r * np.cos(ang + 2 * np.pi * i / k),
                     r * np.sin(ang + 2 * np.pi * i / k)) for i in range(k)]
        else:
            r, ang, m = 0.48, np.pi / 2, k - 1
            offs = [(0.0, 0.0)] + [(r * np.cos(ang + 2 * np.pi * i / m),
                                    r * np.sin(ang + 2 * np.pi * i / m))
                                   for i in range(m)]
        for si, (ox, oy) in zip(members, offs):
            drawn[si].append((cx + ox, cy + oy))
    return drawn


def _label_spots(levels, xlim, ylim):
    """One clabel anchor per contour, on its horizontal run near the right edge."""
    _, xp = balrog_table()
    x = xlim[1] - 0.13 * (xlim[1] - xlim[0])
    spots = []
    for L in levels:
        ys = [k for k in sorted(xp) if xp[k] >= L]
        y = ys[0] if ys else ylim[1]
        spots.append((x, min(y - 0.35, ylim[1] - 0.5)))
    return spots


def draw(ax, legend=True, models=False, xlim=(1, 50), ylim=(1, 30)):
    """Draw the plane onto `ax`. Split out of render() so the merged
    plane-and-pace figure and the Go-Explore overlay reuse it verbatim instead
    of copying it. `models=False` keeps those callers on the two-series plane
    they were written against; the standalone figure passes True."""
    dl, xp = balrog_table()
    hx, hy, _ = human_norm(maxdepth=50, minn=10)

    ds = np.arange(1, 51)
    xs = np.arange(1, 31)
    dv = np.array([_pct(dl, d) for d in ds])
    xv = np.array([_pct(xp, x) for x in xs])
    M = np.minimum(dv[None, :], xv[:, None])          # (xl, dlvl)

    # only contours that actually cross the visible window get drawn, so a
    # zoomed call cannot ask clabel for a level that is not there
    win = M[(xs >= ylim[0]) & (xs <= ylim[1])][:, (ds >= xlim[0]) & (ds <= xlim[1])]
    levels = [L for L in LEVELS if win.min() < L < win.max()]
    cs = ax.contour(ds, xs, M, levels=levels, colors=LIGHT,
                    linewidths=0.9, zorder=1)
    ax.clabel(cs, fmt=lambda v: rf"{v:g}\%", fontsize=11, inline=True,
              inline_spacing=4, colors=GREY,
              manual=_label_spots(levels, xlim, ylim))

    ax.plot(hx, hy, color=GREY, lw=LW, ls="--", alpha=ALPHA, zorder=4,
            label="human norm")

    if models:
        series = _series_points()
        drawn = _dodge(series)
        for si, ((key, col, mk, lab, pts), dd) in enumerate(zip(series, drawn)):
            ax.plot([p[0] for p in dd], [p[1] for p in dd], marker=mk,
                    markersize=MS - 3, linestyle="none", color=col, alpha=.70,
                    markeredgecolor="white", markeredgewidth=.5, zorder=5)
            px = np.array([p[0] for p in pts], float)
            py = np.array([p[1] for p in pts], float)
            n = len(px)
            ax.errorbar(px.mean(), py.mean(),
                        xerr=px.std(ddof=1) / np.sqrt(n),
                        yerr=py.std(ddof=1) / np.sqrt(n),
                        color=col, ecolor=col, elinewidth=1.6, capsize=3,
                        marker=mk, markersize=MS - 2, markeredgecolor="black",
                        markeredgewidth=1.0, zorder=7 + len(SERIES) - si,
                        label=lab)
    else:
        rs = arm("base")["rollouts"]
        ax.plot([r["max_dlvl"] for r in rs], [r["max_xp"] for r in rs],
                marker="^", markersize=MS, linestyle="none", color=AGENT,
                alpha=ALPHA, zorder=5, label=rf"our rollouts ($n{{=}}{len(rs)}$)")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlim == (1, 50):
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    else:
        ax.set_xticks([x for x in (1, 5, 10, 15, 20, 25, 30) if xlim[0] <= x <= xlim[1]])
        ax.set_yticks([y for y in (1, 2, 4, 6, 8, 10, 12, 14) if ylim[0] <= y <= ylim[1]])
    style_axes(ax, "Max Dungeon Level", "Max Experience Level")
    ax.grid(False)
    if legend:
        ax.legend(fontsize=legendsize - 7, loc="upper left", framealpha=.95,
                  borderpad=.45, labelspacing=.32, handletextpad=.5,
                  handlelength=1.4, borderaxespad=.4)
    return ax


def render():
    fig, ax = plt.subplots(figsize=(7, 4.8))
    draw(ax, models=True, xlim=XLIM, ylim=YLIM)
    # a raster twin of the vector figure, for eyeballing the marker dodge
    fig.savefig(f"/root/overleaf/images/{NAME}.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"  wrote {NAME}.png")
    save(fig, NAME)
