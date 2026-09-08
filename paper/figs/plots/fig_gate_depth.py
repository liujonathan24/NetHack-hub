"""The descent gate on a time axis: depth against LM turns, gate firings marked.

WHAT THE GATE IS. In the `p1_gate` arm the harness intercepts every descend and,
if the hero is below the median XL that human winners hold when leaving that
depth, returns an advisory line instead of descending: "you are XL x on Dlvl d.
Typical successful human runs reach XL n before leaving this depth. Repeat the
call to descend anyway." It fires at most once per depth and never blocks -- a
repeated call goes through.

WHY THE NORM CURVE IS NOT ON THE DEPTH PANEL. The gate's conditional is
`xl < norm_xl_for_leaving(dlvl)`, a relation between depth and EXPERIENCE. It has
no curve on a depth-against-time axis. What can be drawn instead is its
inversion, per rollout: given the hero's XL at turn t, the deepest level the gate
would let it leave is `max{d : norm(d) <= xl(t)}`. That is the PERMITTED DEPTH,
and it is exactly the boundary the gate enforces -- a rollout is gated whenever
its depth line sits above its own permitted-depth line. The lower panel draws
both as medians across the fifteen rollouts, which is the only way to show the
conditional without fifteen extra step lines.

EVERY ROLLOUT HERE DIED. Fifteen of fifteen, so the end of each line is a death,
not a truncation, and the ends are marked accordingly.

WHAT THE CROSSES SHOW. Filled = the model repeated the descend on its very next
call, overriding the advice (24 of 43). Open = it did something else first, the
outcome we count as backing off (19 of 43). Only 1 of those 19 went on to die on
the level it had been advised to farm, and it reached the advised XL in 0 of 43
firings, so the visual density of open crosses overstates compliance -- the
caption has to say so.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import AGENT, BAD, GREY, LIGHT, LW, legendsize, style_axes, save

NAME = "fig_gate_depth"
SRC = "/root/overleaf/figs/e15_gate.json"
OUTDIR = "/root/overleaf/images/proposed"


def _permitted(xl, norm):
    """Deepest level the gate would allow leaving at experience level `xl`."""
    ok = [d for d, n in norm if n <= xl]
    return max(ok) if ok else 1


def render():
    doc = json.load(open(SRC))
    norm = [(int(d), int(n)) for d, n in doc["norm"]]
    rolls = doc["rollouts"]

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(10, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12})

    tmax = max(p[0] for r in rolls for p in r["path"])

    # ---- top: depth against LM turns, one line per rollout ------------------
    for r in rolls:
        t = np.array([p[0] for p in r["path"]], float)
        d = np.array([p[1] for p in r["path"]], float)
        ax.plot(t, d, color=AGENT, lw=1.0, alpha=.55, zorder=2)
        if r["died"]:
            ax.plot(t[-1], d[-1], "x", color=BAD, ms=6, mew=1.6, zorder=4)

    # gate firings: open where the model backed off, filled where it overrode
    for r in rolls:
        for g in r["gates"]:
            if g.get("lm_turn") is None:
                continue
            if g["paused"]:
                ax.plot(g["lm_turn"], g["dlvl"], "x", color=GREY, ms=8,
                        mew=2.0, zorder=6)
            else:
                ax.plot(g["lm_turn"], g["dlvl"], "P", color=GREY, ms=7,
                        mew=0, zorder=6)

    ax.set_ylim(0.5, 14.5)
    ax.set_yticks([1, 3, 5, 7, 9, 11, 13])
    style_axes(ax, None, "Dungeon Level")

    # ---- bottom: the conditional, as depth reached vs depth permitted -------
    grid = np.arange(1, tmax + 1)
    D = np.full((len(rolls), len(grid)), np.nan)
    P = np.full((len(rolls), len(grid)), np.nan)
    for i, r in enumerate(rolls):
        t = np.array([p[0] for p in r["path"]])
        d = np.array([p[1] for p in r["path"]])
        x = np.array([p[2] for p in r["path"]])
        idx = np.searchsorted(t, grid, side="right") - 1
        live = (idx >= 0) & (grid <= t[-1])
        D[i, live] = d[np.clip(idx[live], 0, len(d) - 1)]
        P[i, live] = [_permitted(v, norm)
                      for v in x[np.clip(idx[live], 0, len(x) - 1)]]

    alive = np.sum(~np.isnan(D), axis=0)
    keep = alive >= 5
    bx.plot(grid[keep], np.nanmedian(D, axis=0)[keep], color=AGENT, lw=LW,
            label="depth reached (median)")
    bx.plot(grid[keep], np.nanmedian(P, axis=0)[keep], color=GREY, lw=LW,
            ls="--", label="depth the gate permits (median)")
    bx.fill_between(grid[keep], np.nanmedian(P, axis=0)[keep],
                    np.nanmedian(D, axis=0)[keep], color=BAD, alpha=.12,
                    zorder=1)
    bx.set_ylim(0.5, 14.5)
    bx.set_yticks([1, 5, 10])
    bx.set_xlim(1, tmax)
    style_axes(bx, "LM Turns", "Dlvl")
    bx.legend(fontsize=legendsize - 6, loc="upper left", frameon=False)

    handles = [
        Line2D([], [], color=AGENT, lw=1.4, label=rf"rollout ($n{{=}}{len(rolls)}$)"),
        Line2D([], [], color=GREY, lw=0, marker="P", ms=7,
               label="gate fired, advice overridden (24)"),
        Line2D([], [], color=GREY, lw=0, marker="x", ms=8, mew=2.0,
               label="gate fired, model backed off (19)"),
        Line2D([], [], color=BAD, lw=0, marker="x", ms=6, mew=1.6,
               label="death (15 of 15)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.085),
               ncol=4, fontsize=legendsize - 6, frameon=False,
               handletextpad=.6, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.20)
    save(fig, NAME, outdir=OUTDIR)


if __name__ == "__main__":
    render()
