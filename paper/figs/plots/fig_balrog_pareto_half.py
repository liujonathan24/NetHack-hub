"""Half-width Pareto figure: equal-weight mean over MiniHack, TextWorld, Crafter and
NetHack (BAL_max / 100) against cost summed over the four suites, each suite's cost
normalised to the leaderboard's episode count (see fig_balrog_aggregate.py)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from style import AGENT, blue, GREY, LIGHT, SIZE, labelsize, ticksize, style_axes, save, tex
from plots.fig_balrog_pareto import SHORT, frontier, SRC

OUTDIR = "/root/overleaf/final-drafting/v33-claude/images/proposed"
COL = {"base": AGENT, "pae": blue, "lb": "#9A9A9A"}
NAME = {"base": "GLM-5.2 base", "pae": "GLM-5.2 PAE"}
#: hand placement where two leaderboard dots sit almost on top of each other
PIN = {"GLM-5.2 base": (-8, 6, "right", "bottom"), "Mistral-Nemo": (6, -6, "left", "top"), "Gem-3.1-Pro-T": (8, 0, "left", "center"), "Opus 4.5-T": (8, 0, "left", "center"),
       "Opus 4.5": (8, -3, "left", "top"), "Gem-3.1-Pro": (8, 0, "left", "center"),
       "Gem-3-Pro": (-7, 5, "right", "bottom"), "Gem-3-Flash": (-6, 6, "right", "bottom"),
       "Gem-2.5-Pro": (8, -4, "left", "top"), "Haiku 4.5": (-8, 0, "right", "center"),
       "GPT-5 (minimal)": (8, 0, "left", "center"), "Gem-2.5-Flash": (6, -6, "left", "top")}


def label(p):
    return NAME.get(p["kind"], SHORT.get(p["label"], p["label"]))


def main(src=None, name="fig_balrog_pareto_half", wide="", pins=""):
    global PIN
    if pins:
        PIN = json.load(open(pins))
    pts = json.load(open(src)) if src else json.load(open(f"{SRC}/aggregate.json"))["all4_balmax"]
    fig, ax = plt.subplots(figsize=(12, 4.5) if wide else SIZE)
    ax.set_xscale("log")
    xs = [p["cost"] for p in pts]
    ax.set_xlim(min(xs) * 0.5, max(xs) * (3.0 if wide else 6.0))
    ys = [p["prog"] for p in pts]
    ax.set_ylim(max(0, min(ys) - 0.06), max(ys) + 0.06)
    fr = frontier(pts)
    ax.plot([p["cost"] for p in fr], [p["prog"] for p in fr], color="#333333", ls="--",
            lw=2.0, zorder=3)
    for p in pts:
        ours = p["kind"] != "lb"
        ax.scatter([p["cost"]], [p["prog"]], s=150 if ours else 70, color=COL[p["kind"]],
                   edgecolor="white", linewidth=1.0, zorder=5)
    style_axes(ax, "Total cost (USD)", "Mean progression")

    # label placement: avoid other labels, every dot, and the frontier line
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tr = ax.transData
    ab = ax.get_window_extent(r)
    obst = []
    for p in pts:
        x, y = tr.transform((p["cost"], p["prog"]))
        obst.append((x - 7, y - 7, x + 7, y + 7))
    for a, b in zip(fr, fr[1:]):
        (x0, y0), (x1, y1) = tr.transform((a["cost"], a["prog"])), tr.transform((b["cost"], b["prog"]))
        for t in np.linspace(0, 1, 40):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            obst.append((x - 2, y - 2, x + 2, y + 2))
    hit = lambda a, b: a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]
    offs = []
    for rad in (9, 15, 22, 30, 40, 52):
        for ang in range(0, 360, 20):
            dx, dy = rad * np.cos(np.radians(ang)), rad * np.sin(np.radians(ang))
            ha = "left" if dx > 3 else "right" if dx < -3 else "center"
            va = "bottom" if dy > 3 else "top" if dy < -3 else "center"
            offs.append((dx, dy, ha, va))
    placed = []
    for p in sorted(pts, key=lambda q: (q["kind"] == "lb", label(q) not in PIN)):
        ours = p["kind"] != "lb"
        kw = dict(textcoords="offset points", fontsize=13 if ours else 11,
                  color=COL[p["kind"]] if ours else GREY, zorder=6)
        ok = False
        lab = label(p)
        for dx, dy, ha, va in ([PIN[lab]] if lab in PIN else offs):
            t = ax.annotate(tex(label(p)), (p["cost"], p["prog"]), xytext=(dx, dy), ha=ha, va=va, **kw)
            bb = t.get_window_extent(r)
            b = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            inside = b[0] >= ab.x0 and b[2] <= ab.x1 and b[1] >= ab.y0 and b[3] <= ab.y1
            if lab in PIN or (inside and not any(hit(b, q) for q in placed + obst)):
                if abs(dx) > 12 or abs(dy) > 12:
                    t.remove()
                    t = ax.annotate(tex(label(p)), (p["cost"], p["prog"]), xytext=(dx, dy), ha=ha,
                                    va=va, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8, shrinkA=1, shrinkB=4), **kw)
                placed.append(b)
                ok = True
                break
            t.remove()
        if not ok:
            print("  UNPLACED", label(p), tr.transform((p["cost"], p["prog"])), ab)
    save(fig, name, outdir=OUTDIR)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
