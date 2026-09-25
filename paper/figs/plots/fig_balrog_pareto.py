"""BALROG transfer: cost against performance, with Pareto frontiers.

Three PDFs, all from the JSON the review artifacts were built from (no new runs):

  fig_balrog_suites.pdf   -- one panel per benchmark, averaged over its games:
                             MiniHack (8 tasks, Boxoban entered at 0), TextWorld
                             (3 games), Crafter, NetHack BAL_max and NetHack BAL_min.
  fig_balrog_pergame.pdf  -- one panel per game, with our best-of-k curve (k=1..10).
  fig_balrog_bars.pdf     -- per-game mean progression and mean cost per run for
                             base (attempt 1), PAE (best of 10) and the control
                             (best of 10 fresh episodes, no resumption).

COST. Our MiniHack / TextWorld / Crafter arms are provider-reported billing
(`tokens.billed_by_provider_usd`). Our NetHack arms have no billing field; they are
priced at GLM-5.2 $1.54/M fresh input, $0.154/M cached input, $4.84/M output plus
the orchestrator's provider-reported cost -- that card reproduces the three
wallet-measured 30-attempt PAE runs within 6%. Leaderboard points are list price
from each submission's published token totals (no cache split is published).

LABELS are drawn in the plot, not in a legend: leaderboard models are context and
share one muted marker, our arms take the paper's colours (base = AGENT green,
PAE = blue as in fig_main_min, control = purple).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import (AGENT, blue, purple, GREY, LIGHT, LW, labelsize, ticksize,
                   legendsize, style_axes, save, tex)

OUTDIR = "/root/overleaf/final-drafting/v33-claude/images/proposed"
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "balrog")

COL = {"base": AGENT, "pae": blue, "ctl": purple, "lb": "#9A9A9A"}
FRONT = "#333333"
BOX = dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85)
SHORT = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5": "Opus 5", "gpt-5.6-sol": "Sol",
         "gemini-3.1-pro-preview-thinking": "Gem-3.1-Pro-T", "gemini-3.1-pro-preview": "Gem-3.1-Pro",
         "gpt-5.6-terra": "Terra", "gemini-3-flash-preview": "Gem-3-Flash", "gpt-5.6-luna": "Luna",
         "gemini-2.5-pro-exp-03-25": "Gem-2.5-Pro", "gemini-2.5-flash": "Gem-2.5-Flash"}


def short(p):
    if p["kind"] == "base":
        return "Base"
    if p["kind"] == "pae":
        return "PAE"
    if p["kind"] == "ctl":
        return "Control"
    return SHORT.get(p["label"], p["label"])


def frontier(pts):
    fr, best = [], -1.0
    for p in sorted(pts, key=lambda q: q["cost"]):
        if p["prog"] > best:
            fr.append(p)
            best = p["prog"]
    return fr


def place_labels(ax, pts, fs):
    """Greedy in display space: try offsets around each dot, keep the first that
    overlaps neither a placed label nor any dot. Our arms are placed first."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tr = ax.transData
    ab = ax.get_window_extent(r)
    dots = []
    for p in pts:
        x, y = tr.transform((p["cost"], p["prog"]))
        dots.append((x - 5, y - 5, x + 5, y + 5))
    placed = []
    hit = lambda a, b: a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]
    offs = [(7, 0, "left", "center"), (-7, 0, "right", "center"), (0, 8, "center", "bottom"),
            (0, -8, "center", "top"), (7, 7, "left", "bottom"), (-7, 7, "right", "bottom"),
            (7, -7, "left", "top"), (-7, -7, "right", "top"), (0, 18, "center", "bottom"),
            (0, -18, "center", "top"), (12, 16, "left", "bottom"), (-12, 16, "right", "bottom"),
            (12, -16, "left", "top"), (-12, -16, "right", "top")]
    order = sorted(pts, key=lambda p: p["kind"] == "lb")
    for p in order:
        ours = p["kind"] != "lb"
        colr = COL[p["kind"]] if ours else GREY
        done = None
        for dx, dy, ha, va in offs:
            t = ax.annotate(tex(short(p)), (p["cost"], p["prog"]), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va=va, fontsize=fs,
                            color=colr, zorder=6, bbox=BOX)
            bb = t.get_window_extent(r)
            b = (bb.x0, bb.y0, bb.x1, bb.y1)
            inside = b[0] >= ab.x0 and b[2] <= ab.x1 and b[1] >= ab.y0 and b[3] <= ab.y1
            if inside and not any(hit(b, q) for q in placed) and not any(hit(b, q) for q in dots):
                done = b
                if abs(dy) > 12:
                    t.remove()
                    t = ax.annotate(tex(short(p)), (p["cost"], p["prog"]), xytext=(dx, dy),
                                    textcoords="offset points", ha=ha, va=va, fontsize=fs,
                                    color=colr, zorder=6, bbox=BOX,
                                    arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.8))
                break
            t.remove()
        if done is None:  # nothing free: fall back to the right-hand slot
            t = ax.annotate(tex(short(p)), (p["cost"], p["prog"]), xytext=(7, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=fs, color=colr, zorder=6, bbox=BOX)
            bb = t.get_window_extent(r)
            done = (bb.x0, bb.y0, bb.x1, bb.y1)
        placed.append(done)


def scatter_panel(ax, pts, title, ylabel, curve=None, fs=11, pct=False):
    xs = [p["cost"] for p in pts] + [c["cost"] for c in (curve or [])]
    ax.set_xscale("log")
    ax.set_xlim(min(xs) * 0.45, max(xs) * 2.6)
    ymax = max(p["prog"] for p in pts)
    ymax = 1.0 if not pct else ymax
    ax.set_ylim(-0.06 * ymax, ymax * 1.14)
    if curve and len(curve) > 1:
        ax.plot([c["cost"] for c in curve], [c["prog"] for c in curve], color=blue,
                lw=1.4, alpha=0.45, zorder=2)
        ax.scatter([c["cost"] for c in curve], [c["prog"] for c in curve], s=10,
                   color=blue, alpha=0.5, zorder=2)
    fr = frontier(pts)
    if len(fr) > 1:
        ax.plot([p["cost"] for p in fr], [p["prog"] for p in fr], color=FRONT, ls="--",
                lw=1.4, zorder=3)
    for p in pts:
        ours = p["kind"] != "lb"
        ax.scatter([p["cost"]], [p["prog"]], s=90 if ours else 42, color=COL[p["kind"]],
                   edgecolor="white", linewidth=0.8, zorder=5 if ours else 4,
                   marker="o" if p["kind"] != "ctl" else "D")
    style_axes(ax, None, ylabel, None)
    ax.set_title(title, fontsize=labelsize - 4)
    ax.tick_params(axis="both", labelsize=ticksize - 4)
    place_labels(ax, pts, fs)


def suites():
    D = json.load(open(f"{SRC}/fig3/suites.json"))
    names = [("MiniHack (8 tasks)", "MiniHack (8-task mean, Boxoban=0)", "Progression", False),
             ("TextWorld (3 games)", "TextWorld (3-game mean)", None, False),
             ("Crafter", "Crafter (10 seeds)", None, False),
             (r"NetHack BAL$_{\max}$", "NetHack (BALmax %)", r"BAL (\%)", True),
             (r"NetHack BAL$_{\min}$", "NetHack (BALmin %)", None, True)]
    fig, axs = plt.subplots(1, 5, figsize=(24, 5.0))
    for ax, (title, key, yl, pct) in zip(axs, names):
        scatter_panel(ax, D[key]["pts"], title, yl, fs=10.5, pct=pct)
    for ax in axs:
        ax.set_xlabel(r"Cost (USD, log)", fontsize=labelsize - 5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_balrog_suites", outdir=OUTDIR)


def pergame():
    D = json.load(open(f"{SRC}/plotdata4.json"))["games"]
    order = ["crafter", "the_cooking_game", "treasure_hunter", "coin_collector",
             "MiniHack-CorridorBattle-Dark", "MiniHack-Corridor-R3", "MiniHack-MazeWalk-9x9",
             "MiniHack-MazeWalk-15x15", "MiniHack-Quest-Easy", "MiniHack-Boxoban-Medium",
             "MiniHack-Boxoban-Hard"]
    fig, axs = plt.subplots(3, 4, figsize=(22, 15))
    for i, (ax, g) in enumerate(zip(axs.flat, order)):
        title = g.replace("MiniHack-", "MH ").replace("_", r"\_")
        if g.startswith("MiniHack-Boxoban"):
            title += " (ours at 0)"
        scatter_panel(ax, D[g]["pts"], title, "Progression" if i % 4 == 0 else None,
                      curve=D[g].get("curve"), fs=10.5)
        if i >= 7:
            ax.set_xlabel("Cost (USD, log)", fontsize=labelsize - 5)
    lax = axs.flat[-1]
    lax.axis("off")
    hs = [Line2D([], [], marker="o", ls="", ms=10, color=COL["base"], label="Base (attempt 1)"),
          Line2D([], [], marker="o", ls="", ms=10, color=COL["pae"], label="PAE (best of 10)"),
          Line2D([], [], marker="D", ls="", ms=9, color=COL["ctl"], label="Control (10 fresh episodes)"),
          Line2D([], [], color=blue, alpha=0.5, lw=1.4, marker="o", ms=4, label="PAE best of $k$, $k=1..10$"),
          Line2D([], [], marker="o", ls="", ms=7, color=COL["lb"], label="BALROG leaderboard model"),
          Line2D([], [], color=FRONT, ls="--", lw=1.4, label="Pareto frontier")]
    lax.legend(handles=hs, loc="center", fontsize=legendsize - 4, frameon=False)
    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    save(fig, "fig_balrog_pergame", outdir=OUTDIR)


def bars():
    P = [r for r in json.load(open(f"{SRC}/pergame.json")) if r.get("n")]
    names = [r["game"].replace("MiniHack ", "MH ").replace("TextWorld ", "TW ").replace("_", r"\_")
             for r in P]
    y = np.arange(len(P))[::-1]
    h = 0.26
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7.2), sharey=True)
    for off, key, kc, lab in [(h, "base", "base", "Base (attempt 1)"), (0, "pae", "pae", "PAE (best of 10)"),
                              (-h, "ctl", "ctl", "Control (10 fresh episodes)")]:
        pv = [r.get(f"{key}_prog", np.nan) for r in P]
        cv = [r.get(f"{key}_cost_per_run", np.nan) for r in P]
        a.barh(y + off, pv, height=h, color=COL[kc], label=lab, edgecolor="white", linewidth=0.6)
        b.barh(y + off, cv, height=h, color=COL[kc], edgecolor="white", linewidth=0.6)
    a.set_yticks(y)
    a.set_yticklabels(names)
    a.set_xlim(0, 1.05)
    b.set_xscale("log")
    style_axes(a, "Mean progression", None)
    style_axes(b, "Mean cost per run (USD, log)", None)
    for ax in (a, b):
        ax.tick_params(axis="both", labelsize=ticksize - 2)
    a.legend(loc="lower center", bbox_to_anchor=(1.0, 1.01), ncol=3, fontsize=legendsize - 5,
             frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_balrog_bars", outdir=OUTDIR)


if __name__ == "__main__":
    suites()
    pergame()
    bars()
