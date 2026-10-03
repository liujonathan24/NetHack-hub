"""Figure 1: the system, vertical.

TOP (three quarters): one TRACE -- the game engine on the left, showing a real
observation; the Prime Agent harness on the right, showing what it is made of
(an LLM, memory, skills, sub-agents); two curved arrows between them,
observation one way and skill call the other.

BOTTOM (one quarter): what sits between traces. A third of the width is a
person editing the harness by hand. Two thirds is an orchestrator LLM that can
do one of two things: rewrite the harness (memories, notes, skills), or pick a
checkpoint from the archive for the next attempt to resume from.

Pictures over words: every box carries an icon and a name, and nothing else.
Icons are drawn here with matplotlib primitives so the figure has no external
image dependency. No bold anywhere: see style.py.
"""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

from style import GREY, tex, save

NAME = "fig1_system"
OUTDIR = "/root/overleaf/images/proposed"
DATA = "/root/overleaf/figs/turn_example.json"

INK = "#16181d"
LINE = "#8a8f98"          # box strokes
FILL = "#fbfbfc"          # box fills
ENGINE = "#5b7fb5"        # one accent per side of the trace
AGENT = "#5a9a4c"
HUMAN = "#c98a2b"
ORCH = "#7b6bb0"
TINT = {ENGINE: "#eef3fa", AGENT: "#eef6ec", HUMAN: "#fbf3e6", ORCH: "#f1eef8",
        LINE: FILL}

HEAD, BODY, SMALL, MONO = 13.0, 10.5, 9.5, 6.5


def box(ax, x, y, w, h, color, title=None, lw=1.6, r=0.14, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=TINT[color], ec=color, lw=lw,
                                ls=ls, boxstyle=f"round,pad=0,rounding_size={r}",
                                zorder=z))
    if title:
        ax.text(x + 0.14, y + h - 0.12, title, ha="left", va="top",
                fontsize=HEAD, color=INK, zorder=z + 1)


def arrow(ax, p, q, color, rad=0.0, lw=2.0, ls="-", z=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, ls=ls, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def tt(s):
    return r"\texttt{" + tex(s).replace(" ", "~").replace(r"\$", r"{\char36}") + "}"


# ------------------------------------------------------------------- icons --

def chip(ax, cx, cy, s, color, label="LLM"):
    """A processor: rounded square with pins, the model's icon everywhere."""
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s, fc="white",
                                ec=color, lw=1.6, boxstyle="round,pad=0,rounding_size=0.06",
                                zorder=5))
    for k in (-0.5, 0, 0.5):
        for dx, dy in ((k * s * 0.6, s * 0.5), (k * s * 0.6, -s * 0.5)):
            ax.plot([cx + dx, cx + dx], [cy + dy, cy + dy + (0.12 if dy > 0 else -0.12)],
                    color=color, lw=1.6, zorder=4)
        for dx, dy in ((s * 0.5, k * s * 0.6), (-s * 0.5, k * s * 0.6)):
            ax.plot([cx + dx, cx + dx + (0.12 if dx > 0 else -0.12)], [cy + dy, cy + dy],
                    color=color, lw=1.6, zorder=4)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=SMALL + 1,
            color=color, zorder=6)


def person(ax, cx, cy, s, color):
    ax.add_patch(Circle((cx, cy + s * 0.55), s * 0.22, fc="white", ec=color, lw=1.6, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - s * 0.36, cy - s * 0.35), s * 0.72, s * 0.6,
                                fc="white", ec=color, lw=1.6,
                                boxstyle="round,pad=0,rounding_size=0.18", zorder=5))


def cards(ax, x, y, w, h, color, n=3):
    """Memory: a small stack of cards."""
    for i in range(n):
        ax.add_patch(FancyBboxPatch((x + 0.06 * i, y + 0.05 * i), w, h, fc="white",
                                    ec=color, lw=1.3, boxstyle="round,pad=0,rounding_size=0.04",
                                    zorder=5 + i))


def lines_icon(ax, x, y, w, h, color, n=3):
    """Skills: a document with lines."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc="white", ec=color, lw=1.3,
                                boxstyle="round,pad=0,rounding_size=0.04", zorder=5))
    for i in range(n):
        yy = y + h * (0.75 - 0.25 * i)
        ax.plot([x + 0.1, x + w * (0.85 - 0.15 * (i % 2))], [yy, yy], color=color,
                lw=1.3, zorder=6)


def tree(ax, x0, y0, color, sx=1.0, sy=1.0):
    """A checkpoint archive. Every fork is symmetric about its parent, but the
    tree is not balanced: the lower branch has been expanded three more times
    while the upper one stopped after one. Open nodes were never expanded; the
    ringed one is where the next attempt resumes."""
    nodes = {"r": (0.00, 0.30), "a": (0.25, 0.30),
             "b": (0.50, 0.14), "d": (0.50, 0.46),
             "c": (0.75, 0.14), "g": (0.75, 0.37), "h": (0.75, 0.55),
             "f": (1.00, 0.06), "e": (1.00, 0.22), "j": (1.00, 0.37),
             "k": (1.25, 0.22),
             "m": (1.50, 0.14), "n": (1.50, 0.30)}
    edges = [("r", "a"), ("a", "b"), ("a", "d"), ("b", "c"), ("d", "g"), ("d", "h"),
             ("c", "f"), ("c", "e"), ("g", "j"), ("e", "k"), ("k", "m"), ("k", "n")]
    unvisited = {"f", "h", "j", "m"}
    P = {k: (x0 + nx * sx, y0 + ny * sy) for k, (nx, ny) in nodes.items()}
    for u, v in edges:
        ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=color, lw=1.3, zorder=4)
    for k, (px, py) in P.items():
        ax.add_patch(Circle((px, py), 0.04, fc="white" if k in unvisited else color,
                            ec=color, lw=1.3, zorder=5))
    gx, gy = P["n"]
    ax.add_patch(Circle((gx, gy), 0.08, fc="none", ec=color, lw=1.5,
                        ls=(0, (2, 1.5)), zorder=5))


def glyph_icon(ax, x, y, color):
    """Engine edits: a scrap of dungeon, drawn in the map's own alphabet."""
    with mpl.rc_context({"text.usetex": False, "font.family": "monospace"}):
        for i, ln in enumerate(["--+--", "|.>.|", "-----"]):
            ax.text(x, y + 0.36 - 0.18 * i, ln, ha="center", va="center",
                    fontsize=8, color=color, family="monospace", zorder=5)


# ------------------------------------------------------------------ figure --

def observation(d):
    """The map as the model saw it. Every stored row carries a trailing '|'
    from the renderer's column separator (room walls show as '||'); drop it."""
    return [r[:-1] if r.endswith("|") else r for r in d["map"]]


def subbox(ax, x, y, w, h, name, icon, color):
    """A named part with its icon above the name, centred."""
    box(ax, x, y, w, h, LINE, r=0.08, lw=1.1, z=3)
    ax.text(x + w / 2, y + 0.2, name, ha="center", va="center", fontsize=BODY,
            color=INK, zorder=5)
    icon(ax, x + w / 2, y + 0.42 + (h - 0.42) / 2 - 0.05, color)


def render():
    d = json.load(open(DATA))
    fig, ax = plt.subplots(figsize=(12.5, 7.9))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.9)
    ax.axis("off")

    # ---- the rollout: full width, engine left, agent right ---------------------
    CY, CH = 0.2, 2.25                      # the controllers box
    TY, TH = CY + CH + 0.8, 4.5
    box(ax, 0.3, TY, 11.9, TH, LINE, lw=1.4, ls=(0, (4, 3)))
    ax.text(0.44, TY + TH - 0.1, "NetHack rollout", ha="left", va="top",
            fontsize=HEAD, color=GREY)

    EX, EY, EW, EH = 0.55, TY + 0.25, 5.9, TH - 0.75
    box(ax, EX, EY, EW, EH, ENGINE, "game engine (vision on)")
    rows = observation(d)
    px, py, pw, ph = EX + 0.2, EY + 0.55, EW - 0.4, EH - 1.05
    ax.add_patch(Rectangle((px, py), pw, ph, fc="white", ec=LINE, lw=0.8, zorder=3))
    # The map is raw ASCII and goes around the usetex pipeline, as figA1 and
    # figA2 do: LaTeX collapses spaces and eats a trailing '|' at the end of a
    # \texttt group, which is exactly where NetHack draws its right-hand walls.
    step = (ph - 0.16) / len(rows)
    with mpl.rc_context({"text.usetex": False, "font.family": "monospace"}):
        for i, r in enumerate(rows):
            ax.text(px + 0.1, py + ph - 0.08 - step * i, r, ha="left", va="top",
                    fontsize=MONO, color=INK, zorder=4, family="monospace")
        ax.text(px + 0.1, py - 0.1, d["status"], ha="left", va="top",
                fontsize=MONO + 0.5, color=GREY, zorder=4, family="monospace")

    AXX, AY, AW, AH = 7.65, EY, 4.25, EH
    box(ax, AXX, AY, AW, AH, AGENT, "Prime Agent")
    chip(ax, AXX + AW / 2, AY + AH - 1.0, 0.66, AGENT)
    sh, gap = 0.58, 0.1
    for i, name in enumerate(["memory", "skills", "sub-agents"]):
        y = AY + 0.22 + (2 - i) * (sh + gap)
        box(ax, AXX + 0.25, y, AW - 0.5, sh, LINE, r=0.08, lw=1.1, z=3)
        ax.text(AXX + 0.45, y + sh / 2, name, ha="left", va="center",
                fontsize=BODY, color=INK, zorder=5)
        cx = AXX + AW - 0.7
        if name == "memory":
            cards(ax, cx - 0.3, y + 0.1, 0.44, 0.3, AGENT)
        elif name == "skills":
            lines_icon(ax, cx - 0.18, y + 0.08, 0.36, 0.42, AGENT)
        else:
            chip(ax, cx - 0.22, y + sh / 2, 0.26, AGENT, label="")
            chip(ax, cx + 0.18, y + sh / 2, 0.26, AGENT, label="")

    gx = (EX + EW + AXX) / 2
    arrow(ax, (EX + EW, EY + EH - 0.9), (AXX, AY + AH - 0.9), INK, rad=-0.5, lw=2.2)
    ax.text(gx, EY + EH - 0.5, "observation", ha="center", va="bottom",
            fontsize=SMALL, color=INK)
    arrow(ax, (AXX, AY + 0.9), (EX + EW, EY + 0.9), INK, rad=-0.5, lw=2.2)
    ax.text(gx, EY + 0.5, "skill call", ha="center", va="top",
            fontsize=SMALL, color=INK)

    # ---- controllers: human (1/3), orchestrator (2/3) --------------------------
    box(ax, 0.3, CY, 11.9, CH, LINE, lw=1.4, ls=(0, (4, 3)))
    ax.text(0.44, CY + CH - 0.1, "controllers", ha="left", va="top", fontsize=HEAD,
            color=GREY)
    BY, BH = CY + 0.15, CH - 0.6
    HW = 4.4
    box(ax, 0.55, BY, HW, BH, HUMAN, "human")
    person(ax, 0.55 + 0.6, BY + 0.55, 0.62, HUMAN)
    hx0, hw, hy, hh = 0.55 + 1.2, (HW - 1.2 - 0.35) / 2, BY + 0.15, BH - 0.6
    subbox(ax, hx0, hy, hw, hh, "harness edits",
           lambda a, cx, cy, c: lines_icon(a, cx - 0.17, cy - 0.2, 0.34, 0.4, c), HUMAN)
    subbox(ax, hx0 + hw + 0.15, hy, hw, hh, "engine edits",
           lambda a, cx, cy, c: glyph_icon(a, cx, cy - 0.2, c), HUMAN)

    OX, OW = 0.55 + HW + 0.25, 11.9 - 0.5 - HW - 0.25
    box(ax, OX, BY, OW, BH, ORCH, "orchestrator")
    chip(ax, OX + 0.7, BY + 0.15 + (BH - 0.6) / 2, 0.62, ORCH)
    sx0, sw, sy, sh = OX + 1.4, (OW - 1.4 - 0.45) / 2, BY + 0.15, BH - 0.6
    subbox(ax, sx0, sy, sw, sh, "sampling + checkpointing",
           lambda a, cx, cy, c: tree(a, cx - 0.72, cy - 0.38, c, sx=0.96, sy=1.1), ORCH)
    subbox(ax, sx0 + sw + 0.2, sy, sw, sh, "harness optimization",
           lambda a, cx, cy, c: (cards(a, cx - 0.95, cy - 0.18, 0.44, 0.3, c),
                                 lines_icon(a, cx - 0.17, cy - 0.22, 0.34, 0.42, c),
                                 chip(a, cx + 0.7, cy, 0.28, c, label="")), ORCH)

    # the outer loop: controllers evaluate by launching a rollout, and read
    # what it leaves behind
    mid = 0.3 + 11.9 / 2
    arrow(ax, (mid - 0.9, CY + CH), (mid - 0.9, TY), INK, lw=2.2)
    ax.text(mid - 1.05, (CY + CH + TY) / 2, "evaluate", ha="right", va="center",
            fontsize=SMALL, color=INK)
    arrow(ax, (mid + 0.9, TY), (mid + 0.9, CY + CH), INK, lw=2.2)
    ax.text(mid + 1.05, (CY + CH + TY) / 2, "traces, checkpoints", ha="left",
            va="center", fontsize=SMALL, color=INK)
    save(fig, NAME, outdir=OUTDIR)


if __name__ == "__main__":
    render()
