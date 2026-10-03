"""One real turn of the loop, taken verbatim from a rollout.

The abstract block diagram this replaced told the reader nothing they could not
guess. This shows the actual interface: the observation the model acted on, the
call it made, and what came back. Two of the paper's claims are visible in it --
one call advances a hundred game turns, and the reply withholds the map.

Source: base_r1, seed 3, call #42.

Typography: this figure used to opt out of the usetex pipeline, so its labels
were set in DejaVu Sans while every other figure was set in Helvetica. It now
runs through the same pipeline as the rest: labels in the house sans face at
house-ish sizes, and monospace reserved for the game's own bytes. LaTeX collapses
runs of spaces, which would destroy an ASCII map, so `tt()` re-spaces each line
with unbreakable spaces -- in a monospaced face those are exactly one column
wide -- and `block()` sets a whole map as a single LaTeX table so that every row
shares one left origin.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from style import HUMAN, AGENT, BAD, GREY, tex, save

NAME = "fig1_harness_loop"
DATA = "/root/overleaf/figs/turn_example.json"

# Proposals land beside the paper's figures without replacing them; point this
# at style.save's default (images/) to adopt the figure.
OUTDIR = "/root/overleaf/images/proposed"

# Sizes are the house sizes scaled to this figure's 12in width: headings sit
# where a 20pt label sits on a 7in figure, and the game's own text is set as
# small as it can be and stay readable.
HEAD, NOTE, CODE = 15.0, 12.0, 11.0
MONO, MONO_S, MONO_XS = 7.6, 7.4, 7.0
INK = "#16181d"
LW_BOX, LW_ARROW = 1.8, 2.2

TINT = {HUMAN: "#f4f8ff", AGENT: "#f5faf2", BAD: "#fff6f6"}


def tt(s):
    r"""A line of the game's own text, monospaced with its columns preserved.

    Two LaTeX details: runs of spaces collapse, so each space becomes an
    unbreakable space, which in a monospaced face is exactly one column wide;
    and `\$` pulls the dollar from the TS1 companion font, whose subset
    matplotlib writes as an unreadable content stream, so the typewriter font's
    own slot is asked for by number instead.
    """
    return (r"\texttt{" + tex(s).replace(" ", "~").replace(r"\$", r"{\char36}")
            + "}")


def render():
    d = json.load(open(DATA))
    rows = [r[:74] for r in d["map"][:14]]
    ret = ["[skill 'explore_level' ran 100 timesteps without",
           " interruption. GAME: You hear a door open.]",
           "",
           "=== SURROUNDINGS (hidden; call request_map for the",
           "    full map plus ADJACENT / VISIBLE FEATURES /",
           "    VISIBLE MONSTERS / MESSAGES) ===",
           "",
           "=== STATUS ===",
           "HP: 16/16  AC: 6  Dlvl: 4  Turn: 515  XP: 1"]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    def panel(x, w, y, h, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=0.9",
                                    fc=TINT[ec], ec=ec, lw=LW_BOX, zorder=2))

    def head(x, y, n, text, col):
        # The step number carries its weight through size and colour rather than
        # \textbf: matplotlib's subset of Helvetica Bold writes a content stream
        # Ghostscript refuses to read, and this figure has to survive every
        # viewer the paper passes through.
        ax.text(x, y, str(n), fontsize=HEAD + 3, color=col, va="bottom")
        ax.text(x + 3.0, y, tex(text), fontsize=HEAD, color=INK, va="bottom")

    def note(x, y, lines, col, size=NOTE):
        ax.text(x, y, "\n".join(rf"\textit{{{tex(l)}}}" for l in lines),
                fontsize=size, color=col, va="top", ha="left", linespacing=1.35)

    def block(x, y, lines, size, color=INK, stretch=0.9):
        # One LaTeX box, not one per line: matplotlib measures each rendered
        # line from its first glyph, which silently deletes the indentation an
        # ASCII map is made of. Inside a tabular the rows share one origin, so
        # the columns line up the way the game drew them.
        body = r"\\".join(tt(l) for l in lines)
        ax.text(x, y, r"{\renewcommand{\arraystretch}{%s}"
                      r"\begin{tabular}{@{}l@{}}%s\end{tabular}}" % (stretch, body),
                fontsize=size, va="top", ha="left", color=color)

    def arrow(p, q, col, lw=LW_ARROW):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                                     color=col, lw=lw, zorder=4,
                                     shrinkA=0, shrinkB=0))

    # 1 -- the observation the model acted on
    head(0.5, 33.4, "1", "what the model sees", HUMAN)
    panel(0.5, 46, 4.6, 27.9, HUMAN)
    block(1.6, 31.4, rows, MONO)
    block(1.6, 8.6, [d["status"]], MONO_S, color=GREY)

    # 2 -- the call
    head(50.0, 33.4, "2", "what it emits", AGENT)
    panel(50.0, 26, 25.2, 7.4, AGENT)
    block(51.5, 29.6, ["await nethack.np_explore_level()"], CODE)
    note(51.5, 27.0, ["one call, no arguments"], GREY)

    # 3 -- what came back
    head(50.0, 20.2, "3", "what comes back", BAD)
    panel(50.0, 47, 4.6, 13.6, BAD)
    block(51.5, 16.9, ret, MONO_XS, stretch=1.05)

    arrow((47.2, 28.9), (50.0, 28.9), AGENT)
    arrow((63.0, 25.2), (63.0, 18.2), BAD)

    # the loop back
    arrow((50.0, 11.3), (47.4, 11.3), GREY, lw=LW_BOX)
    for p, q in (((98.5, 11.3), (98.5, 1.4)), ((98.5, 1.4), (23.0, 1.4))):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-", color=GREY,
                                     lw=LW_BOX, zorder=1))
    arrow((23.0, 1.4), (23.0, 4.6), GREY, lw=LW_BOX)
    ax.text(60.0, 1.7, r"\textit{" + tex("next turn") + "}", fontsize=NOTE,
            color=GREY, ha="center", va="bottom")

    # the two facts this turn demonstrates
    note(78.5, 33.6, ["one LLM call advanced", "100 game turns"], AGENT)
    note(78.5, 25.2, ["and the reply withholds", "the map, so the next call",
                      "is made without it"], BAD)

    save(fig, NAME, outdir=OUTDIR)
