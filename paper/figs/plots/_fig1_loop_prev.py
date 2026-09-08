"""One real turn of the loop, taken verbatim from a rollout.

The abstract block diagram this replaced told the reader nothing they could not
guess. This shows the actual interface: the observation the model acted on, the
call it made, and what came back. Two of the paper's claims are visible in it --
one call advances a hundred game turns, and the reply withholds the map.

Source: base_r1, seed 3, call #42.
"""
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import HUMAN, AGENT, BAD, GREY, save

NAME = "fig1_harness_loop"
DATA = "/root/overleaf/figs/turn_example.json"

MONO, CODE, LAB, NOTE = 7.0, 8.0, 11.0, 9.5
INK = "#16181d"


def render():
    d = json.load(open(DATA))
    rows = [r[:74] for r in d["map"][:14]]

    with mpl.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(12, 4.6))
        ax.set_xlim(0, 100); ax.set_ylim(0, 38); ax.axis("off")

        def panel(x, w, y, h, ec, fc):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0.4,rounding_size=0.9",
                fc=fc, ec=ec, lw=1.6, zorder=2))

        def head(x, y, n, text, col):
            ax.text(x, y, n, fontsize=LAB, color=col, fontweight="bold",
                    va="bottom", family="DejaVu Sans")
            ax.text(x + 3.0, y, text, fontsize=LAB, color=INK, va="bottom",
                    family="DejaVu Sans")

        def arrow(p, q, col, lw=2.0):
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>",
                mutation_scale=16, color=col, lw=lw, zorder=4,
                shrinkA=0, shrinkB=0))

        # 1 — the observation the model acted on
        head(0.5, 33.4, "1", "what the model sees", HUMAN)
        panel(0.5, 46, 4.6, 27.9, HUMAN, "#f7faff")
        ax.text(1.6, 31.2, "\n".join(rows), family="DejaVu Sans Mono",
                fontsize=MONO, va="top", ha="left", color=INK, linespacing=1.08)
        ax.text(1.6, 8.6, d["status"], family="DejaVu Sans Mono",
                fontsize=MONO - 0.6, va="top", ha="left", color=GREY)

        # 2 — the call
        head(50.0, 33.4, "2", "what it emits", AGENT)
        panel(50.0, 26, 25.2, 7.4, AGENT, "#f6fbf3")
        ax.text(51.5, 29.2, "await nethack.np_explore_level()",
                family="DejaVu Sans Mono", fontsize=CODE, va="top", color=INK)
        ax.text(51.5, 26.7, "one call, no arguments", family="DejaVu Sans",
                fontsize=NOTE, va="top", color=GREY, style="italic")

        # 3 — what came back
        head(50.0, 20.2, "3", "what comes back", BAD)
        panel(50.0, 47, 4.6, 13.6, BAD, "#fff7f7")
        ax.text(51.5, 16.4,
                "[skill 'explore_level' ran 100 timesteps without\n"
                " interruption. GAME: You hear a door open.]\n\n"
                "=== SURROUNDINGS (hidden; call request_map for the\n"
                "    full map plus ADJACENT / VISIBLE FEATURES /\n"
                "    VISIBLE MONSTERS / MESSAGES) ===\n\n"
                "=== STATUS ===\n"
                "HP: 16/16  AC: 6  Dlvl: 4  Turn: 515  XP: 1",
                family="DejaVu Sans Mono", fontsize=5.7, va="top",
                color=INK, linespacing=1.38)

        arrow((47.2, 28.9), (50.0, 28.9), AGENT)
        arrow((63.0, 25.2), (63.0, 18.2), BAD)

        # the loop back
        arrow((50.0, 11.3), (47.4, 11.3), GREY, lw=1.6)
        for p, q in (((98.5, 11.3), (98.5, 2.0)), ((98.5, 2.0), (23.0, 2.0))):
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-", color=GREY,
                                         lw=1.6, zorder=1))
        arrow((23.0, 2.0), (23.0, 4.6), GREY, lw=1.6)
        ax.text(60.0, 2.5, "next turn", fontsize=NOTE, color=GREY,
                ha="center", va="bottom", style="italic", family="DejaVu Sans")

        # the two facts this turn demonstrates
        ax.text(78.0, 33.4, "one LLM call advanced\n100 game turns",
                fontsize=NOTE, color=AGENT, va="top", ha="left",
                linespacing=1.35, family="DejaVu Sans")
        ax.text(78.0, 25.0, "and the reply withholds\nthe map, so the next call\nis made without it",
                fontsize=NOTE, color=BAD, va="top", ha="left",
                linespacing=1.35, family="DejaVu Sans")

        save(fig, NAME)
