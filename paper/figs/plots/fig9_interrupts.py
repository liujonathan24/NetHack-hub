"""What stops a skill mid-execution, base harness against the hand-engineered arm.

The hand-engineered arm carries netplay_telemetry, the interrupt severity filter, so
this is the direct read on whether that flag does what it was built to do.
"""
import numpy as np
import matplotlib.pyplot as plt
from style import (AGENT, OTHER, legendsize, ticksize, ALPHA, style_axes, save)
from common import arm, arm_label

NAME = "fig9_interrupts"
ORDER = ["item / monster appeared", "gold", "corpse", "statue", "boulder",
         "no valid path", "level change", "HP change", "other"]


def render():
    counts, totals = {}, {}
    for a in ("base", "human"):
        c = arm(a)["interrupt_counts"]
        counts[a] = c
        totals[a] = max(1, sum(c.values()))

    cats = [k for k in ORDER if any(counts[a].get(k) for a in counts)][::-1]
    y = np.arange(len(cats))
    h = 0.38

    fig, ax = plt.subplots(figsize=(7, 5.8))
    for off, a, col in [(+h / 2, "base", AGENT), (-h / 2, "human", OTHER)]:
        share = [100 * counts[a].get(k, 0) / totals[a] for k in cats]
        ax.barh(y + off, share, height=h, color=col, alpha=ALPHA,
                label=arm_label(a))
        for yy, v in zip(y + off, share):
            if v > 0.4:
                ax.text(v + 0.7, yy, f"{v:.0f}\\%", va="center", fontsize=13)

    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlim(0, max(100 * counts[a].get(k, 0) / totals[a]
                       for a in counts for k in cats) * 1.22)
    ax.set_ylim(-.6, len(cats) - .4)
    style_axes(ax, "Share of Skill Interruptions (\\%)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=legendsize, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, 1.13), columnspacing=1.2)
    save(fig, NAME)
