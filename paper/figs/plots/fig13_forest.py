"""Every intervention's effect on BALROG, with the reported interval."""
import numpy as np
import matplotlib.pyplot as plt
from style import (HUMAN, AGENT, OTHER, GREY, legendsize, ticksize, LW, MS,
                   ALPHA, SIZE, style_axes, save)
from common import arm_balrog

NAME = "fig13_forest"
CTRL = 3.06
ROWS = [                          # label, arm mean, lo, hi   (None = not yet run)
    ("Hand-engineered flags",   "measured", None, None),
    ("Doors unlocked",          4.08, -2.43,  5.21),
    ("Room density 0.25",       4.48, -2.73,  4.92),
    ("Room density 0.10",       5.70, -3.00, 14.01),
    ("Room density 0.025",      4.35, -3.10, 10.81),
    ("Continual memory",        None, None, None),
    ("Continual $+$ wiki",      None, None, None),
    ("Render policy (V1)",      None, None, None),
    ("State encoding (V2)",     None, None, None),
    ("Crisis directive (P2)",   None, None, None),
    ("Rollback (P3)",           None, None, None),
][::-1]
DELTA = {"Continual memory": -0.20, "Continual $+$ wiki": 0.69}


def render():
    import numpy as np
    b = np.array(arm_balrog("base"), float)
    h = np.array(arm_balrog("human"), float)
    d = h.mean() - b.mean()
    se = np.sqrt(b.var(ddof=1) / len(b) + h.var(ddof=1) / len(h))
    measured = (d, d - 1.96 * se, d + 1.96 * se)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.axvline(0, color=GREY, lw=2.0, ls="--")
    for i, (lab, mean, lo, hi) in enumerate(ROWS):
        if mean == "measured":
            m, lo, hi = measured
            ax.plot([lo, hi], [i, i], color=OTHER, lw=LW + 1, alpha=1.0,
                    solid_capstyle="round")
            ax.plot(m, i, marker="o", markersize=MS + 2, color=OTHER)
        elif mean is not None:
            ax.plot([lo, hi], [i, i], color=HUMAN, lw=LW, alpha=ALPHA,
                    solid_capstyle="round")
            ax.plot(mean - CTRL, i, marker="o", markersize=MS, color=HUMAN)
        elif lab in DELTA:
            ax.plot(DELTA[lab], i, marker="^", markersize=MS, color=AGENT)
        else:
            ax.text(0.5, i, "pending run", fontsize=ticksize, color=GREY,
                    va="center")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.set_xlim(-11, 15)
    ax.set_ylim(-.7, len(ROWS) - .3)
    style_axes(ax, "$\\Delta$ BALROG vs.\\ Control")
    ax.grid(axis="y", visible=False)
    save(fig, NAME)
