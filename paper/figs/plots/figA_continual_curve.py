"""Training and held-out progression across the continual harness iterations.

Four panels, one per reported quantity: max dungeon level, max experience
level, BALROG-max and BALROG-min. Two series, the memory-only reflection run
and the wiki-grounded one. Iteration 0 is the frozen base harness with an
empty store, so every curve starts from the same control.

Solid lines are the five held-out seeds (iterations 1-10, plus iteration 0 =
the empty-store base harness, n=15); dashed are the five training seeds the
edits were written against. The body's claim is about the training curve: if
the loop were memorising its own dungeons, training would rise while held-out
stayed flat. Each point is a mean over five seeds (iteration 0 over 15) with
the standard error banded. Held-out rollouts are deduplicated to one per seed,
keeping the run with the most skill_calls -- the same rule
gen_progression_main.py, superhuman.py and the paper tables use (an
aborted-and-retried rollout writes a record for both attempts; the aborted one
is short and shallow).

F.1: authored at 6.6in against a 5.97in column, so the placement scale is
~0.90 and the point sizes below land near the 9pt caption.
"""
import json
import statistics as st

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from style import AGENT, blue, GREY, LW, save

NAME = "figA_continual_curve"
OUTDIR = "/root/overleaf/images/proposed"
SRC = "/root/overleaf/figs/continual_curve.json"      # held-out, seeds 0-4
SRC_TRAIN = "/root/overleaf/figs/continual_train.json"  # training, seeds 6-10

PANELS = [(0, "Max dungeon level", None),
          (1, "Max experience level", None),
          (2, r"BAL$_{\max}$ (\%)", None),
          (3, r"BAL$_{\min}$ (\%)", None)]
SERIES = [("memory-only", "memory-only", AGENT),
          ("wiki", r"$+$ wiki", blue)]


def series(rounds, idx):
    xs = sorted(int(r) for r in rounds)
    mu, se = [], []
    for r in xs:
        v = [row[idx] for row in rounds[str(r)]]
        mu.append(st.mean(v))
        se.append(st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0)
    return np.array(xs), np.array(mu), np.array(se)


def render():
    data = json.load(open(SRC))
    train = json.load(open(SRC_TRAIN))
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
    for (idx, ylab, _), ax in zip(PANELS, axes.ravel()):
        for key, label, colour in SERIES:
            x, mu, se = series(data[key], idx)
            ax.plot(x, mu, color=colour, lw=LW - 1.2, label=label + ", held out",
                    zorder=3)
            ax.fill_between(x, mu - se, mu + se, color=colour, alpha=.16, lw=0)
            xt, mt, _ = series(train[key], idx)
            ax.plot(xt, mt, color=colour, lw=LW - 1.6, ls="--", alpha=.75,
                    label=label + ", training", zorder=2)
        base = series(data["memory-only"], idx)[1][0]
        ax.axhline(base, color=GREY, ls=":", lw=1.2, zorder=1)
        ax.set_ylabel(ylab, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, linestyle="--", alpha=.45)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Continual harness iteration", fontsize=10)
    axes[0][0].set_xticks(range(0, 11, 2))
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(Line2D([], [], color=GREY, ls=":", lw=1.2))
    l.append("base harness")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(.5, 1.10),
               ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, NAME, outdir=OUTDIR)


if __name__ == "__main__":
    render()
