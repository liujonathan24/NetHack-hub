"""The same seed generated at four settings of room_density, as the model saw it.

ASCII dungeon art is full of TeX-active characters, so these panels opt out of the
usetex pipeline; the surrounding style is otherwise unchanged.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from style import HUMAN, save
from paths import E14, SWEEPS
from common import best_observation

NAME = "figA1_density_gallery"
CELLS = [(f"{E14}/base_r1__prime_agent/traces.jsonl", "room_density 1.0  (control)"),
         (f"{SWEEPS}/e8d_density/D025__prime_agent/transcripts", "room_density 0.25"),
         (f"{SWEEPS}/e8d_density/D010__prime_agent/transcripts", "room_density 0.10"),
         (f"{SWEEPS}/e8d_density/D0025__prime_agent/transcripts", "room_density 0.025")]


def panel(ax, rows, title, color=HUMAN, fs=5.0):
    ax.axis("off")
    ax.set_title(title, fontsize=14, color=color, loc="left", pad=4)
    text = "\n".join(r[:96] for r in (rows or ["(no map rendered in this run)"]))
    ax.text(0, 1, text, family="monospace", fontsize=fs, va="top", ha="left",
            linespacing=1.05, transform=ax.transAxes)


def render():
    with mpl.rc_context({"text.usetex": False}):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10.0))
        fig.subplots_adjust(hspace=.30)
        for ax, (path, title) in zip(axes, CELLS):
            panel(ax, best_observation(path, 16), title)
        save(fig, NAME)
