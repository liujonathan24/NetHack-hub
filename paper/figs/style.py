"""House plotting style. Every figure module imports from here and nowhere else.

The rcParams block below is the project style verbatim; the palette names under it
map the five brand colours onto the roles they play across the paper's figures.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

# DO NOT USE \textbf OR \bfseries IN ANY FIGURE TEXT. Under this usetex setup
# matplotlib 3.11 embeds URW Helvetica Bold (uhvb8a.pfb) in a way ghostscript
# and pdftex-based viewers reject -- "Error reading a content stream. The page
# may be incomplete." -- and the page renders with the plot contents MISSING.
# Regular Helvetica and the Computer Modern maths fonts embed fine, so this is
# specific to bold. Verified 2026-09-03 by bisection on fig4_plane_story: the
# identical figure with plain-weight titles is a valid PDF. Check any new figure
# with  gs -dNOPAUSE -dBATCH -sDEVICE=nullpage file.pdf  before including it.
labelsize = 20
titlesize = 20
legendsize = 20
fontweight = 20
ticksize = 16

red = "#FF8988"
orange = "#FECC81"
blue = "#6098FF"
green = "#77B25D"
purple = "#B28CFF"
color_d = [red, orange, blue, green, purple]

# ---- roles (so a colour means the same thing in every figure) ----
HUMAN = blue        # NAO population
HUMAN_TOP = purple  # NAO top 10
AGENT = green       # our rollouts
BAD = red           # deaths, failures, wasted turns
OTHER = orange      # secondary / structural

GREY = "#555555"
LIGHT = "#cccccc"

LW = 3.0
MS = 10
ALPHA = 0.9

# figure sizes: authored at the project's 7x4.5, taller variants for stacked panels
SIZE = (7, 4.5)
SIZE_TALL = (7, 7.0)
SIZE_WIDE = (12, 4.5)


def tex(s):
    """Escape a plain string for the usetex text pipeline."""
    for a, b in (("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde "),
                 ("^", r"\textasciicircum ")):
        s = s.replace(a, b)
    return s


def mono(s):
    """A code identifier, typeset as monospace and escaped."""
    return r"\texttt{" + tex(s) + "}"


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, linestyle="--")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if title:
        ax.set_title(title, fontsize=titlesize)
    ax.tick_params(axis="both", labelsize=ticksize)
    return ax


def save(fig, name, outdir="/root/overleaf/images"):
    import os
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ".pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  wrote {name}.pdf")
