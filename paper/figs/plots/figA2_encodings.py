"""One game state, every encoding the engine can emit for it.

WHY THIS REPLACED THE GALLERY. The old figure stacked four panels of 5pt text,
each taken from a DIFFERENT run at a different depth, under a caption that said
"the same game state". The encoding-sweep turn records carry both the raw ASCII
grid and the rendered encoding for every turn, so the same state can be shown
honestly in each format, side by side, at a size a reader can read.

THE STATE. Seed 2 of the BBOX_JSON sweep cell, LM turn 60: Dlvl 3, a
well-explored level with 21 entities in view. Three panels:

  (a) ASCII map -- the raw grid the engine drew (what the B0 / "ASCII map
      every turn" harnesses show), with the status line.
  (b) JSON entities -- the same state as the entity list the BBOX_JSON /
      "JSON map every turn" harness shows, one entity per line.
  (c) Lists only -- what the map-hidden variants (BBOX, SPARSE) show instead
      of a map: status, visible features, visible monsters, messages, with the
      map available only through reveal().

All text is the run's own bytes, drawn outside the usetex pipeline (LaTeX
collapses spaces and drops trailing bars, which would corrupt the map).
"""
import glob
import json
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from style import GREY, save
from paths import SWEEPS

NAME = "figA2_encoding_gallery"
CELL = f"{SWEEPS}/e2_encoding_sweep/prime_agent__BBOX_JSON__vison/turns/2_*.ndjson"
LM_TURN = 60
INK = "#16181d"
LINE = "#8a8f98"
MONO = 7.0
HEAD = 12.0


def record():
    f = sorted(glob.glob(CELL))[0]
    for line in open(f):
        r = json.loads(line)
        if r.get("lm_turn") == LM_TURN:
            return r
    raise SystemExit(f"no LM turn {LM_TURN} in {f}")


def sections(ruc):
    """{section name: text} from a rendered observation."""
    out, cur = {}, None
    for ln in ruc.split("\n"):
        m = re.match(r"=== ([^=]+?) ===\s*(.*)", ln)
        if m:
            cur = m.group(1).strip()
            out[cur] = [m.group(2)] if m.group(2) else []
        elif cur is not None:
            out[cur].append(ln)
    return {k: "\n".join(v).strip("\n") for k, v in out.items()}


def panel(ax, title, lines, fs=MONO, wrap=None):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes, fc="white",
                                ec=LINE, lw=0.9, boxstyle="round,pad=0,rounding_size=0.02",
                                zorder=1))
    ax.set_title(title, loc="left", fontsize=HEAD, color=INK, pad=6)
    if wrap:
        import textwrap
        lines = [w for ln in lines for w in (textwrap.wrap(ln, wrap) or [""])]
    with mpl.rc_context({"text.usetex": False, "font.family": "monospace"}):
        ax.text(0.02, 0.975, "\n".join(lines), transform=ax.transAxes, va="top",
                ha="left", fontsize=fs, family="monospace", color=INK,
                linespacing=1.15, zorder=2)


def render():
    r = record()
    ruc = r["rendered_user_content"]
    sec = sections(ruc)
    grid = [row.rstrip() for row in r["raw_grid"]]
    # drop the message line and the trailing empty rows of the 24-row tty
    while grid and not grid[-1].strip():
        grid.pop()
    status = sec["STATUS"].split("\n")[0]

    js = json.loads(sec["MAP (JSON, entities only)"].split("\n")[0])
    jl = ["{", f'  "player": {json.dumps(js["player"])},', '  "entities": [']
    jl += ["    " + json.dumps(e, separators=(",", ":")) + ("," if i < len(js["entities"]) - 1 else "")
           for i, e in enumerate(js["entities"])]
    jl += ["  ]", "}"]

    hidden = next(k for k in sec if k.startswith("MAP (hidden"))
    txt = [f"=== {hidden} ===", "", "=== STATUS ===", *sec["STATUS"].split("\n"), "",
           "=== VISIBLE FEATURES ===", sec["VISIBLE FEATURES"], "",
           "=== VISIBLE MONSTERS ===", sec["VISIBLE MONSTERS"], "",
           "=== MESSAGES ===", *sec.get("MESSAGES", "").split("\n")]

    # map across the top at full width; the two text encodings side by side
    # beneath it. Nine inches wide so the 79-column map still prints at ~4pt.
    fig = plt.figure(figsize=(9, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.14, 1.14], wspace=0.06, hspace=0.2,
                          left=0.012, right=0.988, top=0.945, bottom=0.015)
    top = fig.add_subplot(gs[0, :])
    bl = fig.add_subplot(gs[1, 0])
    br = fig.add_subplot(gs[1, 1])
    panel(top, "(a) ASCII map", grid[1:] + ["", status])
    panel(bl, "(b) JSON entities", jl, fs=MONO - 0.3)
    panel(br, "(c) Lists, map hidden", txt, wrap=70)
    save(fig, NAME)


if __name__ == "__main__":
    render()
