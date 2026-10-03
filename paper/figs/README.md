# Figures

Everything the paper renders lives here. One command regenerates all of it:

```bash
python figs/render_all.py                 # re-extract traces, then render all figures
python figs/render_all.py --no-extract    # skip extraction, reuse figs/figdata.json
python figs/render_all.py fig3 fig7       # render a subset by key
```

Output is vector PDF in `images/`, which the LaTeX source picks up through
`\graphicspath{{images/}}`.

## Layout

| File | Role |
|---|---|
| `style.py` | The house rcParams, the five-colour palette, and the shared `save`/`style_axes` helpers. Every figure imports its colours and sizes from here. |
| `paths.py` | Every input path, in one place. Moving a run directory is a one-line edit. |
| `data.py` | Walks the E14 traces and writes `figdata.json`: one record per LM turn (tool called, whether a map was returned, the status line). Plots never touch the trace tree. |
| `common.py` | Loaders and derived series shared by more than one figure — the BALROG tables, the human norm, the per-rollout progression curves, the observation-gallery extractor. |
| `plots/*.py` | One module per figure. Each defines `NAME` and `render()`; `render_all.py` discovers them automatically, so adding a figure means adding a file. |


## The plane figures

Four modules draw the (Dlvl, XL) plane and they share one convention, so a band
or a colour means the same thing in each:

| Module | Output | What it is for |
|---|---|---|
| `fig4_plane_story.py` | `proposed/fig4_plane_models_corrections.pdf` | Replaces the old `fig4_depth_xp` endpoint scatter. One line per series -- median XL on arrival at each depth, ending where half the rollouts stopped -- for the five models (a) and the correction arms (b), against the same statistic over the NAO ascensions. BALROG-min level sets shaded behind. |
| `fig2_trajectories_ge.py` | `proposed/fig2_trajectories_ge.pdf` | The Go-Explore section's plane in the same grammar: single lives and lineages at 30 and 100 attempts as lines, ascensions as median plus interquartile band. Imports `pace_of` and `min_bands` from `fig4_plane_story`. |
| `fig_plane_models.py` | `proposed/fig_plane_models_*.pdf` | The four rollout-selection variants, kept for the selection argument. |
| `fig4_depth_xp.py` | `fig4_depth_xp.pdf` | The superseded endpoint version, kept for reference. |

`gen_probe_plane.py` builds `e15_probes_plane.json`, the intervention arms'
paths, and refuses to write a file that does not reproduce the paper's
intervention table.

## Style

`style.py` holds the project rcParams verbatim (`text.usetex`, Helvetica, label 20 /
tick 16) and names the palette by role, so a colour means the same thing everywhere:

| Role | Colour |
|---|---|
| `HUMAN` | blue — NAO population |
| `HUMAN_TOP` | purple — NAO top 10 |
| `AGENT` | green — our rollouts |
| `BAD` | red — deaths, failures, wasted turns |
| `OTHER` | orange — secondary / structural |

Two exceptions, both deliberate:

- **`figA1` and `figA2` opt out of `usetex`** via `mpl.rc_context`. They print raw
  ASCII dungeon art, which is full of TeX-active characters (`#`, `$`, `%`, `\`, `_`,
  `{`, `}`, `^`, `~`); escaping it would corrupt the maps.
- **`fig1` sets its own font sizes.** It is a box-and-arrow diagram, so label size is
  governed by box width rather than by `labelsize`.

Use `style.tex()` to escape a plain string for the usetex pipeline and `style.mono()`
for code identifiers (`np_move_to` → `\texttt{np\_move\_to}`).

## No bold text

`\textbf` / `\bfseries` under this usetex setup makes matplotlib embed URW
Helvetica Bold in a way PDF readers reject: the file opens, but the page content
is missing and ghostscript reports "Error reading a content stream". The
matplotlib-side PNG looks fine, which is how it went unnoticed. `style.py` has
the details. Titles are regular weight, and every figure is checked with

```bash
for f in images/*.pdf images/proposed/*.pdf; do
  gs -dNOPAUSE -dBATCH -sDEVICE=nullpage "$f" 2>&1 | grep -q Error && echo "BROKEN $f"; done
```

## Requirements

`usetex` needs a working LaTeX: `texlive-latex-base`, `texlive-latex-extra`,
`texlive-fonts-recommended` (for `helvet`), `dvipng`, `cm-super`.
Python side: `matplotlib`, `numpy`.
