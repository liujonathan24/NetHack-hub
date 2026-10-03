# Paper figures

The figure pipeline for the NetHack paper, mirrored from `/root/overleaf` on the
eval box. `figs/plots/*.py` are one module per figure; `figs/*.json` are the
small extracted datasets they read, so every figure renders from this directory
alone. The `figs/gen_*.py` builders regenerate those datasets from the trace
trees under `/root/nld` and each one checks its extraction against the paper's
tables before writing. Rendered PDFs are in `images/`; `images/proposed/` holds
the figures the current draft includes.

See `figs/README.md` for the layout, the house style, and the two rules that
bit us: no bold text in any figure, and check every PDF with ghostscript
before including it.
