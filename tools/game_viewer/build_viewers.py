#!/usr/bin/env python3
"""Build the standing set of gameplay viewers (blog + artifact) in one place.

    python -m tools.game_viewer.build_viewers                     # all viewers to /tmp
    python -m tools.game_viewer.build_viewers --outdir blog/blog  # into the blog branch
    python -m tools.game_viewer.build_viewers --data-root /path/to/backfilled-outputs

This is the reproducible generator behind the published artifacts and the
blog/*.html viewers. Cells and labels live in CELLS below; edit there. No
cropping by default (export.OBS_CAP is large); reasoning is embedded in full
and read straight from each record's `reasoning` block (see export.py) -- so
the data must be backfilled first (`python -m tools.trace_reasoning <root>`).

`--data-root` points the cell paths at a copy of the run outputs (e.g. a
backfilled copy kept separate from the pristine raw captures). The all-games
ARTIFACT set omits the e6 auto-dismiss cell so the file stays under claude.ai's
16 MB artifact cap without any text truncation; `--with-e6` adds it back for the
blog, which has no such cap.
"""
from __future__ import annotations
import argparse, os
from . import export as ex

# (relative dir under the data root, label). Order is display order.
_CELLS = {
    "npcore_v3":   ("e7_seed/NPCORE_v3__prime_agent",         "NPCORE v3 (control)"),
    "npcore_v2":   ("e7_seed/NPCORE_v2__prime_agent",         "NPCORE v2"),
    "e6":          ("e6_vision/BBOX_MIN__prime_agent",        "e6 reveal+auto-dismiss"),
    "npfull":      ("e7_seed/NPFULL__prime_agent",            "NPFULL 33-tool"),
    "norm":        ("e8_plan/NORM__prime_agent",              "E8a norm gate"),
    "norm_ungated":("e8_plan/NORM_v1_ungated__prime_agent",   "E8a ungated"),
    "prayer":      ("e8_hint/PRAYER__prime_agent",            "E8b prayer hint"),
    "doors":       ("e8c_doors/UNLOCKED__prime_agent",        "E8c doors unlocked"),
    "d025":        ("e8d_density/D025__prime_agent",          "E8d density 0.25"),
    "d010":        ("e8d_density/D010__prime_agent",          "E8d density 0.10"),
    "d0025":       ("e8d_density/D0025__prime_agent",         "E8d density 0.025"),
}


def cells(root):
    """CELLS resolved against `root`: {key: (abs_dir, label)}."""
    return {k: (os.path.join(root, rel), label) for k, (rel, label) in _CELLS.items()}


# name -> (cell keys, keep_frames). Per-experiment viewers keep move-by-move
# frames; multi-cell overviews drop them (final frame per turn) for size. The
# `all_games` key list is filled at build time so `--with-e6` can toggle e6.
VIEWERS = {
    "all_games":  (None, False),
    "e7_viewer":  (["npcore_v3", "npcore_v2", "npfull"], True),
    "e8a_viewer": (["norm", "norm_ungated", "npcore_v3"], True),
    "e8b_viewer": (["prayer", "npcore_v3"], True),
    "e8c_viewer": (["doors", "npcore_v3"], True),
    "e8d_viewer": (["d025", "d010", "d0025"], True),
}


def build(name, keys, keep_frames, root):
    C = cells(root)
    dirs = [C[k][0] for k in keys]
    labels = {C[k][0]: C[k][1] for k in keys}
    games = ex.build_games(dirs, labels)
    if not keep_frames:
        for g in games:
            for t in g["turns"]:
                t["frames"] = []
    return ex.render(games), len(games)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp", help="where to write the .html files")
    ap.add_argument("--data-root", default="outputs",
                    help="root holding the run dirs (default: ./outputs)")
    ap.add_argument("--with-e6", action="store_true",
                    help="include the e6 cell in all_games (blog only; artifact omits it)")
    ap.add_argument("--only", nargs="*", help="subset of viewer names (default: all)")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    all_keys = [k for k in _CELLS if args.with_e6 or k != "e6"]
    for name, (keys, frames) in VIEWERS.items():
        if args.only and name not in args.only:
            continue
        keys = all_keys if keys is None else keys
        html, n = build(name, keys, frames, args.data_root)
        path = os.path.join(args.outdir, f"{name}.html")
        open(path, "w").write(html)
        print(f"{path}: {n} games, {len(html)/1e6:.1f} MB"
              f"{' (frames)' if frames else ' (overview)'}")


if __name__ == "__main__":
    main()
