#!/usr/bin/env python3
"""Build the standing set of gameplay viewers (blog + artifact) in one place.

    python -m tools.game_viewer.build_viewers            # write all viewers to /tmp
    python -m tools.game_viewer.build_viewers --outdir blog   # into the blog branch

This is the reproducible generator behind the published artifacts and the
blog/*.html viewers. Cells and labels live in CELLS below; edit there. No
cropping by default (export.OBS_CAP is large); reasoning is embedded in full.
The all-games ARTIFACT set omits the e6 auto-dismiss cell so the file stays
under claude.ai's 16 MB artifact cap without any text truncation.
"""
from __future__ import annotations
import argparse, os
from . import export as ex

O = "outputs"
# (dir, label). Order is display order.
CELLS = {
    "npcore_v3":  (f"{O}/e7_seed/NPCORE_v3__prime_agent",        "NPCORE v3 (control)"),
    "npcore_v2":  (f"{O}/e7_seed/NPCORE_v2__prime_agent",        "NPCORE v2"),
    "e6":         (f"{O}/e6_vision/BBOX_MIN__prime_agent",       "e6 reveal+auto-dismiss"),
    "npfull":     (f"{O}/e7_seed/NPFULL__prime_agent",           "NPFULL 33-tool"),
    "norm":       (f"{O}/e8_plan/NORM__prime_agent",             "E8a norm gate"),
    "norm_ungated":(f"{O}/e8_plan/NORM_v1_ungated__prime_agent", "E8a ungated"),
    "prayer":     (f"{O}/e8_hint/PRAYER__prime_agent",           "E8b prayer hint"),
    "doors":      (f"{O}/e8c_doors/UNLOCKED__prime_agent",       "E8c doors unlocked"),
    "d025":       (f"{O}/e8d_density/D025__prime_agent",         "E8d density 0.25"),
    "d010":       (f"{O}/e8d_density/D010__prime_agent",         "E8d density 0.10"),
    "d0025":      (f"{O}/e8d_density/D0025__prime_agent",        "E8d density 0.025"),
}

# name -> (cell keys, keep_frames). Per-experiment viewers keep move-by-move
# frames; multi-cell overviews drop them (final frame per turn) for size.
VIEWERS = {
    # all-games artifact: everything EXCEPT e6 (so it fits 16 MB with no crop)
    "all_games":  ([k for k in CELLS if k != "e6"], False),
    "e8a_viewer": (["norm", "norm_ungated", "npcore_v3"], True),
    "e8b_viewer": (["prayer", "npcore_v3"], True),
    "e8c_viewer": (["doors", "npcore_v3"], True),
    "e8d_viewer": (["d025", "d010", "d0025"], True),
}


def build(name, keys, keep_frames):
    dirs = [CELLS[k][0] for k in keys]
    labels = {CELLS[k][0]: CELLS[k][1] for k in keys}
    games = ex.build_games(dirs, labels)
    if not keep_frames:
        for g in games:
            for t in g["turns"]:
                t["frames"] = []
    return ex.render(games), len(games)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp", help="where to write the .html files")
    ap.add_argument("--only", nargs="*", help="subset of viewer names (default: all)")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    for name, (keys, frames) in VIEWERS.items():
        if args.only and name not in args.only:
            continue
        html, n = build(name, keys, frames)
        path = os.path.join(args.outdir, f"{name}.html")
        open(path, "w").write(html)
        print(f"{path}: {n} games, {len(html)/1e6:.1f} MB"
              f"{' (frames)' if frames else ' (overview)'}")


if __name__ == "__main__":
    main()
