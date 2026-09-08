#!/usr/bin/env python3
"""Regenerate every figure in the paper.

    python figs/render_all.py            # extract traces, then render all figures
    python figs/render_all.py fig3 fig7  # render a subset (uses the cached figdata)
    python figs/render_all.py --no-extract

Outputs vector PDFs into images/, which the LaTeX source pulls in via \\graphicspath.
"""
import os, sys, glob, importlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def discover():
    mods = {}
    for path in sorted(glob.glob(os.path.join(HERE, "plots", "*.py"))):
        stem = os.path.basename(path)[:-3]
        if stem.startswith("_"):
            continue
        mods[stem.split("_")[0]] = "plots." + stem
    return mods


def main():
    argv = [a for a in sys.argv[1:] if a != "--no-extract"]
    if "--no-extract" not in sys.argv:
        print("[data] extracting traces")
        import data
        data.main()

    mods = discover()
    want = argv or sorted(mods, key=lambda k: (k[0].isdigit() is False, k))
    unknown = [w for w in want if w not in mods]
    if unknown:
        print(f"unknown figure(s): {unknown}\navailable: {sorted(mods)}")
        return 1

    ok, bad = 0, []
    for key in want:
        mod = importlib.import_module(mods[key])
        print(f"[{key}] {mod.NAME}")
        try:
            mod.render()
            ok += 1
        except Exception:
            bad.append(key)
            traceback.print_exc()
    print(f"\n{ok} figure(s) written to images/" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
