#!/usr/bin/env python
"""Rebuild and print a run's checkpoint tree from archive/*/meta.json alone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run_dir: Path) -> dict:
    out = {}
    for meta in sorted((run_dir / "archive").glob("*/meta.json")):
        m = json.loads(meta.read_text())
        out[m["id"]] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    nodes = load(Path(a.run_dir))
    kids: dict[str | None, list[str]] = {}
    for m in nodes.values():
        kids.setdefault(m.get("parent"), []).append(m["id"])
    roots = kids.get(None, [])
    if len(roots) != 1:
        print(f"WARNING: {len(roots)} roots (expected 1): {roots}")
    orphans = [i for i, m in nodes.items() if m.get("parent") not in (None, *nodes)]
    if orphans:
        print(f"WARNING: orphans {orphans}")

    def walk(cid, depth=0):
        m = nodes[cid]
        print(f"{'  ' * depth}{cid}  a{m['attempt']} step {m['step']:>3}  "
              f"prog {m['progression']:.3f}  aux {m['aux_progress']:.0f}  ({m['reason']})")
        for k in sorted(kids.get(cid, []), key=lambda x: int(x[1:])):
            walk(k, depth + 1)

    for r in roots:
        walk(r)
    print(f"\n{len(nodes)} checkpoints, {len(roots)} root(s), {len(orphans)} orphan(s)")


if __name__ == "__main__":
    main()
