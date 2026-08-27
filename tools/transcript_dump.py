#!/usr/bin/env python3
"""Dump a cell's traces.jsonl as plain-text transcripts, one file per seed.

    python -m tools.transcript_dump <cell_dir> [-o outdir]

Every node in order: role, sampled flag, <REASONING>, <CONTENT>, <TOOL_CALL>
blocks verbatim. No post-processing, no alignment -- this is the raw record of
what the model saw and did, for human reading (the v3 seed-2 post-mortem was
done on exactly this format).
"""
from __future__ import annotations
import argparse
import json
import os


def text(c):
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def dump_trace(t, path):
    seed = ((t.get("task") or {}).get("data") or {}).get("idx")
    out = [f"===== FULL TRANSCRIPT — seed {seed} ====="]
    for i, nd in enumerate(t.get("nodes") or []):
        m = nd.get("message") or {}
        samp = "(sampled)" if nd.get("sampled") else ""
        out.append(f"\n{'─'*70}\n[node {i}] role={m.get('role','?')} {samp}")
        rc = text(m.get("reasoning_content"))
        if rc.strip():
            out.append(f"\n<REASONING>\n{rc}")
        c = text(m.get("content"))
        if c.strip():
            out.append(f"\n<CONTENT>\n{c}")
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            name = fn.get("name") if isinstance(fn, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            out.append(f"\n<TOOL_CALL {name}>\n{args}")
    open(path, "w").write("\n".join(out))
    return seed, os.path.getsize(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir")
    ap.add_argument("-o", "--outdir", default=None,
                    help="default: <cell_dir>/transcripts/")
    args = ap.parse_args(argv)
    tf = os.path.join(args.cell_dir, "traces.jsonl")
    outdir = args.outdir or os.path.join(args.cell_dir, "transcripts")
    os.makedirs(outdir, exist_ok=True)
    for line in open(tf):
        t = json.loads(line)
        seed = ((t.get("task") or {}).get("data") or {}).get("idx")
        path = os.path.join(outdir, f"seed{seed}_transcript.txt")
        seed, size = dump_trace(t, path)
        print(f"seed {seed}: {size//1024} KB -> {path}")


if __name__ == "__main__":
    main()
