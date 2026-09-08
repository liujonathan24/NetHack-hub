#!/usr/bin/env python3
"""Diff two replay dumps step by step and report the FIRST divergence in full,
including a rendered before/after of both engines' 24x80 screens.

    python3 diff_parity.py <a_prefix> <b_prefix>
"""
import json
import sys

import numpy as np

A, B = sys.argv[1], sys.argv[2]

FIELDS = ["tty_chars", "tty_colors", "glyphs", "chars", "colors",
          "inv_strs", "inv_letters", "inv_glyphs",
          "tty_cursor", "message", "blstats", "time", "done", "how_done",
          "all"]


def load(p):
    return [json.loads(l) for l in open(p + ".ndjson")]


def screens(p, n):
    raw = np.fromfile(p + ".tty.bin", dtype=np.uint8)
    return raw.reshape(-1, 24, 80)[n]


def render(scr):
    return [bytes(r).decode("latin1").rstrip() for r in scr]


ra, rb = load(A), load(B)
ea = json.load(open(A + ".events.json"))
eb = json.load(open(B + ".events.json"))

print(f"A = {A}  engine={ea['engine']}")
print(f"    so={ea['ident']['so']}")
print(f"    md5={ea['ident']['so_md5']} size={ea['ident']['so_size']}")
print(f"    status={ea['status']} frames={ea['frames_written']} "
      f"gt_ok={ea['gt_ok']}/{ea['gt_checked']}")
print(f"B = {B}  engine={eb['engine']}")
print(f"    so={eb['ident']['so']}")
print(f"    md5={eb['ident']['so_md5']} size={eb['ident']['so_size']}")
print(f"    status={eb['status']} frames={eb['frames_written']} "
      f"gt_ok={eb['gt_ok']}/{eb['gt_checked']}")
print()

if len(ra) != len(rb):
    print(f"!! FRAME COUNT DIFFERS: A={len(ra)} B={len(rb)}")

n = min(len(ra), len(rb))
first = None
nbad = 0
for i in range(n):
    a, b = ra[i], rb[i]
    bad = [f for f in FIELDS if a.get(f) != b.get(f)]
    if bad:
        nbad += 1
        if first is None:
            first = (i, bad, a, b)

print(f"frames compared : {n}")
print(f"action-byte steps compared : {sum(1 for r in ra[:n] if r['kind'] == 'step')}")
print(f"frames differing: {nbad}")
print()

if first is None and len(ra) == len(rb):
    print("RESULT: IDENTICAL -- every compared field matches on every frame.")
    # per-field belt and braces
    for f in FIELDS:
        assert all(ra[i].get(f) == rb[i].get(f) for i in range(n)), f
    print("        (re-verified field by field)")
    sys.exit(0)

i, bad, a, b = first
print("RESULT: DIVERGENCE")
print(f"  first differing frame : step {i}")
print(f"  record index          : {a['rec']}   kind={a['kind']}")
print(f"  action byte           : {a['action']}"
      + (f" ({chr(a['action'])!r})" if isinstance(a['action'], int) and 32 <= a['action'] < 127 else ""))
print(f"  differing fields      : {bad}")
print()
for f in bad:
    print(f"  {f}:")
    print(f"    A: {a.get(f)}")
    print(f"    B: {b.get(f)}")
print()
for label, pfx, idx in (("A", A, i), ("B", B, i)):
    print(f"  ---- {label} screen AFTER step {idx} ----")
    for line in render(screens(pfx, idx)):
        print("   |" + line)
if i > 0:
    print()
    for label, pfx in (("A", A), ("B", B)):
        print(f"  ---- {label} screen BEFORE (step {i-1}) ----")
        for line in render(screens(pfx, i - 1)):
            print("   |" + line)
sys.exit(1)
