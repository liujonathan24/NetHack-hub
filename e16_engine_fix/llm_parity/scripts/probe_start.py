#!/usr/bin/env python3
"""Probe: does a fresh reset(seeds=(1,1), Val-hum-neu-fem) -- or resume(c1) --
reproduce the recorded run's turn-1 ground-truth observation?

    ENG=<engine root> python3 probe_start.py <mode>     mode = reset | resume
"""
import base64
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.environ["ENG"])
from nethack_core.engine_env import EngineEnv  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NDJSON = os.path.join(HERE, "1_2038677_1787958180.ndjson")
C1 = os.path.join(HERE, "archive_copy", "c1")

MODE = sys.argv[1] if len(sys.argv) > 1 else "reset"


def decode_gt(g):
    raw = zlib.decompress(base64.b64decode(g["data"]))
    out, off = {}, 0
    for name, dt, shape in g["planes"]:
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        out[name] = np.frombuffer(raw[off:off + n], dtype=dt).reshape(shape).copy()
        off += n
    return out


def main():
    with open(NDJSON) as f:
        rec0 = json.loads(f.readline())
    gt = decode_gt(rec0["gt_obs"])

    env = EngineEnv()
    if MODE == "reset":
        obs, _meta = env.reset(seeds=(1, 1), character="Val-hum-neu-fem")
    else:
        env.reset(seeds=(1, 1), character="Val-hum-neu-fem")
        obs = env.resume(C1)

    print(f"ENG={os.environ['ENG']} mode={MODE}")
    print(f"  current_seeds={env.current_seeds}")
    for name in ("glyphs", "chars", "colors", "blstats", "message",
                 "inv_strs", "inv_letters", "inv_glyphs", "tty_colors",
                 "tty_cursor"):
        a = np.asarray(getattr(obs, name))
        b = gt[name]
        same = a.shape == b.shape and bool((a == b).all())
        print(f"  {name:<12} live={a.shape} gt={b.shape} MATCH={same}")
        if not same and name == "blstats":
            print(f"      live={list(map(int, a))}")
            print(f"      gt  ={list(map(int, b))}")
    print("  live blstats:", list(map(int, np.asarray(obs.blstats))))
    print("  gt   blstats:", list(map(int, gt["blstats"])))
    tty = np.asarray(obs.tty_chars)
    print("  --- live tty_chars ---")
    for row in tty:
        print("   |" + bytes(row).decode("latin1").rstrip() + "|")
    print("  --- recorded raw_grid ---")
    for row in rec0["raw_grid"]:
        print("   |" + row.rstrip() + "|")


if __name__ == "__main__":
    main()
