#!/usr/bin/env python3
"""How far does a plain byte replay track the ORIGINAL run, under different
reset options? Diagnostic only -- picks the most faithful start for the real
parity replay.

    ENG=... python3 probe_fidelity.py <nrecords> [tune|notune] [prologue_bytes]
"""
import base64
import json
import os
import sys
import zlib

import numpy as np

ENG = os.environ["ENG"]
os.environ["NLE_LIB_PATH"] = os.path.join(
    ENG, "third_party", "NetHack", "src", "build", "libnethack.so")
sys.path.insert(0, ENG)
from nethack_core.engine_env import EngineEnv  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NREC = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODE = sys.argv[2] if len(sys.argv) > 2 else "notune"
PROLOGUE = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [27]


def dec(g):
    raw = zlib.decompress(base64.b64decode(g["data"]))
    out, off = {}, 0
    for n, dt, sh in g["planes"]:
        c = int(np.prod(sh)) * np.dtype(dt).itemsize
        out[n] = np.frombuffer(raw[off:off + c], dtype=dt).reshape(sh).copy()
        off += c
    return out


recs = [json.loads(l) for l in open(os.path.join(HERE, "1_2038677_1787958180.ndjson"))]

if MODE == "ckpt":
    # Start exactly the way the recorded run did: restore checkpoint c1
    # (the run's resume_checkpoint), which is NOT identical to a fresh reset.
    sys.path.insert(0, "/root/nld/zombie-fix/environments/nethack")
    from nethack_harness.checkpoints import checkpoint_restore
    env = EngineEnv()
    env.reset(seeds=(1, 1), character="Val-hum-neu-fem")
    env, _meta = checkpoint_restore(os.path.join(HERE, "archive_copy", "c1"),
                                    env=env, count_visit=False)
    obs = getattr(env, "_last_restore_obs", None) or env.engine.to_core_observation()
else:
    env = EngineEnv()
    tune = {"reveal_map": 1.0} if MODE == "tune" else None
    obs, _ = env.reset(seeds=(1, 1), character="Val-hum-neu-fem", tune=tune)
    for b in PROLOGUE:
        obs, _d, _i = env.step(b)

print(f"MODE={MODE} prologue={PROLOGUE}")
nglyph_ok = ntime_ok = 0
for i, rec in enumerate(recs[:NREC]):
    for b in (rec["actions"].get("bytes") or []):
        obs, done, _ = env.step(int(b))
        if done:
            break
    gt = dec(rec["gt_obs"])
    g_ok = np.array_equal(np.asarray(obs.glyphs), gt["glyphs"])
    bl = np.asarray(obs.blstats)
    t_ok = int(bl[20]) == int(gt["blstats"][20])
    p_ok = int(bl[0]) == int(gt["blstats"][0]) and int(bl[1]) == int(gt["blstats"][1])
    nglyph_ok += g_ok
    ntime_ok += t_ok
    flag = "OK " if (g_ok and t_ok and p_ok) else "DIFF"
    print(f"  rec{i:>3} {flag} glyphs={g_ok!s:<5} time={int(bl[20])}/{int(gt['blstats'][20])} "
          f"pos=({int(bl[0])},{int(bl[1])})/({int(gt['blstats'][0])},{int(gt['blstats'][1])})")
print(f"SUMMARY {MODE}: glyphs_ok={nglyph_ok}/{NREC} time_ok={ntime_ok}/{NREC}")
