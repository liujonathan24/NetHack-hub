#!/usr/bin/env python3
"""Third fix: a panic raised inside a HOST-context blob API must unwind to the
API entry (a failed call) instead of falling through
really_done -> nh_terminate -> nle_yield -> jump_fcontext and killing the
process with SIGSEGV.

Forced here by handing nle_load_level a corrupt level blob, which makes the
save codec panic. Exit 0 == unwound cleanly; a SIGSEGV/abort == not fixed.
"""
import os, sys
sys.path.insert(0, os.environ.get("ENG", "/root/nld/e16-engine"))
from nethack_core.engine_env import EngineEnv

env = EngineEnv(); env.reset(seeds=(1, 1)); env.step(13)
good = env._engine.save_level()
print("good level blob:", len(good), "bytes", flush=True)

corrupt = good[:len(good) // 3] + b"\xde\xad\xbe\xef" * 64
try:
    env._engine.load_level_raw(corrupt)
    print("load_level_raw returned without raising")
except Exception as exc:
    print("load_level_raw raised:", type(exc).__name__, exc, flush=True)

print("PROCESS STILL ALIVE after a panic inside a host-context call", flush=True)
