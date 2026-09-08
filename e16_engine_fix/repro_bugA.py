"""E16 native-crash repro.

Mimics one treesmoke2 attempt in-process, no LLM:
  fresh EngineEnv -> reset -> checkpoint_restore(archive/cN, env=env)
  -> N "LM turns", each = a few engine steps + push_rollback_snapshot
  -> auto checkpoint_save every 20 turns
  -> occasional rollback(n)

Usage:
  export PYTHONPATH=...
  cd /root/nld/zombie-fix && .venv-cli-eval/bin/python repro_e16.py [turns] [ckpt] [mode]

mode: full (default) | nosave | norollback | nosnapshot
"""
import faulthandler, os, random, shutil, sys, time, glob

faulthandler.enable(all_threads=True)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY,) * 2)
except Exception as e:
    print("core rlimit:", e)

TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
CKPT = sys.argv[2] if len(sys.argv) > 2 else "/root/nld/e16_runs/treesmoke2/archive/c1"
MODE = sys.argv[3] if len(sys.argv) > 3 else "full"
TMP = "/tmp/e16_repro_%d" % os.getpid()
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

from nethack_core.engine_env import EngineEnv
from nethack_core.observations import BLSTATS_IDX
from nethack_harness.checkpoints import checkpoint_restore, checkpoint_save
from nethack_harness.tools.skills import (
    push_rollback_snapshot, rollback, _rollback_ring, ROLLBACK_RING,
)

def rss_kb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS"):
            return int(ln.split()[1])
    return -1

def gt(raw):
    return int(raw.blstats[BLSTATS_IDX["time"]])

def hp(raw):
    return int(raw.blstats[BLSTATS_IDX["hitpoints"]])

print("PID", os.getpid(), "mode", MODE, "turns", TURNS, "ckpt", CKPT, flush=True)

env = EngineEnv()
env.reset(seeds=(1, 1))
env, meta = checkpoint_restore(CKPT, env=env, count_visit=False)
raw = env._engine
print("restored: dlvl=%s gt=%s hp=%s" % (
    int(raw.blstats[BLSTATS_IDX["depth"]]), gt(raw), hp(raw)), flush=True)

rnd = random.Random(20260828)
# keys a np_* skill actually emits: movement, search, pickup, descend, ESC
KEYS = [ord(c) for c in "hjklyubn"] + [ord("s")] * 3 + [ord(","), ord("."), 27]

ck_id = 100
t0 = time.time()
for turn in range(1, TURNS + 1):
    # a skill call is many engine steps (np_explore_level can be 20+)
    nsteps = rnd.choice([1, 1, 2, 3, 5, 8, 12, 20])
    for _ in range(nsteps):
        raw.step(rnd.choice(KEYS))
        if raw.done:
            break
    if raw.done:
        print("turn %d: game over (how_done=%s) -- restarting from ckpt"
              % (turn, int(raw.how_done)), flush=True)
        env, meta = checkpoint_restore(CKPT, env=env, count_visit=False)
        raw = env._engine

    if MODE != "nosnapshot" and hp(raw) > 0:
        push_rollback_snapshot(env, turn, game_time=gt(raw))

    if MODE != "nosave" and turn % 20 == 0:
        ck_id += 1
        d = os.path.join(TMP, "c%d" % ck_id)
        try:
            checkpoint_save(env, d, name="auto", note="cadence")
        except Exception as exc:
            print("  save turn %d -> %s: %s" % (turn, type(exc).__name__, exc), flush=True)

    if MODE != "norollback" and turn % 37 == 0:
        try:
            r = rollback(env, None, n=rnd.choice([1, 1, 2, 5]))
            for a in (r.actions or []):
                raw.step(a)
        except Exception as exc:
            print("  rollback turn %d -> %s: %s" % (turn, type(exc).__name__, exc), flush=True)

    if turn % 20 == 0:
        print("turn %4d gt=%6d hp=%3d ring=%2d snaps=%3d fds=%3d rss=%dMB t=%.0fs"
              % (turn, gt(raw), hp(raw), len(_rollback_ring(env)),
                 len(raw._snapshots), len(os.listdir("/proc/self/fd")),
                 rss_kb() // 1024, time.time() - t0), flush=True)

print("SURVIVED %d turns in %.0fs" % (TURNS, time.time() - t0), flush=True)
print("nle_crash files:", sorted(glob.glob("nle_crash_*.txt")), flush=True)
shutil.rmtree(TMP, ignore_errors=True)
