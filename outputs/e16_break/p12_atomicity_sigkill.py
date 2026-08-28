"""GOAL 3: SIGKILL a save at each point in the write sequence.

Not simulated with exceptions -- the child really receives signal 9 mid-sequence,
so no finally: block, no atexit and no fsync gets to run.

  A  after state.bundle is written into the staging dir
  B  after every staging file is written, before the rename
  C  OVERWRITE PATH: after os.rename(directory, .old-*), before os.rename(tmp, directory)
  D  during checkpoint_restore's meta.json visit-counter atomic_write
"""
import os, subprocess, sys, shutil

TMP = "/tmp/e16_p12"; ROOT = TMP + "/archive/run"
CHILD = r'''
import os, signal, sys
POINT = sys.argv[1]; ROOT = sys.argv[2]
import nethack_harness.checkpoints as ck
from nethack_core.engine_env import EngineEnv
real_write, real_rename, real_replace = ck._write_file, os.rename, os.replace
n = {"w": 0, "r": 0}
def kill(): os.kill(os.getpid(), signal.SIGKILL)
def w(path, data):
    n["w"] += 1; real_write(path, data)
    if POINT == "A" and str(path).endswith("state.bundle"): kill()
def r(a, b):
    n["r"] += 1; real_rename(a, b)
    # C: rename #1 is the creating save; #2 is os.rename(directory, .old-*) in
    # the overwrite path -- kill there, before os.rename(tmp, directory).
    if POINT == "C" and n["r"] == 2:
        sys.stderr.write("KILL after rename #2 (%s -> %s)\n" % (a, b)); sys.stderr.flush(); kill()
def rep(a, b):
    if POINT == "D": kill()
    return real_replace(a, b)
ck._write_file = w; os.rename = r; os.replace = rep
env = EngineEnv(); env.reset(seeds=(1, 1)); env.step(13)
env._engine.goto_depth(3); env._engine.goto_depth(2)
if POINT in ("C", "D"):
    ck.checkpoint_save(env, ROOT + "/c1", name="first", note="MUST SURVIVE")
if POINT == "D":
    ck.checkpoint_restore(ROOT + "/c1")
elif POINT == "B":
    orig = ck._fsync_dir
    def fs(p):
        orig(p)
        if str(p).startswith(ROOT + "/.tmp-"): kill()
    ck._fsync_dir = fs
    ck.checkpoint_save(env, ROOT + "/c1", name="killed", note="")
elif POINT == "C":
    ck.checkpoint_save(env, ROOT + "/c1", name="second", note="", overwrite=True)
else:
    ck.checkpoint_save(env, ROOT + "/c1", name="killed", note="")
print("CHILD SURVIVED (no kill happened)")
'''
open("/tmp/e16_p12_child.py", "w").write(CHILD)

from nethack_harness.checkpoints import checkpoint_list, checkpoint_restore, checkpoint_meta
for point in (sys.argv[1:] or ["A", "B", "C", "D"]):
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(ROOT)
    p = subprocess.run([sys.executable, "/tmp/e16_p12_child.py", point, ROOT],
                       env=dict(os.environ), capture_output=True, text=True, timeout=600)
    print("\n=== POINT %s (child rc=%s%s) ===" % (
        point, p.returncode, ", SIGKILL" if p.returncode == -9 else ""))
    print("archive/run contains:", sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else [])
    try:
        print("checkpoint_list ->", [os.path.basename(str(x)) for x in checkpoint_list(ROOT)])
    except Exception as e:
        print("checkpoint_list RAISED %s: %s" % (type(e).__name__, e))
    ck1 = ROOT + "/c1"
    if not os.path.isdir(ck1):
        print("c1 IS GONE"); continue
    print("c1 files:", sorted(os.listdir(ck1)))
    try:
        m = checkpoint_meta(ck1); print("c1 meta name:", m.get("name"), "visits:", m.get("visits"))
    except Exception as e:
        print("c1 meta UNREADABLE:", type(e).__name__, e)
    try:
        env, _ = checkpoint_restore(ck1, count_visit=False)
        from nethack_core.observations import BLSTATS_IDX
        print("c1 RESTORES OK, depth", int(env._engine.blstats[BLSTATS_IDX["depth"]]))
    except Exception as e:
        print("c1 restore ->", type(e).__name__, str(e)[:150])
