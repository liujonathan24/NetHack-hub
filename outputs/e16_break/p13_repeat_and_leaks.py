"""GOAL 5: repeated / cross-process restores, drift, and resource leaks."""
import hashlib, json, os, subprocess, sys, shutil, glob
TMP = "/tmp/e16_p13"
ROLE = sys.argv[1] if len(sys.argv) > 1 else "driver"

def digest(raw):
    from nethack_core.observations import BLSTATS_IDX
    h = hashlib.sha256()
    for a in (raw.glyphs, raw.chars, raw.colors, raw.blstats, raw._inv_strs):
        h.update(a.tobytes())
    return h.hexdigest()[:16], {k: int(raw.blstats[i]) for k, i in BLSTATS_IDX.items()}

if ROLE == "seed":
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_save
    env = EngineEnv(); env.reset(seeds=(5, 5)); env.step(13)
    raw = env._engine
    raw.goto_depth(3); raw.goto_depth(2); raw.step(27)
    for _ in range(10): raw.step(ord("s"))
    checkpoint_save(env, TMP + "/parent", name="parent", note="")
    print("seeded")
elif ROLE == "one":
    from nethack_harness.checkpoints import checkpoint_restore
    env, meta = checkpoint_restore(TMP + "/parent")
    d, bl = digest(env._engine)
    print(json.dumps({"digest": d, "depth": bl["depth"], "time": bl["time"],
                      "visits": meta["visits"]}))
elif ROLE == "many":
    from nethack_harness.checkpoints import checkpoint_restore
    N = int(sys.argv[2])
    fds = lambda: len(os.listdir("/proc/self/fd"))
    hackdirs = lambda: len(glob.glob("/tmp/nethack_hackdir_*"))
    def rss():
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS"): return int(ln.split()[1])
    rows = []
    for i in range(N):
        env, meta = checkpoint_restore(TMP + "/parent", count_visit=False)
        d, _ = digest(env._engine)
        rows.append({"i": i, "digest": d, "fds": fds(), "hackdirs": hackdirs(), "rss_kb": rss()})
        del env
    for r in rows: print(json.dumps(r))
    print("distinct digests:", len({r["digest"] for r in rows}))
elif ROLE == "branch":
    from nethack_harness.checkpoints import checkpoint_restore, checkpoint_save
    env, _ = checkpoint_restore(TMP + "/parent", count_visit=False); d0, _ = digest(env._engine)
    for _ in range(15): env._engine.step(ord("s"))
    checkpoint_save(env, TMP + "/child", name="child", note="")
    env2, _ = checkpoint_restore(TMP + "/parent", count_visit=False); d1, _ = digest(env2._engine)
    env3, _ = checkpoint_restore(TMP + "/child", count_visit=False);  d2, _ = digest(env3._engine)
    env4, _ = checkpoint_restore(TMP + "/parent", count_visit=False); d3, _ = digest(env4._engine)
    print(json.dumps({"parent_first": d0, "parent_after_child_save": d1, "child": d2,
                      "parent_again": d3, "parent_stable": d0 == d1 == d3,
                      "child_differs_from_parent": d2 != d0}))
else:
    env = dict(os.environ)
    def run(*a):
        p = subprocess.run([sys.executable, __file__, *a], env=env,
                           capture_output=True, text=True, timeout=900)
        return p.returncode, "\n".join(l for l in p.stdout.splitlines()
                                       if not l.startswith("DEF_MREAD")), p.stderr
    print(run("seed")[1])
    print("--- 6 restores, each in a FRESH process ---")
    for i in range(6):
        rc, out, err = run("one"); print("  #%d rc=%s %s" % (i, rc, out.strip()))
    print("--- 12 restores inside ONE process ---")
    rc, out, err = run("many", "12"); print(out, "\n  rc =", rc)
    print("--- parent/child branching ---")
    rc, out, err = run("branch"); print(out, "rc =", rc)
    print("--- leftovers ---")
    print("  /tmp/nethack_hackdir_* :", len(glob.glob("/tmp/nethack_hackdir_*")))
    print("  nle_crash files in cwd :", sorted(os.path.basename(p) for p in
          glob.glob("/root/nld/zombie-fix/nle_crash_*.txt")))
