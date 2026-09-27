#!/usr/bin/env python3
"""Can the checkpoints banked before an env_crash still be restored?

WHY THIS MATTERS. treesmoke11 a021, treesmoke11_r2 and _r3 lost 17 attempts to
`SIGSEGV / panic_msg=extract_nobj: object lost`, clustered by depth (every r3
crash at dlvl 19, r2's at 14 then 20). Those attempts banked checkpoints on
their way down. If the banked states carry the corrupt object chain, then the
archive is poisoned: every later attempt that resumes one crashes again, which
is exactly the depth-clustered pattern the crash logs show. If instead they
restore cleanly, the corruption arose during play and the archive is sound.

SAFETY. Seven orchestrators are live on these archives.
  * Every checkpoint is COPIED to /tmp and the copy is restored. The live
    archive is never opened for write.
  * `count_visit=False` as well, belt and braces -- checkpoint_restore's
    default bumps a visit counter INTO meta.json, which would mutate a live
    run's archive.
  * Each restore runs in its own subprocess. `extract_nobj` is a SIGSEGV, not
    an exception; in-process it would take this script down with it, and the
    exit code is the measurement we want anyway (-11 == SIGSEGV).

  python probe_checkpoint_health.py <run> <c1> <c2> ...
  python probe_checkpoint_health.py --one <path-to-copied-ckpt>   # internal
"""
import json, os, shutil, subprocess, sys, tempfile

REPO = "/root/nld/zombie-fix"
ENG = "/root/nld/e16-engine"
for p in (f"{REPO}/tools/pycompat", ENG, REPO, f"{REPO}/environments/nethack"):
    if p not in sys.path:
        sys.path.insert(0, p)

STEPS_AFTER = 40          # ESC-free actions after restore, to see if it survives
SEARCH = 115


def probe_one(ckpt):
    """Restore a COPY and step it. Prints one JSON line; may die by signal."""
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_restore
    meta = json.load(open(os.path.join(ckpt, "meta.json")))
    seed = (meta.get("seed") or [1, 1])[0]
    env = EngineEnv()
    env.seed(core=int(seed), disp=int(seed))
    env.reset(seeds=(int(seed), int(seed)))
    # count_visit=False: never write back, not even to the copy's meta, so the
    # probe is a pure read of restorability.
    checkpoint_restore(ckpt, env=env, audit=True, count_visit=False)
    raw = env._engine
    out = {"restored": True, "dlvl": int(raw.blstats[12]),
           "stepped": 0, "survived": False}
    for i in range(STEPS_AFTER):
        env.step(SEARCH)
        out["stepped"] = i + 1
    out["survived"] = True
    print("RESULT " + json.dumps(out))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        return probe_one(sys.argv[2])

    run = sys.argv[1]
    ids = sys.argv[2:]
    here = os.path.abspath(__file__)
    py = sys.executable
    tmp = tempfile.mkdtemp(prefix="ckhealth-")
    ok = crash = failed = 0
    print(f"{'ckpt':>6s} {'dlvl':>4s} {'verdict':22s} detail")
    try:
        for cid in ids:
            src = os.path.join(run, "archive", cid)
            if not os.path.isdir(src):
                print(f"{cid:>6s}    -  MISSING"); continue
            meta = json.load(open(os.path.join(src, "meta.json")))
            dst = os.path.join(tmp, cid)
            shutil.copytree(src, dst)
            r = subprocess.run([py, here, "--one", dst],
                               capture_output=True, text=True, timeout=300)
            line = next((l for l in r.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if r.returncode == 0 and line:
                ok += 1
                print(f"{cid:>6s} {meta.get('dlvl'):>4} {'ok: restored+stepped':22s} "
                      f"{json.loads(line[7:])}")
            elif r.returncode < 0:
                crash += 1
                sig = -r.returncode
                tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
                print(f"{cid:>6s} {meta.get('dlvl'):>4} {'CRASH signal '+str(sig):22s} {tail[0][:70]}")
            else:
                failed += 1
                tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
                print(f"{cid:>6s} {meta.get('dlvl'):>4} {'restore FAILED':22s} {tail[0][:70]}")
        print(f"\n  ok={ok}  crashed={crash}  restore-failed={failed}  of {len(ids)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
