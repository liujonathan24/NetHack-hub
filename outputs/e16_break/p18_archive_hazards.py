"""Archive-level hazards: concurrent saves, one bad meta.json poisoning the
whole listing, and level-file names in a bundle used unvalidated as paths."""
import json, os, sys, shutil, struct, subprocess, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lfblob
TMP = "/tmp/e16_p18"

def build():
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_save
    env = EngineEnv(); env.reset(seeds=(5, 5)); env.step(13)
    env._engine.goto_depth(3); env._engine.goto_depth(2); env._engine.step(27)
    return env

CASE = sys.argv[1]

if CASE == "list_poisoned_by_one_bad_meta":
    from nethack_harness.checkpoints import checkpoint_save, checkpoint_list, META_JSON
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    env = build()
    for i in (1, 2, 3):
        checkpoint_save(env, f"{TMP}/c{i}", name=f"c{i}", note="")
    print("healthy archive ->", [os.path.basename(str(p)) for p in checkpoint_list(TMP)])
    # ONE checkpoint's meta.json is damaged (torn write, bad copy, disk error)
    open(f"{TMP}/c2/{META_JSON}", "w").write('{"id": "2", "dlvl":')
    try:
        print("after damaging c2 ->",
              [os.path.basename(str(p)) for p in checkpoint_list(TMP)])
    except Exception as e:
        print(f"after damaging c2 -> checkpoint_list RAISED {type(e).__name__}: {e}")
        print("   ==> c1 and c3, which are perfectly healthy, are now unreachable")

elif CASE == "concurrent_save":
    from nethack_harness.checkpoints import checkpoint_save, checkpoint_meta, checkpoint_list
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    child = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_p18_child.py")
    open(child, "w").write(
        "import os, sys\n"
        "from nethack_core.engine_env import EngineEnv\n"
        "from nethack_harness.checkpoints import checkpoint_save\n"
        "env = EngineEnv(); env.reset(seeds=(int(sys.argv[2]), int(sys.argv[2])))\n"
        "env.step(13); env._engine.goto_depth(int(sys.argv[3])); env._engine.step(27)\n"
        "try:\n"
        "    m = checkpoint_save(env, sys.argv[1], name=sys.argv[2], note='')\n"
        "    print('OK', sys.argv[2], m['dlvl'])\n"
        "except Exception as e:\n"
        "    print('ERR', sys.argv[2], type(e).__name__, str(e)[:80])\n")
    ps = [subprocess.Popen([sys.executable, child, f"{TMP}/c1", s, d],
                           env=dict(os.environ), stdout=subprocess.PIPE, text=True)
          for s, d in (("5", "2"), ("9", "4"))]
    for p in ps:
        print(" ", "\n  ".join(l for l in p.communicate()[0].splitlines()
                               if not l.startswith("DEF_MREAD")))
    print("dir now:", sorted(os.listdir(TMP)))
    print("c1 files:", sorted(os.listdir(TMP + "/c1")))
    m = checkpoint_meta(TMP + "/c1")
    print("c1 meta: name=", m["name"], "dlvl=", m["dlvl"])
    from nethack_harness.checkpoints import checkpoint_restore
    e, mm = checkpoint_restore(TMP + "/c1", count_visit=False)
    from nethack_core.observations import BLSTATS_IDX
    print("c1 restores to depth", int(e._engine.blstats[BLSTATS_IDX["depth"]]),
          "| fidelity ok:", mm["restore_fidelity"]["ok"])

elif CASE == "path_traversal":
    from nethack_harness.checkpoints import (checkpoint_save, checkpoint_restore,
                                             pack_bundle, unpack_bundle, STATE_BUNDLE)
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    env = build()
    checkpoint_save(env, TMP + "/c1", name="t", note="")
    hdr, sec = unpack_bundle(open(f"{TMP}/c1/{STATE_BUNDLE}", "rb").read())
    ents = lfblob.parse(sec["levelfiles"])
    victim = TMP + "/PWNED_BY_A_CHECKPOINT"
    ents.append(("../../../.." + victim, b"written by nle_fr_restore_levelfiles\n"))
    sec["levelfiles"] = lfblob.build(ents)
    open(f"{TMP}/c1/{STATE_BUNDLE}", "wb").write(
        pack_bundle(sec, header_extra={"seed": hdr["seed"]}))
    print("bundle now carries entry:", ents[-1][0])
    try:
        checkpoint_restore(TMP + "/c1", count_visit=False)
        print("restore returned normally")
    except Exception as e:
        print("restore ->", type(e).__name__, str(e)[:120])
    print("victim file written outside the hackdir:", os.path.exists(victim))
    if os.path.exists(victim):
        print("   contents:", repr(open(victim).read().strip()))
