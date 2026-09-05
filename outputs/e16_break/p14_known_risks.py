"""GOAL 6: verify or refute the three flagged known risks, and decide whether
each is reachable from the ONLY sequence E16 actually runs
(fresh process -> env.reset() -> checkpoint_restore(env=env) -> play)."""
import json, os, sys, shutil
CASE = sys.argv[1]
TMP = "/tmp/e16_p14"

def seed_ck():
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_save
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    env = EngineEnv(); env.reset(seeds=(5, 5)); env.step(13)
    env._engine.goto_depth(3); env._engine.goto_depth(2); env._engine.step(27)
    checkpoint_save(env, TMP + "/c1", name="p", note="")
    return env

if CASE == "a_restore_into_drained_gameover":
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_restore
    seed_ck()
    env = EngineEnv(); env.reset(seeds=(9, 9)); env.step(13)
    raw = env._engine
    raw.set_state("hp", 0)
    for i in range(400):
        raw.step(ord("s"))
        if raw.done: break
    print("victim env: done =", bool(raw.done), "how_done =", int(raw.how_done))
    for i in range(30):
        raw.step(13)
    print("drained. done =", bool(raw.done), "-> now restoring a checkpoint into it")
    sys.stdout.flush()
    env2, meta = checkpoint_restore(TMP + "/c1", env=env)
    from nethack_core.observations import BLSTATS_IDX
    print("SURVIVED. depth =", int(env2._engine.blstats[BLSTATS_IDX["depth"]]))

elif CASE == "a_e16_shape":
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_restore
    seed_ck()
    env = EngineEnv(); env.reset(seeds=(9, 9))
    env2, meta = checkpoint_restore(TMP + "/c1", env=env)
    from nethack_core.observations import BLSTATS_IDX
    print("E16 SHAPE OK. depth =", int(env2._engine.blstats[BLSTATS_IDX["depth"]]))

elif CASE == "b_hunger":
    from nethack_core.engine_env import EngineEnv
    env = EngineEnv(); env.reset(seeds=(5, 5)); env.step(13)
    print("calling modify(hunger=-400)"); sys.stdout.flush()
    obs = env.modify(hunger=-400)
    print("SURVIVED modify(hunger=-400)")
    for i in range(20):
        env._engine.step(ord("s"))
    print("SURVIVED 20 steps after; done =", bool(env._engine.done),
          "how_done =", int(env._engine.how_done))

elif CASE == "b_hunger_then_checkpoint":
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_save, checkpoint_restore
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    env = EngineEnv(); env.reset(seeds=(5, 5)); env.step(13)
    env.modify(hunger=-400)
    m = checkpoint_save(env, TMP + "/c1", name="starving", note="")
    print("saved with hunger poke:", {k: m[k] for k in ("dlvl","hp","gameturn")})
    sys.stdout.flush()
    e2, m2 = checkpoint_restore(TMP + "/c1")
    print("restored, fidelity ok:", m2["restore_fidelity"]["ok"])

elif CASE == "c_force_done":
    from nethack_core.engine_env import EngineEnv
    env = EngineEnv(); env.reset(seeds=(5, 5)); env.step(13)
    raw = env._engine
    for f, v in (("hp", 0), ("hpmax", 0)):
        try:
            raw.set_state(f, v); print(f"set_state({f},{v}) ok")
        except Exception as e:
            print(f"set_state({f},{v}) -> {type(e).__name__}: {e}")
    sys.stdout.flush()
    for i in range(60):
        raw.step(ord("s"))
        if raw.done: break
    print("done =", bool(raw.done), "how_done =", int(raw.how_done), "steps =", i)

elif CASE == "c_checkpoint_a_zombie":
    from nethack_core.engine_env import EngineEnv
    from nethack_core.observations import BLSTATS_IDX
    from nethack_harness.checkpoints import checkpoint_save, checkpoint_restore
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    env = EngineEnv(); env.reset(seeds=(5, 5)); env.step(13)
    raw = env._engine
    raw.goto_depth(2); raw.step(27)
    raw.set_state("hp", 0)
    print("live hp =", int(raw.blstats[BLSTATS_IDX["hitpoints"]]), "done =", bool(raw.done))
    m = checkpoint_save(env, TMP + "/c1", name="zombie", note="")
    print("SAVE ACCEPTED at hp=0:", {k: m[k] for k in ("dlvl","hp","max_hp","balrog")})
    sys.stdout.flush()
    e2, m2 = checkpoint_restore(TMP + "/c1")
    print("restored hp =", int(e2._engine.blstats[BLSTATS_IDX["hitpoints"]]),
          "fidelity ok:", m2["restore_fidelity"]["ok"])
