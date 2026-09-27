"""GOAL 6a with a REAL death: die for real, drain the game-over screens fully,
then restore a checkpoint into that same env."""
import os, sys, shutil
from nethack_core.engine_env import EngineEnv
from nethack_core.observations import BLSTATS_IDX
from nethack_harness.checkpoints import checkpoint_save, checkpoint_restore
TMP = "/tmp/e16_p15"; shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)

src = EngineEnv(); src.reset(seeds=(5, 5)); src.step(13)
src._engine.goto_depth(3); src._engine.goto_depth(2); src._engine.step(27)
checkpoint_save(src, TMP + "/c1", name="p", note="")
print("checkpoint written")

env = EngineEnv(); env.reset(seeds=(9, 9)); env.step(13)
raw = env._engine
raw.goto_depth(10); raw.step(27)          # XL1 hero, deep level: dies fast
for i in range(3000):
    raw.step(ord("s"))
    if raw.done: break
print("real death after", i, "steps: done =", bool(raw.done),
      "how_done =", int(raw.how_done),
      "hp =", int(raw.blstats[BLSTATS_IDX["hitpoints"]]))
if not raw.done:
    print("could not produce a genuine game-over"); raise SystemExit(0)
for j in range(40):                        # FULLY drain the game-over screens
    raw.step(13)
print("drained; done =", bool(raw.done))
sys.stdout.flush()
env2, meta = checkpoint_restore(TMP + "/c1", env=env)
print("RESTORE INTO A DRAINED GAME-OVER SURVIVED. depth =",
      int(env2._engine.blstats[BLSTATS_IDX["depth"]]),
      "fidelity ok:", meta["restore_fidelity"]["ok"])
