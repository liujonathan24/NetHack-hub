"""GOAL 6a, take 3: end the game for real with #quit, drain every game-over
screen, THEN restore a checkpoint into the same env."""
import os, sys, shutil
from nethack_core.engine_env import EngineEnv
from nethack_core.observations import BLSTATS_IDX
from nethack_harness.checkpoints import checkpoint_save, checkpoint_restore
TMP = "/tmp/e16_p16"; shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)

src = EngineEnv(); src.reset(seeds=(5, 5)); src.step(13)
src._engine.goto_depth(3); src._engine.goto_depth(2); src._engine.step(27)
checkpoint_save(src, TMP + "/c1", name="p", note="")

env = EngineEnv(); env.reset(seeds=(9, 9)); env.step(13)
raw = env._engine
for ch in "#quit\r":
    raw.step(ord(ch) if ch != "\r" else 13)
for ch in "yy":
    raw.step(ord(ch))
print("after #quit: done =", bool(raw.done), "how_done =", int(raw.how_done))
for j in range(60):
    raw.step(13)
    if j % 20 == 0:
        print(f"  drain {j}: done={bool(raw.done)} how={int(raw.how_done)} "
              f"screen0={''.join(chr(c) for c in raw.tty_chars[0]).rstrip()!r}")
print("FULLY DRAINED: done =", bool(raw.done), "how_done =", int(raw.how_done))
sys.stdout.flush()
env2, meta = checkpoint_restore(TMP + "/c1", env=env)
print("RESTORE INTO DRAINED GAME-OVER SURVIVED. depth =",
      int(env2._engine.blstats[BLSTATS_IDX["depth"]]),
      "fidelity ok:", meta["restore_fidelity"]["ok"])
