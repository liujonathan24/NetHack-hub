"""Does the PROACTIVE guard actually engage on the env the E16 harness uses?

integrity.missing_level_files() returns None on any env it cannot walk down to
a RawEngine, or if the engine call raises; assert_dungeon_on_disk then does
`if n:` -- None is falsy, so the guard PASSES SILENTLY. This checks the real
production env class (nethack_core.env.NetHackCoreEnv, the one
e16_orchestrator.seed_archive and nethack.py hand to checkpoint_save/restore).
"""
import os, sys, shutil, json
from nethack_harness.integrity import missing_level_files, assert_dungeon_on_disk, _raw_engine
from nethack_harness.checkpoints import _engine_of, checkpoint_save

print("--- production env: nethack_core.env.NetHackCoreEnv ---")
from nethack_core.env import NetHackCoreEnv
env = NetHackCoreEnv(task_name="NetHackChallenge-v0", max_episode_steps=100_000)
env.seed(core=5, disp=5)
env.reset(character="Val-hum-neu-fem")
print("  type(env)              :", type(env).__name__)
print("  _engine_of(env)        :", type(_engine_of(env)).__name__)
print("  _raw_engine(env)       :", type(_raw_engine(env)).__name__ if _raw_engine(env) else None)
n = missing_level_files(env)
print("  missing_level_files    :", n, "  <-- None means the guard is a NO-OP")
print("  guard engages          :", n is not None)

print("\n--- and on the engine_env the guard is actually called with ---")
eng = _engine_of(env)
n2 = missing_level_files(eng)
print("  missing_level_files(engine_env):", n2, " guard engages:", n2 is not None)

print("\n--- fail-open demo: an engine whose introspection raises ---")
class Boom:
    def snapshot(self): pass
    def missing_level_files(self): raise RuntimeError("no active game")
class Wrap:
    def __init__(self): self._engine = Boom()
    def checkpoint(self): pass
w = Wrap()
print("  missing_level_files(Wrap):", missing_level_files(w))
try:
    assert_dungeon_on_disk(w, where="fail-open demo")
    print("  assert_dungeon_on_disk: PASSED (no exception) <-- FAIL-OPEN")
except Exception as e:
    print("  assert_dungeon_on_disk raised:", type(e).__name__, e)

print("\n--- fail-open demo 2: env wrapper nested 5 deep (walk limit is 4) ---")
class L:
    def __init__(self, inner): self._engine = inner
deep = L(L(L(L(L(Boom())))))
print("  _raw_engine(deep)       :", _raw_engine(deep))
print("  missing_level_files(deep):", missing_level_files(deep))
try:
    assert_dungeon_on_disk(deep, where="deep wrapper")
    print("  assert_dungeon_on_disk: PASSED (no exception) <-- FAIL-OPEN")
except Exception as e:
    print("  raised:", type(e).__name__)
