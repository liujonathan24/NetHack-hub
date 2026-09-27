"""Re-run the two headline attacks in a version-tolerant way, so the result is
valid against BOTH the committed checkpoints.py (da9f677, no restore-fidelity
audit) and the E16-step-3 working-tree version that adds one."""
import inspect, json, os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lfblob
import nethack_harness.checkpoints as ckmod
from nethack_harness.checkpoints import (checkpoint_save, checkpoint_restore,
                                         pack_bundle, unpack_bundle, STATE_BUNDLE)
from nethack_harness.integrity import (missing_level_files, CheckpointIntegrityError,
                                       assert_not_tricked, tricked_marker_in)
from nethack_core.engine_env import EngineEnv
from nethack_core.observations import BLSTATS_IDX

HAS_AUDIT = hasattr(ckmod, "restore_fidelity")
print(f"### checkpoints.py under test: {ckmod.__file__}")
print(f"### restore-fidelity audit present: {HAS_AUDIT}")
print(f"### checkpoint_restore signature: {inspect.signature(checkpoint_restore)}\n")

CASE = sys.argv[1]
TMP = "/tmp/e16_p19_" + CASE
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)

if CASE == "corrupt_levelfile_is_a_death":
    env = EngineEnv(); env.reset(seeds=(1, 1)); env.step(13)
    raw = env._engine
    raw.goto_depth(3); raw.goto_depth(4); raw.goto_depth(2)
    ck = TMP + "/c1"
    checkpoint_save(env, ck, name="victim", note="")
    hdr, sec = unpack_bundle(open(ck + "/" + STATE_BUNDLE, "rb").read())
    ents = []
    for name, data in lfblob.parse(sec["levelfiles"]):
        if name == ".1":
            b = bytearray(data); b[8:] = b"\x00" * (len(b) - 8); data = bytes(b)
        ents.append((name, data))
    sec["levelfiles"] = lfblob.build(ents)
    open(ck + "/" + STATE_BUNDLE, "wb").write(
        pack_bundle(sec, header_extra={"seed": hdr["seed"]}))
    print("corrupted the BODY of level file '.1'; every file still EXISTS")
    env2, m2 = checkpoint_restore(ck)
    print("checkpoint_restore -> ACCEPTED")
    print("  missing_level_files:", missing_level_files(env2))
    if HAS_AUDIT:
        print("  restore fidelity audit:", m2["restore_fidelity"]["ok"])
    r = env2._engine
    r.step(27); r.goto_depth(1)                       # first staircase
    obs = r.to_core_observation(); terminated = bool(r.done); bl = r.blstats
    from nethack_harness.helpers import _detect_terminal_outcome
    state = {"env": env2, "engine_error": None, "ascended": False, "died": False}
    gate = False
    if terminated or tricked_marker_in(obs):          # nethack.py:1871
        try:
            assert_not_tricked(state["env"], obs, where="after descend")
        except CheckpointIntegrityError as exc:
            gate = True; state["engine_error"] = str(exc)
    if not gate:
        _detect_terminal_outcome(obs, state)          # nethack.py:1890
        if terminated and not state["ascended"] and not state["died"]:
            state["died"] = True                      # nethack.py:1900
    print(json.dumps({
        "screen_line0": "".join(chr(c) for c in obs.tty_chars[0]).rstrip(),
        "terminated": terminated, "how_done": int(r.how_done),
        "hp": int(bl[BLSTATS_IDX["hitpoints"]]),
        "max_hp": int(bl[BLSTATS_IDX["max_hitpoints"]]),
        "E16_INTEGRITY_GATE_FIRED": gate,
        "state.engine_error": state["engine_error"],
        "state.died": state["died"],
    }, indent=2))

elif CASE == "meta_records_a_state_that_never_existed":
    env = EngineEnv(); env.reset(seeds=(11, 11)); env.step(13)
    raw = env._engine
    raw.goto_depth(3); raw.goto_depth(4); raw.goto_depth(2)
    scr = "\n".join("".join(chr(c) for c in r) for r in raw.tty_chars)
    print("at save time: --More-- pending =", "--More--" in scr)
    print("  screen line 0:", scr.split("\n")[0].rstrip())
    print("  live blstats depth =", int(raw.blstats[BLSTATS_IDX["depth"]]))
    meta = checkpoint_save(env, TMP + "/c1", name="liar", note="")
    print("  meta.json written  dlvl =", meta["dlvl"],
          " balrog =", meta["balrog"], " balrog_min =", meta["balrog_min"])
    try:
        env2, m2 = checkpoint_restore(TMP + "/c1")
        d = int(env2._engine.blstats[BLSTATS_IDX["depth"]])
        print("  restore -> ACCEPTED, engine depth =", d)
        print("  VERDICT: archive says dlvl", meta["dlvl"], "; the state that "
              "actually exists is dlvl", d,
              "-- ACCEPTED SILENTLY" if d != meta["dlvl"] else "")
    except CheckpointIntegrityError as e:
        print("  restore -> REJECTED:", str(e)[:180])
        print("  VERDICT: the poisoned meta is still IN THE ARCHIVE (the save "
              "succeeded); the checkpoint is now permanently unrestorable.")
