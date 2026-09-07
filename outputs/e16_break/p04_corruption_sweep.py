"""ATTACK 2: sweep of level-file BODY corruptions inside state.bundle.

The proactive guard (integrity.assert_dungeon_on_disk -> nle_fr_missing_levelfiles,
nle_fast_reset.c:459) only stat()s each "<base>.<ledger>" path. A bundle whose
files all EXIST but whose CONTENT is damaged therefore restores cleanly, and the
damage only lands on the next staircase.

Usage: python p04_corruption_sweep.py <mode>
modes: zero_body trunc_16 trunc_half empty flip_hdr flip_mid swap_pid drop_file
"""
import json, os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lfblob
from nethack_core.engine_env import EngineEnv
from nethack_core.observations import BLSTATS_IDX
from nethack_harness.checkpoints import (checkpoint_save, checkpoint_restore,
                                         pack_bundle, unpack_bundle, STATE_BUNDLE)
from nethack_harness.integrity import (missing_level_files, CheckpointIntegrityError,
                                       assert_not_tricked, tricked_marker_in)
from nethack_harness.helpers import _detect_terminal_outcome

MODE = sys.argv[1]
TMP = "/tmp/e16_p04_" + MODE
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)

def mutate(b):
    b = bytearray(b)
    if MODE == "zero_body":    b[8:] = b"\x00" * (len(b) - 8)
    elif MODE == "trunc_16":   b = b[:16]
    elif MODE == "trunc_half": b = b[:len(b) // 2]
    elif MODE == "empty":      b = bytearray()
    elif MODE == "flip_hdr":   b[8:12] = bytes(x ^ 0xFF for x in b[8:12])
    elif MODE == "flip_mid":   b[len(b) // 2] ^= 0xFF
    elif MODE == "swap_pid":   b[0:4] = b"\xde\xad\xbe\xef"
    elif MODE == "drop_file":  return None
    return bytes(b)

env = EngineEnv(); env.reset(seeds=(1, 1)); env.step(13)
raw = env._engine
raw.goto_depth(3); raw.goto_depth(4); raw.goto_depth(2)
ck = TMP + "/c1"
checkpoint_save(env, ck, name="v", note="")
hdr, sec = unpack_bundle(open(ck + "/" + STATE_BUNDLE, "rb").read())
ents = []
for name, data in lfblob.parse(sec["levelfiles"]):
    if name == ".1":
        d = mutate(data)
        if d is None:
            continue
        data = d
    ents.append((name, data))
sec["levelfiles"] = lfblob.build(ents)
open(ck + "/" + STATE_BUNDLE, "wb").write(
    pack_bundle(sec, header_extra={"seed": hdr["seed"]}))

row = {"mode": MODE}
try:
    env2, m2 = checkpoint_restore(ck)
except CheckpointIntegrityError as e:
    row.update(restore="REJECTED(CheckpointIntegrityError)", detail=str(e)[:120])
    print(json.dumps(row)); raise SystemExit(0)
except Exception as e:
    row.update(restore="REJECTED(%s)" % type(e).__name__, detail=str(e)[:120])
    print(json.dumps(row)); raise SystemExit(0)

row["restore"] = "ACCEPTED"
row["missing_level_files"] = missing_level_files(env2)
if "restore_fidelity" in m2:
    row["fidelity_ok"] = m2["restore_fidelity"]["ok"]
r = env2._engine
r.step(27); r.goto_depth(1)
obs = r.to_core_observation(); terminated = bool(r.done); bl = r.blstats
state = {"env": env2, "engine_error": None, "ascended": False, "died": False}
gate = False
if terminated or tricked_marker_in(obs):
    try:
        assert_not_tricked(state["env"], obs, where="after descend")
    except CheckpointIntegrityError as exc:
        gate = True; state["engine_error"] = str(exc)[:80]
if not gate:
    _detect_terminal_outcome(obs, state)
    if terminated and not state["ascended"] and not state["died"]:
        state["died"] = True
row.update(terminated=terminated, how_done=int(r.how_done),
           hp=int(bl[BLSTATS_IDX["hitpoints"]]),
           maxhp=int(bl[BLSTATS_IDX["max_hitpoints"]]),
           depth=int(bl[BLSTATS_IDX["depth"]]),
           screen0="".join(chr(c) for c in obs.tty_chars[0]).rstrip(),
           GATE_FIRED=gate, died=state["died"])
print(json.dumps(row))
