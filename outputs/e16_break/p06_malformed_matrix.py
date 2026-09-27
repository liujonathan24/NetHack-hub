"""ATTACK 2/3: every way a checkpoint DIRECTORY can be damaged, and what class
of exception (if any) comes out. Usage: python p06_malformed_matrix.py <mode>"""
import json, os, sys, shutil
from nethack_core.engine_env import EngineEnv
from nethack_harness.checkpoints import (checkpoint_save, checkpoint_restore,
                                         checkpoint_list, STATE_BUNDLE, META_JSON)
from nethack_harness.integrity import CheckpointIntegrityError

TMP = "/tmp/e16_p06"; shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
env = EngineEnv(); env.reset(seeds=(1, 1)); env.step(13)
env._engine.goto_depth(3); env._engine.goto_depth(2)
GOLD = TMP + "/gold"
checkpoint_save(env, GOLD, name="gold", note="")
gold_bundle = open(GOLD + "/" + STATE_BUNDLE, "rb").read()
gold_meta = open(GOLD + "/" + META_JSON, "rb").read()

MODES = {}
def mode(fn): MODES[fn.__name__] = fn; return fn

@mode
def bundle_truncated_1pct(d):
    open(d + "/" + STATE_BUNDLE, "wb").write(gold_bundle[:int(len(gold_bundle) * .99)])
@mode
def bundle_truncated_to_header(d):
    open(d + "/" + STATE_BUNDLE, "wb").write(gold_bundle[:200])
@mode
def bundle_empty(d): open(d + "/" + STATE_BUNDLE, "wb").write(b"")
@mode
def bundle_missing(d): os.unlink(d + "/" + STATE_BUNDLE)
@mode
def bundle_one_bitflip_in_player(d):
    b = bytearray(gold_bundle); b[len(b) // 2] ^= 0x01
    open(d + "/" + STATE_BUNDLE, "wb").write(bytes(b))
@mode
def bundle_trailing_garbage(d):
    open(d + "/" + STATE_BUNDLE, "wb").write(gold_bundle + b"\xff" * 4096)
@mode
def meta_truncated(d):
    open(d + "/" + META_JSON, "wb").write(gold_meta[:len(gold_meta) // 2])
@mode
def meta_empty(d): open(d + "/" + META_JSON, "wb").write(b"")
@mode
def meta_missing(d): os.unlink(d + "/" + META_JSON)
@mode
def meta_lies_about_dlvl(d):
    m = json.loads(gold_meta); m["dlvl"] = 17
    open(d + "/" + META_JSON, "w").write(json.dumps(m, indent=2))
@mode
def meta_drops_audited_field(d):
    m = json.loads(gold_meta); m.pop("score")
    open(d + "/" + META_JSON, "w").write(json.dumps(m, indent=2))
@mode
def meta_lies_about_bundle_bytes(d):
    m = json.loads(gold_meta); m["bundle_bytes"] = 12
    open(d + "/" + META_JSON, "w").write(json.dumps(m, indent=2))
@mode
def meta_lies_about_seed(d):
    m = json.loads(gold_meta); m["seed"] = [999, 999]
    open(d + "/" + META_JSON, "w").write(json.dumps(m, indent=2))

name = sys.argv[1]
d = TMP + "/" + name
shutil.rmtree(d, ignore_errors=True); shutil.copytree(GOLD, d)
MODES[name](d)
row = {"mode": name,
       "listed_by_checkpoint_list":
           name in [os.path.basename(str(p)) for p in checkpoint_list(TMP)]}
try:
    env2, m2 = checkpoint_restore(d, count_visit=False)
    from nethack_core.observations import BLSTATS_IDX
    row["result"] = "ACCEPTED"
    row["engine_depth"] = int(env2._engine.blstats[BLSTATS_IDX["depth"]])
    row["meta_dlvl"] = m2.get("dlvl")
    if "restore_fidelity" in m2:
        row["fidelity_ok"] = m2["restore_fidelity"]["ok"]
except CheckpointIntegrityError as e:
    row["result"] = "CheckpointIntegrityError"; row["msg"] = str(e)[:110]
except Exception as e:
    row["result"] = type(e).__name__ + "  <-- NOT a CheckpointIntegrityError"
    row["msg"] = str(e)[:110]
print(json.dumps(row))
