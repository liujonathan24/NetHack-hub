"""GOAL 4: NO engine pokes. Natural play only (moves + search + '>'), save,
resume cross-process, diff all 27 blstats + inventory + glyph map. This is the
control that says how much of the checkpoint layer actually works."""
import json, os, subprocess, sys, shutil
TMP = "/tmp/e16_p11"
AUDITED = {"depth", "experience_level", "hitpoints", "max_hitpoints", "time", "score"}

def snap(raw):
    from nethack_core.observations import BLSTATS_IDX
    bl = raw.blstats
    return {"blstats": {k: int(bl[i]) for k, i in BLSTATS_IDX.items()},
            "glyphs": [int(x) for x in raw.glyphs.reshape(-1)],
            "inv": [bytes(r).split(b"\0")[0].decode("ascii", "replace")
                    for r in raw._inv_strs.reshape(55, 80)],
            "more": "--More--" in "\n".join("".join(chr(c) for c in r)
                                            for r in raw.tty_chars)}

def play(raw, n):
    import random
    rnd = random.Random(1234)
    keys = [ord("h"), ord("j"), ord("k"), ord("l"), ord("s"), ord(">")]
    for _ in range(n): raw.step(rnd.choice(keys))

if len(sys.argv) > 1 and sys.argv[1] == "save":
    seed = int(sys.argv[2]); d = "%s/s%d" % (TMP, seed)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.checkpoints import checkpoint_save
    env = EngineEnv(); env.reset(seeds=(seed, seed)); env.step(13)
    play(env._engine, 120)
    s = snap(env._engine)
    meta = checkpoint_save(env, d + "/c1", name="nat", note="")
    s["meta"] = {k: meta[k] for k in ("dlvl", "xl", "hp", "max_hp", "gameturn", "score")}
    json.dump(s, open(d + "/before.json", "w"))
elif len(sys.argv) > 1 and sys.argv[1] == "restore":
    seed = int(sys.argv[2]); d = "%s/s%d" % (TMP, seed)
    from nethack_harness.checkpoints import checkpoint_restore
    try:
        env, meta = checkpoint_restore(d + "/c1")
        s = snap(env._engine)
    except Exception as e:
        json.dump({"error": "%s: %s" % (type(e).__name__, e)}, open(d + "/after.json", "w"))
        raise SystemExit(0)
    json.dump(s, open(d + "/after.json", "w"))
else:
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    for seed in [1, 2, 3, 5, 7, 11, 13, 17]:
        for role in ("save", "restore"):
            p = subprocess.run([sys.executable, __file__, role, str(seed)],
                               env=dict(os.environ), capture_output=True, text=True, timeout=900)
            if p.returncode: print("seed %d %s rc=%s\n%s" % (seed, role, p.returncode, p.stderr[-400:]))
        d = "%s/s%d" % (TMP, seed)
        b = json.load(open(d + "/before.json")); a = json.load(open(d + "/after.json"))
        if "error" in a:
            print("seed %d: restore error -> %s" % (seed, a["error"][:140])); continue
        bd = {k: (b["blstats"][k], a["blstats"][k]) for k in b["blstats"]
              if b["blstats"][k] != a["blstats"][k]}
        print(json.dumps({"seed": seed, "more_pending_at_save": b["more"],
                          "blstat_diffs": bd,
                          "blstat_diffs_audited": sorted(set(bd) & AUDITED),
                          "blstat_diffs_unaudited": sorted(set(bd) - AUDITED),
                          "glyph_cells_differing":
                              sum(1 for x, y in zip(b["glyphs"], a["glyphs"]) if x != y),
                          "inventory_differs": b["inv"] != a["inv"]}))
