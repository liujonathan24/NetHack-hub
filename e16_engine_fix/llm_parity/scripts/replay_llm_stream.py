#!/usr/bin/env python3
"""Replay the RECORDED LLM ACTION-BYTE STREAM of treesmoke6/a001 against one
engine build, dumping the full observation after EVERY single action byte.

Run once per engine build; the two dumps must be byte-identical.

    ENG=<engine root> python3 replay_llm_stream.py <out_prefix>

Strategy (b): the rollback (`nle_fr_snapshot`/`restore` heap-snapshot rewind)
operations that make 7 of the 667 turn records non-replayable as a flat byte
stream ARE reproduced, by re-implementing the harness's rollback ring exactly
as environments/nethack/nethack_harness/tools/skills.py does it:

  * ROLLBACK_RING = 16 retained per-turn snapshots
  * push AFTER each turn record, only when hp > 0 (nethack.py fix2 guard)
  * rollback(n) -> idx = len(ring)-1-n, clamped to max_n = len(ring)-1,
    restore ring[idx], free+drop ring[idx+1:]
  * forced revive on death -> restore ring[-1] (nethack.py fix2)

Emitted per step, into <out_prefix>.ndjson:
    sha256 of tty_chars, tty_colors, glyphs, chars, colors,
            inv_strs, inv_letters, inv_glyphs
    tty_cursor, message, all 27 blstats, game clock, done, how_done  in clear
    "all" = sha256 over every one of those buffers concatenated
Raw tty_chars for every step also goes to <out_prefix>.tty.bin (24*80 bytes
per step) so a divergence can be rendered without re-running.
"""
import base64
import hashlib
import json
import os
import struct
import sys
import traceback
import zlib

import numpy as np

ENG = os.environ["ENG"]
# Pin BOTH knobs: sys.path decides which nethack_core package is imported, and
# NLE_LIB_PATH decides which libnethack.so that package loads. Leaving the
# second implicit lets the venv's _engine_paths.pth silently supply the other
# tree's .so -- the exact served-bytes drift class this check exists to stop.
os.environ["NLE_LIB_PATH"] = os.path.join(
    ENG, "third_party", "NetHack", "src", "build", "libnethack.so")
sys.path.insert(0, ENG)
from nethack_core.engine_env import EngineEnv  # noqa: E402
import nethack_core  # noqa: E402
from nethack_core import _engine as _eng_mod  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NDJSON = os.path.join(HERE, "1_2038677_1787958180.ndjson")
C1 = os.path.join(HERE, "archive_copy", "c1")
OUT = sys.argv[1]

ROLLBACK_RING = 16
SEEDS = (1, 1)
CHARACTER = "Val-hum-neu-fem"

ARRAYS = ("tty_chars", "tty_colors", "glyphs", "chars", "colors",
          "inv_strs", "inv_letters", "inv_glyphs")


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def tool_names(rec):
    out = []
    for tc in (rec.get("tool_calls") or []):
        out.append(tc.get("name") or (tc.get("function") or {}).get("name"))
    return out


def rollback_n(rec):
    for tc in (rec.get("tool_calls") or []):
        nm = tc.get("name") or (tc.get("function") or {}).get("name")
        if nm == "rollback":
            args = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            try:
                return max(1, int(args.get("n", 1)))
            except Exception:
                return 1
    return None


def decode_gt(g):
    """Decode a turn record's recorded ground-truth observation planes."""
    raw = zlib.decompress(base64.b64decode(g["data"]))
    out, off = {}, 0
    for name, dt, shape in g["planes"]:
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        out[name] = np.frombuffer(raw[off:off + n], dtype=dt).reshape(shape).copy()
        off += n
    return out


def engine_ident():
    """Prove which .so is actually mapped into THIS process."""
    so = str(_eng_mod.library_path())
    md5 = hashlib.md5(open(so, "rb").read()).hexdigest()
    return {"pkg": nethack_core.__file__, "so": so, "so_md5": md5,
            "so_mtime": os.stat(so).st_mtime, "so_size": os.path.getsize(so)}


def main():
    recs = [json.loads(l) for l in open(NDJSON) if l.strip()]

    ident = engine_ident()
    print("ENGINE:", json.dumps(ident, sort_keys=True))

    # START EXACTLY AS THE RECORDED RUN DID.
    # The run's config sets resume_checkpoint=<archive>/c1, so its first frame
    # is a CHECKPOINT RESTORE, not a fresh reset. That is not a cosmetic
    # difference: a fresh reset diverges from the recorded trajectory by turn
    # 4 (measured: pos (19,6) vs (19,5), clock 83 vs 85) and its botl AC reads
    # 6 where the restored game reads 0. Starting from c1 reproduces the
    # original trajectory exactly -- and has the bonus that the very first
    # thing under test is the checkpoint-restore path the fixes touch.
    sys.path.insert(0, "/root/nld/zombie-fix/environments/nethack")
    from nethack_harness.checkpoints import checkpoint_restore  # noqa: E402

    env = EngineEnv()
    env.reset(seeds=SEEDS, character=CHARACTER)
    env, _ckmeta = checkpoint_restore(C1, env=env, count_visit=False)
    obs = getattr(env, "_last_restore_obs", None) or env.engine.to_core_observation()

    ring = []          # [(record_idx, game_time, handle)]
    step = 0           # global action-byte index
    events = []        # non-step operations, for the log

    fh = open(OUT + ".ndjson", "w")
    tb = open(OUT + ".tty.bin", "wb")

    def dump(kind, rec_idx, action, note=""):
        nonlocal step
        raw = env.engine
        bl = [int(x) for x in np.asarray(obs.blstats)]
        row = {
            "step": step,
            "rec": rec_idx,
            "kind": kind,
            "action": action,
            "tty_cursor": [int(x) for x in np.asarray(obs.tty_cursor)],
            "message": bytes(np.asarray(obs.message)).split(b"\0")[0].decode("latin1"),
            "blstats": bl,
            "time": bl[20],
            "done": bool(env.done),
            "how_done": int(raw.how_done),
        }
        cat = hashlib.sha256()
        for name in ARRAYS:
            a = np.ascontiguousarray(np.asarray(getattr(obs, name)))
            row[name] = hashlib.sha256(a.tobytes()).hexdigest()
            cat.update(a.tobytes())
        cat.update(np.ascontiguousarray(np.asarray(obs.tty_cursor)).tobytes())
        cat.update(np.ascontiguousarray(np.asarray(obs.message)).tobytes())
        cat.update(np.ascontiguousarray(np.asarray(obs.blstats)).tobytes())
        cat.update(struct.pack("<i?", int(raw.how_done), bool(env.done)))
        row["all"] = cat.hexdigest()
        if note:
            row["note"] = note
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        tb.write(np.ascontiguousarray(np.asarray(obs.tty_chars), dtype=np.uint8).tobytes())
        step += 1

    # frame 0 = the restored c1 state, before any recorded byte.
    # checkpoint_restore already performs its own welcome---More--- dismissal
    # (reset -> ESC -> load level/player/levelfiles -> ctrl-R), so no extra
    # prologue byte is needed here.
    dump("restore_c1", -1, None, note="checkpoint_restore(c1) seeds=(1,1) Val-hum-neu-fem")

    gt_first_mismatch = None
    gt_checked = 0
    gt_ok = 0

    status = "ok"
    try:
        for i, rec in enumerate(recs):
            a = rec["actions"]
            byts = a.get("bytes") or []
            replayable = bool(a.get("replayable"))
            names = tool_names(rec)

            # ---- reproduce the rollback rewind BEFORE this record's bytes ----
            if not replayable and "rollback" in names:
                n = rollback_n(rec) or 1
                max_n = len(ring) - 1
                if max_n < 1:
                    events.append({"rec": i, "step": step, "op": "rollback",
                                   "n": n, "result": "unavailable"})
                else:
                    n_eff = min(n, max_n)
                    idx = len(ring) - 1 - n_eff
                    tgt_rec, tgt_gt, handle = ring[idx]
                    env.restore(handle)
                    for _r, _g, hh in ring[idx + 1:]:
                        try:
                            env.free_snapshot(hh)
                        except Exception:
                            pass
                    del ring[idx + 1:]
                    events.append({"rec": i, "step": step, "op": "rollback",
                                   "n": n, "n_eff": n_eff, "to_rec": tgt_rec,
                                   "to_time": tgt_gt})
                    dump("rollback", i, None,
                         note=f"restored ring[{idx}] (rec {tgt_rec}, time {tgt_gt}) n={n} n_eff={n_eff}")

            # ---- apply this record's recorded action bytes, one at a time ----
            for b in byts:
                obs_new, done, _info = env.step(int(b))
                obs = obs_new
                dump("step", i, int(b))

                # Death handling, mirroring nethack.py's FORCED ROLLBACK ON
                # DEATH (fix2). Not optional bookkeeping: without it the driver
                # keeps feeding keystrokes into a finished game, which walks the
                # engine into a zombie state (clock 0, hp 0, dlvl 0) and then
                # SIGABRTs inside nle_fr_restore. hp<=0 is tested as well as
                # `done`, because a zero-HP death that the terminated flag
                # misses arrives here with done still False -- the same case
                # nethack.py's hp>0 snapshot guard exists for. The remaining
                # bytes of the record are still applied: in the recorded stream
                # the trailing <ESC> IS the post-revive materializer.
                hp = int(np.asarray(obs.blstats)[10])
                if env.done or hp <= 0:
                    if not ring:
                        events.append({"rec": i, "step": step, "op": "final_death",
                                       "reason": "no snapshot to revive into"})
                        status = "final_death"
                        break
                    tgt_rec, tgt_gt, handle = ring[-1]
                    obs = env.restore(handle) or obs
                    events.append({"rec": i, "step": step, "op": "forced_revive",
                                   "to_rec": tgt_rec, "to_time": tgt_gt,
                                   "was_done": bool(done), "hp": hp})
                    dump("revive", i, None,
                         note=f"forced revive -> ring[-1] (rec {tgt_rec}, time {tgt_gt})")
            if status == "final_death":
                break

            # ---- fidelity: does this replay still track the ORIGINAL run? ----
            # Purely diagnostic -- it never steers the replay. gt_obs is the
            # post-action observation the harness recorded for this turn.
            if rec.get("gt_obs"):
                gt = decode_gt(rec["gt_obs"])
                bad = []
                for nm in ("glyphs", "chars", "colors", "inv_letters",
                           "inv_glyphs", "inv_strs"):
                    if not np.array_equal(np.asarray(getattr(obs, nm)), gt[nm]):
                        bad.append(nm)
                gbl = gt["blstats"]
                obl = np.asarray(obs.blstats)
                for k in range(27):
                    if int(obl[k]) != int(gbl[k]):
                        bad.append(f"blstats[{k}]")
                gt_checked += 1
                if bad:
                    if gt_first_mismatch is None:
                        gt_first_mismatch = {"rec": i, "step": step,
                                             "fields": bad[:12],
                                             "replay_time": int(obl[20]),
                                             "gt_time": int(gbl[20])}
                else:
                    gt_ok += 1

            # ---- push post-turn snapshot (hp>0 guard) ----
            bl = [int(x) for x in np.asarray(obs.blstats)]
            if bl[10] > 0 and not env.done:
                ring.append((i, bl[20], env.snapshot()))
                while len(ring) > ROLLBACK_RING:
                    _r, _g, hh = ring.pop(0)
                    try:
                        env.free_snapshot(hh)
                    except Exception:
                        pass
    except BaseException as exc:
        status = "EXCEPTION"
        events.append({"rec": i, "step": step, "op": "exception",
                       "type": type(exc).__name__, "msg": str(exc),
                       "tb": traceback.format_exc()})
        sys.stderr.write(traceback.format_exc())
    finally:
        fh.close()
        tb.close()
        with open(OUT + ".events.json", "w") as ef:
            json.dump({"engine": ENG, "ident": ident, "status": status,
                       "frames_written": step,
                       "gt_checked": gt_checked, "gt_ok": gt_ok,
                       "gt_first_mismatch": gt_first_mismatch,
                       "events": events}, ef, indent=1)

    print(f"ENG={ENG} status={status} frames={step} "
          f"gt_ok={gt_ok}/{gt_checked} first_gt_mismatch={gt_first_mismatch}",
          flush=True)


if __name__ == "__main__":
    main()
