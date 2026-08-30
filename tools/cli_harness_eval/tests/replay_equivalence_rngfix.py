#!/usr/bin/env python3
"""Does carrying the ISAAC64 state across a checkpoint close the replay gap?

`replay_equivalence.py` measured the gap: a restored game is byte-identical to
the original AT the checkpoint (blstats and full map agree, 0 differing cells)
and then diverges on the very first step. Two independent restores agree with
each other 120/120, so the restore is deterministic -- it is deterministically
replaying a DIFFERENT random stream. That is exactly what checkpoints.py:48
says: resume re-seeds from the saved (core, disp) and replays no history.

This script tests the proposed fix. The engine's RNG lives in
`current_nle_ctx->rng_state[2]` (see src/nle.c:144), each entry a plain
`struct isaac64_ctx { unsigned n; uint64_t r[256], m[256], a, b, c; }` --
4128 bytes, no pointers, so it is memcpy-able and portable across processes.

  leg A  play, checkpoint, and ALSO dump both isaac64_ctx blobs beside it
  leg B  restore, memcpy the blobs back, then replay the tail

`nle_rng_state(idx)` dereferences the CURRENT ctx, so it must be called
through the engine's own loaded library handle (RawEngine._lib) while the
target game is live -- loading a second copy of libnethack.so would hand back
a different process image's globals.

  python replay_equivalence_rngfix.py
"""
import ctypes, json, os, shutil, subprocess, sys

REPO = "/root/nld/zombie-fix"
ENG = "/root/nld/e16-engine"
for p in (f"{REPO}/tools/pycompat", ENG, REPO, f"{REPO}/environments/nethack",
          "/root/nld/e16_runs"):
    if p not in sys.path:
        sys.path.insert(0, p)

from replay_equivalence import (action_list, DISMISS, CHECKPOINT_AT, N_STEPS,
                                SEED, make_env, raw_of, obs_digest)

CK = "/tmp/e16_equiv/ckpt_rng"
RNG_BLOB = CK + ".rng"
ISAAC64_SZ = 4128          # verified: sizeof(struct isaac64_ctx)
N_STREAMS = 2              # CORE and DISP


def _rng_ptr(raw, idx):
    fn = raw._lib.nle_rng_state
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int]
    return fn(idx)


def dump_rng(raw):
    return b"".join(ctypes.string_at(_rng_ptr(raw, i), ISAAC64_SZ)
                    for i in range(N_STREAMS))


def load_rng(raw, blob):
    assert len(blob) == ISAAC64_SZ * N_STREAMS, len(blob)
    for i in range(N_STREAMS):
        chunk = blob[i * ISAAC64_SZ:(i + 1) * ISAAC64_SZ]
        ctypes.memmove(_rng_ptr(raw, i), chunk, ISAAC64_SZ)


def leg_a():
    from nethack_harness.checkpoints import (checkpoint_save,
                                             CheckpointSavepointError)
    env = make_env(); env.seed(core=SEED, disp=SEED)
    env.reset(seeds=(SEED, SEED))
    raw = raw_of(env)
    for d in DISMISS:
        env.step(d)
    acts = action_list(N_STEPS)
    rows = []
    for i, a in enumerate(acts, start=1):
        env.step(a)
        rows.append(obs_digest(None, raw))
        if i == CHECKPOINT_AT:
            if os.path.isdir(CK):
                shutil.rmtree(CK)
            for _ in range(8):
                try:
                    checkpoint_save(env, CK, name="rngfix-probe")
                    break
                except CheckpointSavepointError:
                    env.step(27)
                    rows[-1] = obs_digest(None, raw)
            else:
                raise SystemExit("no clean savepoint")
            # The blob is taken AFTER the save so it reflects the same moment
            # the bundle describes; checkpoint_save does not step the engine.
            open(RNG_BLOB, "wb").write(dump_rng(raw))
    json.dump({"rows": rows}, open("/tmp/e16_equiv/a_rng.json", "w"))
    print(f"[leg A] {len(rows)} steps; rng blob {os.path.getsize(RNG_BLOB)} bytes")


def leg_b(apply_fix):
    from nethack_harness.checkpoints import checkpoint_restore
    env = make_env(); env.seed(core=SEED, disp=SEED)
    env.reset(seeds=(SEED, SEED))
    raw = raw_of(env)
    checkpoint_restore(CK, env=env, audit=True)
    if apply_fix:
        # AFTER the restore, never before: the load would overwrite it, the
        # same ordering constraint checkpoint_restore documents for `reseed`.
        load_rng(raw, open(RNG_BLOB, "rb").read())
    rows = [obs_digest(None, raw) for a in action_list(N_STEPS)[CHECKPOINT_AT:]
            if (env.step(a) or True)]
    tag = "fix" if apply_fix else "nofix"
    json.dump({"rows": rows}, open(f"/tmp/e16_equiv/b_{tag}.json", "w"))
    print(f"[leg B {tag}] restored and stepped {len(rows)} steps")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "a":
            return leg_a()
        return leg_b(sys.argv[1] == "fix")

    os.makedirs("/tmp/e16_equiv", exist_ok=True)
    py = sys.executable
    here = os.path.abspath(__file__)
    for arg in ("a", "nofix", "fix"):
        r = subprocess.run([py, here, arg], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2500:])
            raise SystemExit(f"leg {arg} failed")

    A = json.load(open("/tmp/e16_equiv/a_rng.json"))["rows"][CHECKPOINT_AT:]
    out = {}
    for tag in ("nofix", "fix"):
        B = json.load(open(f"/tmp/e16_equiv/b_{tag}.json"))["rows"]
        n = min(len(A), len(B))
        same = sum(1 for i in range(n) if A[i] == B[i])
        first = next((i for i in range(n) if A[i] != B[i]), None)
        out[tag] = (same, n, first)

    print("\n" + "=" * 66)
    print("restore vs from-scratch replay, post-checkpoint steps")
    for tag, (same, n, first) in out.items():
        label = {"nofix": "today (seed only)", "fix": "with ISAAC64 state"}[tag]
        where = "none" if first is None else f"step {CHECKPOINT_AT+1+first}"
        print(f"  {label:22s} identical {same:3d}/{n}   first diff: {where}")
    print("=" * 66)


if __name__ == "__main__":
    main()
