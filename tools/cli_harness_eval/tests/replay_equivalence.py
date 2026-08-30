#!/usr/bin/env python3
"""Does checkpoint-restore reproduce a from-scratch replay, byte for byte?

THE QUESTION. `checkpoints.py:48` records that a checkpoint does NOT carry the
RNG stream: "Resume re-seeds from the saved (core, disp) and replays no
history, so object appearances/ids match but future random rolls do not
continue the original stream. Two resumes of one checkpoint agree with each
other, not with the original run's future."

That is a claim about fidelity, and it has never been measured against
from-scratch ground truth on the SAVE-BUNDLE path. (`RawEngine.snapshot()` --
the in-memory rollback path -- does carry that verification, but its handle is
pointer-bound to one live engine and cannot cross a process boundary, which is
exactly why the archive uses save bundles instead.)

THE TEST. One fixed action list, no model, no inference.
  leg A  fresh game on seed S, step the whole list, hashing the observation
         after every step; checkpoint at step K on the way past.
  leg B  a SEPARATE PROCESS restores that checkpoint and steps the tail of the
         same list, hashing the same way.
  diff   leg A's hashes for steps K+1..N against leg B's.

Identical hashes mean a restored game is substitutable for a replayed one.
The first differing step is where the divergence begins.

WHY A SIMPLE ACTION LIST IS ENOUGH. Monsters act, and monster movement draws
from the core RNG every turn. A stream restarted at position 0 therefore shows
up within a few turns; reaching a staircase is not required to expose it.

WHY A SEPARATE PROCESS FOR LEG B. In-process, a restore could appear faithful
by inheriting live engine state. The archive's real use is resumption after the
original process is gone, so the test has to match that.

  python replay_equivalence.py            # run both legs, print the diff
  python replay_equivalence.py --leg a    # internal: leg A only
  python replay_equivalence.py --leg b    # internal: leg B only
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, shutil

REPO = "/root/nld/zombie-fix"
ENG = "/root/nld/e16-engine"
for p in (f"{REPO}/tools/pycompat", ENG, REPO, f"{REPO}/environments/nethack"):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED = 1
N_STEPS = 240
CHECKPOINT_AT = 120


# The engine takes RAW ASCII KEYCODES, not indices into an action table --
# verified empirically: after dismissing the intro, 'h'/'j'/'k'/'l'/'s' each
# move the hero and advance the clock, while small integers are mostly no-ops.
W_, S_, N_, E_, SEARCH = 104, 106, 107, 108, 115   # h j k l s
DISMISS = (32, 13, 27)                             # space, enter, esc


def action_list(n):
    """A fixed, seed-independent action list.

    Deliberately dull: a repeating walk-and-search cycle. It is not trying to
    play well, it is trying to burn game turns so monsters act and the core RNG
    advances. Length 15 keeps it from settling into a short loop against room
    geometry (walking into a wall still costs no time, but the search does).
    """
    cycle = [E_, E_, S_, SEARCH, W_, N_, SEARCH, E_, S_, S_,
             SEARCH, W_, W_, N_, SEARCH]
    return [cycle[i % len(cycle)] for i in range(n)]


def obs_digest(env_obs, raw):
    """Hash everything the engine reports for this step.

    glyphs/chars/colors are the map; blstats is the hero. Hashing them together
    means any divergence -- a monster one square off, one point of HP, a
    different level layout -- lands as a hash mismatch at the step it happens.
    """
    h = hashlib.sha256()
    for name in ("glyphs", "chars", "colors", "blstats"):
        arr = getattr(raw, name, None)
        if arr is not None:
            h.update(bytes(memoryview(arr).cast("B")))
    return h.hexdigest()[:16]


def make_env(_unused=None):
    from nethack_core.engine_env import EngineEnv
    return EngineEnv()


def raw_of(env):
    return getattr(env, "_engine", None) or getattr(env, "engine", None)


def leg_a(ckpt_dir, out_path):
    from nethack_harness.checkpoints import checkpoint_save
    env = make_env(None)
    env.seed(core=SEED, disp=SEED)
    env.reset(seeds=(SEED, SEED))
    raw = raw_of(env)
    # A fresh game is parked on the intro screen. Dismissing it is SETUP, not
    # part of the compared sequence: leg B restores a state that is already
    # past it, so these keys must not appear in the action list either leg
    # replays.
    for d in DISMISS:
        env.step(d)
    acts = action_list(N_STEPS)
    rows = []
    for i, a in enumerate(acts, start=1):
        env.step(a)
        rows.append(obs_digest(None, raw))
        if i == CHECKPOINT_AT:
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            # `assert_savepoint` refuses to checkpoint on a --More--, and step
            # 120 lands on one. Clear it with ESC and retry. These keys are
            # SETUP, not compared steps: leg B restores the cleared state, so
            # both legs resume from the same place with the same tail.
            from nethack_harness.checkpoints import CheckpointSavepointError
            for _ in range(8):
                try:
                    checkpoint_save(env, ckpt_dir, name="equivalence-probe",
                                    note=f"leg A step {i}")
                    break
                except CheckpointSavepointError:
                    env.step(27)          # ESC
                    rows[-1] = obs_digest(None, raw)
            else:
                raise SystemExit("could not reach a clean savepoint at "
                                 f"step {i} after 8 ESCs")
    json.dump({"rows": rows, "checkpoint_at": CHECKPOINT_AT,
               "n": N_STEPS, "seed": SEED}, open(out_path, "w"))
    print(f"[leg A] {len(rows)} steps, checkpoint at {CHECKPOINT_AT} -> {ckpt_dir}")


def leg_b(ckpt_dir, out_path):
    from nethack_harness.checkpoints import checkpoint_restore
    env = make_env(None)
    # A restore needs a live game to restore INTO; seed it the same way the
    # orchestrator's resume path does before handing over to the bundle.
    env.seed(core=SEED, disp=SEED)
    env.reset(seeds=(SEED, SEED))
    checkpoint_restore(ckpt_dir, env=env, audit=True)
    raw = raw_of(env)
    acts = action_list(N_STEPS)[CHECKPOINT_AT:]
    rows = []
    for a in acts:
        env.step(a)
        rows.append(obs_digest(None, raw))
    json.dump({"rows": rows, "checkpoint_at": CHECKPOINT_AT}, open(out_path, "w"))
    print(f"[leg B] restored and stepped {len(rows)} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=["a", "b"])
    ap.add_argument("--ckpt", default="/tmp/e16_equiv/ckpt")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.makedirs("/tmp/e16_equiv", exist_ok=True)

    if args.leg == "a":
        return leg_a(args.ckpt, args.out or "/tmp/e16_equiv/a.json")
    if args.leg == "b":
        return leg_b(args.ckpt, args.out or "/tmp/e16_equiv/b.json")

    # Driver: each leg in its own process, so leg B cannot inherit live state.
    py = sys.executable
    here = os.path.abspath(__file__)
    for leg in ("a", "b"):
        r = subprocess.run([py, here, "--leg", leg, "--ckpt", args.ckpt],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-3000:])
            raise SystemExit(f"leg {leg} failed rc={r.returncode}")

    A = json.load(open("/tmp/e16_equiv/a.json"))
    B = json.load(open("/tmp/e16_equiv/b.json"))
    tail = A["rows"][CHECKPOINT_AT:]
    b = B["rows"]
    n = min(len(tail), len(b))
    first = next((i for i in range(n) if tail[i] != b[i]), None)
    same = sum(1 for i in range(n) if tail[i] == b[i])

    print()
    print("=" * 62)
    print(f"compared {n} post-checkpoint steps (global steps "
          f"{CHECKPOINT_AT+1}..{CHECKPOINT_AT+n})")
    print(f"  identical : {same}/{n}")
    if first is None:
        print("  VERDICT   : restore reproduces the replay exactly.")
    else:
        print(f"  first diff: step {CHECKPOINT_AT+1+first} "
              f"(A={tail[first]}  B={b[first]})")
        print(f"  VERDICT   : diverges {first} step(s) after the checkpoint.")
    print("=" * 62)


if __name__ == "__main__":
    main()
