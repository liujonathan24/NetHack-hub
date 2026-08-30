#!/usr/bin/env python3
"""Golden-parity dump: replay one deterministic scripted action sequence and
record, at EVERY step, the full observation the agent would see.

Run once per engine build; the two dumps must be byte-identical.

    ENG=<engine root> python3 parity_dump.py <seed> <nsteps> <out.ndjson>

Recorded per step (byte-for-byte, via sha256 over the raw buffers plus the
values in the clear where they are small enough to eyeball):

    tty_chars, tty_colors, tty_cursor   the rendered terminal
    glyphs, chars, colors               the observation/glyph grid
    message                             the game message
    inv_strs, inv_letters, inv_glyphs   inventory
    blstats                             every status field, in the clear
    time                                the game clock, in the clear
    done, how_done                      game-over state

No LLM, no snapshots, no checkpoints -- normal play only, which is the thing
that has to be unchanged.
"""
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.environ["ENG"])

from nethack_core.engine_env import EngineEnv  # noqa: E402

SEED = int(sys.argv[1])
NSTEPS = int(sys.argv[2])
OUT = sys.argv[3]
CHARACTER = sys.argv[4] if len(sys.argv) > 4 else None

# A realistic key mix: 8-way movement, search, pickup, descend/ascend, look,
# inventory (opens a menu window), discoveries, ESC and CR (prompt handling).
KEYS = ([ord(c) for c in "hjklyubn"] * 3
        + [ord("s")] * 3
        + [ord(","), ord("."), ord(">"), ord("<"), ord(":"),
           ord("i"), ord("\\"), 27, 13])


def h(a):
    return hashlib.sha256(memoryview(a).tobytes()).hexdigest()[:32]


def main():
    env = EngineEnv()
    obs, _meta = env.reset(seeds=(SEED, SEED), character=CHARACTER)
    raw = env._engine
    rnd = random.Random(0xC0FFEE ^ SEED)

    with open(OUT, "w") as f:
        for i in range(NSTEPS):
            action = rnd.choice(KEYS)
            obs, done, _info = env.step(action)
            rec = {
                "i": i,
                "action": action,
                "tty_chars": h(obs.tty_chars),
                "tty_colors": h(obs.tty_colors),
                "tty_cursor": list(map(int, obs.tty_cursor)),
                "glyphs": h(obs.glyphs),
                "chars": h(obs.chars),
                "colors": h(obs.colors),
                "message": bytes(obs.message).split(b"\0")[0].decode(
                    "latin1"),
                "inv_strs": h(obs.inv_strs),
                "inv_letters": h(obs.inv_letters),
                "inv_glyphs": h(obs.inv_glyphs),
                "blstats": [int(x) for x in obs.blstats],
                "done": bool(done),
                "how_done": int(raw.how_done),
            }
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            if done:
                break
    print(f"seed={SEED} steps_written={i + 1} -> {OUT}")


if __name__ == "__main__":
    main()
