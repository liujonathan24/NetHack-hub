#!/usr/bin/env python3
"""Deterministic, no-LLM reproduction of Bug B.

    panic("double buffering unexpected")   save.c:854
    -> really_done(PANICKED) -> nh_terminate -> nle_yield -> jump_fcontext
    -> SIGSEGV (the crash signature in treesmoke3 attempts a001/a002/a003)

MECHANISM
    The E16 cadence auto-checkpoint is EngineEnv.checkpoint(), i.e.
    save_level() + save_player() + save_levelfiles().  Both NLE-added savers

        nle.c:1250   nle_save_level():  bufon(fd); savelev; bflush; bufoff; nhclose(fd)
        save.c:447   nle_save_player(): bufon(fd); savegamestate; bflush; bufoff; nhclose(fd)

    call bufoff(), never bclose().  ONLY def_bclose() resets bw_fd to -1, so
    every checkpoint leaves `bw_fd` holding an fd number that has just been
    closed, and `bw_FILE` a leaked FILE* wrapping it.  It "works" only for as
    long as the next open(2) happens to hand back the SAME number.  The moment
    anything else in the process takes that slot -- a restore's level-file
    installs, a trace file, an LLM HTTP keep-alive socket -- the next bufon()
    sees `bw_fd >= 0 && bw_fd != fd` and panics.

    bw_fd also lives in `nle_save_state`, which is arena-allocated, so
    nle_fr_restore rewinds it to a snapshot-era fd number as well.

REPRODUCTION (mirrors a restored E16 attempt)
    play -> checkpoint -> resume -> cadence checkpoint #1 -> [the attempt
    process acquires one more long-lived fd, exactly as the player's HTTP
    keep-alive socket does] -> cadence checkpoint #2  ==> panic.

    That is why all three treesmoke3 attempts died at the SECOND cadence
    auto-checkpoint of a restored attempt.

Exit 1 == bug reproduced (or the process dies of SIGSEGV, 139).
Exit 0 == fixed.

Run:  ENG=/root/nld/e16-engine python3 repro_bugB.py
"""
import os
import socket
import sys
import tempfile
import traceback

sys.path.insert(0, os.environ.get("ENG", "/root/nld/e16-engine"))

from nethack_core.engine_env import EngineEnv  # noqa: E402

SEED = 1
N_CADENCE = 4


def fds():
    return sorted(int(x) for x in os.listdir("/proc/self/fd"))


def main():
    tmp = tempfile.mkdtemp(prefix="reproB-")
    env = EngineEnv()
    env.reset(seeds=(SEED, SEED))
    for _ in range(12):
        env.step(1)

    ck0 = os.path.join(tmp, "ck0.json")
    print(f"[pre ] fds={fds()}  baseline checkpoint", flush=True)
    env.checkpoint(ck0)

    print(f"[rest] fds={fds()}  resume from that checkpoint", flush=True)
    env.resume(ck0)

    keepalive = []
    for i in range(1, N_CADENCE + 1):
        if i == 2:
            # The one long-lived fd a restored attempt acquires between its
            # first and second cadence checkpoint: the player's HTTP
            # keep-alive socket to the inference provider.
            s = socket.socket()
            keepalive.append(s)
            print(f"       (attempt process opened a long-lived fd "
                  f"{s.fileno()} -- HTTP keep-alive)", flush=True)
        print(f"[cad{i}] fds={fds()}  cadence auto-checkpoint {i} ...",
              flush=True)
        try:
            env.checkpoint(os.path.join(tmp, f"ck{i}.json"))
        except Exception:
            traceback.print_exc()
            print(f"\nREPRO: cadence auto-checkpoint {i} after restore FAILED")
            return 1
        for _ in range(3):
            env.step(1)

    print(f"\nOK: {N_CADENCE} post-restore cadence checkpoints, no panic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
