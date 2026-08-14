"""Empirically enumerate NetHack's blocking UI modes against the live engine.

For each interface we record what an agent can actually *see*:
  * `misc` = [in_yn_function, in_getlin, xwaitingforspace]  (win/rl/winrl.cc)
  * tty row 0 (the message/prompt line)
  * whether the game clock advances while the interface is open
  * which keys dismiss it

The point: `misc` only covers three of the modes. Everything else (menus in
selection mode, getpos pickers, extended-command entry, ...) has to be
recognised from the tty text or not at all.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/NetHack-engine")

import numpy as np  # noqa: E402

from nethack_core.engine_env import EngineEnv  # noqa: E402

PROBE_KEY = ord("s")  # search: always advances the clock, never blocked by a wall


def row(obs, i=0) -> str:
    return "".join(chr(int(c)) for c in obs.tty_chars[i]).rstrip()


def clock(obs) -> int:
    return int(obs.blstats[20])


def send(env, keys):
    obs = None
    for k in keys:
        obs, _, _ = env.step(ord(k) if isinstance(k, str) else k)
    return obs


CASES = [
    ("baseline (no UI open)", []),
    ("inventory menu  (PICK_NONE)", ["i"]),
    ("drop-type menu  (PICK_ANY)", ["D"]),
    ("pickup-type menu", ["m", ","]),
    ("getobj item prompt", ["r"]),
    ("getobj -> menu via ?", ["r", "?"]),
    ("getdir direction prompt", ["F"]),
    ("getpos: travel", ["_"]),
    ("getpos: farlook", [";"]),
    ("getpos: whatis /", ["/"]),
    ("getlin: engrave text", ["E", "-"]),
    ("getlin: annotate level", ["#", "a", "n", "n", "o", "t", "a", "t", "e", "\r"]),
    ("ext-cmd entry (#)", ["#"]),
    ("ext-cmd list (#?)", ["#", "?"]),
    ("yn confirm: #pray", ["#", "p", "r", "a", "y", "\r"]),
    ("yn confirm: quit", ["#", "q", "u", "i", "t", "\r"]),
    ("text window: attributes", [chr(24)]),          # ^X
    ("text window: discoveries", ["\\"]),
    ("text window: overview", [chr(15)]),            # ^O
    ("help menu (?)", ["?"]),
    ("options menu (O)", ["O"]),
    ("count prefix (n)", ["n", "1", "0"]),
    ("message recall (^P)", [chr(16)]),
    ("spell list (+)", ["+"]),
    ("#enhance", ["#", "e", "n", "h", "a", "n", "c", "e", "\r"]),
    ("#terrain", ["#", "t", "e", "r", "r", "a", "i", "n", "\r"]),
    ("look here (:)", [":"]),
]


def main():
    print(f"{'interface':32s} {'misc':10s} {'clock':>11s}  row0")
    print("-" * 110)
    rows = []
    for label, keys in CASES:
        env = EngineEnv()
        env.seed(0, 0)
        obs, _ = env.reset(character="val-hum-fem-neu")
        # clear the opening message so row 0 reflects the probe, not the intro
        obs, _, _ = env.step(ord("\r"))
        try:
            if keys:
                obs = send(env, keys)
            t0 = clock(obs)
            r0 = row(obs)
            misc = [int(x) for x in np.asarray(obs.misc).ravel()[:3]]
            after, _, _ = env.step(PROBE_KEY)  # does a movement key do anything?
            t1 = clock(after)
            frozen = "FROZEN" if t1 == t0 else f"{t0}->{t1}"
            # can esc get us out?
            esc, _, _ = env.step(27)
            esc2, _, _ = env.step(PROBE_KEY)
            escapes = clock(esc2) > clock(esc) or clock(esc) > t1
            print(f"{label:32s} {str(misc):10s} {frozen:>11s}  {r0[:60]}"
                  f"{'' if escapes else '   [ESC did NOT free it]'}")
            rows.append((label, misc, frozen, r0, escapes))
        except Exception as e:
            print(f"{label:32s} ERROR {e!r}")
        finally:
            env.close()

    blocked = [r for r in rows if r[2] == "FROZEN"]
    invisible = [r for r in blocked if not any(r[1])]
    print("\nblocking interfaces:", len(blocked), "of", len(rows))
    print("blocking but INVISIBLE in `misc` (tty text is the only signal):")
    for r in invisible:
        print(f"   {r[0]:32s} row0={r[3][:60]!r}")


if __name__ == "__main__":
    main()
