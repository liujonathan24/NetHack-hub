#!/usr/bin/env python3
"""Watch uncapped rollouts for runaways.

With `max_calls = 0` a rollout ends when the character dies, ascends, or NLE
truncates at 100k in-game steps. That is the point -- a rollout stopped at 200
calls reports a LOWER BOUND on depth, not a score -- but it removes the ceiling
that used to make cost predictable. Two things can go wrong and neither raises
an error:

  * runaway: an agent that never dies because it never risks anything (resting,
    re-reading the map, walking a corridor back and forth). Call count climbs,
    dungeon level does not.
  * stall: the rollout is wedged -- provider hang, daemon wedge -- and the turn
    file simply stops growing.

Emits one line per rollout that crosses a threshold. Read-only by default;
--kill terminates the offending rollout's process group.

    watch_runaway.py <run_dir> [--calls 600] [--stall-min 20] [--kill]
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, signal, sys, time

TURN_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.ndjson$")


def scan(run_dir: pathlib.Path, calls_max: int, stall_min: float):
    now = time.time()
    out = []
    for f in sorted(run_dir.glob("round*/corpus__prime_agent/turns/*.ndjson")):
        m = TURN_RE.match(f.name)
        if not m:
            continue
        seed, pid = int(m.group(1)), int(m.group(2))
        # A finished rollout has a trace row; only live ones matter here.
        tr = f.parent.parent / "traces.jsonl"
        done = set()
        if tr.exists() and tr.stat().st_size:
            for line in tr.read_text().splitlines():
                if not line.strip():
                    continue
                d = (json.loads(line).get("task") or {}).get("data") or {}
                if d.get("seed") is not None:
                    done.add(int(d["seed"]))
        if seed in done:
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        idle = (now - st.st_mtime) / 60.0
        n = dl = 0
        last = None
        with f.open() as fh:
            for line in fh:
                if line.strip():
                    n += 1
                    last = line
        if last:
            try:
                d = json.loads(last)
                dl = int(d.get("max_dlvl_reached") or d.get("dlvl") or 0)
            except Exception:
                pass
        alive = os.path.isdir(f"/proc/{pid}")
        why = []
        if n >= calls_max:
            why.append(f"{n} turns >= {calls_max}")
        if idle >= stall_min and alive:
            why.append(f"stalled {idle:.0f}m")
        if why:
            out.append({"seed": seed, "pid": pid, "turns": n, "dlvl": dl,
                        "idle_min": idle, "alive": alive, "why": "; ".join(why),
                        "round": f.parent.parent.parent.name})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--calls", type=int, default=600,
                    help="flag a rollout at this many turns (default 600; the "
                         "200-call era averaged 151 and the deepest game took 200)")
    ap.add_argument("--stall-min", type=float, default=20.0)
    ap.add_argument("--kill", action="store_true",
                    help="terminate flagged rollouts instead of only reporting")
    a = ap.parse_args()
    hits = scan(a.run_dir, a.calls, a.stall_min)
    for h in hits:
        print(f"[runaway] {h['round']} seed {h['seed']} pid {h['pid']}: "
              f"{h['turns']} turns, Dlvl {h['dlvl']}, idle {h['idle_min']:.0f}m "
              f"-- {h['why']}")
        if a.kill and h["alive"]:
            try:
                os.kill(h["pid"], signal.SIGTERM)
                print(f"[runaway]   SIGTERM -> {h['pid']}")
            except OSError as e:
                print(f"[runaway]   could not signal {h['pid']}: {e}")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
