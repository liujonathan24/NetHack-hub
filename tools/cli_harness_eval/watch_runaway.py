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

# Names that mean the rollout is still talking to the real tool server.
ENGINE_CALL = re.compile(r"^(np_[a-z_]+|request_map|search|reveal|rollback)$")
DEATH_TEXT = re.compile(r"You die|DYWYPI|possessions identified|Do you want your", re.I)


def _turns(f: pathlib.Path) -> list[dict]:
    out = []
    with f.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def zombie_checks(rows: list[dict], window: int = 12) -> list[str]:
    """The failure this exists for: the game ended or broke, and the agent kept
    going anyway.

    Documented precedent on this project: a rollout whose MCP server died had
    the agent rebuild a counterfeit NetHack backend inside its own IPython
    kernel and carry on "playing" it. Nothing errored. Under the old 200-call
    cap that wasted a cell; uncapped, it can burn the full 7200s rollout
    timeout producing fiction.

    Every check needs the rollout to have CONTINUED past the bad state -- a
    rollout that dies on its last turn is just a rollout that ended.
    """
    bad = []
    if len(rows) < 3:
        return bad

    # 1. dead, and still taking turns.
    for i, r in enumerate(rows[:-1]):
        hp = r.get("hp")
        if hp is not None and hp <= 0 and len(rows) - i > 2:
            bad.append(f"died at turn {r.get('turn')} (HP {hp}) but {len(rows)-i-1} "
                       f"more turns were written")
            break
    else:
        for i, r in enumerate(rows[:-2]):
            if DEATH_TEXT.search(r.get("rendered_user_message") or ""):
                bad.append(f"death screen at turn {r.get('turn')} but "
                           f"{len(rows)-i-1} more turns followed")
                break

    tail = rows[-window:]

    # 2. the in-game clock stopped. A counterfeit game cannot advance it.
    clocks = [(t.get("status") or {}).get("time") for t in tail]
    clocks = [c for c in clocks if c is not None]
    if len(clocks) >= window and len(set(clocks)) == 1:
        bad.append(f"in-game clock frozen at {clocks[0]} across {len(clocks)} turns")

    # 3. no contact with the tool server at all.
    named = [c.get("name") for t in tail for c in (t.get("tool_calls") or [])]
    if len(tail) >= window and not any(ENGINE_CALL.match(n or "") for n in named):
        bad.append(f"no engine call in the last {len(tail)} turns "
                   f"(saw: {sorted(set(n for n in named if n))[:4] or 'nothing'})")

    # 4. everything is failing and it has not stopped.
    res = [r.get("status") for t in tail for r in (t.get("tool_results") or [])]
    if len(res) >= 6 and all(x == "failed" for x in res):
        bad.append(f"every one of the last {len(res)} tool results failed")
    return bad


def scan(run_dir: pathlib.Path, calls_max: int, stall_min: float):
    now = time.time()
    out = []
    # Layout-agnostic. This used to hardcode E13's `round*/corpus__prime_agent/`
    # shape, so pointing it at a plain cell tree (the base/human arms, which are
    # `<tier>_r<n>__prime_agent/`) matched ZERO files and reported a clean bill
    # of health for rollouts it had never looked at. A detector that finds
    # nothing must say so, not imply everything is fine -- see the empty-scan
    # warning in main().
    for f in sorted(run_dir.glob("**/turns/*.ndjson")):
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
        rows = _turns(f)
        n = len(rows)
        dl = int((rows[-1].get("max_dlvl_reached") or rows[-1].get("dlvl") or 0)) if rows else 0
        alive = os.path.isdir(f"/proc/{pid}")
        why = []
        # Zombie checks first: they are the reason this tool exists, and they
        # matter even on a rollout that is otherwise short and quiet.
        if alive:
            why += zombie_checks(rows)
        if n >= calls_max:
            why.append(f"{n} turns >= {calls_max}")
        if idle >= stall_min and alive:
            why.append(f"stalled {idle:.0f}m")
        if why:
            out.append({"seed": seed, "pid": pid, "turns": n, "dlvl": dl,
                        "zombie": any("clock frozen" in w or "no engine call" in w
                                      or "died at turn" in w or "death screen" in w
                                      or "tool results failed" in w for w in why),
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
                    help="terminate ZOMBIE rollouts (dead/frozen/counterfeit/erroring "
                         "but still running). Never kills on turn count alone: under "
                         "an uncapped budget a long game is the point, whereas a "
                         "zombie is producing fiction and cannot recover.")
    ap.add_argument("--kill-runaway", action="store_true",
                    help="also terminate rollouts flagged only for length or stall")
    a = ap.parse_args()
    # An empty scan is ambiguous -- no rollouts, or a path that matches nothing --
    # and the two look identical from the exit code. Say which.
    n_turns = len(list(a.run_dir.glob("**/turns/*.ndjson")))
    if n_turns == 0:
        print(f"[runaway] WARNING: no turn files under {a.run_dir} -- nothing was "
              f"checked. This is not an all-clear.")
        return 0
    hits = scan(a.run_dir, a.calls, a.stall_min)
    for h in hits:
        tag = "ZOMBIE" if h.get("zombie") else "runaway"
        print(f"[{tag}] {h['round']} seed {h['seed']} pid {h['pid']}: "
              f"{h['turns']} turns, Dlvl {h['dlvl']}, idle {h['idle_min']:.0f}m "
              f"-- {h['why']}")
        killable = (a.kill and h.get("zombie")) or (a.kill_runaway and not h.get("zombie"))
        if killable and h["alive"]:
            # IDENTITY CHECK before signalling. The pid comes from the turn
            # FILENAME, written at rollout start; once that process dies the
            # kernel recycles the pid, and a stale-but-flagged file then points
            # at an unrelated process. Cost of skipping this: 2026-08-26
            # 03:10, both in-flight reflection orchestrators (fresh processes
            # wearing recycled rollout pids) were killed at rc=137 and the run
            # froze prematurely. A pid is killable only while its environ still
            # names THIS run's directory.
            # The run-dir basename IS the run id (outputs/e13/<run-id>), and
            # every process of the run -- host-side eval children and sandboxed
            # players alike -- carries it in INSTALL_DIR=/tmp/vf-prime-agent-
            # <run-id>. (The worktree path would NOT work: bwrap unsets
            # PYTHONPATH, so sandboxed players never mention it.)
            try:
                with open(f"/proc/{h['pid']}/environ", "rb") as fh:
                    env = fh.read().decode("utf-8", "replace")
            except OSError:
                env = ""
            wanted = a.run_dir.resolve().name
            if wanted not in env:
                print(f"[runaway]   pid {h['pid']} no longer belongs to this "
                      f"run (recycled?) -- NOT killing")
                continue
            try:
                os.kill(h["pid"], signal.SIGTERM)
                print(f"[runaway]   SIGTERM -> {h['pid']}")
                for _ in range(20):
                    time.sleep(0.1)
                    if not os.path.isdir(f"/proc/{h['pid']}"):
                        break
                if os.path.isdir(f"/proc/{h['pid']}"):
                    os.kill(h["pid"], signal.SIGKILL)
                    print(f"[runaway]   SIGKILL -> {h['pid']} (TERM ignored)")
            except OSError as e:
                print(f"[runaway]   could not signal {h['pid']}: {e}")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
