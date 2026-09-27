#!/usr/bin/env python3
"""Progress curve for an E16 run: cumulative LM turns vs BEST-SO-FAR BALROG.

    python3 e16_progress_curve.py <RUN_DIR> [<RUN_DIR> ...] [-o out.json] [--csv]

THE QUESTION THIS ANSWERS. Per-attempt tables say how each life did. They do
not say whether the RUN is getting anywhere. An arm whose first attempt reaches
Dlvl 9 and whose next seven reach Dlvl 1 has a great best-attempt number and has
learned nothing. Plotting the running best against cumulative LM turns puts
every arm -- method, null, rollback -- on one axis where the shape of the curve,
not its endpoint, is the result:

  * a step early then flat  -> the archive is re-treading ground it already has
  * steady staircase        -> attempts are compounding
  * flat at zero            -> nothing is working

X IS LM TURNS, NOT ATTEMPTS, and that matters for fairness. A resumed attempt
can cost 2 calls and a cold restart 489; counting attempts would let an arm look
efficient purely by dying fast. Turns are what is actually spent.

Y IS BALROG, BOTH HALVES. `balrog_both(max_dlvl, max_xl)` returns (max, min):
the max is set by whichever of depth/XL is further along, the min by whichever
is further behind. For every agent this program has run, the max is set by depth
and the min by XL -- at XL 1 the min is 0.00 at ANY depth, and at XL 5 it is
capped at 2.91 however deep the hero goes. So the two curves answer different
questions and both are plotted: `best_max` is "how deep did it get", `best_min`
is "did it ever become strong enough for that depth to count".

Everything is read from the per-turn NDJSON, which the harness writes from the
engine's own blstats -- no model text contributes, and the null arm works
identically even though it keeps no persistent archive.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "environments", "nethack"))
try:
    from nethack_harness.prompt.balrog import balrog_both
except Exception:  # pragma: no cover - import path varies by checkout
    balrog_both = None


def _attempt_dirs(run_dir: str) -> list:
    """Attempt directories in attempt order."""
    out = []
    for d in sorted(glob.glob(os.path.join(run_dir, "attempts", "a*"))):
        base = os.path.basename(d)
        if base[1:].isdigit():
            out.append((int(base[1:]), d))
    return [d for _, d in sorted(out)]


def _turn_file(attempt_dir: str):
    fs = sorted(glob.glob(os.path.join(attempt_dir, "turns", "*.ndjson")))
    return fs[0] if fs else None


def curve(run_dir: str, every: int = 1) -> dict:
    """Cumulative-turn curve of best-so-far BALROG for one run."""
    if balrog_both is None:
        raise RuntimeError("balrog_both is not importable from this checkout")

    run_dir = os.path.abspath(run_dir)
    pts, marks = [], []
    t = 0
    best_max = best_min = 0.0
    hi_d = hi_x = 0

    for adir in _attempt_dirs(run_dir):
        n = int(os.path.basename(adir)[1:])
        f = _turn_file(adir)
        if not f:
            continue
        start_t = t
        # Per-ATTEMPT high-water marks reset; the RUN's best-so-far never does.
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t += 1
            st = rec.get("status") or {}
            d = rec.get("max_dlvl_reached") or rec.get("dlvl") or 0
            x = st.get("experience_level") or 0
            if not (d and x):
                continue
            hi_d, hi_x = max(hi_d, int(d)), max(hi_x, int(x))
            mx, mn = balrog_both(hi_d, hi_x)
            mx, mn = 100.0 * mx, 100.0 * mn
            improved = (mx > best_max + 1e-9) or (mn > best_min + 1e-9)
            best_max, best_min = max(best_max, mx), max(best_min, mn)
            # Keep every improvement, plus a sample every `every` turns, so the
            # staircase keeps its corners without carrying one point per turn.
            if improved or t % every == 0:
                pts.append({"t": t, "attempt": n,
                            "best_max": round(best_max, 2),
                            "best_min": round(best_min, 2),
                            "dlvl": hi_d, "xl": hi_x})
        marks.append({"attempt": n, "t_start": start_t + 1, "t_end": t,
                      "turns": t - start_t,
                      "best_max_at_end": round(best_max, 2),
                      "best_min_at_end": round(best_min, 2)})

    if pts and pts[-1]["t"] != t:
        pts.append({"t": t, "attempt": marks[-1]["attempt"] if marks else None,
                    "best_max": round(best_max, 2), "best_min": round(best_min, 2),
                    "dlvl": hi_d, "xl": hi_x})

    arm = tier = None
    try:
        prov = json.load(open(os.path.join(run_dir, "provenance.json")))
        arm, tier = prov.get("experiment_arm"), prov.get("tier")
    except Exception:
        pass

    # How much of the final best was already reached by the END of attempt 1 --
    # the "did anything after the first life help?" number.
    first = marks[0] if marks else None
    return {
        "run": os.path.basename(run_dir), "run_dir": run_dir,
        "arm": arm, "tier": tier,
        "total_turns": t, "attempts": len(marks),
        "final_best_max": round(best_max, 2), "final_best_min": round(best_min, 2),
        "after_attempt_1_max": (first or {}).get("best_max_at_end"),
        "after_attempt_1_min": (first or {}).get("best_min_at_end"),
        "improvement_after_attempt_1_max": (
            round(best_max - first["best_max_at_end"], 2) if first else None),
        "improvement_after_attempt_1_min": (
            round(best_min - first["best_min_at_end"], 2) if first else None),
        "points": pts, "attempt_marks": marks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("-o", "--out", default=None, help="write JSON here")
    ap.add_argument("--every", type=int, default=10,
                    help="sample every N turns in addition to every improvement")
    ap.add_argument("--csv", action="store_true", help="print CSV instead of a summary")
    a = ap.parse_args()

    out = {}
    for d in a.run_dir:
        try:
            out[os.path.basename(os.path.abspath(d))] = curve(d, a.every)
        except Exception as e:
            print(f"{d}: {e}", file=sys.stderr)

    if a.out:
        json.dump(out, open(a.out, "w"), separators=(",", ":"))
        print(f"{a.out}: {len(out)} run(s)")
    if a.csv:
        print("run,t,attempt,best_max,best_min,dlvl,xl")
        for r, c in out.items():
            for p in c["points"]:
                print(f"{r},{p['t']},{p['attempt']},{p['best_max']},"
                      f"{p['best_min']},{p['dlvl']},{p['xl']}")
        return
    if not a.out:
        for r, c in out.items():
            print(f"{r:<12} arm={str(c['arm']):<16} {c['attempts']} attempts, "
                  f"{c['total_turns']} turns")
            print(f"{'':<12} BALROG max {c['after_attempt_1_max']} -> "
                  f"{c['final_best_max']}  (+{c['improvement_after_attempt_1_max']} "
                  f"after attempt 1)")
            print(f"{'':<12} BALROG min {c['after_attempt_1_min']} -> "
                  f"{c['final_best_min']}  (+{c['improvement_after_attempt_1_min']} "
                  f"after attempt 1)")


if __name__ == "__main__":
    main()
