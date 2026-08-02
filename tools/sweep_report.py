#!/usr/bin/env python
"""One-screen status of every cell in a running sweep.

Reads the per-turn NDJSON directly rather than waiting for `traces.jsonl`, so it
works while rollouts are still in flight. Scores with the same
`balrog_both(max_dlvl, xp_level)` the aggregator uses, and reports BOTH the
published `max` and the `min` -- a rollout with max > 0 and min == 0 scored
entirely off experience level and never descended, which is the degenerate
pattern the min metric exists to expose.

    python -m tools.sweep_report [sweep_root]
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
import time

from nethack_harness.prompt.balrog import balrog_both

DEGENERATE_CALLS = 40  # matches the documented degeneracy rule

#: Cells with fewer than this many surviving rollouts are marked `*` and left out
#: of the pooled ALL row. Every published figure in this project carries a +/- SE,
#: and the measured variance floor between BYTE-IDENTICAL configs is ~2.2 points
#: -- so a mean over one or two rollouts is not a weak estimate, it is noise with
#: a number printed next to it. The row is still shown, because knowing a cell
#: was destroyed matters, but it must not silently move a pooled average.
MIN_USABLE_N = 3


def _mean_se(vals):
    """`(mean, standard error)`. SE is undefined for a single sample -- return
    None rather than 0.0, which would read as a precise measurement."""
    m = st.mean(vals)
    if len(vals) < 2:
        return m, None
    return m, st.stdev(vals) / (len(vals) ** 0.5)


def _fmt(mean, se, width=13, prec=2):
    body = f"{mean:.{prec}f}" + ("" if se is None else f"±{se:.{prec}f}")
    return f"{body:>{width}}"


def _rows(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a live rollout can leave a torn final line
    return out


def _seed(path):
    return os.path.basename(path).split("_")[0]


def rollout(path):
    rows = _rows(path)
    if not rows:
        return None
    status = [r.get("status") or {} for r in rows]
    dlvl = max((r.get("max_dlvl_reached") or r.get("dlvl") or 1) for r in rows)
    xl = max(s.get("experience_level", 1) or 1 for s in status)
    xp = max(s.get("experience_points", 0) or 0 for s in status)
    gt = max(s.get("time", 0) or 0 for s in status)
    hp = rows[-1].get("hp")
    mx, mn = balrog_both(dlvl, xl)
    calls = sum(len(r.get("tool_results") or r.get("tool_calls") or []) for r in rows)
    return dict(
        seed=_seed(path), turns=len(rows), dlvl=dlvl, xl=xl, xp=xp, gturn=gt,
        bmax=mx * 100, bmin=mn * 100, calls=calls or len(rows),
        hp=hp, dead=(hp is not None and hp <= 0),
        idle=time.time() - os.path.getmtime(path),
    )


def main(argv):
    root = argv[1] if len(argv) > 1 else "outputs/e2_encoding_sweep"
    cells = sorted(d for d in glob.glob(os.path.join(root, "*/")) if "__" in os.path.basename(d.rstrip("/")))
    print(f"{'cell':<38}{'n':>3}{'turns':>6}{'calls':>6}{'dlvl':>11}{'XL':>4}"
          f"{'XP':>5}{'gturn':>6}{'BAL_max':>13}{'BAL_min':>13}{'1ax':>4}{'dead':>5}{'idle_s':>7}")
    print("-" * 142)
    grand = []
    for d in cells:
        name = os.path.basename(d.rstrip("/"))
        rs = [r for r in (rollout(f) for f in sorted(glob.glob(d + "turns/*.ndjson"))) if r]
        if not rs:
            print(f"{name:<38}{'--- no turns yet ---':>40}")
            continue
        n = len(rs)
        f = lambda k: st.mean(r[k] for r in rs)
        # `max > 0 and min == 0` means ONE AXIS CONTRIBUTED NOTHING -- it does not
        # say which. At XL 1 the experience term is 0, so a rollout that descended
        # well on a level-1 character trips it as DEPTH-carried; the inverse case
        # (shallow death at high XL) is the xp-carried one the writeups warn about.
        # Reporting it as "xp-carried" regardless, as the aggregator currently
        # does, inverts the meaning for exactly the rollouts this sweep produces.
        oneaxis = sum(1 for r in rs if r["bmax"] > 0 and r["bmin"] == 0)
        dead = sum(1 for r in rs if r["dead"])
        mark = "*" if n < MIN_USABLE_N else " "
        print(f"{name+mark:<38}{n:>3}{f('turns'):>6.0f}{f('calls'):>6.0f}"
              f"{_fmt(*_mean_se([r['dlvl'] for r in rs]), width=11, prec=1)}"
              f"{f('xl'):>4.1f}{f('xp'):>5.0f}{f('gturn'):>6.0f}"
              f"{_fmt(*_mean_se([r['bmax'] for r in rs]))}"
              f"{_fmt(*_mean_se([r['bmin'] for r in rs]))}"
              f"{oneaxis:>4}{dead:>5}{max(r['idle'] for r in rs):>7.0f}")
        if n >= MIN_USABLE_N:
            grand += rs
    if grand:
        print("-" * 142)
        # Pools ROLLOUTS, not cell means, so cells with more survivors weigh more.
        # A sanity check, never a headline -- and cells below MIN_USABLE_N are
        # excluded so a 2-seed cell cannot drag it.
        print(f"{'ALL (usable cells)':<38}{len(grand):>3}"
              f"{st.mean(r['turns'] for r in grand):>6.0f}"
              f"{st.mean(r['calls'] for r in grand):>6.0f}"
              f"{_fmt(*_mean_se([r['dlvl'] for r in grand]), width=11, prec=1)}"
              f"{st.mean(r['xl'] for r in grand):>4.1f}"
              f"{st.mean(r['xp'] for r in grand):>5.0f}"
              f"{st.mean(r['gturn'] for r in grand):>6.0f}"
              f"{_fmt(*_mean_se([r['bmax'] for r in grand]))}"
              f"{_fmt(*_mean_se([r['bmin'] for r in grand]))}"
              f"{sum(1 for r in grand if r['bmax']>0 and r['bmin']==0):>4}"
              f"{sum(1 for r in grand if r['dead']):>5}")
        print(f"\n* cell has n < {MIN_USABLE_N} and is EXCLUDED from the pooled row "
              "-- its mean is noise, not a weak estimate")
        deg = [r for r in grand if r["calls"] < DEGENERATE_CALLS]
        if deg:
            print(f"\nunder the {DEGENERATE_CALLS}-call degeneracy floor: {len(deg)} rollout(s) "
                  "(expected early in a run; only meaningful once a rollout has stopped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
