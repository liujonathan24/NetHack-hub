#!/usr/bin/env python
"""What a sweep will cost, projected from a previous sweep's own traces.

The exp 2 projection was $190 against a $1,230.50 bill -- 6.6x low -- and every
part of that error is now fixed except the one this tool exists for: a rollout's
cost is NOT linear in its call budget, so you cannot price a 400-call rollout by
doubling a 200-call one. 98% of spend is re-sent input, context grows with the
conversation, and the marginal cost of call N therefore climbs until compaction
binds and it plateaus.

So the unit here is the MARGINAL COST OF A 50-CALL BUCKET, measured per arm from
a real sweep, and a rollout of N calls is the integral of that curve to N.

    python tools/cost_projection.py --from outputs/e2_encoding_sweep \
        --horizons 200,400,600,800 --seeds 5 \
        --cells claude_code=2,prime_agent=3

Extrapolation is stated, never silent. Every horizon past the measured range is
marked, and the assumption used to get there is printed with the table. The two
arms need different treatment and that is a finding, not an inconvenience:

  * claude_code's curve PLATEAUS. Measured out to 400 calls, it flattens at
    ~$19-21 per 100 calls from about call 150 -- compaction binding.
  * prime_agent's is still CLIMBING where the data stops (~$31/100 at call 150,
    ~$32 at 200, and only 38 calls of evidence in that bucket). It also costs
    ~1.5x claude_code at the same call index, because its IPython kernel
    transcript rides in the context alongside the game.

A prime_agent projection past ~200 calls is therefore an extrapolation with no
measurement under it, and this tool prints a RANGE for it rather than a number.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.cli_harness_eval.aggregate import _call_cost, is_error_trace  # noqa: E402
from tools.eval_metrics import model_for_cell, price_table_for, refresh_price_tables  # noqa: E402

BUCKET = 50
#: A bucket with fewer than this many observed calls is not a measurement.
MIN_CALLS_PER_BUCKET = 15


def marginal_curve(run_root, arm_prefix, cell_filter=""):
    """`{bucket_start: ($ per call, n observed)}` for one arm.

    Error-terminated rollouts are excluded -- a 402 stub contributes a handful
    of zero-cost calls at index 0 and would drag the first bucket down.
    """
    tot = collections.defaultdict(float)
    n = collections.Counter()
    for path in glob.glob(os.path.join(run_root, f"{arm_prefix}*", "traces.jsonl")):
        cell_dir = os.path.dirname(path)
        cell = os.path.basename(cell_dir)
        if cell_filter and cell_filter not in cell:
            continue
        price = price_table_for(model_for_cell(cell_dir))
        if price is None:
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)
            if is_error_trace(trace):
                continue
            for i, call in enumerate(trace.get("calls") or []):
                c = _call_cost(call, price)
                if c is None:
                    continue
                k = (i // BUCKET) * BUCKET
                tot[k] += c
                n[k] += 1
    return {
        k: (tot[k] / n[k], n[k])
        for k in sorted(tot)
        if n[k] >= MIN_CALLS_PER_BUCKET
    }


def integrate(curve, horizon, tail_per_call):
    """Cost of one rollout of `horizon` calls.

    Buckets past the measured range are charged at `tail_per_call`, and the
    number of calls priced that way is returned so the caller can say how much
    of the figure is measurement and how much is assumption.
    """
    total, extrapolated = 0.0, 0
    for start in range(0, horizon, BUCKET):
        take = min(BUCKET, horizon - start)
        if start in curve:
            total += curve[start][0] * take
        else:
            total += tail_per_call * take
            extrapolated += take
    return total, extrapolated


def tail_rates(curve):
    """`(flat, climbing)` per-call tail assumptions for one arm.

    `flat` holds the last measured bucket -- right for a curve that has already
    plateaued, optimistic for one that has not. `climbing` continues the last
    observed bucket-to-bucket increase for two more buckets and then holds,
    which is what compaction eventually does to any of these curves.
    """
    if not curve:
        return 0.0, 0.0
    keys = sorted(curve)
    flat = curve[keys[-1]][0]
    if len(keys) < 2:
        return flat, flat
    step = curve[keys[-1]][0] - curve[keys[-2]][0]
    return flat, max(flat, flat + step * 1.5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default="outputs/e2_encoding_sweep")
    ap.add_argument("--filter", default="", help="substring a cell name must contain "
                                                 "(e.g. BBOX__vison)")
    ap.add_argument("--horizons", default="200,400,600,800")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cells", default="claude_code=2,prime_agent=3",
                    help="how many CELLS each arm contributes, e.g. "
                         "claude_code=2,prime_agent=3")
    args = ap.parse_args(argv)

    refresh_price_tables()
    horizons = [int(h) for h in args.horizons.split(",")]
    plan = {}
    for part in args.cells.split(","):
        arm, _, k = part.partition("=")
        plan[arm.strip()] = int(k)

    curves, tails = {}, {}
    print(f"marginal cost per 100 calls, measured from {args.src}"
          + (f" (cells matching {args.filter!r})" if args.filter else ""))
    print()
    for arm in plan:
        curves[arm] = marginal_curve(args.src, arm, args.filter)
        tails[arm] = tail_rates(curves[arm])
        row = "  ".join(
            f"{k}-{k+BUCKET-1}: ${v*100:5.2f} (n={c})"
            for k, (v, c) in sorted(curves[arm].items())
        )
        print(f"  {arm:<12} {row}")
        measured_to = (max(curves[arm]) + BUCKET) if curves[arm] else 0
        print(f"  {'':<12} measured to call {measured_to}; "
              f"beyond it: flat ${tails[arm][0]*100:.2f} .. climbing "
              f"${tails[arm][1]*100:.2f} per 100")
    print()

    print(f"projected cost, {args.seeds} seeds per cell, cells: "
          + ", ".join(f"{a}x{k}" for a, k in plan.items()))
    print()
    hdr = f"{'horizon':>9}"
    for arm in plan:
        hdr += f"{arm + ' /rollout':>26}"
    hdr += f"{'SWEEP TOTAL':>26}"
    print(hdr)
    print("-" * len(hdr))
    for h in horizons:
        line = f"{h:>9}"
        lo_tot = hi_tot = 0.0
        any_extrap = False
        for arm, k in plan.items():
            lo, ex = integrate(curves[arm], h, tails[arm][0])
            hi, _ = integrate(curves[arm], h, tails[arm][1])
            lo_tot += lo * k * args.seeds
            hi_tot += hi * k * args.seeds
            mark = "*" if ex else " "
            cell = (f"${lo:,.0f}{mark}" if abs(hi - lo) < 0.5
                    else f"${lo:,.0f}-${hi:,.0f}{mark}")
            line += f"{cell:>26}"
            any_extrap = any_extrap or bool(ex)
        tot = (f"${lo_tot:,.0f}" if abs(hi_tot - lo_tot) < 1
               else f"${lo_tot:,.0f}-${hi_tot:,.0f}")
        line += f"{tot + ('*' if any_extrap else ''):>26}"
        print(line)
    print("\n* includes calls beyond the measured range; the range spans the "
          "flat and climbing tail assumptions printed above.")
    print("  Wall clock is roughly ONE rollout, not N: seeds within a cell and "
          "cells within a batch run concurrently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
