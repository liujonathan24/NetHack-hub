#!/usr/bin/env python
"""BALROG progression *over time*, per strategy, against a human reference.

The cross-cell table answers "how deep did it get". This answers "how FAST",
which is the question that survives a truncated budget: every rollout in a
sweep is censored by a call cap, a death or a wallet, so a terminal depth
compares agents partly on how long they were allowed to run. A rate does not.

Three curves per strategy, all as a function of IN-GAME TURN (`status.time`)
and, separately, of LLM CALL INDEX:

    BALROG max   the published metric: max over the (Dlvl, Xp) achievement axes
    BALROG min   the same table's min -- a rollout with max > 0 and min == 0
                 scored entirely off ONE axis and is not progressing the way the
                 headline number suggests
    dlvl         raw depth, for reading the other two

Averaging convention: CARRY-FORWARD. A rollout that died at turn 900 holds its
final score for the rest of the x-axis instead of dropping out of the mean.
BALROG progression is monotone non-decreasing and death ends accumulation, so
carrying forward is what the metric means; dropping finished rollouts instead
would make every curve bend upward purely as its weakest members exit
(survivorship). `n_alive` is reported at every grid point so a reader can see
how much of a late curve is real play and how much is held constant.

    python tools/progress_rate.py <run_root> [<run_root> ...] \
        [--human-progress PCT --human-turns N] [--json OUT.json]

Human reference
---------------
BALROG's NetHack progression table is ITSELF human data -- each achievement's
value is the empirical probability that a human who reached it went on to
ascend (Paglieri et al., ICLR 2025, built from NAO server games). So "the human
score" is not a separate measurement to look up; a human who ascends ends at
100% by construction. What is NOT in the table is the DENOMINATOR: how many
in-game turns a human spends getting there. That is the number this tool needs
supplied, and it is the only thing standing between these curves and a
same-axes comparison:

    human rate (%/turn) = --human-progress / --human-turns

Both default to None and the human line is omitted rather than guessed. Pass
the pair you trust and the provenance is recorded in the JSON output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "environments", "nethack")
)

from nethack_harness.prompt.balrog import balrog_both  # noqa: E402


def read_rollout(path):
    """`[(game_turn, call_index, dlvl, xl), ...]`, monotone in call index.

    Turn files are written one record per executed skill, so the record index
    IS the call index. `status.time` is the in-game clock and can stand still
    across several calls (a refused call, a menu keystroke), which is exactly
    why both denominators are carried.
    """
    out = []
    max_dlvl, max_xl = 1, 1
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                break  # a SIGKILLed writer leaves a torn final line
            if not isinstance(rec, dict):
                continue
            status = rec.get("status") or {}
            d = rec.get("max_dlvl_reached") or rec.get("dlvl") or 1
            max_dlvl = max(max_dlvl, int(d))
            max_xl = max(max_xl, int(status.get("experience_level") or 1))
            out.append((int(status.get("time") or 0), len(out), max_dlvl, max_xl))
    return out


def curve(rollout, xs, x_index):
    """Carry-forward BALROG (max, min) and dlvl for one rollout on grid `xs`.

    `x_index` is 0 for the in-game clock and 1 for the call index. Before a
    rollout's first sample the value is 0/0/1 (it had not started); after its
    last it holds that last value.
    """
    pts, out, j = rollout, [], 0
    last = (0.0, 0.0, 1)
    for x in xs:
        while j < len(pts) and pts[j][x_index] <= x:
            mx, mn = balrog_both(pts[j][2], pts[j][3])
            last = (mx * 100, mn * 100, pts[j][2])
            j += 1
        out.append(last)
    return out


def alive_at(rollout, xs, x_index):
    """How many grid points this rollout was still producing samples for."""
    end = rollout[-1][x_index] if rollout else 0
    return [x <= end for x in xs]


def strategy_series(rollouts, xs, x_index):
    """Mean carry-forward curves across `rollouts`, plus `n_alive` per point."""
    curves = [curve(r, xs, x_index) for r in rollouts]
    alive = [alive_at(r, xs, x_index) for r in rollouts]
    series = {"x": xs, "balrog_max": [], "balrog_min": [], "dlvl": [],
              "balrog_max_se": [], "n": len(rollouts), "n_alive": []}
    for k in range(len(xs)):
        mx = [c[k][0] for c in curves]
        mn = [c[k][1] for c in curves]
        dl = [c[k][2] for c in curves]
        series["balrog_max"].append(st.mean(mx))
        series["balrog_min"].append(st.mean(mn))
        series["dlvl"].append(st.mean(dl))
        series["balrog_max_se"].append(
            st.stdev(mx) / len(mx) ** 0.5 if len(mx) > 1 else None
        )
        series["n_alive"].append(sum(1 for a in alive if a[k]))
    return series


def rollout_rate(rollout):
    """`(%/game-turn, %/LLM-call, final %, game turns, calls)` for one rollout.

    Progress is measured from the rollout's OWN start (BALROG at turn 1), not
    from zero: every character begins at Dlvl 1 / Xp 1, which the table already
    values above zero, and crediting an agent for the starting position would
    reward a rollout that did nothing.
    """
    if len(rollout) < 2:
        return None
    t0, c0, d0, x0 = rollout[0]
    t1, c1, d1, x1 = rollout[-1]
    start = balrog_both(d0, x0)[0] * 100
    end = balrog_both(d1, x1)[0] * 100
    dt, dc = t1 - t0, c1 - c0
    return {
        "per_game_turn": (end - start) / dt if dt > 0 else None,
        "per_llm_call": (end - start) / dc if dc > 0 else None,
        "final_pct": end,
        "gained_pct": end - start,
        "game_turns": dt,
        "calls": dc,
    }


#: BALROG % checkpoints for the turns-to-milestone table.
MILESTONES = (1.0, 2.0, 5.0, 10.0, 20.0)


def turns_to_milestone(rollout, pct, metric="max"):
    """`(game turns, LLM calls)` to first reach `pct` BALROG, or `None`.

    `metric="min"` measures the same milestone on BALROG's MIN over the two
    achievement axes -- it rises only when depth AND experience have both
    advanced, so it is immune to the single-axis carry the max tolerates and
    is the natural scoreboard for balance-seeking guidance (GUIDE_LAG).

    This is the comparison that survives BALROG's concavity, and the reason the
    headline %/turn rate must not be read across very different horizons. The
    table is a probability of eventual ascension conditioned on a milestone, so
    its first few points are worth a lot and its last ones very little: a
    Valkyrie who walks down four staircases has already banked a few percent,
    while the remaining 90-odd take tens of thousands of turns of endgame. An
    agent measured over its first 600 turns therefore posts a %/turn rate
    several times a human ascender's WITHOUT being faster at anything -- it is
    only ever standing on the steep part of the curve. Comparing turns to the
    SAME milestone removes that entirely.
    """
    idx = 0 if metric == "max" else 1
    for t, c, d, x in rollout:
        if balrog_both(d, x)[idx] * 100 >= pct:
            return t, c
    return None


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.stdev(xs) / len(xs) ** 0.5


def discover(run_root):
    """`{strategy: [rollout, ...]}`.

    Quarantined attempts are included: a rollout the watchdog killed at turn
    400 still measured 400 turns of progress RATE, which is the quantity here,
    even though its terminal depth is censored. It is the terminal-depth table
    that must exclude them, not this one.
    """
    cells = {}
    pattern = os.path.join(run_root, "*", "turns", "*.ndjson")
    stalled = os.path.join(run_root, "*", "turns.stalled", "*", "*.ndjson")
    for path in sorted(glob.glob(pattern)) + sorted(glob.glob(stalled)):
        cell = os.path.relpath(path, run_root).split(os.sep)[0]
        rows = read_rollout(path)
        if len(rows) < 2:
            continue
        cells.setdefault(cell, []).append(rows)
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_roots", nargs="+")
    ap.add_argument("--grid-turns", type=int, default=40,
                    help="grid points on each axis (default 40)")
    ap.add_argument("--human-progress", type=float, default=None,
                    help="BALROG %% a human reference run reaches (e.g. 100 for "
                         "an ascension). Omitted => no human line.")
    ap.add_argument("--human-turns", type=float, default=None,
                    help="in-game turns that reference run takes")
    ap.add_argument("--human-label", default="human reference")
    ap.add_argument("--json", default=None, help="write the full series here")
    args = ap.parse_args(argv)

    cells = {}
    for root in args.run_roots:
        for cell, rollouts in discover(root).items():
            key = cell if len(args.run_roots) == 1 else f"{os.path.basename(root.rstrip('/'))}/{cell}"
            cells.setdefault(key, []).extend(rollouts)
    if not cells:
        print("progress_rate: no rollouts found", file=sys.stderr)
        return 2

    max_turn = max(r[-1][0] for rs in cells.values() for r in rs)
    max_call = max(r[-1][1] for rs in cells.values() for r in rs)
    turn_grid = [round(max_turn * i / args.grid_turns) for i in range(args.grid_turns + 1)]
    call_grid = [round(max_call * i / args.grid_turns) for i in range(args.grid_turns + 1)]

    out = {"cells": {}, "human": None,
           "grid": {"game_turn": turn_grid, "llm_call": call_grid}}
    if args.human_progress is not None and args.human_turns:
        out["human"] = {
            "label": args.human_label,
            "progress_pct": args.human_progress,
            "game_turns": args.human_turns,
            "pct_per_game_turn": args.human_progress / args.human_turns,
            "note": "supplied by the operator; BALROG's own table is human-derived "
                    "(P(ascend | milestone)), so the value is 100 for an ascension "
                    "and the denominator is the only free parameter",
        }

    print(f"{'strategy':<34}{'n':>3}{'BALROG%/1k turns':>19}{'BALROG%/100 calls':>19}"
          f"{'final %':>10}{'turns':>8}{'calls':>7}")
    print("-" * 100)
    ranked = []
    for cell, rollouts in sorted(cells.items()):
        rates = [r for r in (rollout_rate(x) for x in rollouts) if r]
        if not rates:
            continue
        pt, pt_se = mean_se([r["per_game_turn"] for r in rates])
        pc, pc_se = mean_se([r["per_llm_call"] for r in rates])
        fin, _ = mean_se([r["final_pct"] for r in rates])
        gt, _ = mean_se([r["game_turns"] for r in rates])
        ca, _ = mean_se([r["calls"] for r in rates])
        out["cells"][cell] = {
            "n": len(rates),
            "rate_pct_per_game_turn": pt, "rate_pct_per_game_turn_se": pt_se,
            "rate_pct_per_llm_call": pc, "rate_pct_per_llm_call_se": pc_se,
            "final_pct_mean": fin, "game_turns_mean": gt, "calls_mean": ca,
            "by_game_turn": strategy_series(rollouts, turn_grid, 0),
            "by_llm_call": strategy_series(rollouts, call_grid, 1),
        }
        ranked.append((pt or 0, cell))
        print(f"{cell:<34}{len(rates):>3}"
              f"{(pt or 0)*1000:>12.2f}±{(pt_se or 0)*1000:<6.2f}"
              f"{(pc or 0)*100:>12.2f}±{(pc_se or 0)*100:<6.2f}"
              f"{fin:>10.2f}{gt:>8.0f}{ca:>7.0f}")

    # -- turns to milestone: the concavity-proof comparison --------------------
    print()
    header = "".join(f"{'BAL ' + str(m) + '%':>15}" for m in MILESTONES)
    print(f"{'mean in-game turns to reach':<34}{header}")
    print("-" * (34 + 15 * len(MILESTONES)))
    for cell, rollouts in sorted(cells.items()):
        if cell not in out["cells"]:
            continue
        row, cells_out = "", out["cells"][cell].setdefault("milestones", {})
        for m in MILESTONES:
            hits = [turns_to_milestone(r, m) for r in rollouts]
            hits = [h for h in hits if h]
            if not hits:
                cells_out[str(m)] = None
                row += f"{'--':>15}"
                continue
            mu, _ = mean_se([h[0] for h in hits])
            mc, _ = mean_se([h[1] for h in hits])
            cells_out[str(m)] = {"game_turns": mu, "calls": mc,
                                 "n_reached": len(hits), "n": len(rollouts)}
            row += f"{mu:>10.0f} ({len(hits)}){'':>0}"
        print(f"{cell:<34}{row}")
    print("\n(`--` = no rollout in the cell ever reached that milestone; "
          "the count in parentheses is how many of the cell's rollouts did)")

    if out["human"]:
        h = out["human"]["pct_per_game_turn"]
        print("-" * 100)
        print(f"{args.human_label:<34}{'--':>3}{h*1000:>12.2f}{'':>7}{'':>19}"
              f"{args.human_progress:>10.2f}{args.human_turns:>8.0f}{'--':>7}")
        best = max(ranked)[0] if ranked else 0
        if h:
            print(f"\nfastest agent strategy is {best/h:.3f}x the {args.human_label} rate "
                  f"({max(ranked)[1] if ranked else 'n/a'})")
    else:
        print("\n(no human line: pass --human-progress and --human-turns)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
