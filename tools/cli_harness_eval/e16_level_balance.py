#!/usr/bin/env python3
"""How far below the WINNERS' experience norm is the hero for its depth?

    python3 e16_level_balance.py --state DLVL XL
    python3 e16_level_balance.py <RUN_DIR> [<RUN_DIR> ...]
    python3 e16_level_balance.py --table

CORRECTED 2026-08-29. The first version of this scored a state by the MIN-MAX
GAP (BALROG max minus min) and told the orchestrator the gap was "what it costs
you". That is wrong, and the statistical appendix "Lopsidedness, Formally"
(23,860 human games / 433 ascensions; v2: 44,008 / 824) refutes it on two
counts:

  * The gap D = M - m "grows mechanically with score, so it is never compared
    across score levels without conditioning." A deeper run has a bigger gap
    BECAUSE it went deeper. This script reported treesmoke8's 27.19-point gap
    as a deficiency; it was mostly a restatement that the run reached Dlvl 15.

  * The gap does not survive as a predictor. Under a linear-M specification it
    looks like beta ~ -0.12; swap linear M for M-quintile dummies and it
    collapses to +0.03, centred on zero at both horizons. Quoting the appendix:
    "The negative point estimate was residual score confounding absorbed by D,
    not a lopsidedness signal."

WHAT SURVIVES is the deficit against the winners' trajectory:

    delta = normXL(depth) - XL

where normXL(d) is the MEDIAN XL AT FIRST REACHING DEPTH d AMONG THE WINNERS.
Its hazard coefficient is beta = +0.047 [+0.038, +0.054] pooled, rising to
+0.077 [+0.052, +0.099] at depth >= 11: being under the winners' pace predicts
dying, and predicts it harder the deeper you are. The norm is stable -- the
appendix reports it "identical at every depth 1-20" whether computed from 433
or 824 winners -- so it is not an artifact of one decode.

The advice is a SURVIVAL statement, never a scoring one. Under BALROG-max,
experience below XL 7 is worth nothing at all (at Dlvl 7, XL 1 and XL 6 both
score 4.85), so telling a player "levelling scores points" would be false.
What is true is that characters at the winners' pace live long enough to
descend further.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "environments", "nethack"))
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402

#: Depth range the winners' norm is published for. Past it we return None
#: rather than extrapolating a standard the source does not state.
NORM_MAX_DLVL = 20


def pair(dlvl: int, xl: int) -> tuple:
    """(BALROG max, BALROG min) as percentages."""
    a, b = balrog_both(int(dlvl), int(xl))
    return 100.0 * a, 100.0 * b


def norm_xl(dlvl: int):
    """normXL(d): median XL at first reaching depth d among the winners.

    The appendix gives it as a step function -- "XL 1 through depth 3, then
    roughly parity (norm ~ d) to depth 9, flattening to 12 by depth 15" -- and
    that is exactly what is encoded, including the flat 12 out to the published
    limit of depth 20.
    """
    d = int(dlvl)
    if d < 1:
        return None
    if d <= 3:
        return 1
    if d <= 9:
        return d
    if d <= 15:
        return min(12, 9 + round((d - 9) * 3 / 6))
    if d <= NORM_MAX_DLVL:
        return 12
    return None


def deficit(dlvl: int, xl: int):
    """delta = normXL(depth) - XL. Positive means under the winners' pace."""
    n = norm_xl(dlvl)
    return None if n is None else n - int(xl)


def hazard_beta(dlvl: int):
    """The published hazard coefficient per level of deficit, by depth band."""
    d = int(dlvl)
    if d >= 11:
        return 0.077
    if d >= 6:
        return 0.022
    return None


def _advice(dlvl, xl, norm, dd, beta) -> str:
    if norm is None:
        return (f"Dlvl {dlvl} is past the depth range the winners' norm covers "
                f"(1-{NORM_MAX_DLVL}); no experience target is published here.")
    if dd is None or dd <= 0:
        return (f"At Dlvl {dlvl} the hero is XL {xl}; winners first reaching this "
                f"depth are typically XL {norm}. It is at or above the winning "
                f"pace, so depth is the axis with room in it.")
    band = (f" At this depth each level of deficit carries a hazard coefficient "
            f"of {beta:+.3f}." if beta else "")
    return (
        f"UNDER-LEVELLED BY {dd} against the winners' pace. At Dlvl {dlvl} the "
        f"hero is XL {xl}; the median winner first reaching Dlvl {dlvl} is "
        f"XL {norm}.{band} This is about survival, not score: gaining experience "
        f"does not by itself raise the headline number, but characters at the "
        f"winners' pace live long enough to go deeper."
    )


def assess(dlvl: int, xl: int) -> dict:
    mx, mn = pair(dlvl, xl)
    n = norm_xl(dlvl)
    dd = deficit(dlvl, xl)
    beta = hazard_beta(dlvl)
    return {
        "dlvl": int(dlvl), "xl": int(xl),
        "balrog_max": round(mx, 2), "balrog_min": round(mn, 2),
        # Reported for context ONLY. Not a deficiency measure: it grows
        # mechanically with score and does not survive conditioning on it.
        "min_max_gap_not_a_metric": round(mx - mn, 2),
        "winners_norm_xl": n,
        "deficit_vs_winners": dd,
        "hazard_beta_per_level_of_deficit": beta,
        "advice": _advice(int(dlvl), int(xl), n, dd, beta),
    }


def run_state(run_dir: str):
    """The high-water (Dlvl, XL) an existing run actually reached."""
    hi_d = hi_x = 0
    for f in glob.glob(os.path.join(run_dir, "attempts", "a*", "turns", "*.ndjson")):
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = rec.get("status") or {}
            hi_d = max(hi_d, int(rec.get("max_dlvl_reached") or rec.get("dlvl") or 0))
            hi_x = max(hi_x, int(st.get("experience_level") or 0))
    if not (hi_d and hi_x):
        return None
    out = assess(hi_d, hi_x)
    out["run"] = os.path.basename(os.path.abspath(run_dir))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="*")
    ap.add_argument("--state", nargs=2, type=int, metavar=("DLVL", "XL"))
    ap.add_argument("--table", action="store_true",
                    help="print the winners' XL norm by depth")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.table:
        print("Dlvl  winners' XL norm   hazard beta / level of deficit")
        for d in range(1, NORM_MAX_DLVL + 1):
            b = hazard_beta(d)
            print(f"{d:>4}  {str(norm_xl(d)):>16}   {('%+.3f' % b) if b else '—':>8}")
        return

    if a.state:
        r = assess(*a.state)
        print(json.dumps(r, indent=1) if a.json else r["advice"])
        return

    out = {}
    for d in a.run_dir:
        r = run_state(d)
        if not r:
            print(f"{d}: no turn data", file=sys.stderr)
            continue
        out[r["run"]] = r
        if not a.json:
            print(f"{r['run']:<12} D{r['dlvl']} XL{r['xl']}  "
                  f"norm XL{r['winners_norm_xl']}  deficit {r['deficit_vs_winners']:+d}")
            print(f"{'':<12} {r['advice']}")
    if a.json:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
