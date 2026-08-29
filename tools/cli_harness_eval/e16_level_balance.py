#!/usr/bin/env python3
"""Is the hero UNDER-LEVELLED or UNDER-DEPTHED for its own score?

    python3 e16_level_balance.py --state DLVL XL
    python3 e16_level_balance.py <RUN_DIR> [<RUN_DIR> ...]

WHY THIS EXISTS. BALROG is reported as a pair. `balrog_both(dlvl, xl)` returns
(max, min): the max is set by whichever of depth / experience is FURTHER ALONG,
the min by whichever is FURTHER BEHIND. Every arm this program has run scores
on depth and is pinned on experience -- treesmoke8 finished at max 30.88 and
min 3.69, because it reached Dlvl 15 at XL 6.

So "under-levelled" has an exact, deterministic meaning here, and it needs no
reference to anybody's play: you are under-levelled when your XL component is
the smaller of the two, and the gap max - min is precisely what it costs you.

WHY NOT A "MEDIAN WINNER PATH". There are no winning games on disk to take a
median of, and our own rollouts are the wrong reference -- at Dlvl 15 their
median XL is 4.5, which is not a target, it is the symptom. Anchoring on the
metric itself avoids inventing a standard from runs that all failed the same
way. (It also agrees with ordinary NetHack practice, where XL roughly tracking
Dlvl is the survivable ratio in the early-mid game.)

The advice this emits is deliberately about SURVIVABILITY, not about points.
A directive that told a player "experience is worth score" would be false under
the max metric -- at Dlvl 7, XL 1 and XL 6 both score 4.85 -- and players who
are told a false thing stop trusting the true things. What is true is that
levelled heroes reach deeper: across 48 baseline rollouts r(XL, Dlvl) = 0.59,
XL 1 runs cap at Dlvl 7, XL 5 runs average Dlvl 10.5 and reach 15.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "environments", "nethack"))
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402

MAX_XL = 30
MAX_DLVL = 53


def pair(dlvl: int, xl: int) -> tuple:
    a, b = balrog_both(int(dlvl), int(xl))
    return 100.0 * a, 100.0 * b


def depth_component(dlvl: int) -> float:
    """f(Dlvl) -- the depth half of the pair, read at XL 1 where the XL half is 0."""
    return pair(dlvl, 1)[0]


def xl_component(xl: int) -> float:
    """g(XL) -- the experience half, read at Dlvl 1 where the depth half is 0."""
    return pair(1, xl)[0]


def xl_needed(dlvl: int) -> int | None:
    """Lowest XL whose component reaches the depth component at this Dlvl.

    BALROG is (max, min) over the two components, so the binding one is simply
    the smaller. Comparing the returned max and min for equality -- which an
    earlier version of this did -- only ever fires on an exact tie and reported
    every state as unfixable.
    """
    want = depth_component(dlvl)
    for xl in range(1, MAX_XL + 1):
        if xl_component(xl) >= want - 1e-9:
            return xl
    return None


def dlvl_supported(xl: int) -> int:
    """Deepest Dlvl whose depth component this XL still covers."""
    have = xl_component(xl)
    best = 1
    for d in range(1, MAX_DLVL + 1):
        if depth_component(d) <= have + 1e-9:
            best = d
    return best


def assess(dlvl: int, xl: int) -> dict:
    mx, mn = pair(dlvl, xl)
    need = xl_needed(dlvl)
    supported = dlvl_supported(xl)
    fd, gx = depth_component(dlvl), xl_component(xl)
    binding = ("experience" if gx < fd - 1e-9
               else ("depth" if fd < gx - 1e-9 else "balanced"))
    return {
        "dlvl": int(dlvl), "xl": int(xl),
        "balrog_max": round(mx, 2), "balrog_min": round(mn, 2),
        "gap": round(mx - mn, 2),
        "binding_constraint": binding,
        "xl_needed_for_this_depth": need,
        "xl_short_by": (max(0, need - int(xl)) if need else None),
        "dlvl_supported_by_this_xl": supported,
        "advice": _advice(int(dlvl), int(xl), need, supported, mx, mn),
    }


def _advice(dlvl, xl, need, supported, mx, mn) -> str:
    fd, gx = depth_component(dlvl), xl_component(xl)
    if gx > fd + 1e-9:
        return (f"UNDER-DEPTHED. At XL {xl} the hero could support about Dlvl "
                f"{supported} and is only on Dlvl {dlvl}. Descending is what "
                f"raises the score now; more experience adds nothing until the "
                f"depth catches up.")
    if abs(gx - fd) <= 1e-9:
        return (f"Balanced at Dlvl {dlvl} / XL {xl}. Either axis raises the "
                f"score from here.")
    short = (need - xl) if need else None
    return (
        f"UNDER-LEVELLED. At Dlvl {dlvl} the hero is XL {xl}; XL {need} is where "
        f"experience stops being the limiting factor"
        + (f" (short by {short})" if short else "")
        + f". This costs {round(mx - mn, 2)} points of the reported pair "
          f"({round(mx,2)} vs {round(mn,2)}). XL {xl} comfortably supports about "
          f"Dlvl {supported}, so the hero is roughly {max(0, dlvl - supported)} "
          f"floor(s) deeper than its strength. Diving further will not raise the "
          f"lower number at all, and an under-levelled hero dies to ordinary "
          f"monsters at these depths."
    )


def run_state(run_dir: str) -> dict | None:
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
                    help="print the balance point for each Dlvl")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.table:
        print("Dlvl  depth comp  XL needed  XL comp   pair at that XL")
        for d in range(1, 21):
            n = xl_needed(d)
            mx, mn = pair(d, n) if n else (0.0, 0.0)
            print(f"{d:>4}  {depth_component(d):>9.2f}  {str(n):>9}  "
                  f"{(xl_component(n) if n else 0):>7.2f}  {mx:>6.2f} / {mn:<6.2f}")
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
            print(f"{r['run']:<12} D{r['dlvl']} XL{r['xl']} -> "
                  f"{r['balrog_max']} / {r['balrog_min']} min  "
                  f"(gap {r['gap']}, binding: {r['binding_constraint']})")
            print(f"{'':<12} {r['advice']}")
    if a.json:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
