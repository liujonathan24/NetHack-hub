"""Pull the E15 intervention arms onto the (Dlvl, XL) plane.

One record per rollout: the running-max staircase through the plane, the end
point, and the arm's two BALROG gradings. The correction arms are the only
series in the paper that had no plane data -- fig4 used endpoint means from the
table and nothing drew their paths -- so this is what the trajectory version of
that figure needs.

ARM -> CELL MAPPING is not one-to-one and cannot be guessed from the directory
names, so it is written out below and then VERIFIED: the script recomputes each
arm's (Dlvl, XL, BALROG-max, BALROG-min) means and fails loudly if they do not
reproduce the numbers in the paper's intervention table. Two mappings are not
obvious:

  * `gate` and `revive` use the FIXED reruns (`p1_gate_fix1`, `p3_rollback_fix2`)
    and not the r1 cells. P1 r1 is a placebo -- the advisory fired after the
    descent had already executed -- and the p3 rollback death-window did not
    operate until fix2. See /root/nld/e15-data/README.md.
  * `json` is not three clean reps. Rep 1's seeds 1 and 3 were re-run in
    `v2_json_s13retry`, and it is the retries that the table counts; taking
    rep 1 whole gives Dlvl 6.80 / XL 2.20 against the table's 6.4 / 2.13.

Writes figs/e15_probes_plane.json.
"""
import glob
import json
import os
import statistics as st

from common import balrog_table, _pct

B = "/root/nld/zombie-fix/outputs/e15_probes"
OUT = "/root/overleaf/figs/e15_probes_plane.json"

#: arm key -> (label, [(cell, [seeds] or None for all)])
ARMS = {
    "gate":   ("Advisory descent gate", [
        ("p1_gate_fix1__prime_agent", None),
        ("p1_gate_fix1_r2__prime_agent", None),
        ("p1_gate_fix1_r3__prime_agent", None)]),
    "revive": ("Forced revive on death", [
        ("p3_rollback_fix2__prime_agent", None),
        ("p3_rollback_fix2_r2__prime_agent", None),
        ("p3_rollback_fix2_r3__prime_agent", None)]),
    "doors":  ("Locked doors removed", [
        ("c1_doors_r1__prime_agent", None),
        ("c1_doors_r2__prime_agent", None),
        ("c1_doors_r3__prime_agent", None)]),
    "ascii":  ("ASCII map every turn", [
        ("v1_b1_r1__prime_agent", None),
        ("v1_b1_r2__prime_agent", None),
        ("v1_b1_r3__prime_agent", None)]),
    "json":   ("JSON map every turn", [
        ("v2_json_r1__prime_agent", ["0", "2", "4"]),
        ("v2_json_s13retry__prime_agent", ["1", "3"]),
        ("v2_json_r2__prime_agent", None),
        ("v2_json_r3__prime_agent", None)]),
    "crisis": ("Crisis directive", [
        ("p2_crisis_fix1__prime_agent", None),
        ("p2_crisis_fix1_r2__prime_agent", None),
        ("p2_crisis_fix1_r3__prime_agent", None)]),
    "fog":    ("Fog of war restored", [
        ("v0_fog_r1__prime_agent", None),
        ("v0_fog_r2__prime_agent", None),
        ("v0_fog_r3__prime_agent", None)]),
}

#: The paper's intervention table, as (Dlvl, XL, BALROG-max, BALROG-min). The
#: extraction is checked against it rather than trusted.
EXPECT = {
    "gate":   (6.5, 2.93, 5.73, 1.79),
    "revive": (10.3, 4.20, 15.48, 2.64),
    "doors":  (7.3, 2.73, 7.49, 1.76),
    "ascii":  (7.7, 2.93, 8.50, 1.88),
    "json":   (6.4, 2.13, 5.70, 1.43),
    "crisis": (6.1, 3.20, 4.93, 1.88),
}


def _stream(path):
    """[(lm_turn, dlvl, xl)] for one rollout's turn file."""
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        s = r.get("status") or {}
        d = r.get("dlvl") or s.get("depth")
        x = s.get("experience_level")
        t = r.get("lm_turn")
        if isinstance(d, int) and isinstance(x, int) and isinstance(t, int):
            out.append((t, d, x))
    return out


def _staircase(stream):
    """Running (max Dlvl, max XL) at each LM turn, deduplicated.

    The raw stream carries the CURRENT depth, which falls whenever the hero
    climbs back up; the plane is a reach plot, so both axes are cumulative
    maxima. Only the turns that move the pair are kept.
    """
    pts, cd, cx = [], 0, 0
    for _, d, x in stream:
        cd, cx = max(cd, d), max(cx, x)
        if not pts or pts[-1] != [cd, cx]:
            pts.append([cd, cx])
    return pts


def build():
    dl, xp = balrog_table()
    arms = {}
    for key, (label, cells) in ARMS.items():
        rolls = []
        for cell, seeds in cells:
            for f in sorted(glob.glob(f"{B}/{cell}/turns/*.ndjson")):
                seed = os.path.basename(f).split("_")[0]
                if seeds is not None and seed not in seeds:
                    continue
                pts = _staircase(_stream(f))
                if not pts:
                    print(f"  EMPTY {f}")
                    continue
                d, x = pts[-1]
                a, b = _pct(dl, d), _pct(xp, x)
                rolls.append({"cell": cell, "seed": seed, "pts": pts,
                              "end": [d, x], "bal": max(a, b),
                              "bal_min": min(a, b)})
        arms[key] = {"label": label, "rollouts": rolls}
    return arms


def check(arms):
    """Reproduce the intervention table, or say exactly where it diverges."""
    ok = True
    for key, exp in EXPECT.items():
        rs = arms[key]["rollouts"]
        got = (st.mean([r["end"][0] for r in rs]),
               st.mean([r["end"][1] for r in rs]),
               st.mean([r["bal"] for r in rs]),
               st.mean([r["bal_min"] for r in rs]))
        bad = [i for i in range(4) if abs(got[i] - exp[i]) > 0.06]
        ok &= not bad and len(rs) == 15
        flag = "OK " if not bad and len(rs) == 15 else "BAD"
        print(f"  {flag} {key:7s} n={len(rs):2d}  "
              f"Dlvl {got[0]:5.2f} (want {exp[0]:5.2f})  "
              f"XL {got[1]:4.2f} (want {exp[1]:4.2f})  "
              f"BAL-max {got[2]:5.2f} (want {exp[2]:5.2f})  "
              f"BAL-min {got[3]:4.2f} (want {exp[3]:4.2f})")
    return ok


if __name__ == "__main__":
    arms = build()
    good = check(arms)
    json.dump({"arms": arms}, open(OUT, "w"), separators=(",", ":"))
    print(f"wrote {OUT}  ({sum(len(a['rollouts']) for a in arms.values())} rollouts)")
    if not good:
        raise SystemExit("extraction does not reproduce the intervention table")
