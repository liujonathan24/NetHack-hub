"""Which BALROG variant better models 'how close am I to finishing?'

Two tests, both on the top-10 corpus (it contains enough games that actually
reach the end to make 'finishing' measurable):

A. LINEARITY   For games that reached the end, is the metric linear in the
   fraction of the game elapsed? A good progress model rises steadily toward
   the finish rather than sitting flat and then jumping.

B. PREDICTIVENESS  Taken at a fixed turn, how well does the metric separate
   games that eventually reach the end from games that do not (AUC)? A progress
   model that says "you are 40% done" should mean you are far more likely to
   finish than one that says 10%.
"""
from __future__ import annotations

import json
import pathlib
import statistics

FINISH_DLVL = 50  # "reached the end" == got to the bottom of the dungeon


def value_at(curve, t, idx):
    v = 0.0
    for pt in curve:
        if pt[0] > t:
            break
        v = pt[idx]
    return v


def linfit(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return a, b, r2


def auc(pos, neg):
    """Probability a random finisher scores above a random non-finisher."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2  # average rank for ties
        for k in range(i, j):
            ranks[k] = r
        i = j
    rsum = sum(ranks[k] for k, (_v, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    games = json.loads(pathlib.Path("/root/nld/top10_curves.json").read_text())
    games = [g for g in games if g["complete"] and g["turns"] >= 100]
    fin = [g for g in games if g["max_dlvl"] >= FINISH_DLVL]
    los = [g for g in games if g["max_dlvl"] < FINISH_DLVL]
    print(f"{len(games)} complete games: {len(fin)} reached Dlvl {FINISH_DLVL}+, {len(los)} did not\n")

    # ---- A. linearity in completion fraction, over finishers ----
    print("A. LINEARITY over games that reached the end")
    print("   (metric sampled at each 5% of the game's elapsed turns)")
    for name, idx in (("max", 1), ("min", 2)):
        xs, ys, per_game_r2 = [], [], []
        for g in fin:
            gx, gy = [], []
            for p in range(0, 101, 5):
                t = g["turns"] * p / 100
                gx.append(p)
                gy.append(value_at(g["c"], t, idx))
            xs += gx
            ys += gy
            per_game_r2.append(linfit(gx, gy)[2])
        a, b, r2 = linfit(xs, ys)
        print(f"   {name}: pooled R2 {r2:.3f}   slope {b:.3f} %/1% elapsed   "
              f"median per-game R2 {statistics.median(per_game_r2):.3f}")

    # ---- B. predictiveness at a fixed turn ----
    print("\nB. PREDICTIVENESS -- AUC separating eventual finishers, metric read at turn T")
    print(f"   {'turn':>7}  {'n alive':>8}  {'AUC max':>8}  {'AUC min':>8}  "
          f"{'median max fin/not':>22}  {'median min fin/not':>22}")
    for T in (500, 1000, 2000, 5000, 10000, 20000):
        pos_max = [value_at(g["c"], T, 1) for g in fin if g["turns"] >= T]
        neg_max = [value_at(g["c"], T, 1) for g in los if g["turns"] >= T]
        pos_min = [value_at(g["c"], T, 2) for g in fin if g["turns"] >= T]
        neg_min = [value_at(g["c"], T, 2) for g in los if g["turns"] >= T]
        if len(pos_max) < 5 or len(neg_max) < 5:
            continue
        print(f"   {T:7d}  {len(pos_max)+len(neg_max):8d}  {auc(pos_max, neg_max):8.3f}  "
              f"{auc(pos_min, neg_min):8.3f}  "
              f"{statistics.median(pos_max):9.2f} /{statistics.median(neg_max):9.2f}   "
              f"{statistics.median(pos_min):9.2f} /{statistics.median(neg_min):9.2f}")

    # ---- C. the unbalanced-run pathology the two metrics disagree about ----
    print("\nC. Where the two disagree: runs carried by one axis only")
    T = 2000
    alive = [g for g in games if g["turns"] >= T]
    carried = [g for g in alive if value_at(g["c"], T, 1) > 0 and value_at(g["c"], T, 2) == 0]
    print(f"   at turn {T}: {len(carried)}/{len(alive)} games score >0 on max but 0 on min")
    if carried:
        rate_c = sum(1 for g in carried if g["max_dlvl"] >= FINISH_DLVL) / len(carried)
        rest = [g for g in alive if g not in carried]
        rate_r = sum(1 for g in rest if g["max_dlvl"] >= FINISH_DLVL) / max(1, len(rest))
        print(f"   of those, {rate_c:.1%} go on to reach the end, vs {rate_r:.1%} of the rest")


if __name__ == "__main__":
    main()
