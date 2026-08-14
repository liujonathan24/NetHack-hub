"""Turn decoded games into the exact series the artifact plots.

Series produced:
  games[]      per-game step curves: [[turn, balrog_max, balrog_min], ...]
  mean[]       cohort mean at turn t, carry-forward (a finished game holds its
               final score forever, so this is the mean of the cohort's
               best-so-far, not a survivors-only average that drifts upward)
  alive[]      how many games are still being played at turn t (context for the
               mean's right-hand tail)
  ascent{}     win-conditioned percentile fan over NORMALISED progress:
               x = % of that game's total turns, y = BALROG at that point
"""
from __future__ import annotations

import json
import pathlib
import statistics

OUT = pathlib.Path("/root/nld/plot_data.json")


def load(p):
    return json.loads(pathlib.Path(p).read_text())


def value_at(curve, t, idx):
    """Step-function lookup: last point at or before turn t."""
    v = 0.0
    for pt in curve:
        if pt[0] > t:
            break
        v = pt[idx]
    return v


def cohort_series(games, grid):
    mean_max, mean_min, alive = [], [], []
    for t in grid:
        vs_max = [value_at(g["curve"], t, 3) for g in games]
        vs_min = [value_at(g["curve"], t, 4) for g in games]
        mean_max.append(round(statistics.fmean(vs_max), 4))
        mean_min.append(round(statistics.fmean(vs_min), 4))
        alive.append(sum(1 for g in games if g["derived_turns"] >= t))
    return mean_max, mean_min, alive


def fan(games, pcts=(10, 25, 50, 75, 90)):
    """Percentiles over normalised game progress, for win-conditioned games."""
    xs = list(range(0, 101, 2))
    out = {"x": xs, "max": {p: [] for p in pcts}, "min": {p: [] for p in pcts}}
    for x in xs:
        vmax, vmin = [], []
        for g in games:
            total = g["derived_turns"] or 1
            t = total * x / 100.0
            vmax.append(value_at(g["curve"], t, 3))
            vmin.append(value_at(g["curve"], t, 4))
        vmax.sort()
        vmin.sort()
        for p in pcts:
            k = min(len(vmax) - 1, int(round((p / 100) * (len(vmax) - 1))))
            out["max"][p].append(round(vmax[k], 3))
            out["min"][p].append(round(vmin[k], 3))
    return out


def turn_fan(games, grid, pcts=(10, 25, 50, 75, 90)):
    """Percentiles across games on the ABSOLUTE turn axis (carry-forward).

    Same convention as the cohort mean: a finished game holds its final score,
    so the bands describe the cohort's best-so-far rather than a survivors-only
    view that would drift upward as games end.
    """
    out = {"max": {p: [] for p in pcts}, "min": {p: [] for p in pcts}, "alive": []}
    for t in grid:
        vmax = sorted(value_at(g["curve"], t, 3) for g in games)
        vmin = sorted(value_at(g["curve"], t, 4) for g in games)
        for p in pcts:
            k = min(len(vmax) - 1, int(round((p / 100) * (len(vmax) - 1))))
            out["max"][p].append(round(vmax[k], 3))
            out["min"][p].append(round(vmin[k], 3))
        out["alive"].append(sum(1 for g in games if g["derived_turns"] >= t))
    return out


def main():
    sample = load("/root/nld/games_sample100.json")
    try:
        asc = load("/root/nld/games_ascension.json")
    except FileNotFoundError:
        asc = []

    # log-spaced turn grid: games span 10^2..10^5 turns
    grid = sorted({int(round(10 ** (2 + i * (3.6 / 120)))) for i in range(121)} | {1, 10, 50})
    mean_max, mean_min, alive = cohort_series(sample, grid)

    data = {
        "n_games": len(sample),
        "n_ascension": len(asc),
        "grid": grid,
        "mean_max": mean_max,
        "mean_min": mean_min,
        "alive": alive,
        "games": [
            {
                "id": f"{g['player']}/{g['start']}",
                "role": g["role"],
                "turns": g["derived_turns"],
                "dl": g["derived_maxlvl"],
                "xp": g["derived_maxxp"],
                "bmax": g["balrog_max"],
                "bmin": g["balrog_min"],
                "asc": g["ascended"],
                "c": [[p[0], p[3], p[4]] for p in g["curve"]],
            }
            for g in sample
        ],
        "ascension": {
            "n": len(asc),
            "fan": fan(asc) if asc else None,
            "turn_fan": turn_fan(asc, grid) if asc else None,
            "games": [
                {
                    "id": f"{g['player']}/{g['start']}",
                    "role": g["role"],
                    "turns": g["derived_turns"],
                    "files": g["n_files"],
                    "c": [[p[0], p[3], p[4]] for p in g["curve"]],
                }
                for g in asc
            ],
        },
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
    print("sample games", len(sample), "ascension games", len(asc))
    print("mean_max final", mean_max[-1], "mean_min final", mean_min[-1])
    if asc:
        f = data["ascension"]["fan"]
        print("ascension median curve (max):",
              [f["max"][50][i] for i in range(0, 51, 10)])
        print("ascension median curve (min):",
              [f["min"][50][i] for i in range(0, 51, 10)])


if __name__ == "__main__":
    main()
