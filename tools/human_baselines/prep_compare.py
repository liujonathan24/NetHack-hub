"""Merge the three populations onto one turn axis for the artifact.

  nao      NLD-NAO general population (sampled human games)
  top10    DeepMind nao_top10 -- the ten strongest NAO players by Z-score
  agent    our Prime Agent rollouts (local traces)

Every population is summarised the same way: median and p25/p75 across games at
each turn on a shared grid, carry-forward (a finished game holds its final
score), plus the count still playing so the tails can be read honestly.
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics

OUT = pathlib.Path("/root/nld/compare_data.json")


def value_at(curve, t, idx):
    v = 0.0
    for pt in curve:
        if pt[0] > t:
            break
        v = pt[idx]
    return v


def summarise(games, grid, imax=1, imin=2, pcts=(25, 50, 75)):
    res = {"max": {p: [] for p in pcts}, "min": {p: [] for p in pcts},
           "mean_max": [], "mean_min": [], "alive": []}
    for t in grid:
        vmax = sorted(value_at(g["c"], t, imax) for g in games)
        vmin = sorted(value_at(g["c"], t, imin) for g in games)
        for p in pcts:
            k = min(len(vmax) - 1, int(round((p / 100) * (len(vmax) - 1))))
            res["max"][p].append(round(vmax[k], 3))
            res["min"][p].append(round(vmin[k], 3))
        res["mean_max"].append(round(statistics.fmean(vmax), 3))
        res["mean_min"].append(round(statistics.fmean(vmin), 3))
        res["alive"].append(sum(1 for g in games if g["turns"] >= t))
    return res


def win_block(games, grid):
    """Aggregates over the games that actually ascended, if there are any."""
    w = [g for g in games if g.get("won")]
    if len(w) < 3:
        return None
    return {"n": len(w), "series": summarise(w, grid), "stats": endpoint_stats(w),
            "games": [{"c": thin(g["c"], 60), "turns": g["turns"]} for g in draw_set(w, 60)]}


def endpoint_stats(games):
    bm = sorted(g["balrog_max"] for g in games)
    bn = sorted(g["balrog_min"] for g in games)
    tn = sorted(g["turns"] for g in games)
    q = lambda a, p: a[min(len(a) - 1, int(round(p * (len(a) - 1))))]  # noqa: E731
    return {
        "n": len(games),
        "bmax_median": round(statistics.median(bm), 2),
        "bmax_mean": round(statistics.fmean(bm), 2),
        "bmax_p90": round(q(bm, .9), 2),
        "bmax_best": round(bm[-1], 2),
        "bmin_median": round(statistics.median(bn), 2),
        "bmin_mean": round(statistics.fmean(bn), 2),
        "turns_median": int(statistics.median(tn)),
        "turns_max": int(tn[-1]),
        "dlvl_median": int(statistics.median([g["max_dlvl"] for g in games])),
        "dlvl_best": int(max(g["max_dlvl"] for g in games)),
    }


def draw_set(games, cap=120, seed=11):
    """Subsample the lines we DRAW; aggregates are always over every game."""
    if len(games) <= cap:
        return games
    rng = random.Random(seed)
    return rng.sample(games, cap)


def thin(curve, keep=90):
    if len(curve) <= keep:
        return curve
    step = len(curve) / keep
    out = [curve[int(i * step)] for i in range(keep)]
    out.append(curve[-1])
    return out


def main():
    # Prefer the whole-shard decode when it exists; fall back to the sample.
    allp = pathlib.Path("/root/nld/all_games.json")
    if allp.exists():
        nao = [g for g in json.loads(allp.read_text()) if g["turns"] >= 100]
        # One game in 28,040 reports a mangled turn counter: an 80-column status
        # line that fills the row loses its trailing field, so digits splice.
        # The longest genuine NAO game in the xlogfile is 867,759 turns.
        dropped = [g for g in nao if g["turns"] > 1_200_000]
        for g in dropped:
            g["c"] = [p for p in g["c"] if p[0] <= 1_200_000]
            g["turns"] = g["c"][-1][0] if g["c"] else 0
        if dropped:
            print(f"clamped {len(dropped)} game(s) with an impossible turn counter")
        nao_source = f"all {len(nao):,} decoded games (turns >= 100)"
    else:
        raw = json.loads(pathlib.Path("/root/nld/games_sample100.json").read_text())
        nao = [{"c": [[p[0], p[3], p[4]] for p in g["curve"]], "turns": g["derived_turns"],
                "max_dlvl": g["derived_maxlvl"], "balrog_max": g["balrog_max"],
                "balrog_min": g["balrog_min"], "role": g["role"]} for g in raw]
        nao_source = f"{len(nao)} sampled games"
    print("NAO source:", nao_source)

    top = json.loads(pathlib.Path("/root/nld/top10_curves.json").read_text())
    top = [g for g in top if g["complete"] and g["turns"] >= 100]

    # --- who actually beat the game ---
    # NAO: the xlogfile death field is authoritative and already carried through.
    for g in nao:
        g["won"] = str(g.get("death", "")).startswith("ascended")
    # top10 has no xlogfile; use the validated endgame scan, joined on game_key.
    eg = {r["game_key"]: r for r in
          json.loads(pathlib.Path("/root/nld/endgame_states_top10.json").read_text())}
    joined = 0
    for g in top:
        r = eg.get(g.get("game_key"))
        if r:
            joined += 1
        g["won"] = bool(r and r["ascended"])
        g["planes"] = bool(r and r["reached_planes"])
    print(f"top10 endgame join: {joined}/{len(top)} games matched; "
          f"{sum(g['won'] for g in top)} wins, {sum(g['planes'] for g in top)} reached the planes")
    print(f"nao wins (xlogfile): {sum(g['won'] for g in nao)}")

    # Our agent: GLM-5.2 only (the workhorse model behind every headline cell),
    # split by scaffold x observability. Smoke/probe cells are excluded.
    raw = json.loads(pathlib.Path("/root/nld/agent_curves.json").read_text())
    SKIP = {"e5_smoke", "outputs", "trace_probe", "?"}
    glm = [g for g in raw
           if g["turns"] > 1 and g["model"] == "z-ai/glm-5.2"
           and g["run"] not in SKIP and g["vision"] in ("on", "off")]
    variants = {
        "pa_vision": ("Prime Agent · vision", "prime_agent", "on"),
        "pa_fog": ("Prime Agent · fog", "prime_agent", "off"),
        "cc_vision": ("Claude Code · vision", "claude_code", "on"),
        "cc_fog": ("Claude Code · fog", "claude_code", "off"),
    }
    ag = [g for g in glm if g["arm"] == "prime_agent" and g["vision"] == "on"]

    grid = sorted({int(round(10 ** (i * 4.7 / 130))) for i in range(131)} | {1})
    grid = [t for t in grid if t <= 50000]

    data = {
        "grid": grid,
        "pop": {
            "nao": {"label": "NAO population", "stats": endpoint_stats(nao),
                    "series": summarise(nao, grid),
                    "wins": win_block(nao, grid),
                    "games": [{"c": thin(g["c"], 60), "turns": g["turns"]} for g in draw_set(nao)],
                    "n_drawn": min(len(nao), 120)},
            "top10": {"label": "NAO top 10", "stats": endpoint_stats(top),
                      "series": summarise(top, grid),
                    "wins": win_block(top, grid),
                      "games": [{"c": thin(g["c"], 60), "turns": g["turns"]} for g in draw_set(top)],
                      "n_drawn": min(len(top), 120)},
            "agent": {"label": "Prime Agent \u00b7 vision", "stats": endpoint_stats(ag),
                      "series": summarise(ag, grid),
                    "wins": win_block(ag, grid),
                      "games": [{"c": thin(g["c"], 60), "turns": g["turns"]} for g in draw_set(ag)],
                      "n_drawn": min(len(ag), 120)},
        },
        "agent_variants": {
            key: {
                "label": label,
                "stats": endpoint_stats(sub),
                "series": summarise(sub, grid),
                "games": [{"c": thin(g["c"], 60), "turns": g["turns"]} for g in draw_set(sub)],
            }
            for key, (label, arm, vis) in variants.items()
            if (sub := [g for g in glm if g["arm"] == arm and g["vision"] == vis])
        },
        "agent_model": "z-ai/glm-5.2",
        "nao_source": nao_source,
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
    for k, v in list(data["pop"].items()) + list(data["agent_variants"].items()):
        s = v["stats"]
        print(f"{k:7s} n={s['n']:4d}  bmax med {s['bmax_median']:6.2f} mean {s['bmax_mean']:6.2f} "
              f"p90 {s['bmax_p90']:6.2f} best {s['bmax_best']:6.2f} | "
              f"bmin med {s['bmin_median']:5.2f} | turns med {s['turns_median']:6d} "
              f"max {s['turns_max']:6d} | dlvl med {s['dlvl_median']:2d} best {s['dlvl_best']:2d}")


if __name__ == "__main__":
    main()
