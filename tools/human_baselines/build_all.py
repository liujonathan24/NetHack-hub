"""Decode EVERY reconstructable game in the downloaded NLD-NAO shard.

Uses the fast status decoder (fast_scan) and the xlogfile game-bracketing from
build_games, so the output is the same curve the sampled pipeline produced --
just for all ~49k games instead of 96. Curves are thinned for storage; all
summary statistics are computed from the full curve before thinning.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import sys
import time

sys.path.insert(0, "/root/nld")
sys.path.insert(0, "/root/NetHack-hub/environments/nethack")

import fast_scan  # noqa: E402
from build_games import ROOT, file_start, index_games  # noqa: E402
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402


def assign_exclusive(games: list[dict], files: list[pathlib.Path]) -> None:
    """Give every session file to exactly ONE game.

    `build_games.assign_files` gives a file to every game whose
    [starttime, endtime] window contains it, which is wrong for saved
    characters: a game resumed months (measured: up to 4.7 years) after it
    started has a window that swallows every game played in between. Measured
    on 400 players, 13.4% of files were claimed by 2+ games that way.

    A file belongs to the most recently started game that was still open when
    the file was written -- the game actually being played at that moment.
    """
    for g in games:
        g["files"] = []
    if not games:
        return
    by_start = sorted(games, key=lambda g: g["start"])
    for p in files:
        ts = file_start(p)
        if ts is None:
            continue
        owner = None
        for g in by_start:
            if g["start"] - 120 <= ts <= g["end"]:
                owner = g  # later starts overwrite earlier ones
        if owner is not None:
            owner["files"].append(str(p))


def thin(curve, keep=60):
    if len(curve) <= keep:
        return curve
    step = len(curve) / keep
    out = [curve[int(i * step)] for i in range(keep)]
    out.append(curve[-1])
    return out


def one_game(g: dict) -> dict | None:
    samples = []
    for path in g["files"]:
        try:
            got, _mode = fast_scan.scan_any(path)
        except Exception:
            continue
        samples.extend(got)
    if not samples:
        return None
    curve, max_dl, max_xp, last_t = [], 0, 0, -1
    for t, dl, xp in samples:
        if t < last_t:
            continue
        last_t = t
        max_dl, max_xp = max(max_dl, dl), max(max_xp, xp)
        mx, mn = balrog_both(min(max_dl, 50), max_xp)
        pt = [t, round(mx * 100, 3), round(mn * 100, 3)]
        if curve and curve[-1][1:] == pt[1:]:
            curve[-1] = pt
        else:
            curve.append(pt)
    if not curve:
        return None
    return {
        "player": g["player"], "start": g["start"], "role": g["role"],
        "xlog_maxlvl": g["maxlvl"], "xlog_turns": g["turns"],
        "death": g["death"][:60], "n_files": len(g["files"]),
        "turns": curve[-1][0], "max_dlvl": max_dl, "max_xp": max_xp,
        "balrog_max": curve[-1][1], "balrog_min": curve[-1][2],
        "n_points": len(curve), "c": thin(curve),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/root/nld/all_games.json")
    args = ap.parse_args()

    players = {p.name for p in ROOT.iterdir() if p.is_dir()}
    idx = index_games(players)
    games = []
    for name, gs in idx.items():
        assign_exclusive(gs, sorted((ROOT / name).glob("*.ttyrec*")))
        games.extend(g for g in gs if g["files"])
    games.sort(key=lambda g: (g["player"], g["start"]))
    if args.limit:
        games = games[: args.limit]
    n_files = sum(len(g["files"]) for g in games)
    print(f"{len(games)} games, {n_files} session files, {args.procs} procs", flush=True)

    t0 = time.time()
    out, done = [], 0
    with mp.Pool(args.procs) as pool:
        for rec in pool.imap_unordered(one_game, games, chunksize=4):
            done += 1
            if rec:
                out.append(rec)
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(games)} games  {el/60:.1f} min  "
                      f"eta {(el/done*(len(games)-done))/60:.0f} min", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out))
    print(f"decoded {len(out)}/{len(games)} in {(time.time()-t0)/60:.1f} min -> {args.out}")

    import statistics
    played = [g for g in out if g["turns"] >= 100]
    agree = [g for g in out if g["xlog_maxlvl"] == g["max_dlvl"]]
    print(f"derived max Dlvl == xlogfile for {len(agree)}/{len(out)} ({len(agree)/max(1,len(out)):.1%})")
    print(f"games with >=100 turns: {len(played)}")
    if played:
        bm = sorted(g["balrog_max"] for g in played)
        bn = sorted(g["balrog_min"] for g in played)
        q = lambda a, p: a[min(len(a) - 1, int(p * (len(a) - 1)))]  # noqa: E731
        print(f"BALROG max: median {statistics.median(bm):.2f} mean {statistics.fmean(bm):.2f} "
              f"p90 {q(bm,.9):.2f} p99 {q(bm,.99):.2f} best {bm[-1]:.2f}")
        print(f"BALROG min: median {statistics.median(bn):.2f} mean {statistics.fmean(bn):.2f} "
              f"p90 {q(bn,.9):.2f} best {bn[-1]:.2f}")
        print("roles:", collections.Counter(g["role"] for g in played).most_common(5))


if __name__ == "__main__":
    main()
