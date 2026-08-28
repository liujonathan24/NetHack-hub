"""BALROG curves for the DeepMind nao_top10 corpus (top 10 NAO players by Z-score).

The data is already decoded to (T, 24, 80) tty_chars tensors, so no ttyrec
replay is needed -- rows 22-23 are the status block and parse directly.

Same structural problem as NLD-NAO: a file is a login SESSION, not a game. The
published metadata.pkl (which carries `stamp`) is served as 10,240 null bytes,
so sessions cannot be ordered by timestamp. Instead sessions are chained on two
in-frame signals that are monotone within a game and effectively unique across
games: the turn counter T and cumulative experience points.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, "/root/NetHack-hub/environments/nethack")
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402

ROOT = pathlib.Path("/root/nld/top10")
DLVL = re.compile(r"Dlvl:\s*(-?\d+)")
PLANE = re.compile(r"(?:Astral Plane|End Game|Plane of (?:Earth|Air|Fire|Water)|\b(?:Astral|Water|Fire|Air|Earth)\b)")
XP = re.compile(r"\b(?:Xp|Exp):(\d+)(?:/(\d+))?")
T_RE = re.compile(r"\bT:(\d+)")
HP = re.compile(r"\bHP:(-?\d+)\((\d+)\)")


def session_samples(path: pathlib.Path) -> list[tuple[int, int, int, int]]:
    """(turn, depth, xp_level, exp_points) per distinct state in one session."""
    tc = np.load(path)["tty_chars"]
    out = []
    last = None
    for i in range(tc.shape[0]):
        block = "\n".join(
            bytes(tc[i][r]).decode("ascii", "replace") for r in (21, 22, 23)
        )
        mt, mx = T_RE.search(block), XP.search(block)
        if not (mt and mx):
            continue
        md = DLVL.search(block)
        if md:
            depth = int(md.group(1))
        elif PLANE.search(block) and HP.search(block):
            depth = 50  # endgame plane: no Dlvl printed; 50 is our ceiling
        else:
            continue  # Quest ("Home N") hides absolute depth
        s = (int(mt.group(1)), depth, int(mx.group(1)),
             int(mx.group(2)) if mx.group(2) else 0)
        if s != last:
            out.append(s)
            last = s
    return out


def chain(sessions: list[dict]) -> list[list[dict]]:
    """Group sessions into games on (turn, experience-point) continuity."""
    games, used = [], set()
    # a game starts with a session that begins at the very start of a game
    starts = sorted((s for s in sessions if s["minT"] <= 5), key=lambda s: s["minT"])
    pool = sorted(sessions, key=lambda s: s["minT"])
    for s0 in starts:
        if s0["id"] in used:
            continue
        game = [s0]
        used.add(s0["id"])
        cur = s0
        while True:
            nxt = None
            for s in pool:
                if s["id"] in used:
                    continue
                # continues if it picks up at the turn AND the experience total
                # the previous session ended on (both monotone within a game)
                if abs(s["minT"] - cur["maxT"]) <= 50 and s["minXPP"] >= cur["maxXPP"] \
                        and s["minXP"] >= cur["maxXP"] - 1:
                    if nxt is None or s["minT"] < nxt["minT"]:
                        nxt = s
            if nxt is None:
                break
            game.append(nxt)
            used.add(nxt["id"])
            cur = nxt
        games.append(game)
    # sessions never claimed by a start (their opening session is missing)
    for s in pool:
        if s["id"] not in used:
            games.append([s])
            used.add(s["id"])
    return games


def _summarise(f):
    return f, session_samples(pathlib.Path(f))


def build(user: str, procs: int = 14) -> list[dict]:
    files = sorted((ROOT / user).glob("*.npz"))
    sessions = []
    with mp.Pool(procs) as pool:
        results = pool.map(_summarise, [str(f) for f in files], chunksize=1)
    for fstr, sm in results:
        f = pathlib.Path(fstr)
        if not sm:
            continue
        sessions.append({
            "id": f.stem, "file": str(f), "samples": sm,
            "minT": sm[0][0], "maxT": max(s[0] for s in sm),
            "minXP": min(s[2] for s in sm), "maxXP": max(s[2] for s in sm),
            "minXPP": min(s[3] for s in sm), "maxXPP": max(s[3] for s in sm),
        })
    out = []
    for game in chain(sessions):
        samples = [s for sess in game for s in sess["samples"]]
        samples.sort(key=lambda s: s[0])
        curve, max_dl, max_xp, last_t = [], 0, 0, -1
        for t, dl, xp, _pts in samples:
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
            continue
        out.append({
            "user": user, "game_key": f"{user}/{game[0]['id']}", "n_sessions": len(game),
            "complete": game[0]["minT"] <= 5,
            "turns": curve[-1][0], "max_dlvl": max_dl, "max_xp": max_xp,
            "balrog_max": curve[-1][1], "balrog_min": curve[-1][2], "c": curve,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--out", default="/root/nld/top10_curves.json")
    args = ap.parse_args()
    users = args.users or [d.name for d in ROOT.iterdir() if d.is_dir()]

    games = []
    for u in users:
        g = build(u, args.procs)
        games.extend(g)
        print(f"{u:12s} sessions={len(list((ROOT/u).glob('*.npz'))):4d} "
              f"games={len(g):3d} complete={sum(x['complete'] for x in g):3d} "
              f"best={max([x['balrog_max'] for x in g] or [0]):.1f}%", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(games))
    comp = [g for g in games if g["complete"] and g["turns"] >= 100]
    print(f"\n{len(games)} games, {len(comp)} complete with >=100 turns")
    if comp:
        import statistics
        bm = sorted(g["balrog_max"] for g in comp)
        bn = sorted(g["balrog_min"] for g in comp)
        print(f"BALROG max: median {statistics.median(bm):.2f} mean {statistics.fmean(bm):.2f} best {bm[-1]:.2f}")
        print(f"BALROG min: median {statistics.median(bn):.2f} mean {statistics.fmean(bn):.2f} best {bn[-1]:.2f}")
        print("turns median", statistics.median([g["turns"] for g in comp]))
        print("sessions/game:", collections.Counter(g["n_sessions"] for g in comp))


if __name__ == "__main__":
    main()
