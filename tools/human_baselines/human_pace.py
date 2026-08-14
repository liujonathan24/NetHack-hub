"""Human BALROG-progression-vs-turns baseline from NLD-NAO ttyrecs.

A NAO ttyrec is a *login session*, not a game: one game can span several files
(save/resume, disconnects) and one player directory holds many games. This
script chains a player's files in time order and cuts a new game whenever the
turn counter resets, then emits, per game, the turn at which each BALROG
milestone was first reached.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import multiprocessing as mp
import pathlib
import random
import statistics
import sys

sys.path.insert(0, "/root/nld")
sys.path.insert(0, "/root/NetHack-engine")
sys.path.insert(0, "/root/NetHack-hub/environments/nethack")

import pyte  # noqa: E402

from nethack_harness.prompt.balrog import balrog_both  # noqa: E402
from ttyrec_probe import read_frames, status_from_screen  # noqa: E402


def player_games(pdir: pathlib.Path, max_frames_per_file=200_000):
    """Yield games as lists of (turn, dlvl, xp) samples, chaining session files.

    Turn-counter regressions are NOT immediately trusted as a new game. NetHack
    redraws the status line field by field, so a frame captured mid-redraw can
    show a truncated counter ("T:41" while "T:41148" is being written). Taking
    that at face value split single games into dozens of fake ones and produced
    nonsense pace numbers (BALROG 50% "reached" at turn 44). A reset must be
    confirmed by RESET_CONFIRM consecutive low samples before we cut.
    """
    games, cur, last_turn = [], [], -1
    pending: list = []
    for f in sorted(pdir.glob("*.ttyrec*")):
        screen = pyte.Screen(80, 24)
        stream = pyte.ByteStream(screen)
        try:
            frames = list(read_frames(f, max_frames_per_file))
        except Exception:
            continue
        for _, _, payload in frames:
            try:
                stream.feed(payload)
            except Exception:
                continue
            st = status_from_screen(screen.display)
            if st is None or st["turns"] is None or st["depth"] is None:
                continue
            t = st["turns"]
            if st["xp_level"] is None:
                continue
            sample = (t, st["depth"], st["xp_level"])
            if t < last_turn:
                # candidate reset (or a mid-redraw truncation) -- hold it back
                pending.append(sample)
                if len(pending) < RESET_CONFIRM:
                    continue
                # A real new game starts on Dlvl 1 at turn <= 3. Requiring the
                # depth axis too rejects the mid-redraw truncation that can
                # persist for several consecutive frames ("T:11" held on screen
                # while "T:111xx" is drawn), which a turn-only rule accepts.
                if pending[-1][1] == 1 and pending[-1][0] <= 3:
                    if cur:
                        games.append(cur)
                    cur = list(pending)
                    last_turn = pending[-1][0]
                pending = []
                continue
            pending = []
            last_turn = t
            if cur and cur[-1] == sample:
                continue
            cur.append(sample)
    if cur:
        games.append(cur)
    return games


RESET_CONFIRM = 5
MILESTONES = [1, 2, 5, 10, 15, 20, 30, 40, 50]


def game_summary(samples):
    max_dl = max_xp = 0
    first_at = {}
    for t, dl, xp in samples:
        max_dl, max_xp = max(max_dl, dl), max(max_xp, xp)
        mx, mn = balrog_both(max_dl, max_xp)
        for m in MILESTONES:
            if mx * 100 >= m and m not in first_at:
                first_at[m] = t
    mx, mn = balrog_both(max_dl, max_xp)
    return {
        "n_samples": len(samples),
        "turns": samples[-1][0] if samples else 0,
        "max_dlvl": max_dl,
        "max_xp": max_xp,
        "balrog_max": round(mx * 100, 3),
        "balrog_min": round(mn * 100, 3),
        "first_turn_at_pct": first_at,
    }


def work(pdir: str):
    try:
        out = []
        for g in player_games(pathlib.Path(pdir)):
            if len(g) <= 3:
                continue
            s = game_summary(g)
            s["player"] = pathlib.Path(pdir).name
            s["head"] = g[:5]
            s["tail"] = g[-3:]
            out.append(s)
        return out
    except Exception as e:
        return [{"error": repr(e), "player": pdir}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/nld/nld-nao/unz/nld-nao-unzipped")
    ap.add_argument("--players", type=int, default=60)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--max-files", type=int, default=12,
                    help="skip players with more session files than this (cost cap)")
    ap.add_argument("--out", default="/root/nld/human_pace.json")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    rng = random.Random(7)
    rng.shuffle(dirs)
    picked = []
    for d in dirs:
        n = len(list(d.glob("*.ttyrec*")))
        if 1 <= n <= args.max_files:
            picked.append(str(d))
        if len(picked) >= args.players:
            break
    print(f"{len(picked)} players", flush=True)

    with mp.Pool(args.procs) as pool:
        out = pool.map(work, picked, chunksize=1)
    games = [g for gs in out for g in gs if "error" not in g]
    pathlib.Path(args.out).write_text(json.dumps(games, indent=1))

    print(f"games reconstructed: {len(games)}")
    played = [g for g in games if g["turns"] >= 100]
    print(f"games with >=100 turns: {len(played)}")
    if played:
        bm = sorted(g["balrog_max"] for g in played)
        bn = sorted(g["balrog_min"] for g in played)
        print(f"BALROG max  : median {statistics.median(bm):.2f}  mean {statistics.mean(bm):.2f}  p90 {bm[int(.9*len(bm))-1]:.2f}  best {bm[-1]:.2f}")
        print(f"BALROG min  : median {statistics.median(bn):.2f}  mean {statistics.mean(bn):.2f}  p90 {bn[int(.9*len(bn))-1]:.2f}  best {bn[-1]:.2f}")
        print("\nturns to first reach BALROG %  (n games reaching it / median turns / p25 / p75)")
        for m in MILESTONES:
            ts = sorted(g["first_turn_at_pct"][str(m)] if isinstance(next(iter(g["first_turn_at_pct"]), ""), str)
                        else g["first_turn_at_pct"][m]
                        for g in played if (str(m) in g["first_turn_at_pct"] or m in g["first_turn_at_pct"]))
            if ts:
                print(f"  {m:3d}%  n={len(ts):4d}  median={statistics.median(ts):8.0f}  "
                      f"p25={ts[len(ts)//4]:7d}  p75={ts[3*len(ts)//4]:7d}")
            else:
                print(f"  {m:3d}%  n=0")


if __name__ == "__main__":
    main()
