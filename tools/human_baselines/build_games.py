"""Reconstruct whole NLD-NAO games from session ttyrecs, using the xlogfile.

There is no game ID in the data. The join key is (player, starttime..endtime)
from the xlogfile: every session file whose filename timestamp falls inside a
game's [starttime, endtime] window belongs to that game. That bracketing makes
reconstruction deterministic instead of heuristic.

Outputs one JSON per game: the (turn, dlvl, xp, balrog_max, balrog_min) curve,
plus xlog ground truth for validation.
"""
from __future__ import annotations

import argparse
import bz2
import collections
import datetime as dt
import json
import multiprocessing as mp
import pathlib
import re
import struct
import sys

sys.path.insert(0, "/root/nld")
sys.path.insert(0, "/root/NetHack-engine")
sys.path.insert(0, "/root/NetHack-hub/environments/nethack")

import pyte  # noqa: E402

from nethack_harness.prompt.balrog import balrog_both  # noqa: E402
from ttyrec_probe import status_from_screen  # noqa: E402

ROOT = pathlib.Path("/root/nld/nld-nao/unz/nld-nao-unzipped")
XLOG = pathlib.Path("/root/nld/nld-nao/xlogfile.full.txt")
FNAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.(\d{2}):(\d{2}):(\d{2})")


def file_start(p: pathlib.Path) -> int | None:
    m = FNAME_RE.search(p.name)
    if not m:
        return None
    d = dt.datetime.strptime(
        f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def index_games(players: set[str]) -> dict[str, list[dict]]:
    """xlogfile records for the given players that carry start/end times."""
    out: dict[str, list[dict]] = collections.defaultdict(list)
    with open(XLOG, errors="replace") as fh:
        for line in fh:
            sep = "\t" if "\tname=" in line else (":" if ":name=" in line else None)
            if not sep:
                continue
            f = dict(kv.split("=", 1) for kv in line.rstrip("\n").split(sep) if "=" in kv)
            name = f.get("name")
            if name not in players:
                continue
            st, et = f.get("starttime"), f.get("endtime")
            if not (st and et and st.isdigit() and et.isdigit()):
                continue
            out[name].append(
                {
                    "player": name,
                    "start": int(st),
                    "end": int(et),
                    "maxlvl": int(f["maxlvl"]) if f.get("maxlvl", "").lstrip("-").isdigit() else None,
                    "turns": int(f["turns"]) if f.get("turns", "").isdigit() else None,
                    "role": f.get("role"),
                    "death": f.get("death", ""),
                    "points": int(f["points"]) if f.get("points", "").isdigit() else None,
                    "version": f.get("version"),
                }
            )
    return out


def assign_files(games: list[dict], files: list[pathlib.Path]) -> None:
    """Bracket each session file into the game whose window contains it."""
    stamped = sorted(((file_start(p), p) for p in files if file_start(p)), key=lambda t: t[0])
    for g in games:
        # a session belongs to the game if it starts within the window (allow a
        # small lead: the file is opened a beat before the game record starts)
        g["files"] = [str(p) for ts, p in stamped if g["start"] - 120 <= ts <= g["end"]]


def read_frames_raw(path: str, cap: int | None = None):
    data = bz2.open(path, "rb").read() if path.endswith(".bz2") else open(path, "rb").read()
    off, n = 0, 0
    while off + 12 <= len(data):
        sec, usec, ln = struct.unpack("<III", data[off : off + 12])
        off += 12
        if ln > 1 << 22 or off + ln > len(data):
            return
        yield data[off : off + ln]
        off += ln
        n += 1
        if cap and n >= cap:
            return


def curve_for_game(g: dict, max_frames: int = 400_000) -> dict:
    """Decode a game's session files in order into one monotone BALROG curve."""
    samples: list[tuple[int, int, int, bool]] = []
    n_frames = 0
    on_planes = False
    for path in g["files"]:
        screen = pyte.Screen(80, 24)
        stream = pyte.ByteStream(screen)
        for payload in read_frames_raw(path, max_frames - n_frames):
            n_frames += 1
            try:
                stream.feed(payload)
            except Exception:
                continue
            st = status_from_screen(screen.display)
            if st is None or st["turns"] is None or st["xp_level"] is None:
                continue
            if st["kind"] == "endgame":
                # The Elemental/Astral Planes print the plane name instead of
                # "Dlvl:N" -- BALROG scores them via its own achievement key,
                # which outranks any Dlvl value.
                on_planes = True
            if st["depth"] is None:  # Quest ("Home N") hides absolute depth
                continue
            s = (st["turns"], st["depth"], st["xp_level"], on_planes)
            if samples and samples[-1] == s:
                continue
            samples.append(s)
        if n_frames >= max_frames:
            break

    # running maxima -> BALROG; drop non-monotone turn samples (mid-redraw junk)
    curve, max_dl, max_xp, last_t = [], 0, 0, -1
    for t, dl, xp, planes in samples:
        if t < last_t:
            continue
        last_t = t
        max_dl, max_xp = max(max_dl, dl), max(max_xp, xp)
        # Dlvl 50 is the ceiling: no plane / ascension bonus applied.
        mx, mn = balrog_both(min(max_dl, 50), max_xp)
        pt = [t, max_dl, max_xp, round(mx * 100, 3), round(mn * 100, 3)]
        if curve and curve[-1][1:] == pt[1:]:
            curve[-1] = pt  # collapse flat stretches, keep the latest turn
        else:
            curve.append(pt)
    return {
        **{k: v for k, v in g.items() if k != "files"},
        "n_files": len(g["files"]),
        "n_frames": n_frames,
        "curve": curve,
        "derived_maxlvl": max_dl,
        "derived_maxxp": max_xp,
        "derived_turns": curve[-1][0] if curve else 0,
        "balrog_max": curve[-1][3] if curve else 0.0,
        "balrog_min": curve[-1][4] if curve else 0.0,
        "ascended": g["death"].startswith("ascended"),
    }


def _work(g):
    try:
        return curve_for_game(g)
    except Exception as e:
        return {"player": g.get("player"), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "ascension", "stats"], default="sample")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-turns", type=int, default=100)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    players = {p.name for p in ROOT.iterdir() if p.is_dir()}
    idx = index_games(players)
    print(f"{len(players)} players in shard, {sum(len(v) for v in idx.values())} "
          f"xlog games with start/end times", flush=True)

    files_by_player = {name: sorted((ROOT / name).glob("*.ttyrec*")) for name in idx}
    all_games = []
    for name, games in idx.items():
        assign_files(games, files_by_player[name])
        all_games.extend(games)
    covered = [g for g in all_games if g["files"]]
    print(f"{len(covered)} of {len(all_games)} xlog games have >=1 session file "
          f"in this shard", flush=True)

    if args.mode == "stats":
        # How fragmented are games, and does fragmentation track depth?
        by_files = collections.Counter(len(g["files"]) for g in covered)
        print("\nsession files per game -> n games:",
              dict(sorted(by_files.items())[:12]))
        deep = [g for g in covered if (g["maxlvl"] or 0) >= 10]
        long_ = [g for g in covered if (g["turns"] or 0) >= 5000]
        for label, sub in (("all", covered), ("maxlvl>=10", deep), ("turns>=5000", long_)):
            if not sub:
                continue
            single = sum(1 for g in sub if len(g["files"]) == 1)
            print(f"  {label:12s} n={len(sub):6d}  single-session {single/len(sub):6.1%}  "
                  f"median files {sorted(len(g['files']) for g in sub)[len(sub)//2]}")
        asc = [g for g in covered if g["death"].startswith("ascended")]
        print(f"\nascensions in shard: {len(asc)}")
        for g in asc[:10]:
            print(f"   {g['player']:16s} {g['role']} turns={g['turns']:7d} "
                  f"files={len(g['files']):3d} points={g['points']}")
        return

    if args.mode == "ascension":
        picked = [g for g in covered if g["death"].startswith("ascended")]
        picked.sort(key=lambda g: len(g["files"]))
        picked = picked[: args.n]
    else:
        pool = [g for g in covered if (g["turns"] or 0) >= args.min_turns]
        pool.sort(key=lambda g: (g["player"], g["start"]))
        step = max(1, len(pool) // args.n)
        picked = pool[::step][: args.n]

    print(f"decoding {len(picked)} games "
          f"({sum(len(g['files']) for g in picked)} session files)", flush=True)
    with mp.Pool(args.procs) as pool_:
        out = pool_.map(_work, picked, chunksize=1)

    ok = [g for g in out if "error" not in g and g["curve"]]
    pathlib.Path(args.out).write_text(json.dumps(ok))
    agree = [g for g in ok if g["maxlvl"] == g["derived_maxlvl"]]
    print(f"decoded {len(ok)}/{len(picked)}; derived maxlvl == xlog maxlvl for "
          f"{len(agree)} ({len(agree)/max(1,len(ok)):.1%})")
    tagree = [g for g in ok if g["turns"] == g["derived_turns"]]
    print(f"derived final turn == xlog turns for {len(tagree)} "
          f"({len(tagree)/max(1,len(ok)):.1%})")


if __name__ == "__main__":
    main()
