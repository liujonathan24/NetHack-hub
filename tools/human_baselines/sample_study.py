"""Sample NLD-NAO games, derive BALROG curves from ttyrecs, validate vs xlogfile.

Reports the things that decide whether human BALROG-over-time is derivable:
  * what fraction of frames carry a parseable status line
  * how often the optional `T:` (turn counter) field is present
  * how often the level field is NOT "Dlvl:N" (Quest / Ft. Ludios / endgame)
  * how often "Xp:" is replaced by "HD:" (polymorph)
  * derived max-Dlvl vs the xlogfile's ground-truth `maxlvl`
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import multiprocessing as mp
import pathlib
import random
import re
import sys

sys.path.insert(0, "/root/nld")
from ttyrec_probe import scan_game, balrog_curve  # noqa: E402

FNAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.(\d{2}):(\d{2}):(\d{2})")


def sample_files(root: pathlib.Path, n: int, seed: int = 0) -> list[pathlib.Path]:
    players = sorted(p for p in root.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(players)
    out: list[pathlib.Path] = []
    for pl in players:
        files = sorted(pl.glob("*.ttyrec*"))
        if not files:
            continue
        out.append(rng.choice(files))
        if len(out) >= n:
            break
    return out


def analyse(path: pathlib.Path) -> dict:
    try:
        g = scan_game(path)
    except Exception as e:
        return {"file": str(path), "error": repr(e)}
    curve = g["curve"]
    kinds = collections.Counter(s["kind"] for s in curve)
    rec = {
        "file": str(path),
        "player": path.parent.name,
        "role": g["role"],
        "n_frames": g["n_frames"],
        "n_status_frames": g["n_status_frames"],
        "status_frac": round(g["n_status_frames"] / max(1, g["n_frames"]), 4),
        "n_distinct_states": len(curve),
        "has_turns": any(s["turns"] is not None for s in curve),
        "turns_frac": round(
            sum(s["turns"] is not None for s in curve) / max(1, len(curve)), 4
        ),
        "kinds": dict(kinds),
        "n_polyd": sum(s["polyd_hd"] is not None for s in curve),
        "max_dlvl": max([s["depth"] for s in curve if s["depth"]] or [0]),
        "max_xp": max([s["xp_level"] for s in curve if s["xp_level"]] or [0]),
        "max_turns": max([s["turns"] for s in curve if s["turns"]] or [0]),
    }
    bc = balrog_curve(curve)
    rec["balrog_max"] = bc[-1]["balrog_max"] if bc else 0.0
    rec["balrog_min"] = bc[-1]["balrog_min"] if bc else 0.0
    # coarse curve for plotting/inspection: one point per 5% of the trajectory
    step = max(1, len(bc) // 20)
    rec["curve"] = bc[::step][:21]
    m = FNAME_RE.search(path.name)
    if m:
        d = dt.datetime.strptime(
            f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=dt.timezone.utc)
        rec["start_utc"] = int(d.timestamp())
    return rec


def join_xlog(records: list[dict], xlog: pathlib.Path, tol: int = 180) -> None:
    """Attach xlogfile ground truth by (player name, start time within tol)."""
    want = collections.defaultdict(list)
    for r in records:
        if "start_utc" in r:
            want[r["player"]].append(r)
    hits = 0
    with open(xlog, "r", errors="replace") as fh:
        for line in fh:
            if "\tname=" in line:
                fields = dict(
                    kv.split("=", 1) for kv in line.rstrip("\n").split("\t") if "=" in kv
                )
            elif ":name=" in line:
                fields = dict(
                    kv.split("=", 1) for kv in line.rstrip("\n").split(":") if "=" in kv
                )
            else:
                continue
            name = fields.get("name")
            if name not in want:
                continue
            st = fields.get("starttime")
            if not st or not st.isdigit():
                continue
            st = int(st)
            for r in want[name]:
                if abs(st - r["start_utc"]) <= tol and "xlog" not in r:
                    r["xlog"] = {
                        k: fields.get(k)
                        for k in ("maxlvl", "deathlev", "deathdnum", "role", "turns",
                                  "points", "death", "version", "maxhp")
                    }
                    hits += 1
    print(f"xlog join: {hits}/{len(records)} games matched", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/nld/nld-nao/unz/nld-nao-unzipped")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--xlog", default="/root/nld/nld-nao/xlogfile.full.txt")
    ap.add_argument("--out", default="/root/nld/sample_study.json")
    args = ap.parse_args()

    files = sample_files(pathlib.Path(args.root), args.n)
    print(f"sampled {len(files)} games from {args.root}", flush=True)
    with mp.Pool(args.procs) as pool:
        records = pool.map(analyse, files, chunksize=1)

    ok = [r for r in records if "error" not in r]
    print(f"decoded {len(ok)}/{len(records)}")
    join_xlog(ok, pathlib.Path(args.xlog))
    pathlib.Path(args.out).write_text(json.dumps(records, indent=1))

    # ---- summary ----
    nonempty = [r for r in ok if r["n_status_frames"] > 0]
    print(f"games with >=1 status frame: {len(nonempty)}/{len(ok)}")
    fr = sum(r["n_status_frames"] for r in nonempty) / max(
        1, sum(r["n_frames"] for r in nonempty)
    )
    print(f"frames carrying a status line: {fr:.3%}")
    print(
        "games with T: turn counter: "
        f"{sum(r['has_turns'] for r in nonempty)}/{len(nonempty)}"
    )
    kinds = collections.Counter()
    for r in nonempty:
        kinds.update(r["kinds"])
    print("level-field kinds (state transitions):", dict(kinds))
    print("games with polymorph (HD:) frames:",
          sum(r["n_polyd"] > 0 for r in nonempty))

    matched = [r for r in nonempty if "xlog" in r and r["xlog"].get("maxlvl")]
    agree = [r for r in matched if int(r["xlog"]["maxlvl"]) == r["max_dlvl"]]
    print(f"\nxlog-validated: {len(matched)} games; derived max_dlvl == xlog maxlvl "
          f"for {len(agree)} ({len(agree)/max(1,len(matched)):.1%})")
    for r in matched:
        if r not in agree:
            print("  MISMATCH", r["player"], "derived", r["max_dlvl"],
                  "xlog", r["xlog"]["maxlvl"], "deathlev", r["xlog"]["deathlev"],
                  "kinds", r["kinds"], r["xlog"].get("death"))


if __name__ == "__main__":
    main()
