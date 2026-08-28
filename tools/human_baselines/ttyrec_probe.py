"""Decode NLD-NAO human ttyrecs and extract a BALROG progression curve.

Answers: do the human (action-less) NAO recordings carry enough state to derive
the BALROG progression score -- and its `min` variant -- *over time*?

Pipeline per game:
  ttyrec.bz2 -> frames (12-byte LE header: sec, usec, len) -> pyte VT emulator
  -> 24x80 char grid -> bottom status lines -> (Dlvl, Xp level, T, HP) -> BALROG.

Usage:
  python ttyrec_probe.py <root-dir> [--games N] [--max-frames N] [--json out.json]
"""
from __future__ import annotations

import argparse
import bz2
import json
import pathlib
import re
import struct
import sys
import time

import pyte

sys.path.insert(0, "/root/NetHack-engine")
sys.path.insert(0, "/root/NetHack-hub/environments/nethack")
from nethack_core.nld_parse import parse_status, detect_role  # noqa: E402
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402

# Level-description field variants (botl.c describe_level): normal dungeon is
# "Dlvl:N"; the Quest shows "Home N"; Ft. Ludios its dungeon name; the endgame
# planes show the plane name. Polymorph replaces "Xp:N/n" with "HD:n".
_DLVL_RE = re.compile(r"Dlvl:\s*(-?\d+)")
_HOME_RE = re.compile(r"\bHome\s+(\d+)")
_PLANE_RE = re.compile(r"(?:Astral Plane|End Game|Plane of (?:Earth|Air|Fire|Water)|\b(?:Astral|Water|Fire|Air|Earth)\b)")
_XP_RE = re.compile(r"\b(?:Xp|Exp):(\d+)")
_HD_RE = re.compile(r"\bHD:(\d+)")
_T_RE = re.compile(r"\bT:(\d+)")
_HP_RE = re.compile(r"\bHP:(-?\d+)\((\d+)\)")


def read_frames(path: pathlib.Path, max_frames: int | None = None):
    """Yield (sec, usec, payload) frames from a (bz2) ttyrec."""
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rb") as fh:
        data = fh.read()
    off, n = 0, 0
    while off + 12 <= len(data):
        sec, usec, ln = struct.unpack("<III", data[off : off + 12])
        off += 12
        if ln > 1 << 22 or off + ln > len(data):  # corrupt / truncated tail
            break
        yield sec, usec, data[off : off + ln]
        off += ln
        n += 1
        if max_frames and n >= max_frames:
            return


def status_from_screen(lines: list[str]) -> dict | None:
    """Pull (depth, xp_level, turns, hp) out of a rendered screen.

    Scans every row rather than assuming rows 22-23: NAO players use terminals
    taller than 24 rows, which shifts the status block.
    """
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        m_dl, m_home, m_plane = _DLVL_RE.search(line), _HOME_RE.search(line), None
        if not (m_dl or m_home):
            # endgame plane names only count if the line also looks like status
            if _HP_RE.search(line) and _PLANE_RE.search(line):
                m_plane = _PLANE_RE.search(line)
            else:
                continue
        if m_dl:
            depth, kind = int(m_dl.group(1)), "dlvl"
        elif m_home:
            depth, kind = None, "quest"  # Quest hides absolute depth
        else:
            depth, kind = 50, "endgame"
        blob = "\n".join(lines[max(0, i - 1) : i + 2])
        m_xp, m_hd, m_t, m_hp = (
            _XP_RE.search(blob),
            _HD_RE.search(blob),
            _T_RE.search(blob),
            _HP_RE.search(blob),
        )
        return {
            "kind": kind,
            "depth": depth,
            "xp_level": int(m_xp.group(1)) if m_xp else None,
            "polyd_hd": int(m_hd.group(1)) if m_hd and not m_xp else None,
            "turns": int(m_t.group(1)) if m_t else None,
            "hp": int(m_hp.group(1)) if m_hp else None,
            "max_hp": int(m_hp.group(2)) if m_hp else None,
            "row": i,
            "raw": line.strip(),
        }
    return None


def scan_game(path: pathlib.Path, max_frames: int | None = None, rows=24, cols=80):
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)
    curve, role = [], None
    n_frames = n_status = 0
    t0 = None
    for sec, usec, payload in read_frames(path, max_frames):
        n_frames += 1
        if t0 is None:
            t0 = sec + usec / 1e6
        try:
            stream.feed(payload)
        except Exception:
            continue
        lines = screen.display
        if role is None:
            role = detect_role("\n".join(lines))
        st = status_from_screen(lines)
        if st is None:
            continue
        n_status += 1
        st["frame"] = n_frames
        st["t"] = round(sec + usec / 1e6 - t0, 2)
        if curve and _same(curve[-1], st):
            continue
        curve.append(st)
    return {
        "file": str(path),
        "role": role,
        "n_frames": n_frames,
        "n_status_frames": n_status,
        "curve": curve,
    }


def _same(a: dict, b: dict) -> bool:
    return (a["depth"], a["xp_level"], a["turns"]) == (b["depth"], b["xp_level"], b["turns"])


def balrog_curve(curve: list[dict]) -> list[dict]:
    """Running (max, min) BALROG progression over the trajectory."""
    out, max_dl, max_xp = [], 0, 0
    for st in curve:
        if st["depth"] is not None:
            max_dl = max(max_dl, st["depth"])
        if st["xp_level"] is not None:
            max_xp = max(max_xp, st["xp_level"])
        mx, mn = balrog_both(max_dl, max_xp)
        out.append(
            {
                "turns": st["turns"],
                "t": st["t"],
                "dlvl": max_dl,
                "xp": max_xp,
                "balrog_max": round(mx * 100, 3),
                "balrog_min": round(mn * 100, 3),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    files = []
    for p in root.rglob("*.ttyrec*"):
        if p.is_file():
            files.append(p)
        if len(files) >= args.games:
            break

    results = []
    for p in files:
        t0 = time.time()
        try:
            g = scan_game(p, args.max_frames)
        except Exception as e:  # keep going; report the failure rate
            results.append({"file": str(p), "error": repr(e)})
            continue
        g["curve_balrog"] = balrog_curve(g["curve"])
        g["secs"] = round(time.time() - t0, 2)
        last = g["curve_balrog"][-1] if g["curve_balrog"] else None
        results.append(g)
        print(
            f"{p.name:44s} frames={g['n_frames']:6d} status={g['n_status_frames']:6d} "
            f"role={g['role']} final={last} {g['secs']}s",
            flush=True,
        )

    ok = [r for r in results if "error" not in r]
    withstat = [r for r in ok if r["n_status_frames"] > 0]
    print(f"\ngames={len(results)} parsed={len(ok)} with_status_line={len(withstat)}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
