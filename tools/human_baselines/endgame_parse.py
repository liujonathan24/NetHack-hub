"""Detect endgame (the Planes) and ascension in human NetHack recordings.

Emits one record per *game* to `/root/nld/endgame_states.json` so the main
analysis can join on it. Nothing here changes how the existing scripts score the
main plots -- this is a new, additive side-channel that supplies the two BALROG
achievement flags (`reached_planes` -> "Astral Plane" = 0.8746, `ascended` ->
"You ascend t" = 1.0) that a (Dlvl, Xp) curve alone cannot express.

WHAT WE MATCH, AND WHY
----------------------
Level field (status line 2 starts with it -- `botl.c:describe_level`):

  NetHack 3.6.x            NetHack 3.4.3            meaning
  ---------------------    ---------------------    -------------------------
  "Dlvl:N "                "Dlvl:N "                normal dungeon
  "Home N "                "Home N "                the Quest
  <dungeon name>           <dungeon name>           Ft. Ludios
  "Astral Plane "          "Astral Plane "          the Astral Plane
  "Earth "/"Air "/         "End Game "              an Elemental Plane
    "Fire "/"Water "

3.6.2 replaced 3.4.3's generic "End Game " with the per-plane name (with
"Plane of " stripped).  **Every game in the NLD-NAO shard on this machine is
3.4.3** (49,234/49,234 xlog records, all 438 ascensions), and the nao_top10
tensors show the 3.4.3 `Dlvl:%-2d $:%-2ld` layout too -- so `End Game` is the
string that actually does the work here.  Both spellings are matched anyway.

Ascension is a *message*, not a status field.  The winning branch in
`pray.c:dosacrifice()` (identical text in 3.4.3 and 3.6.6) prints, in order:

    You offer the Amulet of Yendor to <god>...            <- NOT proof: the
                                                            wrong-god and
                                                            Moloch branches
                                                            print this too and
                                                            then `done(ESCAPED)`
    An invisible choir sings, and you are bathed in radiance...
    The voice of <god> booms: "Mortal, thou hast done well!"
    "In return for thy service, I grant thee the gift of Immortality!"
    You ascend to the status of Demigod[dess]...
    ... then `done(ASCENDED)` -> "Goodbye <name> the Demigod[dess]..."

The godvoice line is *not* used: its verb is picked by `rn2()` from
`godvoices[]`, and in practice it did not even survive into the frame text of
the ascension we hand-checked.  The four markers we do use are unique to the
winning branch.  `topten.c` writes other players' wins as "ascended to
demigod-hood" (lower-case, different wording), so the high-score list that
NetHack prints at the end of *any* game cannot spoof us.

Usage:
  /root/nld/venv/bin/python endgame_parse.py --corpus both --procs 14
"""
from __future__ import annotations

import argparse
import bz2
import collections
import json
import multiprocessing as mp
import os
import pathlib
import re
import statistics
import struct
import sys
import time

sys.path.insert(0, "/root/nld")
sys.path.insert(0, "/root/NetHack-engine")
sys.path.insert(0, "/root/NetHack-hub/environments/nethack")

import numpy as np  # noqa: E402
import pyte  # noqa: E402

from build_games import ROOT as NAO_ROOT, assign_files, index_games  # noqa: E402
from top10_curves import ROOT as TOP10_ROOT, chain  # noqa: E402

OUT_JSON = pathlib.Path("/root/nld/endgame_states.json")
OUT_MD = pathlib.Path("/root/nld/endgame_summary.md")

# ---------------------------------------------------------------- matchers ---

HP_RE = re.compile(r"HP:(-?\d+)\((\d+)\)")
T_RE = re.compile(r"\bT:(\d+)")
XP_RE = re.compile(r"\b(?:Xp|Exp):(\d+)(?:/(\d+))?")
HD_RE = re.compile(r"\bHD:(\d+)")
DLVL_RE = re.compile(r"Dlvl:\s*(-?\d+)")
HOME_RE = re.compile(r"^ {0,3}Home\s+(\d+)\b")

# Anchored at the start of the status line: `describe_level` writes the level
# field first, so a bare "Air"/"Fire" can never be confused with map or message
# text (and the line must carry HP:(...) as well).
PLANE_RE = re.compile(
    r"^ {0,3}(Astral Plane|End Game|Plane of (?:Earth|Air|Fire|Water)"
    r"|Earth|Air|Fire|Water)\b"
)
_PLANE_CANON = {
    "Astral Plane": "Astral",
    "End Game": "Elemental(3.4.3 'End Game')",
    "Plane of Earth": "Earth", "Earth": "Earth",
    "Plane of Air": "Air", "Air": "Air",
    "Plane of Fire": "Fire", "Fire": "Fire",
    "Plane of Water": "Water", "Water": "Water",
}

ASC_PATTERNS = [
    ("ascend_demigod", re.compile(r"You\s+ascend\s+to\s+the\s+status\s+of\s+Demigod(?:dess)?")),
    ("choir_radiance", re.compile(r"invisible\s+choir\s+sings,\s+and\s+you\s+are\s+bathed\s+in\s+radiance")),
    ("gift_immortality", re.compile(r"grant\s+thee\s+the\s+gift\s+of\s+Immortality")),
    # The farewell is ROLE-dependent (role.c:Goodbye()): Knight "Fare thee
    # well", Samurai "Sayonara", Tourist "Aloha", Valkyrie "Farvel" (Norse),
    # everyone else "Goodbye". Matching only "Goodbye" silently drops this
    # marker for four of the thirteen roles.
    ("goodbye_demigod", re.compile(
        r"(?:Goodbye|Farvel|Sayonara|Aloha|Fare\s+thee\s+well)"
        r"\s+\S+\s+the\s+Demigod(?:dess)?")),
]
OFFER_RE = re.compile(r"You\s+offer\s+the\s+Amulet\s+of\s+Yendor\s+to")

# cheap byte prefilters (raw ttyrec payload contains the literal message text)
ASC_BYTES = (b"Demigod", b"radiance", b"Immortality")
OFFER_BYTES = (b"Amulet of Yendor to",)

_ROLES = ("Archeologist", "Barbarian", "Caveman", "Cavewoman", "Healer", "Knight",
          "Monk", "Priest", "Priestess", "Ranger", "Rogue", "Samurai", "Tourist",
          "Valkyrie", "Wizard")
WELCOME_RE = re.compile(
    r"welcome (?:back )?to NetHack!\s+You are an? [\w\-' ]*?(" + "|".join(_ROLES) + r")\b"
)

# A minority of NAO players ran the *curses* interface on an oversized terminal.
# It has no classic status line at all: the fields sit in a vertical panel down
# the right-hand side, labelled in full ("Dungeon Level: Astral Plane",
# "Hit Points: 285/285", "Time: 60914", "Experience: <points>/<level>").  Both
# labels are required before anything is read, so this can never fire on a
# classic status line.
CURSES_HP = re.compile(r"Hit Points:\s+(-?\d+)/(\d+)")
CURSES_LVL = re.compile(r"Dungeon Level:\s+([A-Za-z0-9][A-Za-z0-9 ]*)")
CURSES_TIME = re.compile(r"Time:\s+(\d+)")
CURSES_XP = re.compile(r"Experience:\s+\d+/(\d+)")
CURSES_PLANE = re.compile(r"^(Astral Plane|End Game|The Elemental Planes"
                          r"|Plane of (?:Earth|Air|Fire|Water)|Earth|Air|Fire|Water)$")

# Messages: the classic UI puts them on row 0; the curses UI puts them in a
# bordered window several rows deep. Scanning the top nine rows covers both and
# costs nothing (map rows cannot contain these sentences).
MSG_ROWS = tuple(range(0, 9))
STATUS_ROWS = (21, 22, 23)

# (columns, rows, status rows to read, try the curses panel) -- tried in order
# until one yields a parseable status line.
PROFILES = (
    (80, 24, STATUS_ROWS, False),
    (80, 24, tuple(range(23, 9, -1)), False),
    (200, 60, tuple(range(59, -1, -1)), True),
)


def parse_curses_panel(text: str) -> dict | None:
    """Read the curses interface's vertical status panel off a whole screen."""
    if "Hit Points:" not in text or "Dungeon Level:" not in text:
        return None
    m_lvl = CURSES_LVL.search(text)
    if not m_lvl or not CURSES_HP.search(text):
        return None
    val = re.sub(r"\s{2,}.*$", "", m_lvl.group(1)).strip()
    m_pl = CURSES_PLANE.match(val)
    if m_pl:
        kind, depth = "endgame", None
        plane = ("Elemental(curses 'The Elemental Planes')"
                 if val == "The Elemental Planes" else _PLANE_CANON.get(val, val))
    elif val.lstrip("-").isdigit():
        kind, depth, plane = "dlvl", int(val), None
    else:
        return None  # the Quest / Ft. Ludios / an unrecognised dungeon name
    m_t, m_x = CURSES_TIME.search(text), CURSES_XP.search(text)
    return {
        "kind": kind, "depth": depth, "plane": plane,
        "turns": int(m_t.group(1)) if m_t else None,
        "xp_level": int(m_x.group(1)) if m_x else None,
        "polyd": False,
        "raw": f"Dungeon Level: {val} | " + (m_t.group(0) if m_t else "Time:?"),
    }


def _new_agg() -> dict:
    """Per-session rollup of the numeric-Dlvl status samples.

    Only aggregates cross process boundaries -- a single long session can hold
    ~20k distinct status states, and shipping every one of them back from the
    worker pool for ~7k files would not fit in memory.  Maxima are unaffected by
    the mid-redraw stale-value problem (a stale value is a real earlier value,
    so it can never inflate a running max), and the level/turn fields we do use
    for evidence come with the raw status line attached.
    """
    return {"n": 0, "minT": None, "maxT": None, "maxDl": 0,
            "minXp": None, "maxXp": 0, "minXPP": 0, "maxXPP": 0}


def _agg_add(a: dict, turn: int, depth: int, xp, xpp: int = 0) -> None:
    a["n"] += 1
    a["minT"] = turn if a["minT"] is None else min(a["minT"], turn)
    a["maxT"] = turn if a["maxT"] is None else max(a["maxT"], turn)
    a["maxDl"] = max(a["maxDl"], depth)
    if xp is not None:
        a["minXp"] = xp if a["minXp"] is None else min(a["minXp"], xp)
        a["maxXp"] = max(a["maxXp"], xp)
    a["minXPP"] = xpp if a["n"] == 1 else min(a["minXPP"], xpp)
    a["maxXPP"] = max(a["maxXPP"], xpp)


def parse_status_block(lines: list[str], width: int = 80) -> dict | None:
    """(kind, depth, plane, xp_level, turns) from rendered rows, bottom-up.

    Requires HP:(n)(m) on the level-field line, which is what rejects mid-redraw
    fragments and stray map rows.  T:/Xp: may be read from the neighbouring rows
    (some terminals split the block) but the level field never is.
    """
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if not HP_RE.search(line):
            continue
        m_pl = PLANE_RE.match(line)
        m_dl = DLVL_RE.search(line)
        m_home = HOME_RE.match(line)
        if not (m_pl or m_dl or m_home):
            continue
        if m_pl:
            kind, depth, plane = "endgame", None, _PLANE_CANON[m_pl.group(1)]
        elif m_dl:
            kind, depth, plane = "dlvl", int(m_dl.group(1)), None
        else:
            kind, depth, plane = "quest", None, None
        blob = "\n".join(lines[max(0, i - 1): i + 2])
        # A status line whose content reaches the terminal's last column may
        # have had its trailing field clipped (seen on a game with 9-digit HP
        # and experience: "... Xp:30/114838732 T" -- the turn count is simply
        # gone). Never trust a value that ends at the edge, and never borrow one
        # from a neighbouring row for such a line: that is how a plane sighting
        # picked up "T:8" from unrelated text and reported turn 8 of an 867k
        # turn game.
        row_full = len(line.rstrip()) >= width
        m_t = T_RE.search(line)
        if m_t and row_full and m_t.end() >= width:
            m_t = None
        if m_t is None and not row_full:
            m_t = T_RE.search(blob)
        m_x = XP_RE.search(line) or (None if row_full else XP_RE.search(blob))
        m_hd = HD_RE.search(line) or (None if row_full else HD_RE.search(blob))
        return {
            "kind": kind, "depth": depth, "plane": plane,
            "turns": int(m_t.group(1)) if m_t else None,
            "xp_level": int(m_x.group(1)) if m_x else None,
            "polyd": bool(m_hd and not m_x),
            "raw": line.rstrip(),
        }
    return None


def scan_messages(lines: list[str], turn, out: dict, where: str) -> None:
    """Record ascension / Amulet-offer evidence from the message rows."""
    text = "\n".join(lines)
    for name, rx in ASC_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        if name not in out["asc"]:
            out["asc"][name] = {
                "marker": name, "turn": turn, "file": where,
                "text": " ".join(text[max(0, m.start() - 10): m.end() + 30].split()),
            }
    m = OFFER_RE.search(text)
    if m and out["offer"] is None:
        out["offer"] = {"turn": turn, "file": where,
                        "text": " ".join(text[m.start(): m.start() + 70].split())}


# ------------------------------------------------------------- NLD-NAO -------

def _frames(data: bytes):
    off = 0
    while off + 12 <= len(data):
        _s, _u, ln = struct.unpack("<III", data[off: off + 12])
        off += 12
        if ln > 1 << 22 or off + ln > len(data):
            return
        yield data[off: off + ln]
        off += ln


# Raw-byte forms of the status fields.  NetHack's tty backend writes each status
# line with a single putstr(), so the field text lands in the ttyrec payload
# contiguously and can be read without emulating the terminal at all.  This is
# used ONLY for the running maxima (max Dlvl / max Xp / last turn), which are
# insensitive to partial redraws -- a stale fragment is still a value the hero
# really had.  Every *endgame claim* is still made from a pyte-rendered screen.
RAW_DLVL = re.compile(rb"Dlvl:\s*(\d{1,2})|Dungeon Level:\s+(\d{1,2})\b")
RAW_XP = re.compile(rb"\b(?:Xp|Exp):(\d{1,2})\b|Experience:\s+\d+/(\d{1,2})\b")
RAW_T = re.compile(rb"\bT:(\d{1,8})\b|Time:\s+(\d{1,8})\b")

# If none of these literals occur anywhere in a session's byte stream, that
# session cannot contain a Plane status line or an ascension message, so it
# never needs to be replayed.
EVIDENCE_BYTES = (b"Astral Plane", b"End Game", b"Plane of ",
                  b"The Elemental Planes", b"Demigod", b"radiance",
                  b"Immortality", b"Amulet of Yendor to")
PRE_FRAMES = 400  # replay this much lead-in so the screen is warm


def _row(screen, y: int) -> str:
    buf = screen.buffer[y]
    return "".join(buf[x].data for x in range(screen.columns))


def scan_ttyrec(path: str, engine: str = "hybrid") -> dict:
    """Endgame evidence + status maxima for one session ttyrec.

    engine="hybrid" (default): read the maxima straight out of the byte stream
    and replay with pyte only the tail of sessions that actually contain an
    endgame literal.  Full replay of every frame of every session costs ~25 CPU
    seconds per file (~5 h for this selection); the hybrid is ~100x cheaper and
    produces the same flags.  engine="pyte" forces the exhaustive path and is
    kept so the two can be diffed (see --verify-engine).
    """
    if engine == "pyte":
        return _scan_ttyrec_pyte(path)
    try:
        data = bz2.open(path, "rb").read() if path.endswith(".bz2") else open(path, "rb").read()
    except Exception as e:
        return {"file": path, "error": repr(e), "agg": _new_agg(), "planes": {},
                "asc": {}, "offer": None, "role": None, "n_frames": 0, "mode": "error"}

    m_role = WELCOME_RE.search(data.decode("latin-1", "replace"))
    out = {"file": path, "agg": _new_agg(), "planes": {}, "asc": {}, "offer": None,
           "role": m_role.group(1) if m_role else None, "mode": "raw"}

    frames = list(_frames(data))
    out["n_frames"] = len(frames)
    agg = out["agg"]
    # Regex the *payloads only*, NUL-joined. Scanning the raw file would splice
    # a digit out of the next frame's 12-byte binary header onto the end of a
    # number that sits at a frame boundary ("T:28805" + header byte -> 288058),
    # and NUL-joining also stops one frame's trailing digits from fusing with
    # the next frame's leading ones. A number truncated by a frame split can
    # only come out smaller, so it can never inflate a maximum.
    body = b"\x00".join(frames)
    _pick = lambda ms: [int(next(g for g in m if g)) for m in ms]  # noqa: E731
    dl = _pick(RAW_DLVL.findall(body))
    xp = _pick(RAW_XP.findall(body))
    tt = _pick(RAW_T.findall(body))
    agg["n"] = len(tt)
    agg["maxDl"] = max(dl) if dl else 0
    agg["maxXp"] = max(xp) if xp else 0
    agg["minXp"] = min(xp) if xp else None
    if tt:
        agg["minT"], agg["maxT"] = min(tt), max(tt)

    first = next((i for i, f in enumerate(frames)
                  if any(b in f for b in EVIDENCE_BYTES)), None)
    if first is None:
        return out  # no plane, no ascension, nothing to render

    out["mode"] = "hybrid"
    tail = frames[max(0, first - PRE_FRAMES):]
    for cols, nrows, rows, try_curses in PROFILES:
        screen = pyte.Screen(cols, nrows)
        stream = pyte.ByteStream(screen)
        planes, prev_st, prev_msg, last_turn, hit = {}, None, None, None, False
        out["asc"], out["offer"] = {}, None
        for f in tail:
            try:
                stream.feed(f)
            except Exception:
                continue
            stext = "\n".join(_row(screen, y) for y in rows)
            if stext != prev_st:
                prev_st = stext
                st = parse_status_block(stext.split("\n"), cols)
                if st is None and try_curses:
                    st = parse_curses_panel(stext)
                if st is not None:
                    hit = True
                    if st["turns"] is not None:
                        last_turn = st["turns"]
                    if st["kind"] == "endgame" and st["plane"] not in planes:
                        planes[st["plane"]] = {"turn": st["turns"], "file": path,
                                               "status_line": st["raw"][:78]}
            mtext = "\n".join(_row(screen, y) for y in MSG_ROWS if y < nrows)
            if mtext != prev_msg:
                prev_msg = mtext
                scan_messages(mtext.split("\n"), last_turn, out, path)
        out["planes"] = planes
        if hit:
            out["mode"] = f"hybrid-{cols}x{nrows}"
            break
        out["mode"] = "hybrid-nostatus"
    return out


def _scan_ttyrec_pyte(path: str) -> dict:
    """Endgame evidence + status maxima, by replaying every frame (reference)."""
    try:
        data = bz2.open(path, "rb").read() if path.endswith(".bz2") else open(path, "rb").read()
    except Exception as e:
        return {"file": path, "error": repr(e), "agg": _new_agg(), "planes": {},
                "asc": {}, "offer": None, "role": None, "n_frames": 0, "mode": "error"}

    want_msg = any(b in data for b in ASC_BYTES) or any(b in data for b in OFFER_BYTES)
    m_role = WELCOME_RE.search(data.decode("latin-1", "replace"))
    out = {"file": path, "agg": _new_agg(), "planes": {}, "asc": {}, "offer": None,
           "role": m_role.group(1) if m_role else None}

    def run(cols, nrows, status_rows, try_curses):
        screen = pyte.Screen(cols, nrows)
        stream = pyte.ByteStream(screen)
        agg, planes, prev_st, prev_msg = _new_agg(), {}, None, None
        last_turn, n_frames, last_s = None, 0, None
        for payload in _frames(data):
            n_frames += 1
            try:
                stream.feed(payload)
            except Exception:
                continue
            stext = "\n".join(_row(screen, y) for y in status_rows)
            if stext != prev_st:
                prev_st = stext
                st = parse_status_block(stext.split("\n"), cols)
                if st is None and try_curses:
                    st = parse_curses_panel(stext)
                if st is not None:
                    if st["turns"] is not None:
                        last_turn = st["turns"]
                        agg["maxT"] = max(agg["maxT"] or 0, st["turns"])
                    # experience is also earned on the Planes, where the level
                    # field is not "Dlvl:N" -- count it whatever the level field
                    if st["xp_level"] is not None:
                        agg["maxXp"] = max(agg["maxXp"], st["xp_level"])
                    if st["kind"] == "endgame" and st["plane"] not in planes:
                        planes[st["plane"]] = {
                            "turn": st["turns"], "file": path,
                            "status_line": st["raw"][:78],
                        }
                    if st["kind"] == "dlvl" and st["turns"] is not None:
                        s = (st["turns"], st["depth"], st["xp_level"])
                        if s != last_s:
                            last_s = s
                            _agg_add(agg, *s)
            if want_msg:
                mtext = "\n".join(_row(screen, y) for y in MSG_ROWS if y < nrows)
                if mtext != prev_msg:
                    prev_msg = mtext
                    scan_messages(mtext.split("\n"), last_turn, out, path)
        return agg, planes, n_frames

    mode = "none"
    agg, planes, n_frames = _new_agg(), {}, 0
    for cols, nrows, rows, try_curses in PROFILES:
        out["asc"], out["offer"] = {}, None
        agg, planes, n_frames = run(cols, nrows, rows, try_curses)
        if agg["n"] or planes:
            mode = f"pyte-{cols}x{nrows}"
            break
    out.update(agg=agg, planes=planes, n_frames=n_frames, mode=mode)
    return out


def _nao_worker(args):
    key, path, engine = args
    try:
        return key, scan_ttyrec(path, engine)
    except Exception as e:  # keep the shard moving; failures are counted
        return key, {"file": path, "error": repr(e), "agg": _new_agg(), "planes": {},
                     "asc": {}, "offer": None, "role": None, "n_frames": 0,
                     "mode": "error"}


def select_nao_games(n_control: int) -> list[dict]:
    players = {p.name for p in NAO_ROOT.iterdir() if p.is_dir()}
    idx = index_games(players)
    for name, games in idx.items():
        assign_files(games, sorted((NAO_ROOT / name).glob("*.ttyrec*")))
    allg = [g for v in idx.values() for g in v if g["files"]]

    asc = [g for g in allg if g["death"].startswith("ascended")]
    rest = [g for g in allg if not g["death"].startswith("ascended")]
    # controls: every deep non-ascension, plus the two "escaped (in celestial
    # disgrace)" games -- those DID stand on the Astral Plane and did NOT win,
    # which is the single sharpest false-positive test available.
    ctrl = {id(g): g for g in rest if (g["maxlvl"] or 0) >= 45}
    for g in rest:
        if "celestial disgrace" in g["death"]:
            ctrl[id(g)] = g
    if len(ctrl) < n_control:  # top up with the deepest remaining games
        for g in sorted(rest, key=lambda g: -(g["maxlvl"] or 0)):
            if id(g) not in ctrl:
                ctrl[id(g)] = g
            if len(ctrl) >= n_control:
                break
    ctrl = list(ctrl.values())[:n_control]
    for g in asc:
        g["_set"] = "ascension"
    for g in ctrl:
        g["_set"] = "control"
    print(f"[nao] {len(allg)} xlog games with session files; selected "
          f"{len(asc)} ascensions + {len(ctrl)} controls "
          f"(maxlvl>=45: {sum(1 for g in ctrl if (g['maxlvl'] or 0) >= 45)}, "
          f"celestial-disgrace: {sum(1 for g in ctrl if 'celestial' in g['death'])})",
          flush=True)
    return asc + ctrl


def run_nao(n_control: int, procs: int, engine: str = "hybrid",
            games: list[dict] | None = None) -> list[dict]:
    games = games if games is not None else select_nao_games(n_control)
    jobs = []
    for g in games:
        key = f"{g['player']}/{g['start']}"
        g["key"] = key
        for f in g["files"]:
            jobs.append((key, f, engine))
    nbytes = sum(os.path.getsize(j[1]) for j in jobs)
    print(f"[nao] decoding {len(jobs)} session files ({nbytes/1e9:.2f} GB compressed) "
          f"on {procs} procs, engine={engine}", flush=True)

    per_key: dict[str, list[dict]] = collections.defaultdict(list)
    t0, done = time.time(), 0
    with mp.Pool(procs) as pool:
        for key, res in pool.imap_unordered(_nao_worker, jobs, chunksize=1):
            per_key[key].append(res)
            done += 1
            if done % 250 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[nao] {done}/{len(jobs)} files  {el:.0f}s  "
                      f"eta {el/done*(len(jobs)-done):.0f}s", flush=True)

    recs = []
    for g in games:
        parts = sorted(per_key.get(g["key"], []), key=lambda r: r["file"])
        recs.append(_assemble(
            corpus="nao", player=g["player"], key=g["key"], parts=parts,
            role=g.get("role"),
            xlog={"maxlvl": g["maxlvl"], "turns": g["turns"], "death": g["death"],
                  "points": g["points"], "role": g["role"], "version": g["version"],
                  "start": g["start"], "end": g["end"], "set": g["_set"],
                  "xlog_ascended": g["death"].startswith("ascended")},
        ))
    return recs


# -------------------------------------------------------------- top10 --------

def _changed_rows(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] == 0:
        return np.zeros(0, dtype=int)
    if arr.shape[0] == 1:
        return np.zeros(1, dtype=int)
    ch = np.any(arr[1:] != arr[:-1], axis=(1, 2))
    return np.concatenate(([0], np.nonzero(ch)[0] + 1))


def _rows_text(tc: np.ndarray, i: int, rows) -> list[str]:
    return [bytes(tc[i][r]).decode("ascii", "replace") for r in rows]


def scan_npz(path: str) -> dict:
    """Same evidence extraction over an already-decoded (T,24,80) tensor."""
    out = {"file": path, "id": pathlib.Path(path).stem, "agg": _new_agg(),
           "planes": {}, "asc": {}, "offer": None, "role": None, "n_frames": 0}
    try:
        tc = np.load(path)["tty_chars"]
    except Exception as e:
        out["error"] = repr(e)
        return out
    if tc.ndim != 3 or tc.shape[0] == 0:
        return out
    out["n_frames"] = int(tc.shape[0])

    last_turn_at: dict[int, int] = {}
    for rows, try_curses in (((22, 23), False), (tuple(range(23, -1, -1)), True)):
        agg, planes, last_turn_at, last_s = _new_agg(), {}, {}, None
        for i in _changed_rows(tc[:, list(rows), :]):
            lines = _rows_text(tc, int(i), rows)
            st = parse_status_block(lines)
            if st is None and try_curses:
                st = parse_curses_panel("\n".join(lines))
            if st is None:
                continue
            if st["turns"] is not None:
                last_turn_at[int(i)] = st["turns"]
            if st["kind"] == "endgame" and st["plane"] not in planes:
                planes[st["plane"]] = {"turn": st["turns"], "file": path,
                                       "status_line": st["raw"][:78]}
            if st["kind"] == "dlvl" and st["turns"] is not None:
                mx = XP_RE.search(st["raw"])
                xpp = int(mx.group(2)) if (mx and mx.group(2)) else 0
                s = (st["turns"], st["depth"], st["xp_level"], xpp)
                if s != last_s:
                    last_s = s
                    _agg_add(agg, *s)
        if agg["n"] or planes:
            break
    out["agg"], out["planes"] = agg, planes

    # messages: rows 0-2, prefiltered by a whole-tensor literal byte search
    msg = tc[:, 0:3, :]
    raw = msg.tobytes()
    if any(b in raw for b in ASC_BYTES) or any(b in raw for b in OFFER_BYTES):
        turns = sorted(last_turn_at.items())
        for i in _changed_rows(msg):
            i = int(i)
            t = None
            for j, tv in turns:
                if j <= i:
                    t = tv
                else:
                    break
            scan_messages(_rows_text(tc, i, (0, 1, 2)), t, out, path)
    if out["role"] is None:
        head = tc[: min(400, tc.shape[0]), 0:3, :].tobytes().decode("latin-1", "replace")
        m = WELCOME_RE.search(head)
        if m:
            out["role"] = m.group(1)
    return out


def _top10_worker(path):
    try:
        return scan_npz(path)
    except Exception as e:
        return {"file": path, "id": pathlib.Path(path).stem, "error": repr(e),
                "agg": _new_agg(), "planes": {}, "asc": {}, "offer": None,
                "role": None, "n_frames": 0}


def run_top10(procs: int, users=None) -> list[dict]:
    users = users or sorted(d.name for d in TOP10_ROOT.iterdir() if d.is_dir())
    recs = []
    for u in users:
        files = sorted(str(f) for f in (TOP10_ROOT / u).glob("*.npz"))
        t0 = time.time()
        with mp.Pool(procs) as pool:
            res = pool.map(_top10_worker, files, chunksize=1)
        sessions, orphans = [], []
        for r in res:
            a = r["agg"]
            if not a["n"]:
                # a session with no numeric-Dlvl frame cannot be chained on turn
                # continuity; keep it only if it carries endgame evidence, as its
                # own singleton "game" (an endgame-only tail session)
                if r["planes"] or r["asc"]:
                    orphans.append(r)
                continue
            sessions.append({
                "id": r["id"], "_r": r,
                "minT": a["minT"], "maxT": a["maxT"],
                "minXP": a["minXp"] or 0, "maxXP": a["maxXp"],
                "minXPP": a["minXPP"], "maxXPP": a["maxXPP"],
            })
        chained = chain(sessions)
        chained += [[{"id": r["id"], "_r": r, "minT": None}] for r in orphans]
        for game in chained:
            parts = [s["_r"] for s in game]
            recs.append(_assemble(
                corpus="top10", player=u, key=f"{u}/{game[0]['id']}", parts=parts,
                role=next((p["role"] for p in parts if p["role"]), None),
                xlog=None, n_sessions=len(game),
                complete=game[0]["minT"] is not None and game[0]["minT"] <= 5,
            ))
        n_pl = sum(1 for r in recs if r["player"] == u and r["reached_planes"])
        n_as = sum(1 for r in recs if r["player"] == u and r["ascended"])
        print(f"[top10] {u:12s} files={len(files):5d} games={len(chained):4d} "
              f"planes={n_pl:3d} ascended={n_as:3d}  {time.time()-t0:.0f}s", flush=True)
    return recs


# ------------------------------------------------------------- assembly ------

def _assemble(*, corpus, player, key, parts, role, xlog, **extra) -> dict:
    max_dl = max_xp = final_turn = 0
    n_status = 0
    for p in parts:
        a = p["agg"]
        n_status += a["n"]
        max_dl = max(max_dl, a["maxDl"])
        max_xp = max(max_xp, a["maxXp"])
        if a["maxT"] is not None:
            final_turn = max(final_turn, a["maxT"])

    planes: dict[str, dict] = {}
    for p in parts:
        for name, ev in p["planes"].items():
            cur = planes.get(name)
            if cur is None or (ev["turn"] is not None and
                               (cur["turn"] is None or ev["turn"] < cur["turn"])):
                planes[name] = ev
    asc: dict[str, dict] = {}
    for p in parts:
        for name, ev in p["asc"].items():
            if name not in asc or (ev["turn"] or 0) < (asc[name]["turn"] or 0):
                asc[name] = ev
    offer = None
    for p in parts:
        if p["offer"] and (offer is None or (p["offer"]["turn"] or 0) < (offer["turn"] or 0)):
            offer = p["offer"]

    plane_turns = [ev["turn"] for ev in planes.values() if ev["turn"] is not None]
    asc_turns = [ev["turn"] for ev in asc.values() if ev["turn"] is not None]
    for t in plane_turns + asc_turns:
        final_turn = max(final_turn, t)

    rec = {
        "corpus": corpus, "player": player, "game_key": key,
        "role": (xlog or {}).get("role") or role,
        "n_files": len(parts),
        "n_frames": sum(p.get("n_frames", 0) for p in parts),
        "n_status_states": n_status,
        "errors": [p["error"] for p in parts if p.get("error")],
        "reached_planes": bool(planes),
        "planes_seen": sorted(planes),
        "reached_astral": "Astral" in planes,
        "first_plane_turn": min(plane_turns) if plane_turns else None,
        "plane_evidence": planes,
        "ascended": bool(asc),
        "ascension_markers": sorted(asc),
        "ascension_turn": min(asc_turns) if asc_turns else None,
        "ascension_evidence": list(asc.values()),
        "offered_amulet": offer is not None,
        "offer_evidence": offer,
        "max_dlvl": max_dl, "max_xp": max_xp, "final_turn": final_turn,
    }
    rec.update(extra)
    if xlog:
        rec["xlog"] = xlog
    return rec


# ------------------------------------------------------------- reporting -----

def _spread(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1)))]  # noqa: E731
    return {"n": len(v), "min": v[0], "p25": q(.25), "median": statistics.median(v),
            "p75": q(.75), "max": v[-1]}


def report(recs: list[dict]) -> str:
    L = []
    w = L.append
    w("# Endgame / ascension detection in human NetHack recordings\n")
    w(f"Generated by `endgame_parse.py`; per-game records in `{OUT_JSON}`.\n")

    for corpus in ("nao", "top10"):
        sub = [r for r in recs if r["corpus"] == corpus]
        if not sub:
            continue
        pl = [r for r in sub if r["reached_planes"]]
        asc = [r for r in sub if r["ascended"]]
        w(f"\n## {corpus}  ({len(sub)} games scanned)\n")
        w(f"- reached the Planes: **{len(pl)}**  (Astral specifically: "
          f"{sum(1 for r in pl if r['reached_astral'])})")
        w(f"- ascended: **{len(asc)}**")
        w(f"- offered the Amulet (not proof of a win): {sum(1 for r in sub if r['offered_amulet'])}")
        sp = _spread([r["first_plane_turn"] for r in pl])
        if sp:
            w(f"- turn of first Plane: median **{sp['median']:.0f}**, "
              f"IQR {sp['p25']}-{sp['p75']}, range {sp['min']}-{sp['max']} (n={sp['n']})")
        sa = _spread([r["ascension_turn"] for r in asc])
        if sa:
            w(f"- turn of ascension: median **{sa['median']:.0f}**, "
              f"IQR {sa['p25']}-{sa['p75']}, range {sa['min']}-{sa['max']} (n={sa['n']})")
        mk = collections.Counter(m for r in asc for m in r["ascension_markers"])
        if mk:
            w(f"- marker hit counts: {dict(mk)}")
        pls = collections.Counter(p for r in pl for p in r["planes_seen"])
        if pls:
            w(f"- plane fields seen: {dict(pls)}")

    # ---- validation against the xlogfile (NLD-NAO only) ----
    nao = [r for r in recs if r["corpus"] == "nao"]
    w("\n## Validation against the NLD-NAO xlogfile\n")
    if not nao:
        w("_no NAO games scanned_")
        return "\n".join(L)
    truth = [r for r in nao if r["xlog"]["xlog_ascended"]]
    tp = [r for r in truth if r["ascended"]]
    fn = [r for r in truth if not r["ascended"]]
    fp = [r for r in nao if r["ascended"] and not r["xlog"]["xlog_ascended"]]
    prec = len(tp) / max(1, len(tp) + len(fp))
    rec_ = len(tp) / max(1, len(truth))
    w(f"| | count |\n|---|---|")
    w(f"| xlog `death=ascended` games scanned | {len(truth)} |")
    w(f"| true positives (parser agrees) | {len(tp)} |")
    w(f"| false negatives (xlog says win, parser missed) | {len(fn)} |")
    w(f"| false positives (parser says win, xlog disagrees) | {len(fp)} |")
    w(f"| **precision** | **{prec:.4f}** |")
    w(f"| **recall** | **{rec_:.4f}** |")
    if fn:
        w("\nFalse negatives:\n")
        for r in fn[:25]:
            w(f"- `{r['game_key']}` role={r['role']} files={r['n_files']} "
              f"planes={r['planes_seen']} maxdlvl={r['max_dlvl']} "
              f"turns={r['final_turn']} (xlog turns={r['xlog']['turns']}) "
              f"errors={len(r['errors'])}")
    if fp:
        w("\nFalse positives:\n")
        for r in fp[:25]:
            w(f"- `{r['game_key']}` xlog death=`{r['xlog']['death']}` "
              f"markers={r['ascension_markers']} evidence={r['ascension_evidence'][:1]}")

    # ---- how accurate are the derived numbers, against xlog ground truth? ----
    def _agree(pairs):
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            return "n/a"
        ex = sum(1 for a, b in pairs if a == b)
        w5 = sum(1 for a, b in pairs if abs(a - b) <= 5)
        return (f"{ex}/{len(pairs)} exact ({ex/len(pairs):.1%}), "
                f"{w5}/{len(pairs)} within 5")

    w("\n### Derived numbers vs the xlogfile\n")
    w(f"- `ascension_turn` vs xlog `turns`: {_agree([(r['ascension_turn'], r['xlog']['turns']) for r in tp])}")
    w(f"- `max_dlvl` vs xlog `maxlvl`: {_agree([(r['max_dlvl'], r['xlog']['maxlvl']) for r in nao])}")
    w(f"- `final_turn` vs xlog `turns`: {_agree([(r['final_turn'], r['xlog']['turns']) for r in nao])}")
    w(f"- wins with no readable turn on the ascension frame: "
      f"{sum(1 for r in tp if r['ascension_turn'] is None)}")

    # the sharpest false-positive test in the corpus
    off_only = [r for r in nao if r["offered_amulet"] and not r["ascended"]]
    w(f"\n**Amulet offered but no win detected**: {len(off_only)} game(s) -- these are "
      f"the players who offered the Amulet to the *wrong* deity on the Astral Plane, "
      f"which NetHack ends as `escaped`, not `ascended`. Getting these right is the "
      f"reason the offering message is recorded but never counted as proof:")
    for r in off_only:
        w(f"- `{r['game_key']}` xlog death=`{r['xlog']['death']}` "
          f"planes={r['planes_seen']} offered at T={r['offer_evidence']['turn']}")

    # containment: every ascension must have reached the planes
    bad = [r for r in nao if r["ascended"] and not r["reached_planes"]]
    w(f"\n**Containment check** (`ascended` ⊆ `reached_planes`): "
      f"{len(bad)} violation(s) out of {sum(1 for r in nao if r['ascended'])} ascensions.")
    for r in bad[:15]:
        w(f"- `{r['game_key']}` markers={r['ascension_markers']} files={r['n_files']} "
          f"maxdlvl={r['max_dlvl']}")

    ctrl = [r for r in nao if r["xlog"]["set"] == "control"]
    cpl = [r for r in ctrl if r["reached_planes"]]
    w(f"\n**Control set** ({len(ctrl)} deep non-ascension games): "
      f"{len(cpl)} reached the Planes, {sum(1 for r in ctrl if r['ascended'])} "
      f"flagged as ascended.")
    for r in cpl[:15]:
        w(f"- `{r['game_key']}` xlog death=`{r['xlog']['death'][:48]}` maxlvl="
          f"{r['xlog']['maxlvl']} planes={r['planes_seen']} @T={r['first_plane_turn']}")

    # planes recall proxy: xlog-ascended games where we saw no plane at all
    nopl = [r for r in truth if not r["reached_planes"]]
    w(f"\n**Planes recall on known wins**: {len(truth)-len(nopl)}/{len(truth)} "
      f"xlog-ascended games show a Plane status line "
      f"({(len(truth)-len(nopl))/max(1,len(truth)):.1%}).")

    derr = [r for r in nao if r["errors"]]
    nodata = [r for r in nao if not r["planes_seen"] and r["max_dlvl"] == 0]
    w(f"\nDecode health: {len(derr)} games with >=1 file-level error, "
      f"{len(nodata)} games yielded no status line at all "
      f"(oversized terminal / unreadable recording).")
    return "\n".join(L)


def _verify_engine(games, recs, n, procs):
    """Re-decode n games frame-by-frame with pyte and diff against the hybrid.

    The hybrid skips replaying sessions with no endgame literal and reads the
    maxima from raw bytes; this proves the two agree on the flags that matter.
    """
    bykey = {r["game_key"]: r for r in recs if r["corpus"] == "nao"}
    asc = [g for g in games if g["_set"] == "ascension"]
    ctl = [g for g in games if g["_set"] == "control"]
    asc.sort(key=lambda g: sum(os.path.getsize(f) for f in g["files"]))
    ctl.sort(key=lambda g: sum(os.path.getsize(f) for f in g["files"]))
    pick = asc[: n // 2] + ctl[: n - n // 2]
    jobs = [(g["key"], f, "pyte") for g in pick for f in g["files"]]
    print(f"\n[verify] re-decoding {len(pick)} games / {len(jobs)} files with the "
          f"exhaustive pyte engine", flush=True)
    per = collections.defaultdict(list)
    with mp.Pool(procs) as pool:
        for key, res in pool.imap_unordered(_nao_worker, jobs, chunksize=1):
            per[key].append(res)
    diffs = []
    for g in pick:
        ref = _assemble(corpus="nao", player=g["player"], key=g["key"],
                        parts=sorted(per[g["key"]], key=lambda r: r["file"]),
                        role=g.get("role"), xlog=None)
        h = bykey[g["key"]]
        for field in ("ascended", "reached_planes", "planes_seen",
                      "ascension_turn", "first_plane_turn", "max_dlvl", "max_xp"):
            if ref[field] != h[field]:
                diffs.append((g["key"], field, h[field], ref[field]))
    print(f"[verify] {len(pick)} games compared, {len(diffs)} field disagreements")
    for k, f, a, b in diffs[:30]:
        print(f"   {k} {f}: hybrid={a!r} pyte={b!r}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["nao", "top10", "both"], default="both")
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--control", type=int, default=200)
    ap.add_argument("--users", nargs="*", default=None)
    ap.add_argument("--engine", choices=["hybrid", "pyte"], default="hybrid")
    ap.add_argument("--verify-engine", type=int, default=0,
                    help="also re-run this many NAO games under the exhaustive "
                         "pyte engine and diff the flags")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the summary from an existing states JSON")
    args = ap.parse_args()

    if args.report_only:
        recs = json.load(open(args.out))
        md = report(recs)
        OUT_MD.write_text(md + "\n")
        print(md)
        return

    recs = []
    if args.corpus in ("nao", "both"):
        games = select_nao_games(args.control)
        recs += run_nao(args.control, args.procs, args.engine, games)
        pathlib.Path(args.out).write_text(json.dumps(recs))
        if args.verify_engine:
            _verify_engine(games, recs, args.verify_engine, args.procs)
    if args.corpus in ("top10", "both"):
        recs += run_top10(args.procs, args.users)

    pathlib.Path(args.out).write_text(json.dumps(recs))
    md = report(recs)
    OUT_MD.write_text(md + "\n")
    print("\n" + md)
    print(f"\nwrote {args.out} ({len(recs)} game records) and {OUT_MD}")


if __name__ == "__main__":
    main()
