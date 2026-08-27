"""Diagnose the ttyrec <-> xlogfile join: clock offset and game boundaries.

Two hypotheses for the mismatches:
  H1  ttyrec filenames are in NAO server-local time, not UTC -> fixed offset.
  H2  a ttyrec is a *login session*, not a game: it can hold several games
      (or only the tail of one resumed via save), so max-Dlvl over the whole
      file need not equal any single game's xlog `maxlvl`.
"""
import bz2
import collections
import datetime as dt
import json
import pathlib
import re
import struct
import sys

FNAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.(\d{2}):(\d{2}):(\d{2})")
recs = json.loads(pathlib.Path("/root/nld/sample_study.json").read_text())
recs = [r for r in recs if "start_utc" in r]
by_player = collections.defaultdict(list)
for r in recs:
    by_player[r["player"]].append(r)

# --- H1: nearest xlog starttime per sampled file, whatever the offset ---
best = {}
xlog = pathlib.Path("/root/nld/nld-nao/xlogfile.full.txt")
with open(xlog, errors="replace") as fh:
    for line in fh:
        sep = "\t" if "\tname=" in line else (":" if ":name=" in line else None)
        if not sep:
            continue
        f = dict(kv.split("=", 1) for kv in line.rstrip("\n").split(sep) if "=" in kv)
        name = f.get("name")
        if name not in by_player:
            continue
        st = f.get("starttime")
        if not (st and st.isdigit()):
            continue
        st = int(st)
        for r in by_player[name]:
            d = st - r["start_utc"]
            if abs(d) < abs(best.get(r["file"], (10**9, None))[0]):
                best[r["file"]] = (d, f)

deltas = collections.Counter()
for fn, (d, f) in best.items():
    deltas[round(d / 3600)] += 1
print("nearest-xlog delta in hours -> count:", dict(sorted(deltas.items())[:20]))
print("within 3 min:", sum(1 for d, _ in best.values() if abs(d) <= 180))

# --- H2: how many games does one ttyrec contain? ---
SPLASH = b"NetHack, Copyright"
DYWYPI = b"Do you want your possessions identified?"
counts = collections.Counter()
multi = []
for r in recs[:120]:
    data = bz2.open(r["file"], "rb").read() if r["file"].endswith(".bz2") else open(r["file"], "rb").read()
    payload = bytearray()
    off = 0
    while off + 12 <= len(data):
        sec, usec, ln = struct.unpack("<III", data[off : off + 12])
        off += 12
        if ln > 1 << 22 or off + ln > len(data):
            break
        payload += data[off : off + ln]
        off += ln
    n_splash = payload.count(SPLASH)
    n_dywypi = payload.count(DYWYPI)
    counts[(n_splash, n_dywypi)] += 1
    if n_splash > 1:
        multi.append((r["player"], pathlib.Path(r["file"]).name, n_splash, n_dywypi,
                      r["max_dlvl"], (r.get("xlog") or {}).get("maxlvl")))
print("\n(splash_count, dywypi_count) -> n_files:", dict(sorted(counts.items())))
print("files with >1 game (splash>1):", len(multi))
for m in multi[:15]:
    print("  ", m)
