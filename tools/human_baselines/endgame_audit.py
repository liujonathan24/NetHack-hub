"""Audit every piece of endgame evidence in endgame_states.json by hand-eye.

The detector's whole claim is "this status line / this message says so". This
prints the distinct raw strings behind every `reached_planes` and `ascended`
flag, so the flags can be checked rather than trusted.
"""
import collections
import json
import re
import sys

recs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/root/nld/endgame_states.json"))

# Does every plane-hit status line look like a real status line, i.e. is the
# level field followed by the gold field ("$:0")?
TIGHT = re.compile(r"^ {0,3}(?:Astral Plane|End Game|Plane of (?:Earth|Air|Fire|Water)"
                   r"|Earth|Air|Fire|Water)\s+\S:-?\d")

for corpus in ("nao", "top10"):
    sub = [r for r in recs if r["corpus"] == corpus]
    if not sub:
        continue
    print(f"\n{'='*78}\n{corpus}: {len(sub)} games\n{'='*78}")

    lines = collections.Counter()
    loose = []
    for r in sub:
        for name, ev in r["plane_evidence"].items():
            sl = ev["status_line"]
            lines[re.sub(r"\d+", "N", sl)] += 1
            if not TIGHT.match(sl):
                loose.append((r["game_key"], name, sl))
    print(f"\n-- distinct plane status-line shapes (digits -> N), top 15 of {len(lines)}")
    for s, n in lines.most_common(15):
        print(f"{n:6d}  {s}")
    print(f"\n-- plane hits whose line is NOT <levelfield> <gold>: {len(loose)}")
    for k, n, sl in loose[:20]:
        print(f"   {k} {n}: {sl!r}")

    msgs = collections.Counter()
    for r in sub:
        for ev in r["ascension_evidence"]:
            msgs[(ev["marker"], re.sub(r"\d+", "N", ev["text"]))] += 1
    print(f"\n-- distinct ascension message texts: {len(msgs)}")
    for (m, s), n in msgs.most_common(20):
        print(f"{n:6d}  [{m}] {s}")

    mk = collections.Counter(tuple(sorted(r["ascension_markers"])) for r in sub if r["ascended"])
    print("\n-- marker combinations on flagged wins")
    for combo, n in mk.most_common():
        print(f"{n:6d}  {combo}")

    # ordering sanity: planes before ascension, ascension at/near the last turn
    bad_order = [r for r in sub if r["ascended"] and r["first_plane_turn"] is not None
                 and r["ascension_turn"] is not None
                 and r["ascension_turn"] < r["first_plane_turn"]]
    print(f"\n-- wins whose ascension turn precedes their first plane turn: {len(bad_order)}")
    for r in bad_order[:10]:
        print("   ", r["game_key"], r["first_plane_turn"], r["ascension_turn"])

    if corpus == "nao":
        off = [r for r in sub if r["offered_amulet"] and not r["ascended"]]
        print(f"\n-- offered the Amulet but did NOT ascend: {len(off)}")
        for r in off[:20]:
            print(f"   {r['game_key']} death={r['xlog']['death'][:46]!r} "
                  f"planes={r['planes_seen']} offerT={r['offer_evidence']['turn']}")
