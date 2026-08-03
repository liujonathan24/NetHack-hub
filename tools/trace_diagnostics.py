#!/usr/bin/env python
"""What is the observation over-supplying, and what is it failing to supply?

Reads a cell's per-turn NDJSON and reports the evidence a variant should be
designed FROM, rather than guessed at:

  A. WASTE      -- per skill, how often a call did not advance the game clock.
                   Zero-clock calls are the harness spending the budget on
                   nothing; a prior audit put 34% of a rollout in this bucket.
  B. REASONS    -- what the game said back on those wasted calls. This is the
                   single most actionable signal: "It's a wall" means the agent
                   acted on terrain it misread, "Never mind." means a tool
                   accepted an argument it could not use.
  C. THRASH     -- identical (name, arguments) repeated. Distinguishes a genuine
                   loop from a model varying coordinates while making no
                   progress -- which look identical in a waste rate.
  D. BAD ARGS   -- calls carrying a null/missing required argument. These are
                   pure harness cost: the tool cannot succeed, and the agent
                   cannot see why.
  E. SECTIONS   -- what the observation actually spends its bytes on, per turn.
                   A section that is large every turn and never acted on is a
                   candidate for removal; one that is absent when a call fails
                   is a candidate for addition.
  F. UNUSED     -- coordinates published in VISIBLE FEATURES that no positional
                   call ever targeted. High unused share means the block is
                   costing tokens without steering behaviour.

Schema-tolerant: prefers v2+ `tool_results` (which carry `clock_advanced`
directly) and falls back to differencing `status.time` across turns for v0/v1
traces, so old and new cells are comparable.

    python -m tools.trace_diagnostics <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

_SECTION = re.compile(r"^=== ([A-Z][A-Z /_-]*[A-Z]) ===", re.M)
_COORD = re.compile(r"\((\d{1,2}),\s*(\d{1,2})\)")


def _rows(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed rollout can leave a torn final line; the rest is good.
                break
    return out


def _calls(rows):
    """Yield (name, args, advanced, message) per executed call.

    v2+ records carry the outcome explicitly. For older records the clock is the
    only ground truth available, so the NEXT turn's `status.time` decides whether
    this turn's call did anything.
    """
    for i, r in enumerate(rows):
        results = r.get("tool_results") or []
        if results:
            for tr in results:
                yield (
                    tr.get("name"),
                    tr.get("arguments"),
                    tr.get("clock_advanced"),
                    tr.get("game_message") or "",
                )
            continue
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        t0 = (r.get("status") or {}).get("time")
        t1 = (nxt.get("status") or {}).get("time") if nxt else None
        adv = None if (t0 is None or t1 is None) else t1 > t0
        msg = ((nxt or {}).get("messages") or [""])[0] if nxt else ""
        for tc in r.get("tool_calls") or []:
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            yield tc.get("name"), args, adv, msg


def _obs_text(r):
    return r.get("rendered_user_message") or r.get("rendered_user_content") or ""


def analyse(run_dir):
    turn_files = []
    for base, _dirs, files in os.walk(run_dir):
        if os.path.basename(base) != "turns":
            continue
        turn_files += [os.path.join(base, f) for f in files if f.endswith(".ndjson")]
    if not turn_files:
        print(f"  (no turn files under {run_dir})")
        return

    calls = Counter()
    waste = Counter()
    reasons = Counter()
    thrash = Counter()
    badargs = Counter()
    sect_bytes = Counter()
    sect_turns = Counter()
    n_turns = 0
    published = set()
    targeted = set()

    for f in sorted(turn_files):
        rows = _rows(f)
        n_turns += len(rows)
        seen = Counter()
        for name, args, adv, msg in _calls(rows):
            if not name:
                continue
            calls[name] += 1
            if adv is False:
                waste[name] += 1
                if msg:
                    reasons[msg.strip()[:72]] += 1
            seen[(name, json.dumps(args, sort_keys=True, default=str))] += 1
            if isinstance(args, dict):
                if args and all(v is None for v in args.values()):
                    badargs[name] += 1
                x, y = args.get("x"), args.get("y")
                if isinstance(x, int) and isinstance(y, int):
                    targeted.add((x, y))
        for sig, c in seen.items():
            if c > 1:
                thrash[sig] += c - 1

        for r in rows:
            text = _obs_text(r)
            if not text:
                continue
            marks = list(_SECTION.finditer(text))
            for j, m in enumerate(marks):
                end = marks[j + 1].start() if j + 1 < len(marks) else len(text)
                nm = m.group(1).strip()
                sect_bytes[nm] += end - m.start()
                sect_turns[nm] += 1
            fb = text.find("=== VISIBLE FEATURES ===")
            if fb != -1:
                # Start AFTER this header's own trailing `===`, otherwise the
                # very next `===` found is the closing marker of the header
                # itself and the segment collapses to nothing. The block's
                # content also begins on the SAME line as the header
                # ("=== VISIBLE FEATURES === stairs DOWN at (57,13); ..."), so
                # the segment runs to the next header, not the next newline.
                body = fb + len("=== VISIBLE FEATURES ===")
                nxt = text.find("\n===", body)
                for mx, my in _COORD.findall(text[body : nxt if nxt > 0 else len(text)]):
                    published.add((int(mx), int(my)))

    total = sum(calls.values())
    tw = sum(waste.values())
    print(f"\n{'='*78}\n{run_dir}   turns={n_turns}  calls={total}  "
          f"wasted={tw} ({100*tw/max(1,total):.0f}%)\n{'='*78}")

    print("\nA. WASTE BY SKILL")
    print(f"   {'skill':<24}{'calls':>7}{'wasted':>8}{'waste%':>8}   share of all waste")
    for name, c in calls.most_common(10):
        w = waste[name]
        bar = "#" * int(round(20 * w / max(1, tw)))
        print(f"   {name:<24}{c:>7}{w:>8}{100*w/c:>7.0f}%   {bar}")

    print("\nB. WHY THE CLOCK DID NOT MOVE  (top game messages on wasted calls)")
    for m, c in reasons.most_common(10):
        print(f"   {c:>5}  {m}")
    if not reasons:
        print("   (none recorded)")

    print("\nC. THRASH  (identical name+args repeated)")
    if thrash:
        for (name, args), c in thrash.most_common(6):
            print(f"   {c:>5}  {name}{args[:56]}")
    else:
        print("   (none)")

    print("\nD. CALLS WITH ALL-NULL ARGUMENTS  (tool cannot succeed)")
    if badargs:
        for name, c in badargs.most_common(6):
            print(f"   {c:>5}  {name}")
    else:
        print("   (none)")

    print("\nE. OBSERVATION BUDGET  (mean bytes/turn, and how often present)")
    for nm, b in sect_bytes.most_common(12):
        per = b / max(1, sect_turns[nm])
        print(f"   {nm:<26}{per:>8.0f} B/turn   present {100*sect_turns[nm]/max(1,n_turns):>3.0f}% of turns")

    print("\nF. PUBLISHED vs USED COORDINATES")
    if published:
        used = published & targeted
        print(f"   VISIBLE FEATURES published {len(published)} distinct coords; "
              f"{len(used)} ever targeted ({100*len(used)/len(published):.0f}%)")
        print(f"   -> {100*(1-len(used)/len(published)):.0f}% of published coordinates never steered a call")
    else:
        print("   (no VISIBLE FEATURES block found)")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for d in argv[1:]:
        analyse(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
