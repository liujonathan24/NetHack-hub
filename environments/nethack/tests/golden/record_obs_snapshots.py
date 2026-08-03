#!/usr/bin/env python
"""Re-bless the golden observation snapshots.

    cd /root/NetHack-hub
    export ENG=/root/NetHack-engine
    PYTHONPATH=$PWD/tools/pycompat:$ENG:$PWD:$PWD/environments/nethack \
      .venv-cli-eval/bin/python environments/nethack/tests/golden/record_obs_snapshots.py

Writes one `obs/<name>.txt` per entry in `obs_configs.CONFIGS`, containing the
turn-1 observation verbatim -- no header, no wrapper -- so `git diff` on a
re-bless reads as the format change itself.  The config that produced each file
lives in `obs_configs.CONFIGS`, keyed by filename; that is the only place it is
recorded, so the two cannot drift.

Flags:
  --check    render but do not write; exit 1 if anything differs (this is the
             blocking form, for a human who *wants* a gate; the pytest check is
             deliberately non-blocking -- see ../test_golden_obs.py)
  --twice    render every cell twice and abort unless the two renders are
             byte-identical (determinism guard; also run by default)
  --only N   restrict to one cell name

DETERMINISM
-----------
Everything in these observations is a pure function of (seed, character,
variant, tune, compaction): the dungeon comes from the pinned core/display
seed, and the turn-1 state is post-reset with no action applied.  Nothing
wall-clock, PID- or path-derived reaches the rendered text -- the trace writer's
`<seed>_<pid>_<time>` run id is filename-only and never rendered.  Verified by
rendering every cell twice in-process and twice in separate interpreters:
byte-identical both ways.  Consequently NOTHING is normalized out; the files are
the raw observation.  If a future field does become nondeterministic, normalize
it HERE and say so in this docstring rather than loosening the comparison in the
test -- a normalization that lives in the test is invisible to whoever reads the
snapshot.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import obs_configs as OC  # noqa: E402


async def _render(name: str, twice: bool) -> str:
    text = await OC.render_observation(name)
    if twice:
        again = await OC.render_observation(name)
        if again != text:
            diff = "\n".join(difflib.unified_diff(
                text.splitlines(), again.splitlines(),
                fromfile="render#1", tofile="render#2", lineterm="", n=1))
            raise SystemExit(
                f"NONDETERMINISTIC: cell {name!r} rendered differently twice in "
                f"one process. Do not commit this snapshot; normalize the "
                f"offending field in record_obs_snapshots.py first.\n{diff}")
    return text


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 on any difference")
    ap.add_argument("--twice", dest="twice", action="store_true", default=True,
                    help="render each cell twice and require identical output (default)")
    ap.add_argument("--no-twice", dest="twice", action="store_false")
    ap.add_argument("--only", default=None, help="restrict to one cell name")
    args = ap.parse_args()

    names = [args.only] if args.only else list(OC.CONFIGS)
    for n in names:
        if n not in OC.CONFIGS:
            raise SystemExit(f"unknown cell {n!r}; known: {', '.join(OC.CONFIGS)}")

    os.makedirs(OC.SNAPSHOT_DIR, exist_ok=True)
    differed = False
    for name in names:
        text = await _render(name, args.twice)
        path = OC.snapshot_path(name)
        old = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                old = fh.read()
        if old == text:
            print(f"  unchanged  {name}  ({len(text)} chars)")
            continue
        differed = True
        if args.check:
            print(f"  DIFFERS    {name}")
            for line in difflib.unified_diff(
                    (old or "").splitlines(), text.splitlines(),
                    fromfile=f"{name} (snapshot)", tofile=f"{name} (live)",
                    lineterm="", n=2):
                print("    " + line)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"  {'wrote' if old is None else 'REBLESSED'}  {name}  ({len(text)} chars) -> {path}")

    if args.check and differed:
        print("\n--check: snapshots are stale. Re-bless with:\n  " + OC.REBLESS_CMD)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
