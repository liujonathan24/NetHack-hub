#!/usr/bin/env python3
"""Infer what killed a rollout. NOT WIRED INTO ANYTHING -- parked for later.

NetHack states the cause plainly on its death screen ("Killed by a kobold
lord"), but closed-loop skills auto-dismiss that screen before the harness
records it (see nethack.py's note on the death / "Do you want your possessions
identified?" screen), so the string never reaches the trace. What is left is
inference from the last few turns, and it is not good enough to report as fact:

  * "kitten hits" is the PET attacking something else, not the killer.
  * "You are hit by the 1st dart" names the projectile, not who threw it.
  * the nearest visible monster at the end is often a bystander -- on seed 6 it
    was a floating eye standing 1 step away while a kobold lord's darts did the
    killing.

Cross-checked against the orchestrator's own cited evidence, the last-attacker
heuristic below agreed on seeds 6, 7, 9 and 10 and found nothing on seed 8.
That is a useful hint and a bad column in a results table.

The real fix is upstream: capture the death screen before auto-dismissing it,
and publish the cause as a metric. Until then, treat anything here as a guess.

    death_cause.py <cell_dir>          # e.g. .../round1/corpus__prime_agent
"""
from __future__ import annotations
import glob, pathlib, re, sys

PET = re.compile(r"^(kitten|little dog|large dog|pony|cat|dog|horse|large cat)$", re.I)
PROJ = re.compile(
    r"^(\d+(st|nd|rd|th) )?(dart|arrow|crossbow bolt|rock|dagger|spear|boulder|missile)$", re.I)
# Strongest first: something that explicitly hit the PLAYER.
HIT_YOU = re.compile(
    r"([a-z][a-z' \-]{2,24}?)\s+(?:hits|bites|stings|kicks|touches|claws|butts)\s+you", re.I)
# Weaker: any attack verb at all in the closing turns.
ANY_ATTACK = re.compile(
    r"(?:The |A |An )?([a-z][a-z' \-]{2,24}?)\s+"
    r"(?:hits|bites|stings|kicks|touches|claws|butts|casts|zaps)", re.I)


def infer(cell: pathlib.Path, seed: int, window: int = 15) -> str | None:
    tail: list[str] = []
    for f in sorted(glob.glob(str(cell / "turns" / f"{seed}_*.ndjson"))):
        tail += [l for l in open(f) if l.strip()]
    for rx in (HIT_YOU, ANY_ATTACK):
        for line in reversed(tail[-window:]):
            for m in reversed(list(rx.finditer(line))):
                name = m.group(1).strip().lower()
                if PET.match(name) or PROJ.match(name):
                    continue
                return name
    return None


def main() -> int:
    cell = pathlib.Path(sys.argv[1])
    seeds = sorted({int(pathlib.Path(f).name.split("_")[0])
                    for f in glob.glob(str(cell / "turns" / "*.ndjson"))})
    for s in seeds:
        print(f"seed {s}: {infer(cell, s) or '(unknown)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
