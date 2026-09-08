#!/usr/bin/env python3
"""Fabricate a plausible seed-1 E16 checkpoint archive and render its ledger.

The numbers are INVENTED. They are drawn from real E15 seed-1 findings (the
Mines entrance around D3, the Grey-elf that pursues through corridors, the
owlbear wall at D11) so the orchestrator is reasoning over something shaped
like the real thing -- but nothing here was measured by an engine, and nothing
produced here may ever be quoted as a result. This file exists only so SIM 1
can ask "does the model reason over ledger FIELDS".

The meta.json schema and the rendered table come from the real orchestrator
(``tools/cli_harness_eval/e16_orchestrator.py``) when it is importable, so the
bytes served to the model in this simulation are the bytes the real run would
serve. If it is not importable the fallback renderer below reproduces its
format; a banner records which path was taken.

OUT is the scratchpad, not the shared worktree: the first run of this
simulation wrote into ``outputs/e16_sim/`` and a concurrent agent removed the
whole directory at 10:31.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/root/nld/zombie-fix")
OUT = Path(os.environ.get(
    "E16_SIM_OUT",
    "/tmp/claude-0/-root/3382b9f1-b1ec-4e18-9c3e-4585c4167d08/scratchpad/"
    "e16sim/artifacts"))

CHECKPOINTS = [
    dict(id="c03", parent=None, name="dlvl2-first-descent",
         note="Cleared D1 and D2, pet still alive, 2 food rations. Safe base.",
         dlvl=2, xl=2, hp=18, max_hp=18, gameturn=612, score=241,
         balrog=0.041, balrog_min=0.041, attempts_from=1, visits=1,
         created_by="auto"),
    dict(id="c07", parent="c03", name="mines-entrance",
         note="At the Gnomish Mines branch staircase on D3, full HP. Saved "
              "before choosing Mines vs main dungeon.",
         dlvl=3, xl=2, hp=21, max_hp=21, gameturn=1044, score=388,
         balrog=0.062, balrog_min=0.062, attempts_from=2, visits=2,
         created_by="save"),
    dict(id="c12", parent="c07", name="d5-arrival",
         note="Arrived D5 with the Grey-elf still on D4 behind me. Escape "
              "plan: north corridor back to the up-stair.",
         dlvl=5, xl=3, hp=27, max_hp=31, gameturn=1893, score=702,
         balrog=0.104, balrog_min=0.104, attempts_from=4, visits=2,
         created_by="save"),
    dict(id="c15", parent="c12", name="d7-fountain-room",
         note="Big fountain room on D7, cleared. Long sword and +1 ring mail "
              "worn. Good place to rest up before pushing on.",
         dlvl=7, xl=4, hp=38, max_hp=41, gameturn=2760, score=1180,
         balrog=0.148, balrog_min=0.148, attempts_from=1, visits=1,
         created_by="save"),
    dict(id="c19", parent="c15", name="d9-post-elf",
         note="Killed the Grey-elf on D9 at 12 HP, then rested to near full. "
              "Down staircase is known and two rooms east.",
         dlvl=9, xl=5, hp=44, max_hp=47, gameturn=3921, score=1904,
         balrog=0.191, balrog_min=0.191, attempts_from=3, visits=1,
         created_by="save"),
    dict(id="c21", parent="c19", name="d10-corridor-cache",
         note="Stashed spare armour in the D10 corridor junction. Nothing "
              "hostile in sight; HP not yet topped up.",
         dlvl=10, xl=5, hp=31, max_hp=47, gameturn=4302, score=2011,
         balrog=0.212, balrog_min=0.212, attempts_from=0, visits=1,
         created_by="auto"),
    dict(id="c22", parent="c21", name="d11-stairs-known",
         note="On D11 with the down staircase already mapped. An owlbear "
              "wanders the west half of the level; the stair is south-east.",
         dlvl=11, xl=5, hp=29, max_hp=47, gameturn=4588, score=2140,
         balrog=0.233, balrog_min=0.233, attempts_from=1, visits=1,
         created_by="save"),
]

LESSONS = {
    "c07": "## attempt 2\nWent into the Mines. Gnome lord with a crossbow at "
           "the second Mines level; died at XL2. The Mines are not safe at "
           "this XL.\n",
    "c12": "## attempt 3\nGrey-elf pursues through corridors; it does not "
           "give up when you break line of sight. Outrun it in the open or "
           "fight it with a corridor at your back -- do not try to lose it.\n"
           "## attempt 5\nDescended immediately on arrival and met the elf "
           "again one level down at 14 HP. Rest first.\n"
           "## attempt 8\nTook the north corridor per the escape plan; it "
           "dead-ends. The south door is the real exit.\n"
           "## attempt 11\nRested to full before descending. Reached D7. This "
           "is the line that works.\n",
    "c15": "## attempt 12\nPrayed at 6 HP and it worked, but the cooldown is "
           "now live -- do not count on prayer again for ~1000 turns.\n",
    "c19": "## attempt 13\nDescended to D11, met an owlbear in the open, "
           "melee'd it and died in 6 calls at XL5.\n"
           "## attempt 15\nDescended to D11, owlbear again, tried to flee "
           "north, cornered, died.\n"
           "## attempt 17\nDescended to D11, owlbear a third time. Died. "
           "Three attempts from this checkpoint, three owlbear deaths on "
           "D11. The problem is not the route, it is the fight.\n",
    "c22": "## attempt 18\nStarted on D11 south-east of the owlbear, reached "
           "the down staircase without a fight, descended to D12 at 29 HP, "
           "then died to a soldier ant on D12 at 29/47.\n",
}

SUMMARY_LAST_ATTEMPT = (
    "Attempt 18 (from c22, 'd11-stairs-known'). I went straight for the "
    "south-east staircase without engaging the owlbear and it worked -- I "
    "reached D12 on game turn 4731 at 29/47 HP. Then a soldier ant came out "
    "of a side room while I was still under half HP and killed me in four "
    "turns. Lesson: the D11 staircase route is solved, but arriving on D12 "
    "under 60% HP is not survivable. Rest before descending, not after."
)


def write_archive(root: Path) -> Path:
    archive = root / "archive" / "sim1"
    archive.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for i, ck in enumerate(CHECKPOINTS):
        d = archive / ck["id"]
        d.mkdir(exist_ok=True)
        meta = dict(ck)
        meta["created_at"] = now - (len(CHECKPOINTS) - i) * 3600
        meta["dungeon_number"] = 0
        meta["level_number"] = ck["dlvl"]
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (d / "lessons.md").write_text(LESSONS.get(ck["id"], ""))
        (d / "state.bundle").write_bytes(b"")
    return archive


def render(archive: Path) -> tuple:
    """Return (ledger_text, provenance-of-renderer)."""
    sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))
    try:
        import e16_orchestrator as eo  # type: ignore
        rows = [eo.row_from_meta(p, json.loads((p / "meta.json").read_text()))
                for p in sorted(archive.iterdir())
                if (p / "meta.json").is_file()]
        return eo.render_ledger(rows), "e16_orchestrator.render_ledger"
    except Exception as exc:  # pragma: no cover - fallback path
        rows = [json.loads((p / "meta.json").read_text())
                for p in sorted(archive.iterdir())
                if (p / "meta.json").is_file()]
        rows.sort(key=lambda r: (-r["dlvl"], -r["xl"], -r["score"]))
        lines = ["CHECKPOINT ARCHIVE (%d saved state(s)). All numbers below "
                 "are measured by the harness from the game engine, not "
                 "written by anyone." % len(rows),
                 "  id  Dlvl  XL   HP     turn   score  attempts  "
                 "name / why it was saved"]
        for r in rows:
            lines.append(
                f"  {r['id']:>4}  {r['dlvl']:>4}  {r['xl']:>2}  "
                f"{r['hp']:>3}/{r['max_hp']:<3} {r['gameturn']:>6} "
                f"{r['score']:>6}  {r['attempts_from']:>8}  "
                f"{r['name'][:28]} | {r['note'][:60]}")
        return "\n".join(lines), f"fallback ({exc!r})"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = write_archive(OUT)
    ledger, who = render(archive)
    (OUT / "sim1_ledger.txt").write_text(ledger + "\n")
    lessons = []
    for ck in CHECKPOINTS:
        body = LESSONS.get(ck["id"])
        if body:
            lessons.append(f"--- lessons.md for {ck['id']} "
                           f"({ck['name']}) ---\n{body}")
    (OUT / "sim1_lessons.txt").write_text("\n".join(lessons))
    (OUT / "sim1_last_summary.txt").write_text(SUMMARY_LAST_ATTEMPT + "\n")
    print(f"renderer: {who}")
    print(ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
