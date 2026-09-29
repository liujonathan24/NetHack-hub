#!/usr/bin/env python3
"""SIM 1 counterfactual: the deepest checkpoint is now the WRONG choice.

The point of SIM 1 is not "did the model produce a plausible paragraph" -- a
model that always names the deepest row produces a plausible paragraph too.
The discriminating test is whether the choice MOVES when the fields move. So
this builds ledger B, identical to ledger A except in the fields round A's
answer explicitly cited:

  c22 (D11, the frontier)  attempts_from 1 -> 7,  HP 29/47 -> 11/47
                           lessons: seven attempts, seven fast deaths, and
                           the owlbear now camps ON the staircase.
  c19 (D9)                 attempts_from 3 -> 0,  HP 44/47 -> 47/47
                           lessons: the owlbear deaths are gone; the level is
                           cleared and the down stair is mapped.

Dlvl/XL/score ordering is UNCHANGED, so "pick the deepest" and "pick the top
row" both still answer c22. Only reasoning over attempts_from, HP and the
lesson text can move the answer off c22.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path("/root/nld/zombie-fix")
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("E16_SIM_OUT", str(HERE.parent / "artifacts")))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))

import make_ledger as ml  # noqa: E402

OVERRIDES = {
    "c22": dict(attempts_from=7, hp=11,
                note="On D11 with the down staircase already mapped. An "
                     "owlbear has been camping ON the staircase since "
                     "attempt 19."),
    "c19": dict(attempts_from=0, hp=47,
                note="D9 is cleared and the down staircase is mapped. Full "
                     "HP, nothing hostile on the level."),
}

LESSONS_B = dict(ml.LESSONS)
LESSONS_B["c22"] = (
    "## attempts 18-24\nSeven attempts have now started from this "
    "checkpoint. All seven died: five on D11 within 40 game turns (the "
    "owlbear is sitting on the down staircase and there is no way past it "
    "at 11 HP), two on D12 immediately after descending. Nothing has been "
    "learned from this state in four rounds.\n")
LESSONS_B["c19"] = (
    "## attempt 25\nRe-explored D9 properly instead of rushing the descent. "
    "Cleared the level, found the down staircase, rested to 47/47. The three "
    "old owlbear deaths came from descending at low HP into an unmapped "
    "D11; that is no longer the situation from this state.\n")

SUMMARY_B = (
    "Attempt 24 (from c22, 'd11-stairs-known'). Seventh attempt from this "
    "state. The owlbear is on the staircase and I started at 11/47 HP; I "
    "tried to Elbereth it off the stair, the engraving smudged, and it "
    "killed me on game turn 4623 -- 35 turns after the restore. This "
    "checkpoint has produced nothing in seven tries."
)


def main() -> int:
    for ck in ml.CHECKPOINTS:
        if ck["id"] in OVERRIDES:
            ck.update(OVERRIDES[ck["id"]])
    ml.LESSONS = LESSONS_B
    archive = ml.write_archive(OUT / "cf")
    ledger, who = ml.render(archive)
    (OUT / "sim1b_ledger.txt").write_text(ledger + "\n")
    lessons = []
    for ck in ml.CHECKPOINTS:
        body = LESSONS_B.get(ck["id"])
        if body:
            lessons.append(f"--- lessons.md for {ck['id']} "
                           f"({ck['name']}) ---\n{body}")
    (OUT / "sim1b_lessons.txt").write_text("\n".join(lessons))
    (OUT / "sim1b_last_summary.txt").write_text(SUMMARY_B + "\n")
    print(f"renderer: {who}")
    print(ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
