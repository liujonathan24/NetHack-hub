"""Golden observation snapshots — a WARN, not a gate.

WHY
---
`docs/HARNESS_DEFECTS.md` §4.1: the same `BBOX` config scored 3.58 ± 1.55, then
2.45 ± 0.34, then 3.71 ± 0.49 across harness-wide fixes — two of which (statue
naming, feature un-truncation) changed the observation *text*. The numbers had
silently stopped being comparable and nothing recorded what the agent had been
shown. `golden/obs/*.txt` is that record.

WHERE THE LINE IS DRAWN, AND WHY
--------------------------------
The observation format is under active iteration; a blocking golden test would
be red on most working days, and a test that is red on most days is a test
everyone learns to `-k 'not golden'` away. So:

  WARN (test PASSES, prints a full diff + emits a pytest warning):
      any content drift at all — a line changed, a section added, a
      non-required section removed, ordering changed, whitespace changed.
      This is "the format moved; results from before this commit are not
      comparable to results after it". That is information, not an error.

  FAIL (test errors):
      1. the snapshot file is missing or unreadable  -> the record itself is
         broken; nothing can be compared and a silent pass would be a lie.
      2. the live observation is empty or whitespace-only -> the agent is being
         shown nothing. Not a format change; a broken renderer.
      3. a REQUIRED section vanished entirely (see `_REQUIRED_SECTIONS`) ->
         the agent lost a whole channel of state.
      4. rendering raised -> the observation path is broken.

  Deliberately NOT a failure: a large diff, a big size change, or a *renamed*
  section. Compaction work and section renames are legitimate iteration, and
  guessing at a "too big" threshold would reintroduce the flakiness this
  design is avoiding. Drift size is reported in the warning; a human decides.

`_REQUIRED_SECTIONS` is the minimum an agent needs to play at all — where it
is, what it has, what just happened, and *some* map channel. It is intentionally
short: JOURNAL, ADJACENT, HINT, VISIBLE FEATURES and VISIBLE MONSTERS are all
WARN-only, because each has been legitimately added, removed or restructured
during this project's normal iteration.

RE-BLESSING
-----------
After an intentional format change, one command (also printed by this test):

    PYTHONPATH=$PWD/tools/pycompat:$ENG:$PWD:$PWD/environments/nethack \
      .venv-cli-eval/bin/python environments/nethack/tests/golden/record_obs_snapshots.py

`record_obs_snapshots.py --check` is the blocking form of this same comparison,
for a human who explicitly wants a gate (e.g. just before cutting a sweep).
"""
from __future__ import annotations

import asyncio
import difflib
import os
import re
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden"))
import obs_configs as OC  # noqa: E402

# A section header line: `=== NAME ===` optionally followed by inline content
# (`=== ADJACENT === N=$(gold) ...`). Non-greedy so inline content containing
# `===` cannot swallow the name.
_SECTION_RE = re.compile(r"^=== (.+?) ===")

# Normalized names of the sections whose disappearance is real breakage rather
# than iteration. Kept deliberately small — see module docstring.
#   MAP       some map channel must exist, even the BBOX "hidden, call reveal"
#             stub; with none, an agent under BBOX has no spatial channel at all.
#   STATUS    HP / Dlvl / position. Without it there is no state and no reward
#             signal the agent can reason about.
#   INVENTORY the only view of what the character carries.
#   MESSAGES  the game's own text. Defect §2.3 was caused by this channel being
#             *thin*; losing it entirely is strictly worse.
_REQUIRED_SECTIONS = frozenset({"MAP", "STATUS", "INVENTORY", "MESSAGES"})


def _section_name(line: str) -> str | None:
    """Normalized section name for a header line, else None.

    `=== MAP (hidden; call reveal(...)) ===` normalizes to `MAP` so the B0 and
    BBOX map headers count as the same section — the BBOX stub IS the map
    channel, just an empty one.
    """
    m = _SECTION_RE.match(line)
    if not m:
        return None
    return m.group(1).split("(")[0].strip().upper()


def _sections(text: str) -> list[str]:
    return [n for n in (_section_name(ln) for ln in text.splitlines()) if n]


def _render(name: str) -> str:
    try:
        return asyncio.run(OC.render_observation(name))
    except Exception as exc:  # FAIL condition 4
        raise AssertionError(
            f"GOLDEN OBS FAIL [{name}]: rendering the observation raised "
            f"{type(exc).__name__}: {exc}. The observation path is broken; this "
            f"is not a format change."
        ) from exc


def _drift_report(name: str, snapshot: str, live: str) -> str:
    """Human-readable 'what changed', section-level then line-level."""
    old_secs, new_secs = _sections(snapshot), _sections(live)
    lines = [f"GOLDEN OBS DRIFT [{name}]: the observation format has changed."]

    added = [s for s in new_secs if s not in old_secs]
    removed = [s for s in old_secs if s not in new_secs]
    if added:
        lines.append(f"  sections ADDED:   {', '.join(added)}")
    if removed:
        lines.append(f"  sections REMOVED: {', '.join(removed)}")
    if not added and not removed and old_secs != new_secs:
        lines.append(f"  section ORDER changed: {' | '.join(old_secs)}"
                     f"  ->  {' | '.join(new_secs)}")

    o, n = snapshot.splitlines(), live.splitlines()
    changed = sum(1 for d in difflib.ndiff(o, n) if d[0] in "+-")
    lines.append(f"  {len(snapshot)} -> {len(live)} chars, "
                 f"{len(o)} -> {len(n)} lines, ~{changed} line(s) differ")
    lines.append("  --- unified diff (snapshot -> live) ---")
    lines += ["  " + d for d in difflib.unified_diff(
        o, n, fromfile=f"golden/obs/{name}.txt", tofile=f"live {name}",
        lineterm="", n=2)]
    lines.append("")
    lines.append("  Results produced BEFORE this change are not comparable to "
                 "results produced after it.")
    lines.append("  If this change was intentional, re-bless with:")
    lines.append("    " + OC.REBLESS_CMD)
    return "\n".join(lines)


class GoldenObsDrift(UserWarning):
    """Emitted when a pinned observation's content has drifted. Not an error."""


class GoldenObsBreakage(AssertionError):
    """Raised for the FAIL conditions only — see module docstring."""


def classify(name: str, snapshot: str, live: str) -> str | None:
    """The whole warn-vs-fail decision, as one pure function.

    Returns None when identical, a drift report string when it should WARN, and
    raises `GoldenObsBreakage` for the FAIL conditions. Kept free of pytest and
    of the engine so the policy itself is unit-testable (see the tests at the
    bottom of this file) — the line between "iterating" and "broken" is the
    part of this design most likely to be argued with later, so it is executable
    documentation rather than prose.
    """
    # FAIL 2: the agent is being shown nothing.
    if not live.strip():
        raise GoldenObsBreakage(
            f"GOLDEN OBS FAIL [{name}]: the rendered observation is empty / "
            f"whitespace-only. This is a broken renderer, not a format change.")

    # FAIL 3: a whole channel of state disappeared. Compared against the
    # sections the SNAPSHOT had, so a section that was never in this cell (e.g.
    # HINT under fog) cannot trip it.
    lost = sorted((_REQUIRED_SECTIONS & set(_sections(snapshot))) - set(_sections(live)))
    if lost:
        raise GoldenObsBreakage(
            f"GOLDEN OBS FAIL [{name}]: required section(s) {lost} disappeared "
            f"from the observation. Required sections are "
            f"{sorted(_REQUIRED_SECTIONS)}; losing one removes a whole channel "
            f"of state from the agent. If this really is intentional, edit "
            f"_REQUIRED_SECTIONS in this file (and say why) before re-blessing.")

    return None if live == snapshot else _drift_report(name, snapshot, live)


@pytest.mark.parametrize("name", sorted(OC.CONFIGS))
def test_observation_matches_golden_snapshot(name, capsys):
    path = OC.snapshot_path(name)

    # FAIL 1: the record itself is broken.
    if not os.path.exists(path):
        pytest.fail(
            f"GOLDEN OBS FAIL [{name}]: no snapshot at {path}. The pinned "
            f"observation record is missing, so nothing can be compared. "
            f"Create it with:\n  {OC.REBLESS_CMD}")
    try:
        with open(path, encoding="utf-8") as fh:
            snapshot = fh.read()
    except OSError as exc:
        pytest.fail(f"GOLDEN OBS FAIL [{name}]: snapshot {path} is unreadable: {exc}")

    report = classify(name, snapshot, _render(name))

    # Everything else: WARN. Test passes.
    if report is not None:
        # print() so the full diff lands in `-s` / `-rA` output and in CI logs;
        # stderr via capsys.disabled() so it is visible even on a passing `-q`
        # run; warnings.warn() so it also lands in pytest's warnings summary.
        print("\n" + report)
        with capsys.disabled():
            sys.stderr.write("\n" + report + "\n")
        warnings.warn(report, GoldenObsDrift, stacklevel=2)


def test_every_config_has_a_snapshot():
    """The matrix and the committed files must not drift apart.

    A cell added to `CONFIGS` without re-blessing would otherwise fail only as
    "missing snapshot" inside the parametrized test, which reads like breakage.
    An orphan .txt (a cell deleted from CONFIGS) is reported here too, since
    nothing else would ever look at it again.
    """
    on_disk = {f[:-4] for f in os.listdir(OC.SNAPSHOT_DIR) if f.endswith(".txt")}
    expected = set(OC.CONFIGS)
    assert not (expected - on_disk), (
        f"config(s) {sorted(expected - on_disk)} have no snapshot; run:\n  "
        f"{OC.REBLESS_CMD}")
    assert not (on_disk - expected), (
        f"orphan snapshot(s) {sorted(on_disk - expected)} in {OC.SNAPSHOT_DIR} "
        f"— no CONFIGS entry produces them; delete them or restore the config.")


# --------------------------------------------------------------------------
# Policy tests: these pin the warn-vs-fail LINE, with no engine involved.
# --------------------------------------------------------------------------

_SYNTH = (
    "=== JOURNAL ===\nObjective: descend.\n\n"
    "=== MAP ===\n  |@..|\n\n"
    "=== STATUS ===\nHP: 16/16  Dlvl: 1\n\n"
    "=== INVENTORY ===\n  b: a long sword\n\n"
    "=== ADJACENT === N=. E=.\n\n"
    "=== MESSAGES ===\n  Welcome to NetHack!\n"
)


def test_identical_observation_is_silent():
    assert classify("synth", _SYNTH, _SYNTH) is None


@pytest.mark.parametrize("mutation,label", [
    (lambda t: t.replace("HP: 16/16", "HP: 9/16"), "a content line changed"),
    (lambda t: t.replace("=== ADJACENT ===", "=== SURROUNDINGS ==="), "a section renamed"),
    (lambda t: t.replace("=== ADJACENT === N=. E=.\n\n", ""), "a NON-required section removed"),
    (lambda t: t + "\n=== NEW THING ===\nhello\n", "a section added"),
    (lambda t: t.replace("  |@..|", "  |@.{2}|"), "the map re-encoded (compaction)"),
    (lambda t: "=== MESSAGES ===\n  x\n" + t.replace("=== MESSAGES ===\n  Welcome to NetHack!\n", ""),
     "sections reordered"),
])
def test_content_drift_only_warns(mutation, label):
    """Every one of these is normal iteration: report it, never block on it."""
    report = classify("synth", _SYNTH, mutation(_SYNTH))
    assert report is not None, f"{label} produced no report"
    assert "GOLDEN OBS DRIFT" in report
    assert OC.REBLESS_CMD in report, "the report must say how to re-bless"


@pytest.mark.parametrize("live", ["", "   \n\n  \t\n"])
def test_empty_observation_fails(live):
    with pytest.raises(GoldenObsBreakage, match="empty / whitespace-only"):
        classify("synth", _SYNTH, live)


@pytest.mark.parametrize("section", sorted(_REQUIRED_SECTIONS))
def test_losing_a_required_section_fails(section):
    live = "\n".join(
        ln for ln in _SYNTH.splitlines() if _section_name(ln) != section)
    with pytest.raises(GoldenObsBreakage, match="disappeared"):
        classify("synth", _SYNTH, live)


def test_bbox_hidden_map_header_still_counts_as_the_map_section():
    """The BBOX stub `=== MAP (hidden; call reveal(...)) ===` is the map
    channel. Normalizing it to `MAP` is what keeps the required-section rule
    usable across both variants — regress that and BBOX would fail every run."""
    assert _section_name("=== MAP (hidden; call reveal(x1,y1,x2,y2) to view a region) ===") == "MAP"
    assert _section_name("=== ADJACENT === N=$(gold) NE=. E=.") == "ADJACENT"
    assert _section_name("  |....+#####  ===  weird") is None


def test_required_sections_present_in_every_committed_snapshot():
    """Guards the guard: if a committed snapshot is missing a required section,
    FAIL 3 above can never fire for that cell (it only fires on sections the
    snapshot had). Cheap, no engine boot."""
    for name in sorted(OC.CONFIGS):
        with open(OC.snapshot_path(name), encoding="utf-8") as fh:
            secs = set(_sections(fh.read()))
        missing = sorted(_REQUIRED_SECTIONS - secs)
        assert not missing, (
            f"committed snapshot {name}.txt is missing required section(s) "
            f"{missing}; it was blessed from a broken observation.")
