"""`reveal` must return what it was asked for, or say that it did not.

Regression for docs/HARNESS_DEFECTS.md 3.3: `_REVEAL_MAX_W = 40` was applied to
the exclusive difference `x2 - x1`, so a full-width request (`x1=0, x2=78`)
came back with 41 of 79 columns and *no indication whatsoever* that anything had
been dropped. Under `BBOX` / `SPARSE_ONDEMAND`, `reveal` is the agent's only
access to the map, so this was half a dungeon presented as a whole one.

Two properties are tested here, and they are different properties:
  1. a full-width request returns the full width;
  2. any clamp that DOES happen is visible -- in the returned text, and in a
     counter a sweep can read.
"""

import pathlib
import sys

import numpy as np
import pytest

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.helpers import classify_tool_result  # noqa: E402
from nethack_harness.tools import skills as SK  # noqa: E402


@pytest.fixture()
def env():
    from nethack_core.env import NetHackCoreEnv
    e = NetHackCoreEnv(task_name="NetHackScore-v0")
    e.seed(42, 42)
    e.reset()
    SK.reset_reveal_clamp_counts()
    try:
        yield e
    finally:
        e.close()


def _rows(feedback: str) -> list[str]:
    """The `y N: ...` grid lines only (no header, no NOTE)."""
    return [ln for ln in feedback.split("\n") if ln.startswith("y")]


def _width(line: str) -> int:
    return len(line) - len("y 0: ")


def test_full_width_request_returns_all_79_columns(env):
    """The headline defect. Was 41 columns; the NetHack map is 79 wide."""
    res = SK.reveal(env, None, 0, 0, SK._MAP_MAX_X, SK._MAP_MAX_Y)
    rows = _rows(res.feedback)

    assert res.feedback.startswith("reveal (x0-78, y0-20):")
    assert len(rows) == 21, "the map is 21 rows tall"
    assert {_width(r) for r in rows} == {79}, "every row must carry all 79 columns"


def test_a_full_map_request_is_not_a_clamp_and_says_nothing_about_one(env):
    res = SK.reveal(env, None, 0, 0, SK._MAP_MAX_X, SK._MAP_MAX_Y)

    assert "NOTE:" not in res.feedback
    counts = SK.reveal_clamp_counts()
    assert counts["calls"] == 1
    assert counts["out_of_range"] == counts["too_wide"] == counts["too_tall"] == 0
    assert counts["truncated"] == 0


def test_the_full_map_fits_the_character_budget_with_room_to_spare(env):
    """The size blowup the old cap was protecting against, as a number.

    A full-map reveal measured 1,807 characters on a live game -- of the same
    order as the `=== MAP ===` section a full-map variant sends every turn
    (1,328) and well inside a whole observation (2,613). This asserts the cost
    stays in that range, so a future format change that doubles it fails here
    instead of quietly inflating every BBOX prompt.
    """
    res = SK.reveal(env, None, 0, 0, SK._MAP_MAX_X, SK._MAP_MAX_Y)

    assert len(res.feedback) < 2000, len(res.feedback)
    assert len(res.feedback) <= SK._REVEAL_MAX_CHARS
    assert "(truncated)" not in res.feedback


def test_out_of_range_request_is_clamped_and_says_so(env):
    res = SK.reveal(env, None, -5, -5, 200, 40)

    assert res.feedback.startswith("reveal (x0-78, y0-20):")
    assert {_width(r) for r in _rows(res.feedback)} == {79}
    assert "NOTE: partial view" in res.feedback
    assert "clamped to the grid" in res.feedback
    assert SK.reveal_clamp_counts()["out_of_range"] == 1
    # Per-episode mirror, for anything holding an env rather than the module.
    assert getattr(env, "_reveal_clamps", {}).get("out_of_range") == 1


def test_a_clamped_reveal_is_still_classified_completed(env):
    """The NOTE must not read as a failure to the trace classifier.

    `helpers._STATUS_MARKERS` matches `"reveal: "` -> failed BEFORE
    `"reveal (x"` -> completed, so wording that starts a line with `reveal:`
    would silently reclassify every clamped call as a failure.
    """
    res = SK.reveal(env, None, -5, -5, 200, 40)
    assert classify_tool_result(res.feedback) == "completed"


def test_a_size_cap_that_does_fire_names_the_columns_it_withheld(env, monkeypatch):
    """Keep the cap usable for an experiment -- but never silent."""
    monkeypatch.setattr(SK, "_REVEAL_MAX_COLS", 40)
    res = SK.reveal(env, None, 0, 0, 78, 20)

    rows = _rows(res.feedback)
    assert {_width(r) for r in rows} == {40}
    assert "columns 40-78 are NOT shown" in res.feedback
    assert SK.reveal_clamp_counts()["too_wide"] == 1


def test_a_height_cap_that_does_fire_names_the_rows_it_withheld(env, monkeypatch):
    """The height cap is handled exactly like the width cap.

    It was the same off-by-one (`y2 - y1 > 20` on a 21-row map), it just never
    bit, because the grid clamp bounded y first.
    """
    monkeypatch.setattr(SK, "_REVEAL_MAX_ROWS", 10)
    res = SK.reveal(env, None, 0, 0, 78, 20)

    assert len(_rows(res.feedback)) == 10
    assert "rows 10-20 are NOT shown" in res.feedback
    assert SK.reveal_clamp_counts()["too_tall"] == 1


def test_caps_are_the_full_map_so_no_legal_request_can_be_shortened(env):
    assert SK._REVEAL_MAX_COLS == SK._MAP_MAX_X + 1 == 79
    assert SK._REVEAL_MAX_ROWS == SK._MAP_MAX_Y + 1 == 21


def test_reveal_matches_the_engine_grid_it_claims_to_show(env):
    """Faithfulness: the text must be the tty rows, not a re-render.

    Map row y is tty row y+1 (row 0 is the message line). Getting this wrong
    would be invisible in a width test and fatal in play.
    """
    res = SK.reveal(env, None, 0, 0, SK._MAP_MAX_X, SK._MAP_MAX_Y)
    tty = np.asarray(env._last_observation.tty_chars)

    for i, line in enumerate(_rows(res.feedback)):
        expected = "".join(chr(int(c)) if int(c) else " " for c in tty[i + 1, 0:79])
        assert line[len("y 0: "):] == expected, f"row {i} does not match tty row {i + 1}"


def test_counters_are_resettable_for_per_batch_measurement():
    SK.reset_reveal_clamp_counts()
    assert set(SK.reveal_clamp_counts().values()) == {0}
