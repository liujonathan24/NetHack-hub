"""The post-baseline tool fixes must be reconstructible, both ways.

The E10 baseline is the denominator of every claim in this series. It was
measured before PR #29, so unless those fixes can be switched OFF, "run the
baseline" silently means "run the baseline plus undated improvements" and the
published reference numbers describe nothing reproducible.

These tests pin the contract: default OFF, string-coerced, and each flag
actually changing the behaviour it names.
"""
from __future__ import annotations

import pytest

from nethack_harness import tool_flags


@pytest.fixture(autouse=True)
def _clean_flags():
    tool_flags.reset()
    yield
    tool_flags.reset()


def test_every_flag_defaults_off_so_base_is_the_default():
    snap = tool_flags.snapshot()
    assert snap == {
        "netplay_telemetry": False,
        "melee_hints": False,
        "skill_doc_coords": False,
    }, "a flag defaulting ON would make the baseline unreachable from config"


def test_env_args_strings_coerce_the_way_the_other_knobs_do():
    """env_args arrive as strings through the CLI, so "false" must not read as
    True the way a non-empty string otherwise would -- the bug `describe_args`
    and `reflect` both carry explicit coercion for."""
    tool_flags.configure(netplay_telemetry="false", melee_hints="0")
    assert tool_flags.enabled("netplay_telemetry") is False
    assert tool_flags.enabled("melee_hints") is False

    tool_flags.configure(netplay_telemetry="true", melee_hints="1")
    assert tool_flags.enabled("netplay_telemetry") is True
    assert tool_flags.enabled("melee_hints") is True

    for falsey in ("", "off", "no", "False", None):
        tool_flags.configure(netplay_telemetry=falsey)
        assert tool_flags.enabled("netplay_telemetry") is False, falsey


def test_unrelated_kwargs_are_ignored():
    """`configure(**kwargs)` is handed the env's whole kwarg bag."""
    tool_flags.configure(skill_set="np_core", max_turns=200, tune={"reveal_map": 1.0})
    assert tool_flags.snapshot() == {
        "netplay_telemetry": False, "melee_hints": False, "skill_doc_coords": False,
    }


def test_melee_report_is_the_bare_coordinate_when_the_flag_is_off():
    """The stale-target report (c1a0bec) is the whole point of `melee_hints`."""
    from netplay.nethack_agent.skills import _melee_target_report

    off = _melee_target_report(agent=None, target_glyph=42, tx=3, ty=4)
    assert off == "Unable to reach the target at (3, 4)."
    assert "matching monster" not in off


def test_melee_report_names_the_new_position_when_the_flag_is_on():
    from netplay.nethack_agent.skills import _melee_target_report

    class _Pos:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Level:
        def get_monsters(self):
            return [(42, _Pos(9, 9)), (42, _Pos(5, 5)), (7, _Pos(3, 4))]

    class _Agent:
        current_level = _Level()

    tool_flags.configure(melee_hints=True)
    on = _melee_target_report(_Agent(), 42, 3, 4)
    # Nearest same-glyph monster by Manhattan distance, and never a claimed
    # identity -- glyph values cannot distinguish two of the same species.
    assert "(5, 5)" in on
    assert "a matching monster" in on


def test_the_added_keys_are_unavailable_until_the_flag_is_on():
    """`-` is the one that matters: it is "your fingers" in the engrave prompt,
    i.e. the only path to dust-engraving Elbereth. The baseline could not press
    it, so a baseline cell must not be able to either."""
    from netplay.nethack_utils.nle_wrapper import RawKeyPress

    with pytest.raises(ValueError, match="Cannot parse the given key"):
        RawKeyPress.parse("-")

    tool_flags.configure(netplay_telemetry=True)
    assert RawKeyPress.parse("-") == RawKeyPress.KEYPRESS_MINUS


def test_the_gate_covers_every_key_the_commit_added():
    from netplay.nethack_utils.nle_wrapper import RawKeyPress, _POST_BASELINE_KEYS

    for key in "-']{}|~":
        with pytest.raises(ValueError):
            RawKeyPress.parse(key)
    assert len(_POST_BASELINE_KEYS) == 6 + 1

    # Keys that predate the baseline must stay available with everything off.
    for key in "abz>%`":
        assert RawKeyPress.parse(key) == RawKeyPress(ord(key))


def test_keys_that_predate_the_baseline_are_never_gated():
    from netplay.nethack_utils.nle_wrapper import RawKeyPress

    assert RawKeyPress.parse("enter") == RawKeyPress.KEYPRESS_ENTER
    assert RawKeyPress.parse("esc") == RawKeyPress.KEYPRESS_ESC


# --------------------------------------------------------------------------- #
# ea8cc15: the nearest-monster hint on the no-monster-here path (same flag)    #
# --------------------------------------------------------------------------- #
def _drive_no_monster_failure():
    """Run melee_attack against an empty tile and return the failure text."""
    from netplay.nethack_agent.skills import melee_attack

    class _Pos:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Level:
        def get_monster_glyph(self, x, y):
            return None                      # nothing at the target tile

        def get_monsters(self):
            return [(42, _Pos(10, 12)), (7, _Pos(2, 2))]

    class _Agent:
        current_level = _Level()

        def waiting_for_popup(self):
            return False          # @fail_on_popup gate

    steps = list(melee_attack(_Agent(), 3, 4))
    assert len(steps) == 1 and "fail" in str(steps[0].status).lower()
    return steps[0].thoughts


def test_no_monster_failure_is_bare_when_the_flag_is_off():
    """ea8cc15 shares `melee_hints` with the stale-target report: one
    behaviour split across two commits (see tool_flags._DEFAULTS)."""
    text = _drive_no_monster_failure()
    assert text == "There is no monster at (3,4)."


def test_no_monster_failure_names_the_nearest_monster_when_on():
    tool_flags.configure(melee_hints=True)
    text = _drive_no_monster_failure()
    # nearest by Manhattan distance to the requested tile, either glyph --
    # there is no target glyph to match on this path.
    assert "Nearest visible monster is at (2, 2)." in text


# --------------------------------------------------------------------------- #
# aee5c43 / skill_doc_coords: the SKILL.md coordinate-frame note               #
# --------------------------------------------------------------------------- #
def test_skill_doc_serves_the_baseline_wording_when_off():
    """OFF must be byte-exact baseline: the note is swapped back, not deleted."""
    import importlib
    hp = importlib.import_module("nethack_prime_agent")

    shipped = (importlib.resources.files("nethack_prime_agent") / "skill" / "SKILL.md").read_bytes()
    assert hp._COORD_FRAME_NOTE in shipped.decode()   # ON-state is the file itself

    off = hp._restore_baseline_coord_note(shipped).decode()
    assert hp._COORD_FRAME_NOTE not in off
    assert hp._COORD_FRAME_BASELINE in off


def test_skill_doc_strip_fails_loudly_when_the_note_drifts():
    import importlib
    hp = importlib.import_module("nethack_prime_agent")

    with pytest.raises(RuntimeError, match="stale"):
        hp._restore_baseline_coord_note(b"# NetHack\nsome other doc\n")


def test_launcher_tier_mapping_matches_tool_tiers_toml():
    """launch_cell.sh duplicates the [human] mapping on purpose (no TOML
    parsing in bash); this is the pin that keeps the two in sync."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    tiers = (root / "tools/cli_harness_eval/configs/tool_tiers.toml").read_text()
    launcher = (root / "tools/cli_harness_eval/launch_cell.sh").read_text()
    for flag in ("netplay_telemetry", "melee_hints", "skill_doc_coords"):
        assert f"{flag} = true" in tiers          # [human] section lists it
        assert f"{flag} = false" in tiers         # [base] section lists it
        assert flag in launcher                   # launcher expands it
