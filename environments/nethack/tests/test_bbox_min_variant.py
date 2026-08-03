"""Variant BBOX_MIN: the quiet turn pushes feedback + STATUS and nothing else;
the entity/message sections are delivered only on `reveal` turns.

The economics this pins: under plain BBOX the coordinate feed (ADJACENT /
VISIBLE FEATURES / VISIBLE MONSTERS / MESSAGES) is pushed free on every turn,
so `reveal` competes with it and fires on ~2% of turns while the feed costs
~700 chars/turn forever. BBOX_MIN moves the whole feed behind `reveal`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "environments" / "nethack"))

from nethack_harness.prompt.prompt_spec import (  # noqa: E402
    VARIANT_REGISTRY,
    _BBOX_MIN_PLACEHOLDER,
)


class _Obs:
    """The attribute surface format_observation_as_chat reads."""

    def __init__(self):
        self.status = {"hitpoints": 16, "max_hitpoints": 16, "armor_class": 6,
                       "depth": 1, "time": 7, "experience_level": 1, "gold": 0,
                       "x": 3, "y": 6}
        self.character = {"role": "valkyrie", "race": "human", "alignment": "neutral"}
        self.inventory = [type("I", (), {"letter": "b", "description": "a +1 long sword"})()]
        self.under_player = "stairs UP (<)"
        self.adjacent = {"N": ".", "E": "."}
        self.messages = ["You see here a lichen corpse."]
        self.menu = []
        self.inventory_prompt = None


def _render(state):
    tpl = VARIANT_REGISTRY["BBOX_MIN"].turn_template
    return tpl(_Obs(), None, state, compact=True, journal_max_chars=2000)


WITHHELD = ("=== INVENTORY", "=== UNDER PLAYER", "=== ADJACENT",
            "=== VISIBLE FEATURES", "=== VISIBLE MONSTERS", "=== MESSAGES")


def test_quiet_turn_is_feedback_plus_status_only():
    text = _render({"_self_dispatch": True, "_last_skill_name": "np_move_to"})
    assert "=== STATUS ===" in text
    assert _BBOX_MIN_PLACEHOLDER in text
    for section in WITHHELD:
        assert section not in text, f"{section} leaked into a quiet turn"


def test_reveal_turn_delivers_the_withheld_sections():
    text = _render({"_self_dispatch": True, "_last_skill_name": "reveal"})
    assert "=== STATUS ===" in text
    for section in ("=== INVENTORY", "=== UNDER PLAYER", "=== ADJACENT",
                    "=== MESSAGES"):
        assert section in text, f"{section} missing from a reveal turn"
    # FEATURES/MONSTERS need raw_obs and are exercised in the live render;
    # their gate is the same `minimal` flag asserted above.


def test_first_turn_with_no_skill_history_is_quiet():
    # Turn 1 renders before any skill has run; it must not crash and must not
    # leak the sections.
    text = _render({"_self_dispatch": True})
    assert "=== STATUS ===" in text
    for section in WITHHELD:
        assert section not in text


def test_the_placeholder_names_what_reveal_buys():
    # The placeholder is the agent's only standing notice of the contract; if
    # it stops mentioning reveal and the withheld sections, the variant is
    # unlearnable.
    for token in ("reveal(x1,y1,x2,y2)", "ADJACENT", "VISIBLE FEATURES",
                  "VISIBLE MONSTERS", "MESSAGES"):
        assert token in _BBOX_MIN_PLACEHOLDER


def test_plain_bbox_is_unchanged_by_the_new_flag():
    # BBOX must keep pushing the feed — it is the baseline arm of the delivery
    # experiment and must not drift when BBOX_MIN was added.
    tpl = VARIANT_REGISTRY["BBOX"].turn_template
    text = tpl(_Obs(), None, {"_self_dispatch": True, "_last_skill_name": "np_move_to"},
               compact=True, journal_max_chars=2000)
    for section in ("=== INVENTORY", "=== UNDER PLAYER", "=== ADJACENT",
                    "=== MESSAGES"):
        assert section in text, f"BBOX lost its per-turn {section} block"


def test_game_over_still_renders_on_a_quiet_turn():
    # Death must never arrive silently, whatever the variant withholds.
    obs = _Obs()
    obs.status["hitpoints"] = 0
    tpl = VARIANT_REGISTRY["BBOX_MIN"].turn_template
    text = tpl(obs, None, {"_self_dispatch": True, "_last_skill_name": "np_move_to"},
               compact=True, journal_max_chars=2000)
    assert "GAME OVER" in text.upper() or "dead" in text.lower()
