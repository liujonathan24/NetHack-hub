"""Variant BBOX_BJSON_OD: BBOX_MIN delivery with the bought look upgraded.

Pins the treatment's surgical scope (E15 on-demand x encoding cell): quiet
turns and reveal turns render byte-identical to BBOX_MIN; ONLY a request_map
turn differs, adding a `=== MAP (JSON) ===` block (the structured,
coordinate-addressable body) alongside the ASCII map. If a quiet turn ever
diverges, the arm is no longer "one treatment, one axis" and the comparison
against the frozen base control is void.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "environments" / "nethack"))

from nethack_harness.prompt.prompt_spec import VARIANT_REGISTRY  # noqa: E402


class _Obs:
    """The attribute surface format_observation_as_chat reads."""

    def __init__(self):
        self.status = {"hitpoints": 16, "max_hitpoints": 16, "armor_class": 6,
                       "depth": 1, "time": 7, "experience_level": 1, "gold": 0,
                       "x": 3, "y": 6}
        self.character = {"role": "valkyrie", "race": "human",
                          "alignment": "neutral"}
        self.inventory = [type("I", (), {"letter": "b",
                                         "description": "a +1 long sword"})()]
        self.under_player = "stairs UP (<)"
        self.adjacent = {"N": ".", "E": "."}
        self.messages = ["You see here a lichen corpse."]
        self.menu = []
        self.inventory_prompt = None


def _render(variant, state):
    tpl = VARIANT_REGISTRY[variant].turn_template
    return tpl(_Obs(), None, state, compact=True, journal_max_chars=2000)


def test_quiet_turn_byte_identical_to_bbox_min():
    st = {"_self_dispatch": True, "_last_skill_name": "np_move_to",
          "_published_tools": ("np_move_to", "request_map")}
    assert _render("BBOX_BJSON_OD", dict(st)) == _render("BBOX_MIN", dict(st))


def test_reveal_turn_byte_identical_to_bbox_min():
    st = {"_self_dispatch": True, "_last_skill_name": "reveal"}
    assert _render("BBOX_BJSON_OD", dict(st)) == _render("BBOX_MIN", dict(st))


# ---- request_map turn against the live engine ----------------------------- #

def _v0_env_and_state(variant):
    import nethack as m0
    import verifiers as vf

    env = m0.load_environment(
        task_spec="full_nle", skill_set="np_core,request_map,search",
        n_examples=1, explicit_seeds=[0], character="Val-hum-neu-fem",
        variant=variant,
    )
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    state.setdefault("trajectory", [])
    return env, state


def _request_map_obs(variant):
    env, state = _v0_env_and_state(variant)
    out = asyncio.run(env._apply_tool_call(state, "request_map", {}))
    if isinstance(out, list):  # chat-shaped content parts
        out = " ".join(p.get("text", "") for p in out if isinstance(p, dict))
    return str(out)


def test_request_map_returns_ascii_map_plus_json_body():
    text = _request_map_obs("BBOX_BJSON_OD")
    assert "=== MAP ===" in text, "ASCII map missing from the bought look"
    assert "=== MAP (JSON) ===" in text, "JSON body missing from the bought look"
    # The JSON body must be coordinate-addressable: the player record carries
    # an explicit position.
    assert '"player"' in text or '"x"' in text


def test_bbox_min_request_map_has_no_json_body():
    text = _request_map_obs("BBOX_MIN")
    assert "=== MAP ===" in text
    assert "=== MAP (JSON) ===" not in text
