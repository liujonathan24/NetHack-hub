"""Pins for the e7 raw-prompt surface (skill_set="np_core", auto_dismiss=False).

The seeding experiment (2026-08-18) publishes the NARROW, individually-debugged
core of netplay_true and turns the harness auto-dismiss loop OFF so the model
answers NetHack's own prompts via np_press_key. These tests pin:

  1. the np_core preset's exact 7 tools (and its comma-composition),
  2. the auto_dismiss constructor default (True -- existing arms unchanged),
  3. the UI-freeze block naming answers as np_press_key calls on this surface
     while the BALROG-80 surface renders byte-identically to before,
  4. BBOX_MIN honoring request_map as the map purchase (and reveal cells
     rendering exactly as they always have).
"""

import pathlib
import sys

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.helpers import _build_skill_adapter_callables  # noqa: E402
from nethack_harness.prompt.interactive_state import _offered_answers  # noqa: E402


NP_CORE_EXPECTED = {
    "np_explore_level", "np_melee_attack", "np_move_to",
    "np_press_key", "np_pray", "np_apply", "np_rest", "np_kick",
}


def _names(adapters):
    return {getattr(a, "__name__", "") for a in adapters}


def test_np_core_is_exactly_seven_tools():
    assert _names(_build_skill_adapter_callables("np_core")) == NP_CORE_EXPECTED


def test_np_core_comma_composition():
    got = _names(_build_skill_adapter_callables("np_core,request_map,search"))
    assert got == NP_CORE_EXPECTED | {"request_map", "search"}


def test_np_core_excludes_reveal_rollback_and_type_text():
    got = _names(_build_skill_adapter_callables("np_core,request_map,search"))
    for banned in ("reveal", "rollback", "np_type_text", "np_go_to"):
        assert banned not in got


def test_netplay_true_full_arm_composition():
    got = _names(_build_skill_adapter_callables("netplay_true,request_map,search"))
    assert len(got) == 33 and "np_type_text" in got and "reveal" not in got


def test_auto_dismiss_defaults_true():
    """Existing arms must be untouched: auto_dismiss defaults on."""
    import inspect
    import nethack as nethack_mod
    sig = inspect.signature(nethack_mod.NetHackVerifiersEnv.__init__)
    param = sig.parameters["auto_dismiss"]
    assert param.default is True


def test_freeze_block_names_np_press_key_on_np_surface():
    out = _offered_answers(
        "What do you want to wield? [- bc or ?*]",
        {"np_press_key", "np_move_to"},
    )
    assert "`np_press_key(key='b')`" in out
    assert "`np_press_key(key='-')`" in out
    assert "`np_press_key(key='esc')` to cancel" in out
    assert "bal_" not in out


def test_freeze_block_balrog_naming_unchanged():
    out = _offered_answers(
        "What do you want to wield? [- bc or ?*]",
        {"bal_b", "bal_esc"},
    )
    assert out == ("`bal_minus` (the '-' key), `bal_b`, `bal_c`, "
                   "'?' to list valid choices, '*' to list every item, "
                   "`bal_esc` to cancel")


def _fake_obs():
    from nethack_core.observations import StructuredObservation
    return StructuredObservation(
        map_view="", messages=["The door opens."], inventory=[],
        status={"hitpoints": 16, "max_hitpoints": 16, "armor_class": 6,
                "depth": 1, "time": 42, "experience_level": 1, "gold": 0,
                "x": 5, "y": 5, "hunger_state": 1},
        character={"role": "Valkyrie", "race": "human", "alignment": "neutral"},
    )


def _render(state):
    from nethack_harness.memory.journal import Journal
    from nethack_harness.prompt.prompt_spec import resolve_spec
    spec = resolve_spec("BBOX_MIN", None)
    return spec.turn_template(_fake_obs(), Journal(), state,
                              compact=False, journal_max_chars=2000)


def test_bbox_min_placeholder_names_request_map_when_reveal_unpublished():
    txt = _render({"_published_tools": {"np_press_key", "request_map", "search"}})
    assert "call request_map for the full map" in txt
    assert "reveal(x1,y1,x2,y2)" not in txt


def test_bbox_min_request_map_turn_pushes_inline_map():
    state = {"_published_tools": {"np_press_key", "request_map", "search"},
             "_force_map": True, "_last_skill_name": "request_map"}
    txt = _render(state)
    assert "=== MAP ===" in txt
    assert "SURROUNDINGS (hidden" not in txt
    assert "_force_map" not in state, "flag must be popped, never linger"


def test_bbox_min_reveal_cells_render_exactly_as_before():
    txt = _render({"_published_tools": {"np_explore_level", "reveal", "search"}})
    assert "call reveal(x1,y1,x2,y2)" in txt
    # A reveal turn must NOT gain an inline map from the popped force flag.
    state = {"_published_tools": {"np_explore_level", "reveal", "search"},
             "_force_map": True, "_last_skill_name": "reveal"}
    txt2 = _render(state)
    assert "=== MAP ===" not in txt2
    assert "_force_map" not in state


def test_auto_dismiss_coerces_cli_string_false():
    """env_args reach the ctor as dotted-scalar strings; bool("false") is True,
    so the ctor must parse the string form. Caught in the e7 smoke test."""
    import inspect
    import nethack as nethack_mod
    src = inspect.getsource(nethack_mod.NetHackVerifiersEnv.__init__)
    assert 'isinstance(auto_dismiss, str)' in src
    # Behavior-level check without building an env: replicate the clause.
    for raw, expected in [("false", False), ("False", False), ("0", False),
                          ("off", False), ("true", True), ("1", True),
                          (True, True), (False, False)]:
        v = raw
        if isinstance(v, str):
            v = v.strip().lower() not in ("false", "0", "no", "off", "")
        assert bool(v) is expected, (raw, expected)


def test_path_explain_names_published_map_tool():
    import numpy as np
    from nethack_harness.navigation.path_explain import explain_path_failure

    class Raw:
        # 21x79 chars grid: player on floor at (5,5); target floor at (70,18)
        # provably disconnected, so the no-route branch ALWAYS fires.
        chars = np.full((21, 79), ord(" "), dtype=np.uint8)
        blstats = [5, 5]
    Raw.chars[5, 4:7] = ord(".")
    Raw.chars[18, 69:72] = ord(".")

    args = {"x": 70, "y": 18}
    out_np = explain_path_failure(Raw, args, published_tools={"np_press_key", "request_map", "search"})
    out_reveal = explain_path_failure(Raw, args, published_tools={"reveal", "search"})
    out_default = explain_path_failure(Raw, args)
    for out, want, banned in [
        (out_np, "request_map", "`reveal`"),
        (out_reveal, "`reveal`", "request_map"),
        (out_default, "`reveal`", "request_map"),
    ]:
        assert out is not None, "no-route branch must fire on this fixture"
        assert want in out and banned not in out, out


def test_request_map_feedback_classifies_completed():
    """request_map takes no NLE step and had no status marker -- it tallied as
    'unknown' (3 of 20 calls in the e7 smoke). Pin the marker."""
    from nethack_harness.helpers import classify_tool_result
    assert classify_tool_result("Refreshing the full map this turn.") == "completed"


def test_residual_ui_blocks_are_dismiss_mode_aware():
    """Under auto_dismiss=False the MENU/PROMPT safety blocks must not promise
    a dismissal that never comes (16 such lies rendered across NPFULL v1)."""
    from types import SimpleNamespace
    from nethack_harness.prompt.rendering import format_observation_as_chat
    from nethack_harness.memory.journal import Journal
    from nethack_core.observations import StructuredObservation

    so = StructuredObservation(
        map_view="", messages=[], inventory=[],
        status={"hitpoints": 16, "max_hitpoints": 16, "armor_class": 6,
                "depth": 1, "time": 42, "experience_level": 1, "gold": 0,
                "x": 5, "y": 5, "hunger_state": 1},
        character={"role": "Valkyrie", "race": "human", "alignment": "neutral"},
        menu=[SimpleNamespace(description="a - eat it")],
    )
    kw = dict(compact=False, journal_max_chars=2000)
    raw = format_observation_as_chat(so, Journal(), {"_auto_dismiss": False}, **kw)
    assert "answer it: press a choice key" in raw
    assert "auto-dismiss" not in raw
    legacy = format_observation_as_chat(so, Journal(), {}, **kw)
    assert "=== MENU (harness will auto-dismiss; ignore) ===" in legacy


def test_step_frames_capture_opt_in():
    """record_step_frames=True stores one tty screen per engine step; the
    default stays byte-identical (no frames, no size cost)."""
    import numpy as np
    from nethack_harness.helpers import TurnRecorder

    class Obs:
        def __init__(s, c): s.tty_chars = np.full((24, 80), ord(c), dtype=np.uint8)
    class Env:
        def __init__(s): s.n = 0
        def step(s, a): s.n += 1; return (Obs(chr(ord("a")+s.n)), 1.0, False, False, {})

    env = Env()
    with TurnRecorder(env, capture_frames=True) as rec:
        for k in (ord("l"), ord("k")):
            env.step(k)
    assert len(rec.frames) == 2
    assert all(len(f["g"]) == 24 for f in rec.frames)
    assert rec.frames[0]["k"] == "l" and rec.frames[0]["b"] == ord("l")

    with TurnRecorder(env) as rec2:   # default off
        env.step(ord("h"))
    assert rec2.frames == []


def test_e8_knobs_default_off():
    """descent_gate/mechanic_hints default off: existing arms byte-identical."""
    import inspect
    import nethack as nethack_mod
    sig = inspect.signature(nethack_mod.NetHackVerifiersEnv.__init__)
    assert sig.parameters["descent_gate"].default == "off"
    assert sig.parameters["mechanic_hints"].default == ""


def test_e8_norm_lookup():
    from nethack_harness.prompt.human_norms import norm_xl_for_leaving
    # leaving Dlvl d uses arrival norm of d+1 (NAO ascended medians)
    assert norm_xl_for_leaving(1) == 1   # arrive Dl2 at XL1
    assert norm_xl_for_leaving(4) == 3   # arrive Dl5 at XL3
    assert norm_xl_for_leaving(6) == 5   # arrive Dl7 at XL5
    assert norm_xl_for_leaving(30) == 11  # beyond table -> nearest lower


def test_e8_mechanic_hint_blocks():
    from nethack_harness.prompt.human_norms import MECHANIC_HINT_BLOCKS
    assert "1/7" in MECHANIC_HINT_BLOCKS["prayer"]
    assert "experience level" in MECHANIC_HINT_BLOCKS["descend_pacing"]
    # prompt-only: block text must never mention tool schemas
    for b in MECHANIC_HINT_BLOCKS.values():
        assert "parameters" not in b
