"""Tests for the vendored NetPlay skill layer (skill_set="netplay_true").

Upstream: github.com/CommanderCero/NetPlay @ 6acb90d865411f28d440e042372d9034f971e54a
See environments/nethack/vendor/PROVENANCE.md.
"""

import inspect
import pathlib
import sys

import pytest

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.tools.skills import SkillResult, registry  # noqa: E402
from nethack_harness.tools import netplay_true as npt_module  # noqa: E402


# --------------------------------------------------------------------------
# Regression: the registry must not drop closed-loop bookkeeping.
# --------------------------------------------------------------------------

def test_registry_preserves_pre_executed_when_stripping_unknown_args():
    """A closed-loop skill called with a stray arg must not be re-executed.

    `SkillRegistry.call` filters kwargs it cannot pass through, and used to
    rebuild the SkillResult with only four fields -- silently dropping
    `pre_executed`/`pre_reward`/`final_obs`/`pre_terminated`/`pre_truncated`.
    For a skill that already stepped the env (every NetPlay skill does), losing
    `pre_executed` makes the harness replay `actions` on top of the steps the
    skill already took.
    """
    sentinel = object()

    @registry.register("_test_closed_loop", schema={"description": "", "parameters": {}})
    def _skill(env, obs):
        return SkillResult(
            actions=[],
            feedback="done",
            pre_executed=True,
            pre_reward=3.5,
            final_obs=sentinel,
            pre_terminated=True,
            pre_truncated=True,
        )

    try:
        res = registry.call("_test_closed_loop", None, None, bogus_arg=1)
        assert res.pre_executed is True
        assert res.pre_reward == 3.5
        assert res.final_obs is sentinel
        assert res.pre_terminated is True
        assert res.pre_truncated is True
        assert "ignored unknown args" in res.feedback
    finally:
        registry._skills.pop("_test_closed_loop", None)
        registry._schemas.pop("_test_closed_loop", None)


# --------------------------------------------------------------------------
# The vendored action surface.
# --------------------------------------------------------------------------

def test_vendored_surface_matches_upstream_repository():
    """The exposed skill list must equal upstream netplay/__init__.py:9-18.

    Upstream builds SkillRepository([*ALL_COMMAND_SKILLS, set_avoid_monster_flag,
    melee_attack, explore_level, move_to, go_to, press_key, type_text]) =
    24 command skills + 7 named = 31.
    """
    from nethack_harness.tools import netplay_true as npt

    names = set(npt.NETPLAY_TRUE_SKILL_NAMES)
    assert len(names) == 31, sorted(names)

    # The seven explicitly-listed skills.
    for expected in ("set_avoid_monster_flag", "melee_attack", "explore_level",
                     "move_to", "go_to", "press_key", "type_text"):
        assert expected in names

    # Skills upstream defines but deliberately leaves OUT of the exposed set
    # (commented out in ALL_COMMAND_SKILLS, or internal-only helpers).
    for excluded in ("search", "open", "close", "take_off_all",
                     "explore", "search_room", "open_door", "open_neighbor_doors"):
        assert excluded not in names, f"{excluded} should not be exposed"


def test_skill_set_netplay_true_resolves_and_is_isolated():
    """`netplay_true` exposes exactly the np_ tools; other sets are untouched."""
    from nethack_harness.helpers import _build_skill_adapter_callables
    from nethack_harness.tools import netplay_true as npt  # noqa: F401  (registers)

    true_names = {f.__name__ for f in _build_skill_adapter_callables("netplay_true")}
    assert len(true_names) == 31
    assert all(n.startswith("np_") for n in true_names)

    # The pre-existing 18-tool `netplay` set must be unchanged, and no np_ tool
    # may leak into it or into `full`.
    netplay_names = {f.__name__ for f in _build_skill_adapter_callables("netplay")}
    assert len(netplay_names) == 18
    assert not any(n.startswith("np_") for n in netplay_names)

    full_names = {f.__name__ for f in _build_skill_adapter_callables("full")}
    assert not any(n.startswith("np_") for n in full_names), (
        "vendored NetPlay skills leaked into the default 'full' action surface"
    )


def test_boulders_resolve_to_rock_class():
    """G.BOULDERS drives the walkable mask, so a wrong table breaks pathfinding."""
    from nethack_harness.tools import netplay_true  # noqa: F401  (sys.path)
    import netplay.nethack_utils.glyphs as G
    import nle.nethack as nh

    otyps = [b - nh.GLYPH_OBJ_OFF for b in G.BOULDERS]
    assert otyps, "no boulder glyphs resolved"
    for o in otyps:
        assert ord(nh.objclass(o).oc_class) == nh.ROCK_CLASS
    assert nh.objdescr.from_idx(otyps[0]).oc_name == "boulder"


# --------------------------------------------------------------------------
# End-to-end against a real engine.
# --------------------------------------------------------------------------

def _fresh_env():
    from nethack_core.env import NetHackCoreEnv
    from nethack_harness.tools import netplay_true as npt

    npt.reset_agent_cache()
    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(42, 42)
    env.reset()
    return env


def test_explore_level_moves_the_game():
    """NetPlay's explore_level must actually drive the engine, not no-op."""
    env = _fresh_env()
    try:
        before = env.last_observation[env.observation_keys.index("blstats")]
        start = (int(before[0]), int(before[1]), int(before[20]))

        res = registry.call("np_explore_level", env, None)

        assert res.pre_executed is True, "closed-loop skill must report pre_executed"
        after = res.final_obs.blstats
        end = (int(after[0]), int(after[1]), int(after[20]))
        assert end[2] > start[2], f"game time did not advance: {start} -> {end}"
        assert end[:2] != start[:2], f"hero did not move: {start} -> {end}"
    finally:
        env.close()


def test_melee_attack_pursues_and_kills():
    """NetPlay's melee_attack is pursue-until-dead, not a directional bump."""
    import nle.nethack as nh
    from nethack_harness.tools import netplay_true as npt

    env = _fresh_env()
    try:
        def hostiles(agent):
            return [
                (int(g), p)
                for g, p in agent.current_level.get_monsters()
                if not nh.glyph_is_pet(int(g))
            ]

        target = None
        for _ in range(8):
            registry.call("np_explore_level", env, None)
            found = hostiles(npt.get_agent(env))
            if found:
                target = found[0]
                break
        if target is None:
            pytest.skip("no hostile encountered on this seed within the budget")

        glyph, _ = target
        killed = False
        for _ in range(12):
            agent = npt.get_agent(env)
            here = [p for g, p in hostiles(agent) if g == glyph]
            if not here:
                killed = True
                break
            pos = here[0]
            res = registry.call(
                "np_melee_attack", env, None, x=int(pos.x), y=int(pos.y)
            )
            assert res.pre_executed is True
            if "Killed the target" in res.feedback:
                killed = True
                break

        assert killed, "melee_attack never removed the target"
    finally:
        env.close()


def test_press_key_and_type_text_reach_the_engine():
    env = _fresh_env()
    try:
        res = registry.call("np_press_key", env, None, key="i")
        assert res.pre_executed is True
        # ESC out of whatever the inventory opened.
        registry.call("np_press_key", env, None, key="esc")

        res = registry.call("np_type_text", env, None, text="12")
        assert res.pre_executed is True
    finally:
        env.close()


def test_set_avoid_monster_flag_toggles_agent_state():
    from nethack_harness.tools import netplay_true as npt

    env = _fresh_env()
    try:
        assert npt.get_agent(env).avoid_monsters is False
        registry.call("np_set_avoid_monster_flag", env, None, value=True)
        assert npt.get_agent(env).avoid_monsters is True
    finally:
        env.close()


def test_move_to_reaches_a_reachable_tile():
    from nethack_harness.tools import netplay_true as npt

    env = _fresh_env()
    try:
        agent = npt.get_agent(env)
        sx, sy = agent.blstats.x, agent.blstats.y
        # Pick a reachable tile a few steps away using NetPlay's own BFS.
        candidates = [
            (x, y, d)
            for (x, y, d) in agent.last_bfs_results.bfs_results.get_reachable_tiles()
            if 2 <= d <= 6
        ]
        if not candidates:
            pytest.skip("no reachable tile in range on this seed")
        tx, ty, _ = candidates[0]

        res = registry.call("np_move_to", env, None, x=int(tx), y=int(ty))
        assert res.pre_executed is True
        agent = npt.get_agent(env)
        assert (agent.blstats.x, agent.blstats.y) != (sx, sy), "move_to did not move"
    finally:
        env.close()


def test_popup_guard_fires_with_zero_engine_steps():
    """Entry-time `fail_on_popup` must see a real popup and refuse to step.

    `NetPlayEngineEnv._current_core_obs` used to rebuild its observation from
    `NetHackCoreEnv.last_observation`, which is filtered to `observation_keys` --
    and `misc` (the in_yn_function/in_getlin/xwaitforspace flags `waiting_for_yn`
    etc. read) is not in the default key set. That stamped `misc=None` into
    `last_raw` on every `get_agent()` refresh, so the popup guard read False even
    with a real `[yn]` prompt open, and a skill's first keystroke (e.g. `move_to`'s
    first leg) landed in the prompt instead -- `y`/`n` are also compass keys.

    Open a real prompt with a raw keystroke sequence (outside any netplay skill),
    then call a `@fail_on_popup`-decorated skill and assert it fired the guard
    without consuming a single engine turn.
    """
    env = _fresh_env()
    try:
        # '#pray<CR>' reaches "Are you sure you want to pray? [yn] (n)" without
        # answering it -- a real, deterministic in_yn_function prompt.
        for c in "#pray":
            env.step(ord(c))
        obs, _, _, _, _ = env.step(ord("\r"))
        msg = bytes(obs.message).split(b"\x00", 1)[0]
        assert b"Are you sure you want to pray" in msg, "setup did not open a yn prompt"
        before_turn = int(obs.blstats[20])
        # A reachable adjacent tile (NW leg = 'y', a compass key AND the yn-prompt
        # "yes" answer) -- exactly the collision the finding describes.
        tx, ty = int(obs.blstats[0]) - 1, int(obs.blstats[1]) - 1

        res = registry.call("np_move_to", env, None, x=tx, y=ty)

        assert res.pre_executed is True
        assert "Cannot handle the popup" in res.feedback
        after = env.last_observation[env.observation_keys.index("blstats")]
        assert int(after[20]) == before_turn, "guard should consume zero engine turns"
        after_msg = bytes(env._last_observation.message).split(b"\x00", 1)[0]
        assert b"Are you sure you want to pray" in after_msg, (
            "the pending prompt must still be open -- nothing answered it"
        )
    finally:
        env.close()


def test_optional_skill_params_are_callable_without_them():
    """Upstream-optional skill params must publish as optional, not required.

    `down()`/`up()`/`pickup()`/`loot()`/`offer()` mean "act where I stand" when
    called with no (x, y); the twelve `create_inventory_command` skills
    (eat/drink/wield/...) mean "open the menu" when called with no `item_letter`;
    `rest()` defaults to 5 turns. `_schema_for` used to describe these as
    "(optional)" in prose but never emit a `"default"` key, and
    `_make_skill_adapter` treats a missing `"default"` as `inspect.Parameter.empty`
    -- i.e. required -- so the no-arg forms were unreachable through the
    published tool surface.
    """
    from nethack_harness.helpers import _build_skill_adapter_callables

    callables = {f.__name__: f for f in _build_skill_adapter_callables("netplay_true")}
    for name, upstream_optional_params in (
        ("np_down", {"x", "y"}),
        ("np_up", {"x", "y"}),
        ("np_pickup", {"x", "y"}),
        ("np_loot", {"x", "y"}),
        ("np_offer", {"x", "y"}),
        ("np_eat", {"item_letter"}),
        ("np_drop", {"item_letter"}),
        ("np_rest", {"count"}),
    ):
        fn = callables[name]
        sig = inspect.signature(fn)
        for pname in upstream_optional_params:
            assert sig.parameters[pname].default is None, (
                f"{name}({pname}=...) must be optional (default None), "
                f"got {sig.parameters[pname].default!r}"
            )
        # The no-arg call itself must be constructible (no TypeError binding).
        sig.bind()

    env = _fresh_env()
    try:
        agent = npt_module.get_agent(env)
        sx, sy = agent.blstats.x, agent.blstats.y
        res = registry.call("np_down", env, None)
        assert res.pre_executed is True
        # `down()` means "descend where I stand" -- it must not have required
        # us to pass a position, and it must not have moved the hero to reach
        # for one either (create_position_command only moves when x/y given).
        agent = npt_module.get_agent(env)
        assert (agent.blstats.x, agent.blstats.y) == (sx, sy)
    finally:
        env.close()


def test_raw_keystroke_surface_present_in_netplay_true_absent_in_netplay():
    """The literal `move` gate holds, but netplay_true is NOT gate-equivalent.

    No `move` tool is published in either set. But `netplay_true` (faithful to
    upstream) publishes `press_key`/`type_text`, which pass raw keystrokes
    straight to the engine -- `RawKeyPress.parse("h")` is the keystroke for a
    westward step, so `np_type_text(text="hhhh")` is an unrestricted walk. This
    is correct for fidelity to upstream, not a bug, but the two netplay sets
    must not be mistaken for gate-equivalent: pin the asymmetry explicitly.
    """
    from nethack_harness.helpers import _build_skill_adapter_callables

    netplay_true_names = {f.__name__ for f in _build_skill_adapter_callables("netplay_true")}
    netplay_names = {f.__name__ for f in _build_skill_adapter_callables("netplay")}

    assert "move" not in netplay_true_names
    assert "move" not in netplay_names

    assert "np_press_key" in netplay_true_names
    assert "np_type_text" in netplay_true_names
    assert "press_key" not in netplay_names
    assert "type_text" not in netplay_names


def test_scout_tiles_accumulate_for_closed_loop_netplay_true_skills():
    """`scout_reward` must not be structurally zero for `netplay_true`.

    Every `np_` skill returns `pre_executed=True` with `action_indices = []`, so
    `nethack.py`'s step loop (which is what populates `state["scout_tiles_seen"]`)
    never runs for them. `run_netplay_skill` must report the tiles its own
    internal step loop revealed so `nethack.py` can still accumulate them --
    without changing behaviour for the hand-written `netplay` set's own
    closed-loop skill (`explore_and_descend`), which does not opt in.
    """
    from nethack_harness.tools.netplay_true import run_netplay_skill, NETPLAY_SKILL_REPOSITORY

    npt_module.reset_agent_cache()
    from nethack_core.env import NetHackCoreEnv
    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(42, 42)
    env.reset()
    try:
        skill = NETPLAY_SKILL_REPOSITORY.skills["explore_level"]
        result = run_netplay_skill(env, skill, {})
        assert result.pre_executed is True
        assert getattr(result, "pre_visible_obs", None), (
            "run_netplay_skill must report per-step observations so nethack.py "
            "can populate scout_tiles_seen for a closed-loop np_ skill"
        )
        # explore_level took at least one real engine step on a fresh level.
        assert len(result.pre_visible_obs) >= 1
    finally:
        env.close()


def test_scout_reward_nonzero_for_netplay_true_end_to_end():
    """End-to-end: an `np_` closed-loop skill call must grow scout_tiles_seen.

    Drives the real `nethack.py` env_response path (`_apply_tool_call`), not
    just `run_netplay_skill` in isolation, so the `state["scout_tiles_seen"]`
    back-fill added to the `pre_executed` branch is exercised the same way a
    real rollout would exercise it.
    """
    import asyncio

    import nethack as m
    import verifiers as vf

    env = m.load_environment(task_spec="full_nle", skill_set="netplay_true",
                              n_examples=1, explicit_seeds=[42],
                              character="Val-hum-neu-fem")
    state = vf.State({"task": {"tier": "full_nle", "seed": 42}, "info": {}})
    asyncio.run(env.setup_state(state))

    before = len(state["scout_tiles_seen"])
    asyncio.run(env._apply_tool_call(state, "np_explore_level", {}))
    after = len(state["scout_tiles_seen"])

    assert after > before, (
        f"scout_tiles_seen did not grow from an np_ closed-loop skill call "
        f"({before} -> {after})"
    )


def test_scout_reward_unchanged_for_netplay_explore_and_descend():
    """The hand-written `netplay` set's own closed-loop skill must not change.

    `explore_and_descend` is `netplay`'s dominant tool and is ALSO
    `pre_executed=True`; it never sets `pre_visible_obs`, so the new back-fill
    branch in `nethack.py` must be a no-op for it -- `scout_tiles_seen` stays
    exactly as unpopulated as before this fix (a pre-existing, separate
    limitation of the `netplay` set, not something this task changes).
    """
    import asyncio

    import nethack as m
    import verifiers as vf

    env = m.load_environment(task_spec="full_nle", skill_set="netplay",
                              n_examples=1, explicit_seeds=[42],
                              character="Val-hum-neu-fem")
    state = vf.State({"task": {"tier": "full_nle", "seed": 42}, "info": {}})
    asyncio.run(env.setup_state(state))

    before = len(state["scout_tiles_seen"])
    asyncio.run(env._apply_tool_call(state, "explore_and_descend", {}))
    after = len(state["scout_tiles_seen"])

    assert after == before, (
        f"netplay's explore_and_descend must not gain scout-tile bookkeeping "
        f"as a side effect of the netplay_true fix ({before} -> {after})"
    )


def test_explore_level_plus_down_descends_past_dlvl_1():
    """The capability this port exists to restore.

    Half of a prior experiment's rollouts never left dungeon level 1. NetPlay
    splits this into two LLM decisions -- explore_level until a down-staircase
    is visible, then down(x, y), which pathfinds to the stairs and descends --
    where our own `explore_and_descend` fuses them and caps its search.
    """
    import netplay.nethack_utils.glyphs as G
    from nethack_harness.tools import netplay_true as npt

    env = _fresh_env()
    try:
        start_depth = int(npt.get_agent(env).blstats.depth)

        for _ in range(25):
            agent = npt.get_agent(env)
            stairs = list(agent.current_level.get_features([G.SS.S_dnstair]))
            if stairs:
                _, pos = stairs[0]
                res = registry.call(
                    "np_down", env, None, x=int(pos.x), y=int(pos.y)
                )
                assert res.pre_executed is True
                if int(npt.get_agent(env).blstats.depth) > start_depth:
                    return
            else:
                registry.call("np_explore_level", env, None)

        pytest.fail(
            f"never descended below dlvl {start_depth} within the explore budget"
        )
    finally:
        env.close()
