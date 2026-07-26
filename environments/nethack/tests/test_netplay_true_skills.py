"""Tests for the vendored NetPlay skill layer (skill_set="netplay_true").

Upstream: github.com/CommanderCero/NetPlay @ 6acb90d865411f28d440e042372d9034f971e54a
See environments/nethack/vendor/PROVENANCE.md.
"""

import pathlib
import sys

import pytest

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.tools.skills import SkillResult, registry  # noqa: E402


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
