"""Zombie detection: the rollout is over, or broken, and nobody stopped it.

Documented precedent on this project -- a rollout whose MCP server died had the
agent rebuild a counterfeit NetHack backend inside its own IPython kernel and
carry on "playing" it. Nothing raised. Under the old 200-call cap that wasted a
cell; with the cap removed it can burn the full 7200s rollout timeout producing
fiction, so these checks are what replaces the cap as a safety net.

Every check must ALSO stay quiet on a rollout that simply ended, which is the
failure mode that would make the whole guard useless.
"""
import importlib.util, pathlib

_MOD = pathlib.Path(__file__).resolve().parent / "watch_runaway.py"
_spec = importlib.util.spec_from_file_location("watch_runaway", _MOD)
wr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(wr)


def turn(t, hp=20, clock=None, calls=("np_move_to",), status="ok", obs=""):
    return {"turn": t, "hp": hp, "dlvl": 3, "max_dlvl_reached": 3,
            "status": {"time": t if clock is None else clock},
            "rendered_user_message": obs,
            "tool_calls": [{"name": c} for c in calls],
            "tool_results": [{"name": c, "status": status} for c in calls]}


def test_a_healthy_rollout_is_not_flagged():
    assert wr.zombie_checks([turn(i) for i in range(30)]) == []


def test_a_rollout_that_dies_on_its_last_turn_is_not_flagged():
    """Dying IS how a game ends. Flagging it would fire on every rollout."""
    rows = [turn(i) for i in range(20)] + [turn(20, hp=0)]
    assert wr.zombie_checks(rows) == []


def test_dead_but_still_taking_turns():
    rows = [turn(i) for i in range(10)] + [turn(10, hp=0)] + [turn(i) for i in range(11, 25)]
    assert any("died at turn 10" in w for w in wr.zombie_checks(rows))


def test_death_screen_followed_by_more_turns():
    rows = ([turn(i) for i in range(10)]
            + [turn(10, obs="You die... Do you want your possessions identified?")]
            + [turn(i) for i in range(11, 25)])
    assert any("death screen" in w for w in wr.zombie_checks(rows))


def test_frozen_in_game_clock():
    """A counterfeit backend can fake observations but not advance NetHack's clock."""
    rows = [turn(i) for i in range(10)] + [turn(i, clock=500) for i in range(10, 30)]
    assert any("clock frozen" in w for w in wr.zombie_checks(rows))


def test_counterfeit_game_stops_calling_the_tool_server():
    rows = [turn(i) for i in range(10)] + [
        turn(i, clock=500, calls=("ipython",)) for i in range(10, 30)]
    flags = wr.zombie_checks(rows)
    assert any("no engine call" in w for w in flags)
    assert any("clock frozen" in w for w in flags)


def test_every_recent_tool_result_failing():
    rows = [turn(i) for i in range(10)] + [turn(i, status="failed") for i in range(10, 30)]
    assert any("tool results failed" in w for w in wr.zombie_checks(rows))


def test_a_short_rollout_is_never_flagged():
    """Startup has few turns and no history; it must not look like a zombie."""
    assert wr.zombie_checks([turn(0), turn(1)]) == []
