"""Environment problems found by models playing full games, each pinned where it was fixed."""

from types import SimpleNamespace

import numpy as np

from nethack_harness.tools import netplay_true as npt
from nethack_harness.prompt import rendering, tty_screen as ts
from netplay.nethack_agent import skills, tracking
from netplay.nethack_agent.agent import NetHackAgent
from netplay.nethack_agent.pathfinding import LevelPathfinder
from netplay.nethack_utils import screen_symbols as SS


def _tty(rows, cursor, misc=(0, 0, 1), blstats=None):
    chars = np.full((24, 80), ord(" "), np.uint8)
    for r, (col, text) in rows.items():
        chars[r, col:col + len(text)] = np.frombuffer(text.encode(), np.uint8)
    return SimpleNamespace(tty_chars=chars, tty_cursor=np.array(cursor), misc=np.array(misc),
                           blstats=np.zeros(27, int) if blstats is None else blstats)


def test_full_screen_text_window_keeps_its_first_column():
    obs = _tty({22: (0, "Agent the Valkyrie's attributes:"), 23: (1, "--More--")}, (23, 9))
    assert ts.find_window(obs).lines == ["Agent the Valkyrie's attributes:"]


def test_game_over_shows_the_last_live_status_and_the_cause():
    import nethack

    live = np.zeros(27, int)
    live[[0, 1, 10, 11, 12, 20]] = [40, 7, 2, 16, 4, 125]
    log = ts.CallScreenLog(None, baseline_obs=SimpleNamespace(blstats=live))
    log._keep_live(SimpleNamespace(blstats=np.zeros(27, int)))
    so = SimpleNamespace(status={"depth": 0, "hitpoints": 0}, messages=["You die..."])
    top_ten = _tty({5: (27, "1        154  Agent-Val-Hum-Fem-Neu died in The Gnomi"), 6: (0, "sh Mines on"),
                    7: (40, "level 4.  Killed by a crossbow bolt.")}, (9, 23))
    state = {"structured_obs": so, "_call_log": log, "raw_obs": top_ten}
    nethack._keep_live_status(state)
    assert (so.status["hitpoints"], so.status["max_hitpoints"], so.status["depth"], so.status["time"]) == (0, 16, 4, 125)
    assert rendering._death_cause(so, state) == "Killed by a crossbow bolt."


def _agent(features, hero, time=1):
    a = NetHackAgent.__new__(NetHackAgent)
    level = tracking.Level(0, 1)
    level.features[:] = features
    a.data = SimpleNamespace(blstats=SimpleNamespace(x=hero[0], y=hero[1], time=time), current_level=level)
    a.last_bfs_results = LevelPathfinder.from_level(hero[0], hero[1], level, False)
    return a


def test_a_run_or_travel_is_not_a_teleport():
    a = _agent(SS.S_room, (30, 4), time=6)
    walked = tracking.TeleportEvent((26, 4), (30, 4))
    assert a._walked(SimpleNamespace(time=2), walked)
    assert not a._walked(SimpleNamespace(time=5), tracking.TeleportEvent((10, 4), (30, 4)))
    assert not a._walked(SimpleNamespace(time=6), walked)


def test_live_spell_menu_stays_open_for_its_answer():
    from nethack_core.env import NetHackCoreEnv
    from nethack_harness.tools.skills import registry

    npt.reset_agent_cache()
    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(202, 202)
    env.reset(character="Wiz-elf-cha-mal")
    registry.call("np_press_key", env, None, key="ESC")
    res = registry.call("np_press_key", env, None, key="Z")
    screen = "\n".join(ts.tty_rows(res.final_obs))
    assert "Choose which spell to cast" in screen and "a - force bolt" in screen
    env.close()

