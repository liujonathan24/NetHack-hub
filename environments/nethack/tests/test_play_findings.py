"""Environment problems found by models playing full games, each pinned where it was fixed."""

from types import SimpleNamespace

import numpy as np

from nethack_harness.tools import netplay_true as npt
from nethack_harness.prompt import rendering, tty_screen as ts
from netplay.nethack_agent import skills, tracking
from netplay.nethack_agent.agent import NetHackAgent
from netplay.nethack_agent.pathfinding import LevelPathfinder
from netplay.nethack_utils import screen_symbols as SS


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
