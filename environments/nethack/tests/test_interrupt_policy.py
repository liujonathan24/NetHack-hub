"""The NetCore skill-interrupt policy (netplay agent.py), driven on synthetic observations."""

from types import SimpleNamespace

import numpy as np
import pytest

from nethack_core import glyphs as G
from nethack_harness.tools import netplay_true  # noqa: F401  (puts vendor/ on sys.path)
from netplay.core.agent_base import Step, StepData
from netplay.nethack_agent import tracking
from netplay.nethack_agent.agent import NetHackAgent, describe_level_change


def _mon_id(name):
    return next(i for i in range(G.NUMMONS) if G.monster_name(i) == name)


NEWT, GNOME, SHK, HERO = (G.GLYPH_MON_OFF + _mon_id(name)
                          for name in ("newt", "gnome", "shopkeeper", "valkyrie"))
KITTEN_PET = G.GLYPH_PET_OFF + _mon_id("kitten")
GEM = G.GLYPH_OBJ_OFF + 200  # any object glyph


def _set_obs(a, hero, things, depth=1, dnum=0, time=1):
    glyphs = np.full((21, 79), G.GLYPH_CMAP_OFF, dtype=np.int16)  # S_stone
    for (x, y), g in things.items():
        glyphs[y, x] = g
    glyphs[hero[1], hero[0]] = HERO
    a.last_observation = {"glyphs": glyphs}
    a.data.blstats = SimpleNamespace(x=hero[0], y=hero[1], depth=depth, level_number=depth,
                                     dungeon_number=dnum, time=time, hitpoints=16, max_hitpoints=16)


def _agent(start=None, descr=None, exempt=None, depth=1, dnum=0):
    """An agent with only the state the policy reads; tracking begins on `start` ({pos: glyph})."""
    a = NetHackAgent.__new__(NetHackAgent)
    a.last_interrupt = None
    a._tracked_monsters = []
    a._reported_last_call = []
    a.max_skill_gamesteps = 100
    a.data = SimpleNamespace(blstats=None)
    a.describe_tile = None if descr is None else (lambda x, y: descr.get((x, y)))  # farlook text
    _set_obs(a, (40, 10), start or {}, depth=depth, dnum=dnum)
    a._begin_monster_tracking(exempt)
    return a


def _interrupt(a, things, hero=(40, 10)):
    _set_obs(a, hero, things)
    return a._hostile_monster_interrupt()


@pytest.mark.parametrize("things,descr,reason", [
    ({(78, 10): NEWT}, None, None),
    ({(78, 10): NEWT, (45, 12): NEWT}, None,
     "A hostile newt came into view at (45,12), 5 squares away"),
    ({(x, 10): GEM for x in range(41, 46)}, None, None),
    ({(42, 10): KITTEN_PET, (43, 11): GNOME, (44, 9): SHK},
     {(42, 10): "tame kitten", (43, 11): "peaceful gnome", (44, 9): "peaceful Asidonhopo"}, None),
    ({(44, 9): SHK}, None, None),
    ({(44, 9): SHK}, {(44, 9): "Asidonhopo"},
     "A hostile Asidonhopo came into view at (44,9), 4 squares away"),
    ({(43, 11): GNOME}, {(43, 11): "gnome, leashed to you"},
     "A hostile gnome came into view at (43,11), 3 squares away"),
    ({(43, 11): NEWT}, {(43, 11): "corridor"},
     "A hostile newt came into view at (43,11), 3 squares away"),
], ids=["far-hostile", "near-hostile", "items", "pet-and-peacefuls", "always-peaceful-species",
        "angry-shopkeeper", "farlook-name", "farlook-describes-the-square"])
def test_what_appears_mid_call(things, descr, reason):
    assert _interrupt(_agent(descr=descr), things) == reason


def test_reported_monster_does_not_interrupt_the_next_calls_while_it_stays_put():
    a = _agent()
    assert _interrupt(a, {(45, 10): NEWT})
    a._end_monster_tracking()
    a._begin_monster_tracking()
    assert [_interrupt(a, {pos: NEWT}) for pos in [(45, 10), (45, 11), (46, 11)]] == [None] * 3
    a._end_monster_tracking()
    a._begin_monster_tracking()
    assert _interrupt(a, {(46, 11): NEWT}) is None


def test_monster_on_screen_at_call_start_is_not_new():
    assert _interrupt(_agent({(44, 10): NEWT}), {(44, 10): NEWT}) is None


def test_known_monster_interrupts_when_it_comes_into_range():
    a = _agent({(60, 10): NEWT})
    first = None
    for x in range(41, 61):
        if _interrupt(a, {(60, 10): NEWT}, hero=(x, 10)):
            first = x
            break
    assert first == 60 - NetHackAgent.HOSTILE_NEAR_RADIUS


def test_approaching_monster_interrupts_before_radius():
    a = _agent({(51, 10): NEWT})
    reason = ""
    for x in (50, 49, 48):
        reason = _interrupt(a, {(x, 10): NEWT})
        if reason:
            break
    assert "approaching" in (reason or "")


def test_melee_target_is_exempt():
    a = _agent({(41, 10): NEWT}, exempt=(41, 10))
    for x in (43, 41):  # it backs off and comes back
        assert _interrupt(a, {(x, 10): NEWT}) is None


@pytest.mark.parametrize("old,new,text", [
    ((2, 0), (3, 2), "You went down from Dlvl 2 to Dlvl 3 and entered the Gnomish Mines"),
    ((1, 0), (2, 0), "You went down from Dlvl 1 to Dlvl 2"),
    ((4, 2), (3, 0), "You went up from Dlvl 4 to Dlvl 3 and entered the Dungeons of Doom"),
])
def test_level_change_is_described_in_dlvl_terms(old, new, text):
    assert describe_level_change(*old, *new) == text


def test_execute_skill_stops_on_a_level_change_not_on_a_new_glyph():
    a = _agent(depth=2)

    def skill(agent):
        _set_obs(agent, (40, 10), {(41, 10): GEM}, depth=2, time=2)
        new_glyph = tracking.NewGlyphEvent(GEM, (41, 10))
        yield Step.act(StepData(0, None, 0.0, False, False, {}, [new_glyph]))
        _set_obs(agent, (5, 5), {}, depth=3, dnum=2, time=3)
        level_change = tracking.DungeonLevelChangeEvent((0, 2), (2, 1))
        yield Step.act(StepData(0, None, 0.0, False, False, {}, [level_change]))
        yield Step.completed("should not get here")

    skill.name = "press_key"
    thoughts = " ".join(s.thoughts for s in a._execute_skill(skill, {}) if s.thoughts)
    assert "appeared" not in thoughts and "glass" not in thoughts
    assert "should not get here" not in thoughts
    assert "You went down from Dlvl 2 to Dlvl 3 and entered the Gnomish Mines" in thoughts
    assert a.last_interrupt
