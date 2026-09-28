"""The NetCore skill-interrupt policy (vendor/.../nethack_agent/agent.py).

Play-tests found closed-loop skills interrupted by monsters 60 columns away, by
every shop item and fresh corpse "appearing", by the same werejackal on two
calls in a row, with the event text naming unidentified objects by their TRUE
identity, and with level changes reported as the level index within a branch
rather than the Dlvl the status line shows. These tests pin the replacement:

  * a HOSTILE monster interrupts when it comes within HOSTILE_NEAR_RADIUS, or
    keeps closing in from within APPROACH_RADIUS; far monsters do not;
  * items never interrupt (so their names never reach the text at all);
  * pets and peaceful monsters (farlook says "tame"/"peaceful") never do;
  * the same monster does not interrupt again on the next call while it stays
    put;
  * level changes are described in Dlvl terms, naming a newly entered branch.

They drive the policy on synthetic observations: no engine is needed.
"""

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.tools import netplay_true  # noqa: E402,F401  (puts vendor/ on sys.path)
from netplay.core.agent_base import Step, StepData  # noqa: E402
from netplay.nethack_agent import agent as agent_mod  # noqa: E402
from netplay.nethack_agent.agent import NetHackAgent, describe_level_change  # noqa: E402
import netplay.nethack_agent.tracking as tracking  # noqa: E402
from nethack_core import glyphs as G  # noqa: E402

H, W = 21, 79
STONE = G.GLYPH_CMAP_OFF  # S_stone


def _mon_id(name):
    for i in range(G.NUMMONS):
        if G.monster_name(i) == name:
            return i
    raise KeyError(name)


NEWT = G.GLYPH_MON_OFF + _mon_id("newt")
GNOME = G.GLYPH_MON_OFF + _mon_id("gnome")
KITTEN_PET = G.GLYPH_PET_OFF + _mon_id("kitten")
SHK = G.GLYPH_MON_OFF + _mon_id("shopkeeper")
HERO = G.GLYPH_MON_OFF + _mon_id("valkyrie")
GEM = G.GLYPH_OBJ_OFF + 200  # some object glyph; which one does not matter


class _Blstats(SimpleNamespace):
    pass


def _make_agent(hero=(40, 10), depth=1, dnum=0):
    """A NetHackAgent with just the state the interrupt policy reads."""
    a = NetHackAgent.__new__(NetHackAgent)
    a.describe_tile = None
    a.last_interrupt = None
    a._tracked_monsters = []
    a._reported_last_call = []
    a.max_skill_gamesteps = 100
    a.data = SimpleNamespace(blstats=None)
    a._descr = {}
    a.describe_tile = lambda x, y: a._descr.get((x, y))
    _set_obs(a, hero, {}, depth=depth, dnum=dnum)
    return a


def _set_obs(a, hero, things, depth=1, dnum=0, time=1):
    glyphs = np.full((H, W), STONE, dtype=np.int16)
    for (x, y), g in things.items():
        glyphs[y, x] = g
    glyphs[hero[1], hero[0]] = HERO
    a.last_observation = {"glyphs": glyphs}
    a.data.blstats = _Blstats(x=hero[0], y=hero[1], depth=depth,
                              dungeon_number=dnum, level_number=depth,
                              time=time, hitpoints=16, max_hitpoints=16)


def test_far_hostile_does_not_interrupt_near_one_does():
    a = _make_agent()
    a._begin_monster_tracking()
    # 60 columns away: the play-test lichen that stopped a kick.
    _set_obs(a, (40, 10), {(78, 10): NEWT})
    assert a._hostile_monster_interrupt() is None
    _set_obs(a, (40, 10), {(78, 10): NEWT, (45, 12): NEWT})
    reason = a._hostile_monster_interrupt()
    assert reason and "hostile newt" in reason and "(45,12)" in reason


def test_items_never_interrupt_and_are_never_named():
    a = _make_agent()
    a._begin_monster_tracking()
    for x in range(41, 46):  # a shop's worth of items right next to the hero
        _set_obs(a, (40, 10), {(xx, 10): GEM for xx in range(41, x + 1)})
        assert a._hostile_monster_interrupt() is None


def test_pets_and_peacefuls_do_not_interrupt():
    a = _make_agent()
    a._begin_monster_tracking()
    a._descr = {(42, 10): "tame kitten", (43, 11): "peaceful gnome",
                (44, 9): "peaceful Asidonhopo"}
    _set_obs(a, (40, 10), {(42, 10): KITTEN_PET, (43, 11): GNOME, (44, 9): SHK})
    assert a._hostile_monster_interrupt() is None


def test_always_peaceful_species_without_description_is_skipped():
    a = _make_agent()
    a.describe_tile = None
    a._begin_monster_tracking()
    _set_obs(a, (40, 10), {(44, 9): SHK})
    assert a._hostile_monster_interrupt() is None


def test_angry_shopkeeper_is_hostile():
    a = _make_agent()
    a._begin_monster_tracking()
    a._descr = {(44, 9): "Asidonhopo"}
    _set_obs(a, (40, 10), {(44, 9): SHK})
    assert "Asidonhopo" in (a._hostile_monster_interrupt() or "")


def test_hostile_gnome_uses_the_farlook_name():
    a = _make_agent()
    a._begin_monster_tracking()
    a._descr = {(43, 11): "gnome, leashed to you"}
    _set_obs(a, (40, 10), {(43, 11): GNOME})
    assert "hostile gnome came into view at (43,11)" in a._hostile_monster_interrupt()


def test_description_about_the_square_falls_back_to_species():
    # The map can show a monster the hero cannot look at; farlook then
    # describes the terrain. Never call a monster "a hostile corridor".
    a = _make_agent()
    a._begin_monster_tracking()
    a._descr = {(43, 11): "corridor"}
    _set_obs(a, (40, 10), {(43, 11): NEWT})
    reason = a._hostile_monster_interrupt()
    assert "hostile newt" in reason and "corridor" not in reason


def test_same_monster_does_not_interrupt_on_the_next_call_while_it_stays_put():
    a = _make_agent()
    a._begin_monster_tracking()
    _set_obs(a, (40, 10), {(45, 10): NEWT})
    assert a._hostile_monster_interrupt()
    a._end_monster_tracking()
    # Next call: the newt is still there (or one square over).
    a._begin_monster_tracking()
    for pos in [(45, 10), (45, 11), (46, 11)]:
        _set_obs(a, (40, 10), {pos: NEWT})
        assert a._hostile_monster_interrupt() is None
    a._end_monster_tracking()
    # And the call after that.
    a._begin_monster_tracking()
    _set_obs(a, (40, 10), {(46, 11): NEWT})
    assert a._hostile_monster_interrupt() is None


def test_monster_on_screen_at_call_start_is_not_new():
    a = _make_agent()
    _set_obs(a, (40, 10), {(44, 10): NEWT})
    a._begin_monster_tracking()
    _set_obs(a, (40, 10), {(44, 10): NEWT})
    assert a._hostile_monster_interrupt() is None


def test_known_monster_interrupts_when_it_comes_into_range():
    a = _make_agent()
    _set_obs(a, (40, 10), {(60, 10): NEWT})
    a._begin_monster_tracking()
    hero = 40
    reasons = []
    for _ in range(20):
        hero += 1
        _set_obs(a, (hero, 10), {(60, 10): NEWT})
        r = a._hostile_monster_interrupt()
        if r:
            reasons.append((hero, r))
            break
    assert reasons and reasons[0][0] == 60 - agent_mod.NetHackAgent.HOSTILE_NEAR_RADIUS


def test_approaching_monster_interrupts_before_radius():
    a = _make_agent()
    _set_obs(a, (40, 10), {(51, 10): NEWT})
    a._begin_monster_tracking()
    r = None
    for x in (50, 49, 48):
        _set_obs(a, (40, 10), {(x, 10): NEWT})
        r = a._hostile_monster_interrupt()
        if r:
            break
    assert r and "approaching" in r


def test_melee_target_is_exempt():
    a = _make_agent()
    _set_obs(a, (40, 10), {(41, 10): NEWT})
    a._begin_monster_tracking(exempt=(41, 10))
    for x in (43, 41):  # it backs off and comes back
        _set_obs(a, (40, 10), {(x, 10): NEWT})
        assert a._hostile_monster_interrupt() is None


@pytest.mark.parametrize("old,new,text", [
    ((2, 0), (3, 2), "You went down from Dlvl 2 to Dlvl 3 and entered the Gnomish Mines"),
    ((1, 0), (2, 0), "You went down from Dlvl 1 to Dlvl 2"),
    ((4, 2), (3, 0), "You went up from Dlvl 4 to Dlvl 3 and entered the Dungeons of Doom"),
])
def test_level_change_is_described_in_dlvl_terms(old, new, text):
    assert describe_level_change(old[0], old[1], new[0], new[1]) == text


def test_execute_skill_reports_level_change_and_ignores_new_glyph_events():
    """End to end through `_execute_skill`: a NewGlyphEvent (a gem appearing)
    must not stop the skill; a level change must, in Dlvl terms."""
    a = _make_agent(depth=2, dnum=0)

    def fake_skill_fn(agent):
        _set_obs(agent, (40, 10), {(41, 10): GEM}, depth=2, dnum=0, time=2)
        yield Step.act(StepData(0, None, 0.0, False, False, {},
                                [tracking.NewGlyphEvent(GEM, (41, 10))]))
        _set_obs(agent, (5, 5), {}, depth=3, dnum=2, time=3)
        yield Step.act(StepData(0, None, 0.0, False, False, {},
                                [tracking.DungeonLevelChangeEvent((0, 2), (2, 1))]))
        yield Step.completed("should not get here")

    class _Skill:  # skills are callables returning a Step generator
        name = "press_key"

        def __call__(self, agent, **kw):
            return fake_skill_fn(agent)

    out = [s.thoughts for s in a._execute_skill(_Skill(), {}) if s.thoughts]
    joined = " ".join(out)
    assert "appeared" not in joined and "glass" not in joined
    assert "You went down from Dlvl 2 to Dlvl 3 and entered the Gnomish Mines" in joined
    assert "should not get here" not in joined
    assert a.last_interrupt
