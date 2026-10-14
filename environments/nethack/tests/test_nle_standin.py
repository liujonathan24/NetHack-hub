"""The `nle` stand-in: the names and records CodeHack reads; NetPlay's action values unchanged."""

import importlib.util
import pathlib

import pytest

from nethack_harness.tools import netplay_true  # puts vendor/ on sys.path
import nethack_core.actions as core_actions
import nle.nethack as nh
import nle.nethack.actions as nle_actions
from nle_shim_data import codehack_tables

_VENDOR = pathlib.Path(netplay_true.__file__).resolve().parents[2] / "vendor"


def _obj(name):
    return next(o for o in map(nh.objclass, range(nh.NUM_OBJECTS)) if nh.OBJ_NAME(o) == name)


def test_frozen_tables_equal_the_engine_library():
    spec = importlib.util.spec_from_file_location("gen", _VENDOR / "tools/gen_codehack_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.read_tables() == (codehack_tables.OBJECTS, codehack_tables.MONSTERS)


@pytest.mark.parametrize("enum", ["CompassDirection", "CompassDirectionLonger", "MiscDirection",
                                  "MiscAction", "Command", "TextCharacters"])
def test_actions_package_keeps_netplay_values(enum):
    from nle.nethack import actions

    assert actions is nle_actions and isinstance(nh.ACTIONS, tuple) and len(nh.ACTIONS) == 121
    members = [(m.name, int(m)) for m in getattr(nle_actions, enum)]
    assert members == [(m.name, int(m)) for m in getattr(core_actions, enum)]


@pytest.mark.parametrize("name, fields", [
    ("long sword", {"oc_class": chr(nh.WEAPON_CLASS), "oc_skill": 7, "oc_weight": 40}),
    ("arrow", {"oc_skill": -21}),
    ("plate mail", {"oc_armcat": 0, "a_ac": 7, "oc_weight": 450}),
    ("elven cloak", {"oc_armcat": 5, "a_ac": 1}),
    ("food ration", {"oc_nutrition": 800, "oc_weight": 20}),
])
def test_object_records(name, fields):
    obj = _obj(name)
    assert {k: getattr(obj, k) for k in fields} == fields


def test_monsters_descriptions_and_masks():
    ant = nh.permonst(0)
    assert (ant.mname, ant.mlet, ant.difficulty, ant.ac, ant.cnutrit, ant.cwt) == (
        "giant ant", "\x01", 4, 3, 10, 10)
    assert nh.permonst(nh.NUMMONS).mname is None
    assert nh.OBJ_DESCR(_obj("elven cloak")) == "faded pall"
    assert nh.OBJ_DESCR(_obj("long sword")) is None
    masks = (nh.BL_MASK_STONE, nh.BL_MASK_LEV, nh.BL_MASK_RIDE, nh.BL_MASK_BITS)
    assert nh.RANDOM_CLASS == 0 and masks == (1, 0x400, 0x1000, 13)
