"""Compatibility shim: ``nle.nethack`` backed by our engine (``nethack_core``).

The vendored NetPlay skill layer imports ``nle.nethack`` for *constants and pure
glyph arithmetic only* -- it never drives NLE's engine through this module. We
run our own NetHack fork behind ``nethack_core``, and ``nle`` is not installed,
so this module re-exports the equivalents instead of pulling in a second engine.

Everything here is either
  (a) re-exported straight from ``nethack_core`` (actions, glyph offsets, glyph
      predicates) -- ``nethack_core.actions`` mirrors NLE's enums verbatim, and
      the glyph offsets are the same NetHack 3.6 layout; or
  (b) derived with the exact arithmetic from the fork's ``include/display.h``
      (zap/explode/swallow/warning offsets and the ``glyph_to_*`` accessors); or
  (c) read from ``nle_shim_data.objects_table``, generated from the fork's
      ``src/objects.c`` by ``vendor/tools/gen_object_table.py`` (the same source
      NetHack's own ``makedefs`` reads).

This file is OURS, not upstream NetPlay code. See vendor/PROVENANCE.md.
"""

from dataclasses import dataclass
from typing import Optional

from nethack_core import actions  # noqa: F401  (re-exported for NetPlay)
from nethack_core import glyphs as _g

from nle_shim_data.objects_table import OBJECT_TABLE

# --- core sizes -----------------------------------------------------------
NUMMONS = _g.NUMMONS
NUM_OBJECTS = _g.NUM_OBJECTS
MAXPCHARS = _g.MAXPCHARS

# Fork constants (include/rm.h, include/display.h, include/decl.h, include/hack.h)
MAXEXPCHARS = 9
EXPL_MAX = 7
NUM_ZAP = 8
WARNCOUNT = 6
NO_GLYPH = _g.MAX_GLYPH

# --- glyph offsets --------------------------------------------------------
GLYPH_MON_OFF = _g.GLYPH_MON_OFF
GLYPH_PET_OFF = _g.GLYPH_PET_OFF
GLYPH_INVIS_OFF = _g.GLYPH_INVIS_OFF
GLYPH_DETECT_OFF = _g.GLYPH_DETECT_OFF
GLYPH_BODY_OFF = _g.GLYPH_BODY_OFF
GLYPH_RIDDEN_OFF = _g.GLYPH_RIDDEN_OFF
GLYPH_OBJ_OFF = _g.GLYPH_OBJ_OFF
GLYPH_CMAP_OFF = _g.GLYPH_CMAP_OFF
GLYPH_STATUE_OFF = _g.GLYPH_STATUE_OFF
MAX_GLYPH = _g.MAX_GLYPH
GLYPH_INVISIBLE = _g.GLYPH_INVISIBLE

# Derived exactly as include/display.h does. nethack_core does not export these
# four because it has no use for zap/explosion/swallow/warning glyphs.
GLYPH_EXPLODE_OFF = (MAXPCHARS - MAXEXPCHARS) + GLYPH_CMAP_OFF
GLYPH_ZAP_OFF = (MAXEXPCHARS * EXPL_MAX) + GLYPH_EXPLODE_OFF
GLYPH_SWALLOW_OFF = (NUM_ZAP << 2) + GLYPH_ZAP_OFF
GLYPH_WARNING_OFF = (NUMMONS << 3) + GLYPH_SWALLOW_OFF

# Consistency check against nethack_core's independently-derived values: if the
# fork's glyph layout ever shifts, fail loudly at import rather than silently
# mis-decode glyphs.
assert GLYPH_STATUE_OFF == WARNCOUNT + GLYPH_WARNING_OFF, "glyph layout drift"
assert MAX_GLYPH == NUMMONS + GLYPH_STATUE_OFF, "glyph layout drift"

# --- glyph predicates (vectorised, straight from nethack_core) ------------
glyph_is_pet = _g.glyph_is_pet
glyph_is_monster = _g.glyph_is_monster
glyph_is_normal_monster = _g.glyph_is_normal_monster
glyph_is_ridden_monster = _g.glyph_is_ridden_monster
glyph_is_detected_monster = _g.glyph_is_detected_monster
glyph_is_normal_object = _g.glyph_is_normal_object
glyph_is_object = _g.glyph_is_object
glyph_is_body = _g.glyph_is_body
glyph_is_statue = _g.glyph_is_statue
glyph_is_cmap = _g.glyph_is_cmap
glyph_is_trap = _g.glyph_is_trap
glyph_is_invisible = _g.glyph_is_invisible


def glyph_is_swallow(glyph):
    """display.h: GLYPH_SWALLOW_OFF <= g < GLYPH_SWALLOW_OFF + (NUMMONS << 3)"""
    return GLYPH_SWALLOW_OFF <= glyph < GLYPH_SWALLOW_OFF + (NUMMONS << 3)


def glyph_is_warning(glyph):
    """display.h: GLYPH_WARNING_OFF <= g < GLYPH_WARNING_OFF + WARNCOUNT"""
    return GLYPH_WARNING_OFF <= glyph < GLYPH_WARNING_OFF + WARNCOUNT


# --- glyph accessors ------------------------------------------------------
glyph_to_mon = _g.glyph_to_mon


def glyph_to_pet(glyph):
    return int(glyph) - GLYPH_PET_OFF


def glyph_to_obj(glyph):
    """display.h glyph_to_obj, restricted to the normal-object case.

    The CORPSE/STATUE branches need otyp constants that only matter for body and
    statue glyphs; NetPlay only calls this after glyph_is_normal_object().
    """
    if glyph_is_normal_object(glyph):
        return int(glyph) - GLYPH_OBJ_OFF
    return NO_GLYPH


def glyph_to_cmap(glyph):
    return int(glyph) - GLYPH_CMAP_OFF if glyph_is_cmap(glyph) else NO_GLYPH


def glyph_to_swallow(glyph):
    return ((int(glyph) - GLYPH_SWALLOW_OFF) & 0x7) if glyph_is_swallow(glyph) else 0


def glyph_to_warning(glyph):
    return int(glyph) - GLYPH_WARNING_OFF if glyph_is_warning(glyph) else NO_GLYPH


# --- object classes (include/objclass.h) ----------------------------------
ILLOBJ_CLASS = 1
WEAPON_CLASS = 2
ARMOR_CLASS = 3
RING_CLASS = 4
AMULET_CLASS = 5
TOOL_CLASS = 6
FOOD_CLASS = 7
POTION_CLASS = 8
SCROLL_CLASS = 9
SPBOOK_CLASS = 10
WAND_CLASS = 11
COIN_CLASS = 12
GEM_CLASS = 13
ROCK_CLASS = 14
BALL_CLASS = 15
CHAIN_CLASS = 16
VENOM_CLASS = 17
MAXOCLASSES = 18


@dataclass(frozen=True)
class _ObjClass:
    """Stands in for NLE's ``objclass``.

    NLE returns ``oc_class`` as a one-character ``bytes``/``str`` and NetPlay
    does ``ord(nh.objclass(i).oc_class)``, so we store the class as a 1-char
    string whose ordinal is the class number -- keeping the upstream ``ord(...)``
    call correct and unmodified.
    """

    oc_class: str


@dataclass(frozen=True)
class _ObjDescr:
    oc_name: Optional[str]
    oc_descr: Optional[str]

    @staticmethod
    def from_idx(idx: int) -> "_ObjDescr":
        name, descr, _ = OBJECT_TABLE[int(idx)]
        return _ObjDescr(name, descr)


def objclass(idx: int) -> _ObjClass:
    return _ObjClass(chr(OBJECT_TABLE[int(idx)][2]))


def permonst(mon_idx: int):
    """Stands in for NLE's ``permonst``. NetPlay only ever reads ``.mname``."""
    return _PerMonst(_g.monster_name(int(mon_idx)))


@dataclass(frozen=True)
class _PerMonst:
    mname: Optional[str]


objdescr = _ObjDescr
