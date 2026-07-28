"""BALROG's 80-action NetHack surface, registered as `bal_*` skills.

BALROG (balrog-ai/BALROG, `balrog/environments/nle/__init__.py`) hands its agent
an ACTIONS dict of exactly 80 text commands and nothing else. That is a very
different bargain from our own skill sets: no pathfinding, no closed loops, no
in-skill item selection. The agent presses one key, sees what happened, presses
another. `travel` and `far <dir>` are the only labour-saving devices, and both
are stock NetHack commands rather than harness code.

This module reproduces that surface so encoding results can be read against
BALROG's published numbers on a matched action space.

WHY THESE ARE JUST KEYSTROKES
-----------------------------
`nethack_core.actions` mirrors `nle.nethack`'s enums verbatim, and the engine
consumes those enum values *as keystroke bytes* -- `_to_action_indices`
(helpers.py) is a documented identity pass-through. So every BALROG action is
one integer, and a skill is `SkillResult(actions=[that integer])`.

This is why the surface needs no special-casing to stay coherent. `travel`
prompts for a target, and the agent answers with `up`/`down` -- which are
MiscDirection.UP/DOWN, i.e. `<` (60) and `>` (62), the very characters NetHack's
travel prompt wants. Menu answers work the same way: digits are TextCharacters.
The 80 actions compose because they are keys, not wrappers.

THE `bal_` PREFIX
-----------------
Registry names are prefixed, following the `np_` precedent in netplay_true.py,
because three separate mechanisms would otherwise silently capture the bare
names and we would measure something other than what we think:

  * `_SKILL_ALIASES` (tools/skills.py) already maps `travel` -> `move_to`,
    `look` -> `search`, `down` -> `descend`.
  * `env_response` rebinds bare `north`..`northwest` to `move(direction=...)`
    for the `dir8` preset.
  * `_build_skill_adapter_callables("full")` excludes only the `np_` prefix, so
    unprefixed additions would leak into every pre-existing experiment arm.

The prefix is cosmetic -- `bal_north` sends byte 107 exactly as BALROG's `north`
does -- and it keeps this surface strictly additive to the arms already run.

A NOTE ON PROMPTS
-----------------
These are open-loop commands: `bal_eat` opens "What do you want to eat?" and
leaves it open. The agent is expected to answer on its next turn, as BALROG's
does. That only works if the harness stops auto-dismissing menus, which is
gated on this skill set in nethack.py; see `balrog_raw_prompts` there. BALROG
runs `skip_more: True`, so --More-- auto-acknowledgement stays on in both.
"""

from __future__ import annotations

from nethack_core import actions as _A
from nethack_harness.tools.skills import SkillResult, registry

_C = _A.Command
_D = _A.CompassDirection
_DL = _A.CompassDirectionLonger
_M = _A.MiscDirection
_MA = _A.MiscAction
_T = _A.TextCharacters

#: Namespace prefix; see module docstring.
TOOL_PREFIX = "bal_"

#: BALROG action name -> (keystroke byte, BALROG's own description verbatim).
#: Transcribed mechanically from their ACTIONS dict, not by hand.
BALROG_ACTIONS: dict[str, tuple[int, str]] = {
    'north': (_D.N, 'move north'),
    'east': (_D.E, 'move east'),
    'south': (_D.S, 'move south'),
    'west': (_D.W, 'move west'),
    'northeast': (_D.NE, 'move northeast'),
    'southeast': (_D.SE, 'move southeast'),
    'southwest': (_D.SW, 'move southwest'),
    'northwest': (_D.NW, 'move northwest'),
    'far north': (_DL.N, 'move far north'),
    'far east': (_DL.E, 'move far east'),
    'far south': (_DL.S, 'move far south'),
    'far west': (_DL.W, 'move far west'),
    'far northeast': (_DL.NE, 'move far northeast'),
    'far southeast': (_DL.SE, 'move far southeast'),
    'far southwest': (_DL.SW, 'move far southwest'),
    'far northwest': (_DL.NW, 'move far northwest'),
    'up': (_M.UP, 'go up a staircase'),
    'down': (_M.DOWN, 'go down a staircase (tip: you can only go down if you are standing on the stairs)'),
    'wait': (_M.WAIT, 'rest one move while doing nothing'),
    'more': (_MA.MORE, 'display more of the message (tip: ONLY ever use when current message ends with --More--)'),
    'annotate': (_C.ANNOTATE, 'leave a note about the level'),
    'apply': (_C.APPLY, 'apply (use) a tool'),
    'call': (_C.CALL, 'name a monster or object, or add an annotation'),
    'cast': (_C.CAST, 'cast a spell'),
    'close': (_C.CLOSE, 'close an adjacent door'),
    'open': (_C.OPEN, 'open an adjacent door'),
    'dip': (_C.DIP, 'dip an object into something'),
    'drop': (_C.DROP, 'drop an item'),
    'droptype': (_C.DROPTYPE, 'drop specific item types (specify in the next prompt)'),
    'eat': (_C.EAT, 'eat something (tip: replenish food when hungry)'),
    'esc': (_C.ESC, 'exit menu or message'),
    'engrave': (_C.ENGRAVE, 'engrave writing on the floor (tip: Elbereth)'),
    'enhance': (_C.ENHANCE, 'advance or check weapons skills'),
    'fire': (_C.FIRE, 'fire ammunition from quiver'),
    'fight': (_C.FIGHT, 'fight a monster (even if you only guess one is there)'),
    'force': (_C.FORCE, 'force a lock'),
    'inventory': (_C.INVENTORY, 'show your inventory'),
    'invoke': (_C.INVOKE, 'invoke '),
    'jump': (_C.JUMP, 'jump to a location'),
    'kick': (_C.KICK, 'kick an enemy or a locked door or chest'),
    'look': (_C.LOOK, 'look at what is under you'),
    'loot': (_C.LOOT, 'loot a box on the floor'),
    'monster': (_C.MONSTER, "use a monster's special ability (when polymorphed)"),
    'offer': (_C.OFFER, 'offer a sacrifice to the gods (tip: on an aligned altar)'),
    'overview': (_C.OVERVIEW, 'display an overview of the dungeon'),
    'pay': (_C.PAY, 'pay your shopping bill'),
    'pickup': (_C.PICKUP, 'pick up things at the current location'),
    'pray': (_C.PRAY, 'pray to the gods for help'),
    'puton': (_C.PUTON, 'put on an accessory'),
    'quaff': (_C.QUAFF, 'quaff (drink) something'),
    'quiver': (_C.QUIVER, 'select ammunition for quiver'),
    'read': (_C.READ, 'read a scroll or spellbook'),
    'remove': (_C.REMOVE, 'remove an accessory'),
    'rub': (_C.RUB, 'rub a lamp or a stone'),
    'search': (_C.SEARCH, 'search for hidden doors and passages'),
    'swap': (_C.SWAP, 'swap wielded and secondary weapons'),
    'takeoff': (_C.TAKEOFF, 'take off one piece of armor'),
    'takeoffall': (_C.TAKEOFFALL, 'take off all armor'),
    'teleport': (_C.TELEPORT, 'teleport to another level (if you have the ability)'),
    'throw': (_C.THROW, 'throw something (e.g. a dagger or dart)'),
    'travel': (_C.TRAVEL, 'travel to a specific location on the map (tip: in the next action, specify > or < for stairs, { for fountain, and _ for altar)'),
    'twoweapon': (_C.TWOWEAPON, 'toggle two-weapon combat'),
    'untrap': (_C.UNTRAP, 'untrap something'),
    'wear': (_C.WEAR, 'wear a piece of armor'),
    'wield': (_C.WIELD, 'wield a weapon'),
    'wipe': (_C.WIPE, 'wipe off your face'),
    'zap': (_C.ZAP, 'zap a wand'),
    'minus': (_T.MINUS, '-'),
    'space': (_T.SPACE, ' '),
    'apos': (_T.APOS, "'"),
    '0': (_T.NUM_0, '0'),
    '1': (_T.NUM_1, '1'),
    '2': (_T.NUM_2, '2'),
    '3': (_T.NUM_3, '3'),
    '4': (_T.NUM_4, '4'),
    '5': (_T.NUM_5, '5'),
    '6': (_T.NUM_6, '6'),
    '7': (_T.NUM_7, '7'),
    '8': (_T.NUM_8, '8'),
    '9': (_T.NUM_9, '9'),
}


def _make_action_skill(key: int, action_name: str):
    """One skill that presses `key` once and returns control to the agent."""

    def _skill(env, obs) -> SkillResult:
        return SkillResult(actions=[int(key)], feedback=f"pressed: {action_name}")

    return _skill


def register_all() -> list[str]:
    """Register all 80 actions as `bal_*` skills. Idempotent. Returns the names."""
    names = []
    for action_name, (key, desc) in BALROG_ACTIONS.items():
        tool = TOOL_PREFIX + action_name.replace(" ", "_")
        registry.register(tool, {"description": desc, "parameters": {}})(
            _make_action_skill(key, action_name)
        )
        names.append(tool)
    return names


#: The 80 registered tool names, in BALROG's own order.
BALROG_TOOL_NAMES: list[str] = register_all()


def register_menu_letters() -> list[str]:
    """Register `bal_a`..`bal_z` -- the item-selection keys.

    BALROG documents 80 actions but its *valid* action space is 248 strings:
    the 86 USEFUL_ACTIONS names plus `a-z`, `A-Z` and `0-99`
    (balrog/environments/nle/base.py). The bare letters are how their agent
    answers NetHack's own item prompts -- their tips block says so explicitly:
    "What do you want to eat? [dgh or ?*]" -> answer "d", "g" or "h".

    Our agent acts by calling tools, so without these it simply CANNOT answer an
    item prompt except by the accident of a letter coinciding with a command
    byte (`h/j/k/l/y/u/b/n` are compass keys, `i` is inventory). That makes
    eat/quaff/read/wield/wear unusable on the raw surface -- a functional gap,
    not a stylistic one.
    """
    names = []
    for ch in "abcdefghijklmnopqrstuvwxyz":
        tool = TOOL_PREFIX + ch
        if tool in BALROG_TOOL_NAMES:
            continue
        registry.register(
            tool, {"description": f"answer a menu/item prompt with '{ch}'", "parameters": {}}
        )(_make_action_skill(ord(ch), ch))
        names.append(tool)
    return names


#: Menu-answer letters, registered alongside the 80.
BALROG_MENU_LETTERS: list[str] = register_menu_letters()


def balrog_instruction_prompt() -> str:
    """BALROG's own NLE system prompt, verbatim except for the action names.

    Reproduced from `get_instruction_prompt()` in their
    `balrog/environments/nle/__init__.py`: the full action list with each
    description, their tips block, then "PLAY!".

    THE ONE DEVIATION is the action names: BALROG's agent emits bare strings
    (`north`), ours calls tools (`bal_north`). Listing the bare names would
    describe an action set the model cannot actually invoke. The mapping is 1:1
    and every description is theirs, so the semantics are unchanged.

    Why this exists: the first `balrog80` sweep published the 80 tools but kept
    OUR system prompt, which is written for the closed-loop macro skills and
    never explains that descent needs `down` *while standing on the stairs*, nor
    that `travel` wants a `>`/`<` follow-up. Across 12,446 turns the agent
    called `bal_down` ONCE and spent 24.8% of its actions on `bal_search`,
    ending every episode on Dlvl 1 for a flat 0.00%. That measured a missing
    prompt, not the action space -- BALROG scores 3.96 on this same surface.
    """
    action_strings = ",\n".join(
        f"{TOOL_PREFIX}{name.replace(' ', '_')}: {desc}"
        for name, (_key, desc) in BALROG_ACTIONS.items()
    )
    return f"""
You are an agent playing NetHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

Tips:
- When the message asks for a completion, such as: "What do you want to eat? [d or ?*]", you should respond with a single character corresponding to the item you want to eat/use.
    - For example, "What do you want to eat? [dgh or ?*]" -> Possible answers are "d", "g", or "h" to eat the associated food.
- When the message asks for a direction, such as: "In what direction?" you should respond with a direction.
- When the message has --More-- at the end, your next action should be "more" to see the rest of the message.
- Explore the environment to find the stairs down to the next level.
- Always carefully read the last message to understand the current state of the game and decide your next action accordingly.
- If you keep moving in the same direction, you will eventually hit a wall and stop moving. Your message might be: "It's solid stone", or "It's a wall". Change your action to move in another direction to continue exploring the environment.
- Read the language observation carefully and look at ascii map or image observation provided to decide the next action to take and where to move next.
- You can attack monsters by moving into them.

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible in the game.

PLAY!
""".strip()
