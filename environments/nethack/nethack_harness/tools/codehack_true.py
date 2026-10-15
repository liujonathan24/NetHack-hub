"""CodeHack's skills on our engine, published as `ch_<name>` tools.

CodeHack (github.com/BartekCupial/codehack) comes from the fork
liujonathan24/codehack, branch nethack-env, installed in the game image. Its
`Bot` drives `CodeHackGame`, an NLE-shaped view of our core env. Every key goes
through `core_env.step`, looked up at call time, so the turn's TurnRecorder
records it and the grader's replay reproduces the game.
"""

from __future__ import annotations

import contextlib
import copy
import io
import logging
import re
import weakref

import numpy as np

from nethack_harness.helpers import CARRIAGE_RETURN, _cr_would_be_unknown_command
from nethack_harness.prompt import tty_screen
from nethack_harness.tools import netplay_true as _npt  # registers np_press_key; puts the `nle` stand-in on sys.path
from nethack_harness.tools.skills import SkillResult, registry

import codehack.bot.strategies as strategies  # noqa: E402
from codehack.bot import Bot  # noqa: E402
from codehack.bot.exceptions import BotFinished, BotPanic, EnemyAppeared, LostHP  # noqa: E402
from codehack.bot.level import LEVEL_CHANGERS  # noqa: E402
from codehack.bot.panics import enemy_appeared, lost_hp  # noqa: E402
from nle.nethack import ACTIONS  # noqa: E402

logger = logging.getLogger(__name__)

#: The paper's NetHack skill list without its way down, in the paper's order.
CH_DESCRIPTIONS = {
    "engrave_elbereth": "Engraves Elbereth on the floor with your fingers.",
    "explore": "Explores undiscovered tiles, starting from the closest ones. Stops when something new comes into view.",
    "fight_engulfed": "Fights from inside a monster that has engulfed you, until you are out.",
    "fight_melee": "Fights the closest monster in melee. Hostile monsters come first.",
    "fight_ranged": "Fights the closest monster with a ranged weapon.",
    "zap_monster": "Zaps the closest monster with a wand, without hitting yourself on a bounce.",
    "approach_monster": "Moves toward the closest monster until it is within reach of your wands.",
    "goto_choke_point": "Moves to the nearest choke point. Use it only when there are several monsters around.",
    "wait_for_monster": "Waits for monsters to come within attack range.",
    "goto_corridor": "Moves to the closest other corridor (not the one you are in).",
    "goto_corridor_east": "Moves to the nearest other corridor to the east (none if you are already easternmost).",
    "goto_corridor_north": "Moves to the nearest other corridor to the north (none if you are already northernmost).",
    "goto_corridor_south": "Moves to the nearest other corridor to the south (none if you are already southernmost).",
    "goto_corridor_west": "Moves to the nearest other corridor to the west (none if you are already westernmost).",
    "goto_room": "Moves to the closest other room (not the one you are in).",
    "goto_room_east": "Moves to the nearest other room to the east (none if you are already easternmost).",
    "goto_room_north": "Moves to the nearest other room to the north (none if you are already northernmost).",
    "goto_room_south": "Moves to the nearest other room to the south (none if you are already southernmost).",
    "goto_room_west": "Moves to the nearest other room to the west (none if you are already westernmost).",
    "ascend_stairs": "Goes to the nearest up staircase and climbs it.",
    "goto_corpse": "Moves to the closest corpse.",
    "examine_items": "Goes to items on the floor and looks at them, starting with the ones where you stand.",
    "open_doors": "Tries to open the closest closed door.",
    "open_doors_kick": "Tries to open the closest closed door by kicking it.",
    "open_doors_key": "Tries to open the closest closed door with a key from your inventory.",
    "pickup_amulet": "Picks up amulets from the floor.",
    "pickup_armor": "Picks up armor and shields from the floor.",
    "pickup_coin": "Picks up coins from the floor.",
    "pickup_food": "Picks up food from the floor.",
    "pickup_gem": "Picks up gems from the floor.",
    "pickup_potion": "Picks up potions from the floor.",
    "pickup_ring": "Picks up rings from the floor.",
    "pickup_scroll": "Picks up scrolls from the floor.",
    "pickup_spellbook": "Picks up spellbooks from the floor.",
    "pickup_tool": "Picks up tools from the floor.",
    "pickup_wand": "Picks up wands from the floor.",
    "pickup_weapon": "Picks up weapons from the floor.",
    "pickup_corpse": "Picks up a corpse from the floor.",
    "puton_ring": "Puts on a ring from your inventory.",
    "puton_amulet": "Puts on an amulet from your inventory.",
    "wear_boots": "Wears boots from your inventory.",
    "wear_cloak": "Wears a cloak from your inventory.",
    "wear_gloves": "Wears gloves from your inventory.",
    "wear_helm": "Wears a helm from your inventory.",
    "wear_shield": "Wears a shield from your inventory.",
    "wear_shirt": "Wears a shirt from your inventory.",
    "wear_suit": "Wears a suit from your inventory.",
    "goto_boulder": "Moves next to the closest boulder.",
    "push_boulder_east": "Pushes a boulder next to you one square east.",
    "push_boulder_north": "Pushes a boulder next to you one square north.",
    "push_boulder_south": "Pushes a boulder next to you one square south.",
    "push_boulder_west": "Pushes a boulder next to you one square west.",
    "search_corridor_for_hidden_doors": "Searches the current corridor for hidden doors.",
    "search_room_for_hidden_doors": "Searches rooms for hidden doors: goes to the most likely room and searches several spots.",
    "eat_corpse_floor": "Eats the corpse on the floor where you stand.",
    "eat_corpse_inventory": "Eats a corpse from your inventory.",
    "eat_food_inventory": "Eats food from your inventory.",
    "escape_trap": "Tries again and again to get out of a trap you are stuck in.",
    "yes": "Answers 'yes' to a question.",
    "no": "Answers 'no' to a question.",
    "cancel": "Cancels the current prompt (ESC).",
    "more": "Moves past a --More-- in the game's text.",
    "leave_shop": "Leaves the shop through its door. Drops all unpaid items.",
    "emergency_escape": "Tries to escape from a dangerous situation by teleporting.",
    "fix_trouble": (
        "Tries to fix major and minor troubles: lycanthropy, stoning, sliming, food poisoning, sickness, "
        "hallucination, blindness, stunning, confusion, strangulation, deafness."
    ),
    "pray": "Prays, unless you prayed too recently for it to be safe.",
    "heal": "Tries to heal by casting healing spells, drinking healing potions, or praying.",
    "rest_until_full_health": "Rests to regain hit points until they are full.",
    "drop_cursed_items": "Drops all cursed items from your inventory.",
    "drop_unequipped_armor": "Drops all armor you are not wearing.",
    "drop_unequipped_weapons": "Drops all weapons you are not wielding.",
    "drop_unidentified_potions": "Drops all unidentified potions from your inventory.",
    "drop_unidentified_scrolls": "Drops all unidentified scrolls from your inventory.",
    "drop_unidentified_spellbooks": "Drops all unidentified spellbooks from your inventory.",
    "drop_unidentified_wands": "Drops all unidentified wands from your inventory.",
    "identify_items_altar": "Learns whether your items are blessed, uncursed or cursed by dropping them on an altar.",
    "dip_for_excalibur": "Dips a long sword into a fountain to get Excalibur.",
}

NO_PROGRESS = "This skill did not result in any progress. Try a different one."

#: A call ends quietly after this many game turns, as NetCore's skills do,
_CALL_TURNS = 100
#: or after this many keys in a row that leave the game clock where it was.
_FROZEN_KEYS = 200

_MORE, _ESC, _DOWN = CARRIAGE_RETURN, 27, ord(">")
_ANSWERS = {"yes": ord("y"), "no": ord("n"), "cancel": _ESC, "more": _MORE}
_STEPS = dict(zip(map(ord, "hjklyubn"), [(0, -1), (1, 0), (-1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]))
_MOVES = {**_STEPS, **{key - 32: step for key, step in _STEPS.items()}}  # HJKLYUBN run the same way
_GETPOS_OPEN = {ord(";"), ord("_")}
_GETPOS_PICK = {ord("."), ord(","), ord(";"), ord(":"), _ESC}
_CALL_ITEM = re.compile(r"Call (a|an|the) .*:")
_NO_CLASSES = np.zeros(55, np.uint8)


class _Stop(BotPanic):
    """Ends the call; the game goes on."""


def _flags(obs) -> tuple:
    return tty_screen.misc_flags(obs) or (False, False, False)


def _waiting(obs) -> bool:
    """True unless the game is at the command prompt: a flag is set, the cursor is on the
    message line (a question, a count, `#`) or off the hero (a position prompt)."""
    cur = tty_screen.tty_cursor(obs)
    if any(_flags(obs)) or cur is None or cur[0] == 0:
        return True
    hero = (int(obs.blstats[1]) + 1, int(obs.blstats[0]))
    return cur != hero or tty_screen.is_getpos_text(tty_screen.obs_message(obs))


def _more_at_cursor(obs) -> bool:
    cur = tty_screen.tty_cursor(obs)
    return cur is not None and tty_screen.tty_rows(obs)[cur[0]][: cur[1]].rstrip().endswith("--More--")


def _alnum(text: str) -> str:
    return "".join(c for c in text if c.isalnum())


def _text(obs) -> str:
    """What CodeHack reads after a key, joined as nle-utils' AutoMore does: the message,
    then the lines of the window the cursor is in, then that window's marker."""
    message = tty_screen.obs_message(obs)
    parts = [message]
    win = tty_screen.find_window(obs)
    if win is not None:
        # A long message wraps onto the rows above its --More--; those are not a window.
        parts += [ln.strip() for ln in win.lines if _alnum(ln) and _alnum(ln) not in _alnum(message)]
        parts.append(win.terminator)
    elif _more_at_cursor(obs):
        parts.append("--More--")
    return "\n".join(p for p in parts if p)


class CodeHackGame:
    """The NLE env CodeHack's Bot drives (`Bot.env`), over our core env."""

    actions = ACTIONS
    _internal_index = 0

    def __init__(self, core, character: str) -> None:
        self.character = character
        self.bot = None
        self.begin(core)

    @property
    def unwrapped(self):
        return self

    @property
    def last_observation(self):
        return (np.array([0, *map(int, _flags(self.obs))]),)

    def begin(self, core) -> None:
        """Start a call from wherever the other tools left the game."""
        self._core = weakref.ref(core)
        self.obs = core._last_observation
        self.frames, self.notes, self.reward, self.done = [], [], 0.0, False
        self.turn0 = self.turn = int(self.obs.blstats[20])
        self.frozen, self.getpos, self.spells_read = 0, False, False

    def reset(self, **kwargs):
        return self._obs(""), {}

    def step(self, index):
        key = int(self.actions[index])
        if key == _DOWN or (key in _MOVES and not self.getpos and not _waiting(self.obs) and self._leaves_level(key)):
            logger.info("refused key %d", key)
            raise _Stop("a key that can leave the level")
        if key == _MORE and _cr_would_be_unknown_command(self.obs):
            return self._obs(""), 0.0, self.done, False, {}
        texts = [self._send(key)]
        while not self.done and _flags(self.obs)[2] and "--More--" in texts[-1]:
            texts[-1] = texts[-1].replace("--More--", "").strip()
            texts.append(self._send(_MORE))
        while not self.done and _flags(self.obs)[1] and _CALL_ITEM.match(tty_screen.obs_message(self.obs)):
            texts.append(self._send(_ESC))
        over = self.turn - self.turn0 >= _CALL_TURNS or self.frozen >= _FROZEN_KEYS
        # A count, `#` or position prompt in progress is finished by the next keys.
        if over and not self.done and (any(_flags(self.obs)) or not _waiting(self.obs)):
            raise _Stop("quiet stop")
        return self._obs("\n".join(t for t in texts if t)), 0.0, self.done, False, {}

    def _send(self, key: int) -> str:
        cursor_key = self.getpos and key in _MOVES
        if self.getpos:
            self.getpos = key not in _GETPOS_PICK
        elif key in _GETPOS_OPEN and not _waiting(self.obs):
            self.getpos = True
        obs, reward, terminated, truncated, _ = self._core().step(key)
        self.obs = obs
        self.frames.append(obs)
        self.reward += float(reward)
        self.done = self.done or bool(terminated or truncated)
        turn = int(obs.blstats[20])
        self.frozen = 0 if turn != self.turn else self.frozen + (not cursor_key)
        self.turn = turn
        return _text(obs)

    def _leaves_level(self, key: int) -> bool:
        dy, dx = _MOVES[key]
        pos = (int(self.obs.blstats[1]) + dy, int(self.obs.blstats[0]) + dx)
        glyphs = self.obs.glyphs
        if not (0 <= pos[0] < glyphs.shape[0] and 0 <= pos[1] < glyphs.shape[1]):
            return False
        level = getattr(self.bot, "current_level", None)
        return int(glyphs[pos]) in LEVEL_CHANGERS or (level is not None and level.leads_off_level(pos))

    def _obs(self, text: str) -> dict:
        o = self.obs
        return {"message": o.message, "blstats": o.blstats, "tty_chars": o.tty_chars, "tty_colors": o.tty_colors,
                "tty_cursor": o.tty_cursor, "glyphs": o.glyphs, "inv_strs": o.inv_strs,
                "inv_letters": o.inv_letters, "inv_glyphs": o.inv_glyphs, "inv_oclasses": _NO_CLASSES,
                "text_message": text}


#: core env -> [seeds, game, bot]: one Bot per game.
_BOTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def reset_bot_cache() -> None:
    _BOTS.clear()


def _character(core) -> str:
    """The setup character in CodeHack's form, or "@" for CodeHack to read it off ^X."""
    char = (getattr(core, "setup_character", None) or "@").lower()
    return char if char.count("-") == 3 and "@" not in char else "@"


def _entry(core) -> list:
    seeds = _npt._episode_key(core)
    entry = _BOTS.get(core)
    if entry is None or entry[0] != seeds:
        entry = _BOTS[core] = [seeds, CodeHackGame(core, _character(core)), None]
    else:
        entry[1].begin(core)
    return entry


def _ready_bot(entry) -> Bot:
    """The game's Bot, synced to the current screen; it is made on first use."""
    seeds, game, bot = entry
    if bot is not None:
        bot.current_obs, bot.current_info = game.reset()
        bot.last_obs = copy.deepcopy(bot.current_obs)
        bot.update()
        return bot
    bot = entry[2] = Bot(game, seed=int(seeds[0]) if seeds else 0)
    game.bot = bot
    bot.add_message = lambda note: game.notes.append(note)
    bot.panic(enemy_appeared)
    bot.panic(lost_hp)
    read_spells = bot.character.update_spells

    def update_spells():
        # Character.update opens the spell menu after every key until it has read it.
        if not game.spells_read and not _waiting(game.obs):
            game.spells_read = True
            read_spells()

    bot.character.update_spells = update_spells
    bot.reset()
    return bot


def run_codehack_skill(core, name: str) -> SkillResult:
    """Run one CodeHack skill until it returns or stops; the game is never ended here."""
    tool = f"ch_{name}"
    waiting = _waiting(core._last_observation)
    if name in _ANSWERS:
        if not waiting:
            return SkillResult(actions=[], feedback=f"{tool}: {NO_PROGRESS}")
        game = CodeHackGame(core, "@")
        try:
            game.step(ACTIONS.index(_ANSWERS[name]))
        except BotPanic:
            pass
        return _result(core, game, f"{tool}: done.")
    if waiting:
        return SkillResult(actions=[], feedback=f"{tool} did not run: the game is waiting for an answer to a prompt.")
    entry = _entry(core)
    game = entry[1]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            returned = getattr(strategies, name)(_ready_bot(entry))
        reason = "stopped before finishing." if returned is False else "done."
    except EnemyAppeared:
        reason = "stopped early, a monster came into reach."
    except LostHP:
        reason = "stopped early, you lost hit points."
    except (BotPanic, BotFinished) as exc:
        logger.debug("%s stopped: %r", tool, exc)
        reason = "stopped early."
    except Exception as exc:
        _BOTS.pop(core, None)
        _npt._swallowed(f"codehack.{name}", exc)
        return _result(core, game, f"{tool}: could not run here.")
    if not game.frames:
        reason = NO_PROGRESS
    notes = " ".join(dict.fromkeys(game.notes))
    return _result(core, game, f"{tool}: {reason}" + (f" {notes}" if notes else ""))


def _result(core, game: CodeHackGame, feedback: str) -> SkillResult:
    if game.frames:
        _npt.reset_agent_cache()
    return SkillResult(actions=[], feedback=feedback, pre_executed=True, pre_reward=game.reward,
                       final_obs=core._last_observation, pre_terminated=game.done,
                       pre_visible_obs=list(game.frames))


def _tool(name: str):
    def run(env, obs):
        return run_codehack_skill(env, name)

    run.__name__ = f"codehack_{name}"
    return run


for _name, _text_ in CH_DESCRIPTIONS.items():
    registry.register(f"ch_{_name}", schema={"description": _text_, "parameters": {}})(_tool(_name))

CODEHACK_TOOL_NAMES = tuple(f"ch_{name}" for name in CH_DESCRIPTIONS)
