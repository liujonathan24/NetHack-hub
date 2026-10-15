"""CodeHack in fixed mode: the published tools, the strip, and recorded play on the engine."""

import ast
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

ENV_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from _golden_replays import core_env, core_from_golden  # noqa: E402
from nethack_harness.helpers import _build_skill_adapter_callables  # noqa: E402
from nethack_harness.tools import codehack_true as ch  # noqa: E402
from nethack_harness.tools.skills import registry  # noqa: E402
import codehack  # noqa: E402
from codehack.bot.strategies.fight_monster import hostile_targets  # noqa: E402
from codehack.nle_utils.glyph import SS  # noqa: E402
from nle.nethack import actions as A  # noqa: E402

#: seedpack.LIMIT_WORDS, and words about going down, stops and counts.
LIMIT = re.compile(
    r"(?<![A-Za-z])(?:budgets?|caps?|capped|capping|limits?|limited|limiting|limitations?|unlimited|steps?[ _]used|"
    r"\d+\s*calls|calls?\s+(?:used|left|remaining|spent|made)|"
    r"max[ _]?(?:keys?|steps?|idle\w*|skill\w*|gamesteps|calls?|runs?|turns?|commands?)|"
    r"costs?|costly|free|charged?|charges|charging|remaining|per[- _](?:run|command|call))(?![A-Za-z])",
    re.IGNORECASE,
)
DESCEND = re.compile(
    r"descend|downstair|down_stair|go_down|dnstair|dnladder|\bdown\b|deeper|lower level|\bmax|timeout|"
    r"up to \d|\d+ (?:tries|times|turns|keys)|budget|\bcalls?\b", re.IGNORECASE)
BANNED_DEFS = {"descend_stairs", "escape_downstairs", "zap_down_wand_of_digging", "read_scroll_of_teleportation",
               "check_abort", "cache_overview", "cache_terrain"}

CALLS = [(n, {}) if isinstance(n, str) else n for n in (
    "ch_explore", "ch_pickup_coin", "ch_open_doors", "ch_goto_room", "ch_search_corridor_for_hidden_doors",
    ("np_press_key", {"key": "#pray<ENTER>"}), "ch_explore", "ch_no", "ch_pray", "ch_eat_food_inventory",
    ("search", {"times": 3}), "ch_rest_until_full_health", "ch_fight_melee", "ch_examine_items",
    "ch_engrave_elbereth", "ch_wear_shield", "request_map", "ch_goto_corridor", "ch_explore", "ch_explore",
    "ch_goto_room_east", "ch_escape_trap", "ch_drop_unequipped_weapons", "ch_yes", "ch_cancel", "ch_more",
    "ch_ascend_stairs", "ch_no", "ch_explore", "ch_heal", "explore",
)]
GAMES = [(0, "Val-hum-neu-fem"), (3, "Wiz-elf-cha-mal")]

DRIVER = textwrap.dedent(
    """
    import asyncio, hashlib, json, re, sys
    from nethack import load_environment
    from nethack_harness.prompt.content import content_to_text
    from nethack_harness.tools import codehack_true as ch
    seed, character, trace_dir, calls = int(sys.argv[1]), sys.argv[2] or None, sys.argv[3], json.loads(sys.argv[4])
    limit = re.compile(sys.argv[5], re.IGNORECASE)

    async def main():
        env = load_environment(variant="BBOX_MIN", map_detail="full", auto_dismiss=False,
                               skill_set="codehack,request_map,search", character=character,
                               tune={"reveal_map": 1.0}, n_examples=1, seed=seed, self_dispatch=True, trace_dir=trace_dir)
        state = await env.setup_state({"task": {"seed": seed}, "info": {}})
        out = {"feedback": [], "depth": [], "clean": [], "limit_hits": [], "character": state["env"].setup_character}
        for name, args in calls:
            text = content_to_text(await env._apply_tool_call(state, name, args))
            obs = state["raw_obs"]
            out["feedback"].append(text.split("\\n")[0])
            out["depth"].append(int(obs.blstats[12]))
            out["clean"].append(not ch._waiting(obs))
            out["limit_hits"] += [line for line in text.splitlines() if limit.search(line)]
        out["terminated"] = bool(state.get("terminated"))
        out["final"] = [obs.blstats.tolist(), hashlib.sha1(obs.glyphs.tobytes()).hexdigest(),
                        hashlib.sha1(obs.tty_chars.tobytes()).hexdigest()]
        print(json.dumps(out))

    asyncio.run(main())
    """
)


def _names(skill_set):
    return {a.__name__ for a in _build_skill_adapter_callables(skill_set)}


def test_codehack_preset_is_the_77_skills_plus_the_key():
    assert len(ch.CODEHACK_TOOL_NAMES) == 77
    assert _names("codehack") == set(ch.CODEHACK_TOOL_NAMES) | {"np_press_key"}
    assert _names("codehack,request_map,search") == _names("codehack") | {"request_map", "search"}
    assert not {n for n in _names("full") | _names("np_core") if n.startswith("ch_")}


@pytest.mark.parametrize("name", ch.CODEHACK_TOOL_NAMES)
def test_tool_takes_no_arguments_and_reads_clean(name):
    schema = registry.all_schemas()[name]
    assert schema["parameters"] == {} and schema["description"] == ch.CH_DESCRIPTIONS[name[3:]]
    text = f"{name} {schema['description']}"
    assert not LIMIT.search(text) and not DESCEND.search(text), text


def test_feedback_lines_read_clean():
    tree = ast.parse(pathlib.Path(ch.__file__).read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("run_codehack_skill", "_result")]
    texts = [ch.NO_PROGRESS] + [n.value for f in funcs for stmt in f.body[1:] for n in ast.walk(stmt)
                                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert len(texts) > 8 and not [t for t in texts if LIMIT.search(t) or DESCEND.search(t)]


def test_installed_fork_has_no_way_down_and_no_hidden_keys():
    root = pathlib.Path(codehack.__file__).parent
    hits = []
    for path in sorted((root / "bot").rglob("*.py")) + sorted((root / "nle_utils").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name in BANNED_DEFS:
                hits.append((path.name, node.name))
            elif isinstance(node, ast.Attribute) and node.attr in ("DOWN", "OVERVIEW", "EXTCMD"):
                hits.append((path.name, node.attr))
            elif isinstance(node, ast.Constant) and (node.value == ">" or node.value == 62 and path.name != "glyph.py"):
                hits.append((path.name, node.value))
    assert not hits, hits
    assert not hasattr(ch.strategies, "descend_stairs") and not hasattr(ch.Bot, "check_abort")


def _turn(env):
    return int(env._last_observation.blstats[20])


def _hp(env):
    return int(env._last_observation.blstats[10])


def _messages(result):
    from nethack_harness.prompt.tty_screen import obs_message

    return [obs_message(o) for o in result.pre_visible_obs or []]


def test_guard_refuses_down_and_known_trap_doors():
    env = core_env()
    obs, t0 = env._last_observation, _turn(env)
    y, x = int(obs.blstats[1]), int(obs.blstats[0])
    glyphs = obs.glyphs.copy()
    glyphs[y, x + 1] = SS.S_trap_door
    game = ch.CodeHackGame(env, "val-hum-neu-fem")
    game.obs = dataclasses.replace(obs, glyphs=glyphs)
    for key in (ord(">"), ord("l"), ord("L")):
        with pytest.raises(ch._Stop):
            game.step(ch.ACTIONS.index(key))
    bot = ch._ready_bot(ch._entry(env))
    bot.current_level.known_traps[y, x + 1] = SS.S_trap_door
    bot.current_level.update(bot.glyphs, bot.blstats)
    assert not bot.current_level.walkable[y, x + 1]
    with pytest.raises(ch.BotPanic):
        bot.pathfinder.direction((y, x + 1))
    assert game.frames == [] and _turn(env) == t0


def test_stops_end_the_call_and_never_the_game(monkeypatch):
    monkeypatch.delenv("NETHACK_STRICT_SKILL_ERRORS", raising=False)
    env = core_env()

    def rest(bot):
        while True:
            bot.search(20)

    def spin(bot):
        while True:
            bot.step(ord(":"))

    def raises(exc):
        def skill(bot):
            raise exc
        return skill

    cases = [(rest, "stopped early."), (spin, "stopped early."), (raises(ch.BotPanic("x")), ch.NO_PROGRESS),
             (raises(ch.BotFinished()), ch.NO_PROGRESS), (raises(ValueError("x")), "could not run here.")]
    turns, keys = [], []
    for i, (skill, text) in enumerate(cases):
        monkeypatch.setattr(ch.strategies, f"t{i}", skill, raising=False)
        t0 = _turn(env)
        res = ch.run_codehack_skill(env, f"t{i}")
        assert res.feedback == f"ch_t{i}: {text}" and not res.pre_terminated and not ch._waiting(env._last_observation)
        turns.append(_turn(env) - t0)
        keys.append(len(res.pre_visible_obs))
    assert ch._CALL_TURNS <= turns[0] < ch._CALL_TURNS + 40 and turns[1:] == [0, 0, 0, 0]
    assert keys[1] == ch._FROZEN_KEYS and keys[2:] == [0, 0, 0]
    for _ in range(30):
        assert ch.run_codehack_skill(env, "pray").feedback.startswith(f"ch_pray: {ch.NO_PROGRESS}")
    assert int(env._last_observation.blstats[10]) > 0 and _turn(env) == sum(turns) + 1


def _play(seed, character, trace_dir, hashseed, calls=CALLS):
    env = {**os.environ, "PYTHONHASHSEED": hashseed,
           "PYTHONPATH": os.pathsep.join([str(ENV_DIR), os.environ.get("PYTHONPATH", "")])}
    argv = [sys.executable, "-c", DRIVER, str(seed), character or "", str(trace_dir), json.dumps(calls), LIMIT.pattern]
    proc = subprocess.run(argv, cwd=ENV_DIR, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    (record,) = pathlib.Path(trace_dir).glob("*.ndjson")
    actions = [json.loads(line)["actions"] for line in record.read_text().splitlines() if line.strip()]
    return json.loads(proc.stdout.splitlines()[-1]), actions


def _check_record(out, actions, calls):
    """Every key recorded and replayable; no ^X, no '>', no Dlvl gain and at most one spell-menu read in ch_ calls."""
    names = [n for n, _ in calls]
    keys = [a["bytes"] for a in actions]
    assert all(a["recorded"] and a["replayable"] for a in actions)
    ch_keys = [k for n, k in zip(names, keys) if n.startswith("ch_")]
    assert all(62 not in k and 24 not in k for k in ch_keys)
    assert sum(k[i:i + 2] == [ord("Z"), 13] for k in ch_keys for i in range(len(k))) <= 1
    assert all(d <= prev for n, prev, d in zip(names, [1] + out["depth"], out["depth"]) if n.startswith("ch_"))
    assert not out["terminated"] or out["final"][0][10] == 0
    assert out["limit_hits"] == [] and out["character"].count("-") == 3
    return keys


def _replay_matches(seed, character, keys, final):
    env = core_env(seed, character)
    obs = env._last_observation
    for key in (k for ks in keys for k in ks):
        obs = env.step(key)[0]
    return [obs.blstats.tolist(), hashlib.sha1(obs.glyphs.tobytes()).hexdigest(),
            hashlib.sha1(obs.tty_chars.tobytes()).hexdigest()] == final


@pytest.mark.parametrize("seed, character", GAMES)
def test_scripted_game_repeats_replays_and_never_goes_down(tmp_path, seed, character):
    out, actions = _play(seed, character, tmp_path / "a", "0")
    assert (out, actions) == _play(seed, character, tmp_path / "b", "1")
    keys = _check_record(out, actions, CALLS)
    # A prompt left open by a pressed key stops the next skill; an answer skill answers it.
    assert out["feedback"][6] == "[ch_explore did not run: the game is waiting for an answer to a prompt.]"
    assert keys[6] == [] and keys[7] == [ord("n")]
    # Menus and --More-- chains leave a clean screen; the no-return question `<` asks on
    # Dlvl 1 is left to the model, which answers it with the next call.
    assert all(clean for i, clean in enumerate(out["clean"]) if i not in (5, 6, 26))
    assert out["clean"][26] == (keys[26][-1:] != [ord("<")]) and keys[27] == [ord("n")]
    assert out["feedback"][-1].startswith("[Tool 'explore' is not available") and keys[-1] == []
    assert _replay_matches(seed, character, keys, out["final"])


def test_unpinned_character_plays_without_reading_attributes(tmp_path):
    """character=None: the game picks the role; CodeHack learns it from the welcome, never from ^X."""
    out, actions = _play(5, None, tmp_path, "0", CALLS[:12])
    keys = _check_record(out, actions, CALLS[:12])
    assert out["feedback"][6] == "[ch_explore did not run: the game is waiting for an answer to a prompt.]"
    assert _replay_matches(5, None, keys, out["final"])


def test_bot_memory_survives_an_error(monkeypatch):
    monkeypatch.delenv("NETHACK_STRICT_SKILL_ERRORS", raising=False)
    env = core_env()
    ch.run_codehack_skill(env, "explore")
    for _ in range(400):
        if _turn(env) > 320:
            break
        env.step(13 if ch._flags(env._last_observation)[2] else ord("s"))
    bot = ch._BOTS[env][2]
    prayed = ch.run_codehack_skill(env, "pray")
    assert prayed.feedback == "ch_pray: done." and bot.last_prayer is not None
    assert any("You begin praying" in m for m in _messages(prayed))

    def boom(bot):
        raise ValueError("x")

    monkeypatch.setattr(ch.strategies, "boom", boom, raising=False)
    assert ch.run_codehack_skill(env, "boom").feedback == "ch_boom: could not run here."
    assert ch._BOTS[env][2] is bot
    again = ch.run_codehack_skill(env, "pray")
    assert again.feedback.startswith(f"ch_pray: {ch.NO_PROGRESS}") and again.pre_visible_obs == []


@pytest.mark.parametrize("character, heals", [("Hea-hum-neu-fem", True), ("Pri-hum-law-mal", False)])
def test_casters_heal_by_spell_or_potion(character, heals):
    env = core_env(character=character)
    env.modify(hp=2)
    feedback = [ch.run_codehack_skill(env, "heal").feedback for _ in range(3)]
    assert all("could not run here" not in f and "question" not in f for f in feedback), feedback
    assert not ch._waiting(env._last_observation)
    spells = ch._BOTS[env][2].character.known_spells
    assert spells and all(isinstance(s.fail, float) and isinstance(s.level, int) for s in spells.values())
    assert (_hp(env) > 2) == heals, (feedback, _hp(env))


def test_fight_skills_ignore_the_shopkeeper():
    env = core_from_golden("shop_next_door")
    bot = ch._ready_bot(ch._entry(env))
    assert "shopkeeper" in {e.name for e in bot.entities}
    assert "shopkeeper" not in {e.name for _, e in hostile_targets(bot)}
    before = (_turn(env), _hp(env), env._last_observation.blstats[:2].tolist())
    for name in ("fight_melee", "fight_ranged"):
        result = ch.run_codehack_skill(env, name)
        assert result.feedback == f"ch_{name}: {ch.NO_PROGRESS}" and not result.pre_terminated
        assert not any("store" in m or "angry" in m for m in _messages(result))
    assert (_turn(env), _hp(env), env._last_observation.blstats[:2].tolist()) == before


def test_open_question_is_reported(monkeypatch):
    env = core_env()

    def ask(bot):
        bot.step(A.Command.PRAY)
        return True

    monkeypatch.setattr(ch.strategies, "ask", ask, raising=False)
    assert ch.run_codehack_skill(env, "ask").feedback == "ch_ask: stopped at a question."
    assert ch._flags(env._last_observation)[0]
    assert ch.run_codehack_skill(env, "no").feedback == "ch_no: done." and not ch._waiting(env._last_observation)
