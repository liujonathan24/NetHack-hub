"""np_press_key key sequences, the full key surface, the extended-command
ENTER, and peaceful-monster handling in the vendored pathing skills.

See vendor/netplay/nethack_utils/nle_wrapper.py (RawKeyPress.parse_sequence)
and vendor/netplay/nethack_agent/skills.py (press_key, move_to).
"""

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.tools.skills import registry  # noqa: E402
from nethack_harness.tools import netplay_true as npt  # noqa: E402,F401  (vendor/ on sys.path)
from netplay.nethack_utils.nle_wrapper import RawKeyPress  # noqa: E402
import netplay.nethack_agent.skills as skills  # noqa: E402
from netplay.core.agent_base import Step, StepData  # noqa: E402


def _codes(key):
    return [int(k) for k in RawKeyPress.parse_sequence(key)]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_every_printable_ascii_key_is_pressable():
    for c in range(32, 127):
        assert _codes(chr(c)) == [c]
        assert int(RawKeyPress.parse(chr(c))) == c


@pytest.mark.parametrize("key,expected", [
    (">", [62]), ("<", [60]), (",", [44]), ("-", [45]), ("|", [124]),
    ("ESC", [27]), ("esc", [27]), ("Enter", [13]), ("SPACE", [32]),
    ("qf", [113, 102]),
    ("E-Elbereth<ENTER>", [69, 45] + [ord(c) for c in "Elbereth"] + [13]),
    ("#loot<ENTER>", [35] + [ord(c) for c in "loot"] + [13]),
    ("<C-d>", [4]), ("<c-D>h", [4, 104]),
    ("<esc><Space><TAB><backspace>", [27, 32, 9, 8]),
    ("<LT><GT>", [60, 62]), ("<<", [60, 60]), ("a<b", [97, 60, 98]),
])
def test_parse_sequence(key, expected):
    assert _codes(key) == expected


@pytest.mark.parametrize("bad", ["", "<FOO>", "<C-1>", "é", "abé"])
def test_parse_sequence_rejects_without_partial_send(bad):
    with pytest.raises(ValueError):
        RawKeyPress.parse_sequence(bad)


def test_descent_gate_sees_gt_anywhere_in_a_sequence():
    import nethack as nethack_mod
    f = nethack_mod._press_key_sends_gt
    assert f(">") and f(" > ") and f("s>") and f("<ESC>>") and f("<GT>")
    assert not f("<") and not f("qf") and not f("<FOO>>")


# --------------------------------------------------------------------------
# press_key against a fake agent (game end partway through)
# --------------------------------------------------------------------------

def _run(skill, agent, **kw):
    return list(getattr(skill, "fn", skill)(agent, **kw))


class _FakeAgent:
    def __init__(self, done_after=None, yn_prompt=None):
        self.sent = []
        self.done_after = done_after
        self.blstats = SimpleNamespace(hitpoints=10, x=1, y=1)
        rows = np.full((24, 80), ord(" "), dtype=np.uint8)
        self.last_observation = {"tty_chars": rows, "misc": np.zeros(3), "tty_cursor": np.array([5, 5])}
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(waiting_for_yn=False))
        self.yn_prompt = yn_prompt
        self.current_game_message = ""

    def step(self, action, thoughts=None):
        self.sent.append(int(action))
        done = self.done_after is not None and len(self.sent) >= self.done_after
        if self.yn_prompt and len(self.sent) == 1:
            self.env.unwrapped.waiting_for_yn = True
            self.current_game_message = self.yn_prompt
        else:
            self.env.unwrapped.waiting_for_yn = False
            self.current_game_message = ""
        data = StepData(action=action, observation=None, reward=0.0, done=done,
                        truncated=False, info={}, events=[])
        return Step.act(data, thoughts)


def test_press_key_sends_one_step_per_key_and_reports_all_sent():
    agent = _FakeAgent()
    steps = _run(skills.press_key, agent, key="E-Elbereth<ENTER>")
    assert agent.sent == _codes("E-Elbereth<ENTER>")
    assert steps[-1].has_completed()
    assert "Pressed all 11 keys" in steps[-2].thoughts  # on the last key's step


def test_press_key_stops_when_the_game_ends():
    agent = _FakeAgent(done_after=3)
    steps = _run(skills.press_key, agent, key="abcdef")
    assert agent.sent == [ord(c) for c in "abc"]
    assert "remaining 3 keys were not sent" in (steps[-1].thoughts or "")


def test_bad_key_sends_nothing():
    agent = _FakeAgent()
    steps = _run(skills.press_key, agent, key="ab<FOO>")
    assert agent.sent == [] and steps[-1].status.name == "error"


def test_really_attack_prompt_is_answered_no():
    agent = _FakeAgent(yn_prompt="Really attack Sablja? [yn] (n)")
    agent.current_level = SimpleNamespace(get_monster_glyph=lambda x, y: 1234)
    gen = skills._step_declining_attack(agent, 108, 2, 1)
    out = []
    try:
        while True:
            out.append(next(gen))
    except StopIteration as stop:
        declined = stop.value
    assert declined is True
    assert agent.sent == [108, ord("n")]
    assert 1234 in agent.known_peaceful_glyphs
    assert ord("y") not in agent.sent


# --------------------------------------------------------------------------
# Live engine
# --------------------------------------------------------------------------

def _fresh_env():
    from nethack_core.env import NetHackCoreEnv
    npt.reset_agent_cache()
    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(42, 42)
    env.reset()
    registry.call("np_press_key", env, None, key="ESC")  # intro --More--
    return env


def _screen(res) -> str:
    """The terminal text the skill left on screen. Game messages are rendered
    into the observation, not the skill's feedback string, so read them here."""
    import numpy as np
    rows = np.asarray(res.final_obs.tty_chars).astype(np.uint8)
    return "\n".join(bytes(r).decode("latin-1").rstrip() for r in rows)


def test_live_fingers_elbereth_in_one_call():
    env = _fresh_env()
    try:
        res = registry.call("np_press_key", env, None, key="E-Elbereth<ENTER>")
        assert "Pressed all 11 keys" in res.feedback, res.feedback
        res = registry.call("np_press_key", env, None, key=":")
        assert "Elbereth" in _screen(res), _screen(res)
    finally:
        env.close()


def test_live_single_dash_answers_write_with_prompt():
    env = _fresh_env()
    try:
        registry.call("np_press_key", env, None, key="E")
        res = registry.call("np_press_key", env, None, key="-")
        assert "error" not in res.feedback, res.feedback
        assert "write in the dust" in _screen(res), _screen(res)
        registry.call("np_press_key", env, None, key="ESC")
    finally:
        env.close()


def test_live_extended_command_enter_submits():
    """The #-prompt does not set in_getlin, so a CR there used to be dropped
    as stray and '# pray' never ran. ENTER must submit it."""
    env = _fresh_env()
    try:
        res = registry.call("np_press_key", env, None, key="#pray<ENTER>")
        assert "Are you sure you want to pray" in _screen(res), _screen(res)
        registry.call("np_press_key", env, None, key="n")
        # Split over calls, as the play-tester did.
        registry.call("np_press_key", env, None, key="#")
        registry.call("np_press_key", env, None, key="p")
        res = registry.call("np_press_key", env, None, key="ENTER")
        assert "Are you sure you want to pray" in _screen(res), _screen(res)
        registry.call("np_press_key", env, None, key="n")
    finally:
        env.close()
