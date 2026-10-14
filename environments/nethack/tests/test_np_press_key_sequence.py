"""np_press_key: key sequences, the full key surface, ENTER handling, and "Really attack?"."""

from types import SimpleNamespace

import numpy as np
import pytest

from nethack_harness.helpers import _cr_would_be_unknown_command, _ext_cmd_prompt_open
from nethack_harness.tools import netplay_true as npt  # puts vendor/ on sys.path
from nethack_harness.tools.skills import registry
from netplay.core.agent_base import Step, StepData
from netplay.nethack_agent import skills
from netplay.nethack_utils.nle_wrapper import RawKeyPress


def _codes(key):
    return [int(k) for k in RawKeyPress.parse_sequence(key)]


def test_every_printable_ascii_key_is_pressable():
    for c in range(32, 127):
        assert _codes(chr(c)) == [c] and int(RawKeyPress.parse(chr(c))) == c


@pytest.mark.parametrize("key,expected", [
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
    from nethack import _press_key_sends_gt as f

    assert f(">") and f(" > ") and f("s>") and f("<ESC>>") and f("<GT>")
    assert not f("<") and not f("qf") and not f("<FOO>>")


class _FakeAgent:
    def __init__(self, done_after=None, yn_prompt=None, misc=(0, 0, 0)):
        self.sent = []
        self.done_after = done_after
        self.yn_prompt = yn_prompt
        self.blstats = SimpleNamespace(hitpoints=10, x=1, y=1)
        self.last_observation = {"tty_chars": np.full((24, 80), ord(" "), dtype=np.uint8),
                                 "misc": np.array(misc), "tty_cursor": np.array([5, 5])}
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(waiting_for_yn=False))
        self.current_game_message = ""

    def step(self, action, thoughts=None):
        self.sent.append(int(action))
        prompt = self.yn_prompt if len(self.sent) == 1 else None  # the first key raises the prompt
        self.env.unwrapped.waiting_for_yn = prompt is not None
        self.current_game_message = prompt or ""
        done = self.done_after is not None and len(self.sent) >= self.done_after
        return Step.act(StepData(action, None, 0.0, done, False, {}, []), thoughts)


def test_press_key_sends_one_step_per_key_and_reports_all_sent():
    agent = _FakeAgent(misc=(0, 1, 0))  # a text prompt reads a line, so its ENTER is sent
    steps = list(skills.press_key(agent, key="E-Elbereth<ENTER>"))
    assert agent.sent == _codes("E-Elbereth<ENTER>")
    assert steps[-1].has_completed() and "Pressed all 11 keys" in steps[-2].thoughts


def test_press_key_stops_when_the_game_ends():
    agent = _FakeAgent(done_after=3)
    steps = list(skills.press_key(agent, key="abcdef"))
    assert agent.sent == [ord(c) for c in "abc"]
    assert "remaining 3 keys were not sent" in (steps[-1].thoughts or "")


def test_bad_key_sends_nothing():
    agent = _FakeAgent()
    steps = list(skills.press_key(agent, key="ab<FOO>"))
    assert agent.sent == [] and steps[-1].status.name == "error"


def _returned(gen):
    """The value a generator returns once it is run to its end."""
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def test_really_attack_prompt_is_answered_no():
    agent = _FakeAgent(yn_prompt="Really attack Sablja? [yn] (n)")
    agent.current_level = SimpleNamespace(get_monster_glyph=lambda x, y: 1234)
    assert _returned(skills._step_declining_attack(agent, 108, 2, 1)) is True
    assert agent.sent == [108, ord("n")] and 1234 in agent.known_peaceful_glyphs


def _obs(top, cursor):
    rows = [top.ljust(80)] + [" " * 80] * 23
    return SimpleNamespace(misc=np.zeros(3), tty_cursor=np.asarray(cursor),
                           tty_chars=np.array([[ord(c) for c in r] for r in rows], dtype=np.uint8))


def test_stray_cr_guard_at_the_extended_command_prompt():
    # The `#` prompt raises no misc flag; it is told apart by the cursor on the top line.
    assert _cr_would_be_unknown_command(_obs("# offer", cursor=(0, 7))) is False
    assert _cr_would_be_unknown_command(_obs("# offer", cursor=(12, 40))) is True  # an old `#` line


def _press(env, key):
    return registry.call("np_press_key", env, None, key=key)


def _screen(res):
    rows = np.asarray(res.final_obs.tty_chars).astype(np.uint8)
    return "\n".join(bytes(r).decode("latin-1").rstrip() for r in rows)


@pytest.fixture
def env():
    from nethack_core.env import NetHackCoreEnv

    npt.reset_agent_cache()
    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(42, 42)
    env.reset()
    _press(env, "ESC")  # the intro --More--
    yield env
    env.close()


def test_live_elbereth_in_one_call(env):
    assert "Pressed all 11 keys" in _press(env, "E-Elbereth<ENTER>").feedback
    assert "Elbereth" in _screen(_press(env, ":"))


def test_live_single_dash_answers_write_with_prompt(env):
    _press(env, "E")
    res = _press(env, "-")
    assert "error" not in res.feedback and "write in the dust" in _screen(res)


def test_live_guard_recognises_the_extended_command_prompt(env):
    obs = _press(env, "#pray").final_obs
    assert _ext_cmd_prompt_open(obs) and not _cr_would_be_unknown_command(obs)


@pytest.mark.parametrize("calls,sent", [
    (["#pray<ENTER>"], True), (["#pray", "<ENTER>"], True), (["#", "p", "ENTER"], True),
    (["<ENTER>"], False), ([";", "<ENTER>"], False),
], ids=["one-call", "two-calls", "split", "command-prompt", "cursor-prompt"])
def test_live_enter_submits_the_extended_command_or_is_reported_as_not_sent(env, calls, sent):
    for key in calls:
        res = _press(env, key)
    assert ("ENTER was not sent" not in res.feedback) is sent
    assert ("Are you sure you want to pray" in _screen(res)) is sent
