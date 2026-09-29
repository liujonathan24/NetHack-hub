"""ENTER reaches the `#` extended-command prompt on every path, and an ENTER the
harness drops is reported to the agent instead of looking pressed.

The `#` prompt raises none of the engine's `misc` input flags, so the stray-CR
guard (helpers._cr_would_be_unknown_command) used to drop the CR that submits it
unless np_press_key had rewritten it to LF first. Cursor (getpos) prompts also
leave the flags clear; an ENTER there does nothing and is dropped, and the agent
used to be told "completed" with no key sent.
"""

import pathlib
import sys
from types import SimpleNamespace

import numpy as np

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.helpers import _cr_would_be_unknown_command, _ext_cmd_prompt_open  # noqa: E402
from nethack_harness.tools.skills import registry  # noqa: E402
from nethack_harness.tools import netplay_true as npt  # noqa: E402


def _obs(top: str, cursor=(0, 0), misc=(0, 0, 0)):
    rows = [top.ljust(80)[:80]] + [" " * 80] * 23
    return SimpleNamespace(
        misc=np.asarray(misc),
        tty_cursor=np.asarray(cursor),
        tty_chars=np.array([[ord(c) for c in r] for r in rows], dtype=np.uint8),
    )


def test_guard_passes_cr_at_the_extended_command_prompt():
    assert _ext_cmd_prompt_open(_obs("# offer", cursor=(0, 7)))
    assert not _cr_would_be_unknown_command(_obs("# offer", cursor=(0, 7)))


def test_guard_still_drops_a_stray_cr():
    # Command prompt: flags clear, cursor on the map.
    assert _cr_would_be_unknown_command(_obs("You see here a dagger.", cursor=(12, 40)))
    # A '#' left on the top line from an earlier message, cursor back on the map.
    assert _cr_would_be_unknown_command(_obs("# offer", cursor=(12, 40)))


def test_guard_passes_cr_to_any_flagged_prompt():
    assert not _cr_would_be_unknown_command(_obs("Really attack? [yn] (n)", cursor=(0, 24), misc=(1, 0, 0)))


def _fresh_env():
    from nethack_core.env import NetHackCoreEnv
    npt.reset_agent_cache()
    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(42, 42)
    env.reset()
    registry.call("np_press_key", env, None, key="ESC")  # intro --More--
    return env


def test_live_guard_recognises_the_real_extended_command_prompt():
    env = _fresh_env()
    try:
        res = registry.call("np_press_key", env, None, key="#pray")
        assert _ext_cmd_prompt_open(res.final_obs)
        assert not _cr_would_be_unknown_command(res.final_obs)
        registry.call("np_press_key", env, None, key="ESC")
    finally:
        env.close()


def test_live_enter_at_the_command_prompt_is_reported_not_sent():
    env = _fresh_env()
    try:
        res = registry.call("np_press_key", env, None, key="<ENTER>")
        assert "ENTER was not sent" in res.feedback, res.feedback
    finally:
        env.close()


def test_live_enter_at_a_cursor_prompt_is_reported_not_sent():
    env = _fresh_env()
    try:
        registry.call("np_press_key", env, None, key=";")  # farlook: "Pick an object."
        res = registry.call("np_press_key", env, None, key="<ENTER>")
        assert "ENTER was not sent" in res.feedback, res.feedback
        registry.call("np_press_key", env, None, key="ESC")
    finally:
        env.close()


def test_live_enter_still_submits_the_extended_command():
    env = _fresh_env()
    try:
        registry.call("np_press_key", env, None, key="#pray")
        res = registry.call("np_press_key", env, None, key="<ENTER>")
        assert "ENTER was not sent" not in res.feedback, res.feedback
        screen = "\n".join(bytes(r).decode("latin-1") for r in np.asarray(res.final_obs.tty_chars).astype(np.uint8))
        assert "Are you sure you want to pray" in screen, screen
        registry.call("np_press_key", env, None, key="n")
    finally:
        env.close()
