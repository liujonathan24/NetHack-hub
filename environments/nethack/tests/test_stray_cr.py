"""The stray `^M`.

Run 1's Prime Agent rollouts carried `Unknown command '^M'.` on the top line for
31 turns of one seed and 17 of another. Every occurrence was *originated* by the
same twelve-byte action list -- `E - E l b e r e t h \r \r`, i.e.
`engrave_elbereth` -- whose second, unconditional carriage return reached
NetHack's command dispatcher instead of a prompt. `rhack()` has no binding for
byte 13, so it takes its bad-command branch and `visctrl()` renders the byte as
`^M`. NetHack's top line is not cleared until something repaints it, so the one
stray CR then showed up in the observation of every following `move_to` turn --
which is why it looked like `move_to` was the culprit, and why one agent wrote
"when move_to fails to advance turn, call engrave_elbereth to unstick the game
(clears ^M from input buffer)" into its notes and burned an 18-turn streak on it.

Three independent fixes, tested here:

1. the source -- `engrave_elbereth` no longer emits the second CR;
2. the funnel -- a CR that would reach command context is swallowed as a no-op
   in the harness's own step loop (nethack.py `env_response`, interface.py
   `TypedNetHackInterface.step`), so no *other* hand-written skill (or a future
   one) can resurface the message either;
3. the `netplay_true` funnel -- the 31 vendored NetPlay skills (`np_` prefix)
   drive the engine through a *separate* path (`netplay_true.py`'s
   `run_netplay_skill` / `NetPlayEngineEnv.step`) that bypasses (1) and (2)
   entirely, since a pre-executed `SkillResult` reports `actions=[]` -- the
   engine steps already happened by the time `env_response` sees it. Upstream's
   own `press_key`/`type_text` skills can pass a bare CR (`RawKeyPress.
   KEYPRESS_ENTER = 13`) straight through, so `np_press_key(key="enter")`
   outside a prompt reproduced the exact same `Unknown command '^M'.` -- same
   guard, applied at `NetPlayEngineEnv.step`, the one place every netplay_true
   skill's engine calls funnel through.
"""
from __future__ import annotations

import pytest

from nethack_core.env import NetHackCoreEnv
from nethack_harness.helpers import (
    CARRIAGE_RETURN,
    _cr_would_be_unknown_command,
)
from nethack_harness.tools.skills import engrave_elbereth


def _fresh_env(seed: int = 19) -> NetHackCoreEnv:
    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(seed, seed)
    obs = env.reset()
    return env, obs[0] if isinstance(obs, tuple) else obs


def _top_line(obs) -> str:
    tty = obs["tty_chars"] if isinstance(obs, dict) else obs.tty_chars
    return "".join(chr(int(c)) for c in tty[0]).rstrip()


# --- 1. the source ---------------------------------------------------------


def test_engrave_elbereth_emits_exactly_one_carriage_return():
    """The trailing "any prompt close" CR is gone; the getlin submit stays."""
    result = engrave_elbereth(None, None)
    assert result.actions.count(CARRIAGE_RETURN) == 1
    assert result.actions[-1] == CARRIAGE_RETURN
    # The rest of the sequence is untouched: E, -, then the word.
    assert result.actions[:2] == [ord("E"), ord("-")]
    assert result.actions[2:-1] == [ord(c) for c in "Elbereth"]


def test_engrave_elbereth_action_list_is_no_longer_the_run1_signature():
    """The exact byte list every run-1 `^M` was traced back to."""
    run1_signature = [69, 45, 69, 108, 98, 101, 114, 101, 116, 104, 13, 13]
    assert engrave_elbereth(None, None).actions != run1_signature


# --- 2. the discriminator --------------------------------------------------


def test_misc_flags_discriminate_command_context_from_prompts():
    """`misc` is (in_yn_function, in_getlin, xwaitingforspace).

    Only the all-clear state is command context, and only there is a CR stray.
    """
    env, obs = _fresh_env()
    try:
        assert _cr_would_be_unknown_command(obs) is True  # idle, awaiting a command

        obs = env.step(ord("E"))[0]  # "What do you want to write with? [- or ?*]"
        assert _cr_would_be_unknown_command(obs) is False

        obs = env.step(ord("-"))[0]  # "You write in the dust...--More--"
        assert _cr_would_be_unknown_command(obs) is False

        obs = env.step(CARRIAGE_RETURN)[0]  # getlin: "write what in the dust here?"
        assert _cr_would_be_unknown_command(obs) is False

        obs = env.step(CARRIAGE_RETURN)[0]  # back to command context
        assert _cr_would_be_unknown_command(obs) is True

        obs = env.step(ord("i"))[0]  # inventory menu, "(end)"
        assert _cr_would_be_unknown_command(obs) is False
    finally:
        env.close()


def test_detector_fails_open_on_an_unreadable_observation():
    """No `misc` -> assume a prompt may be open and send the CR unchanged."""
    assert _cr_would_be_unknown_command(None) is False
    assert _cr_would_be_unknown_command({}) is False
    assert _cr_would_be_unknown_command(object()) is False


# --- 3. the defect itself --------------------------------------------------


def test_a_bare_cr_in_command_context_still_produces_unknown_command():
    """The engine behaviour we are guarding against, pinned so it cannot drift.

    If NetHack ever starts ignoring a bare CR on its own, this test fails and
    the funnel guard can be revisited.
    """
    env, obs = _fresh_env()
    try:
        assert "Unknown command" not in _top_line(obs)
        obs = env.step(CARRIAGE_RETURN)[0]
        assert "Unknown command '^M'" in _top_line(obs)
    finally:
        env.close()


# The engrave sequence, byte for byte, as run 1 sent it: `E - Elbereth \r \r`.
# It lands back in command context, and it is the trailing CR -- the one this
# task removed -- that was the stray. To reproduce the *condition* that CR was
# in, without depending on whether a given seed puts a --More-- in the way, the
# tests below drive the sequence to completion and then issue one more CR: that
# is exactly "a CR arriving while the engine is awaiting a command".
_RUN1_ENGRAVE_SEQUENCE = [69, 45, 69, 108, 98, 101, 114, 101, 116, 104, 13, 13]


def test_a_trailing_cr_after_engrave_produces_the_run1_message():
    """Control: without the guard, the extra CR is the `^M` the agents saw."""
    env, obs = _fresh_env()
    try:
        for action in _RUN1_ENGRAVE_SEQUENCE:
            obs = env.step(action)[0]
        assert _cr_would_be_unknown_command(obs) is True  # command context
        assert "Unknown command '^M'" not in _top_line(obs)
        obs = env.step(CARRIAGE_RETURN)[0]  # the stray CR, unguarded
        assert "Unknown command '^M'" in _top_line(obs)
    finally:
        env.close()


def test_funnel_swallow_prevents_unknown_command_for_that_same_cr():
    """Same trajectory, but every CR goes through the funnel rule first."""
    env, obs = _fresh_env()
    try:
        swallowed = 0
        for action in [*_RUN1_ENGRAVE_SEQUENCE, CARRIAGE_RETURN]:
            if action == CARRIAGE_RETURN and _cr_would_be_unknown_command(obs):
                swallowed += 1
                continue
            obs = env.step(action)[0]
            assert "Unknown command '^M'" not in _top_line(obs), (
                f"stray ^M surfaced after byte {action}"
            )
        assert swallowed == 1, f"expected exactly one stray CR, swallowed {swallowed}"
    finally:
        env.close()


def test_new_engrave_sequence_never_surfaces_the_message_through_the_funnel():
    """The shipped skill, replayed live through the funnel, three times over."""
    env, obs = _fresh_env()
    try:
        for _ in range(3):
            for action in engrave_elbereth(None, None).actions:
                if action == CARRIAGE_RETURN and _cr_would_be_unknown_command(obs):
                    continue
                obs = env.step(action)[0]
                assert "Unknown command '^M'" not in _top_line(obs)
    finally:
        env.close()


# --- 4. the prompt CRs that must survive -----------------------------------


@pytest.mark.parametrize("skill_name", ["descend", "ascend", "pray"])
def test_skills_that_need_a_cr_still_carry_one(skill_name):
    """The guard is conditional, not a blanket filter: these still emit a CR.

    `descend`/`ascend` prefix `>`/`<` with a CR to clear a pending --More--, and
    `pray` submits `#pray` with one. Those CRs are only dropped when the engine
    says no prompt is open, which is checked live in the funnel.
    """
    from nethack_harness.tools import skills as skills_mod

    result = getattr(skills_mod, skill_name)(None, None)
    assert CARRIAGE_RETURN in result.actions


# --- 5. netplay_true has its own funnel, and its own way in --------------


def test_np_press_key_enter_in_command_context_no_longer_says_unknown_command():
    """The gap the `netplay_true` switch opened, closed the same way.

    `netplay_true` skills don't go through `env_response`'s `action_indices`
    loop at all -- `run_netplay_skill` reports `SkillResult(actions=[],
    pre_executed=True, ...)` because the engine steps already happened inside
    `NetPlayEngineEnv.step`. Before this test, `np_press_key(key="enter")`
    called with no prompt open reproduced the run-1 message byte for byte
    (`RawKeyPress.KEYPRESS_ENTER == 13`, vendor/netplay/nethack_utils/
    nle_wrapper.py). RED: this test fails on `NetPlayEngineEnv.step` before its
    guard is added -- the top line reads "Unknown command '^M'." instead of the
    welcome banner.
    """
    from nethack_core.env import NetHackCoreEnv
    from nethack_harness.tools import netplay_true as npt

    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(19, 19)
    env.reset()
    try:
        skill = npt.NETPLAY_SKILL_REPOSITORY.get_skill("press_key")
        result = npt.run_netplay_skill(env, skill, {"key": "enter"})
        assert "Unknown command" not in _top_line(result.final_obs)
    finally:
        env.close()
        npt.reset_agent_cache()


def test_netplay_engine_env_step_swallows_a_stray_cr_directly():
    """Unit-level: the funnel itself, isolated from the skill/agent machinery."""
    from nethack_core.env import NetHackCoreEnv
    from nethack_harness.tools.netplay_true import NetPlayEngineEnv

    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(19, 19)
    env.reset()
    try:
        wrapped = NetPlayEngineEnv(env)
        wrapped.reset()
        before = _top_line(wrapped.last_raw)
        obs, reward, terminated, truncated, info = wrapped.step(CARRIAGE_RETURN)
        assert _top_line(obs.raw) == before
        assert reward == 0.0
        assert not terminated and not truncated
    finally:
        env.close()


def test_netplay_engine_env_step_still_forwards_a_cr_a_prompt_is_waiting_for():
    """The netplay_true guard is conditional too: a real getlin CR still lands.

    Same `E - <CR>` trajectory as the harness-side test above (up to the
    getlin prompt), driven through `NetPlayEngineEnv.step` this time. Each CR
    here is expected to actually reach the engine and change its state --
    proof the guard isn't a blanket CR filter.
    """
    from nethack_core.env import NetHackCoreEnv
    from nethack_harness.tools.netplay_true import NetPlayEngineEnv

    env = NetHackCoreEnv(task_name="NetHackScore-v0")
    env.seed(19, 19)
    env.reset()
    try:
        wrapped = NetPlayEngineEnv(env)
        wrapped.reset()
        wrapped.step(ord("E"))  # "What do you want to write with? [- or ?*]"
        obs, *_ = wrapped.step(ord("-"))  # "You write in the dust...--More--"
        assert _cr_would_be_unknown_command(obs.raw) is False  # a prompt is open
        before = _top_line(obs.raw)
        obs, *_ = wrapped.step(CARRIAGE_RETURN)  # dismiss the --More--
        after = _top_line(obs.raw)
        assert after != before, "the CR should have dismissed the prompt, not been dropped"
        assert "Unknown command" not in after
    finally:
        env.close()
