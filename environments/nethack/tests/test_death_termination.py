"""Task 14 — death termination.

The committed arm-2 acceptance rollout
(``tools/cli_harness_eval/acceptance/task10_prime_agent_seed0.turns.ndjson``)
shows ``hp = 0`` from turn 6 through turn 12 (7 of 12 calls spent on a
tombstone), yet the rollout never set ``died`` and only ended on budget
exhaustion. Two existing detectors both miss this:

  * ``_detect_terminal_outcome`` scans the raw tty for death/ascension text
    markers and never fires on a raw ``hp`` poke (there is no death message
    to scan for) — exactly what a state-mutation-only kill looks like.
  * The NLE ``terminated`` fallback (``nethack.py`` around the
    ``_detect_terminal_outcome`` call) never arrives either, for the same
    reason: nothing routed through the engine's own death codepath.

``hitpoints == 0`` sits in the shaped status every turn already
(``tools/encoding_eval/aggregate_run.py`` uses exactly this signal
post-hoc). This test drives a character to ``hp == 0`` via the engine's
``modify`` hook (a direct ``u.uhp`` poke — see
``nethack_core/engine_env.py``'s ``_MODIFY_BOUNDS`` and
``third_party/NetHack/src/src/nle.c``'s ``nle_set_state`` "hp" case, which
sets ``u.uhp`` with no death-codepath side effect) so death is deterministic
and reproduces the same "gap" the two existing detectors have: no death
message, no NLE ``terminated`` flag, just ``hitpoints == 0`` in the status.
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack as m
import verifiers as vf


def _env_and_state(modify, trace_dir=None):
    env = m.load_environment(
        task_spec="full_nle", skill_set="netplay",
        n_examples=1, explicit_seeds=[0],
        character="Val-hum-neu-fem", modify=modify,
        trace_dir=trace_dir,
    )
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    return env, state


def test_zero_hp_is_detected_as_death():
    env, state = _env_and_state({"hp": 0})
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    assert state["died"] is True
    assert state["terminated"] is True


def test_zero_hp_records_death_dlvl():
    env, state = _env_and_state({"hp": 0})
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    assert state.get("death_dlvl") is not None


def test_dead_character_does_not_step_the_engine_on_a_later_call():
    """A CLI agent driving the game over MCP calls ``_apply_tool_call``
    directly, per-tool-call, with no ``is_completed`` gate in between (that
    gate only exists in the native verifiers rollout loop). Once the
    character is dead, a further call must be a no-op against the engine —
    otherwise a 150-call budget burns the remainder on a tombstone, exactly
    as the committed arm-2 acceptance rollout did (7 of 12 calls after
    death).
    """
    env, state = _env_and_state({"hp": 0})
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    assert state["died"] is True
    frozen_time = state["structured_obs"].status.get("time")

    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))

    assert state["structured_obs"].status.get("time") == frozen_time, (
        "the in-game clock advanced on a call made after death — the "
        "engine was stepped for a character that is already dead"
    )


def test_alive_character_is_unaffected():
    """Sanity check: a normal (non-zero-hp) rollout must not be terminated
    by this new check."""
    env, state = _env_and_state(None)
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    assert state["died"] is False
    assert not state.get("terminated")


def test_trace_lines_carry_a_monotonic_timestamp(tmp_path):
    """Every trace line needs a wall-clock field to measure seconds/call and
    detect superlinear latency growth (Task 15's job). `t_wall` already
    existed here (`time.time()`, present in both committed acceptance
    NDJSONs) — this test pins the new `t_mono` (`time.monotonic()`) field,
    added so a mid-rollout clock adjustment can't corrupt the latency
    measurement `t_wall` alone would be exposed to.
    """
    env, state = _env_and_state(None, trace_dir=str(tmp_path))
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    import json
    lines = [json.loads(l) for l in files[0].read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert isinstance(lines[0].get("t_wall"), float)
    assert isinstance(lines[0].get("t_mono"), float)


def test_dead_character_calls_still_land_in_the_trace(tmp_path):
    """A call that hits the death guard (no engine step) must still be
    visible in the trace — otherwise Task 15 can't count how many of a
    150-call budget were burned on a dead character, which is the whole
    point of measuring this."""
    env, state = _env_and_state({"hp": 0}, trace_dir=str(tmp_path))
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))  # dies
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))  # no-op
    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    import json
    lines = [json.loads(l) for l in files[0].read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[1]["action_indices"] == []
