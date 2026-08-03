"""The trace must be enough to REPLAY and ANALYSE a rollout, not just re-read it.

Everything here runs the engine in process with no LLM anywhere (the scripted
client below is keyless and returns canned tool calls), so the whole file is
free to run.

The six properties under test, and the version-0/1 defect each one pins:

A. `actions` -- the full ordered low-level command stream.
   Before: `action_indices` was EMPTY on 100% of `netplay_true` turns, because
   every one of those skills is `pre_executed` and the harness step loop that
   populates the field never runs for them (`nethack.py`, the
   `if getattr(result, "pre_executed", False)` branch). A `move_to` that walked
   30 tiles recorded `[]`.
B. `gt_obs` -- the engine's own observation planes, per turn.
   Before: only `rendered_user_message` / `raw_grid`, i.e. analysis could only
   ever see what the agent was SHOWN.
C. `tool_results` -- machine-readable outcome per call.
   Before: outcome existed only as prose inside `rendered_user_message`, so
   success/failure needed a regex -- and the regex would have been wrong:
   `completed` is emitted by the vendored skills without inspecting the result.
D. `all_messages` -- every in-game message, including a macro's intermediates.
   Before: `messages` held only the LAST one, because `shape()` only ever sees
   the final observation of the turn.
E. record count == rollout metrics.
   Before: `is_completed` fired on the max_turns cap before `env_response`
   wrote the final entry, so the NDJSON was short by exactly one on EVERY
   rollout (`num_turns=100` vs 99 records, measured).
F. schema validity.
   Before: the writer emitted `t_mono`, which was not in `ALL_FIELDS`, so
   `validate_record()` failed on 100% of new records; and it used a bare
   `json.dumps`, so `schema_version` was never stamped.
"""

import asyncio
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack as m  # noqa: E402
from nethack_core import trace_schema as TS  # noqa: E402
from nethack_harness.helpers import (  # noqa: E402
    TurnRecorder,
    build_tool_result,
    classify_tool_result,
)

SEED = 0
CHARACTER = "Val-hum-neu-fem"
SKILLS = "netplay_true,reveal,rollback"


# --------------------------------------------------------------------------- #
# rigs                                                                         #
# --------------------------------------------------------------------------- #
def _make_env(trace_dir, **kw):
    return m.load_environment(
        task_spec="full_nle", variant="B0", skill_set=SKILLS, n_examples=1,
        max_turns=50, explicit_seeds=[SEED], character=CHARACTER,
        compact_obs=False, trace_dir=str(trace_dir) if trace_dir else None, **kw)


async def _setup(env):
    ex = env.dataset[0]
    state = {"task": {"seed": SEED}, "info": ex["info"], "prompt": ex["prompt"],
             "responses": [], "turn": 0, "id": "probe", "model": "probe"}
    return await env.setup_state(state)


def _drive(trace_dir, plan, **kw):
    """Run `plan` (a list of (skill, kwargs)) in process and return (state, records)."""
    async def go():
        env = _make_env(trace_dir, **kw)
        state = await _setup(env)
        for name, args in plan:
            await env._apply_tool_call(state, name, args)
        return env, state
    env, state = asyncio.run(go())
    files = sorted(pathlib.Path(trace_dir).glob("*.ndjson"))
    assert len(files) == 1, files
    return state, TS.read_trace(files[0])


#: A plan that exercises a closed-loop macro, a no-op call, and a single key.
PLAN = [
    ("np_explore_level", {}),
    ("np_press_key", {"key": "s"}),
    ("np_explore_level", {}),
    ("np_move_to", {"x": 40, "y": 8}),
]


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    return _drive(tmp_path_factory.mktemp("trace_driven"), PLAN)


# --------------------------------------------------------------------------- #
# A. every action a skill executed, in order                                   #
# --------------------------------------------------------------------------- #
def test_pre_executed_skills_now_record_their_command_stream(driven):
    """`np_explore_level` expands into many engine steps; all of them land."""
    _state, recs = driven
    macro = recs[0]
    # `tool_calls` USED to stay empty on this route (it was filled only from a
    # parsed assistant message, which exists only in the native rollout loop),
    # so both CLI arms recorded nothing at all about what they called. It is now
    # synthesized from the dispatch arguments -- see
    # `test_cli_arm_reasoning.py`, which owns that property. `tool_results[i]`
    # has been route-independent since version 2.
    assert macro["tool_results"][0]["name"] == "np_explore_level"
    assert macro["tool_calls"][0]["name"] == "np_explore_level"
    # The version-0/1 field is still exactly as empty as it always was...
    assert macro["action_indices"] == []
    # ...and the new one is not.
    assert macro["actions"]["recorded"] is True
    assert macro["actions"]["source"] == TS.ACTION_SOURCE_ENGINE_HOOK
    assert macro["actions"]["n"] > 1, macro["actions"]
    assert len(macro["actions"]["bytes"]) == macro["actions"]["n"]
    assert all(isinstance(b, int) for b in macro["actions"]["bytes"])
    # `keys` is a derived, human-readable rendering of the same sequence.
    assert macro["actions"]["keys"] == "".join(
        TS.key_repr(b) for b in macro["actions"]["bytes"])


def test_the_recorded_actions_replay_the_game_byte_for_byte(tmp_path):
    """The point of the whole exercise: actions alone reproduce the state.

    Replays the concatenated command stream against a virgin engine of the same
    seed and asserts every ground-truth plane matches the original run.
    """
    state, recs = _drive(tmp_path / "orig", PLAN)
    stream = [b for r in recs for b in (r["actions"]["bytes"] or [])]
    assert len(stream) > 10, "nothing to replay"

    async def replay():
        env = _make_env(None)
        st = await _setup(env)
        for b in stream:
            st["env"].step(int(b))
        return st["env"]._last_observation

    replayed = asyncio.run(replay())
    original = state["raw_obs"]
    # `tty_chars` is deliberately NOT compared: the harness scrubs the intro
    # banner out of it IN PLACE every turn for rendering (`_scrub_intro_banner`),
    # so it is a presentation surface, not engine state. Everything the engine
    # actually simulates must match exactly.
    for plane in ("glyphs", "chars", "colors", "blstats", "inv_strs",
                  "inv_letters", "inv_glyphs", "message"):
        a = np.asarray(getattr(original, plane))
        b = np.asarray(getattr(replayed, plane))
        assert a.shape == b.shape and (a == b).all(), f"{plane} diverged on replay"


def test_an_unobservable_action_stream_says_so_instead_of_faking_an_empty_list():
    """`recorded: False` and `n: 0` are different facts and must stay different.

    Version 0/1 wrote `[]` both when a turn executed nothing and when the turn's
    actions were simply not visible to the writer, which is exactly why "100% of
    netplay_true turns had no actions" read as a metric rather than as a bug.
    """
    unobservable = TS.empty_action_record("no engine access")
    genuinely_empty = TS.action_record([])
    assert unobservable["recorded"] is False
    assert unobservable["reason"]
    assert unobservable["replayable"] is False
    assert genuinely_empty["recorded"] is True
    assert genuinely_empty["reason"] is None
    assert genuinely_empty["replayable"] is True
    assert unobservable["n"] == genuinely_empty["n"] == 0
    assert unobservable != genuinely_empty


def test_a_turn_that_rewinds_the_engine_is_marked_not_replayable(tmp_path):
    """`rollback` rewinds the engine heap via `engine.restore(handle)` -- it
    never goes through `step`, so the byte stream cannot reproduce it. Saying so
    is the whole point: a trace that implied replayability here would be quietly
    wrong instead of usefully incomplete."""
    _state, recs = _drive(tmp_path / "rb", [
        ("np_press_key", {"key": "s"}),
        ("np_press_key", {"key": "s"}),
        ("np_press_key", {"key": "s"}),
        ("rollback", {"n": 2}),
        ("np_press_key", {"key": "s"}),
    ])
    assert [r["actions"]["replayable"] for r in recs] == [True, True, True, False, True]
    rb = recs[3]
    assert rb["actions"]["recorded"] is True        # we DID see its keystrokes
    assert "rollback" in rb["actions"]["reason"]
    assert rb["tool_results"][0]["status"] == "completed"


def test_the_recorder_captures_steps_taken_anywhere_in_the_turn(tmp_path):
    """The hook is on the engine, not on a call site, so nothing can bypass it."""
    async def go():
        env = _make_env(None)
        state = await _setup(env)
        core = state["env"]
        rec = TurnRecorder(core)
        with rec:
            core.step(ord("s"))     # "call site" 1
            for _ in range(3):      # "call site" 2
                core.step(ord("s"))
        return core, rec
    core, rec = asyncio.run(go())
    assert rec.actions == [ord("s")] * 4
    # And the patch is gone afterwards: no leak into the next turn.
    assert "step" not in core.__dict__
    before = len(rec.actions)
    core.step(ord("s"))
    assert len(rec.actions) == before


# --------------------------------------------------------------------------- #
# B. ground-truth observations                                                 #
# --------------------------------------------------------------------------- #
def test_ground_truth_observation_is_persisted_and_round_trips(driven):
    state, recs = driven
    blob = recs[-1]["gt_obs"]
    assert blob is not None and blob["codec"] == TS.GT_OBS_CODEC
    planes = TS.decode_obs_blob(blob)
    assert set(planes) >= {"glyphs", "chars", "colors", "blstats", "inv_strs"}
    live = state["raw_obs"]
    for name, arr in planes.items():
        ref = np.asarray(getattr(live, name))
        assert arr.shape == ref.shape, name
        assert (arr == ref).all(), f"{name} does not match the live engine obs"


def test_ground_truth_is_stored_compressed_not_as_json_integer_lists(driven):
    """Size guard. Naive JSON lists would be ~45 KB/turn; this must stay small."""
    _state, recs = driven
    raw = sum(r["gt_obs"]["raw_bytes"] for r in recs)
    stored = sum(len(json.dumps(r["gt_obs"])) for r in recs)
    assert stored < raw * 0.4, (
        f"gt_obs costs {stored/len(recs):.0f} B/turn against {raw/len(recs):.0f} "
        "B of raw planes -- the codec regressed")
    # And it must not dominate the record it rides in.
    whole = sum(len(TS.to_json_line(r)) for r in recs)
    assert stored < whole * 0.5


def test_a_record_without_ground_truth_is_still_readable():
    """`gt_obs: None` is legal; readers must not require it."""
    assert TS.decode_obs_blob(None) == {}
    assert TS.decode_obs_blob({"codec": "something-else"}) == {}


# --------------------------------------------------------------------------- #
# C. machine-readable tool results                                             #
# --------------------------------------------------------------------------- #
def test_every_turn_carries_a_structured_result(driven):
    _state, recs = driven
    for r in recs:
        assert len(r["tool_results"]) == 1, r["tool_results"]
        tr = r["tool_results"][0]
        assert tr["status"] in TS.TOOL_STATUSES, tr["status"]
        assert set(tr) >= {"status", "game_message", "clock_advanced",
                           "clock_before", "clock_after", "engine_steps"}
        assert tr["name"] == r["tool_calls"][0]["name"] if r["tool_calls"] else True


def test_a_blocked_move_that_claims_completed_is_distinguishable(tmp_path):
    """The headline defect C exists to fix.

    `move_to` onto the tile the hero is already standing on reports
    `Skill 'move_to X Y' completed: Reached position (X, Y)` -- so a "completed"
    metric counts it as a success -- while taking zero engine steps and leaving
    the game clock exactly where it was. Measured on the probe, 6 of 10
    zero-clock turns reported `completed`, overstating success by ~21 points.
    `status` alone cannot see this; `status` + `clock_advanced` can.
    """
    async def here():
        env = _make_env(None)
        st = await _setup(env)
        bl = st["raw_obs"].blstats
        return int(bl[0]), int(bl[1])
    x, y = asyncio.run(here())

    _state, recs = _drive(tmp_path / "blocked", [("np_move_to", {"x": x, "y": y})])
    tr = recs[0]["tool_results"][0]
    assert tr["status"] == "completed", tr
    assert "completed" in tr["feedback"]
    assert tr["clock_advanced"] is False, tr
    assert tr["engine_steps"] == 0, tr
    assert recs[0]["actions"]["recorded"] is True and recs[0]["actions"]["n"] == 0


def test_failure_is_a_status_not_a_regex_over_the_rendered_prose(tmp_path):
    _state, recs = _drive(
        tmp_path / "fail", [("np_melee_attack", {"x": 1, "y": 1})])
    tr = recs[0]["tool_results"][0]
    assert tr["status"] == "failed", tr
    assert tr["clock_advanced"] is False


def test_status_classification_covers_the_shapes_the_skills_actually_emit():
    cases = {
        "Executing skill 'move_to 57 13'. Skill 'move_to 57 13' failed: "
        "No valid path found to reach position (57, 13).": "failed",
        "Executing skill 'move_to 36 10'. Skill 'move_to 36 10' completed: "
        "Tile (36,10) is blocked, stopping adjacent to it": "completed",
        "REFUSED: (3,4) is your pet little dog.": "refused",
        "Skill raised ValueError: boom": "error",
        "Executing skill 'explore_level'. Interrupting skill to rethink "
        "because 'A gold piece appeared at (8,7).'": "interrupted",
        "[INTERRUPTED: newt corpse. Stopped early so you can eat before it "
        "spoils.]": "interrupted",
        "Executing skill 'explore_level'. Skill has been running for 100 "
        "timesteps without interruption. Rethinking.": "interrupted",
        "No effect.": "no_op",
        "rolled back 2 turn(s) to the state after turn 7.": "completed",
        "rollback unavailable: no snapshots recorded yet.": "failed",
        "cannot roll back 9 turns; only 3 earlier turn(s) are retained.": "failed",
        "reveal: map unavailable this turn.": "failed",
        "reveal (x1-x20, y3-y9):\ny 3: ....": "completed",
    }
    for feedback, expected in cases.items():
        assert classify_tool_result(feedback) == expected, feedback
    # `unknown` is an honest answer for a skill whose prose has no recognised
    # marker -- better than guessing "completed". `engine_steps` and
    # `clock_advanced` on the same record stay exact regardless.
    assert classify_tool_result("something nobody has seen") == "unknown"
    for status in TS.TOOL_STATUSES:
        assert isinstance(status, str)


def test_clock_advanced_is_none_when_the_clock_is_unknown():
    r = build_tool_result(name="x", arguments={}, feedback="", status="unknown",
                          clock_before=None, clock_after=7)
    assert r["clock_advanced"] is None


# --------------------------------------------------------------------------- #
# D. all intermediate messages                                                 #
# --------------------------------------------------------------------------- #
def test_a_macro_keeps_every_message_not_just_the_last(driven):
    """`messages` is last-only by construction; `all_messages` is the stream."""
    _state, recs = driven
    macro = recs[0]
    assert len(macro["messages"]) <= 1, "shape() has started keeping more?"
    assert len(macro["all_messages"]) > len(macro["messages"]), macro["all_messages"]
    # The last-known message is still the last entry of the full list, so the
    # two views agree where they overlap.
    if macro["messages"]:
        assert macro["all_messages"][-1] == macro["messages"][-1]
    # Every kept message is non-empty and no two neighbours repeat.
    for i, msg in enumerate(macro["all_messages"]):
        assert msg.strip()
        if i:
            assert msg != macro["all_messages"][i - 1]


def test_messages_produced_mid_macro_are_not_dropped(tmp_path):
    """A message emitted at step k of a macro must survive to the trace.

    Seed 0 opens with the welcome banner and then the pet picking up / dropping
    gold several steps into the first `explore_level`; before this, only the
    final observation's message was ever recorded.
    """
    _state, recs = _drive(tmp_path / "msgs", [("np_explore_level", {})])
    msgs = recs[0]["all_messages"]
    assert len(msgs) >= 2, msgs
    # The welcome banner is emitted on the FIRST step of the macro and would be
    # the first thing a last-message-only writer loses.
    assert any("welcome to NetHack" in msg for msg in msgs), msgs
    assert msgs[0] != msgs[-1]


# --------------------------------------------------------------------------- #
# E. one record per LM turn, including the last one                            #
# --------------------------------------------------------------------------- #
class _Scripted:
    """Keyless client that emits one canned tool call per turn, forever."""

    def __new__(cls, *a, **kw):
        from verifiers.clients import Client

        class Impl(Client):
            def __init__(self):
                self._config = self._client = None
                self.calls = 0

            def setup_client(self, config):
                return None

            async def close(self):
                return None

            async def to_native_tool(self, tool):
                return tool

            async def to_native_prompt(self, messages):
                return messages, {}

            async def get_native_response(self, *a, **kw):
                return None

            async def raise_from_native_response(self, response):
                return None

            async def from_native_response(self, response):
                return response

            async def get_response(self, prompt, model, sampling_args, tools=None, **kw):
                from verifiers.types import Response, ResponseMessage, ToolCall
                i = self.calls
                self.calls += 1
                msg = ResponseMessage(
                    role="assistant", content="",
                    tool_calls=[ToolCall(id=f"c{i}", name="np_press_key",
                                         arguments=json.dumps({"key": "s"}))],
                    finish_reason="tool_calls", is_truncated=False)
                return Response(id=f"r{i}", created=0, model=model, message=msg)

        return Impl()


def _scripted_rollout(trace_dir, max_turns):
    from verifiers.v1.legacy import rollout_output_to_trace

    env = m.load_environment(
        task_spec="full_nle", n_examples=1, max_turns=max_turns,
        explicit_seeds=[SEED], character=CHARACTER, skill_set="netplay_true",
        trace_dir=str(trace_dir))
    out = asyncio.run(env.run_rollout(
        input=dict(env.get_eval_dataset()[0]), client=_Scripted(),
        model="scripted", sampling_args={}, state_columns=["trajectory"]))
    trace = rollout_output_to_trace(out, 0)
    metrics = trace.metrics if hasattr(trace, "metrics") else trace["metrics"]
    recs = TS.read_trace(sorted(pathlib.Path(trace_dir).glob("*.ndjson"))[0])
    return metrics, recs


@pytest.mark.parametrize("max_turns", [4, 7])
def test_the_ndjson_and_the_rollout_metrics_agree(tmp_path, max_turns):
    """Defect E. Measured before the fix: `num_turns` / `total_tool_calls` = 30
    against 29 NDJSON records -- every per-turn aggregation disagreed with
    rollout metrics by exactly one, on every rollout."""
    metrics, recs = _scripted_rollout(tmp_path / f"n{max_turns}", max_turns)
    assert len(recs) == int(metrics["num_turns"]) == int(metrics["total_tool_calls"])
    assert [r["lm_turn"] for r in recs] == list(range(1, len(recs) + 1))


def test_the_final_unapplied_call_is_recorded_as_unapplied(tmp_path):
    """The fix must not pretend the last call ran: the rollout loop generates it
    and counts it, but `env_response` is never invoked for it. Stepping the
    engine one extra time to balance the books would change the game the metrics
    describe, so we record the call, mark it `applied: False`, and say in the
    action record why there is no command stream."""
    _metrics, recs = _scripted_rollout(tmp_path / "final", 5)
    assert [r["applied"] for r in recs] == [True] * 4 + [False]
    last = recs[-1]
    assert last["tool_calls"][0]["name"] == "np_press_key"
    assert last["tool_results"][0]["name"] == "np_press_key"
    assert last["tool_results"][0]["arguments"] == {"key": "s"}
    assert last["tool_results"][0]["status"] == "not_applied"
    assert last["actions"]["recorded"] is False
    assert "never applied" in last["actions"]["reason"]
    assert last["actions"]["bytes"] == []
    # It still carries the observation the call was made against.
    assert last["rendered_user_message"]
    assert last["gt_obs"] is not None


def test_turns_that_never_reach_the_engine_still_get_a_record(tmp_path):
    """A hallucinated tool consumes an LM turn and used to produce no record at
    all, which made `len(records)` unreconcilable with `num_turns` independently
    of the max_turns off-by-one."""
    _state, recs = _drive(tmp_path / "reject", [
        ("np_press_key", {"key": "s"}),
        ("no_such_tool", {}),
        ("np_press_key", {"key": "s"}),
    ])
    assert len(recs) == 3
    assert [r["lm_turn"] for r in recs] == [1, 2, 3]
    rejected = recs[1]
    assert rejected["tool_results"][0]["status"] == "rejected"
    assert rejected["applied"] is True          # dispatched, just not honoured
    assert rejected["actions"]["recorded"] is True
    assert rejected["actions"]["n"] == 0        # ...and provably did nothing


def test_lm_turn_is_the_authoritative_counter_and_turn_is_unchanged(tmp_path):
    """`turn` keeps its version-0/1 meaning (it only advances on turns that
    reached the engine), so existing readers are untouched; `lm_turn` is the
    one that counts LM turns."""
    _state, recs = _drive(tmp_path / "counters", [
        ("np_press_key", {"key": "s"}),
        ("no_such_tool", {}),
        ("np_press_key", {"key": "s"}),
    ])
    assert [r["lm_turn"] for r in recs] == [1, 2, 3]
    turns = [r["turn"] for r in recs]
    assert turns == sorted(turns)
    assert turns[1] == turns[0]                 # the rejected turn did not advance it


# --------------------------------------------------------------------------- #
# F. schema validity, old and new                                              #
# --------------------------------------------------------------------------- #
def test_new_records_validate_and_carry_a_version_stamp(driven):
    """Both halves of defect F: `t_mono` was an unknown field on 100% of new
    records, and the bare `json.dumps` meant `record_version()` read 0."""
    _state, recs = driven
    for r in recs:
        assert TS.validate_record(r) == [], (r["lm_turn"], TS.validate_record(r))
        assert TS.record_version(r) == TS.TRACE_SCHEMA_VERSION >= 2
        assert "t_mono" in r


def test_t_mono_is_a_documented_field():
    """It is consumed by tools/cli_harness_eval/aggregate.py, so the schema --
    not the writer -- was the stale side."""
    assert "t_mono" in TS.ALL_FIELDS


COMMITTED_TRACES = sorted(
    (pathlib.Path(__file__).resolve().parents[3]
     / "tools" / "cli_harness_eval" / "acceptance").glob("*.turns.ndjson"))


@pytest.mark.parametrize(
    "path", COMMITTED_TRACES, ids=[p.name for p in COMMITTED_TRACES])
def test_committed_legacy_traces_still_validate(path):
    recs = TS.read_trace(path)
    assert recs
    for r in recs:
        assert TS.validate_record(r) == [], (path.name, TS.validate_record(r))
        assert TS.record_version(r) == 0        # unstamped == legacy


def test_a_legacy_record_carrying_t_mono_now_validates():
    """The pilot_reveal traces (uncommitted, 4 x 99 turns) are exactly this
    shape: version 0, but written by a `t_mono`-emitting writer."""
    legacy = {f: None for f in TS.REQUIRED_FIELDS}
    legacy["t_mono"] = 1234.5
    assert TS.validate_record(legacy) == []
    assert TS.record_version(legacy) == 0


def test_no_version_0_field_name_was_renamed(tmp_path):
    """The aggregators (`tools/cli_harness_eval/{progress,aggregate}.py`) read
    these by name. Version 2 is purely additive."""
    _metrics, recs = _scripted_rollout(tmp_path / "fields", 4)
    v0_and_v1 = (
        "turn", "t_wall", "variant", "raw_grid", "status", "dlvl", "hp",
        "max_hp", "max_dlvl_reached", "continual_life", "rendered_user_message",
        "rendered_user_content", "assistant_message", "tool_calls",
        "action_indices", "reward", "messages",
    )
    for field in v0_and_v1:
        assert field in recs[0], field
    # `tool_calls` keeps its exact `{"name", "arguments"}` shape, and
    # `arguments` is still the raw JSON *string* the model emitted.
    call = recs[0]["tool_calls"][0]
    assert set(call) == {"name", "arguments"}
    assert isinstance(call["arguments"], str)
    json.loads(call["arguments"])
