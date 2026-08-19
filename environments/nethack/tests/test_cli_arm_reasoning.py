"""The CLI arms must record WHAT they called and WHY, like the control arm does.

The defect (`docs/HARNESS_DEFECTS.md` Sec 3.5, "We can see *what* it decided,
never *why*") had two halves, and they have different fixes because they have
different causes:

* `tool_calls` was empty on 100% of CLI-arm turns for no good reason at all.
  The dispatched name and arguments are literally the arguments to
  `_apply_tool_call` -- `tool_results[i]["name"]` had been populated from them
  since schema version 2 -- but the field was filled only from a PARSED
  assistant message, which exists only inside the in-process v0 rollout loop.
  Fixed at the write site; `test_the_mcp_route_records_the_call_it_dispatched`.

* `assistant_message` was empty on 0-of-12 / 0-of-20 of the committed CLI
  reference traces because the tool server genuinely does not have it: a CLI
  agent talks to the interception endpoint in a different process. That cannot
  be fixed at the write site, so the record now says so explicitly
  (`reasoning.available = false` with a reason) and `NetHackTask.finalize`
  backfills it from the rollout trace, where the messages actually are.

Measured before: `assistant_message` non-empty on 9 of 29 control-arm turns and
0 of 32 CLI-arm turns; `tool_calls` empty on 100% of CLI-arm turns.
"""

import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import nethack as m  # noqa: E402
import nethack_v1 as v1  # noqa: E402
from nethack_core import trace_schema as TS  # noqa: E402
from nethack_harness.helpers import MCP_REASONING_UNAVAILABLE  # noqa: E402

from _mcp_server_harness import booted_toolset, build_task_and_trace  # noqa: E402

SEED = 0
CHARACTER = "Val-hum-neu-fem"
SKILLS = "netplay_true,reveal,rollback"
ACCEPTANCE = pathlib.Path(__file__).resolve().parents[3] / "tools/cli_harness_eval/acceptance"


# --------------------------------------------------------------------------- #
# rigs                                                                         #
# --------------------------------------------------------------------------- #
def _drive_mcp_route(trace_dir, plan, **env_kwargs):
    """Dispatch `plan` the way an MCP tool call does: straight into
    `_apply_tool_call`, with no assistant message anywhere. Returns the trace
    records AND the result payloads exactly as they went back over the wire."""

    async def go():
        env = m.load_environment(
            task_spec="full_nle", variant="B0", skill_set=SKILLS, n_examples=1,
            max_turns=50, explicit_seeds=[SEED], character=CHARACTER,
            compact_obs=False, trace_dir=str(trace_dir), **env_kwargs)
        ex = env.dataset[0]
        state = await env.setup_state(
            {"task": {"seed": SEED}, "info": ex["info"], "prompt": ex["prompt"],
             "responses": [], "turn": 0, "id": "probe", "model": "probe"})
        contents = []
        for name, args in plan:
            contents.append(await env._apply_tool_call(state, name, args))
        return contents

    contents = asyncio.run(go())
    files = sorted(pathlib.Path(trace_dir).glob("*.ndjson"))
    assert len(files) == 1, files
    return TS.read_trace(files[0]), contents


PLAN = [("np_press_key", {"key": "s"}), ("np_move_to", {"x": 40, "y": 8})]


@pytest.fixture(scope="module")
def mcp_route(tmp_path_factory):
    return _drive_mcp_route(tmp_path_factory.mktemp("mcp_route"), PLAN)


@pytest.fixture(scope="module")
def mcp_records(mcp_route):
    return mcp_route[0]


# --------------------------------------------------------------------------- #
# 1. the dispatched call is recorded on BOTH routes                            #
# --------------------------------------------------------------------------- #
def test_the_mcp_route_records_the_call_it_dispatched(mcp_records):
    """The headline defect: measured empty on 100% of CLI-arm turns before."""
    assert len(mcp_records) == len(PLAN)
    for record, (name, args) in zip(mcp_records, PLAN):
        assert record["tool_calls"], "tool_calls empty on the MCP route"
        call = record["tool_calls"][0]
        assert set(call) == {"name", "arguments"}
        assert call["name"] == name
        # Already parsed on this route (the MCP client parsed the JSON), which
        # is the same type `tool_results[i]["arguments"]` carries everywhere.
        assert call["arguments"] == args
        # ... and it agrees with the route-independent field, which is the
        # whole point of having recorded both.
        assert record["tool_results"][0]["name"] == name


def test_each_record_says_which_route_dispatched_it(mcp_records):
    assert {r["dispatch_route"] for r in mcp_records} == {"mcp"}
    for record in mcp_records:
        assert record["dispatch_route"] in TS.DISPATCH_ROUTES


def test_the_control_route_is_unchanged(tmp_path):
    """The control arm's records must be byte-identical in every version-0/1
    field. Its `tool_calls[i]["arguments"]` in particular is still the RAW JSON
    STRING the model emitted, not a dict -- readers depend on that."""
    from test_trace_reproducibility import _scripted_rollout

    _metrics, recs = _scripted_rollout(tmp_path / "control", 4)
    assert {r["dispatch_route"] for r in recs} == {"harness"}
    call = recs[0]["tool_calls"][0]
    assert call["name"] == "np_press_key"
    assert isinstance(call["arguments"], str)
    assert json.loads(call["arguments"]) == {"key": "s"}


# --------------------------------------------------------------------------- #
# 2. reasoning: available, or explicitly unavailable WITH A REASON             #
# --------------------------------------------------------------------------- #
def test_the_mcp_route_says_why_the_reasoning_is_missing(mcp_records):
    """"Unavailable, because <reason>" is a different fact from `""`, and the
    empty string could not express it."""
    for record in mcp_records:
        reasoning = record["reasoning"]
        assert reasoning["available"] is False
        assert reasoning["text"] == ""
        assert reasoning["reason"] == MCP_REASONING_UNAVAILABLE
        # It names the thing that fixes it, so a reader is never stuck.
        assert "tools.trace_reasoning" in reasoning["reason"]


def test_a_silent_model_and_an_unseeable_model_are_distinguishable(tmp_path):
    """The scripted control-arm model returns `content=""` on every turn. That
    is the model saying nothing -- NOT the route being unable to see it -- and
    the two carry different reasons."""
    from test_trace_reproducibility import _scripted_rollout

    _metrics, recs = _scripted_rollout(tmp_path / "silent", 4)
    for record in recs:
        assert record["reasoning"]["available"] is False
        assert record["reasoning"]["reason"] != MCP_REASONING_UNAVAILABLE
        assert "no assistant text" in record["reasoning"]["reason"]


def test_reasoning_is_captured_inline_when_the_harness_has_it(tmp_path):
    """The control arm's existing behaviour, now also carrying the provider's
    separate `reasoning_content` channel, which was previously dropped."""
    from nethack_harness.helpers import _reasoning_block

    block = _reasoning_block(
        {"content": "I should go downstairs.", "reasoning_content": "thinking..."},
        "harness")
    assert block["available"] is True
    assert block["source"] == "assistant_message"
    assert block["alignment"] == "inline"
    assert block["text"] == "I should go downstairs."
    assert block["reasoning_text"] == "thinking..."


# --------------------------------------------------------------------------- #
# 2b. the call-id barrier: one id on BOTH log streams                          #
# --------------------------------------------------------------------------- #
def test_every_dispatched_call_gets_a_monotonic_correlation_id(mcp_route):
    """The game-side half of the barrier: the server assigns the id at the
    single shared dispatch path and stamps it on the record, so it exists on
    every route regardless of what the scaffold can see."""
    records, _contents = mcp_route
    results = [r["tool_results"][0] for r in records]
    assert [tr["call_id"] for tr in results] == [1, 2]
    assert all(tr["call_id_echoed"] is True for tr in results)
    # The MCP transport surfaces no native tool-call id to the server -- which
    # is exactly why the counter is the primary key, not the corroboration.
    assert all(tr["native_call_id"] is None for tr in results)
    for record in records:
        assert TS.validate_record(record) == [], TS.validate_record(record)


def test_the_marker_rides_the_result_payload_on_the_wire(mcp_route):
    """The model-side half: `[call#N]` is appended to the RESULT -- the one
    channel that reaches the transcript verbatim in every scaffold -- and the
    trace's `rendered_user_message` records exactly what the model saw."""
    from tools.trace_align import marker_call_ids

    records, contents = mcp_route
    for i, (record, content) in enumerate(zip(records, contents), start=1):
        text = content if isinstance(content, str) else str(content)
        assert text.rstrip().endswith(f"[call#{i}]"), text[-60:]
        assert marker_call_ids(text) == [i]
        assert record["rendered_user_message"].rstrip().endswith(f"[call#{i}]")


def test_the_echo_can_be_disabled_without_losing_the_game_side_id(tmp_path):
    """`call_id_in_results=False` restores byte-identical result payloads for
    token-matched cells. The record still carries the id (stamping is free);
    `call_id_echoed: false` tells a post-hoc join not to expect markers."""
    records, contents = _drive_mcp_route(
        tmp_path / "no_echo", [("np_press_key", {"key": "s"})],
        call_id_in_results=False)
    text = contents[0] if isinstance(contents[0], str) else str(contents[0])
    assert "[call#" not in text
    assert "[call#" not in records[0]["rendered_user_message"]
    tr = records[0]["tool_results"][0]
    assert tr["call_id"] == 1
    assert tr["call_id_echoed"] is False


def _scripted_rollout_with_trace(trace_dir, max_turns):
    """A full scripted rollout returning BOTH log streams: the turn NDJSON and
    the v1 `Trace` (the `traces.jsonl` line) the legacy bridge produces."""
    from test_trace_reproducibility import _Scripted
    from verifiers.v1.legacy import rollout_output_to_trace

    env = m.load_environment(
        task_spec="full_nle", n_examples=1, max_turns=max_turns,
        explicit_seeds=[SEED], character=CHARACTER, skill_set="netplay_true",
        trace_dir=str(trace_dir))
    out = asyncio.run(env.run_rollout(
        input=dict(env.get_eval_dataset()[0]), client=_Scripted(),
        model="scripted", sampling_args={}, state_columns=["trajectory"]))
    trace = rollout_output_to_trace(out, 0)
    recs = TS.read_trace(sorted(pathlib.Path(trace_dir).glob("*.ndjson"))[0])
    return trace, recs


def test_the_call_id_join_is_total_and_unique_end_to_end(tmp_path):
    """THE BARRIER, proven end to end: run a real rollout, then join the two
    logs on the id alone -- every dispatched record finds exactly one issuing
    assistant turn, no id is claimed twice, and the never-dispatched flush is
    explicitly id-less rather than misattributed."""
    from tools.trace_align import (
        align_coverage, align_records_to_turns_by_call_id, record_call_id)
    from tools.trace_reasoning import assistant_turns_from_trace

    trace, recs = _scripted_rollout_with_trace(tmp_path / "join", 5)
    turns = assistant_turns_from_trace(trace)

    mapping, mode, reason = align_records_to_turns_by_call_id(recs, turns)
    assert mode == "call_id", reason

    applied = [r for r in recs if r["applied"]]
    # Total: every dispatched record carries an id and found its turn...
    assert [record_call_id(r) for r in applied] == list(range(1, len(applied) + 1))
    assert all(mapping[i] is not None for i, r in enumerate(recs) if r["applied"])
    # ...and the mapping is the ground-truth identity of the scripted rollout
    # (assistant turn k issued call k+1), which no ordinal guess was consulted
    # to produce.
    assert mapping == list(range(len(applied))) + [None]
    # Unique: no two records share an id; no id maps to two turns.
    ids = [record_call_id(r) for r in applied]
    assert len(set(ids)) == len(ids)
    # The flush record (generated but never dispatched) is explicit about
    # having no id -- absence is a stated fact, not a missing key.
    assert recs[-1]["applied"] is False
    assert record_call_id(recs[-1]) is None
    assert recs[-1]["tool_results"][0]["call_id"] is None

    coverage = align_coverage(recs, turns)
    assert coverage["total"] is True
    assert coverage["joined"] == coverage["with_id"] == len(applied)

    # And the reasoning backfill now rides the same join: its strongest
    # alignment mode is the barrier, not a name-walk or an ordinal guess.
    from tools.trace_reasoning import align_records_to_turns
    _mapping2, mode2, _ = align_records_to_turns(recs, turns)
    assert mode2 == "call_id"


# --------------------------------------------------------------------------- #
# 3. schema: additive, and old traces still parse                              #
# --------------------------------------------------------------------------- #
def test_the_new_records_still_validate(mcp_records):
    for record in mcp_records:
        assert TS.validate_record(record) == [], TS.validate_record(record)
        assert TS.record_version(record) == TS.TRACE_SCHEMA_VERSION >= 3


def test_the_committed_cli_reference_traces_still_parse():
    """Both new fields are optional; the pre-versioning CLI artifacts carry
    neither and must keep validating unchanged."""
    files = sorted(ACCEPTANCE.glob("*.turns.ndjson"))
    assert files, f"no reference traces under {ACCEPTANCE}"
    for path in files:
        records = TS.read_trace(path)
        assert records
        for record in records:
            assert TS.validate_record(record) == []
            assert "dispatch_route" not in record
            assert "reasoning" not in record


# --------------------------------------------------------------------------- #
# 4. the backfill: the reasoning IS recoverable, from the rollout trace        #
# --------------------------------------------------------------------------- #
def _sampled_assistant_nodes(pairs):
    """`[(text, skill_name)]` -> verifiers `MessageNode`s, as the interception
    server commits them."""
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage, ToolCall

    return [
        MessageNode(
            message=AssistantMessage(
                content=text,
                reasoning_content=f"[reasoning for {name}]",
                tool_calls=[ToolCall(id=f"c{i}", name=f"mcp__nethack__{name}",
                                     arguments="{}")],
            ),
            sampled=True,
        )
        for i, (text, name) in enumerate(pairs)
    ]


def test_the_missing_reasoning_is_recoverable_from_the_rollout_trace(mcp_records):
    """The join `NetHackTask.finalize` and `tools/trace_reasoning` both perform.

    Note the tool names arrive `mcp__nethack__`-prefixed, which is what Claude
    Code's client emits; the trace records the bare skill name.
    """
    from tools.trace_reasoning import assistant_turns_from_nodes, backfill_records

    records = json.loads(json.dumps(mcp_records))       # deep copy
    nodes = _sampled_assistant_nodes(
        [(f"I will {name}.", name) for name, _ in PLAN])
    stats = backfill_records(records, assistant_turns_from_nodes(nodes))

    assert stats["mode"] == "tool_call_sequence"
    assert stats["recovered"] == len(PLAN)
    for record, (name, _args) in zip(records, PLAN):
        assert record["assistant_message"] == f"I will {name}."
        assert record["reasoning"]["available"] is True
        assert record["reasoning"]["source"] == "trace_nodes"
        assert record["reasoning"]["reasoning_text"] == f"[reasoning for {name}]"
        assert TS.validate_record(record) == []


# --------------------------------------------------------------------------- #
# 4b. timestamp fallback: recover PRE-BARRIER runs with no call-id markers      #
# --------------------------------------------------------------------------- #
def test_timestamp_alignment_recovers_pre_barrier_ipython_runs():
    """The `timestamp` strategy, for a pre-barrier run whose game tools are
    invisible to the model (Prime Agent: every assistant turn calls `ipython`).

    No `[call#N]` markers (strategy 0 out), no game tool name to walk (1 out),
    and the counts disagree -- one plan, several silent moves (2 out). The
    wall-clock join then attributes each move to the last turn emitted at or
    before it, and carries a silent move back to the plan it is still executing,
    so several moves honestly share one narration rather than being dropped.
    """
    from tools.trace_reasoning import align_records_to_turns, backfill_records

    def turn(text, ts):
        return {"content": text, "reasoning_content": "", "tool_names": ["ipython"],
                "game_tool_names": [], "result_call_ids": [], "timestamp": ts}

    turns = [turn("Plan A: head for the downstairs.", 100.0),
             turn("", 110.0),                                   # silent tool turn
             turn("Plan B: the newt is next to me, kill it.", 120.0)]

    def rec(name, t_wall):
        return {"tool_results": [{"name": name}], "tool_calls": [{"name": name}],
                "assistant_message": "", "t_wall": t_wall}

    records = [rec("np_move_to", 101.0), rec("np_move_to", 112.0),
               rec("np_move_to", 115.0), rec("np_melee_attack", 121.0)]

    mapping, mode, reason = align_records_to_turns(records, turns)
    assert mode == "timestamp", reason
    assert mapping == [0, 0, 0, 2]              # silent moves carried back to Plan A

    stats = backfill_records(records, turns)
    assert stats["mode"] == "timestamp"
    assert stats["recovered"] == 4 and stats["unavailable"] == 0
    assert [r["assistant_message"] for r in records[:3]] == ["Plan A: head for the downstairs."] * 3
    assert records[3]["assistant_message"] == "Plan B: the newt is next to me, kill it."
    for r in records:
        assert r["reasoning"]["alignment"] == "timestamp"
        assert r["reasoning"]["source"] == "trace_nodes"


def test_timestamp_alignment_refuses_when_clocks_are_unusable():
    """A missing stamp, or a first move that predates the first completion,
    means the clocks are not comparable -- refuse, do not guess."""
    from tools.trace_reasoning import align_records_to_turns

    turns = [{"content": "x", "reasoning_content": "", "tool_names": ["ipython"],
              "game_tool_names": [], "result_call_ids": [], "timestamp": 100.0}]
    early = [{"tool_results": [{"name": "np_move_to"}], "t_wall": 50.0},
             {"tool_results": [{"name": "np_move_to"}], "t_wall": 60.0}]
    _m, mode, reason = align_records_to_turns(early, turns)
    assert mode is None
    assert "wrong move" in reason


# --------------------------------------------------------------------------- #
# 5. end to end, over a REAL MCP tool server                                   #
# --------------------------------------------------------------------------- #
SEQUENCE = [("search", {"times": 1}), ("search", {"times": 2})]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """Drive a real `python -m nethack_v1` server over real MCP, then run the
    task's own `finalize`. Costs one engine boot. Also captures what actually
    crossed the wire: the published tool schemas and the raw result texts."""
    trace_dir = tmp_path_factory.mktemp("served_turns")

    async def go():
        task, trace = build_task_and_trace(trace_dir=str(trace_dir))
        async with booted_toolset(task, trace) as (session, mini):
            listed = await session.list_tools()
            schemas = {t.name: t.inputSchema for t in listed.tools}
            wire_texts = []
            for name, args in SEQUENCE:
                result = await session.call_tool(name, args)
                wire_texts.append("".join(
                    getattr(block, "text", "") or "" for block in result.content))
            trace.nodes.extend(
                _sampled_assistant_nodes(
                    [(f"Turn {i}: searching.", name)
                     for i, (name, _a) in enumerate(SEQUENCE)]
                )
            )
            await task.finalize(trace, None)
            return trace, schemas, wire_texts

    trace, schemas, wire_texts = asyncio.run(go())
    files = sorted(pathlib.Path(trace_dir).glob("*.ndjson"))
    assert len(files) == 1, files
    return trace, TS.read_trace(files[0]), schemas, wire_texts


def test_a_real_mcp_rollout_records_its_calls_and_recovers_its_reasoning(served):
    trace, records, _schemas, _wire = served
    assert len(records) == len(SEQUENCE)
    for record, (name, args) in zip(records, SEQUENCE):
        assert record["dispatch_route"] == "mcp"
        assert record["tool_calls"][0] == {"name": name, "arguments": args}
        # `finalize` ran the backfill, so the words are there too.
        assert record["assistant_message"].startswith("Turn ")
        assert record["reasoning"]["available"] is True
        assert record["reasoning"]["source"] == "trace_nodes"
    # And the run's own metrics say how much was captured, so a reader does not
    # have to open the NDJSON to find out.
    assert trace.metrics["reasoning_recovered"] == float(len(SEQUENCE))
    assert trace.metrics["reasoning_unavailable"] == 0.0


def test_the_real_mcp_schema_is_untouched_and_the_result_carries_the_id(served):
    """The barrier's contract, checked on the actual wire: every published
    inputSchema is exactly the v0 adapter's parameters (no `reasoning`, no
    `call_id`, no instrumentation of any kind), while every RESULT ends with
    the correlation marker."""
    import inspect

    from nethack_harness.helpers import _build_skill_adapter_callables

    _trace, records, schemas, wire_texts = served
    assert schemas
    adapters = {a.__name__: a for a in _build_skill_adapter_callables("netplay")}
    for name, schema in schemas.items():
        props = set((schema or {}).get("properties") or {})
        assert props == set(inspect.signature(adapters[name]).parameters), name
        assert not ({"reasoning", "call_id", "state"} & props), name
    for i, (text, record) in enumerate(zip(wire_texts, records), start=1):
        assert text.rstrip().endswith(f"[call#{i}]"), text[-60:]
        assert record["tool_results"][0]["call_id"] == i
        assert record["tool_results"][0]["call_id_echoed"] is True


def test_the_toolset_publishes_the_trace_file_it_is_writing(served):
    """`finalize` runs in the DRIVER process and the NDJSON is written by the
    TOOL SERVER, so the run id has to cross the state channel; without it the
    join has nothing to open."""
    trace, _records, _schemas, _wire = served
    assert trace.state.trace_run_id
    assert trace.state.trace_run_id.startswith("0_")   # <seed>_<pid>_<epoch>
