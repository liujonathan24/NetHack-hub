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
def _drive_mcp_route(trace_dir, plan):
    """Dispatch `plan` the way an MCP tool call does: straight into
    `_apply_tool_call`, with no assistant message anywhere."""

    async def go():
        env = m.load_environment(
            task_spec="full_nle", variant="B0", skill_set=SKILLS, n_examples=1,
            max_turns=50, explicit_seeds=[SEED], character=CHARACTER,
            compact_obs=False, trace_dir=str(trace_dir))
        ex = env.dataset[0]
        state = await env.setup_state(
            {"task": {"seed": SEED}, "info": ex["info"], "prompt": ex["prompt"],
             "responses": [], "turn": 0, "id": "probe", "model": "probe"})
        for name, args in plan:
            await env._apply_tool_call(state, name, args)

    asyncio.run(go())
    files = sorted(pathlib.Path(trace_dir).glob("*.ndjson"))
    assert len(files) == 1, files
    return TS.read_trace(files[0])


PLAN = [("np_press_key", {"key": "s"}), ("np_move_to", {"x": 40, "y": 8})]


@pytest.fixture(scope="module")
def mcp_records(tmp_path_factory):
    return _drive_mcp_route(tmp_path_factory.mktemp("mcp_route"), PLAN)


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
# 5. end to end, over a REAL MCP tool server                                   #
# --------------------------------------------------------------------------- #
SEQUENCE = [("search", {"times": 1}), ("search", {"times": 2})]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """Drive a real `python -m nethack_v1` server over real MCP, then run the
    task's own `finalize`. Costs one engine boot."""
    trace_dir = tmp_path_factory.mktemp("served_turns")

    async def go():
        task, trace = build_task_and_trace(trace_dir=str(trace_dir))
        async with booted_toolset(task, trace) as (session, mini):
            for name, args in SEQUENCE:
                await session.call_tool(name, args)
            trace.nodes.extend(
                _sampled_assistant_nodes(
                    [(f"Turn {i}: searching.", name)
                     for i, (name, _a) in enumerate(SEQUENCE)]
                )
            )
            await task.finalize(trace, None)
            return trace

    trace = asyncio.run(go())
    files = sorted(pathlib.Path(trace_dir).glob("*.ndjson"))
    assert len(files) == 1, files
    return trace, TS.read_trace(files[0])


def test_a_real_mcp_rollout_records_its_calls_and_recovers_its_reasoning(served):
    trace, records = served
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


def test_the_toolset_publishes_the_trace_file_it_is_writing(served):
    """`finalize` runs in the DRIVER process and the NDJSON is written by the
    TOOL SERVER, so the run id has to cross the state channel; without it the
    join has nothing to open."""
    trace, _records = served
    assert trace.state.trace_run_id
    assert trace.state.trace_run_id.startswith("0_")   # <seed>_<pid>_<epoch>
