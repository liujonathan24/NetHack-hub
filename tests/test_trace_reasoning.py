"""`tools/trace_reasoning.py`: recover the agent's words, or refuse to guess.

The join is between two files written by two different processes -- the per-turn
NDJSON (tool server) and `traces.jsonl` (eval driver) -- so the only thing that
makes it trustworthy is that it is CHECKED rather than assumed. These tests pin
both halves: the cases where it recovers, and the cases where it must produce an
explicit "unavailable, reason=..." instead of an attractive wrong answer.

The ground-truth case at the bottom is the one that matters most: on the control
arm the record ALREADY carries the assistant text inline, so re-deriving it from
the trace and comparing is a free correctness proof of the join.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trace_reasoning import (  # noqa: E402
    align_records_to_turns,
    assistant_turns_from_nodes,
    backfill_records,
    backfill_run_dir,
    normalize_tool_name,
    record_call_name,
    verify_alignment,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = os.path.join(REPO, "outputs", "pilot_reveal")


def _node(content, names, reasoning="", sampled=True):
    return {
        "sampled": sampled,
        "message": {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": [{"name": n, "arguments": "{}"} for n in names],
        },
    }


def _record(name, assistant_message=""):
    return {
        "tool_calls": [{"name": name, "arguments": {}}],
        "tool_results": [{"name": name, "arguments": {}}],
        "assistant_message": assistant_message,
    }


# --------------------------------------------------------------------------- #
# extraction                                                                   #
# --------------------------------------------------------------------------- #
def test_only_model_sampled_assistant_messages_count():
    """`nodes` is a cumulative prefix replay of the conversation and also holds
    prompt-supplied messages; `sampled` is the provenance flag that separates
    the agent's own turns from context it was handed."""
    nodes = [
        {"sampled": False, "message": {"role": "system", "content": "rules"}},
        {"sampled": False, "message": {"role": "assistant", "content": "few-shot"}},
        _node("mine", ["np_look"]),
        {"sampled": True, "message": {"role": "user", "content": "obs"}},
    ]
    turns = assistant_turns_from_nodes(nodes)
    assert [t["content"] for t in turns] == ["mine"]


def test_the_mcp_tool_prefix_is_stripped():
    """Claude Code's client names the same skill `mcp__nethack__np_look`; the
    trace records `np_look`. Without this the two sides never match and every
    Claude Code rollout would report its reasoning as unrecoverable."""
    assert normalize_tool_name("mcp__nethack__np_look") == "np_look"
    assert normalize_tool_name("np_look") == "np_look"
    turns = assistant_turns_from_nodes([_node("x", ["mcp__nethack__np_move_to"])])
    assert turns[0]["game_tool_names"] == ["np_move_to"]


def test_the_agents_own_scaffolding_tools_are_not_game_calls():
    """Prime Agent's only tool is `ipython`; the skills run inside it. Counting
    `ipython` as a game call would fabricate a 1:1 mapping that does not exist."""
    turns = assistant_turns_from_nodes([_node("x", ["ipython"])])
    assert turns[0]["tool_names"] == ["ipython"]
    assert turns[0]["game_tool_names"] == []


def test_the_dispatched_name_is_read_route_independently():
    assert record_call_name(_record("np_kick")) == "np_kick"
    # version-0/1 record: no `tool_results` at all.
    assert record_call_name({"tool_calls": [{"name": "np_kick", "arguments": "{}"}]}) == "np_kick"
    assert record_call_name({"tool_calls": [], "tool_results": []}) is None


# --------------------------------------------------------------------------- #
# alignment                                                                    #
# --------------------------------------------------------------------------- #
def test_matching_call_sequences_align():
    records = [_record("np_look"), _record("np_move_to")]
    turns = assistant_turns_from_nodes(
        [_node("a", ["np_look"]), _node("b", ["np_move_to"])])
    mapping, mode, reason = align_records_to_turns(records, turns)
    assert (mapping, mode, reason) == ([0, 1], "tool_call_sequence", "")


def test_model_turns_that_called_no_skill_are_skipped_not_misattributed():
    """A CLI agent thinks out loud, reads a file, then acts. Those turns exist
    on the trace and have no record of their own; the walk must step over them
    rather than shift every later pairing by one."""
    records = [_record("np_look"), _record("np_move_to")]
    turns = assistant_turns_from_nodes([
        _node("planning", []),
        _node("a", ["np_look"]),
        _node("reading the wiki", ["ipython"]),
        _node("b", ["np_move_to"]),
    ])
    mapping, mode, _ = align_records_to_turns(records, turns)
    assert mode == "tool_call_sequence"
    assert [turns[i]["content"] for i in mapping] == ["a", "b"]


def test_an_unmatchable_rollout_reports_unavailable_rather_than_guessing():
    """Prime Agent's shape: the model never names a game skill, and the counts
    differ. Positional pairing would attach the wrong paragraph to the wrong
    move, which is worse than no paragraph at all."""
    records = [_record("np_look"), _record("np_move_to"), _record("np_kick")]
    turns = assistant_turns_from_nodes(
        [_node("wrote some python", ["ipython"]), _node("more python", ["ipython"])])
    mapping, mode, reason = align_records_to_turns(records, turns)
    assert mapping == [None, None, None]
    assert mode is None
    assert "disagree on length" in reason

    stats = backfill_records(records, turns)
    assert stats["recovered"] == 0
    assert stats["unavailable"] == 3
    for record in records:
        assert record["reasoning"]["available"] is False
        assert record["reasoning"]["reason"] == reason


def test_equal_counts_fall_back_to_ordinal_pairing():
    records = [_record("np_look"), _record("np_move_to")]
    turns = assistant_turns_from_nodes(
        [_node("a", ["ipython"]), _node("b", ["ipython"])])
    mapping, mode, _ = align_records_to_turns(records, turns)
    assert (mapping, mode) == ([0, 1], "ordinal")


def test_an_empty_trace_says_the_calls_were_never_intercepted():
    records = [_record("np_look")]
    mapping, mode, reason = align_records_to_turns(records, [])
    assert mapping == [None] and mode is None
    assert "no sampled assistant messages" in reason


def test_assistant_turns_harvest_the_call_id_markers_that_follow_them():
    """The barrier's transcript half: `[call#N]` markers in the messages that
    FOLLOW a sampled assistant turn are that turn's issued calls
    (`tools/trace_align.py` owns the join built on them)."""
    turns = assistant_turns_from_nodes([
        _node("I look.", ["np_look"]),
        {"sampled": False, "message": {"role": "tool", "content": "obs\n[call#1]"}},
        _node("I run a loop.", ["ipython"]),
        {"sampled": False,
         "message": {"role": "tool", "content": "[call#2]\n...\n[call#3]"}},
    ])
    assert [t["result_call_ids"] for t in turns] == [[1], [2, 3]]


def test_the_call_id_barrier_outranks_the_name_walk():
    """Two turns call the SAME skill, and the transcript order is crossed
    relative to the records: the name-walk would pair them positionally and
    lie. The id join is tried first and gets the crossed order right."""
    records = [_record("np_look"), _record("np_look")]
    records[0]["tool_results"][0]["call_id"] = 1
    records[1]["tool_results"][0]["call_id"] = 2
    turns = assistant_turns_from_nodes([
        _node("second words", ["np_look"]),
        {"sampled": False, "message": {"role": "tool", "content": "[call#2]"}},
        _node("first words", ["np_look"]),
        {"sampled": False, "message": {"role": "tool", "content": "[call#1]"}},
    ])
    mapping, mode, _ = align_records_to_turns(records, turns)
    assert mode == "call_id"
    assert [turns[j]["content"] for j in mapping] == ["first words", "second words"]

    stats = backfill_records(records, turns)
    assert stats["mode"] == "call_id"
    assert [r["assistant_message"] for r in records] == ["first words", "second words"]
    assert {r["reasoning"]["alignment"] for r in records} == {"call_id"}


def test_pre_barrier_traces_fall_back_to_the_name_walk_and_say_why():
    """Old records carry no ids; the join refuses explicitly and the name-walk
    keeps working exactly as before -- and an unalignable rollout's reason now
    names the missing barrier too."""
    records = [_record("np_look"), _record("np_move_to")]
    turns = assistant_turns_from_nodes(
        [_node("a", ["np_look"]), _node("b", ["np_move_to"])])
    mapping, mode, _ = align_records_to_turns(records, turns)
    assert (mapping, mode) == ([0, 1], "tool_call_sequence")

    unalignable = [_record("np_look"), _record("np_move_to"), _record("np_kick")]
    ipython_only = assistant_turns_from_nodes(
        [_node("python", ["ipython"]), _node("python", ["ipython"])])
    _mapping, mode, reason = align_records_to_turns(unalignable, ipython_only)
    assert mode is None
    assert "call-id join unavailable" in reason
    assert "before the call-id barrier" in reason


# --------------------------------------------------------------------------- #
# the self-check                                                               #
# --------------------------------------------------------------------------- #
def test_a_mapping_that_contradicts_the_inline_text_is_rejected():
    """The control arm audits the join for free: it already knows the answer.
    An `ordinal` pairing that disagrees with it is proof the pairing is wrong,
    and must not be used to overwrite anything."""
    records = [_record("np_look", "I look."), _record("np_move_to", "I move.")]
    turns = assistant_turns_from_nodes(
        [_node("I move.", ["ipython"]), _node("I look.", ["ipython"])])  # swapped
    mapping, mode, _ = align_records_to_turns(records, turns)
    assert mode == "ordinal"
    assert verify_alignment(records, turns, mapping)

    stats = backfill_records(records, turns)
    assert stats["mode"] is None
    assert "attribute reasoning to the wrong move" in stats["reason"]
    # ... and the good records were left exactly as the harness wrote them.
    assert [r["assistant_message"] for r in records] == ["I look.", "I move."]


def test_inline_text_is_never_replaced_but_gains_the_reasoning_channel():
    """`reasoning_content` is a separate provider field the writer never had.
    Attaching it is additive; the visible message stays the harness's."""
    records = [_record("np_look", "I look.")]
    turns = assistant_turns_from_nodes(
        [_node("I look.", ["np_look"], reasoning="considered kicking")])
    stats = backfill_records(records, turns)
    assert stats["already"] == 1 and stats["recovered"] == 0
    assert records[0]["assistant_message"] == "I look."
    assert records[0]["reasoning"]["source"] == "assistant_message"
    assert records[0]["reasoning"]["reasoning_text"] == "considered kicking"


# --------------------------------------------------------------------------- #
# the directory driver                                                         #
# --------------------------------------------------------------------------- #
def _fake_run(tmp_path, *, seed=0, pid=999, epoch=1700000000):
    cell = tmp_path / "claude_code"
    (cell / "turns").mkdir(parents=True)
    records = [_record("np_look"), _record("np_move_to")]
    for record in records:
        record["t_mono"] = 1.0
    (cell / "turns" / f"{seed}_{pid}_{epoch}.ndjson").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    trace = {
        "task": {"data": {"idx": seed}},
        "nodes": [_node("first", ["mcp__nethack__np_look"]),
                  _node("second", ["mcp__nethack__np_move_to"])],
    }
    (cell / "traces.jsonl").write_text(json.dumps(trace) + "\n")
    return cell / "turns" / f"{seed}_{pid}_{epoch}.ndjson"


def test_the_directory_driver_backfills_a_whole_run(tmp_path):
    path = _fake_run(tmp_path)
    stats = backfill_run_dir(tmp_path, warn=lambda _m: None)
    assert [s["recovered"] for s in stats] == [2]
    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["assistant_message"] for r in written] == ["first", "second"]


def test_a_dry_run_writes_nothing(tmp_path):
    path = _fake_run(tmp_path)
    before = path.read_text()
    backfill_run_dir(tmp_path, warn=lambda _m: None, dry_run=True)
    assert path.read_text() == before


def test_the_rewrite_is_atomic_and_leaves_no_temp_file(tmp_path):
    """These files are the only per-turn record of a run that cost real money;
    a half-written replacement would destroy it."""
    path = _fake_run(tmp_path)
    backfill_run_dir(tmp_path, warn=lambda _m: None)
    assert list(path.parent.glob("*.tmp")) == []
    assert len(path.read_text().strip().splitlines()) == 2


def test_the_cli_reports_per_rollout_and_exits_zero(tmp_path):
    _fake_run(tmp_path)
    out = subprocess.run(
        [sys.executable, "-m", "tools.trace_reasoning", str(tmp_path), "--dry-run"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert out.returncode == 0, out.stderr
    assert "tool_call_sequence" in out.stdout
    assert "dry run" in out.stdout


# --------------------------------------------------------------------------- #
# ground truth                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(PILOT), reason="pilot run not on this box")
def test_the_recovered_text_equals_the_control_arms_own_record():
    """The strongest available check, on real data: recompute from
    `traces.jsonl` what the harness wrote inline and require them to be equal.
    Measured 396/396 across `outputs/pilot_reveal`'s four 99-turn rollouts."""
    from tools.eval_metrics import read_ndjson, select_turn_files
    from tools.trace_reasoning import assistant_turns_from_trace

    checked = 0
    for cell in sorted(os.listdir(PILOT)):
        cell_dir = os.path.join(PILOT, cell)
        traces = os.path.join(cell_dir, "traces.jsonl")
        if not os.path.exists(traces):
            continue
        by_seed = {}
        for line in open(traces):
            if line.strip():
                trace = json.loads(line)
                by_seed[trace["task"]["data"]["idx"]] = trace
        chosen, _ = select_turn_files(cell_dir, seeds=set(by_seed))
        for seed, path in sorted(chosen.items()):
            records = read_ndjson(path)
            inline = [r.get("assistant_message") or "" for r in records]
            stats = backfill_records(
                records, assistant_turns_from_trace(by_seed[seed]), overwrite=True)
            assert stats["mode"] == "tool_call_sequence", (cell, seed, stats)
            recovered = [r.get("assistant_message") or "" for r in records]
            assert recovered == inline, (cell, seed)
            checked += len(records)
    assert checked >= 99, f"expected real pilot data, checked {checked} turns"
