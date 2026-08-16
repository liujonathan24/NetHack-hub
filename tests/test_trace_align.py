"""`tools/trace_align.py`: the call-id join between the two log streams.

The barrier's whole value is that the join is exact or refused -- never a
guess. These tests pin both halves on synthetic data (the end-to-end proof
over a real rollout lives in
`environments/nethack/tests/test_cli_arm_reasoning.py::
test_the_call_id_join_is_total_and_unique_end_to_end`).
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trace_align import (  # noqa: E402
    align_coverage,
    align_records_to_turns_by_call_id,
    content_text,
    marker_call_ids,
    record_call_id,
)
from tools.trace_reasoning import assistant_turns_from_nodes  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _assistant(content, sampled=True, names=()):
    return {
        "sampled": sampled,
        "message": {
            "role": "assistant",
            "content": content,
            "reasoning_content": "",
            "tool_calls": [{"name": n, "arguments": "{}"} for n in names],
        },
    }


def _result(text, role="tool"):
    return {"sampled": False, "message": {"role": role, "content": text}}


def _record(call_id, name="np_look"):
    return {
        "tool_calls": [{"name": name, "arguments": {}}],
        "tool_results": [{"name": name, "arguments": {}, "call_id": call_id}],
        "assistant_message": "",
    }


# --------------------------------------------------------------------------- #
# primitives                                                                   #
# --------------------------------------------------------------------------- #
def test_markers_parse_in_order_including_several_in_one_message():
    """An ipython block prints several results in execution order; that order
    is the sub-ordering of the block's calls and must survive parsing."""
    text = "ran skills:\n[call#3]\nmoved\n[call#4]\nsearched [call#5] done"
    assert marker_call_ids(text) == [3, 4, 5]
    assert marker_call_ids("no markers here") == []
    assert marker_call_ids(None) == []


def test_content_text_flattens_block_lists():
    blocks = [{"type": "text", "text": "a [call#1]"},
              {"type": "image_url", "image_url": {}},
              {"type": "text", "text": "[call#2]"}]
    assert marker_call_ids(content_text(blocks)) == [1, 2]
    assert content_text("plain") == "plain"
    assert content_text({"weird": True}) == ""


def test_record_call_id_reads_the_stamp_and_is_honest_about_absence():
    assert record_call_id(_record(7)) == 7
    # Explicit null (the never-dispatched flush) and pre-barrier records
    # (no key at all) both mean "nothing to join on".
    assert record_call_id(_record(None)) is None
    assert record_call_id({"tool_results": [{"name": "np_look"}]}) is None
    assert record_call_id({"tool_results": []}) is None
    assert record_call_id({}) is None


# --------------------------------------------------------------------------- #
# harvesting: markers attach to the assistant turn that issued the calls       #
# --------------------------------------------------------------------------- #
def test_markers_are_attributed_to_the_preceding_sampled_assistant_turn():
    nodes = [
        _assistant("I will look.", names=["np_look"]),
        _result("=== MAP ===\n...\n[call#1]"),
        _assistant("Now I move.", names=["np_move_to"]),
        _result("=== MAP ===\n...\n[call#2]"),
    ]
    turns = assistant_turns_from_nodes(nodes)
    assert [t["result_call_ids"] for t in turns] == [[1], [2]]


def test_an_ipython_block_owns_every_call_its_output_echoes():
    """One assistant turn -> many game calls: exactly the shape that defeats
    the name-walk (the model only ever called `ipython`), and exactly what the
    barrier exists to solve."""
    nodes = [
        _assistant("Running a loop over three skills.", names=["ipython"]),
        _result("out:\n[call#4]\nmoved\n[call#5]\nsearched\n[call#6]\nlooked"),
    ]
    turns = assistant_turns_from_nodes(nodes)
    assert turns[0]["result_call_ids"] == [4, 5, 6]

    records = [_record(4), _record(5), _record(6)]
    mapping, mode, reason = align_records_to_turns_by_call_id(records, turns)
    assert (mapping, mode, reason) == ([0, 0, 0], "call_id", "")


def test_markers_before_any_sampled_turn_are_prompt_context_and_dropped():
    """A replayed prefix (resume-from-trace, few-shot) can contain old markers;
    attributing them to a turn of THIS rollout would be fabrication."""
    nodes = [
        _result("old session tail ... [call#9]", role="user"),
        _assistant("fresh turn", names=["np_look"]),
        _result("[call#1]"),
    ]
    turns = assistant_turns_from_nodes(nodes)
    assert [t["result_call_ids"] for t in turns] == [[1]]


# --------------------------------------------------------------------------- #
# the join: exact or refused                                                   #
# --------------------------------------------------------------------------- #
def test_the_join_is_exact_even_where_ordinal_pairing_would_lie():
    """Interleaved non-game turns give the two channels different lengths and
    orders; the id join does not care."""
    nodes = [
        _assistant("thinking out loud", names=[]),          # no call issued
        _assistant("first move", names=["np_look"]),
        _result("[call#1]"),
        _assistant("reading the wiki", names=["ipython"]),  # non-game call
        _result("wiki text, no marker"),
        _assistant("second move", names=["np_move_to"]),
        _result("[call#2]"),
    ]
    turns = assistant_turns_from_nodes(nodes)
    records = [_record(1), _record(2)]
    mapping, mode, _ = align_records_to_turns_by_call_id(records, turns)
    assert mode == "call_id"
    assert [turns[j]["content"] for j in mapping] == ["first move", "second move"]


def test_records_without_ids_say_so_instead_of_guessing():
    records = [_record(None), {"tool_calls": [], "tool_results": []}]
    turns = assistant_turns_from_nodes([_assistant("x"), _result("[call#1]")])
    mapping, mode, reason = align_records_to_turns_by_call_id(records, turns)
    assert mapping == [None, None] and mode is None
    assert "before the call-id barrier" in reason


def test_a_transcript_without_markers_names_the_echo_as_the_gap():
    records = [_record(1)]
    turns = assistant_turns_from_nodes([_assistant("x", names=["np_look"])])
    mapping, mode, reason = align_records_to_turns_by_call_id(records, turns)
    assert mapping == [None] and mode is None
    assert "no [call#N] markers" in reason


def test_an_ambiguous_id_refuses_the_whole_join():
    """The same id under two assistant turns can only mean corruption; a join
    that picked one silently would be worse than no join."""
    nodes = [
        _assistant("a", names=["np_look"]), _result("[call#1]"),
        _assistant("b", names=["np_look"]), _result("[call#1]"),
    ]
    turns = assistant_turns_from_nodes(nodes)
    mapping, mode, reason = align_records_to_turns_by_call_id([_record(1)], turns)
    assert mode is None
    assert "more than one assistant turn" in reason


def test_an_id_with_no_marker_refuses_rather_than_partially_joining():
    turns = assistant_turns_from_nodes(
        [_assistant("a", names=["np_look"]), _result("[call#1]")])
    mapping, mode, reason = align_records_to_turns_by_call_id(
        [_record(1), _record(2)], turns)
    assert mode is None
    assert "found no matching" in reason


def test_coverage_reports_the_barriers_promise():
    nodes = [_assistant("a", names=["np_look"]), _result("[call#1]"),
             _assistant("b", names=["np_move_to"]), _result("[call#2]")]
    turns = assistant_turns_from_nodes(nodes)
    records = [_record(1), _record(2), _record(None)]     # incl. a flush
    coverage = align_coverage(records, turns)
    assert coverage["n"] == 3
    assert coverage["with_id"] == 2 and coverage["without_id"] == 1
    assert coverage["joined"] == 2
    assert coverage["mode"] == "call_id"
    assert coverage["total"] is True


# --------------------------------------------------------------------------- #
# integration with the reasoning backfill                                      #
# --------------------------------------------------------------------------- #
def test_the_backfill_rides_the_barrier_and_beats_the_ordinal_guess():
    """Same channel lengths, crossed order: ordinal pairing would attach each
    paragraph to the wrong move. The id join gets it right, and the recovered
    provenance says which strategy was used."""
    from tools.trace_reasoning import align_records_to_turns, backfill_records

    nodes = [
        _assistant("I search here.", names=["ipython"]),
        _result("[call#2]"),        # served SECOND record's call
        _assistant("I descend now.", names=["ipython"]),
        _result("[call#1]"),        # served FIRST record's call
    ]
    turns = assistant_turns_from_nodes(nodes)
    records = [_record(1, "np_down"), _record(2, "np_press_key")]

    mapping, mode, _ = align_records_to_turns(records, turns)
    assert mode == "call_id"
    assert [turns[j]["content"] for j in mapping] == \
        ["I descend now.", "I search here."]

    stats = backfill_records(records, turns)
    assert stats["mode"] == "call_id" and stats["recovered"] == 2
    assert records[0]["assistant_message"] == "I descend now."
    assert records[1]["assistant_message"] == "I search here."
    assert {r["reasoning"]["alignment"] for r in records} == {"call_id"}
    assert {r["reasoning"]["source"] for r in records} == {"trace_nodes"}


# --------------------------------------------------------------------------- #
# the directory driver                                                         #
# --------------------------------------------------------------------------- #
def _fake_run(tmp_path, *, seed=0, pid=999, epoch=1700000000):
    cell = tmp_path / "claude_code"
    (cell / "turns").mkdir(parents=True)
    records = [_record(1, "np_look"), _record(2, "np_move_to")]
    for record in records:
        record["t_mono"] = 1.0
    (cell / "turns" / f"{seed}_{pid}_{epoch}.ndjson").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    trace = {
        "task": {"data": {"idx": seed}},
        "nodes": [
            _assistant("first", names=["mcp__nethack__np_look"]),
            _result("obs\n[call#1]"),
            _assistant("second", names=["mcp__nethack__np_move_to"]),
            _result("obs\n[call#2]"),
        ],
    }
    (cell / "traces.jsonl").write_text(json.dumps(trace) + "\n")


def test_the_driver_reports_coverage_per_rollout(tmp_path):
    from tools.trace_align import report_run_dir

    _fake_run(tmp_path)
    stats = report_run_dir(tmp_path, warn=lambda _m: None)
    assert len(stats) == 1
    assert stats[0]["total"] is True
    assert stats[0]["joined"] == stats[0]["with_id"] == 2


def test_the_cli_reports_and_exits_zero(tmp_path):
    _fake_run(tmp_path)
    out = subprocess.run(
        [sys.executable, "-m", "tools.trace_align", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert out.returncode == 0, out.stderr
    assert "call_id" in out.stdout
    assert "True" in out.stdout
