"""The three accounting repairs from the exp2 post-mortem, pinned.

Each of these certifies a defect that already cost a sweep:

  * `_call_cost` read `cached_input_tokens` as a SUBSET of `prompt_tokens` and
    subtracted it. It is a disjoint addend -- `verifiers.v1.types.Usage` says
    so, and the traces show a first call with `prompt_tokens=28,
    cached_input_tokens=384`. The sweep's 334M cache-read tokens (52.4% of all
    input) were priced at zero.

  * Rollouts the PROVIDER killed (`upstream 402 insufficient_funds`) were
    written into `traces.jsonl` as ordinary records and averaged into every
    column. 31 of 80.

  * The stall watchdog SIGKILLed rollouts and quarantined their turn files, and
    the eval CLI -- which writes `traces.jsonl` only when a rollout finishes --
    wrote nothing at all for them. 50 attempts, 32,145 in-game turns, invisible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import stall_watchdog as sw  # noqa: E402
from tools.cli_harness_eval import aggregate as agg  # noqa: E402


def _turn(turn, dlvl, hp=10, xl=1, time_=None):
    return {
        "turn": turn,
        "dlvl": dlvl,
        "max_dlvl_reached": dlvl,
        "hp": hp,
        "max_hp": 16,
        "t_wall": 1000.0 + turn,
        "status": {"experience_level": xl, "time": time_ if time_ is not None else turn * 10,
                   "hitpoints": hp},
        "tool_calls": [{"name": "np_move_to"}],
    }


def _cell(tmp_path, name, traces, turn_files=()):
    cell = tmp_path / name
    (cell / "turns").mkdir(parents=True)
    (cell / "config.toml").write_text('model = "z-ai/glm-5.2"\n')
    if traces is not None:
        (cell / "traces.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in traces)
        )
    for fname, rows in turn_files:
        (cell / "turns" / fname).write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return cell


# ---------------------------------------------------------------------------
# cost: cache reads are an ADDEND, and the price table is the provider's
# ---------------------------------------------------------------------------


def test_cached_input_tokens_are_added_not_subtracted():
    """The exact usage shape from the sweep: cached > prompt. Under the old
    whole-and-part reading this charged 28 tokens; it is 412."""
    price = {"input_per_million": 1.0, "cached_input_per_million": 1.0,
             "output_per_million": 0.0}
    call = {"usage": {"prompt_tokens": 28, "completion_tokens": 0,
                      "cached_input_tokens": 384}}
    assert agg._call_cost(call, price) == (28 + 384) / 1_000_000


def test_reasoning_tokens_are_not_billed_twice():
    """`reasoning_tokens` is a subset of `completion_tokens`, so adding it
    again would inflate output spend by ~40% on a reasoning model."""
    price = {"input_per_million": 0.0, "cached_input_per_million": 0.0,
             "output_per_million": 1.0}
    call = {"usage": {"prompt_tokens": 0, "completion_tokens": 100,
                      "cached_input_tokens": 0, "reasoning_tokens": 60}}
    assert agg._call_cost(call, price) == 100 / 1_000_000


def test_glm_5_2_is_priced_at_primes_published_rate():
    """$1.68 / $5.28, from `/models`. The committed table said $1.40 / $4.40,
    which is where 20% of the 6.6x projection error came from. Cache reads
    carry the full input rate because the endpoint publishes no discount."""
    table = agg.PRICE_TABLES["z-ai/glm-5.2"]
    assert table["input_per_million"] == 1.68
    assert table["output_per_million"] == 5.28
    assert table["cached_input_per_million"] == table["input_per_million"]


# ---------------------------------------------------------------------------
# error stubs: excluded from scores, still counted in spend
# ---------------------------------------------------------------------------


def test_is_error_trace_recognises_both_spellings_and_the_all_errors_shape():
    assert agg.is_error_trace({"stop_condition": "error"})
    assert agg.is_error_trace({"stop_condition": "ProviderError"})
    assert agg.is_error_trace({"stop_condition": "HarnessError"})
    assert agg.is_error_trace({"stop_condition": "harness_timeout"})
    assert agg.is_error_trace({"calls": [{"error": {"status_code": 402}}]})
    assert not agg.is_error_trace({"stop_condition": "game_over"})
    assert not agg.is_error_trace({"stop_condition": "agent_completed"})
    # A watchdog kill is NOT an infrastructure stub: the depth it reached is a
    # real lower bound on what the agent did.
    assert not agg.is_error_trace({"stop_condition": "watchdog_stall"})


def test_a_402_stub_does_not_dilute_the_cells_depth(tmp_path):
    """Two real rollouts at dlvl 8 and a 402 stub. The stub must not appear as
    a third, shallow rollout -- that is how three exp2 cells fell to n=1."""
    traces = [
        {"task": {"data": {"idx": 0}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 8.0, "skill_calls": 100.0},
         "calls": [{"usage": {"prompt_tokens": 1000, "completion_tokens": 10}}]},
        {"task": {"data": {"idx": 1}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 8.0, "skill_calls": 100.0},
         "calls": [{"usage": {"prompt_tokens": 1000, "completion_tokens": 10}}]},
        {"task": {"data": {"idx": 2}}, "stop_condition": "error",
         "metrics": {}, "calls": [{"error": {"status_code": 402}}]},
    ]
    turns = [
        ("0_100_1.ndjson", [_turn(1, 1), _turn(2, 8)]),
        ("1_101_1.ndjson", [_turn(1, 1), _turn(2, 8)]),
    ]
    cell = _cell(tmp_path, "claude_code", traces, turns)
    row = agg.aggregate_run(str(tmp_path), warn=lambda _m: None)[0]
    assert row["n"] == 2
    assert row["n_error"] == 1
    assert row["depth_mean"] == 8.0
    assert cell.exists()


def test_a_402_that_truncated_a_real_rollout_is_not_scored_at_its_cut_depth(tmp_path):
    """The nastier half: the 402 landed mid-game, so a turn file exists and
    holds a real but CENSORED depth. Scoring it would report a rollout the
    provider stopped as one the agent could not push past."""
    traces = [
        {"task": {"data": {"idx": 0}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 9.0}, "calls": []},
        {"task": {"data": {"idx": 1}}, "stop_condition": "ProviderError",
         "metrics": {"max_dlvl_reached": 2.0}, "calls": [{"error": {"status_code": 402}}]},
    ]
    turns = [
        ("0_100_1.ndjson", [_turn(1, 1), _turn(2, 9)]),
        ("1_101_1.ndjson", [_turn(1, 1), _turn(2, 2)]),
    ]
    _cell(tmp_path, "claude_code", traces, turns)
    row = agg.aggregate_run(str(tmp_path), warn=lambda _m: None)[0]
    assert row["n"] == 1
    assert row["n_error"] == 1
    assert row["n_turnfiles_error_only"] == 1
    assert row["depth_mean"] == 9.0


def test_error_stub_spend_is_still_counted(tmp_path):
    """Excluded from the SCORE, not from the bill. A retried attempt that
    burned 190 calls before the 402 spent real money."""
    traces = [
        {"task": {"data": {"idx": 0}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 3.0},
         "calls": [{"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}]},
        {"task": {"data": {"idx": 1}}, "stop_condition": "ProviderError",
         "metrics": {"max_dlvl_reached": 2.0},
         "calls": [{"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}]},
    ]
    _cell(tmp_path, "claude_code", traces,
          [("0_100_1.ndjson", [_turn(1, 3)])])
    row = agg.aggregate_run(str(tmp_path), warn=lambda _m: None)[0]
    # One scored rollout at 1M prompt tokens ...
    assert round(row["cost_mean"], 4) == round(1.68, 4)
    # ... but two rollouts' worth of tokens left the wallet.
    assert round(row["spend_traced"], 4) == round(2 * 1.68, 4)


# ---------------------------------------------------------------------------
# depth from metrics: a missing turn file is not dlvl 1
# ---------------------------------------------------------------------------


def test_depth_comes_from_metrics_when_the_turn_file_is_gone(tmp_path):
    """Quarantined or never written, the turn file is not the only record --
    `NetHackTask.finalize` copies the real depth onto `trace.metrics`. Both
    scorers used to reach it through `max_dlvl_reached or 1` and score 1."""
    traces = [
        {"task": {"data": {"idx": 0}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 13.0, "max_xp_level": 6.0}, "calls": []},
    ]
    _cell(tmp_path, "claude_code", traces, turn_files=())
    row = agg.aggregate_run(str(tmp_path), warn=lambda _m: None)[0]
    assert row["n"] == 1
    assert row["n_from_metrics"] == 1
    assert row["depth_mean"] == 13.0
    # No turn file means no latency and no game-turn denominator; those stay
    # unavailable rather than being synthesized from the trace.
    assert row["sec_per_call_first_half_mean"] is None


def test_the_turns_channel_wins_when_both_exist(tmp_path):
    """`metrics` is the fallback, never an extra rollout: a seed present in
    both channels must count ONCE, through the richer per-turn record."""
    traces = [
        {"task": {"data": {"idx": 0}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 5.0}, "calls": []},
    ]
    _cell(tmp_path, "claude_code", traces,
          [("0_100_1.ndjson", [_turn(1, 1), _turn(2, 5)])])
    row = agg.aggregate_run(str(tmp_path), warn=lambda _m: None)[0]
    assert row["n"] == 1
    assert row["n_from_metrics"] == 0


# ---------------------------------------------------------------------------
# partial traces: a killed attempt is not a rollout that never happened
# ---------------------------------------------------------------------------


def test_partial_trace_reconstructs_depth_and_leaves_cost_unavailable(tmp_path):
    path = tmp_path / "2_4242_1785603356.ndjson"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        _turn(1, 1, xl=1, time_=10),
        _turn(2, 4, xl=3, time_=250),
    ]) + '{"turn": 3, "dlvl": 4, "hp"')  # torn final line, as a SIGKILL leaves
    trace = sw.partial_trace_from_turns(str(path), 2, {"pid": 4242})
    assert trace["partial"] is True
    assert trace["stop_condition"] == "watchdog_stall"
    assert trace["task"]["data"]["idx"] == 2
    assert trace["metrics"]["max_dlvl_reached"] == 4.0
    assert trace["metrics"]["game_turns"] == 250.0
    # Token usage died with the process. Report nothing rather than guess.
    assert trace["calls"] == []
    assert agg.rollout_cost(trace, agg.PRICE_TABLE_GLM_5_2) is None


def test_partial_trace_is_none_for_a_file_with_no_parseable_record(tmp_path):
    path = tmp_path / "0_1_2.ndjson"
    path.write_text('{"turn": 1, "dlv')
    assert sw.partial_trace_from_turns(str(path), 0, {}) is None


def test_a_killed_attempt_is_scored_when_nothing_superseded_it(tmp_path):
    traces = [
        {"task": {"data": {"idx": 0}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 2.0}, "calls": []},
    ]
    cell = _cell(tmp_path, "prime_agent", traces,
                 [("0_100_1.ndjson", [_turn(1, 2)])])
    (cell / "traces.partial.jsonl").write_text(json.dumps({
        "partial": True,
        "task": {"data": {"idx": 1}},
        "stop_condition": "watchdog_stall",
        "calls": [],
        "metrics": {"max_dlvl_reached": 6.0, "max_xp_level": 1.0},
    }) + "\n")
    row = agg.aggregate_run(str(tmp_path), warn=lambda _m: None)[0]
    assert row["n"] == 2
    assert row["n_partial"] == 1
    assert row["depth_mean"] == 4.0  # (2 + 6) / 2


def test_a_killed_attempt_that_was_retried_is_not_counted_twice(tmp_path):
    """The retry produced a real record for the same seed. Counting both would
    inflate `n` with one seed twice -- the failure `select_turn_files` already
    guards against on the turns channel."""
    traces = [
        {"task": {"data": {"idx": 0}}, "stop_condition": "game_over",
         "metrics": {"max_dlvl_reached": 7.0}, "calls": []},
    ]
    cell = _cell(tmp_path, "prime_agent", traces,
                 [("0_100_1.ndjson", [_turn(1, 7)])])
    (cell / "traces.partial.jsonl").write_text(json.dumps({
        "partial": True,
        "task": {"data": {"idx": 0}},
        "stop_condition": "watchdog_stall",
        "calls": [],
        "metrics": {"max_dlvl_reached": 2.0},
    }) + "\n")
    row = agg.aggregate_run(str(tmp_path), warn=lambda _m: None)[0]
    assert row["n"] == 1
    assert row["n_partial"] == 0
    assert row["n_partial_superseded"] == 1
    assert row["depth_mean"] == 7.0


def test_backfill_is_idempotent(tmp_path):
    """Reruns must not double a run's `n`. The file is rewritten, not
    appended."""
    cell = tmp_path / "prime_agent"
    qdir = cell / "turns.stalled" / "20260801T171254Z_pid88064"
    qdir.mkdir(parents=True)
    (cell / "turns").mkdir()
    (qdir / "3_88064_1785603356.ndjson").write_text(
        json.dumps(_turn(1, 5)) + "\n"
    )
    assert sw.backfill(str(tmp_path), say=lambda _m: None) == 1
    assert sw.backfill(str(tmp_path), say=lambda _m: None) == 1
    lines = (cell / "traces.partial.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["metrics"]["max_dlvl_reached"] == 5.0
