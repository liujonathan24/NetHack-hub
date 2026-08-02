"""tools/cli_harness_eval/aggregate.py: turns raw per-arm rollouts (a v1
`traces.jsonl` + the per-turn `turns/*.ndjson` the engine writes alongside
it) into the cross-arm results table.

Every column here has a wrong-but-obvious version the design brief rejected;
these tests pin the right one:
  - `died` from `hitpoints == 0` in the turns trace, NEVER `metrics.died`
    (measured wrong on the committed Task 10 artifact: dead from turn 6,
    scored `died = 0`).
  - "actions used" from the MEASURED `total_tool_calls` (control) /
    `skill_calls` (CLI arms) metric, never the nominal call budget.
  - cost from real `calls[].usage` token counts against one GLM 5.2 price
    table, marked unavailable (not estimated) when a rollout's calls carry
    no usage at all.
  - seconds/call from consecutive `t_wall` (or `t_mono`, if present) deltas,
    split into first-half / second-half means.
  - post-death drain: calls issued strictly after the first zero-HP turn.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.cli_harness_eval.aggregate import (
    PRICE_TABLE_GLM_5_2,
    actions_used,
    aggregate_arm,
    post_death_drain,
    rollout_cost,
    rollout_from_turns,
    seconds_per_call_halves,
    to_markdown,
)

ACCEPTANCE = pathlib.Path(__file__).resolve().parents[1] / "tools/cli_harness_eval/acceptance"


def _turns(*lines):
    return list(lines)


# -- died: derived from hitpoints, never metrics.died ------------------------


def test_died_is_derived_from_hitpoints_not_metrics_died():
    """Reproduces the Task 10 finding exactly: hp=0 turns with a rollout that
    would score `metrics.died = 0.0` if that field were trusted."""
    turns = _turns(
        {"turn": 1, "status": {"hitpoints": 16, "experience_level": 1}, "max_dlvl_reached": 1},
        {"turn": 2, "status": {"hitpoints": 0, "experience_level": 1}, "max_dlvl_reached": 2},
        {"turn": 3, "status": {"hitpoints": 0, "experience_level": 1}, "max_dlvl_reached": 2},
    )
    row = rollout_from_turns(turns)
    assert row["died"] is True


def test_alive_rollout_is_not_flagged_died():
    turns = _turns(
        {"turn": 1, "status": {"hitpoints": 16, "experience_level": 1}, "max_dlvl_reached": 1},
        {"turn": 2, "status": {"hitpoints": 10, "experience_level": 2}, "max_dlvl_reached": 2},
    )
    row = rollout_from_turns(turns)
    assert row["died"] is False


def test_top_level_hp_field_is_a_fallback_when_status_is_missing():
    turns = _turns({"turn": 1, "hp": 0, "max_dlvl_reached": 1})
    row = rollout_from_turns(turns)
    assert row["died"] is True


# -- actions used: measured, never nominal -----------------------------------


def test_actions_used_reads_total_tool_calls_for_control():
    trace = {"metrics": {"total_tool_calls": 87.0, "skill_calls": 999.0}}
    assert actions_used(trace, "control") == 87.0


def test_actions_used_reads_skill_calls_for_cli_arms():
    trace = {"metrics": {"skill_calls": 20.0, "total_tool_calls": 999.0}}
    assert actions_used(trace, "claude_code") == 20.0
    assert actions_used(trace, "prime_agent") == 20.0


def test_actions_used_is_none_when_the_metric_is_absent():
    assert actions_used({"metrics": {}}, "control") is None


# -- cost: computed from real usage, marked unavailable when absent ---------


def test_rollout_cost_matches_a_hand_computed_price():
    trace = {
        "calls": [
            {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cached_input_tokens": 0}},
            {"usage": {"prompt_tokens": 2000, "completion_tokens": 50, "cached_input_tokens": 1000}},
        ]
    }
    price = {"input_per_million": 2.0, "cached_input_per_million": 0.5, "output_per_million": 10.0}
    # `prompt_tokens` EXCLUDES cache reads (verifiers.v1.types.Usage: "`input_tokens`
    # adds them back"), so the two are added, never subtracted. The earlier version
    # of this test encoded the opposite -- `1000 fresh = 2000 prompt - 1000 cached`
    # -- and so certified the sign error that dropped 52% of the exp2 sweep's input
    # tokens out of every cost this module reported.
    # call 1: 1000 prompt * 2.0/1e6 + 0 cached + 100 * 10.0/1e6 = 0.002 + 0.001 = 0.003
    # call 2: 2000 prompt * 2.0/1e6 + 1000 cached * 0.5/1e6 + 50 * 10.0/1e6
    #       = 0.004 + 0.0005 + 0.0005 = 0.005
    assert rollout_cost(trace, price) == 0.008


def test_rollout_cost_is_none_when_no_call_carries_usage():
    """Do NOT invert cost into tokens and do NOT estimate: a rollout whose
    calls carry no usage data must report unavailable, not a guess."""
    assert rollout_cost({"calls": []}, PRICE_TABLE_GLM_5_2) is None
    assert rollout_cost({"calls": [{"usage": None}]}, PRICE_TABLE_GLM_5_2) is None
    assert rollout_cost({}, PRICE_TABLE_GLM_5_2) is None


def test_rollout_cost_matches_the_real_claude_code_acceptance_trace():
    """Hand-verified against the committed artifact with the GLM 5.2 price
    table below (independently, outside the aggregator): 22 calls, total
    ~$0.1258."""
    trace = json.loads(
        (ACCEPTANCE / "task13_claude_code_seed0.traces.jsonl").read_text().splitlines()[0]
    )
    cost = rollout_cost(trace, PRICE_TABLE_GLM_5_2)
    assert cost is not None
    assert round(cost, 4) == 0.1258


# -- seconds per call: first half vs second half, from t_wall/t_mono --------


def test_seconds_per_call_splits_into_first_and_second_half():
    turns = _turns(
        {"t_wall": 0.0}, {"t_wall": 10.0}, {"t_wall": 20.0}, {"t_wall": 40.0}, {"t_wall": 80.0}
    )
    # deltas: 10, 10, 20, 40 -> half=2 -> first=[10,10] second=[20,40]
    first, second = seconds_per_call_halves(turns)
    assert first == 10.0
    assert second == 30.0


def test_seconds_per_call_prefers_t_mono_when_present():
    turns = _turns(
        {"t_wall": 0.0, "t_mono": 100.0},
        {"t_wall": 999.0, "t_mono": 105.0},  # t_wall corrupted by a clock step
    )
    first, second = seconds_per_call_halves(turns)
    assert first == second == 5.0


def test_seconds_per_call_growth_measured_on_the_real_acceptance_artifacts():
    """Pins the exact regression the brief measured: ~17s -> ~37s per call."""
    turns = [
        json.loads(l)
        for l in (ACCEPTANCE / "task13_claude_code_seed0.turns.ndjson").read_text().splitlines()
        if l.strip()
    ]
    first, second = seconds_per_call_halves(turns)
    assert 16 < first < 18
    assert 36 < second < 38
    assert second > 2 * first * 0.9  # "roughly 2x growth"


# -- post-death drain ---------------------------------------------------------


def test_post_death_drain_counts_calls_after_the_first_zero_hp_turn():
    turns = _turns(
        {"status": {"hitpoints": 16}},
        {"status": {"hitpoints": 0}},  # death call itself: not drain
        {"status": {"hitpoints": 0}},  # drain
        {"status": {"hitpoints": 0}},  # drain
    )
    assert post_death_drain(turns) == 2


def test_post_death_drain_is_zero_when_never_died():
    turns = _turns({"status": {"hitpoints": 16}}, {"status": {"hitpoints": 10}})
    assert post_death_drain(turns) == 0


def test_post_death_drain_is_near_zero_on_the_real_claude_code_acceptance_artifact():
    """Task 14 fixed live termination; on a healthy rollout this must be ~0 --
    a nonzero value here is the regression signal."""
    turns = [
        json.loads(l)
        for l in (ACCEPTANCE / "task13_claude_code_seed0.turns.ndjson").read_text().splitlines()
        if l.strip()
    ]
    assert post_death_drain(turns) == 0


# -- aggregate_arm + to_markdown: the real acceptance artifacts -------------


def test_aggregate_arm_on_the_real_prime_agent_acceptance_artifact():
    row = aggregate_arm(
        "prime_agent",
        [ACCEPTANCE / "task10_prime_agent_seed0.traces.jsonl"],
        [ACCEPTANCE / "task10_prime_agent_seed0.turns.ndjson"],
    )
    assert row["n"] == 1
    assert row["depth_mean"] == 2.0
    assert row["died_pct"] == 100.0
    assert row["actions_metric_name"] == "skill_calls"
    assert row["actions_mean"] == 12.0
    assert row["cost_n_available"] == 1
    assert row["cost_mean"] is not None
    assert row["post_death_drain_mean"] == 6.0  # died at (1-indexed) turn 6 of 12


def test_aggregate_arm_on_the_real_claude_code_acceptance_artifact():
    row = aggregate_arm(
        "claude_code",
        [ACCEPTANCE / "task13_claude_code_seed0.traces.jsonl"],
        [ACCEPTANCE / "task13_claude_code_seed0.turns.ndjson"],
    )
    assert row["n"] == 1
    assert row["depth_mean"] == 3.0
    assert row["died_pct"] == 0.0
    assert row["actions_metric_name"] == "skill_calls"
    assert row["actions_mean"] == 20.0
    assert row["post_death_drain_mean"] == 0.0


def test_a_rollout_missing_the_actions_metric_does_not_crash_the_arm_aggregate(tmp_path):
    """Scope note: if the aggregation needs a field a trace doesn't carry, the
    aggregator must report it (None / excluded), not synthesize a value."""
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"metrics": {}, "calls": []}) + "\n")
    turns = tmp_path / "r0.ndjson"
    turns.write_text(json.dumps({"status": {"hitpoints": 12, "experience_level": 1}, "max_dlvl_reached": 1}) + "\n")
    row = aggregate_arm("control", [traces], [turns])
    assert row["actions_mean"] is None
    assert row["cost_mean"] is None


def test_to_markdown_renders_a_table_from_both_committed_acceptance_arms():
    rows = [
        aggregate_arm(
            "claude_code",
            [ACCEPTANCE / "task13_claude_code_seed0.traces.jsonl"],
            [ACCEPTANCE / "task13_claude_code_seed0.turns.ndjson"],
        ),
        aggregate_arm(
            "prime_agent",
            [ACCEPTANCE / "task10_prime_agent_seed0.traces.jsonl"],
            [ACCEPTANCE / "task10_prime_agent_seed0.turns.ndjson"],
        ),
    ]
    md = to_markdown(rows)
    assert "claude_code" in md
    assert "prime_agent" in md
    assert "|" in md
