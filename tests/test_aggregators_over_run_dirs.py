"""End-to-end tests for the two path-based aggregators, over REAL run dirs.

`tools/cli_harness_eval/aggregate.py` and `tools/encoding_eval/aggregate.py`
are the things that actually produce published numbers, so these tests drive
them the way the runbook does -- pointed at `outputs/pilot_reveal`, a real
2-cell / 2-seed GLM-5.2 pilot, and at `outputs/trace_probe`, a real
glm-4.7-flash probe.

Pinned here:
  - a sweep laid out by VARIANT name (`{fog,reveal}`) produces rows, not an
    empty table with exit 0;
  - an input with no cells fails loudly and non-zero;
  - a retried rollout stays ONE row;
  - a cell whose model has no price table reports cost unavailable rather than
    at another model's rates;
  - every emitted table carries BOTH BALROG numbers and the pace columns.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.cli_harness_eval import aggregate as cli_agg  # noqa: E402
from tools.cli_harness_eval import progress as cli_progress  # noqa: E402
from tools.encoding_eval import aggregate as enc_agg  # noqa: E402
from tools.eval_metrics import read_ndjson, turn_file_parts  # noqa: E402

PILOT = REPO / "outputs/pilot_reveal"
PROBE_RUN = REPO / "outputs/trace_probe"

_needs_pilot = pytest.mark.skipif(
    not (PILOT / "reveal/traces.jsonl").exists(), reason="outputs/pilot_reveal not present"
)
_needs_probe = pytest.mark.skipif(
    not (PROBE_RUN / "B0_reveal/traces.jsonl").exists(), reason="outputs/trace_probe not present"
)


def _rows(run_dir):
    """`aggregate_run` with warnings captured instead of printed."""
    warnings: list[str] = []
    rows = cli_agg.aggregate_run(str(run_dir), warn=warnings.append)
    return rows, warnings


# ---------------------------------------------------------------------------
# BUG 3: unknown / non-arm cell names must not yield a silent empty table
# ---------------------------------------------------------------------------


@_needs_pilot
def test_a_variant_named_sweep_is_aggregated_not_silently_empty():
    """`outputs/pilot_reveal/{fog,reveal}` are not named control/claude_code/
    prime_agent. The old hardcoded arm loop returned zero rows and exit 0."""
    assert cli_agg.discover_cells(str(PILOT)) == ["fog", "reveal"]
    rows, _ = _rows(PILOT)
    assert [r["cell"] for r in rows] == ["fog", "reveal"]
    assert all(r["n"] == 2 for r in rows)


def test_the_canonical_arm_names_still_sort_first(tmp_path):
    for name in ("zzz_variant", "claude_code", "control"):
        (tmp_path / name / "turns").mkdir(parents=True)
    assert enc_agg.discover_cells(str(tmp_path)) == ["claude_code", "control", "zzz_variant"]
    assert cli_agg.discover_cells(str(tmp_path)) == ["control", "claude_code", "zzz_variant"]


def test_an_input_with_no_cells_is_loud_and_non_zero(tmp_path, capsys):
    (tmp_path / "not_a_cell").mkdir()
    rows, warnings = _rows(tmp_path)
    assert rows == []
    assert any("no cells found" in w for w in warnings)

    assert cli_agg.main([str(tmp_path)]) == 1
    assert enc_agg.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "ZERO rows" in err
    assert not (tmp_path / "table.md").exists(), "must not write an empty table"


def test_a_missing_run_directory_is_non_zero(tmp_path):
    assert cli_agg.main([str(tmp_path / "nope")]) == 2
    assert enc_agg.main([str(tmp_path / "nope")]) == 2
    assert cli_agg.main([]) == 2


# ---------------------------------------------------------------------------
# BUG 4: retries must not inflate n, through the whole aggregator
# ---------------------------------------------------------------------------


@_needs_pilot
def test_a_duplicate_pid_file_does_not_double_count_a_seed(tmp_path):
    """Simulated retry on real data: the fog cell gets a second attempt file
    for seed 0 (same PID, later timestamp -- an in-process retry) and a third
    from a relaunch (different PID). `glob('turns/*.ndjson')` would report
    n=4; the aggregator must still report n=2 with unchanged depth."""
    run = tmp_path / "run"
    run.mkdir()
    cell = run / "fog"
    shutil.copytree(PILOT / "fog", cell, ignore=shutil.ignore_patterns("*.log"))

    before, _ = _rows(run)
    assert before[0]["n"] == 2

    turns = cell / "turns"
    real = next(turns.glob("0_*.ndjson"))
    _, pid, ts = turn_file_parts(real)
    head = "\n".join(real.read_text().splitlines()[:3]) + "\n"
    (turns / f"0_{pid}_{ts + 30}.ndjson").write_text(head)
    (turns / f"0_{pid + 5}_{ts + 600}.ndjson").write_text(head)
    assert len(list(turns.glob("*.ndjson"))) == 4

    after, warnings = _rows(run)
    assert after[0]["n"] == 2, "one row per seed, not one per retry attempt"
    assert after[0]["depth_mean"] == before[0]["depth_mean"]
    assert sum("superseded retry" in w for w in warnings) == 2


@_needs_pilot
def test_progress_report_also_keeps_one_row_per_seed_under_a_retry(tmp_path):
    cell = tmp_path / "fog"
    shutil.copytree(PILOT / "fog", cell, ignore=shutil.ignore_patterns("*.log"))
    turns = cell / "turns"
    real = next(turns.glob("1_*.ndjson"))
    _, pid, ts = turn_file_parts(real)
    (turns / f"1_{pid}_{ts + 5}.ndjson").write_text(
        "\n".join(real.read_text().splitlines()[:2]) + "\n"
    )
    rows, skills, _, dropped = cli_progress.rows_for_cell(str(cell))
    assert [r["seed"] for r in rows] == [0, 1]
    assert len(dropped) == 1
    assert sum(skills.values()) == 198  # 99 applied calls x 2 seeds, not 10,080


# ---------------------------------------------------------------------------
# BUG 1 / 2, through the aggregator
# ---------------------------------------------------------------------------


@_needs_pilot
def test_progress_reports_real_skill_totals_not_replayed_nodes():
    rows, skills, _, _ = cli_progress.rows_for_cell(str(PILOT / "reveal"))
    assert sum(skills.values()) == 198
    assert skills.get("reveal", 0) == 2  # was reported as 131
    assert all(r["calls"] == 100 for r in rows)
    assert all(r["calls_source"] == "metrics.total_tool_calls" for r in rows)


@_needs_pilot
def test_actions_used_survives_the_v0_legacy_path_without_skill_calls():
    """`metrics.skill_calls` is absent in every pilot trace. `actions_used`
    must fall through the explicit chain rather than raise or report None."""
    trace = json.loads((PILOT / "reveal/traces.jsonl").read_text().splitlines()[0])
    assert "skill_calls" not in trace["metrics"]
    assert cli_agg.actions_used(trace, "claude_code") == 100
    assert cli_agg.actions_used(trace, "control") == 100
    assert cli_agg.actions_used(trace, "some_new_cell") == 100


def test_actions_used_still_prefers_the_arms_own_metric():
    trace = {"metrics": {"total_tool_calls": 87.0, "skill_calls": 20.0}}
    assert cli_agg.actions_used(trace, "control") == 87.0
    assert cli_agg.actions_used(trace, "claude_code") == 20.0
    assert cli_agg.actions_used({"metrics": {}}, "control") is None


@_needs_probe
def test_a_short_rollout_is_flagged_degenerate_and_never_silently_passed():
    """The probe emitted 30 calls, i.e. < 40: degenerate. Under
    `metrics.get('skill_calls', 999)` it passed as a healthy rollout."""
    rows, _ = _rows(PROBE_RUN)
    (row,) = rows
    assert row["degenerate"], row
    assert "30" in row["degenerate"][0][1]
    assert row["unknown_degeneracy"] == []


# ---------------------------------------------------------------------------
# BUG 5: model-aware pricing
# ---------------------------------------------------------------------------


@_needs_probe
def test_a_non_glm_5_2_cell_is_priced_with_its_OWN_table():
    """The defect was a `z-ai/glm-4.7-flash` run reported at GLM 5.2 rates.

    That model is now in PRICE_TABLES with the provider's own numbers, so the
    assertion moved from "reports unavailable" to "reports ITS price": the row
    must be priced strictly below what the GLM 5.2 table would have produced
    (flash is $0.10/$0.43 against $1.68/$5.28).
    """
    rows, _warnings = _rows(PROBE_RUN)
    (row,) = rows
    assert row["model"] == "z-ai/glm-4.7-flash"
    assert row["cost_priced_model"] == "z-ai/glm-4.7-flash"
    assert row["cost_mean"] is not None
    glm52 = cli_agg.PRICE_TABLES["z-ai/glm-5.2"]
    flash = cli_agg.PRICE_TABLES["z-ai/glm-4.7-flash"]
    assert flash["input_per_million"] < glm52["input_per_million"]


def test_a_model_with_no_table_at_all_reports_cost_unavailable(tmp_path):
    """The invariant the test above used to carry: a model nobody has a price
    for is reported as unavailable, NEVER priced with a neighbour's table."""
    cell = tmp_path / "cell"
    (cell / "turns").mkdir(parents=True)
    (cell / "config.toml").write_text('model = "acme/not-a-real-model"\n')
    (cell / "turns" / "0_100_1.ndjson").write_text(
        json.dumps({"turn": 1, "dlvl": 1, "hp": 10, "status": {"time": 1}}) + "\n"
    )
    (cell / "traces.jsonl").write_text(
        json.dumps({
            "task": {"data": {"idx": 0}},
            "stop_condition": "game_over",
            "calls": [{"usage": {"prompt_tokens": 10, "completion_tokens": 1}}],
        }) + "\n"
    )
    rows, warnings = _rows(tmp_path)
    (row,) = rows
    assert row["cost_mean"] is None
    assert row["cost_priced_model"] is None
    assert any("not in PRICE_TABLES" in w for w in warnings)
    assert "unavailable (no price table for acme/not-a-real-model)" in cli_agg.to_markdown(rows)


@_needs_pilot
def test_a_priced_model_still_reports_a_real_cost():
    rows, _ = _rows(PILOT)
    for row in rows:
        assert row["model"] == "z-ai/glm-5.2"
        assert row["cost_mean"] is not None and row["cost_mean"] > 0


def test_rollout_cost_refuses_to_price_without_a_table():
    trace = {"calls": [{"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}]}
    assert cli_agg.rollout_cost(trace, None) is None
    assert cli_agg.rollout_cost(trace, cli_agg.PRICE_TABLE_GLM_5_2) is not None


# ---------------------------------------------------------------------------
# BUG 7 + 8: both BALROG numbers and the pace columns in every table
# ---------------------------------------------------------------------------


@_needs_pilot
def test_the_cli_table_carries_both_balrog_numbers_and_the_slope_columns():
    rows, _ = _rows(PILOT)
    md = cli_agg.to_markdown(rows)
    for header in (
        "BALROG % max",
        "BALROG % min",
        "xp-carried",
        "dlvl / game-turn",
        "dlvl / LLM call",
        "BALROG% / game-turn",
        "BALROG% / LLM call",
    ):
        assert header in md, header
    for row in rows:
        for key in (
            "balrog_pct_mean",
            "balrog_min_pct_mean",
            "xp_carried_n",
            "depth_per_game_turn_mean",
            "depth_per_llm_call_mean",
            "balrog_pct_per_game_turn_mean",
            "balrog_pct_per_llm_call_mean",
        ):
            assert row[key] is not None, key


@_needs_pilot
def test_the_pilot_reveal_cell_reports_its_xp_carried_rollout():
    rows, _ = _rows(PILOT)
    fog, reveal = rows
    assert reveal["xp_carried_n"] == 1  # seed 1: Dlvl 4 at XP 1 -> 2.12 / 0.00
    assert fog["xp_carried_n"] == 0
    assert reveal["balrog_min_pct_mean"] < reveal["balrog_pct_mean"]


@_needs_pilot
def test_the_pilot_pace_signal_reveal_descends_faster_per_game_turn():
    """Reveal reached deeper in far fewer GAME turns than fog even though both
    cells are capped at the same number of LLM calls -- which is invisible in a
    depth-only table, and is the reason the slope columns exist."""
    rows, _ = _rows(PILOT)
    fog, reveal = rows
    assert fog["depth_mean"] == reveal["depth_mean"] == 4.5
    assert reveal["game_turns_mean"] < fog["game_turns_mean"]
    assert reveal["depth_per_game_turn_mean"] > fog["depth_per_game_turn_mean"]


# ---------------------------------------------------------------------------
# BUG 6: encoding_eval scores with the real BALROG metric and reads directories
# ---------------------------------------------------------------------------


def test_encoding_aggregate_scores_with_balrog_not_the_deprecated_proxy():
    """`progression_score` is the DEPRECATED analytic proxy `(DL/50)^1.3 *
    (XL/30)^0.6`; its own docstring says "do not quote it as BALROG" and it
    reads ~2x off. At DL=10 / XL=6 the real table gives 12.56% and the proxy
    6.42% -- this module used to emit the second while claiming the first."""
    from nethack_harness.prompt.balrog import balrog_progress, progression_score

    samples = [{"seed": 0, "max_dlvl": 10, "xp_level": 6, "reward": 1.0, "num_turns": 10}]
    row = enc_agg.aggregate_cells({"B0": samples})["rows"]["B0"]
    assert row["balrog_pct_mean"] == pytest.approx(100 * balrog_progress(10, 6))
    assert row["balrog_pct_mean"] == pytest.approx(12.56, abs=0.01)
    assert row["balrog_pct_mean"] != pytest.approx(100 * progression_score(10, 6), abs=0.01)
    assert "progression_score" not in row and "progression_tier" not in row


def test_encoding_aggregate_cells_emits_both_balrog_numbers():
    samples = [
        {"seed": 0, "max_dlvl": 4, "xp_level": 1, "reward": 1.0, "num_turns": 50},
        {"seed": 1, "max_dlvl": 5, "xp_level": 3, "reward": 2.0, "num_turns": 50},
    ]
    table = enc_agg.aggregate_cells({"B0": samples})
    row = table["rows"]["B0"]
    assert row["n"] == 2
    assert row["balrog_pct_mean"] > row["balrog_min_pct_mean"]
    assert row["xp_carried_n"] == 1  # the Dlvl 4 / XP 1 rollout
    assert row["depth_per_llm_call_mean"] is not None
    assert "progression_score" not in row
    md = enc_agg.table_to_markdown(table)
    assert "BALROG % max" in md and "BALROG % min" in md


@_needs_pilot
def test_encoding_aggregate_has_a_working_path_entry_point():
    """It previously had NO file I/O at all -- `aggregate_cells` took in-memory
    dicts, so it could not be run over a completed sweep."""
    warnings: list[str] = []
    rows = enc_agg.aggregate_run_dir(str(PILOT), warn=warnings.append)
    assert [r["cell"] for r in rows] == ["fog", "reveal"]
    assert all(r["n"] == 2 for r in rows)
    fog, reveal = rows
    assert reveal["xp_carried_n"] == 1
    assert reveal["depth_per_game_turn_mean"] > fog["depth_per_game_turn_mean"]
    md = enc_agg.table_to_markdown(rows)
    assert "BALROG % min" in md and "dlvl / game-turn" in md


def test_encoding_aggregate_reads_the_older_trace_subdirectory_name(tmp_path):
    cell = tmp_path / "B0" / "trace"
    cell.mkdir(parents=True)
    (cell / "0_1_1.ndjson").write_text(
        "\n".join(
            json.dumps(
                {"max_dlvl_reached": d, "status": {"depth": d, "experience_level": 1, "time": 10 * d}}
            )
            for d in (1, 2, 3)
        )
        + "\n"
    )
    rows = enc_agg.aggregate_run_dir(str(tmp_path), warn=lambda _m: None)
    assert len(rows) == 1
    assert rows[0]["max_dlvl"] == 3
    assert rows[0]["depth_per_game_turn_mean"] == pytest.approx(2 / 30)


@pytest.mark.skipif(
    not (REPO / "outputs/encoding_eval/calib_flash_b0_c3_partial/trace").is_dir(),
    reason="calib_flash_b0_c3_partial not present",
)
def test_the_real_retried_encoding_cell_reports_five_seeds_not_six_files(tmp_path):
    """`outputs/encoding_eval/calib_flash_b0_c3_partial/trace` genuinely holds
    6 attempt files for 5 seeds (seed 1 ran twice, under two different PIDs) --
    the exact shape the runbook warns about."""
    src = REPO / "outputs/encoding_eval/calib_flash_b0_c3_partial"
    assert len(list((src / "trace").glob("*.ndjson"))) == 6
    (tmp_path / "run").mkdir()
    shutil.copytree(src, tmp_path / "run" / "c3", ignore=shutil.ignore_patterns("*.log"))
    warnings: list[str] = []
    rows = enc_agg.aggregate_run_dir(str(tmp_path / "run"), warn=warnings.append)
    assert rows[0]["n"] == 5
    assert sum("superseded retry" in w for w in warnings) == 1


@pytest.mark.skipif(
    not (REPO / "outputs/encoding_eval/calib_flash_b0_c3_partial/trace").is_dir(),
    reason="calib_flash_b0_c3_partial not present",
)
def test_aggregate_run_py_also_joins_on_the_pid_and_reports_both_balrog_numbers(tmp_path):
    """`tools/encoding_eval/aggregate_run.py` is the third aggregator and had
    the same two defects: it globbed `trace/*.ndjson` (n=6 for 5 seeds) and
    quoted BALROG's max alone."""
    from tools.encoding_eval import aggregate_run as agg_run

    shutil.copytree(
        REPO / "outputs/encoding_eval/calib_flash_b0_c3_partial",
        tmp_path / "B0",
        ignore=shutil.ignore_patterns("*.log"),
    )
    table = agg_run.aggregate(str(tmp_path))
    (row,) = table
    assert row["n"] == 5, "6 attempt files, 5 seeds"
    assert row["balrog_min_pct_mean"] is not None
    assert row["xp_carried_n"] == 2
    md = agg_run.to_markdown(table)
    assert "BALROG % min" in md and "xp-carried" in md


def test_read_ndjson_skips_a_half_written_final_line(tmp_path):
    p = tmp_path / "x.ndjson"
    p.write_text(json.dumps({"turn": 1}) + "\n" + '{"turn": 2, "stat')
    assert read_ndjson(p) == [{"turn": 1}]
