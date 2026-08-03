"""Regression tests for `tools/eval_metrics.py`, the shared scoring layer.

Every test here is driven by a REAL trace committed to / produced in this repo
(`outputs/pilot_reveal`, `outputs/trace_probe`,
`tools/cli_harness_eval/acceptance`), because each bug it pins was a
*measured* wrong number in a published table, not a hypothetical:

  - the ~15x skill-call over-count from treating `traces.jsonl`'s cumulative
    prefix-replay `nodes` as a list of calls;
  - `skill_calls` not existing at all on the v0-legacy path, so the degeneracy
    rule ran off a KeyError or a silent `999` sentinel;
  - `glob('turns/*.ndjson')` counting one row per retry ATTEMPT instead of one
    per seed;
  - a GLM 5.2 price applied to a `z-ai/glm-4.7-flash` run;
  - BALROG quoted as its `max` alone, hiding a rollout whose entire score came
    from experience level.

Tests that need a rollout the repo does not contain are skipped, never faked.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.eval_metrics import (  # noqa: E402
    PRICE_TABLES,
    balrog_columns,
    degeneracy,
    executed_call_histogram,
    game_turns,
    model_for_cell,
    pace_columns,
    price_table_for,
    read_ndjson,
    select_turn_files,
    skill_call_count,
    turn_file_parts,
)

PROBE = REPO / "outputs/trace_probe/B0_reveal"
FOG = REPO / "outputs/pilot_reveal/fog"
REVEAL = REPO / "outputs/pilot_reveal/reveal"
ACCEPTANCE = REPO / "tools/cli_harness_eval/acceptance"

_needs_probe = pytest.mark.skipif(
    not (PROBE / "traces.jsonl").exists(), reason="outputs/trace_probe not present"
)
_needs_pilot = pytest.mark.skipif(
    not (REVEAL / "traces.jsonl").exists(), reason="outputs/pilot_reveal not present"
)


def _one_trace(cell: pathlib.Path, seed: int = 0) -> dict:
    for line in (cell / "traces.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        if ((trace.get("task") or {}).get("data") or {}).get("idx") == seed:
            return trace
    raise AssertionError(f"seed {seed} not in {cell}/traces.jsonl")


def _turn_rows(cell: pathlib.Path, seed: int) -> list[dict]:
    chosen, _ = select_turn_files(cell)
    return read_ndjson(chosen[seed])


# ---------------------------------------------------------------------------
# BUG 1: `nodes` is a cumulative prefix replay, not a list of calls
# ---------------------------------------------------------------------------


@_needs_probe
def test_nodes_really_is_a_cumulative_prefix_replay():
    """The premise of the ~15x bug, pinned so the fix cannot be argued away:
    the probe rollout emitted 30 calls but `nodes` holds 911 entries, 455 of
    them assistant messages carrying `tool_calls`."""
    trace = _one_trace(PROBE)
    nodes = trace["nodes"]
    assistant_with_calls = [
        n
        for n in nodes
        if (n.get("message") or {}).get("role") == "assistant"
        and (n.get("message") or {}).get("tool_calls")
    ]
    assert len(nodes) == 911
    assert len(assistant_with_calls) == 455
    assert sum(1 for n in nodes if n.get("sampled")) == 30


@_needs_probe
def test_executed_call_count_on_the_probe_is_29_not_455():
    """THE regression test for the 15x over-count.

    `outputs/trace_probe/B0_reveal` is a 30-turn probe. The turn NDJSON --
    authoritative for the control / v0-legacy arm, which is what this cell ran
    -- records 29 applied calls, 22 of them `np_move_to`. The old code derived
    its histogram from every assistant node and reported 455 skills with 292
    `np_move_to`."""
    trace = _one_trace(PROBE)
    rows = _turn_rows(PROBE, 0)
    hist, source = executed_call_histogram(trace=trace, turn_rows=rows)
    assert source == "turns.tool_calls"
    assert sum(hist.values()) == 29
    assert hist["np_move_to"] == 22
    assert sum(hist.values()) < 40  # the broken value was 455


@_needs_probe
def test_sampled_nodes_reproduce_the_engine_side_referee_counters_exactly():
    """The CLI arms have no `tool_calls` in their turn files, so they must read
    the trace `nodes` -- restricted to `sampled`. That path is validated here
    against `metrics`' independent per-skill counters: they agree to the call."""
    trace = _one_trace(PROBE)
    hist, source = executed_call_histogram(trace=trace, turn_rows=None)
    assert source == "trace.nodes[sampled]"
    metrics = trace["metrics"]
    assert sum(hist.values()) == metrics["total_tool_calls"] == 30
    for name, n in hist.items():
        assert n == metrics[f"{name}_calls"], name


@_needs_pilot
@pytest.mark.parametrize(
    "cell,seed,applied_reveal,emitted_reveal",
    [
        (FOG, 0, 7, 8),
        (FOG, 1, 3, 3),
        (REVEAL, 0, 1, 1),
        (REVEAL, 1, 1, 1),
    ],
)
def test_pilot_reveal_adoption_counts_are_per_call_not_per_replayed_node(
    cell, seed, applied_reveal, emitted_reveal
):
    """Reveal adoption is exactly what the sweep measures, so it gets pinned
    per seed. The old code reported 279 `reveal` out of 10,080 "skills" for the
    fog cell (really 10 out of 198) and 131 out of 10,080 for the reveal cell
    (really 2 out of 198).

    Both channels are pinned because they differ by exactly the one call each
    rollout emitted but never got to apply -- every pilot rollout stopped on
    `max_output_tokens`, so the referee counted 100 calls and the environment
    wrote 99 turn records. Neither number is 455 or 10,080."""
    trace = _one_trace(cell, seed)
    applied, applied_src = executed_call_histogram(
        trace=trace, turn_rows=_turn_rows(cell, seed)
    )
    emitted, emitted_src = executed_call_histogram(trace=trace, turn_rows=None)

    assert applied_src == "turns.tool_calls"
    assert sum(applied.values()) == 99
    assert applied.get("reveal", 0) == applied_reveal

    assert emitted_src == "trace.nodes[sampled]"
    assert sum(emitted.values()) == int(trace["metrics"]["total_tool_calls"]) == 100
    assert emitted.get("reveal", 0) == emitted_reveal == int(trace["metrics"]["reveal_calls"])


def test_prime_agent_skills_nested_in_ipython_are_unwrapped():
    """Counting the outer tool name reports 100% `ipython` / 0% everything
    else. The code payload is parsed instead."""
    trace = {
        "nodes": [
            {
                "sampled": True,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": "ipython",
                            "arguments": json.dumps(
                                {"code": "await nethack.np_move_to(3, 4)\nawait nethack.reveal()"}
                            ),
                        }
                    ],
                },
            }
        ]
    }
    hist, source = executed_call_histogram(trace=trace)
    assert source == "trace.nodes[sampled]"
    assert hist["np_move_to"] == 1
    assert hist["reveal"] == 1
    assert "ipython" not in hist


def test_unsampled_replayed_nodes_are_never_counted():
    """Synthetic minimal reproduction of the prefix replay: the same assistant
    turn appearing three times must count once."""
    call = {"name": "np_move_to", "arguments": "{}"}
    node = lambda sampled: {  # noqa: E731
        "sampled": sampled,
        "message": {"role": "assistant", "tool_calls": [call]},
    }
    hist, _ = executed_call_histogram(trace={"nodes": [node(True), node(False), node(False)]})
    assert hist["np_move_to"] == 1


# ---------------------------------------------------------------------------
# BUG 2: `skill_calls` does not exist on the v0-legacy path
# ---------------------------------------------------------------------------


@_needs_pilot
def test_the_pilot_traces_really_lack_skill_calls():
    """`eval.log` says `running 2x1 v0 rollouts ... (legacy: nethack)`, so
    `NetHackTask.finalize` never ran. A literal `metrics['skill_calls']` here
    raises KeyError; `.get('skill_calls', 999)` marks everything non-degenerate."""
    metrics = _one_trace(REVEAL, 0)["metrics"]
    assert "skill_calls" not in metrics
    assert "max_dlvl_reached" not in metrics
    assert "died" not in metrics
    with pytest.raises(KeyError):
        metrics["skill_calls"]


@_needs_pilot
def test_skill_call_count_falls_back_explicitly_and_names_its_source():
    trace = _one_trace(REVEAL, 0)
    calls, source = skill_call_count(trace=trace, turn_rows=_turn_rows(REVEAL, 0))
    assert calls == 100
    assert source == "metrics.total_tool_calls"


def test_skill_call_count_prefers_skill_calls_then_total_then_the_histogram():
    assert skill_call_count(trace={"metrics": {"skill_calls": 7, "total_tool_calls": 99}}) == (
        7,
        "metrics.skill_calls",
    )
    assert skill_call_count(trace={"metrics": {"total_tool_calls": 99}}) == (
        99,
        "metrics.total_tool_calls",
    )
    assert skill_call_count(trace={"metrics": {"np_move_to_calls": 4, "reveal_calls": 1}}) == (
        5,
        "trace.metrics.*_calls",
    )


def test_skill_call_count_is_none_not_a_sentinel_when_nothing_is_measurable():
    assert skill_call_count(trace={"metrics": {}}, turn_rows=[]) == (None, "unavailable")


def test_degeneracy_rule():
    assert degeneracy("error", 500)[0] is True
    assert degeneracy("max_output_tokens", 39)[0] is True
    assert degeneracy("max_output_tokens", 40)[0] is False
    assert degeneracy("agent_completed", 79)[0] is True
    assert degeneracy("agent_completed", 80)[0] is False


def test_unknown_skill_calls_is_unknown_degeneracy_not_a_pass():
    """`metrics.get('skill_calls', 999)` silently marked every v0-legacy
    rollout non-degenerate. Unmeasurable must be its own answer."""
    verdict, reason = degeneracy("max_output_tokens", None)
    assert verdict is None
    assert "unavailable" in reason


# ---------------------------------------------------------------------------
# BUG 4: one row per seed -- the PID join over <seed>_<pid>_<epoch>.ndjson
# ---------------------------------------------------------------------------


def test_turn_file_parts_parses_the_documented_filename_shape():
    assert turn_file_parts("turns/12_33959_1785556711.ndjson") == (12, 33959, 1785556711)
    assert turn_file_parts("task13_claude_code_seed0.turns.ndjson") is None


@_needs_pilot
def test_select_turn_files_is_one_row_per_seed_on_the_real_pilot_cell():
    chosen, dropped = select_turn_files(REVEAL)
    assert sorted(chosen) == [0, 1]
    assert dropped == []


@_needs_pilot
def test_a_simulated_retry_does_not_double_count_the_seed(tmp_path):
    """The documented trap, reproduced on real data: a rollout that retried
    writes a NEW `<seed>_<pid>_<epoch>.ndjson` per attempt, so
    `glob('turns/*.ndjson')` reports one row per ATTEMPT. Here the real reveal
    cell is copied and seed 0 is given two extra attempts -- a later stub from
    the same eval process (same PID) and a full relaunch under a different PID.
    Globbing would say n=4; the join must still say n=2, and must keep the
    longest attempt so a stub can never outvote the real rollout."""
    cell = tmp_path / "reveal"
    shutil.copytree(REVEAL, cell, ignore=shutil.ignore_patterns("*.log"))
    turns = cell / "turns"
    real = next(turns.glob("0_*.ndjson"))
    seed, pid, ts = turn_file_parts(real)
    real_rows = len(read_ndjson(real))

    stub = turns / f"0_{pid}_{ts + 60}.ndjson"  # in-process retry, same PID
    stub.write_text("\n".join(real.read_text().splitlines()[:2]) + "\n")
    relaunch = turns / f"0_{pid + 7}_{ts + 900}.ndjson"  # relaunch, new PID
    relaunch.write_text("\n".join(real.read_text().splitlines()[:5]) + "\n")

    assert len(list(turns.glob("*.ndjson"))) == 4  # what the old glob counted

    chosen, dropped = select_turn_files(cell)
    assert sorted(chosen) == [0, 1], "one row per seed, never one per attempt"
    assert pathlib.Path(chosen[0]) == real
    assert len(read_ndjson(chosen[0])) == real_rows
    assert len(dropped) == 2
    assert all("superseded retry" in reason for _, reason in dropped)


@_needs_pilot
def test_a_stale_seed_not_in_traces_jsonl_cannot_inflate_n(tmp_path):
    """"A stale directory silently inflates n" -- `traces.jsonl` is the
    authority for which seeds the run actually scheduled."""
    cell = tmp_path / "reveal"
    shutil.copytree(REVEAL, cell, ignore=shutil.ignore_patterns("*.log"))
    turns = cell / "turns"
    real = next(turns.glob("0_*.ndjson"))
    (turns / "9_11111_1700000000.ndjson").write_text(real.read_text())

    chosen, dropped = select_turn_files(cell, seeds={0, 1})
    assert sorted(chosen) == [0, 1]
    assert any("stale directory" in reason for _, reason in dropped)


def test_an_empty_attempt_file_is_reported_not_silently_selected(tmp_path):
    turns = tmp_path / "turns"
    turns.mkdir()
    (turns / "0_100_1.ndjson").write_text("")
    (turns / "0_100_2.ndjson").write_text(json.dumps({"turn": 1}) + "\n")
    chosen, dropped = select_turn_files(tmp_path)
    assert sorted(chosen) == [0]
    assert any("no turn rows" in reason for _, reason in dropped)


# ---------------------------------------------------------------------------
# BUG 5: model-aware pricing -- never silently misprice
# ---------------------------------------------------------------------------


@_needs_probe
def test_the_probe_cell_ran_a_model_that_is_not_glm_5_2():
    """The measured mispricing: this run was `z-ai/glm-4.7-flash` and was
    reported at GLM 5.2 rates.

    `glm-4.7-flash` now HAS a table of its own (the provider's, via
    `refresh_price_tables`), so the guard is no longer "it is unpriced" -- it
    is "it is priced as ITSELF". That is the property that actually mattered:
    the defect was one model's rates being applied to another's run, and a
    table that happens to be missing is only an accidental way to avoid it.
    """
    assert model_for_cell(PROBE) == "z-ai/glm-4.7-flash"
    probe_price = price_table_for("z-ai/glm-4.7-flash")
    assert probe_price is not None
    assert probe_price is not PRICE_TABLES["z-ai/glm-5.2"]
    assert probe_price["input_per_million"] != PRICE_TABLES["z-ai/glm-5.2"]["input_per_million"]


@_needs_pilot
def test_model_is_read_from_the_cells_own_config():
    assert model_for_cell(REVEAL) == "z-ai/glm-5.2"
    assert price_table_for(model_for_cell(REVEAL)) is PRICE_TABLES["z-ai/glm-5.2"]


def test_an_unknown_or_missing_model_has_no_price_table(tmp_path):
    assert price_table_for(None) is None
    assert price_table_for("openai/gpt-9") is None
    assert model_for_cell(tmp_path) is None


def test_model_falls_back_to_the_eval_log_banner(tmp_path):
    (tmp_path / "eval.log").write_text(
        "03:58:34    INFO results: /x\n"
        "03:58:34    INFO running 2x1 v0 rollouts on z-ai/glm-5.2 (legacy: nethack)\n"
    )
    assert model_for_cell(tmp_path) == "z-ai/glm-5.2"


# ---------------------------------------------------------------------------
# BUG 7: both BALROG numbers, always
# ---------------------------------------------------------------------------


def test_balrog_columns_carry_max_min_and_the_xp_carried_flag():
    """The real pilot example: reveal seed 1 reached Dlvl 4 at XP 1 -- 2.12% on
    BALROG's published max, 0.00% on the min. The `xp_carried` flag is exactly
    `max > 0 and min == 0`: one axis alone is holding up the headline score, so
    the max is not a summary of the rollout."""
    cols = balrog_columns(4, 1)
    assert round(cols["balrog_pct"], 2) == 2.12
    assert cols["balrog_min_pct"] == 0.0
    assert cols["xp_carried"] is True


def test_balrog_min_is_not_the_max_when_both_axes_advanced():
    cols = balrog_columns(5, 3)
    assert cols["balrog_pct"] > cols["balrog_min_pct"] > 0.0
    assert cols["xp_carried"] is False


@_needs_pilot
def test_pilot_reveal_seed1_is_the_xp_carried_case_end_to_end():
    rows = _turn_rows(REVEAL, 1)
    dlvl = max(r.get("max_dlvl_reached") or 1 for r in rows)
    xp = max((r.get("status") or {}).get("experience_level") or 1 for r in rows)
    assert (dlvl, xp) == (4, 1)
    cols = balrog_columns(dlvl, xp)
    assert round(cols["balrog_pct"], 2) == 2.12
    assert cols["balrog_min_pct"] == 0.0
    assert cols["xp_carried"] is True


# ---------------------------------------------------------------------------
# BUG 8: progression slope / pace
# ---------------------------------------------------------------------------


@_needs_pilot
def test_game_turns_is_status_time_not_the_number_of_llm_calls():
    """One LLM call runs a whole pathfinding macro, so the two denominators are
    an order of magnitude apart: reveal seed 0 spans 388 GAME turns over 99
    recorded calls."""
    rows = _turn_rows(REVEAL, 0)
    assert game_turns(rows) == 388
    assert len(rows) == 99


@_needs_pilot
def test_reveal_descends_faster_per_game_turn_than_fog_on_the_pilot():
    """The pilot's headline pace signal: reveal seed 0 reached Dlvl 5 in 388
    game turns, fog seed 0 reached Dlvl 3 in 936."""
    def _pace(cell, seed):
        rows = _turn_rows(cell, seed)
        dlvl = max(r.get("max_dlvl_reached") or 1 for r in rows)
        return pace_columns(dlvl, None, game_turns(rows), len(rows)), dlvl

    rev, rev_dlvl = _pace(REVEAL, 0)
    fog, fog_dlvl = _pace(FOG, 0)
    assert (rev_dlvl, rev["game_turns"]) == (5, 388)
    assert (fog_dlvl, fog["game_turns"]) == (3, 936)
    assert rev["depth_per_game_turn"] == pytest.approx(4 / 388)
    assert fog["depth_per_game_turn"] == pytest.approx(2 / 936)
    assert rev["depth_per_game_turn"] > 4 * fog["depth_per_game_turn"]


def test_pace_uses_depth_GAINED_so_a_rollout_that_never_moved_scores_zero():
    p = pace_columns(1, 0.0, 500, 100)
    assert p["depth_gained"] == 0
    assert p["depth_per_game_turn"] == 0.0  # measured zero, not "unmeasurable"
    assert p["depth_per_llm_call"] == 0.0


def test_pace_is_unavailable_not_infinite_on_a_zero_or_missing_denominator():
    p = pace_columns(5, 2.65, 0, 0)
    assert p["depth_per_game_turn"] is None
    assert p["depth_per_llm_call"] is None
    p = pace_columns(5, 2.65, None, None)
    assert p["balrog_pct_per_game_turn"] is None
    assert p["balrog_pct_per_llm_call"] is None


def test_pace_is_unavailable_when_the_numerator_is_unknown():
    p = pace_columns(5, None, 100, 50)
    assert p["depth_per_game_turn"] is not None
    assert p["balrog_pct_per_game_turn"] is None
