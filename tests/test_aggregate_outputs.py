"""Two aggregator-infrastructure defects, both of which silently lost a result.

DEFECT 1 -- one filename, two aggregators.
`tools/cli_harness_eval/aggregate.py` and `tools/encoding_eval/aggregate.py`
(and `tools/encoding_eval/aggregate_run.py`) each wrote `<run_dir>/table.md` and
`<run_dir>/table.json`. They compute DIFFERENT tables from the same directory
layout, so running both over one run directory replaced the first result with
the second, with nothing printed and nothing to tell the two files apart
afterwards. `tools/cli_harness_eval/run_sweep.sh` then made it worse by ALSO
teeing the aggregator's stdout into the same `table.md` that `main()` writes --
a second writer of the same path in the same invocation, and the worse of the
two, because a pipe carries stdout only and the aggregator's warnings go to
stderr.

DEFECT 2 -- `sec/call` on a concurrently-run cell.
The column is a mean of consecutive `t_mono` deltas. `t_mono` is
`time.monotonic()`, which is PROCESS-wide, and the eval CLI runs a cell's seeds
concurrently in one process, so those deltas contain the co-tenant rollouts'
work as well. The pilot cell printed `1.3s -> 11.5s` for two rollouts whose
durations differed by 7.6x in one interpreter. A wrong number that looks right
is worse than an honest "unavailable".
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.cli_harness_eval import aggregate as cli_aggregate  # noqa: E402
from tools.encoding_eval import aggregate as enc_aggregate  # noqa: E402
from tools.eval_metrics import (  # noqa: E402
    concurrent_rollout_files,
    rollout_clock_span,
    table_paths,
    write_table,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# defect 1: the shared filename                                                #
# --------------------------------------------------------------------------- #
def test_each_producer_owns_a_filename_no_other_producer_writes(tmp_path):
    write_table(tmp_path, [{"cell": "a"}], "# cli table", producer="cli_harness_eval",
                warn=lambda _m: None)
    write_table(tmp_path, [{"cell": "b"}], "# encoding table", producer="encoding_eval",
                warn=lambda _m: None)

    cli = table_paths(tmp_path, "cli_harness_eval")
    enc = table_paths(tmp_path, "encoding_eval")
    # The second run did NOT destroy the first: both results are still on disk.
    assert "cli table" in open(cli["primary_md"]).read()
    assert "encoding table" in open(enc["primary_md"]).read()
    assert json.load(open(cli["primary_json"])) == [{"cell": "a"}]
    assert json.load(open(enc["primary_json"])) == [{"cell": "b"}]


def test_overwriting_the_canonical_table_is_loud_and_points_at_the_survivor(tmp_path):
    warnings = []
    write_table(tmp_path, [], "# cli", producer="cli_harness_eval", warn=warnings.append)
    assert warnings == [], "the first producer has nothing to displace"

    result = write_table(tmp_path, [], "# encoding", producer="encoding_eval",
                         warn=warnings.append)
    assert result["overwrote"] == "cli_harness_eval"
    assert len(warnings) == 1
    assert "table.cli_harness_eval.md" in warnings[0]
    assert "NOT lost" in warnings[0]


def test_the_documented_filenames_still_exist_and_say_who_wrote_them(tmp_path):
    """RUNBOOK Sec 6, `docs/experiments/*` and several shell scripts name
    `table.md` / `table.json`, so the convention is kept -- but the markdown now
    identifies its producer, and `table.json` keeps its exact old shape so
    `json.load(open("table.json"))` still returns what it always returned."""
    rows = [{"cell": "fog", "n": 2}]
    write_table(tmp_path, rows, "| a |", producer="cli_harness_eval", warn=lambda _m: None)

    canonical_md = (tmp_path / "table.md").read_text()
    assert canonical_md.startswith("<!--")
    assert "cli_harness_eval" in canonical_md.splitlines()[0]
    assert "| a |" in canonical_md
    assert json.load(open(tmp_path / "table.json")) == rows

    provenance = json.load(open(tmp_path / "table.provenance.json"))
    assert provenance["producer"] == "cli_harness_eval"
    assert provenance["primary_md"] == "table.cli_harness_eval.md"


def test_both_aggregators_declare_distinct_producer_names():
    assert cli_aggregate.PRODUCER != enc_aggregate.PRODUCER


def test_run_sweep_no_longer_double_writes_the_table():
    """The shell must not tee into a file `main()` also writes, and must keep
    the aggregator's own exit status instead of tee's."""
    script = open(os.path.join(REPO, "tools/cli_harness_eval/run_sweep.sh")).read()
    # Comments still name the old command (that is where the finding is
    # recorded); only executable lines are the contract.
    code = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )
    assert "table.md" not in code
    assert "aggregate.log" in script
    assert "PIPESTATUS" in script
    # stderr is captured now: the warnings about ignored retry files and
    # unpriced models used to be dropped on the floor by the stdout-only pipe.
    assert "2>&1" in script


# --------------------------------------------------------------------------- #
# defect 2: sec/call on concurrent rollouts                                    #
# --------------------------------------------------------------------------- #
def _turns(start, step, n=10):
    return [{"t_mono": start + i * step, "status": {"hitpoints": 10}} for i in range(n)]


def test_two_rollouts_sharing_a_process_and_a_time_window_are_detected():
    overlaps = concurrent_rollout_files([
        ("turns/0_500_1700.ndjson", _turns(100.0, 1.0)),   # 100 .. 109
        ("turns/1_500_1700.ndjson", _turns(105.0, 1.0)),   # 105 .. 114
    ])
    assert set(overlaps) == {"turns/0_500_1700.ndjson", "turns/1_500_1700.ndjson"}


def test_sequential_rollouts_in_one_process_are_not_flagged():
    """Same interpreter, but one finished before the other began -- its deltas
    are its own, so the measurement stands."""
    overlaps = concurrent_rollout_files([
        ("turns/0_500_1700.ndjson", _turns(100.0, 1.0)),   # 100 .. 109
        ("turns/1_500_1700.ndjson", _turns(200.0, 1.0)),   # 200 .. 209
    ])
    assert overlaps == {}


def test_rollouts_in_different_processes_are_never_compared():
    """`t_mono` values from two processes are not on the same clock at all, so
    an apparent overlap between them means nothing."""
    overlaps = concurrent_rollout_files([
        ("turns/0_500_1700.ndjson", _turns(100.0, 1.0)),
        ("turns/1_600_1700.ndjson", _turns(100.0, 1.0)),
    ])
    assert overlaps == {}


def test_a_rollout_with_one_timestamp_has_no_span():
    assert rollout_clock_span([{"t_mono": 1.0}]) is None
    assert rollout_clock_span([]) is None
    assert rollout_clock_span([{"t_mono": 1.0}, {"t_mono": 5.0}]) == (1.0, 5.0, "t_mono")


def _cell(tmp_path, files):
    cell = tmp_path / "cell"
    (cell / "turns").mkdir(parents=True)
    for name, rows in files.items():
        (cell / "turns" / name).write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
    return cell


def test_a_concurrent_cell_reports_sec_per_call_as_unavailable(tmp_path):
    cell = _cell(tmp_path, {
        "0_500_1700.ndjson": _turns(100.0, 1.0),
        "1_500_1700.ndjson": _turns(105.0, 9.0),
    })
    paths = sorted(str(p) for p in (cell / "turns").glob("*.ndjson"))
    row = cli_aggregate.aggregate_arm("cell", [], paths)

    assert row["sec_per_call_first_half_mean"] is None
    assert row["sec_per_call_second_half_mean"] is None
    assert row["sec_per_call_n_available"] == 0
    assert len(row["sec_per_call_unavailable"]) == 2
    for _path, why in row["sec_per_call_unavailable"]:
        assert "process-wide" in why and "pid 500" in why

    md = cli_aggregate.to_markdown([row])
    assert "unavailable (2 concurrent rollouts/process)" in md
    # The wrong-looking number must not appear anywhere in the table.
    assert "s → " not in md.split("Notes:")[0]


def test_a_sequential_cell_still_reports_a_real_number(tmp_path):
    """The fix must not throw the measurement away when it IS measurable."""
    cell = _cell(tmp_path, {
        "0_500_1700.ndjson": _turns(100.0, 2.0),
        "1_501_1800.ndjson": _turns(100.0, 2.0),
    })
    paths = sorted(str(p) for p in (cell / "turns").glob("*.ndjson"))
    row = cli_aggregate.aggregate_arm("cell", [], paths)

    assert row["sec_per_call_unavailable"] == []
    assert row["sec_per_call_first_half_mean"] == 2.0
    assert row["sec_per_call_second_half_mean"] == 2.0
    assert "2.0s → 2.0s" in cli_aggregate.to_markdown([row])


def test_a_partly_concurrent_cell_says_how_many_rollouts_it_measured(tmp_path):
    cell = _cell(tmp_path, {
        "0_500_1700.ndjson": _turns(100.0, 1.0),    # shares pid 500 with seed 1
        "1_500_1700.ndjson": _turns(105.0, 1.0),
        "2_777_1900.ndjson": _turns(100.0, 3.0),    # alone in its process
    })
    paths = sorted(str(p) for p in (cell / "turns").glob("*.ndjson"))
    row = cli_aggregate.aggregate_arm("cell", [], paths)

    assert row["sec_per_call_n_available"] == 1
    assert row["sec_per_call_first_half_mean"] == 3.0
    assert "(1/3 rollouts)" in cli_aggregate.to_markdown([row])


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #
def test_running_both_aggregators_over_one_directory_keeps_both_results(tmp_path):
    run_dir = tmp_path / "run"
    cell = _cell(run_dir, {"0_500_1700.ndjson": _turns(100.0, 1.0)})
    (cell / "config.toml").write_text('model = "z-ai/glm-5.2"\n')

    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(
               [REPO, os.path.join(REPO, "environments", "nethack"),
                os.environ.get("PYTHONPATH", "")])}
    for module in ("tools.cli_harness_eval.aggregate", "tools.encoding_eval.aggregate"):
        out = subprocess.run([sys.executable, "-m", module, str(run_dir)],
                             cwd=REPO, capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr

    names = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert {"table.md", "table.json", "table.cli_harness_eval.md",
            "table.cli_harness_eval.json", "table.encoding_eval.md",
            "table.encoding_eval.json", "table.provenance.json"} <= names
    # The CLI table's own columns survived the encoding aggregator's run.
    assert "Post-death drain" in (run_dir / "table.cli_harness_eval.md").read_text()
    assert "tokens/turn" in (run_dir / "table.encoding_eval.md").read_text()
