"""tools/stall_watchdog.py — the out-of-process mitigation for the two hangs in
`docs/HARNESS_DEFECTS.md` §3.1 that `no_progress_timeout` can never catch.

Every test here drives REAL processes writing REAL `<seed>_<pid>_<ts>.ndjson`
files. A watchdog that has only been unit-tested against fabricated state is not
evidence of anything: the whole mechanism is "notice that a process stopped
writing and kill it", and both halves of that are OS behaviour.

The two headline cases are `test_kills_quarantines_and_logs_a_stalled_rollout`
(drives the real CLI end to end) and
`test_does_not_kill_a_rollout_that_is_still_writing`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WATCHDOG = REPO / "tools" / "stall_watchdog.py"

sys.path.insert(0, str(REPO / "tools"))
import stall_watchdog as sw  # noqa: E402


# A stand-in rollout: writes `n` turn records `interval` apart into a correctly
# named turn file, then sleeps forever. `n=2, interval=0.05` reproduces the
# §3.1 hang shape (a few normal turns, then a wedged process that never returns
# and never reaches env.step); a large `n` reproduces a healthy rollout.
FAKE_ROLLOUT = r'''
import json, os, sys, time
turns, seed, n, interval = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
path = os.path.join(turns, "%s_%d_%d.ndjson" % (seed, os.getpid(), int(time.time())))
with open(path, "a") as fh:
    for i in range(n):
        fh.write(json.dumps({
            "turn": i + 1, "dlvl": 1, "hp": 16, "max_hp": 16, "variant": "B0",
            "t_wall": time.time(), "tool_calls": [{"name": "np_explore_level"}],
            "rendered_user_message": "=== STATUS ===\nHP: 16/16  Turn: %d" % (i + 1),
        }) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        time.sleep(interval)
time.sleep(3600)
'''


@pytest.fixture
def turns(tmp_path) -> Path:
    d = tmp_path / "cell" / "turns"
    d.mkdir(parents=True)
    return d


def _spawn(turns: Path, seed: int, n: int, interval: float) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", FAKE_ROLLOUT,
                          str(turns), str(seed), str(n), str(interval)])
    deadline = time.time() + 15
    while time.time() < deadline:
        if list(turns.glob(f"{seed}_{p.pid}_*.ndjson")):
            return p
        if p.poll() is not None:
            raise AssertionError("fake rollout exited before writing a turn file")
        time.sleep(0.05)
    raise AssertionError("fake rollout never wrote a turn file")


def _reap(procs):
    for p in procs:
        if p.poll() is None:
            p.kill()
        try:
            p.wait(timeout=5)
        except Exception:
            pass


def _wd(turns, **kw):
    kw.setdefault("timeout", 1.0)
    kw.setdefault("poll", 0.2)
    kw.setdefault("kill_grace", 1.0)
    return sw.Watchdog([str(turns)], **kw)


# --------------------------------------------------------------------------
# filename contract
# --------------------------------------------------------------------------
def test_parses_the_run_id_filename_the_harness_actually_writes():
    """`helpers.py` builds `f"{seeds[0]}_{os.getpid()}_{int(time.time())}"`.
    The PID is right there in the name — that is what makes an out-of-process
    watchdog possible at all, so the parse is pinned."""
    m = sw.RUN_ID_RE.match("13_652628_1785108044.ndjson")
    assert m and (m["seed"], m["pid"], m["ts"]) == ("13", "652628", "1785108044")
    for bad in ("traces.jsonl", "0_123.ndjson", "a_1_2.ndjson",
                "0_1_2.ndjson.bak", "0_1_2_3.ndjson"):
        assert sw.RUN_ID_RE.match(bad) is None, bad


# --------------------------------------------------------------------------
# THE KILL PATH — end to end through the real CLI
# --------------------------------------------------------------------------
def test_kills_quarantines_and_logs_a_stalled_rollout(turns):
    """A rollout that writes two turns and then wedges must be killed, its turn
    files moved out of `turns/`, and the kill recorded in enough detail to
    reconstruct what happened."""
    proc = _spawn(turns, seed=7, n=2, interval=0.05)
    try:
        fname = os.path.basename(list(turns.glob(f"7_{proc.pid}_*.ndjson"))[0])
        time.sleep(1.2)  # exceed the 1s timeout below

        rc = subprocess.run(
            [sys.executable, str(WATCHDOG), "--turns-dir", str(turns),
             "--timeout", "1", "--poll", "0.2", "--kill-grace", "1", "--once"],
            capture_output=True, text=True, timeout=60)

        assert rc.returncode == 3, f"expected exit 3 (killed something)\n{rc.stderr}"
        assert "STALL" in rc.stderr and f"pid={proc.pid}" in rc.stderr

        # 1. the process is dead
        proc.wait(timeout=10)
        assert proc.poll() is not None
        assert not sw.pid_alive(proc.pid)

        # 2. its turn file no longer sits in turns/, so a relaunch of seed 7
        #    cannot merge with it (HARNESS_DEFECTS §4.6)
        assert list(turns.glob("*.ndjson")) == []
        qroot = Path(str(turns) + ".stalled")
        moved = list(qroot.glob(f"*_pid{proc.pid}/{fname}"))
        assert len(moved) == 1, f"turn file not quarantined; found {list(qroot.rglob('*'))}"
        # and it is intact, not truncated by the move
        assert len(moved[0].read_text().strip().splitlines()) == 2

        # 3. the log reconstructs the incident
        log = qroot / "stall_watchdog.jsonl"
        recs = [json.loads(ln) for ln in log.read_text().splitlines()]
        kills = [r for r in recs if r["event"] == "kill"]
        assert len(kills) == 1
        k = kills[0]
        assert k["pid"] == proc.pid
        assert k["seeds"] == [7] and k["likely_hung_seed"] == 7
        assert k["idle_s"] >= 1.0 and k["timeout_s"] == 1.0
        assert k["kill"]["survivors"] == []
        assert fname in k["quarantined_files"]
        last = k["per_seed"]["7"]["last_turn"]
        assert last["turn"] == 2 and last["dlvl"] == 1 and last["hp"] == 16
        assert last["n_lines"] == 2
        assert last["last_tool_call"] == "np_explore_level"
        # the manifest next to the quarantined files says the same thing
        manifest = json.loads((moved[0].parent / "MANIFEST.json").read_text())
        assert manifest["pid"] == proc.pid and manifest["event"] == "kill"
    finally:
        _reap([proc])


# --------------------------------------------------------------------------
# THE NO-KILL PATH
# --------------------------------------------------------------------------
def test_does_not_kill_a_rollout_that_is_still_writing(turns):
    """The failure mode that would make this watchdog unusable is killing
    healthy work. A rollout appending a turn every 0.2s survives a watchdog with
    a 1.5s timeout running for several full poll cycles."""
    proc = _spawn(turns, seed=3, n=200, interval=0.2)
    try:
        wd = _wd(turns, timeout=1.5, poll=0.25)
        rc = wd.run(max_seconds=4.0)
        assert rc == 0, "watchdog killed a healthy rollout"
        assert wd.kills == []
        assert proc.poll() is None, "healthy rollout was killed"
        assert sw.pid_alive(proc.pid)
        assert len(list(turns.glob("*.ndjson"))) == 1, "healthy turn file was moved"
        assert list(Path(str(turns) + ".stalled").glob("*_pid*")) == [], \
            "healthy rollout was quarantined"
    finally:
        _reap([proc])


def test_a_finished_seed_does_not_arm_a_kill_while_the_process_still_writes(turns):
    """WHY the stall is measured per-PID and not per-seed.

    One eval process owns every seed of a cell. A seed that simply FINISHED
    stops writing forever, so a per-seed rule would kill the whole cell
    `timeout` seconds after its first seed completed — turning a successful run
    into a dead one. Here seed 0's file is long-stale while seed 1 is still
    writing under the same PID: no kill.
    """
    proc = _spawn(turns, seed=1, n=200, interval=0.2)
    try:
        finished = turns / f"0_{proc.pid}_{int(time.time())}.ndjson"
        finished.write_text(json.dumps({"turn": 412, "dlvl": 5, "hp": 11}) + "\n")
        old = time.time() - 3600
        os.utime(finished, (old, old))

        wd = _wd(turns, timeout=1.5, poll=0.25)
        assert wd.run(max_seconds=3.0) == 0
        assert wd.kills == []
        assert proc.poll() is None
        assert finished.exists(), "a finished seed's data must not be quarantined"
    finally:
        _reap([proc])


def test_leftover_files_from_an_exited_process_are_left_alone(turns):
    """Turn files whose PID is gone are a COMPLETED (or previously handled) run.
    They are final data for grading — never quarantine them, and never try to
    kill a dead PID."""
    proc = _spawn(turns, seed=2, n=1, interval=0.0)
    proc.kill()
    proc.wait(timeout=10)
    time.sleep(1.2)

    wd = _wd(turns, timeout=1.0)
    assert wd.check_dir(str(turns)) == []
    assert len(list(turns.glob("*.ndjson"))) == 1
    assert list(Path(str(turns) + ".stalled").glob("*_pid*")) == []


def test_refuses_to_kill_a_recycled_pid(turns):
    """PID numbers wrap. If /proc says the process started AFTER the newest file
    it supposedly owns, it is a different process and killing it would take out
    something unrelated — on a busy box that could be another cell."""
    proc = _spawn(turns, seed=4, n=1, interval=0.0)
    try:
        f = list(turns.glob(f"4_{proc.pid}_*.ndjson"))[0]
        # File predates the process => the process cannot have written it.
        old = sw.pid_start_time(proc.pid) - 600
        os.utime(f, (old, old))

        wd = _wd(turns, timeout=1.0)
        assert wd.check_dir(str(turns)) == []
        assert proc.poll() is None, "killed a process that could not have written the file"
        assert f.exists()
    finally:
        _reap([proc])

# --------------------------------------------------------------------------
# RELAUNCHED ROLLOUTS — the hole that wedged a run for 45 minutes
# --------------------------------------------------------------------------
#
# `max_relaunches: 2`: when a player wedges, the harness starts a NEW one. The
# ORIGINAL turn file's owner PID then dies, and the watchdog used to file that
# entry as "finished, its files are final" and stop inspecting it — while the
# relaunched player owns no turn file at all, so scan() saw nothing to watch.
# A retried or relaunched rollout therefore had ZERO coverage. Confirmed twice
# independently (e16_runs/mechcheck, e16_runs/methodtest).
#
# The fix keeps watching when the writer PID is dead but the ROLLOUT is not,
# and kills the live process instead of the dead file owner. These four tests
# pin both halves: it now fires on a relaunch, and it still leaves a genuinely
# completed rollout alone.


def _eval_cli(seconds: int = 3600) -> subprocess.Popen:
    """Stand-in for the eval CLI process the watchdog is armed against
    (`--parent-pid`). Must be started BEFORE any turn file, exactly as the real
    one is — the PID-reuse guard rejects a "parent" younger than its own data."""
    return subprocess.Popen([sys.executable, "-c",
                             f"import time; time.sleep({seconds})"])


def _write_traces(turns: Path, *seeds: int) -> Path:
    """The `traces.jsonl` the eval CLI writes when a rollout FINISHES, in the
    cell dir beside `turns/`. Its presence is what makes a dead writer's files
    genuinely final."""
    path = turns.parent / "traces.jsonl"
    with path.open("a") as fh:
        for seed in seeds:
            fh.write(json.dumps({
                "id": f"trace-{seed}", "task": {"data": {"idx": seed}},
                "is_completed": True, "stop_condition": "game_over",
            }) + "\n")
    return path


def test_a_relaunched_rollout_is_still_watched_after_its_writer_dies(turns):
    """THE BUG. Writer PID dies (the harness relaunched the player), the eval
    process is still alive, no new turn file appears. That is a wedged relaunch
    and the watchdog must fire on it — previously it filed the entry as
    finished and never looked again."""
    parent = _eval_cli()
    proc = _spawn(turns, seed=5, n=2, interval=0.05)
    try:
        fname = os.path.basename(list(turns.glob(f"5_{proc.pid}_*.ndjson"))[0])
        proc.kill()                       # the relaunch: this writer is gone
        proc.wait(timeout=10)
        assert not sw.pid_alive(proc.pid)
        time.sleep(1.2)                   # and nothing writes in its place

        wd = _wd(turns, timeout=1.0, parent_pid=parent.pid)
        kills = wd.check_dir(str(turns))

        assert len(kills) == 1, "a relaunched rollout got no watchdog coverage"
        k = kills[0]
        # The record separates the dead evidence-owner from the live victim.
        assert k["pid"] == proc.pid, "writer pid identifies the evidence"
        assert k["kill_pid"] == parent.pid, "the LIVE rollout process is killed"
        assert k["relaunched"] is True
        assert k["seeds"] == [5]

        # The live rollout is actually dead, not merely logged about.
        parent.wait(timeout=15)
        assert not sw.pid_alive(parent.pid)

        # And the truncated original file is quarantined, so the relaunch's
        # NDJSON cannot merge with it during grading (HARNESS_DEFECTS §4.6).
        assert list(turns.glob("*.ndjson")) == []
        qroot = Path(str(turns) + ".stalled")
        assert len(list(qroot.glob(f"*_pid{proc.pid}/{fname}"))) == 1
        # The attempt is still counted rather than vanishing from every table.
        partial = turns.parent / sw.PARTIAL_TRACES_BASENAME
        assert partial.exists()
        assert json.loads(partial.read_text().splitlines()[0])["task"]["data"]["idx"] == 5
    finally:
        _reap([proc, parent])


def test_a_genuinely_completed_rollout_is_still_left_alone(turns):
    """THE REGRESSION GUARD. The branch this fix reaches into exists to protect
    a rollout that simply FINISHED: its writer is gone and its files are final.
    A finished rollout has a `traces.jsonl` record, and a relaunched one does
    not — that is the discriminator. With the record present, the watchdog must
    keep its hands off even though the eval process is still alive (it is still
    running the cell's other work)."""
    parent = _eval_cli()
    proc = _spawn(turns, seed=6, n=2, interval=0.05)
    try:
        proc.kill()
        proc.wait(timeout=10)
        _write_traces(turns, 6)           # the CLI finalized this rollout
        time.sleep(1.2)

        wd = _wd(turns, timeout=1.0, parent_pid=parent.pid)
        assert wd.check_dir(str(turns)) == [], "killed a completed rollout"
        assert sw.pid_alive(parent.pid), "killed the eval process over final files"
        assert len(list(turns.glob("*.ndjson"))) == 1, "final data was quarantined"
        assert list(Path(str(turns) + ".stalled").glob("*_pid*")) == []
    finally:
        _reap([proc, parent])


def test_a_relaunch_that_is_writing_again_is_not_killed(turns):
    """The other half of the no-kill contract. A relaunch that came back and is
    producing turns under a NEW pid is progress. The dead original's entry must
    not fire on it — the dead writer is judged on the whole CELL's silence, not
    on its own orphaned file's age."""
    parent = _eval_cli()
    dead = _spawn(turns, seed=8, n=2, interval=0.05)
    dead.kill()
    dead.wait(timeout=10)
    time.sleep(1.2)                        # the original file is now stale
    live = _spawn(turns, seed=8, n=200, interval=0.2)   # the relaunch, writing
    try:
        wd = _wd(turns, timeout=1.0, poll=0.25, parent_pid=parent.pid)
        assert wd.run(max_seconds=3.0) == 0
        assert wd.kills == []
        assert sw.pid_alive(parent.pid), "killed a rollout that was making progress"
        assert live.poll() is None
    finally:
        _reap([dead, live, parent])


def test_no_parent_pid_means_a_dead_writer_is_still_treated_as_finished(turns):
    """Un-parented invocations are unchanged. Without `--parent-pid` there is no
    evidence the rollout is still live, so a dead writer's files stay final —
    the conservative reading, and the pre-existing behaviour."""
    proc = _spawn(turns, seed=9, n=1, interval=0.0)
    proc.kill()
    proc.wait(timeout=10)
    time.sleep(1.2)

    wd = _wd(turns, timeout=1.0)           # no parent_pid
    assert wd.check_dir(str(turns)) == []
    assert len(list(turns.glob("*.ndjson"))) == 1


# --------------------------------------------------------------------------
# arming / re-arming
# --------------------------------------------------------------------------
def test_exits_when_its_parent_pid_exits(turns):
    """`--parent-pid` is how the watchdog is re-armed per BATCH rather than
    outliving one. RUNBOOK's documented failure mode is a watchdog that exits
    when the eval queue drains and leaves later batches unguarded; tying the
    lifetime to the batch process makes arming and the batch the same event."""
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        started = time.time()
        rc = subprocess.run(
            [sys.executable, str(WATCHDOG), "--turns-dir", str(turns),
             "--timeout", "300", "--poll", "0.3", "--parent-pid", str(parent.pid)],
            capture_output=True, text=True, timeout=60)
        elapsed = time.time() - started
        assert rc.returncode == 0, rc.stderr
        assert elapsed < 20, "watchdog outlived its parent"
        recs = [json.loads(ln) for ln in
                (Path(str(turns) + ".stalled") / "stall_watchdog.jsonl").read_text().splitlines()]
        assert recs[0]["event"] == "armed" and recs[0]["parent_pid"] == parent.pid
        assert recs[-1]["event"] == "exit" and recs[-1]["reason"] == "parent_exited"
    finally:
        _reap([parent])


def test_wrapper_arms_for_exactly_the_commands_lifetime(turns):
    """tools/with_stall_watchdog.sh is the on-ramp for anything not launched
    through the two cell scripts. It must pass the command's exit code through
    (so it is a drop-in prefix) and leave no watchdog behind."""
    rc = subprocess.run(
        ["bash", str(REPO / "tools" / "with_stall_watchdog.sh"), str(turns),
         "--", "bash", "-c", "sleep 2; exit 17"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "STALL_TIMEOUT": "300", "STALL_POLL": "0.3"})
    assert rc.returncode == 17, f"exit code not passed through\n{rc.stderr}"
    assert "armed" in rc.stderr and "disarmed" in rc.stderr
    log = Path(str(turns) + ".stalled") / "stall_watchdog.jsonl"
    assert log.exists(), f"watchdog never armed\n{rc.stderr}"
    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert any(r["event"] == "armed" for r in recs)
    # and it is gone once the command is: nothing left guarding a dead batch
    wd_pids = {r["watchdog_pid"] for r in recs}
    assert not any(sw.pid_alive(p) for p in wd_pids), "watchdog outlived the command"


@pytest.mark.parametrize("script", [
    "tools/cli_harness_eval/launch_cell.sh",
    "tools/encoding_eval/launch_encoding_cell.sh",
])
def test_launchers_arm_the_watchdog_only_when_asked(script):
    """Opt-in by design: an unset STALL_WATCHDOG must leave a normal foreground
    run byte-for-byte as it was, so nobody's interactive debugging session gets
    SIGKILLed by a background process. `--parent-pid $$` is what ties the
    watchdog to THIS invocation (the `exec` at the end keeps the PID)."""
    text = (REPO / script).read_text()
    assert 'if [ -n "${STALL_WATCHDOG:-}" ]; then' in text
    assert "tools/stall_watchdog.py" in text
    assert '--parent-pid "$$"' in text
    assert '--timeout "${STALL_TIMEOUT:-300}"' in text


# --------------------------------------------------------------------------
# safety rails
# --------------------------------------------------------------------------
def test_refuses_a_quarantine_dir_inside_the_watched_turns_dir(turns):
    """Aggregation globs `turns/*.ndjson`. A quarantine dir nested under it
    would still be graded, so the move would accomplish nothing — refuse loudly
    rather than pretend."""
    rc = subprocess.run(
        [sys.executable, str(WATCHDOG), "--turns-dir", str(turns), "--once",
         "--quarantine-dir", str(turns / "dead")],
        capture_output=True, text=True, timeout=60)
    assert rc.returncode == 2
    assert "INSIDE the watched turns dir" in rc.stderr
    # the default location is a sibling, not a child
    assert not sw.quarantine_root(str(turns), None).startswith(str(turns) + os.sep)


def test_dry_run_detects_without_killing_or_moving(turns):
    proc = _spawn(turns, seed=5, n=1, interval=0.0)
    try:
        time.sleep(1.2)
        wd = _wd(turns, timeout=1.0, dry_run=True)
        kills = wd.check_dir(str(turns))
        assert len(kills) == 1 and kills[0]["dry_run"] is True
        assert proc.poll() is None, "dry run killed a process"
        assert len(list(turns.glob("*.ndjson"))) == 1, "dry run moved a file"
    finally:
        _reap([proc])


def test_last_turn_record_survives_a_truncated_tail(turns):
    """A wedged process is usually killed mid-write, so the final NDJSON line is
    often half-written. The evidence gatherer must still report the last COMPLETE
    turn instead of throwing away the whole file."""
    p = turns / "9_424242_1785108044.ndjson"
    p.write_text(json.dumps({"turn": 41, "dlvl": 3, "hp": 7}) + "\n"
                 + '{"turn": 42, "dlv')
    rec = sw.last_turn_record(str(p))
    assert rec["turn"] == 41 and rec["dlvl"] == 3 and rec["hp"] == 7
    assert rec["truncated_tail"] is True


def test_scan_groups_by_pid_and_tracks_each_seed(turns):
    now = time.time()
    for name, age in (("0_111_1.ndjson", 300), ("1_111_2.ndjson", 10),
                      ("0_222_3.ndjson", 50)):
        f = turns / name
        f.write_text("{}\n")
        os.utime(f, (now - age, now - age))
    by_pid = sw.scan(str(turns))
    assert sorted(by_pid) == [111, 222]
    # process-level freshness is the MAX over its seeds — seed 0 being 300s
    # stale does not make the process stale while seed 1 is 10s fresh.
    assert now - by_pid[111]["newest_mtime"] == pytest.approx(10, abs=2)
    assert sorted(by_pid[111]["seeds"]) == [0, 1]


def test_config_ports_extracts_mcp_urls():
    """Agent-first kill: the writer->agent join matches on ports referenced by
    per-rollout config URLs (settings.json / .mcp.json)."""
    import stall_watchdog as sw
    settings = ('{"mcpServers": {"nethack": {"type": "http", '
                '"url": "http://127.0.0.1:43571/mcp", '
                '"bearerTokenEnvVar": "NETHACK_MCP_TOKEN"}}}')
    assert sw.config_ports(settings) == {43571}
    assert sw.config_ports('{"url": "https://api.pinference.ai/api/v1"}') == set()
    assert sw.config_ports("") == set()
    assert sw.config_ports(None) == set()


def test_agent_pids_for_ports_empty_and_no_match():
    import stall_watchdog as sw
    assert sw.agent_pids_for_ports(set(), {1}) == []
    # A port nothing on the box references: scan must return [] and not raise.
    assert sw.agent_pids_for_ports({1}, set()) == []
