"""E16 durability gate: an interrupted attempt must not vanish.

WHAT THIS SUITE IS WRITTEN AGAINST -- a real run, `e16_runs/mechcheck`.

One attempt resumed c1 and played 249 LM turns to Dlvl 8 / XL 4 over 26
minutes, writing 23 auto-checkpoints into the archive. Every one of those
checkpoints was a valid, restorable bundle -- the persistent-state design
worked. What did not survive was the BOOKKEEPING: `attempts.jsonl` was never
created, `summary.json` was never written, and not one dollar of that attempt's
spend was ever attributed, because every record of an attempt was written at
the END of `ingest`, which only runs after the player subprocess returns. The
archive gained 24 checkpoints that belonged to an attempt that, as far as any
artifact in the run directory was concerned, had never happened.

That is the exact corruption this experiment exists to prevent. `censored` has
to be distinguishable from `died`; spend has to be attributable per attempt;
and a checkpoint in the archive has to name the attempt that produced it. An
attempt that disappears while its side effects persist breaks all three at once
and does it SILENTLY, which is worse than crashing.

So the tests below kill things for real. A child orchestrator plays real engine
steps through the real checkpoint path, is SIGKILLed (no handler runs, nothing
gets to clean up) or SIGTERMed (the handler runs) mid-attempt, and the
assertions are all about what is left ON DISK afterwards.

NO INFERENCE IS SPENT HERE. The player is injected and the orchestrator's
session is either absent (scripted selector) or restored from a recorded round
log, so the whole suite runs against a real engine and a real archive with no
model call anywhere.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))
sys.path.insert(0, str(REPO / "environments" / "nethack"))

import e16_orchestrator as E  # noqa: E402
from nethack_harness.checkpoints import (  # noqa: E402
    checkpoint_list, checkpoint_meta, checkpoint_restore, checkpoint_save,
)

WIKI_SRC = Path("/root/nld/e15-wiki/configs/continual/wiki")


def cfg_for(run_dir, **kw):
    kw.setdefault("selector", "scripted")
    kw.setdefault("budget_ceiling_usd", 1000.0)
    kw.setdefault("milestone_dlvl", 99)
    kw.setdefault("milestone_dungeon", -1)
    kw.setdefault("stall_attempts", 99)
    kw.setdefault("progress_interval_s", 0.5)
    return E.OrchestratorConfig(run_dir=Path(run_dir), wiki_src=WIKI_SRC, **kw)


# --------------------------------------------------------------------------- #
# the child orchestrator, killed for real
# --------------------------------------------------------------------------- #

DRIVER = r'''
"""A real Orchestrator whose second attempt never returns. Killed by the test."""
import json, os, sys, time
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))
sys.path.insert(0, str(REPO / "environments" / "nethack"))
import e16_orchestrator as E
from nethack_harness.checkpoints import checkpoint_restore, checkpoint_save

RUN = Path(sys.argv[2])
WIKI = Path(sys.argv[3])
HANG_ON = int(sys.argv[4])          # the attempt that never returns
CHECKPOINTS = int(sys.argv[5])      # how many it writes before hanging


def save_at(env, path, name, created_by):
    """checkpoint_save, but dismiss a --More-- first if the game is parked on one.

    The real save skill refuses to checkpoint mid-prompt (a savepoint taken on
    a --More-- restores into a state the player cannot act from). This stub
    plays blind `s` keys, so it hits that legitimately -- clearing the prompt
    is what a real player does, not a workaround for the checkpoint code.
    """
    last = None
    for _ in range(16):
        try:
            return checkpoint_save(env, path, name=name,
                                   note="automatic checkpoint",
                                   created_by=created_by)
        except Exception as exc:
            last = exc
            env.step(ord(" "))
    raise last


def launcher(ctx):
    env, _meta = checkpoint_restore(ctx.checkpoint_dir,
                                    fidelity_log=ctx.fidelity_log)
    turns = ctx.out_dir / "turns"
    turns.mkdir(parents=True, exist_ok=True)
    tf = turns / ("1_%d_%d.ndjson" % (os.getpid(), int(time.time())))
    if ctx.attempt != HANG_ON:
        for _ in range(4):
            env.step(ord("s"))
        env.modify(gold=500)
        saved = save_at(env, ctx.archive_dir / ("c%d" % (200 + ctx.attempt)),
                        "attempt %d done" % ctx.attempt, "save")
        with open(tf, "a") as fh:
            fh.write(json.dumps({"turn": 1, "dlvl": saved["dlvl"],
                                 "max_dlvl_reached": saved["dlvl"],
                                 "tool_calls": [{"name": "np_press_key"}],
                                 "applied": {"time": 40, "experience_level": 1}}) + "\n")
        return E.PlayerResult(stop_condition="died", died=True, calls=7,
                              spend_usd=2.0, max_dlvl=saved["dlvl"],
                              max_xl=saved["xl"], summary="died on purpose")

    # THE ATTEMPT THAT IS KILLED. It does what mechcheck's did: plays, and
    # writes auto-checkpoints into the shared archive as it goes.
    for i in range(1, CHECKPOINTS + 1):
        for _ in range(3):
            env.step(ord("s"))
        env.modify(gold=1000 * i)
        saved = save_at(env, ctx.archive_dir / ("c%d" % (100 + i)),
                        "auto %d" % i, "auto")
        with open(tf, "a") as fh:
            fh.write(json.dumps({"turn": i, "dlvl": saved["dlvl"],
                                 "max_dlvl_reached": saved["dlvl"],
                                 "hp": saved["hp"], "max_hp": saved["max_hp"],
                                 "tool_calls": [{"name": "np_press_key",
                                                 "arguments": {"key": "s"}}],
                                 "applied": {"time": 150 * i,
                                             "experience_level": 3,
                                             "score": saved["score"]}}) + "\n")
        (RUN / "READY").write_text(str(i))
    time.sleep(3600)      # never returns; the test kills this process
    return E.PlayerResult()


DIRECTIVE = ("Take the down-stairs on each level as soon as you see them, and "
             "save a checkpoint each time you reach a new deepest level.")


class Directing(E.Orchestrator):
    """The scripted selector writes no directive; this run needs one to lose.

    A directive is the TREATMENT, so "was the directive recovered?" is not a
    cosmetic question -- an attempt recovered without one would enter the
    dataset looking like a control.
    """

    def decide(self, rows):
        out = super().decide(rows)
        out["directive"] = DIRECTIVE
        return out


cfg = E.OrchestratorConfig(run_dir=RUN, wiki_src=WIKI, selector="scripted",
                           budget_ceiling_usd=1000.0, max_attempts=9,
                           stall_attempts=99, milestone_dlvl=99,
                           milestone_dungeon=-1, progress_interval_s=0.5)
orch = Directing(cfg, launcher)
orch.prepare()
E.seed_archive(cfg)
orch.run()
'''


def _spawn_driver(tmp_path, *, hang_on=1, checkpoints=3):
    run = Path(tmp_path) / "run"
    run.mkdir(parents=True, exist_ok=True)
    driver = Path(tmp_path) / "driver.py"
    driver.write_text(DRIVER)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO / "tools" / "pycompat"), os.environ.get("ENG", "/root/NetHack-engine"),
        str(REPO), str(REPO / "environments" / "nethack"),
        str(REPO / "tools" / "cli_harness_eval"),
        env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(REPO), str(run), str(WIKI_SRC),
         str(hang_on), str(checkpoints)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    ready = run / "READY"
    deadline = time.time() + 180
    while time.time() < deadline:
        if ready.is_file() and ready.read_text().strip() == str(checkpoints):
            time.sleep(1.0)  # let the journal + progress stream settle
            return proc, run
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"driver died before playing:\n{err.decode()[-4000:]}")
        time.sleep(0.25)
    proc.kill()
    pytest.fail("driver never reached the checkpoint it was waiting on")


def _kill(proc, sig):
    """Signal the driver's whole process GROUP, the way a real teardown does."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except OSError:
        proc.send_signal(sig)
    try:
        proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=30)


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# 1. the provisional row exists BEFORE the player does
# --------------------------------------------------------------------------- #

def test_the_attempt_is_on_disk_before_the_player_is_launched(tmp_path):
    """The whole fix in one assertion: the record precedes the side effects.

    mechcheck's attempt wrote 24 checkpoints before anything on disk said an
    attempt existed. Here the launcher asserts, from INSIDE the launch, that
    its own journal entry is already written and already names it.
    """
    cfg = cfg_for(tmp_path / "run", max_attempts=1)
    seen = {}

    def launcher(ctx):
        seen["journal"] = _rows(cfg.journal_path)
        seen["open_file"] = json.loads((ctx.out_dir / "attempt_open.json").read_text())
        seen["summary_exists"] = cfg.summary_path.is_file()
        seen["summary"] = json.loads(cfg.summary_path.read_text())
        env, _ = checkpoint_restore(ctx.checkpoint_dir, fidelity_log=ctx.fidelity_log)
        env.step(ord("s"))
        return E.PlayerResult(died=True, spend_usd=0.5, calls=3, summary="ok")

    orch = E.Orchestrator(cfg, launcher)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=1)

    opened = seen["journal"]
    assert [r["event"] for r in opened] == ["open"], \
        "an `open` line must be on disk before the player runs"
    op = opened[0]
    for key in ("attempt", "from_checkpoint", "directive", "ts", "pid", "out_dir"):
        assert key in op, f"the provisional row must carry {key}"
    assert op["pid"] == os.getpid()
    assert op["from_checkpoint"] == "1"
    assert seen["open_file"]["attempt"] == 1
    # summary.json exists DURING the attempt and says which one is in flight.
    assert seen["summary_exists"]
    assert seen["summary"]["in_flight"]["attempt"] == 1

    # ...and it is closed exactly once when the row goes final.
    events = _rows(cfg.journal_path)
    assert [r["event"] for r in events] == ["open", "close"]
    assert events[1]["outcome"] == E.OUTCOME_DIED
    assert json.loads(cfg.summary_path.read_text())["in_flight"] is None


def test_the_progress_stream_is_tailable_while_the_attempt_plays(tmp_path):
    """`tail -f progress.jsonl` must show a run moving -- or not moving.

    The column that matters is `idle_s`: seconds since the harness last wrote a
    turn record. mechcheck looked dead for 12 minutes while its harness sat in
    a relaunch, and there was no artifact anywhere that would have shown the
    difference between that and a wedge.
    """
    cfg = cfg_for(tmp_path / "run", max_attempts=1, progress_interval_s=0.5)

    def launcher(ctx):
        env, _ = checkpoint_restore(ctx.checkpoint_dir, fidelity_log=ctx.fidelity_log)
        turns = ctx.out_dir / "turns"
        turns.mkdir(parents=True, exist_ok=True)
        tf = turns / f"1_{os.getpid()}_{int(time.time())}.ndjson"
        for i in range(1, 4):
            env.step(ord("s"))
            env.modify(gold=100 * i)
            checkpoint_save(env, ctx.archive_dir / f"c{300 + i}", name=f"auto {i}",
                            note="auto", created_by="auto")
            tf.write_text(json.dumps({
                "turn": i, "dlvl": 1, "max_dlvl_reached": 1, "hp": 9, "max_hp": 12,
                "tool_calls": [{"name": "np_press_key"}],
                "applied": {"time": 111 * i, "experience_level": 2, "score": 5 * i},
            }) + "\n")
            time.sleep(0.8)
        return E.PlayerResult(died=True, spend_usd=1.0, calls=3, summary="ok")

    orch = E.Orchestrator(cfg, launcher)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=1)

    stream = _rows(cfg.progress_path)
    kinds = [r["event"] for r in stream]
    assert kinds[0] == "attempt_start" and kinds[-1] == "attempt_end"
    samples = [r for r in stream if r["event"] == "progress"]
    assert samples, "no progress samples were written while the attempt played"
    last = samples[-1]
    for key in ("attempt", "iso", "wall_s", "idle_s", "lm_turn", "game_turn",
                "dlvl", "xl", "hp", "checkpoints_total", "checkpoints_new"):
        assert key in last, f"the progress stream must carry {key}"
    assert last["idle_s"] is not None and last["idle_s"] >= 0
    # It saw the archive grow WHILE the attempt was still running -- which is
    # the whole point: mechcheck's checkpoints were the only live signal, and
    # they were only visible by staring at the directory.
    assert max(len(r["checkpoints_new"]) for r in samples) >= 1
    assert stream[-1]["outcome"] == E.OUTCOME_DIED
    # The per-attempt mirror is there too, so an attempt directory is
    # self-describing when it is copied off the box on its own.
    assert (cfg.run_dir / "attempts" / "a001" / "progress.jsonl").is_file()


# --------------------------------------------------------------------------- #
# 2. SIGKILL -- nothing gets to clean up, and recovery still works
# --------------------------------------------------------------------------- #

def test_a_SIGKILLED_run_is_reconciled_into_a_censored_attempt(tmp_path):
    """THE MECHCHECK CASE, reproduced and then repaired.

    SIGKILL is the honest test: no handler runs, no `finally` runs, nothing in
    the orchestrator gets a chance to write anything. What is left is exactly
    what mechcheck was left as -- an archive full of checkpoints and an attempt
    that never reported -- and reconciliation has to turn that into a coherent
    run state without inventing anything.
    """
    proc, run = _spawn_driver(tmp_path, hang_on=1, checkpoints=3)
    _kill(proc, signal.SIGKILL)
    cfg = cfg_for(run)

    # -- the wreckage, before recovery --------------------------------------
    assert not cfg.attempts_path.exists(), \
        "precondition: a SIGKILLed attempt leaves no final row (this is the bug)"
    archive_before = sorted(p.name for p in checkpoint_list(cfg.archive_dir))
    assert len(archive_before) == 4, archive_before          # seed + 3 auto
    orphans = [p.name for p in checkpoint_list(cfg.archive_dir)
               if not checkpoint_meta(p).get("created_in_attempt")
               and checkpoint_meta(p).get("created_by") != "orchestrator"]
    assert len(orphans) == 3, "precondition: the checkpoints are orphaned"
    # But the provisional row survived the kill, because it was written first.
    journal = _rows(cfg.journal_path)
    assert [r["event"] for r in journal] == ["open"]
    assert journal[0]["from_checkpoint"] == "1"

    # -- recovery -----------------------------------------------------------
    report = E.reconcile_run(cfg)

    assert report["attempts_finalized_now"] == [1]
    assert len(report["checkpoints_attributed"]) == 3
    assert report["unattributed_checkpoints"] == [], \
        "every checkpoint an attempt produced must be attributable to it"
    assert report["seed_checkpoints"] == ["c1"]

    rows = _rows(cfg.attempts_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == E.OUTCOME_CENSORED
    assert row["censored"] is True
    assert row["censor_reason"] == E.CENSOR_INTERRUPTED
    assert row["recovered"] is True
    assert row["from_checkpoint"] == "1"
    # THE TREATMENT SURVIVES. An attempt recovered without its directive would
    # enter the dataset looking like a control.
    assert row["directive"].startswith("Take the down-stairs")
    assert row["directive"] == (
        cfg.run_dir / "attempts" / "a001" / "directive_served.txt"
    ).read_text().strip()
    assert row["directive_kind"] != "none"
    assert sorted(row["new_checkpoints"]) == ["c101", "c102", "c103"]
    # Lower bounds recovered from the attempt's own turn records, not invented.
    assert row["calls"] >= 3 and row["lm_turns_written"] == 3
    assert row["max_dlvl"] >= 1
    assert row["spend_known"] is False, \
        "no traces.jsonl means no honest spend number, and it must say so"

    # -- the archive is untouched apart from attribution --------------------
    assert sorted(p.name for p in checkpoint_list(cfg.archive_dir)) == archive_before
    for name in ("c101", "c102", "c103"):
        meta = checkpoint_meta(cfg.archive_dir / name)
        assert meta["created_in_attempt"] == 1
        assert meta["attributed_by"] == "recovery"
        assert (cfg.archive_dir / name / "state.bundle").stat().st_size > 0
    # RECOVERY REBUILDS THE CHAIN, NOT A FAN. It used to stamp every orphan
    # with the checkpoint the attempt resumed, which is how a recovered archive
    # came out a star: c101, c102 and c103 all children of c1, with nothing
    # recording that c102 was written after c101 in the same life.
    assert [checkpoint_meta(cfg.archive_dir / n)["parent"]
            for n in ("c101", "c102", "c103")] == ["1", "101", "102"]
    assert E.archive_tree(E.ledger_rows(cfg.archive_dir))["problems"] == []
    assert "created_in_attempt" not in checkpoint_meta(cfg.archive_dir / "c1"), \
        "the seed state belongs to no attempt and must not be attributed to one"

    # -- and it is IDEMPOTENT: a second reconcile changes nothing -----------
    again = E.reconcile_run(cfg)
    assert again["attempts_finalized_now"] == []
    assert again["checkpoints_attributed"] == []
    assert len(_rows(cfg.attempts_path)) == 1


def test_a_run_with_no_journal_at_all_is_still_recoverable(tmp_path):
    """mechcheck EXACTLY: no attempts.jsonl, no journal, just a directory.

    The real run predates the journal, so recovery cannot depend on it. The
    attempt directory -- its served directive and its turn records -- is
    evidence on its own, and refusing to read it is how the attempt was lost.
    """
    proc, run = _spawn_driver(tmp_path, hang_on=1, checkpoints=2)
    _kill(proc, signal.SIGKILL)
    cfg = cfg_for(run)
    cfg.journal_path.unlink()                       # pretend it never existed
    (cfg.run_dir / "attempts" / "a001" / "attempt_open.json").unlink()

    report = E.reconcile_run(cfg)

    assert report["attempts_finalized_now"] == [1]
    assert report["unattributed_checkpoints"] == []
    row = _rows(cfg.attempts_path)[0]
    assert row["outcome"] == E.OUTCOME_CENSORED
    assert row["censor_reason"] == E.CENSOR_INTERRUPTED
    assert sorted(row["new_checkpoints"]) == ["c101", "c102"]
    # The directive is recovered from the bytes that were SERVED, which is the
    # only copy that survives when the orchestrator's memory does not.
    assert row["directive"] == (
        cfg.run_dir / "attempts" / "a001" / "directive_served.txt"
    ).read_text().strip()


# --------------------------------------------------------------------------- #
# 3. SIGTERM -- the handler runs, and the row is final before the process goes
# --------------------------------------------------------------------------- #

def test_SIGTERM_finalizes_the_in_flight_attempt_before_exiting(tmp_path):
    """A signal must not be able to skip the bookkeeping.

    Python's default SIGTERM terminates without running one `finally`, so the
    orchestrator installs a handler that raises instead. By the time the
    process is gone the attempt is a final `censored:interrupted` row and the
    summary has been flushed -- no reconciliation needed.
    """
    proc, run = _spawn_driver(tmp_path, hang_on=1, checkpoints=2)
    _kill(proc, signal.SIGTERM)
    cfg = cfg_for(run)

    rows = _rows(cfg.attempts_path)
    assert len(rows) == 1, "the in-flight attempt must be finalized in-process"
    row = rows[0]
    assert row["outcome"] == E.OUTCOME_CENSORED
    assert row["censor_reason"] == E.CENSOR_INTERRUPTED
    assert "SIGTERM" in (row["error"] or "") or "RunInterrupted" in (row["error"] or "")
    assert sorted(row["new_checkpoints"]) == ["c101", "c102"]
    assert [r["event"] for r in _rows(cfg.journal_path)] == ["open", "close"]

    summary = json.loads(cfg.summary_path.read_text())
    assert summary["attempts"] == 1
    assert summary["attempts_censored"] == 1
    assert summary["attempts_died"] == 0, \
        "an interrupted attempt is CENSORED; pooling it with deaths is the bug"
    assert summary["censor_reasons"] == {E.CENSOR_INTERRUPTED: 1}
    assert summary["in_flight"] is None

    # Nothing is left for recovery to find, and it says so.
    report = E.reconcile_run(cfg)
    assert report["attempts_finalized_now"] == []
    assert report["unattributed_checkpoints"] == []


# --------------------------------------------------------------------------- #
# 4. resume: continue without losing or double-counting anything
# --------------------------------------------------------------------------- #

def test_resume_continues_the_run_without_double_counting(tmp_path):
    """Kill mid-attempt, resume, and check every number is continuous.

    Attempt 1 completes and costs $2. Attempt 2 is SIGKILLed mid-play with two
    checkpoints already in the archive. The resumed run must: finalize attempt
    2 as censored, attribute its checkpoints, keep attempt 1's spend, launch
    attempt 3 as a NEW attempt, and never re-charge or re-run anything.
    """
    proc, run = _spawn_driver(tmp_path, hang_on=2, checkpoints=2)
    _kill(proc, signal.SIGKILL)
    cfg = cfg_for(run, max_attempts=3)

    before = sorted(p.name for p in checkpoint_list(cfg.archive_dir))
    assert before == ["c1", "c101", "c102", "c201"], before
    assert [r["attempt"] for r in _rows(cfg.attempts_path)] == [1], \
        "precondition: attempt 1 finalized normally, attempt 2 did not"

    launched = []

    def launcher(ctx):
        launched.append(ctx.attempt)
        env, _ = checkpoint_restore(ctx.checkpoint_dir, fidelity_log=ctx.fidelity_log)
        env.step(ord("s"))
        return E.PlayerResult(died=True, spend_usd=1.5, calls=4,
                              summary="third attempt")

    orch = E.Orchestrator(cfg, launcher)
    orch.prepare()
    state = orch.resume()

    assert state["attempts_recovered"] == 2, "1 completed + 1 recovered"
    assert state["recovery"]["attempts_finalized_now"] == [2]
    assert state["recovery"]["unattributed_checkpoints"] == []
    # SPEND IS CARRIED FORWARD, not restarted.
    assert orch.budget.player_usd == pytest.approx(2.0)

    summary = orch.run(max_attempts=3)

    # -- no attempt was re-run and none was lost ----------------------------
    assert launched == [3], "resume must not re-launch an attempt that already ran"
    rows = _rows(cfg.attempts_path)
    assert [r["attempt"] for r in rows] == [1, 2, 3]
    assert len({r["attempt"] for r in rows}) == 3, "no attempt id may repeat"
    assert [r["outcome"] for r in rows] == [
        E.OUTCOME_DIED, E.OUTCOME_CENSORED, E.OUTCOME_DIED]
    assert rows[1]["censor_reason"] == E.CENSOR_INTERRUPTED

    # -- spend is continuous across the restart -----------------------------
    assert summary["budget"]["player_usd"] == pytest.approx(3.5)  # 2.0 + 0.0 + 1.5
    assert summary["cumulative_spend_usd"] == pytest.approx(3.5)
    assert rows[2]["cumulative_spend_usd"] == pytest.approx(3.5), \
        "the new attempt's cumulative line must include the pre-crash spend"
    assert summary["resumed"] is True
    assert summary["attempts_recovered"] == 1
    assert summary["attempts_with_unknown_spend"] == 1

    # -- the archive is intact, and every checkpoint has an owner -----------
    after = sorted(p.name for p in checkpoint_list(cfg.archive_dir))
    assert set(before).issubset(after), "resume must not lose a checkpoint"
    unowned = [p.name for p in checkpoint_list(cfg.archive_dir)
               if not checkpoint_meta(p).get("created_in_attempt")
               and checkpoint_meta(p).get("created_by") != "orchestrator"]
    assert unowned == [], f"orphaned checkpoints after resume: {unowned}"
    assert summary["unattributed_checkpoints"] == []
    owners = [checkpoint_meta(cfg.archive_dir / n)["created_in_attempt"]
              for n in ("c101", "c102")]
    assert owners == [2, 2]
    # No checkpoint is claimed by two attempts.
    claimed = [n for r in rows for n in (r.get("new_checkpoints") or [])]
    assert len(claimed) == len(set(claimed)), f"double-counted: {claimed}"


def test_resume_reattaches_the_orchestrator_session_by_its_recorded_id(tmp_path):
    """The conversation continues; it does not restart.

    The orchestrator's value is that one session accumulates a strategy across
    rounds. A resume that opened a fresh session would silently convert the run
    into "N independent calls wearing a session's name" -- and would pay for a
    second opening plan while replacing the one every earlier attempt was
    steered by. The session id is read from the round log, which is appended
    and fsynced per round and therefore survives a crash that provenance.json
    does not.
    """
    cfg = cfg_for(tmp_path / "run", selector="llm")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    sid = "01a048b8-92ce-734f-bae0-6f871815b188"       # mechcheck's real id
    for rec in (
        {"round": 1, "kind": "opening", "session_id": sid, "spend_usd": 0.11,
         "sent_chars_cumulative": 4000, "session_cwd": str(cfg.orchestrator_dir)},
        {"round": 2, "kind": "round1", "session_id": sid, "spend_usd": 0.07,
         "sent_chars_cumulative": 9000, "session_cwd": str(cfg.orchestrator_dir)},
    ):
        E._append_jsonl(cfg.orchestrator_log, rec)

    class FakeSession:
        kind = "prime_agent_session"
        session_id = ""
        session_ids: list = []
        session_cwd = ""
        session_file = ""
        rounds = 0
        sent_chars = 0
        spend_usd = 0.0
        compactions = 0
        continuity_breaks = 0
        usage_available = False

        def budget_lines(self):
            return {"rounds": self.rounds, "spend_usd": self.spend_usd,
                    "session_ids": list(self.session_ids)}

    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(), session=FakeSession())
    orch.prepare()
    E.seed_archive(cfg)
    state = orch.resume()

    sess = state["session"]
    assert sess["attached"] is True
    assert sess["session_id"] == sid, "the SAME conversation, not a new one"
    assert sess["session_ids"] == [sid]
    assert sess["rounds"] == 2
    assert sess["sent_chars"] == 9000
    assert orch.session.spend_usd == pytest.approx(0.18)
    # The opening plan is already bought and on the record: buying another one
    # would cost money AND replace the plan the run has been following.
    assert orch.opened is True
    assert sess["opening_already_bought"] is True
    # The orchestrator's own budget line is rebuilt from the round log.
    assert orch.budget.orchestrator_usd == pytest.approx(0.18)


def test_a_checkpoint_no_attempt_can_claim_is_reported_not_absorbed(tmp_path):
    """Silence is the failure mode; an unattributable checkpoint must be loud."""
    cfg = cfg_for(tmp_path / "run")
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult())
    orch.prepare()
    E.seed_archive(cfg)
    # A checkpoint from nowhere: created long before any attempt window.
    env, _ = checkpoint_restore(cfg.archive_dir / "c1", fidelity_log=cfg.fidelity_path)
    env.step(ord("s"))
    checkpoint_save(env, cfg.archive_dir / "c999", name="from nowhere",
                    note="no attempt produced this", created_by="auto")
    meta = checkpoint_meta(cfg.archive_dir / "c999")
    meta["created_at"] = 1.0                     # 1970
    E.atomic_write(cfg.archive_dir / "c999" / "meta.json",
                   json.dumps(meta, indent=2, sort_keys=True) + "\n")

    report = E.reconcile_run(cfg)

    assert [c["checkpoint"] for c in report["unattributed_checkpoints"]] == ["c999"]
    assert report["unattributed_checkpoints"][0]["why"]
    assert "created_in_attempt" not in checkpoint_meta(cfg.archive_dir / "c999"), \
        "an unattributable checkpoint must not be given an owner it does not have"


def test_reconcile_refuses_a_run_another_orchestrator_still_holds(tmp_path):
    """Recovery must not become the corruption it exists to repair.

    THIS IS NOT HYPOTHETICAL. The mechcheck run that motivated this whole
    section looked dead from its file timestamps -- 26 minutes of activity and
    then nothing -- while its orchestrator, its eval process and its stall
    watchdog were all still alive and its harness was 12 minutes into a
    relaunch. Reconciling it in that state would have written a `censored` row
    for an attempt that was still playing and handed its future checkpoints to
    a closed attempt: a corrupted dataset produced by the repair tool.
    """
    proc, run = _spawn_driver(tmp_path, hang_on=1, checkpoints=1)
    cfg = cfg_for(run)
    try:
        # The driver is STILL RUNNING, mid-attempt, right now.
        assert E.live_orchestrator_pids(run) == [proc.pid]
        with pytest.raises(E.RunStillLive) as exc:
            E.reconcile_run(cfg)
        assert str(proc.pid) in str(exc.value)
        with pytest.raises(E.RunStillLive):
            E.Orchestrator(cfg, lambda ctx: E.PlayerResult()).resume()
        # Nothing was written: the live attempt has no row and no owner stamps.
        assert not cfg.attempts_path.exists()
        assert not checkpoint_meta(cfg.archive_dir / "c101").get("created_in_attempt")
    finally:
        _kill(proc, signal.SIGKILL)

    # Once it is really gone, the same call goes through.
    report = E.reconcile_run(cfg)
    assert report["live_orchestrator_pids"] == []
    assert report["attempts_finalized_now"] == [1]


def test_a_recovered_row_reads_the_key_the_harness_actually_writes(tmp_path):
    """Locked against the real `mechcheck` turn schema, byte shapes and all.

    `helpers._write_trace_entry` puts the engine's blstats under `status`;
    `applied` in that schema is a BOOLEAN. Reading the wrong one fails
    silently -- an empty dict, and every recovered number quietly defaults.
    The first recovery of mechcheck's attempt came back claiming XL 1 and game
    turn 0 for a game whose own last record says XL 4 at turn 2208, which is
    exactly the class of "a number nobody measured" this run's records exist to
    keep out. And `from_checkpoint` came back null, because a run predating the
    journal keeps that fact in `selection.jsonl` instead.
    """
    cfg = cfg_for(tmp_path / "run")
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult())
    orch.prepare()
    E.seed_archive(cfg)
    E.clear_run_lock(cfg.run_dir)

    # `selection.jsonl` is appended BEFORE the launch, so it survives too.
    E._append_jsonl(cfg.selection_path, {
        "attempt": 1, "source": "llm", "chosen_id": "1",
        "directive": "Take the down-stairs on each level as soon as you see them.",
        "directive_kind": "descend_fast"})
    out = cfg.run_dir / "attempts" / "a001"
    (out / "turns").mkdir(parents=True)
    (out / "directive_served.txt").write_text(
        "Take the down-stairs on each level as soon as you see them.")
    (out / "turns" / "1_67460_1787926656.ndjson").write_text("\n".join(
        json.dumps({
            "turn": t, "dlvl": 8, "max_dlvl_reached": 8, "hp": 43, "max_hp": 43,
            "applied": True,                       # a BOOLEAN, as in the real file
            "status": {"depth": 8, "experience_level": 4, "time": 2200 + t,
                       "hitpoints": 43, "max_hitpoints": 43, "score": 731},
            "tool_calls": [{"name": "request_map", "arguments": {}}],
        }) for t in (247, 248, 249)) + "\n")
    # A checkpoint the player wrote and nobody attributed, exactly as found.
    env, _ = checkpoint_restore(cfg.archive_dir / "c1", fidelity_log=cfg.fidelity_path)
    env.step(ord("s"))
    checkpoint_save(env, cfg.archive_dir / "c24", name="auto level_up_xl4",
                    note="automatic checkpoint", created_by="auto")

    report = E.reconcile_run(cfg)

    assert report["attempts_finalized_now"] == [1]
    assert report["unattributed_checkpoints"] == []
    row = _rows(cfg.attempts_path)[0]
    assert row["from_checkpoint"] == "1", "selection.jsonl names the source state"
    assert row["max_xl"] == 4, "XL must come from `status`, not from `applied`"
    assert row["game_turn_reached"] == 2449
    assert row["max_dlvl"] == 8
    assert row["lm_turns_written"] == 249
    assert row["calls"] == 3
    assert row["directive_kind"] == "descend_fast"
    assert row["new_checkpoints"] == ["c24"]
    assert checkpoint_meta(cfg.archive_dir / "c24")["parent"] == "1"
