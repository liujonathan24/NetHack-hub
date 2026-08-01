#!/usr/bin/env python
"""Stall watchdog for NetHack eval rollouts.

WHY THIS IS NOT OPTIONAL FOR AN UNATTENDED RUN
----------------------------------------------
`docs/HARNESS_DEFECTS.md` §3.1 lists two unbounded hangs that are still open:

  * `np_explore_level` intermittently spins inside the vendored pathfinding
    (`nethack_bfs`) — measured NO return in 600 s, versus 0.0–0.5 s normally.
  * post-death `rollback` followed by any NetPlay skill deadlocks the adapter
    (~60 leaked threads). Reproduced 3 of 3.

**Neither reaches `env.step`**, so the engine's own `no_progress_timeout` can
never fire, and nothing in-process will ever notice. The only signal that
survives a wedged event loop is the absence of new bytes on disk: a healthy
rollout appends to `turns/<seed>_<pid>_<ts>.ndjson` every few seconds.

WHAT IT DOES
------------
Polls one or more `turns/` directories. A rollout process is declared STALLED
when no file it owns has been written for `--timeout` seconds (default 300).
On a stall it:

  1. records the evidence — seed(s), PID, stall duration, and the last NDJSON
     record actually written (turn number, dlvl, hp, wall clock);
  2. kills the PID and its descendants (SIGTERM, then SIGKILL after
     `--kill-grace`) — the tool-server children are killed too, or the MCP
     subprocesses outlive the rollout;
  3. QUARANTINES every turn file that PID owned, moving them out of `turns/`
     into `<turns>.stalled/<utc>_pid<pid>/` with a `MANIFEST.json`;
  4. appends a JSON record to `<turns>.stalled/stall_watchdog.jsonl` and a
     human line to stderr.

Step 3 is not optional. `docs/HARNESS_DEFECTS.md` §4.6: "`turns/` is shared per
cell and grading groups by seed prefix, so a dead run's partial NDJSON MERGES
with the relaunch." Killing without quarantining converts a hang into silently
corrupted results, which is strictly worse than the hang.

WHY THE STALL IS MEASURED PER-PID, NOT PER-SEED
-----------------------------------------------
One eval process owns every seed of a cell — the committed traces show a single
PID across all of `0_..`, `1_..`, `2_..` in one `turns/` dir. Two consequences:

  * A seed that simply FINISHED stops writing forever. Per-seed staleness would
    therefore kill a perfectly healthy cell `--timeout` seconds after its first
    seed completed. That failure mode is worse than the one being mitigated.
  * Both known hangs wedge the whole process anyway (a blocking C call in the
    pathfinding, and a thread deadlock in the adapter), so process-level silence
    is the true signal for both.

So: kill only when NOTHING in the process has moved. That cannot fire on a cell
where any seed is still making progress. The per-seed timings are still recorded
in the kill record, because that is what identifies WHICH seed hung.

RE-ARMING (the documented failure mode)
---------------------------------------
`RUNBOOK.md`: "Re-arm it per batch — it exits when the queue drains." A watchdog
started once for a whole sweep leaves every later batch unguarded. Three ways to
get per-batch arming, in order of preference:

  1. `STALL_WATCHDOG=1 tools/cli_harness_eval/launch_cell.sh ...` — both
     launchers arm one watchdog per invocation and tie its lifetime to the eval
     process via `--parent-pid`, so arming and the batch are the same event and
     cannot get out of step.
  2. `tools/with_stall_watchdog.sh <turns-dir> -- <any command>` — same
     property for anything not launched through those scripts.
  3. `--parent-pid PID` by hand: the watchdog exits when PID does.

Usage:
    tools/stall_watchdog.py --turns-dir OUT/turns [--turns-dir OUT2/turns] \
        [--timeout 300] [--poll 15] [--parent-pid $$] [--dry-run] [--once]

Exit codes: 0 clean, 3 it killed at least one rollout (greppable in a log), 2
bad arguments.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone

# `<seed>_<pid>_<unixtime>.ndjson`, built by
# environments/nethack/nethack_harness/helpers.py:
#     run_id = f"{seeds[0]}_{os.getpid()}_{int(time.time())}"
RUN_ID_RE = re.compile(r"^(?P<seed>\d+)_(?P<pid>\d+)_(?P<ts>\d+)\.ndjson$")

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
# A process cannot have written a file before it started. If /proc says the PID
# started AFTER the newest file it supposedly owns (beyond this slack), the PID
# has been recycled by an unrelated process and must not be killed.
_PID_REUSE_SLACK_S = 5.0


# --------------------------------------------------------------------------
# process helpers
# --------------------------------------------------------------------------
def pid_alive(pid: int) -> bool:
    """True only if `pid` is a RUNNING process.

    A zombie must count as dead. `os.kill(pid, 0)` succeeds against a zombie —
    the PID entry survives until the parent reaps it — so the naive check makes
    an already-killed rollout look like it survived SIGKILL, and makes a
    `--parent-pid` whose parent has not yet been waited on look alive forever
    (the watchdog would then never exit, i.e. never re-arm). Both were observed.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            raw = fh.read()
        return raw[raw.rindex(")") + 2:].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM  # alive but not ours to signal
    return True


def _boot_time() -> float:
    try:
        with open("/proc/stat") as fh:
            for line in fh:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        pass
    return 0.0


def pid_start_time(pid: int) -> float | None:
    """Wall-clock start time of `pid`, or None if unknown/unreadable."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            raw = fh.read()
    except OSError:
        return None
    # comm (field 2) is parenthesized and may contain spaces/parens.
    try:
        tail = raw[raw.rindex(")") + 2:].split()
        return _boot_time() + float(tail[19]) / _CLK_TCK  # field 22 -> tail[19]
    except (ValueError, IndexError):
        return None


def descendants(pid: int) -> list[int]:
    """All transitive children of `pid`, excluding this process.

    Excluding self matters: when armed from a launcher, the watchdog is itself a
    child of the eval process it is guarding, and would otherwise kill itself
    before it could quarantine anything.
    """
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                raw = fh.read()
            ppid = int(raw[raw.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry))

    me = os.getpid()
    out, stack = [], list(children.get(pid, []))
    while stack:
        p = stack.pop()
        if p == me or p in out:
            continue
        out.append(p)
        stack.extend(children.get(p, []))
    return out


def kill_tree(pid: int, grace: float, dry_run: bool = False) -> dict:
    """SIGTERM the process tree, then SIGKILL whatever is left after `grace`."""
    targets = [pid] + descendants(pid)
    record = {"targets": targets, "sigterm": [], "sigkill": [], "survivors": []}
    if dry_run:
        record["dry_run"] = True
        return record

    for p in targets:
        try:
            os.kill(p, signal.SIGTERM)
            record["sigterm"].append(p)
        except OSError:
            pass

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not any(pid_alive(p) for p in targets):
            break
        time.sleep(0.2)

    for p in targets:
        if pid_alive(p):
            try:
                os.kill(p, signal.SIGKILL)
                record["sigkill"].append(p)
            except OSError:
                pass
    time.sleep(0.2)
    record["survivors"] = [p for p in targets if pid_alive(p)]
    return record


# --------------------------------------------------------------------------
# turn-file inspection
# --------------------------------------------------------------------------
def last_turn_record(path: str) -> dict:
    """Summary of the last complete NDJSON record in `path`.

    A stalled rollout's file is usually mid-write, so the final line can be
    truncated; walk backwards until a line parses. This is the "what was the
    last recorded turn" evidence required to reconstruct the hang.
    """
    out: dict = {"file": os.path.basename(path)}
    try:
        with open(path, "rb") as fh:
            lines = fh.read().splitlines()
        out["n_lines"] = len(lines)
        for raw in reversed(lines):
            try:
                rec = json.loads(raw.decode("utf-8", "replace"))
            except (ValueError, UnicodeDecodeError):
                out["truncated_tail"] = True
                continue
            if not isinstance(rec, dict):
                continue
            for key in ("turn", "dlvl", "hp", "max_hp", "variant",
                        "max_dlvl_reached", "t_wall", "reward"):
                if key in rec:
                    out[key] = rec[key]
            calls = rec.get("tool_calls") or []
            if calls:
                first = calls[0]
                out["last_tool_call"] = (
                    first.get("name") if isinstance(first, dict) else str(first)[:80])
            msg = rec.get("rendered_user_message")
            if isinstance(msg, str):
                out["last_message_head"] = msg.strip().splitlines()[0][:200] if msg.strip() else ""
            break
    except OSError as exc:
        out["error"] = str(exc)
    return out


def scan(turns_dir: str) -> dict[int, dict]:
    """Group `turns/*.ndjson` by owning PID.

    Returns {pid: {"newest_mtime": float, "seeds": {seed: {...}}, "files": [...]}}.
    """
    by_pid: dict[int, dict] = {}
    try:
        entries = os.listdir(turns_dir)
    except OSError:
        return by_pid
    for fname in entries:
        m = RUN_ID_RE.match(fname)
        if not m:
            continue
        path = os.path.join(turns_dir, fname)
        try:
            st = os.stat(path)
        except OSError:
            continue
        pid, seed = int(m.group("pid")), int(m.group("seed"))
        slot = by_pid.setdefault(pid, {"newest_mtime": 0.0, "seeds": {}, "files": []})
        slot["files"].append(path)
        slot["newest_mtime"] = max(slot["newest_mtime"], st.st_mtime)
        seed_slot = slot["seeds"].setdefault(seed, {"newest_mtime": 0.0, "files": []})
        seed_slot["files"].append(path)
        seed_slot["newest_mtime"] = max(seed_slot["newest_mtime"], st.st_mtime)
    return by_pid


# --------------------------------------------------------------------------
# watchdog
# --------------------------------------------------------------------------
def quarantine_root(turns_dir: str, override: str | None) -> str:
    """Where dead rollouts' turn files go.

    Default is a SIBLING of `turns/`, never a subdirectory of it: aggregation
    globs `turns/*.ndjson`, so a nested quarantine would still be picked up and
    the whole point (§4.6, "a dead run's partial NDJSON merges with the
    relaunch") would be lost.
    """
    return override or (turns_dir.rstrip(os.sep) + ".stalled")


class Watchdog:
    def __init__(self, turns_dirs, timeout=300.0, poll=15.0, kill_grace=10.0,
                 quarantine=None, log_path=None, dry_run=False, verbose=False):
        self.turns_dirs = [os.path.abspath(d) for d in turns_dirs]
        self.timeout = float(timeout)
        self.poll = float(poll)
        self.kill_grace = float(kill_grace)
        self.quarantine = quarantine
        self.log_path = log_path
        self.dry_run = dry_run
        self.verbose = verbose
        self.kills: list[dict] = []
        self._handled: set[tuple[str, int]] = set()
        self._warned_reuse: set[int] = set()

    # -- logging ----------------------------------------------------------
    def _log_path_for(self, turns_dir: str) -> str:
        if self.log_path:
            return self.log_path
        return os.path.join(quarantine_root(turns_dir, self.quarantine),
                            "stall_watchdog.jsonl")

    def log(self, turns_dir: str, event: str, **fields) -> dict:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "watchdog_pid": os.getpid(),
            "turns_dir": turns_dir,
            **fields,
        }
        path = self._log_path_for(turns_dir)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError as exc:
            sys.stderr.write(f"[stall_watchdog] cannot write log {path}: {exc}\n")
        return rec

    def say(self, msg: str) -> None:
        sys.stderr.write(f"[stall_watchdog] {msg}\n")
        sys.stderr.flush()

    # -- the check --------------------------------------------------------
    def check_dir(self, turns_dir: str, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        killed = []
        for pid, slot in sorted(scan(turns_dir).items()):
            key = (turns_dir, pid)
            if key in self._handled:
                continue
            idle = now - slot["newest_mtime"]
            if idle <= self.timeout:
                if self.verbose:
                    self.say(f"ok pid={pid} idle={idle:.0f}s "
                             f"seeds={sorted(slot['seeds'])}")
                continue
            if not pid_alive(pid):
                # Finished or already dead. Its files are final; leave them for
                # grading. Remember it so we do not re-inspect every poll.
                self._handled.add(key)
                continue
            start = pid_start_time(pid)
            if start is not None and start > slot["newest_mtime"] + _PID_REUSE_SLACK_S:
                if pid not in self._warned_reuse:
                    self._warned_reuse.add(pid)
                    self.say(f"pid={pid} started after its newest turn file — "
                             f"PID reuse, refusing to kill an unrelated process")
                self._handled.add(key)
                continue
            killed.append(self.handle_stall(turns_dir, pid, slot, idle))
            self._handled.add(key)
        return killed

    def handle_stall(self, turns_dir: str, pid: int, slot: dict, idle: float) -> dict:
        # Evidence first: read the last recorded turn for every seed this PID
        # owns BEFORE anything is killed or moved.
        seeds = {}
        for seed, s in sorted(slot["seeds"].items()):
            newest = max(s["files"], key=lambda p: os.stat(p).st_mtime)
            seeds[str(seed)] = {
                "idle_s": round(time.time() - s["newest_mtime"], 1),
                "newest_file": os.path.basename(newest),
                "last_turn": last_turn_record(newest),
            }
        # Heuristic, and labelled as one in the record: when the process wedges,
        # every seed stops at once, so the seed that went quiet FIRST is the
        # likeliest culprit. It can also be a seed that simply finished early —
        # which is why the full per-seed table is logged alongside it rather
        # than this single number being the answer.
        stalled_seeds = sorted(slot["seeds"], key=lambda s: slot["seeds"][s]["newest_mtime"])

        self.say(f"STALL turns_dir={turns_dir} pid={pid} idle={idle:.0f}s "
                 f"(timeout={self.timeout:.0f}s) seeds={sorted(slot['seeds'])} "
                 f"likely_hung_seed={stalled_seeds[0]}"
                 + (" [DRY RUN]" if self.dry_run else ""))

        kill = kill_tree(pid, self.kill_grace, dry_run=self.dry_run)
        moved, qdir = self.quarantine_files(turns_dir, pid, slot["files"])

        rec = self.log(
            turns_dir, "kill",
            pid=pid,
            idle_s=round(idle, 1),
            timeout_s=self.timeout,
            seeds=sorted(slot["seeds"]),
            likely_hung_seed=stalled_seeds[0],
            per_seed=seeds,
            kill=kill,
            quarantine_dir=qdir,
            quarantined_files=[os.path.basename(p) for p in moved],
            dry_run=self.dry_run,
        )
        # Self-describing artifact next to the moved files, so the quarantine
        # dir explains itself without the log.
        if qdir and not self.dry_run:
            try:
                with open(os.path.join(qdir, "MANIFEST.json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, default=str)
            except OSError:
                pass
        self.say(f"killed pid={pid} (survivors={kill['survivors']}); "
                 f"quarantined {len(moved)} file(s) -> {qdir}")
        self.kills.append(rec)
        return rec

    def quarantine_files(self, turns_dir: str, pid: int, files: list[str]):
        """Move every turn file this PID owned out of `turns/`.

        ALL of the PID's files, not just the hung seed's: the process is dead,
        so every seed it was running is truncated mid-rollout, and any of them
        merging with a relaunch corrupts that seed's grade (§4.6). A completed
        seed's data is not lost — it is in the quarantine dir, and can be moved
        back deliberately.
        """
        if self.dry_run:
            return [], None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        qdir = os.path.join(quarantine_root(turns_dir, self.quarantine),
                            f"{stamp}_pid{pid}")
        try:
            os.makedirs(qdir, exist_ok=True)
        except OSError as exc:
            self.say(f"cannot create quarantine dir {qdir}: {exc}")
            return [], None
        moved = []
        for path in sorted(files):
            dest = os.path.join(qdir, os.path.basename(path))
            try:
                shutil.move(path, dest)
                moved.append(dest)
            except OSError as exc:
                self.say(f"cannot quarantine {path}: {exc}")
        return moved, qdir

    # -- loop -------------------------------------------------------------
    def run(self, once=False, parent_pid=None, max_seconds=None) -> int:
        stop = {"flag": False}

        def _stop(_signum, _frame):
            stop["flag"] = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _stop)
            except (ValueError, OSError):
                pass

        for d in self.turns_dirs:
            self.log(d, "armed", timeout_s=self.timeout, poll_s=self.poll,
                     parent_pid=parent_pid, dry_run=self.dry_run,
                     quarantine_root=quarantine_root(d, self.quarantine))
        self.say(f"armed on {self.turns_dirs} timeout={self.timeout:g}s "
                 f"poll={self.poll:g}s parent_pid={parent_pid}"
                 + (" [DRY RUN]" if self.dry_run else ""))

        started = time.monotonic()
        reason = "once"
        while not stop["flag"]:
            for d in self.turns_dirs:
                self.check_dir(d)
            if once:
                break
            if parent_pid is not None and not pid_alive(parent_pid):
                reason = "parent_exited"
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                reason = "max_seconds"
                break
            # Sleep in slices so a signal or a departed parent is noticed
            # promptly rather than one full poll interval later.
            slept = 0.0
            while slept < self.poll and not stop["flag"]:
                time.sleep(min(0.25, self.poll - slept))
                slept += 0.25
                if parent_pid is not None and not pid_alive(parent_pid):
                    break
        if stop["flag"]:
            reason = "signal"

        for d in self.turns_dirs:
            self.log(d, "exit", reason=reason, kills=len(self.kills))
        self.say(f"exit ({reason}); {len(self.kills)} kill(s)")
        return 3 if self.kills else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stall_watchdog",
        description="Kill and quarantine NetHack eval rollouts that stop writing turn files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--turns-dir", action="append", required=True, dest="turns_dirs",
                   metavar="DIR", help="a rollout `turns/` directory (repeatable)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="seconds of process-wide silence before a kill (default 300, per RUNBOOK)")
    p.add_argument("--poll", type=float, default=15.0, help="seconds between scans (default 15)")
    p.add_argument("--kill-grace", type=float, default=10.0,
                   help="seconds between SIGTERM and SIGKILL (default 10)")
    p.add_argument("--quarantine-dir", default=None,
                   help="where killed rollouts' turn files go "
                        "(default: <turns-dir>.stalled — must NOT be inside turns/)")
    p.add_argument("--log", default=None,
                   help="JSONL log path (default: <quarantine-dir>/stall_watchdog.jsonl)")
    p.add_argument("--parent-pid", type=int, default=None,
                   help="exit when this PID exits — this is how the watchdog is "
                        "re-armed per batch instead of outliving one")
    p.add_argument("--once", action="store_true", help="one scan pass, then exit")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="stop after this long (testing / bounded runs)")
    p.add_argument("--dry-run", action="store_true",
                   help="detect and log, but never kill or move anything")
    p.add_argument("--verbose", action="store_true", help="log healthy rollouts too")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for d in args.turns_dirs:
        q = os.path.abspath(quarantine_root(os.path.abspath(d), args.quarantine_dir))
        if q.startswith(os.path.abspath(d) + os.sep):
            sys.stderr.write(
                f"[stall_watchdog] refusing: quarantine dir {q} is INSIDE the "
                f"watched turns dir. Aggregation globs turns/*.ndjson, so "
                f"quarantined files would still be graded.\n")
            return 2
    wd = Watchdog(
        args.turns_dirs, timeout=args.timeout, poll=args.poll,
        kill_grace=args.kill_grace, quarantine=args.quarantine_dir,
        log_path=args.log, dry_run=args.dry_run, verbose=args.verbose,
    )
    return wd.run(once=args.once, parent_pid=args.parent_pid,
                  max_seconds=args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
