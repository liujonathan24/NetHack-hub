"""Where to cut a player's session so its memory matches a checkpoint.

A checkpoint is written DURING game call K (the ledger's ``call``). The
player's conversation is a list of kernel cells. Each cell's tool result is
what the model printed, and one cell can run several game calls (a loop of
moves). So the exchange the game state belongs to is found two ways, in
order:

1. the first tool result whose text carries the environment's ``[call#K]``
   marker (present whenever the model printed the observation);
2. otherwise, by time: the first tool result recorded after call K finished
   (``call_times``: call number -> epoch seconds, from the env's trace).

If that cell also ran call K+1, a cut AFTER it would give the resumed model a
transcript of actions that have not happened in the restored game. The cut
is then made BEFORE the cell, at the previous tool result: the transcript
lags the game by the calls the cell ran before the save (``lag_calls``), and
the first observation the model requests shows the true state.

The result names a session row id. The daemon's fork copies the branch from
the root to that row (inclusive), so the forked child wakes up exactly at
that exchange. This is a port of ``e16_orchestrator.seal_checkpoint_session``
without the file copy: the copy is the daemon's job now.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CALL_MARKER_RE = re.compile(r"\[call#(\d+)\]")


@dataclass
class SessionCut:
    """The fork source for one checkpoint, or the reason there is none."""

    ok: bool
    session_file: str
    entry_id: str | None = None
    method: str | None = None  # "marker" | "timestamp"
    position: str | None = None  # "after_call" | "before_batch"
    transcript_end_call: int | None = None
    lag_calls: int | None = None
    reason: str = ""
    records_total: int = 0

    def fork_source(self) -> dict[str, str] | None:
        """The ``fork_from`` wire object, or None when the cut failed."""
        if not self.ok or self.entry_id is None:
            return None
        return {"session_file": self.session_file, "entry_id": self.entry_id}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_texts(record: dict[str, Any]) -> list[str]:
    """Every text a session row's message carries (content parts, thinking,
    and the bash/ipython stdout detail)."""
    message = record.get("message") if isinstance(record, dict) else None
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    out: list[str] = []
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                for key in ("text", "thinking"):
                    if isinstance(part.get(key), str):
                        out.append(part[key])
    details = message.get("details")
    if isinstance(details, dict) and isinstance(details.get("stdout"), str):
        out.append(details["stdout"])
    return out


def record_time(record: dict[str, Any]) -> float | None:
    """A row's wall-clock time in epoch seconds: ``message.timestamp`` is
    epoch milliseconds; the row-level ``timestamp`` is ISO-8601 with a Z."""
    message = record.get("message") if isinstance(record, dict) else None
    if isinstance(message, dict) and isinstance(message.get("timestamp"), (int, float)):
        return float(message["timestamp"]) / 1000.0
    stamp = record.get("timestamp") if isinstance(record, dict) else None
    if isinstance(stamp, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return _dt.datetime.strptime(stamp, fmt).replace(tzinfo=_dt.timezone.utc).timestamp()
            except ValueError:
                continue
    return None


def _is_tool_result(record: dict[str, Any]) -> bool:
    if record.get("type") != "message":
        return False
    message = record.get("message")
    return isinstance(message, dict) and message.get("role") == "toolResult"


def load_rows(session_file: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = Path(session_file).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def cut_session(
    session_file: str | Path,
    call: int | None,
    *,
    call_times: dict[int, float] | None = None,
) -> SessionCut:
    """Find the row to fork at for a checkpoint written during ``call``."""
    path = Path(session_file)
    cut = SessionCut(ok=False, session_file=str(path))
    if not path.is_file():
        cut.reason = f"session file does not exist: {path}"
        return cut
    if call is None:
        cut.reason = "checkpoint has no call id (written outside a call?)"
        return cut
    call = int(call)
    rows = load_rows(path)
    cut.records_total = len(rows)
    if not rows or rows[0].get("type") != "session":
        cut.reason = "first record is not a session header"
        return cut
    results = [row for row in rows if _is_tool_result(row)]
    marker = f"[call#{call}]"
    next_marker = f"[call#{call + 1}]"
    times = call_times or {}
    time_k, time_next = times.get(call), times.get(call + 1)

    hit: int | None = None
    method: str | None = None
    spans_next = False
    for index, row in enumerate(results):
        texts = record_texts(row)
        if any(marker in text for text in texts):
            hit, method = index, "marker"
            spans_next = any(next_marker in text for text in texts)
            if not spans_next and time_next is not None:
                stamp = record_time(row)
                spans_next = stamp is not None and stamp >= time_next
            break
    if hit is None and time_k is not None:
        for index, row in enumerate(results):
            stamp = record_time(row)
            if stamp is not None and stamp >= time_k:
                hit, method = index, "timestamp"
                spans_next = time_next is not None and stamp >= time_next
                break
    if hit is None:
        cut.reason = f"no tool result carrying {marker} in {path.name}" + (
            "" if time_k is None else f" and none recorded after call {call}"
        )
        return cut
    cut.method = method
    if spans_next:
        if hit == 0:
            cut.reason = (
                f"call {call} was the first cell's batch and the cell ran past it; "
                "no earlier exchange to cut at"
            )
            return cut
        row = results[hit - 1]
        marks = [int(value) for text in record_texts(row) for value in CALL_MARKER_RE.findall(text)]
        stamp = record_time(row)
        by_time = [number for number, when in times.items() if stamp is not None and when <= stamp]
        end_call = max(marks) if marks else (max(by_time) if by_time else None)
        cut.position = "before_batch"
        cut.transcript_end_call = end_call
        cut.lag_calls = (call - end_call) if end_call is not None else None
    else:
        row = results[hit]
        cut.position = "after_call"
        cut.transcript_end_call = call
        cut.lag_calls = 0
    entry_id = row.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        cut.reason = "the tool result row has no id"
        return cut
    cut.entry_id = entry_id
    cut.ok = True
    return cut
