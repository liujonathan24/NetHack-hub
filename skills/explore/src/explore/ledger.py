"""The checkpoint ledger: the one record of a run that is not model-written.

A run directory holds one ``ledger.jsonl``. Each line is one record:

* ``{"kind": "checkpoint", ...}``: written by the environment when it saves.
  Every number in ``score`` is harness-computed. ``label`` and ``note`` are
  text the model may have written; they are displayed and never measured.
* ``{"kind": "outcome", ...}``: written by the harness when a player exits.
  ``result`` is one of ``died``, ``censored``, ``budget``, ``alive``.
* ``{"kind": "player", ...}``: written by the machine driver when the factory
  spawns a player: the attempt index and the player's session file. The
  select() call reads it to find the conversation a checkpoint belongs to.

Appends are single ``write`` calls of one line, so concurrent writers (the
env process and the kernel) never interleave inside a line.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

LEDGER_FILE = "ledger.jsonl"
RUN_DIR_ENV = "EXPLORE_RUN_DIR"
OUTCOMES: tuple[str, ...] = ("died", "censored", "budget", "alive")


def run_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The run directory: the argument, else ``$EXPLORE_RUN_DIR``."""
    if explicit is not None:
        return Path(explicit)
    value = os.environ.get(RUN_DIR_ENV)
    if not value:
        raise RuntimeError(
            f"no explore run directory: pass run_dir= or set {RUN_DIR_ENV}"
        )
    return Path(value)


def _number(value: Any) -> int | float | None:
    """A JSON number, or None. Booleans and strings are not numbers here."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def clean_score(score: Any) -> dict[str, int | float]:
    """Keep only the numeric fields of a score dict (integrity rule 1: no
    model-written text can become a number)."""
    if not isinstance(score, dict):
        return {}
    out: dict[str, int | float] = {}
    for key, value in score.items():
        number = _number(value)
        if isinstance(key, str) and number is not None:
            out[key] = number
    return out


@dataclass
class Checkpoint:
    id: str
    attempt: int
    created_at: float
    score: dict[str, int | float] = field(default_factory=dict)
    env_handle: Any = None
    call: int | None = None
    parent: str | None = None
    label: str = ""
    note: str = ""

    def record(self) -> dict[str, Any]:
        return {"kind": "checkpoint", **asdict(self)}


@dataclass
class Outcome:
    attempt: int
    result: str
    checkpoint_from: str | None
    created_at: float
    reason: str = ""
    final_score: dict[str, int | float] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"kind": "outcome", **asdict(self)}


@dataclass
class PlayerRecord:
    attempt: int
    child: str
    session_file: str | None
    created_at: float

    def record(self) -> dict[str, Any]:
        return {"kind": "player", **asdict(self)}


class Ledger:
    """Read and append the run's ledger."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self.dir = run_dir(directory)
        self.path = self.dir / LEDGER_FILE

    # -- reads ---------------------------------------------------------------

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn tail never poisons the ledger
            if isinstance(record, dict):
                out.append(record)
        return out

    def checkpoints(self) -> list[Checkpoint]:
        out: list[Checkpoint] = []
        for record in self.records():
            if record.get("kind") != "checkpoint":
                continue
            out.append(
                Checkpoint(
                    id=str(record.get("id")),
                    attempt=int(record.get("attempt") or 0),
                    created_at=float(record.get("created_at") or 0.0),
                    score=clean_score(record.get("score")),
                    env_handle=record.get("env_handle"),
                    call=(int(record["call"]) if _number(record.get("call")) is not None else None),
                    parent=(str(record["parent"]) if record.get("parent") is not None else None),
                    label=str(record.get("label") or ""),
                    note=str(record.get("note") or ""),
                )
            )
        return out

    def checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        for checkpoint in self.checkpoints():
            if checkpoint.id == str(checkpoint_id):
                return checkpoint
        return None

    def outcomes(self) -> list[Outcome]:
        out: list[Outcome] = []
        for record in self.records():
            if record.get("kind") != "outcome":
                continue
            out.append(
                Outcome(
                    attempt=int(record.get("attempt") or 0),
                    result=str(record.get("result") or ""),
                    checkpoint_from=(
                        str(record["checkpoint_from"]) if record.get("checkpoint_from") is not None else None
                    ),
                    created_at=float(record.get("created_at") or 0.0),
                    reason=str(record.get("reason") or ""),
                    final_score=clean_score(record.get("final_score")),
                )
            )
        return out

    def players(self) -> list[PlayerRecord]:
        out: list[PlayerRecord] = []
        for record in self.records():
            if record.get("kind") != "player":
                continue
            session_file = record.get("session_file")
            out.append(
                PlayerRecord(
                    attempt=int(record.get("attempt") or 0),
                    child=str(record.get("child") or ""),
                    session_file=str(session_file) if session_file else None,
                    created_at=float(record.get("created_at") or 0.0),
                )
            )
        return out

    def player_session(self, attempt: int) -> str | None:
        """The session file of the player that played ``attempt`` (the last
        record wins: a retried attempt has two)."""
        found = None
        for player in self.players():
            if player.attempt == attempt and player.session_file:
                found = player.session_file
        return found

    def attempts_started(self) -> int:
        """The number of attempts the ledger knows about."""
        attempts = {player.attempt for player in self.players()}
        attempts.update(outcome.attempt for outcome in self.outcomes())
        attempts.update(checkpoint.attempt for checkpoint in self.checkpoints())
        return max(attempts) + 1 if attempts else 0

    def attempts_from(self, checkpoint_id: str) -> int:
        """How many attempts started from ``checkpoint_id``."""
        return sum(1 for outcome in self.outcomes() if outcome.checkpoint_from == str(checkpoint_id))

    def next_checkpoint_id(self) -> str:
        return f"c{len(self.checkpoints())}"

    # -- appends -------------------------------------------------------------

    def append(self, record: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def add_checkpoint(
        self,
        *,
        attempt: int,
        score: dict[str, Any],
        env_handle: Any,
        call: int | None,
        parent: str | None,
        label: str = "",
        note: str = "",
        checkpoint_id: str | None = None,
        now: float | None = None,
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            id=checkpoint_id or self.next_checkpoint_id(),
            attempt=int(attempt),
            created_at=time.time() if now is None else now,
            score=clean_score(score),
            env_handle=env_handle,
            call=(int(call) if call is not None else None),
            parent=(str(parent) if parent is not None else None),
            label=str(label or ""),
            note=str(note or ""),
        )
        self.append(checkpoint.record())
        return checkpoint

    def add_outcome(
        self,
        *,
        attempt: int,
        result: str,
        checkpoint_from: str | None,
        reason: str = "",
        final_score: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> Outcome:
        if result not in OUTCOMES:
            raise ValueError(f"result must be one of {OUTCOMES}, got {result!r}")
        outcome = Outcome(
            attempt=int(attempt),
            result=result,
            checkpoint_from=(str(checkpoint_from) if checkpoint_from is not None else None),
            created_at=time.time() if now is None else now,
            reason=str(reason or ""),
            final_score=clean_score(final_score),
        )
        self.append(outcome.record())
        return outcome

    def add_player(self, *, attempt: int, child: str, session_file: str | None, now: float | None = None) -> PlayerRecord:
        player = PlayerRecord(
            attempt=int(attempt),
            child=str(child),
            session_file=str(session_file) if session_file else None,
            created_at=time.time() if now is None else now,
        )
        self.append(player.record())
        return player


# -- the leaderboard --------------------------------------------------------

#: The score keys the leaderboard shows, in order, with their column labels.
LEADERBOARD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dlvl", "Dlvl"),
    ("xl", "XL"),
    ("hp", "HP"),
    ("max_hp", "maxHP"),
    ("hunger", "hunger"),
    ("score", "score"),
    ("gameturn", "turn"),
    ("progress", "progress"),
)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def leaderboard_rows(ledger: Ledger) -> list[dict[str, Any]]:
    """One row per checkpoint: id, parent, attempt, the numeric score fields,
    how many attempts started there and how they ended, and the label."""
    outcomes = ledger.outcomes()
    rows: list[dict[str, Any]] = []
    for checkpoint in ledger.checkpoints():
        from_here = [outcome for outcome in outcomes if outcome.checkpoint_from == checkpoint.id]
        rows.append(
            {
                "id": checkpoint.id,
                "parent": checkpoint.parent,
                "attempt": checkpoint.attempt,
                "call": checkpoint.call,
                **checkpoint.score,
                "attempts_from": len(from_here),
                "outcomes": sorted(outcome.result for outcome in from_here),
                "label": checkpoint.label,
            }
        )
    return rows


def leaderboard_text(ledger: Ledger, limit: int = 30) -> str:
    """The leaderboard the orchestrator reads. Every number comes from the
    ledger's checkpoint scores; the label is the only model-written text and
    it is marked as such."""
    rows = leaderboard_rows(ledger)
    if not rows:
        return "No checkpoints yet."
    columns = [(key, title) for key, title in LEADERBOARD_COLUMNS if any(key in row for row in rows)]
    header = ["id", "parent", "attempt"] + [title for _, title in columns] + ["tried", "ended", "label (model text)"]
    lines = [" | ".join(header)]
    for row in rows[-limit:]:
        cells = [row["id"], row["parent"] or "-", str(row["attempt"])]
        cells += [_fmt(row.get(key)) for key, _ in columns]
        cells.append(str(row["attempts_from"]))
        cells.append(",".join(row["outcomes"]) or "-")
        cells.append(row["label"][:40])
        lines.append(" | ".join(cells))
    total = ledger.attempts_started()
    lines.append(f"{len(rows)} checkpoints . {total} attempts started . {len(ledger.outcomes())} finished")
    return "\n".join(lines)


def latest_checkpoint_of_attempt(checkpoints: Iterable[Checkpoint], attempt: int) -> Checkpoint | None:
    """The last checkpoint ``attempt`` wrote (the A2 fixed-rule target)."""
    latest = None
    for checkpoint in checkpoints:
        if checkpoint.attempt == attempt and (latest is None or checkpoint.created_at >= latest.created_at):
            latest = checkpoint
    return latest
