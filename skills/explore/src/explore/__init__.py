"""Checkpointed exploration (the paper's Prime-Agent-Explore) for the kernel.

The loop: a player plays and the environment saves checkpoints; an
orchestrator reads the leaderboard, selects a checkpoint, and writes a
directive; a new player starts from that checkpoint's game AND the player's
own conversation at that point. The factory machine ``explore`` runs the
loop; this package is the orchestrator's and the environment's API to the
run's ledger.

Who calls what:

* the environment (or harness) calls ``checkpoint(...)`` when it saves and
  ``outcome(...)`` when a player exits;
* the orchestrator child calls ``leaderboard()``, then ``select(...)`` or
  ``stop()``, and ends its answer with the fenced json block ``answer()``
  renders;
* the root session calls ``run(...)``: it starts the machine and records
  each spawned player's session file into the ledger, which is how
  ``select`` finds the conversation a checkpoint belongs to.

Every number the orchestrator reads comes from the ledger's checkpoint
scores, which the environment wrote. No text the orchestrator writes becomes
a number: only a checkpoint id and a directive cross into the run.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from .adapters import EnvFork, ScriptedAdapter, load_adapter
from .ledger import (
    Checkpoint,
    Ledger,
    Outcome,
    PlayerRecord,
    latest_checkpoint_of_attempt,
    leaderboard_rows,
    leaderboard_text,
    run_dir,
)
from .session_cut import SessionCut, cut_session

__all__ = [
    "Checkpoint",
    "EnvFork",
    "Ledger",
    "Outcome",
    "PlayerRecord",
    "ScriptedAdapter",
    "SessionCut",
    "answer",
    "checkpoint",
    "cut_session",
    "leaderboard",
    "load_adapter",
    "outcome",
    "rows",
    "run",
    "run_dir",
    "select",
    "select_fixed",
    "stop",
    "validate_directive",
    "watch_players",
]

DIRECTIVE_BLOCK_FORMAT = "[ORCHESTRATOR DIRECTIVE for this attempt: {directive}]"
BLIND_DIRECTIVE = "Continue playing."
MACHINE_NAME = "explore"

# -- the environment's side ------------------------------------------------


def checkpoint(
    *,
    attempt: int,
    score: dict[str, Any],
    env_handle: Any,
    call: int | None,
    parent: str | None = None,
    label: str = "",
    note: str = "",
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Record one saved checkpoint. ``score`` keeps only its numeric fields.
    ``call`` is the game call number the save happened in (the one the
    environment stamps as ``[call#N]`` on that call's result)."""
    record = Ledger(run_dir).add_checkpoint(
        attempt=attempt, score=score, env_handle=env_handle, call=call, parent=parent, label=label, note=note
    )
    return record.record()


def outcome(
    *,
    attempt: int,
    result: str,
    checkpoint_from: str | None,
    reason: str = "",
    final_score: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Record how an attempt ended: ``died``, ``censored``, ``budget``, or
    ``alive``. Death and censoring stay different results."""
    record = Ledger(run_dir).add_outcome(
        attempt=attempt, result=result, checkpoint_from=checkpoint_from, reason=reason, final_score=final_score
    )
    return record.record()


# -- the orchestrator's side -----------------------------------------------


def leaderboard(limit: int = 30, *, run_dir: str | Path | None = None) -> str:
    """The leaderboard text: one line per checkpoint with its harness-computed
    numbers, how many attempts started there, and how they ended."""
    return leaderboard_text(Ledger(run_dir), limit=limit)


def rows(*, run_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """The leaderboard as data."""
    return leaderboard_rows(Ledger(run_dir))


_KEY_TOKEN_RE = re.compile(
    r"\b(press|key|keys|keystroke|type the letter|action:)\b|ctrl-[a-z]|<esc>|'[^']'",
    re.IGNORECASE,
)


def validate_directive(directive: str, *, max_sentences: int = 2, max_chars: int = 300) -> list[str]:
    """Directives are strategy, never keystrokes: at most two sentences, at
    most 300 characters, no literal key or action tokens. Returns the list of
    problems (empty when the directive is acceptable)."""
    problems: list[str] = []
    text = directive.strip()
    if len(text) > max_chars:
        problems.append(f"directive has {len(text)} characters; the limit is {max_chars}")
    sentences = [part for part in re.split(r"[.!?]+(?:\s|$)", text) if part.strip()]
    if len(sentences) > max_sentences:
        problems.append(f"directive has {len(sentences)} sentences; the limit is {max_sentences}")
    match = _KEY_TOKEN_RE.search(text)
    if match:
        problems.append(f"directive names a key or action ({match.group(0)!r}); give strategy, not keystrokes")
    return problems


def select(
    checkpoint_id: str,
    directive: str = "",
    *,
    blind: bool = False,
    run_dir: str | Path | None = None,
    adapter: EnvFork | None = None,
) -> dict[str, Any]:
    """Choose the checkpoint the next player starts from.

    Restores the checkpoint's game through the adapter, finds the player
    conversation the checkpoint belongs to and where to cut it, and returns
    the two output ports the explore machine binds:

    ``{"decision": {"next": {...}}, "directive": "<text>"}``

    ``next`` carries ``checkpoint``, ``attempt`` (the new player's index),
    ``session_ref`` (the fork source, or null when no conversation is known:
    the player then starts with fresh memory), ``env_handle``,
    ``mcp_servers``, and ``memory`` (how the cut went). ``blind`` replaces
    the directive with the neutral "Continue playing." (the A1/A2 arms).
    A directive that fails ``validate_directive`` raises ``ValueError``: fix
    it and call again.
    """
    ledger = Ledger(run_dir)
    chosen = ledger.checkpoint(str(checkpoint_id))
    if chosen is None:
        known = ", ".join(entry.id for entry in ledger.checkpoints()) or "none"
        raise ValueError(f"unknown checkpoint {checkpoint_id!r}; the ledger has: {known}")
    if blind:
        directive_text = BLIND_DIRECTIVE
    else:
        directive_text = directive.strip()
        problems = validate_directive(directive_text)
        if problems:
            raise ValueError("; ".join(problems))
    attempt = ledger.attempts_started()
    adapter = adapter or load_adapter(run_dir=ledger.dir)
    restored = adapter.restore(chosen.env_handle, attempt=attempt, run_dir=ledger.dir)
    session_file = ledger.player_session(chosen.attempt)
    if session_file is None:
        cut = SessionCut(ok=False, session_file="", reason=f"no player session recorded for attempt {chosen.attempt}")
    else:
        cut = cut_session(session_file, chosen.call)
    decision = {
        "next": {
            "checkpoint": chosen.id,
            "attempt": attempt,
            "session_ref": cut.fork_source(),
            "env_handle": restored.get("env_handle", chosen.env_handle),
            "mcp_servers": list(restored.get("mcp_servers") or []),
            "directive": directive_text,
            "memory": cut.as_dict(),
        }
    }
    ledger.append(
        {
            "kind": "selection",
            "attempt": attempt,
            "checkpoint": chosen.id,
            "directive": directive_text,
            "blind": blind,
            "memory_ok": cut.ok,
        }
    )
    return {"decision": decision, "directive": DIRECTIVE_BLOCK_FORMAT.format(directive=directive_text)}


def stop(reason: str = "") -> dict[str, Any]:
    """End the run. The decision has NO ``next`` key, so the machine's guard
    ``decision.next exists`` fails (a ``null`` value would still exist)."""
    return {"decision": {"stop": True, "reason": reason}, "directive": ""}


def answer(result: dict[str, Any]) -> str:
    """The fenced json block the orchestrator's final answer must end with."""
    return "```json\n" + json.dumps(result, sort_keys=True) + "\n```"


def select_fixed(*, run_dir: str | Path | None = None) -> str | None:
    """The A2 rule: the latest checkpoint the previous attempt wrote, else
    the latest checkpoint overall, else None (nothing to resume)."""
    ledger = Ledger(run_dir)
    checkpoints = ledger.checkpoints()
    if not checkpoints:
        return None
    previous = ledger.attempts_started() - 1
    latest = latest_checkpoint_of_attempt(checkpoints, previous)
    if latest is None:
        latest = max(checkpoints, key=lambda entry: entry.created_at)
    return latest.id


# -- the root session's side -----------------------------------------------


def _player_attempt(state_id: str, instance_index: int, *, player_state: str, opener_state: str) -> int | None:
    if state_id == opener_state:
        return 0
    if state_id == player_state:
        return instance_index + 1
    return None


async def watch_players(
    run_id: str,
    *,
    run_dir: str | Path | None = None,
    player_state: str = "player",
    opener_state: str = "opener",
    poll_seconds: float = 5.0,
) -> None:
    """Record every spawned player's session file into the ledger until the
    run ends. Runs in the root kernel beside ``rlm.factory.run``."""
    from rlm import rlm as _rlm  # the kernel namespace; imported late so tests can patch it

    ledger = Ledger(run_dir)
    seen: set[str] = set()
    while True:
        snapshot = await _rlm.factory.watch(run_id, poll_seconds)
        for node in snapshot.get("nodes") or []:
            state_id = node.get("id")
            for instance in node.get("instances") or []:
                child = instance.get("child")
                session_file = instance.get("session_file")
                if not child or child in seen:
                    continue
                attempt = _player_attempt(
                    str(state_id), int(instance.get("index") or 0), player_state=player_state, opener_state=opener_state
                )
                if attempt is None:
                    continue
                seen.add(child)
                ledger.add_player(attempt=attempt, child=str(child), session_file=session_file)
        if snapshot.get("state") not in ("running", "stopping", "paused"):
            return
        if not snapshot.get("changed", True):
            await asyncio.sleep(0)


async def run(machine: str = MACHINE_NAME, *, run_dir: str | Path | None = None, name: str | None = None) -> dict[str, Any]:
    """Start the explore machine and the player watcher. Returns the
    ``rlm.factory.run`` result; the watcher task rides ``result["watcher"]``."""
    from rlm import rlm as _rlm

    directory = Ledger(run_dir).dir
    directory.mkdir(parents=True, exist_ok=True)
    result = await _rlm.factory.run(machine, name=name)
    task = asyncio.create_task(watch_players(result["run_id"], run_dir=directory))
    result["watcher"] = task
    return result
