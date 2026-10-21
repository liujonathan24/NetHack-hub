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
import os
import re
import shutil
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
from .checkpoint_kernel import KERNEL_SNAPSHOT_FILES, handoff_kernel, session_artifact_dir  # noqa: F401
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
    "NotTheOrchestrator",
]

DIRECTIVE_BLOCK_FORMAT = "[Strategy for this attempt: {directive}]"
BLIND_DIRECTIVE = "Continue playing."
MACHINE_NAME = "explore"


class NotTheOrchestrator(PermissionError):
    """Raised by the explore service when a session other than the
    orchestrator (or the root) reads the ledger or selects a checkpoint."""


class ExploreServiceError(RuntimeError):
    """A remote explore call failed in the service (message and type kept)."""


def _remote(call: str, **kwargs: Any) -> Any:
    """Run ``call`` in the run's explore service (``EXPLORE_SERVICE_URL``).

    A sandboxed kernel cannot see the run directory, so the ledger, archive
    and game servers live in the service; this sends it the call and returns
    its result, re-raising the service's error type where it is ours."""
    import urllib.request

    url = os.environ["EXPLORE_SERVICE_URL"].rstrip("/") + "/call"
    # The caller's identity, attached by the client, never by the model: the
    # kernel's private session directory (the sandbox shows each kernel only
    # its own). The service maps it to the factory state that spawned it.
    caller = os.environ.get("RLM_SESSION_DIR", "")
    body = json.dumps({"call": call, "kwargs": kwargs, "caller": caller}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        reply = json.loads(response.read())
    if reply.get("ok"):
        return reply.get("result")
    kind, message = reply.get("type"), reply.get("error") or "explore service error"
    if kind == "NotTheOrchestrator":
        raise NotTheOrchestrator(message)
    if kind in ("ValueError", "KeyError", "FileNotFoundError"):
        raise ValueError(message)
    raise ExploreServiceError(f"{kind}: {message}")


def _use_service(run_dir: Any, adapter: Any) -> bool:
    return bool(os.environ.get("EXPLORE_SERVICE_URL")) and run_dir is None and adapter is None



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


def _settle_previous(ledger: Ledger, adapter: EnvFork | None) -> None:
    """Make sure the latest attempt's GAME is over and its outcome recorded
    before the orchestrator reads anything. A player's turn can end while its
    game calls keep running in the kernel; the adapter's ``settle`` waits for
    the game (or a quiet period), stops the server, and writes the outcome."""
    previous = ledger.attempts_started() - 1
    if previous < 0 or any(outcome.attempt == previous for outcome in ledger.outcomes()):
        return
    adapter = adapter or load_adapter(run_dir=ledger.dir)
    settle = getattr(adapter, "settle", None)
    if settle is not None:
        settle(attempt=previous, run_dir=ledger.dir)


def leaderboard(limit: int = 30, *, run_dir: str | Path | None = None, adapter: EnvFork | None = None) -> str:
    """The leaderboard text: one line per checkpoint with its harness-computed
    numbers, how many attempts started there, and how they ended. Orchestrator
    only (the explore service checks the caller's session). Waits for the
    previous attempt's game to be over first."""
    if _use_service(run_dir, adapter):
        return _remote("leaderboard", limit=limit)
    ledger = Ledger(run_dir)
    _settle_previous(ledger, adapter)
    return leaderboard_text(ledger, limit=limit)


def rows(*, run_dir: str | Path | None = None, adapter: EnvFork | None = None) -> list[dict[str, Any]]:
    """The leaderboard as data. Orchestrator only."""
    if _use_service(run_dir, adapter):
        return _remote("rows")
    ledger = Ledger(run_dir)
    _settle_previous(ledger, adapter)
    return leaderboard_rows(ledger)


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

    ``next`` carries only what the player's spawn needs: ``checkpoint``,
    ``attempt`` (the new player's index), ``session_ref`` (the fork source,
    or null when no conversation is known: the player then starts with fresh
    memory), and ``mcp_servers``. The block stays short on purpose: the
    orchestrator model copies it into its final answer, and a long block
    gets mistyped. How the cut went and the env handle are in the ledger's
    ``selection`` record. ``blind`` replaces
    the directive with the neutral "Continue playing." (the A1/A2 arms).
    A directive that fails ``validate_directive`` raises ``ValueError``: fix
    it and call again.
    """
    if _use_service(run_dir, adapter):
        return _remote("select", checkpoint_id=checkpoint_id, directive=directive, blind=blind)
    ledger = Ledger(run_dir)
    adapter = adapter or load_adapter(run_dir=ledger.dir)
    _settle_previous(ledger, adapter)
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
    # A later select() in the same round (the orchestrator changed its mind,
    # or a retried orchestrator chose again) SUPERSEDES the earlier one: its
    # game server stops and its handoff goes, so only the latest decision
    # block can start a player; an earlier block fails at spawn.
    earlier = [r for r in ledger.records() if r.get("kind") == "selection" and r.get("attempt") == attempt]
    if earlier:
        finalize = getattr(adapter, "finalize", None)
        if finalize is not None:
            finalize(ledger.dir, attempt, reason="superseded by a later select")
        for stale in (ledger.dir / "handoff").glob(f"attempt-{attempt}-*.jsonl"):
            stale.unlink()
    restored = adapter.restore(chosen.env_handle, attempt=attempt, run_dir=ledger.dir)
    session_file = ledger.player_session(chosen.attempt)
    if session_file is None:
        cut = SessionCut(ok=False, session_file="", reason=f"no player session recorded for attempt {chosen.attempt}")
    else:
        cut = cut_session(session_file, chosen.call)
    # The player forks from a PRIVATE copy of the cut conversation, which this
    # attempt's game server deletes at the player's first game call. A decision
    # block re-emitted from an orchestrator's memory therefore names a copy that
    # is gone, and its player's spawn fails before any model call instead of
    # replaying a finished attempt.
    session_ref = cut.fork_source()
    kernel_snapshot, kernel_reason = False, "no forked conversation"
    if session_ref:
        source = Path(session_ref["session_file"] if isinstance(session_ref, dict) else session_ref)
        handoff = ledger.dir / "handoff" / f"attempt-{attempt}-{len(earlier) + 1}.jsonl"
        handoff.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, handoff)
        # The player's kernel is the one it had AT the checkpoint, which the
        # attempt's game server stored (checkpoint_kernel), never the source
        # player's latest: that one knows how the attempt went on.
        kernel_snapshot, kernel_reason = handoff_kernel(ledger.dir, chosen.id, handoff)
        session_ref = {**session_ref, "session_file": str(handoff)} if isinstance(session_ref, dict) else str(handoff)
        arm = getattr(adapter, "arm_handoff", None)
        if arm is not None:
            arm(ledger.dir, attempt, handoff)
    # The block the machine binds stays small: the child's final answer is
    # the binding lane and hosts cap it, so only what the player's spawn
    # needs rides here. The cut's details go to the ledger.
    decision = {
        "next": {
            "checkpoint": chosen.id,
            "attempt": attempt,
            "session_ref": session_ref,
            "mcp_servers": list(restored.get("mcp_servers") or []),
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
            "memory": cut.as_dict(),
            "kernel_snapshot": kernel_snapshot,
            "kernel_reason": kernel_reason,
            "env_handle": restored.get("env_handle", chosen.env_handle),
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
    if _use_service(run_dir, None):
        return _remote("select_fixed")
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
    max_resumes: int = 3,
) -> None:
    """Watch the run to its end from the root kernel, beside
    ``rlm.factory.run``: resume a paused run (at most ``max_resumes``
    times), and record every spawned player's session file into the ledger.

    In service mode the service reads players from the daemon's spawn
    ledger instead, and this reports the run's final status to it
    (``run_ended``): the root ends its turn after starting the run (a child's
    task prompt waits for the parent turn that spawned it, so a root that
    watched in its own turn would delay every child by a minute), and the
    driver waits for this report."""
    from rlm import rlm as _rlm  # the kernel namespace; imported late so tests can patch it

    remote = bool(os.environ.get("EXPLORE_SERVICE_URL")) and run_dir is None
    ledger = None if remote else Ledger(run_dir)
    seen: set[str] = set()
    resumes = 0
    while True:
        snapshot = await _rlm.factory.watch(run_id, poll_seconds)
        state = snapshot.get("state")
        if state == "paused" and resumes < max_resumes:
            resumes += 1
            await _rlm.factory.resume(run_id)
            continue
        for node in [] if ledger is None else snapshot.get("nodes") or []:
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
        if state not in ("running", "stopping") and not (state == "paused" and resumes < max_resumes):
            break
        if not snapshot.get("changed", True):
            await asyncio.sleep(0)
    if remote:
        status = await _rlm.factory.status(run_id)
        _remote(
            "run_ended",
            run_id=run_id,
            state=status.get("state"),
            resumes=resumes,
            pause_reason=status.get("pause_reason"),
            usage=status.get("usage"),
            events=status.get("events"),
        )


async def run(machine: str = MACHINE_NAME, *, run_dir: str | Path | None = None, name: str | None = None) -> dict[str, Any]:
    """Start the explore machine and the player watcher. Returns the
    ``rlm.factory.run`` result; the watcher task rides ``result["watcher"]``."""
    from rlm import rlm as _rlm

    remote = bool(os.environ.get("EXPLORE_SERVICE_URL")) and run_dir is None
    directory = None
    if not remote:
        directory = Ledger(run_dir).dir
        directory.mkdir(parents=True, exist_ok=True)
    result = await _rlm.factory.run(machine, name=name)
    task = asyncio.create_task(watch_players(result["run_id"], run_dir=directory))
    result["watcher"] = task
    return result
