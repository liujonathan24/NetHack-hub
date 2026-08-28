"""Checkpoint integrity: a broken checkpoint must never present as a death.

WHY THIS MODULE EXISTS
----------------------
Three E15 forced-revive rollouts recorded ``died=True`` for a hero that was at
full HP. The engine had printed::

    Cannot open file ".N" for level N (errno 2).
    Probably someone removed it.

``tricked_fileremoved`` (NetHack ``src/save.c:487``) prints exactly those two
lines and then calls ``done(TRICKED)`` — an instant game-over that ignores HP.
``goto_level`` (``src/do.c:1462``) reaches it whenever the in-memory flag
``level_info[ledger].linfo_flags & LFILE_EXISTS`` is set but the on-disk level
file is gone. Downstream, that arrives as: NLE ``terminated``, zeroed blstats
(HP 0/0, Dlvl 0) — byte-for-byte the shape of a monster kill. The harness
scored it as a death and wrote it into the dataset.

A corrupted checkpoint is an ENGINE fault, not a game outcome. Every path that
can produce one therefore raises :class:`CheckpointIntegrityError` instead of
letting the run continue as a death. Detection is two-layered:

* PROACTIVE — :func:`assert_dungeon_on_disk` asks the engine whether any ledger
  is flagged as having a level file that is not there. This fires while the
  hero is still alive, before ``goto_level`` can convert it into a game-over,
  and is the check to run right after any restore/resume.
* REACTIVE — :func:`tricked_marker_in` / :func:`engine_reported_tricked` catch
  the game-over itself, for corruption arriving from a path we did not guard.

Both are cheap; nothing here steps the engine.
"""
from __future__ import annotations

from typing import Optional

try:  # the engine defines the exception so RawEngine.restore can raise it
    from nethack_core._engine import CheckpointIntegrityError
except Exception:  # pragma: no cover - engine-less import (docs/tooling)
    class CheckpointIntegrityError(RuntimeError):
        """A restored game's heap disagrees with the level files on disk."""


#: ``how_done`` code for ``done(TRICKED)`` (NetHack ``include/hack.h``: 12).
#: This is NOT a death: it is NetHack's "this save/level data is inconsistent"
#: abort, reached from ``tricked_fileremoved`` (save.c:487) and ``trickery``
#: (restore.c:1171).
HOW_TRICKED = 12

#: Message fragments the two TRICKED paths print immediately before the
#: game-over. ``tricked_fileremoved`` prints the first pair; ``trickery`` (a
#: level file whose embedded pid/dlvl header does not match, which is what a
#: naive cross-process resume produces) prints the third.
TRICKED_MARKERS = (
    "Cannot open file",
    "Probably someone removed it",
    "Strange, this map is not as I remember it",
    "Somebody is trying some trickery here",
)


def _message_text(obs) -> str:
    """Best-effort text of an observation's message line, for any obs shape."""
    if obs is None:
        return ""
    msg = getattr(obs, "message", None)
    if msg is None and isinstance(obs, dict):
        msg = obs.get("message")
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    try:
        raw = bytes(msg)
    except Exception:
        return str(msg)
    return raw.split(b"\0")[0].decode("ascii", "replace")


def tricked_marker_in(obs) -> bool:
    """True if this observation carries one of NetHack's TRICKED banners.

    Worth checking even while the game still reports ``done == False``: the
    banner is printed several steps before the game-over actually lands, so
    this is the earliest reactive signal available.
    """
    text = _message_text(obs)
    return any(m in text for m in TRICKED_MARKERS)


def engine_reported_tricked(env) -> bool:
    """True if the underlying engine ended this game with ``done(TRICKED)``."""
    for holder in (env, getattr(env, "_engine", None), getattr(env, "engine", None)):
        if holder is None:
            continue
        try:
            if bool(holder.done) and int(holder.how_done) == HOW_TRICKED:
                return True
        except Exception:
            continue
    return False


def _raw_engine(env):
    """Walk down to the RawEngine under whatever env wrapper we were handed."""
    node = env
    for _ in range(4):
        if node is None:
            return None
        if hasattr(node, "missing_level_files") and hasattr(node, "snapshot"):
            return node
        node = getattr(node, "_engine", None) or getattr(node, "engine", None)
    return None


def missing_level_files(env) -> Optional[int]:
    """Ledgers the heap flags ``LFILE_EXISTS`` with no file on disk.

    ``None`` when the engine cannot answer (no live game, non-engine backend).
    Anything ``> 0`` is a latent TRICKED game-over waiting for a stair.
    """
    eng = _raw_engine(env)
    if eng is None:
        return None
    try:
        n = eng.missing_level_files()
    except Exception:
        return None
    return None if n < 0 else int(n)


def assert_dungeon_on_disk(env, where: str = "") -> None:
    """Raise :class:`CheckpointIntegrityError` if the dungeon has holes.

    Call this immediately after any restore/resume. It is the proactive half of
    the guard: it fires while the hero is alive, so the corruption is reported
    as an engine fault instead of being converted into a death by the next
    stair transition.
    """
    n = missing_level_files(env)
    if n:
        raise CheckpointIntegrityError(
            f"checkpoint integrity failure{f' ({where})' if where else ''}: "
            f"{n} dungeon level(s) are flagged as present in the restored game "
            f"but their level files are missing on disk; the next stair "
            f"transition would be reported as a death (done(TRICKED)) rather "
            f"than an error"
        )


def assert_not_tricked(env, obs=None, where: str = "") -> None:
    """Raise if the engine has taken (or is about to take) a TRICKED game-over.

    The reactive half of the guard. ``obs`` is optional; passing the latest
    observation catches the banner a few steps before ``done`` flips.
    """
    if engine_reported_tricked(env) or tricked_marker_in(obs):
        raise CheckpointIntegrityError(
            f"engine reported done(TRICKED){f' ({where})' if where else ''}: "
            f"NetHack aborted the game because its level data is inconsistent "
            f"with the dungeon on disk. This is a checkpoint/engine fault, not "
            f"a death, and must not be recorded as one"
        )


__all__ = [
    "CheckpointIntegrityError",
    "HOW_TRICKED",
    "TRICKED_MARKERS",
    "assert_dungeon_on_disk",
    "assert_not_tricked",
    "engine_reported_tricked",
    "missing_level_files",
    "tricked_marker_in",
]
