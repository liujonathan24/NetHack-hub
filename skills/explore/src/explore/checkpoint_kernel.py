"""A restored player's Python kernel, as it was at its checkpoint.

Prime Agent's ``fork_from`` copies the kernel snapshot it finds next to the
fork source. A player's LATEST snapshot holds what happened after the
checkpoint (its death report, its notes), so a player restored from that
checkpoint could read its own future. Instead the attempt's game server
keeps the player's recent snapshots (``KernelBuffer``) and, for each
checkpoint it mirrors, stores the newest one written BEFORE the checkpoint
(``store_for_checkpoint``). ``select()`` hands that one to the player
(``handoff_kernel``).

The kernel writes its snapshot 1.5 s after each successful cell, so a
checkpoint's kernel is the kernel after an earlier cell: names the
checkpoint's own cell (and any cell in the 1.5 s before it) assigned may be
missing. It never holds a name from after the checkpoint.

Everything here runs outside the agents' sandbox: in the game server
(``environments/nethack/explore_server.py``) and the explore service.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

KERNEL_SNAPSHOT_FILES = ("kernel-state.dill", "kernel-state.json")
#: The manifest: Prime Agent writes it after the payload, so its time is the
#: snapshot's time.
MANIFEST = "kernel-state.json"
#: Per-checkpoint kernels, under the run directory: ``<dir>/<checkpoint id>/``.
CHECKPOINT_KERNELS_DIR = "checkpoint-kernels"
ABOUT = "about.json"


def session_artifact_dir(session_file: Path) -> Path | None:
    """Where Prime Agent keeps a session's artifacts, the kernel snapshot
    among them: ``dirname(session dir)/session-artifacts/<header id>``."""
    try:
        with Path(session_file).open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
    except (OSError, ValueError):
        return None
    if not isinstance(header, dict) or header.get("type") != "session" or not header.get("id"):
        return None
    return Path(session_file).parent.parent / "session-artifacts" / str(header["id"])


def checkpoint_kernel_dir(run_dir: Path, checkpoint_id: str) -> Path:
    return Path(run_dir) / CHECKPOINT_KERNELS_DIR / str(checkpoint_id)


def attempt_player_session(run_dir: Path, attempt: int) -> str | None:
    """The session file of the agent playing ``attempt``, from the daemon's
    spawn ledger: the opener plays attempt 0, a player the attempt its
    handoff copy names. A respawned player replaces the earlier one (the
    latest record wins). None until the daemon has recorded the spawn."""
    from .service import ROLE_NAME, handoff_attempt, spawn_records

    found = None
    for record in spawn_records(Path(run_dir)):
        match = ROLE_NAME.search(str(record.get("name") or ""))
        child_file = str(record.get("child") or "")
        if match is None or match.group(1) == "orchestrator" or not child_file:
            continue
        playing = 0 if match.group(1) == "opener" else handoff_attempt(child_file)
        if playing == attempt:
            found = child_file
    return found


class KernelBuffer:
    """The last few kernel snapshots of one session, each copied with its
    file times (``<buffer>/<manifest mtime ns>/``)."""

    def __init__(self, directory: Path, keep: int = 8) -> None:
        self.dir = Path(directory)
        self.keep = keep

    def versions(self) -> list[tuple[int, Path]]:
        if not self.dir.is_dir():
            return []
        out = []
        for entry in self.dir.iterdir():
            if entry.is_dir() and entry.name.isdigit() and all((entry / name).is_file() for name in KERNEL_SNAPSHOT_FILES):
                out.append((int(entry.name), entry))
        return sorted(out)

    def watch(self, artifact_dir: Path) -> bool:
        """Copy the session's current snapshot when it is new. A snapshot
        replaced while it was copied is dropped; the next call takes the
        newer one. Returns whether a version was added."""
        manifest = Path(artifact_dir) / MANIFEST
        try:
            stamp = manifest.stat().st_mtime_ns
        except OSError:
            return False
        if not (Path(artifact_dir) / KERNEL_SNAPSHOT_FILES[0]).is_file():
            return False
        target = self.dir / str(stamp)
        if target.is_dir():
            return False
        partial = self.dir / f".{stamp}.tmp"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        try:
            for name in KERNEL_SNAPSHOT_FILES:
                shutil.copy2(Path(artifact_dir) / name, partial / name)
            if manifest.stat().st_mtime_ns != stamp:
                raise OSError("snapshot replaced during the copy")
        except OSError:
            shutil.rmtree(partial, ignore_errors=True)
            return False
        os.replace(partial, target)
        for _, old in self.versions()[: -self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        return True

    def before(self, at: float) -> tuple[int, Path] | None:
        """The newest buffered snapshot written at or before ``at`` (epoch s)."""
        limit = int(at * 1_000_000_000)
        earlier = [version for version in self.versions() if version[0] <= limit]
        return earlier[-1] if earlier else None


def store_for_checkpoint(run_dir: Path, checkpoint_id: str, buffer: KernelBuffer | None, created_at: float, session_file: str | None) -> bool:
    """Keep the kernel the player had when ``checkpoint_id`` was written
    (``about.json`` says which snapshot, or why there is none)."""
    out = checkpoint_kernel_dir(run_dir, checkpoint_id)
    out.mkdir(parents=True, exist_ok=True)
    chosen = buffer.before(created_at) if buffer is not None else None
    about = {"checkpoint": str(checkpoint_id), "checkpoint_at": created_at, "session_file": session_file}
    if chosen is None:
        about.update(ok=False, reason="no kernel snapshot written before the checkpoint" if session_file else "the attempt's player session is not known yet")
    else:
        stamp, source = chosen
        for name in KERNEL_SNAPSHOT_FILES:
            shutil.copy2(source / name, out / name)
        about.update(ok=True, snapshot_at=stamp / 1_000_000_000)
    (out / ABOUT).write_text(json.dumps(about, indent=2) + "\n", encoding="utf-8")
    return bool(about["ok"])


def handoff_kernel(run_dir: Path, checkpoint_id: str, handoff: Path) -> tuple[bool, str]:
    """Put the checkpoint's kernel where Prime Agent looks for the handoff
    copy's (``session_artifact_dir(handoff)``), so the player's fork gets it.

    The files get the CURRENT time on purpose (a plain copy): Prime Agent
    copies a fork source's snapshot only when it is at least as new as the
    source conversation's last Python result, to skip a snapshot older than
    the source's latest kernel. Here the older kernel is the point: the
    handoff carries the whole source conversation and the daemon cuts it at
    the checkpoint, but the kernel must stop at the checkpoint too."""
    source = checkpoint_kernel_dir(run_dir, checkpoint_id)
    target = session_artifact_dir(handoff)
    if target is None:
        return False, f"handoff {handoff} has no session header"
    # Every handoff of one source conversation shares this directory (it is
    # named by the source's session id), so an earlier handoff's kernel goes
    # first: a checkpoint without a kernel must not inherit another's.
    for name in KERNEL_SNAPSHOT_FILES:
        (target / name).unlink(missing_ok=True)
    if not all((source / name).is_file() for name in KERNEL_SNAPSHOT_FILES):
        try:
            reason = json.loads((source / ABOUT).read_text(encoding="utf-8")).get("reason") or "no kernel stored"
        except (OSError, ValueError):
            reason = "no kernel stored for this checkpoint"
        return False, reason
    target.mkdir(parents=True, exist_ok=True)
    for name in KERNEL_SNAPSHOT_FILES:
        shutil.copyfile(source / name, target / name)
    return True, ""
