"""Checkpoint archive for checkpointed exploration (Prime-Agent-Explore).

Built on the engine's own ``EngineEnv.checkpoint()`` / ``resume()``. One
checkpoint is one directory of the run's archive::

    <archive>/c<N>/checkpoint.json   the engine checkpoint (seed, blobs, level files)
    <archive>/c<N>/meta.json         what the explore ledger reads: the numbers,
                                     the [call#K] it was taken after, why it was taken

The game process takes checkpoints itself (:class:`AutoCheckpointer`, after
every tool call), so the archive grows whether or not the model ever asks.
Triggers: every ``every_n`` calls, the first entry to a Dlvl, and an XL gain.
A trigger that lands while a prompt, menu or ``--More--`` is open is DEFERRED
to the next call: the engine refuses such a checkpoint
(``CheckpointAtPromptError``), and the level entry that matters most is also
the moment a ``--More--`` is most likely to be up.

Resume (:func:`resume`) restores everything EXCEPT the RNG blob by default:
the engine then resets to the game's seed and the random stream restarts from
there, which is the semantics the paper's PAE runs used
(``reseed_on_restore=False``: two restores of one checkpoint continue
identically; restoring never reveals the original run's future rolls).
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
from typing import Any, Optional

CHECKPOINT_FILE = "checkpoint.json"
META_FILE = "meta.json"


class CheckpointMetaError(RuntimeError):
    """A checkpoint directory without a readable meta.json."""


def _checkpoint_id(directory: pathlib.Path) -> int:
    try:
        return int(directory.name.lstrip("c"))
    except ValueError:
        return -1


def list_checkpoints(archive) -> list[pathlib.Path]:
    """The archive's complete checkpoint directories, oldest first."""
    root = pathlib.Path(archive)
    if not root.is_dir():
        return []
    found = [
        d for d in root.iterdir()
        if d.is_dir() and _checkpoint_id(d) >= 0
        and (d / CHECKPOINT_FILE).is_file() and (d / META_FILE).is_file()
    ]
    return sorted(found, key=_checkpoint_id)


def read_meta(directory) -> dict:
    path = pathlib.Path(directory) / META_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointMetaError(f"{path}: {error}") from error


def _new_directory(archive: pathlib.Path) -> pathlib.Path:
    """Claim the next free c<N> (attempts run one after another, but mkdir is
    atomic, so a racing writer just takes the next number)."""
    archive.mkdir(parents=True, exist_ok=True)
    n = max((_checkpoint_id(d) for d in archive.iterdir() if d.is_dir()), default=0) + 1
    while True:
        candidate = archive / f"c{n}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            n += 1


def _write_json(path: pathlib.Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def game_numbers(status: dict) -> dict:
    """The leaderboard numbers from a structured status dict."""
    def num(key):
        value = status.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return {
        "dlvl": num("depth"),
        "xl": num("experience_level"),
        "hp": num("hitpoints"),
        "max_hp": num("max_hitpoints"),
        "hunger_state": num("hunger_state"),
        "score": num("score"),
        "gameturn": num("time"),
        "gold": num("gold"),
        "ac": num("armor_class"),
    }


class AutoCheckpointer:
    """Decides when to checkpoint and writes the checkpoints of one attempt."""

    def __init__(self, archive, *, every_n: int, attempt_base: Optional[dict] = None,
                 start_call: int = 0, now=time.time):
        self.archive = pathlib.Path(archive)
        self.every_n = max(0, int(every_n or 0))
        self.now = now
        base = attempt_base or {}
        # The lineage's best so far: a resumed attempt starts from its
        # checkpoint's own record, so BALROG progress is never re-earned.
        self.max_dlvl = int(base.get("max_dlvl_reached") or 0)
        self.max_xl = int(base.get("max_xp_level") or 0)
        self.seen_dlvl: set[int] = set()
        self.last_xl: Optional[int] = None
        self.last_periodic = int(start_call)
        self.seeded = False
        self.pending: list[str] = []
        self.deferrals = 0
        self.errors: list[str] = []
        self.written: list[str] = []

    def _triggers(self, call: int, dlvl: int, xl: int) -> list[str]:
        out = []
        if dlvl and dlvl not in self.seen_dlvl:
            self.seen_dlvl.add(dlvl)
            out.append(f"level_{dlvl}")
        if xl and self.last_xl is not None and xl > self.last_xl:
            out.append(f"xl_{xl}")
        if xl:
            self.last_xl = xl
        if self.every_n and call - self.last_periodic >= self.every_n:
            self.last_periodic = call
            out.append(f"call_{call}")
        return out

    def after_call(self, core_env, *, call: int, status: dict, finished: bool) -> Optional[str]:
        """Run after every tool call. Returns the new checkpoint's id, if any."""
        numbers = game_numbers(status)
        dlvl, xl = numbers["dlvl"] or 0, numbers["xl"] or 0
        self.max_dlvl = max(self.max_dlvl, dlvl)
        self.max_xl = max(self.max_xl, xl)
        if not self.seeded:
            # The first call only sets the watermarks: the state it sees is the
            # start (or the resumed checkpoint), which the archive already has
            # or which is not worth a checkpoint of its own.
            self.seeded = True
            if dlvl:
                self.seen_dlvl.add(dlvl)
            self.last_xl = xl or None
            return None
        if finished:
            self.pending.clear()
            return None
        self.pending.extend(self._triggers(call, dlvl, xl))
        if not self.pending:
            return None
        from nethack_core.engine_env import CheckpointAtPromptError

        directory = _new_directory(self.archive)
        try:
            core_env.underlying.checkpoint(directory / CHECKPOINT_FILE)
        except CheckpointAtPromptError:
            directory.rmdir()
            self.deferrals += 1
            return None
        except Exception as error:  # noqa: BLE001 - reported, never fatal to the game
            for leftover in directory.iterdir():
                leftover.unlink()
            directory.rmdir()
            self.errors.append(f"call {call}: {type(error).__name__}: {error}")
            return None
        from nethack_harness.prompt.balrog import balrog_both

        balrog_max, balrog_min = balrog_both(self.max_dlvl, self.max_xl)
        label = "auto " + "+".join(self.pending)
        meta = {
            **numbers,
            "id": directory.name.lstrip("c"),
            "call": int(call),
            "created_at": self.now(),
            "created_by": "auto",
            "name": label,
            "note": f"automatic checkpoint ({', '.join(self.pending)})",
            "max_dlvl_reached": self.max_dlvl,
            "max_xp_level": self.max_xl,
            "balrog": balrog_max,
            "balrog_min": balrog_min,
            "deferrals": self.deferrals,
        }
        _write_json(directory / META_FILE, meta)
        self.pending.clear()
        self.written.append(meta["id"])
        return meta["id"]


def resume(core_env, directory, *, restore_rng: bool = False, reseed: Optional[tuple] = None):
    """Resume the game in ``core_env`` (a reset ``NetHackCoreEnv``) from a
    checkpoint directory. Returns ``(observation, meta)``.

    ``restore_rng=False`` (the default) drops the checkpoint's RNG blob, so the
    random stream restarts from the game's seed. ``reseed=(core, disp)`` then
    reseeds it, for runs where two restores of one checkpoint must diverge.
    """
    directory = pathlib.Path(directory)
    meta = read_meta(directory)
    source = directory / CHECKPOINT_FILE
    engine = core_env.underlying
    if restore_rng:
        obs = engine.resume(source)
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
        data.pop("rng", None)
        with tempfile.TemporaryDirectory() as tmp:
            stripped = pathlib.Path(tmp) / CHECKPOINT_FILE
            stripped.write_text(json.dumps(data), encoding="utf-8")
            obs = engine.resume(stripped)
    if reseed is not None:
        engine._engine.reseed(core=int(reseed[0]), disp=int(reseed[1]))
    # NetHackCoreEnv caches the observation its own reset/step produced.
    core_env._last_observation = obs
    return obs, meta
