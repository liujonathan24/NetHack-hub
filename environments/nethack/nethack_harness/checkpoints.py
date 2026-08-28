"""E16 persistent checkpoints — dungeon state that outlives the process.

WHY NOT THE EXISTING SNAPSHOT
-----------------------------
``RawEngine.snapshot()`` (``nle_fr_snapshot``) is the fastest and highest
fidelity checkpoint the engine has, and it CANNOT be written to disk. Reading
``third_party/NetHack/src/src/nle_fast_reset.c``, the handle holds:

* ``saved_ctx`` — a byte copy of ``nle_ctx_t``, which is full of live pointers
  (``s_arena_base``, ``stack.sptr``, ``s_netHackRL_instance``, ``fmon``/``fobj``
  chains that point into the arena);
* ``saved_stack`` — an image of the coroutine (fcontext) stack, restored by
  ``memcpy`` to ``nle->stack.sptr - stack_size``, i.e. meaningful only at that
  exact address in that exact process;
* ``saved_arena`` — arena bytes whose internal pointers are absolute;
* ``saved_mirror`` — the libc-malloc'd rl display mirror.

Restoring it is a set of ``memcpy``s into a LIVE process's address space
(``nle_fr_restore``), and ``restore()`` explicitly refuses a handle another
engine instance created. There is no serialization, and adding one would mean
pointer-swizzling the whole arena. So a disk checkpoint uses NetHack's OWN save
machinery instead.

THE DISK FORMAT AND ITS FIDELITY
--------------------------------
``state.bundle`` is three engine blobs plus the seed:

* ``player`` — ``nle_save_player`` -> ``savegamestate`` (save.c:405). The ``u``
  struct, inventory, attributes, timers, light sources, killers, the DUNGEON
  GRAPH (``save_dungeon``, dungeon.c:175, including the whole ``level_info[]``
  array), fruit/names/waterlevel/msghistory.
* ``level`` — ``nle_save_level`` -> ``savelev`` (nle.c). The current floor:
  terrain, objects, monsters, traps, engravings, hero memory.
* ``levelfiles`` — ``nle_fr_levelfiles_blob``. Every OFF-current dungeon level,
  which NetHack keeps as ``<base>.<ledger>`` files rather than in memory.

The third is MANDATORY, not an optimization. Because the player blob carries
``level_info[]``, the resumed heap believes in every level the original game
visited; NetHack answers a level file that the heap believes in but cannot open
with ``done(TRICKED)`` — a full-HP game-over that looks exactly like a monster
kill (this is the same failure that corrupted three E15 rollouts; see
``nethack_harness/integrity.py``). ``checkpoint_restore`` therefore verifies
the invariant explicitly and raises rather than returning a game that will
"die" on its next staircase.

Does NOT survive a checkpoint (fidelity limits, measured):

* The RNG STREAM. Resume re-seeds from the saved ``(core, disp)`` and replays
  no history, so object appearances/ids match but future random rolls do not
  continue the original stream. Two resumes of one checkpoint agree with each
  other, not with the original run's future.
* The rl-port display mirror and the tty frame. ``resume`` re-renders from
  scratch (one ctrl-R), so ``tty_chars`` is regenerated rather than restored;
  ``glyphs``/``chars``/``colors``/``blstats`` come from the reloaded state.
* The message history line that was on screen at save time, and any pending
  ``--More--``/menu/prompt. A checkpoint resumes at a clean command prompt; it
  cannot be taken mid-prompt and come back mid-prompt.
* The conversation. ``prefix.jsonl`` is reserved for the harness to write.

WHAT A CHECKPOINT DIRECTORY LOOKS LIKE
--------------------------------------
``archive/<run>/c<id>/``

    state.bundle   engine state + level files (binary, self-describing)
    meta.json      id, parent, name, note, dlvl, xl, hp/max, gameturn, score,
                   balrog, balrog_min, visits, attempts_from, created_by
    prefix.jsonl   conversation prefix (reserved; the harness writes it later)
    lessons.md     free-text lessons carried forward

ATOMICITY
---------
A checkpoint directory appears whole or not at all: everything is written into
a sibling ``.tmp-*`` directory, every file is ``fsync``ed, the directory is
``fsync``ed, and only then is it ``rename``d into place (rename of a directory
onto a non-existent name is atomic on POSIX). A crash at any point leaves the
previous checkpoint untouched and at worst a ``.tmp-*`` directory behind, which
:func:`checkpoint_list` and :func:`checkpoint_restore` ignore. Later mutations
of ``meta.json`` (visit/attempt counters) go through the same
temp-file + fsync + ``os.replace`` dance, which is atomic per file.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from nethack_harness.integrity import (
    CheckpointIntegrityError,
    assert_dungeon_on_disk,
)

#: Magic + version for ``state.bundle``. Bump the version if the section list
#: or the engine blob formats change incompatibly.
BUNDLE_MAGIC = b"NLECKPT\x01"
BUNDLE_VERSION = 1

#: Filenames inside a checkpoint directory.
STATE_BUNDLE = "state.bundle"
META_JSON = "meta.json"
PREFIX_JSONL = "prefix.jsonl"
LESSONS_MD = "lessons.md"

#: Sections of ``state.bundle``, in load order. The order is the engine's hard
#: contract for a resume: level, then player, then the off-current dungeon.
_SECTIONS = ("level", "player", "levelfiles")


# --------------------------------------------------------------------------- #
# atomic write primitives
# --------------------------------------------------------------------------- #

def _fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/creation inside it is durable."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_file(path: Path, data: bytes) -> None:
    """Write and fsync one file (the caller fsyncs the directory)."""
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def atomic_write(path, data) -> None:
    """Replace one file atomically: temp file in the same dir, fsync, rename.

    Used for ``meta.json`` updates after a checkpoint exists. A reader either
    sees the whole old file or the whole new one, never a truncated middle.
    """
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns():x}")
    try:
        _write_file(tmp, data)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


# --------------------------------------------------------------------------- #
# state.bundle container
# --------------------------------------------------------------------------- #

def pack_bundle(sections: dict, header_extra: Optional[dict] = None) -> bytes:
    """Serialize ``{section_name: bytes}`` into one self-describing blob.

    Layout: ``BUNDLE_MAGIC`` | uint32 header length | JSON header | sections.
    The header carries the section order and lengths, so the format can gain
    sections without breaking older readers' ability to find the ones they know.
    """
    order = [name for name in _SECTIONS if name in sections]
    order += [name for name in sections if name not in order]
    header = {
        "version": BUNDLE_VERSION,
        "sections": [{"name": n, "len": len(sections[n])} for n in order],
    }
    header.update(header_extra or {})
    raw_header = json.dumps(header, sort_keys=True).encode("utf-8")
    body = b"".join(sections[n] for n in order)
    return BUNDLE_MAGIC + struct.pack("<I", len(raw_header)) + raw_header + body


def unpack_bundle(blob: bytes) -> tuple:
    """Inverse of :func:`pack_bundle`. Returns ``(header, {name: bytes})``.

    Raises :class:`CheckpointIntegrityError` on anything malformed — a
    half-written bundle must be an error, never a silently emptier game.
    """
    if len(blob) < len(BUNDLE_MAGIC) + 4 or not blob.startswith(BUNDLE_MAGIC):
        raise CheckpointIntegrityError(
            "state.bundle is not a NetHack checkpoint bundle (bad magic)"
        )
    off = len(BUNDLE_MAGIC)
    (hlen,) = struct.unpack("<I", blob[off:off + 4])
    off += 4
    if hlen > len(blob) - off:
        raise CheckpointIntegrityError("state.bundle header is truncated")
    try:
        header = json.loads(blob[off:off + hlen].decode("utf-8"))
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"state.bundle header is not valid JSON: {exc}"
        ) from exc
    off += hlen
    out = {}
    for entry in header.get("sections", []):
        n = int(entry["len"])
        if off + n > len(blob):
            raise CheckpointIntegrityError(
                f"state.bundle section {entry.get('name')!r} is truncated "
                f"({len(blob) - off} of {n} bytes present)"
            )
        out[entry["name"]] = blob[off:off + n]
        off += n
    return header, out


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

def _engine_of(env):
    """The EngineEnv under whatever env wrapper we were handed."""
    for cand in (getattr(env, "_engine", None), getattr(env, "engine", None), env):
        if cand is not None and hasattr(cand, "checkpoint"):
            return cand
    raise TypeError(
        "checkpoint_save needs an engine-backed env (NetHackCoreEnv or "
        f"EngineEnv); got {type(env).__name__}"
    )


def _raw_of(engine_env):
    return getattr(engine_env, "_engine", None) or getattr(engine_env, "engine", None)


def _status_snapshot(engine_env) -> dict:
    """Read the fields meta.json records straight off the live engine."""
    from nethack_core.observations import BLSTATS_IDX

    raw = _raw_of(engine_env)
    bl = raw.blstats
    get = lambda k: int(bl[BLSTATS_IDX[k]])  # noqa: E731
    return {
        "dlvl": get("depth"),
        "xl": get("experience_level"),
        "hp": get("hitpoints"),
        "max_hp": get("max_hitpoints"),
        "gameturn": get("time"),
        "score": get("score"),
    }


def _balrog_pair(status: dict, max_dlvl: Optional[int], max_xl: Optional[int]) -> tuple:
    """``(balrog, balrog_min)`` for a checkpoint.

    ``balrog`` is the run's BALROG progression at save time, computed from the
    high-water marks the harness tracks (``max_dlvl_reached`` / ``max_xp_level``)
    when the caller supplies them. ``balrog_min`` is the progression this
    checkpoint on its own GUARANTEES on resume — computed from the checkpoint's
    own depth and XL — because a resume starts where the checkpoint is, not
    where the run had once been. They are equal unless the run had descended
    deeper (or levelled higher) than the point being saved.
    """
    from nethack_harness.prompt.balrog import progression_score

    dlvl, xl = status["dlvl"], status["xl"]
    balrog_min = progression_score(dlvl, xl)
    balrog = progression_score(
        max(int(max_dlvl or dlvl), dlvl), max(int(max_xl or xl), xl)
    )
    return balrog, balrog_min


# --------------------------------------------------------------------------- #
# save / restore
# --------------------------------------------------------------------------- #

def checkpoint_save(
    env,
    directory,
    name: str = "",
    note: str = "",
    *,
    parent: Optional[str] = None,
    created_by: Optional[str] = None,
    max_dlvl: Optional[int] = None,
    max_xl: Optional[int] = None,
    visits: int = 0,
    attempts_from: int = 0,
    lessons: str = "",
    extra: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """Write ``directory`` as a complete, resumable checkpoint. Returns its meta.

    ``directory`` is the checkpoint itself (``archive/<run>/c<id>/``); its id is
    the trailing path component with any leading ``c`` stripped. The write is
    atomic: the directory appears whole or not at all (see the module docstring).

    ``name`` is a short human label, ``note`` the purpose ("try the mines
    branch"). Neither is interpreted.
    """
    directory = Path(directory)
    engine_env = _engine_of(env)
    raw = _raw_of(engine_env)

    # Never checkpoint a game that is already inconsistent: the resume would
    # inherit the divergence and present it as a death.
    assert_dungeon_on_disk(engine_env, where="checkpoint_save")

    seeds = list(getattr(engine_env, "_current_seeds", None) or (0, 0))
    sections = {
        "level": raw.save_level(),
        "player": raw.save_player(),
        "levelfiles": raw.save_levelfiles(),
    }
    bundle = pack_bundle(sections, header_extra={"seed": seeds})

    status = _status_snapshot(engine_env)
    balrog, balrog_min = _balrog_pair(status, max_dlvl, max_xl)
    ident = directory.name[1:] if directory.name.startswith("c") else directory.name
    meta = {
        "id": ident,
        "parent": parent,
        "name": name,
        "note": note,
        "seed": seeds,
        "created_at": time.time(),
        "created_by": created_by or os.environ.get("NLD_AGENT_ID") or "unknown",
        "balrog": balrog,
        "balrog_min": balrog_min,
        "visits": int(visits),
        "attempts_from": int(attempts_from),
        "bundle_bytes": len(bundle),
        "bundle_version": BUNDLE_VERSION,
    }
    meta.update(status)
    if extra:
        meta.update(extra)

    if directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"checkpoint {directory} already exists; checkpoints are "
                f"immutable (pass overwrite=True only to replace one)"
            )
    directory.parent.mkdir(parents=True, exist_ok=True)

    # Stage everything in a sibling temp dir so the checkpoint is never
    # observable half-written, then swap it in with one rename.
    tmp = Path(tempfile.mkdtemp(
        prefix=f".tmp-{directory.name}-", dir=str(directory.parent)
    ))
    try:
        _write_file(tmp / STATE_BUNDLE, bundle)
        _write_file(tmp / META_JSON,
                    (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
        # Reserved for the harness: the conversation prefix is written later.
        _write_file(tmp / PREFIX_JSONL, b"")
        _write_file(tmp / LESSONS_MD, lessons.encode("utf-8"))
        _fsync_dir(tmp)
        if directory.exists():
            # Replacing an existing checkpoint: park the old one, swap, delete.
            # os.rename cannot overwrite a non-empty directory, so this is the
            # only ordering that never leaves the name pointing at a partial
            # write -- the gap between the two renames exposes "absent", never
            # "corrupt".
            doomed = directory.with_name(
                f".old-{directory.name}-{time.time_ns():x}"
            )
            os.rename(directory, doomed)
            os.rename(tmp, directory)
            shutil.rmtree(doomed, ignore_errors=True)
        else:
            os.rename(tmp, directory)
        tmp = None  # renamed away; nothing to clean up
        _fsync_dir(directory.parent)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return meta


def checkpoint_meta(directory) -> dict:
    """Read a checkpoint's ``meta.json``."""
    return json.loads((Path(directory) / META_JSON).read_text())


def checkpoint_list(root) -> list:
    """Every complete checkpoint directly under ``root``, oldest first.

    Skips ``.tmp-*`` / ``.old-*`` staging directories and anything missing a
    bundle, so a crashed write is never mistaken for a checkpoint.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / STATE_BUNDLE).is_file() and (child / META_JSON).is_file():
            out.append(child)
    return sorted(out, key=lambda p: checkpoint_meta(p).get("created_at", 0))


def checkpoint_restore(directory, env=None, *, count_visit: bool = True):
    """Rebuild a playable env from a checkpoint. Returns ``(env, meta)``.

    Works in a process that has never seen the original game: everything comes
    off disk. ``env`` may be an existing engine-backed env to reuse; by default
    a fresh ``EngineEnv`` is created.

    Load order is the engine's contract and is not negotiable — reset to the
    saved seed, ``load_level_raw``, ``load_player_raw``, then the off-current
    level files, then ONE render. The level files must go last: ``nle_load_level``
    stamps the level blob over ``<lock>.<ledger_no(u.uz)>``, and until
    ``load_player_raw`` runs, ``u.uz`` is still the freshly-reset level 1 — so
    installing them earlier gets the level-1 file overwritten with the
    checkpoint's floor, and returning to level 1 later trips ``getlev``'s dlvl
    header check -> ``trickery()`` -> ``done(TRICKED)``.

    Raises :class:`CheckpointIntegrityError` if the bundle is malformed or if
    the resumed game's dungeon has holes.
    """
    directory = Path(directory)
    meta = checkpoint_meta(directory)
    header, sections = unpack_bundle((directory / STATE_BUNDLE).read_bytes())
    for required in ("level", "player"):
        if required not in sections:
            raise CheckpointIntegrityError(
                f"state.bundle is missing its {required!r} section"
            )
    if "levelfiles" not in sections:
        raise CheckpointIntegrityError(
            "state.bundle has no 'levelfiles' section; resuming without the "
            "off-current dungeon leaves the restored heap referencing level "
            "files that do not exist, which NetHack reports as a death"
        )

    if env is None:
        from nethack_core.engine_env import EngineEnv
        env = EngineEnv()
    engine_env = _engine_of(env)
    raw = _raw_of(engine_env)

    core, disp = (header.get("seed") or meta.get("seed") or [0, 0])[:2]
    engine_env.reset(seeds=(int(core), int(disp)))
    # Dismiss the fresh game's welcome --More-- BEFORE loading anything, or the
    # single ctrl-R below is swallowed by that prompt instead of running
    # docrt(): the returned observation then still shows the reset's level-1
    # starting room while blstats describe the checkpoint's floor. ESC rather
    # than CR because ESC is a no-op in command context.
    raw.step(27)
    raw.load_level_raw(sections["level"])      # LEVEL first
    raw.load_player_raw(sections["player"])    # THEN player
    raw.load_levelfiles(sections["levelfiles"])  # THEN the rest of the dungeon
    raw.step(18)                               # single ctrl-R render
    assert_dungeon_on_disk(engine_env, where=f"checkpoint_restore({directory})")

    if count_visit:
        try:
            meta["visits"] = int(meta.get("visits", 0)) + 1
            atomic_write(directory / META_JSON,
                         json.dumps(meta, indent=2, sort_keys=True) + "\n")
        except OSError:
            pass  # a read-only archive must not break a resume
    return env, meta


__all__ = [
    "BUNDLE_MAGIC",
    "BUNDLE_VERSION",
    "LESSONS_MD",
    "META_JSON",
    "PREFIX_JSONL",
    "STATE_BUNDLE",
    "atomic_write",
    "checkpoint_list",
    "checkpoint_meta",
    "checkpoint_restore",
    "checkpoint_save",
    "pack_bundle",
    "unpack_bundle",
]
