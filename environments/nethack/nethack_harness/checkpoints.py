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
* The CONVERSATION, in the sense the design wants. ``prefix.jsonl`` now carries
  the LM-visible transcript (what the model said, what it called, what it was
  served), written by the harness at save time -- but a resumed player REPLAYS
  it as quoted text rather than resuming an actual conversation, because both
  player scaffolds launch with sessions off (``--no-session`` /
  ``--no-session-persistence``). Continuity of content, not of session.

WHAT A CHECKPOINT DIRECTORY LOOKS LIKE
--------------------------------------
``archive/<run>/c<id>/``

    state.bundle   engine state + level files (binary, self-describing)
    meta.json      id, parent, name, note, dlvl, xl, hp/max, gameturn, score,
                   balrog, balrog_min, visits, attempts_from, created_by
    prefix.jsonl   the LM-visible transcript at save time (see above)
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
import hashlib
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
#: 2 adds the mandatory body ``sha256`` (see :func:`pack_bundle`). Version-1
#: bundles predate the checksum and are still readable, unverified.
BUNDLE_VERSION = 2

#: Filenames inside a checkpoint directory.
STATE_BUNDLE = "state.bundle"
META_JSON = "meta.json"
PREFIX_JSONL = "prefix.jsonl"
LESSONS_MD = "lessons.md"

#: Where the harness mirrors the RUN's BALROG high-water marks so the `save`
#: skill can reach them. The skill is handed only `(env, obs)`, and BALROG
#: scores the deepest level a rollout TOUCHED -- a checkpoint taken after
#: climbing back up would otherwise record a pair for a shallower run than the
#: one that actually happened. Set in nethack.py's env_response.
HIGH_WATER_DLVL_ATTR = "_max_dlvl_reached"
HIGH_WATER_XL_ATTR = "_max_xp_level"

#: Where the harness accumulates the LM-visible transcript, for
#: :func:`checkpoint_save` to write into ``prefix.jsonl``. See
#: ``nethack.py:_append_conversation_prefix`` for what that transcript is and,
#: more importantly, what it is NOT: the player scaffolds run with
#: ``--no-session`` / ``--no-session-persistence``, so a resumed player replays
#: this as quoted TEXT rather than resuming an actual conversation.
CONVERSATION_PREFIX_ATTR = "_conversation_prefix"

#: Attribute :func:`checkpoint_restore` publishes the restored frame on. The
#: caller needs the observation itself; it cannot travel inside ``meta``, which
#: is JSON that gets written back to disk.
LAST_RESTORE_OBS_ATTR = "_last_restore_obs"

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
    body = b"".join(sections[n] for n in order)
    header = {
        "version": BUNDLE_VERSION,
        "sections": [{"name": n, "len": len(sections[n])} for n in order],
        # THE CHECKSUM. There was none, and the gap was not theoretical: the
        # section lengths are self-declared, so `unpack_bundle` accepted a
        # lying length, trailing garbage, and any single bit flip inside a
        # section body. A damaged level-file body restores CLEANLY -- every
        # file exists, so the file-existence guard sees nothing -- and then
        # NetHack `panic()`s and the run is recorded as a full-HP death. Bit
        # rot, a torn copy and deliberate tampering all land in that same path,
        # and this is the only place that can tell.
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    header.update(header_extra or {})
    raw_header = json.dumps(header, sort_keys=True).encode("utf-8")
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
    body_start = off
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
    if off != len(blob):
        # Trailing bytes mean the declared lengths and the file disagree. That
        # is a damaged or rewritten bundle, and the sections read out of it are
        # not trustworthy just because they happened to parse.
        raise CheckpointIntegrityError(
            f"state.bundle has {len(blob) - off} trailing byte(s) after its "
            f"declared sections; the file does not match its own header"
        )
    want = header.get("sha256")
    if want:
        got = hashlib.sha256(blob[body_start:]).hexdigest()
        if got != want:
            raise CheckpointIntegrityError(
                f"state.bundle checksum mismatch (header {want[:16]}..., body "
                f"{got[:16]}...): the engine blobs are damaged. Restoring them "
                f"would produce a game that panics on a later turn and reports "
                f"as a full-HP death"
            )
    elif int(header.get("version", 1)) >= 2:
        raise CheckpointIntegrityError(
            "state.bundle claims version >= 2 but carries no sha256; its "
            "header has been rewritten"
        )
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
        # Branch identity, for the E16 stop condition "Sokoban entrance". Depth
        # alone cannot express it: Sokoban's levels have small `depth` values
        # that a Dungeons-of-Doom checkpoint also has. NetHack numbers the
        # branches in dungeon.def order -- 0 Dungeons of Doom, 1 Gehennom,
        # 2 Gnomish Mines, 3 Quest, 4 Sokoban -- and blstats carries the number.
        "dungeon_number": get("dungeon_number"),
        "level_number": get("level_number"),
    }


#: The meta fields a restore re-derives from the engine and checks. Every one
#: is HARNESS-COMPUTED from blstats at save time (:func:`_status_snapshot`) —
#: no model-authored text contributes to any of them, which is what makes the
#: audit meaningful rather than a comparison of one narrative with another.
AUDIT_FIELDS = ("dlvl", "xl", "hp", "max_hp", "gameturn", "score",
                # Branch identity was written into meta and drives the Sokoban
                # stop condition while being unauditable -- so a restore could
                # disagree about WHICH BRANCH the hero is in and nothing would
                # notice.
                "dungeon_number", "level_number")

#: tty markers that mean the game is parked on a prompt rather than at a clean
#: command prompt. Checkpointing there is refused: see :func:`assert_savepoint`.
PENDING_PROMPT_MARKERS = ("--More--", "--more--")


class CheckpointSavepointError(ValueError):
    """This MOMENT cannot be checkpointed. The game is fine; the timing is not.

    Deliberately NOT a :class:`CheckpointIntegrityError`. That exception means
    "something is broken and a rollout must not continue as if it were not",
    and the `save` skill re-raises it for exactly that reason. A pending
    ``--More--`` is neither: the game is healthy, the model simply asked at a
    moment where meta.json and the bundle could disagree. So it comes back to
    the model as skill feedback ("clear the prompt, then save") instead of
    ending the rollout.
    """


def assert_savepoint(engine_env, where: str = "") -> None:
    """Refuse to checkpoint a game that is parked on a prompt.

    THE BUG THIS CLOSES. ``meta.json`` is built from ``raw.blstats`` -- the
    numpy observation buffer, refreshed only by an engine step -- while
    ``state.bundle`` serializes the LIVE HEAP. Nothing reconciled them. With a
    ``--More--`` pending over a deferred level transition the two disagree:
    measured, meta recorded Dlvl 4 while the restorable state was Dlvl 2. The
    save SUCCEEDED and the poisoned meta landed in the archive; only a later
    restore failed. That is the worse ordering, because every consumer that
    reads meta WITHOUT restoring -- the frontier selector, the stop conditions,
    the run summary, the BALROG fields -- believed a state that never existed.

    Refusing is the right fix rather than "refresh blstats first": a refresh
    means stepping the engine, which dismisses the prompt and changes the game,
    and a checkpoint taken mid-prompt cannot come back mid-prompt anyway (see
    the module docstring's fidelity limits). So a caller that wants to save
    here must clear the prompt itself, deliberately, and then save.
    """
    raw = _raw_of(engine_env)
    try:
        tty = bytes(raw.tty_chars.reshape(-1).tobytes())
    except Exception:
        return  # no tty to inspect; the fidelity audit still runs on restore
    text = tty.decode("ascii", "replace")
    for marker in PENDING_PROMPT_MARKERS:
        if marker in text:
            raise CheckpointSavepointError(
                f"refusing to checkpoint{f' ({where})' if where else ''}: the "
                f"game is parked on a {marker} prompt. blstats (which meta.json "
                f"is built from) and the live heap (which state.bundle "
                f"serializes) can disagree here -- measured: meta said Dlvl 4 "
                f"for a state that restores to Dlvl 2. Clear the prompt, then "
                f"save"
            )


def restore_fidelity(env, meta: dict, fields=AUDIT_FIELDS) -> dict:
    """Compare a just-restored game's blstats against the checkpoint's meta.

    Returns a record: ``{"ok": bool, "fields": {name: {"meta":…, "engine":…,
    "match": bool}}}``. It reads the LIVE engine, so it cannot be satisfied by
    anything written into meta.json after the fact.

    WHY THIS IS NOT OPTIONAL. The archive is the experiment's dataset: every
    depth claim E16 makes is a claim about a state that a checkpoint's meta
    describes. If a restore can quietly hand back a different state than its
    meta advertises, the archive records states that never existed and the
    selector optimises over fiction. So the audit runs on EVERY restore and a
    mismatch is an error, not a log line.
    """
    engine_env = _engine_of(env)
    live = _status_snapshot(engine_env)
    out = {}
    ok = True
    for name in fields:
        want = meta.get(name)
        got = live.get(name)
        match = (want is not None) and int(want) == int(got)
        ok = ok and match
        out[name] = {"meta": want, "engine": got, "match": match}
    return {"ok": ok, "fields": out, "checked_at": time.time()}


def _balrog_pair(status: dict, max_dlvl: Optional[int], max_xl: Optional[int],
                 *, ascended: bool = False) -> tuple:
    """``(balrog, balrog_min)`` for a checkpoint — THE HOUSE PAIR, not a proxy.

    ``balrog_both(max_dlvl, max_xl)``: the real BALROG metric off the vendored
    achievement table, scored ``max`` over the (Dlvl, XL) axes and ``min`` over
    them. Identical call, identical arguments, to what ``nethack_v1.py``'s
    ``finalize`` publishes as ``balrog_pct`` / ``balrog_min_pct``, so an
    archive row and a trace row describe the same state with the same number.

    THIS WAS WRONG AND THE WRONGNESS WAS SILENT. The first implementation used
    ``progression_score``, the analytic proxy ``(DL/50)^1.3 * (XL/30)^0.6``,
    whose own module docstring says "DEPRECATED ... do not quote it as BALROG".
    It is not a rescaling of the real metric; the two disagree in both
    directions at once. Measured at the states seed 1 actually reaches:

        D12 XL3   proxy 0.0393   real (0.2061, 0.0208)
        D13 XL4   proxy 0.0518   real (0.2566, 0.0242)
        D15 XL5   proxy 0.0713   real (0.3088, 0.0291)

    So the archive's ``balrog`` read ~0.2x the real max while its
    ``balrog_min`` read ~1.9-2.4x high -- and both looked entirely plausible.
    Compounding it, ``save()`` passed no high-water marks, so ``balrog`` and
    ``balrog_min`` were two copies of one deprecated number and the "BALROG max
    and min" pair carried no information at all.

    ``max_dlvl``/``max_xl`` are the RUN's high-water marks (the harness's
    ``max_dlvl_reached`` / ``max_xp_level``), because BALROG scores the deepest
    level a rollout touched, not where it is standing. They default to the
    checkpoint's own depth/XL when a caller cannot supply them, which is the
    correct floor rather than a guess.
    """
    from nethack_harness.prompt.balrog import balrog_both

    dlvl, xl = status["dlvl"], status["xl"]
    hi, lo = balrog_both(
        max(int(max_dlvl or dlvl), dlvl),
        max(int(max_xl or xl), xl),
        reached_planes=False,
        ascended=bool(ascended),
    )
    return hi, lo


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
    prefix: Optional[list] = None,
    extra: Optional[dict] = None,
    overwrite: bool = False,
) -> dict:
    """Write ``directory`` as a complete, resumable checkpoint. Returns its meta.

    ``directory`` is the checkpoint itself (``archive/<run>/c<id>/``); its id is
    the trailing path component with any leading ``c`` stripped. The write is
    atomic: the directory appears whole or not at all (see the module docstring).

    ``name`` is a short human label, ``note`` the purpose ("try the mines
    branch"). Neither is interpreted.

    Raises :class:`CheckpointSavepointError` if the game is parked on a prompt
    (a freshly reset game is: it sits on the welcome ``--More--``). Clear it
    first -- ``env.step(13)`` -- then save.
    """
    directory = Path(directory)
    engine_env = _engine_of(env)
    raw = _raw_of(engine_env)

    # Never checkpoint a game that is already inconsistent: the resume would
    # inherit the divergence and present it as a death.
    assert_dungeon_on_disk(engine_env, where="checkpoint_save")
    # ...and never checkpoint one whose meta and heap can disagree.
    assert_savepoint(engine_env, where="checkpoint_save")

    seeds = list(getattr(engine_env, "_current_seeds", None) or (0, 0))
    sections = {
        "level": raw.save_level(),
        "player": raw.save_player(),
        "levelfiles": raw.save_levelfiles(),
    }
    bundle = pack_bundle(sections, header_extra={"seed": seeds})

    status = _status_snapshot(engine_env)
    # High-water marks, in this order of preference: what the caller passed;
    # what the harness mirrored onto the env (see HIGH_WATER_ATTRS -- the
    # `save` skill is handed only `env`, so this is the only channel it has);
    # then the checkpoint's own depth/XL as the floor.
    if max_dlvl is None:
        max_dlvl = getattr(env, HIGH_WATER_DLVL_ATTR,
                           getattr(engine_env, HIGH_WATER_DLVL_ATTR, None))
    if max_xl is None:
        max_xl = getattr(env, HIGH_WATER_XL_ATTR,
                         getattr(engine_env, HIGH_WATER_XL_ATTR, None))
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
        # WHAT THE PAIR WAS COMPUTED FROM. Without these, an archive row's
        # BALROG numbers cannot be re-derived or checked against the trace
        # metrics, and a missing high-water mark is indistinguishable from a
        # run that never went deeper than this checkpoint.
        "balrog_metric": "balrog_both (vendored BALROG achievement table)",
        "max_dlvl_reached": max(int(max_dlvl or status["dlvl"]), status["dlvl"]),
        "max_xp_level": max(int(max_xl or status["xl"]), status["xl"]),
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
        # The conversation prefix -- the plan that was live when this state was
        # saved. Falls back to the transcript the harness accumulates on the
        # env (CONVERSATION_PREFIX_ATTR), so the published `save` skill, which
        # is handed only `(env, obs)`, still writes one. Empty when neither is
        # available, and `checkpoint_list`/`restore` treat that as "no prefix"
        # rather than an error.
        if prefix is None:
            prefix = getattr(env, CONVERSATION_PREFIX_ATTR, None) or \
                getattr(engine_env, CONVERSATION_PREFIX_ATTR, None)
        _write_file(tmp / PREFIX_JSONL, _render_prefix_jsonl(prefix))
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


def _render_prefix_jsonl(prefix) -> bytes:
    """One JSON object per line, or b"" for an absent/empty transcript."""
    if not prefix:
        return b""
    out = []
    for rec in prefix:
        if isinstance(rec, dict):
            out.append(json.dumps(rec, sort_keys=True))
    return ("\n".join(out) + "\n").encode("utf-8") if out else b""


def checkpoint_prefix(directory) -> list:
    """A checkpoint's conversation prefix, or ``[]`` if it has none."""
    path = Path(directory) / PREFIX_JSONL
    if not path.is_file() or path.stat().st_size == 0:
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def checkpoint_meta(directory) -> dict:
    """Read a checkpoint's ``meta.json``.

    A missing or malformed one is a :class:`CheckpointIntegrityError`, not a
    bare ``FileNotFoundError``/``JSONDecodeError``: callers already treat that
    exception as "this checkpoint is not usable" and would otherwise have to
    know three exception types to say the same thing.
    """
    path = Path(directory) / META_JSON
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"checkpoint {directory} has no {META_JSON}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointIntegrityError(
            f"checkpoint {directory}'s {META_JSON} is not valid JSON: {exc}") from exc


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

    def _created_at(path):
        # ONE unreadable meta.json used to make the WHOLE archive unlistable,
        # which turns a single damaged entry into a total loss of the run's
        # state. Sort it to the front and let the caller's own read raise on
        # the entry that is actually broken.
        try:
            return checkpoint_meta(path).get("created_at", 0) or 0
        except (OSError, ValueError, CheckpointIntegrityError):
            return 0

    return sorted(out, key=_created_at)


def checkpoint_restore(directory, env=None, *, count_visit: bool = True,
                       audit: bool = True, fidelity_log=None):
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

    # Publish the restored frame the way a normal step would. NetHackCoreEnv
    # caches the last observation and every caller above it (shape_observation,
    # the renderers, the death detector) reads that cache -- so without this a
    # resumed rollout's first observation would be the FRESH RESET's level-1
    # room while blstats describe the checkpoint's floor. That mismatch is
    # invisible in the status line and was exactly the class of bug the engine's
    # own `resume()` docstring records.
    try:
        restored_obs = raw.to_core_observation()
    except Exception:
        restored_obs = None
    if restored_obs is not None:
        if hasattr(env, "_last_observation"):
            env._last_observation = restored_obs
        # Also published under a name of our own, because the caller needs the
        # frame ITSELF (to shape and serve as the first observation) and must
        # not have to reach into another module's private cache to get it. NOT
        # returned inside `meta`: meta is JSON that gets written to disk, and a
        # CoreObservation in it makes the whole dict unserializable.
        setattr(env, LAST_RESTORE_OBS_ATTR, restored_obs)

    # RESTORE FIDELITY AUDIT (E16 research-integrity requirement 2). Re-read
    # the engine and prove it is the state meta.json advertises. A mismatch is
    # an integrity error, not a silent continue: continuing would let the
    # archive -- and every depth claim read off it -- describe a state the
    # engine never actually produced.
    record = None
    if audit:
        record = restore_fidelity(env, meta)
        record["checkpoint"] = str(directory)
        record["id"] = meta.get("id")
        if fidelity_log is not None:
            try:
                with open(fidelity_log, "a") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError:
                pass  # the audit itself must never be what breaks a resume
        if not record["ok"]:
            bad = ", ".join(
                f"{k}: meta={v['meta']} engine={v['engine']}"
                for k, v in record["fields"].items() if not v["match"]
            )
            raise CheckpointIntegrityError(
                f"restore fidelity failure ({directory}): the resumed game does "
                f"not match the checkpoint's recorded state [{bad}]. The archive "
                f"would otherwise record a state that never existed."
            )
    if count_visit:
        try:
            meta["visits"] = int(meta.get("visits", 0)) + 1
            atomic_write(directory / META_JSON,
                         json.dumps(meta, indent=2, sort_keys=True) + "\n")
        except OSError:
            pass  # a read-only archive must not break a resume
    # Attached to the RETURNED meta only, deliberately after the persist above:
    # the audit is a property of THIS resume, not of the checkpoint, and
    # meta.json must stay the checkpoint's identity.
    if record is not None:
        meta["restore_fidelity"] = record
    return env, meta


__all__ = [
    "AUDIT_FIELDS",
    "CheckpointSavepointError",
    "PENDING_PROMPT_MARKERS",
    "assert_savepoint",
    "CONVERSATION_PREFIX_ATTR",
    "HIGH_WATER_DLVL_ATTR",
    "HIGH_WATER_XL_ATTR",
    "LAST_RESTORE_OBS_ATTR",
    "BUNDLE_MAGIC",
    "BUNDLE_VERSION",
    "LESSONS_MD",
    "META_JSON",
    "PREFIX_JSONL",
    "STATE_BUNDLE",
    "atomic_write",
    "checkpoint_list",
    "checkpoint_meta",
    "checkpoint_prefix",
    "checkpoint_restore",
    "checkpoint_save",
    "pack_bundle",
    "restore_fidelity",
    "unpack_bundle",
]
