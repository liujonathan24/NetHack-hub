"""Multi-level snapshot completeness (GATE B / Task 12b).

Off-current dungeon levels are persisted by NetHack to disk as "<lock>.<n>"
files in the hackdir (savelev/getlev); only the level in the arena is captured
by the in-memory snapshot. nle_fr_snapshot now also bundles those level files
into the handle and nle_fr_restore rewrites them, so a snapshot spans the whole
dungeon: travelling back to a level left before the snapshot reads the
snapshot-time level, not a later mutation.

These tests exercise the bundling machinery directly (the new code): a snapshot
captures the "<lock>.*" level files, restore reverts post-snapshot changes to
them, and the static template files (<role>.lev / *.des) are NOT bundled. The
in-game read path (getlev reading those files) is stock NetHack.
"""
import os

from nethack_core import _engine

# Dynamic dungeon-level files are "<base>.<ledger>". The base is empty in this
# build, so they are ".<digits>" (e.g. ".50") in the hackdir.
#
# NB the base is NOT the same thing as ``s_lock``: set_levelfile_name()
# (files.c:499) rewrites s_lock IN PLACE to "<base>.<lev>" on every level-file
# operation, so s_lock is ".2" once the hero has been to ledger 2. Reading it
# as a prefix -- which the bundler did until E16 -- matches nothing at all, and
# every test in this file that only ever touches a fresh game (s_lock still
# empty) passes anyway. See test_levelfile_integrity.py for the coverage that
# actually exercises a transitioned game.
_LOCK_PREFIX = "."


def _write(path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _read(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_snapshot_bundles_and_restores_level_files():
    """A post-snapshot change to a level file is reverted on restore."""
    env = _engine.RawEngine()
    env.start(core=42, disp=42)
    hd = env._hackdir

    # Simulate an off-current dungeon level on disk.
    lvl = os.path.join(hd, _LOCK_PREFIX + "7")
    _write(lvl, b"ORIGINAL-LEVEL-7-CONTENTS")

    h = env.snapshot()  # bundles 1lock.7

    # Mutate the level file after the snapshot (as revisiting + leaving would).
    _write(lvl, b"MUTATED-AFTER-SNAPSHOT-DIFFERENT-LENGTH")
    assert _read(lvl) != b"ORIGINAL-LEVEL-7-CONTENTS"

    env.restore(h)

    assert _read(lvl) == b"ORIGINAL-LEVEL-7-CONTENTS", (
        "restore did not revert the off-current level file from the snapshot"
    )
    env.free_snapshot(h)
    env.end()


def test_static_template_files_are_not_bundled():
    """Static templates (<role>.lev / *.des) live in the shared read-only datadir,
    never in the per-env hackdir, so the hackdir-scanning snapshot bundler cannot
    capture or revert them — only "<lock>.<n>" level files are bundled."""
    env = _engine.RawEngine()
    env.start(core=42, disp=42)
    hd = env._hackdir

    # The template exists in the shared datadir (read directly, never copied)...
    datadir = env._build_dat_path()
    assert (datadir / "Val-strt.lev").exists(), "expected template in shared datadir"
    # ...and is NOT present in the writable per-env hackdir, so snapshot bundling
    # (which only scans the hackdir for level files) can never touch it.
    assert not os.path.exists(os.path.join(hd, "Val-strt.lev")), (
        "static template must not live in the per-env hackdir"
    )

    # A snapshot/restore round-trip leaves the shared datadir untouched.
    before = _read(str(datadir / "Val-strt.lev"))
    h = env.snapshot()
    env.restore(h)
    assert _read(str(datadir / "Val-strt.lev")) == before, (
        "snapshot/restore must not modify the shared read-only datadir"
    )
    env.free_snapshot(h)
    env.end()


def test_levelfile_created_after_snapshot_is_deleted_on_restore():
    """A level file that did not exist at snapshot time is DELETED on restore.

    CONTRACT CHANGE (E16, 2026-08). This test previously asserted the opposite
    ("post-snapshot level files are left alone"), on the reasoning that NetHack
    regenerates unvisited levels via mklev rather than reading them, so a stale
    file is harmless. It is harmless for READS, but it makes the restored
    (heap, disk) pair a function of the abandoned future rather than of the
    snapshot -- and that whole class of divergence is what produced three
    rollouts' worth of corrupted data: when the heap says a level file exists
    and it does not, goto_level hands the run to done(TRICKED), a full-HP
    game-over indistinguishable from a monster kill.

    The contract is now an equality in both directions: after restore, the set
    of level files on disk is EXACTLY the set that existed at snapshot time.
    """
    env = _engine.RawEngine()
    env.start(core=42, disp=42)
    hd = env._hackdir

    h = env.snapshot()  # no ".42" at this point
    newfile = os.path.join(hd, _LOCK_PREFIX + "42")
    _write(newfile, b"CREATED-AFTER-SNAPSHOT")

    env.restore(h)

    assert not os.path.exists(newfile), (
        "restore must drop level files that did not exist at snapshot time, so "
        "the on-disk level set is exactly the snapshot's"
    )
    env.free_snapshot(h)
    env.end()
