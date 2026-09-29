"""E16 GATE — dungeon level files must never desync from the restored heap.

THE INCIDENT
------------
Three E15 forced-revive rollouts recorded a death that never happened. Right
after a post-revive stair transition the game printed::

    Cannot open file ".N" for level N (errno 2).
    Probably someone removed it.

and the run ended at FULL HP with zeroed state (HP 0/0, Dlvl 0). That is not a
monster kill: ``tricked_fileremoved`` (save.c:487) prints exactly those two
lines and calls ``done(TRICKED)``, an instant game-over independent of HP. Its
only caller is ``goto_level`` (do.c:1462), which reaches it when the IN-MEMORY
flag ``level_info[ledger].linfo_flags & LFILE_EXISTS`` is set but the on-disk
``<base>.<ledger>`` file is gone (do.c:1451).

ROOT CAUSE (two defects, both required)
---------------------------------------
1. ``done()`` calls ``clearlocks()`` (end.c:1371 -> files.c:614), which unlinks
   EVERY level file. So the instant the hero dies, the whole dungeon vanishes
   from disk — while the forced-revive snapshot's heap still believes in it.

2. ``nle_fr_bundle_levelfiles`` was supposed to make the snapshot whole across
   the dungeon, but it matched filenames against ``nle->s_lock`` as if that
   were a stable base name. It is not: ``set_levelfile_name`` (files.c:499)
   rewrites ``lock`` IN PLACE to ``"<base>.<lev>"`` on every level-file
   operation. After any level transition the "prefix" is ".2", which prefixes
   no real level file, so the bundle captured ZERO files and restore rewrote
   nothing. The pre-existing bundling tests all passed because they only ever
   snapshot a fresh game, where ``s_lock`` is still the empty base.

Every test here fails on the pre-fix engine and passes on the fixed one.
"""
import os

import pytest

from nethack_core import _engine
from nethack_core._engine import CheckpointIntegrityError

#: ``how_done`` for ``done(TRICKED)`` (hack.h: TRICKED = 12).
HOW_TRICKED = 12

#: The two lines ``tricked_fileremoved`` prints before killing the game.
TRICKED_FILE_MARKERS = ("Cannot open file", "Probably someone removed it")


def _levelfiles(hackdir) -> list:
    """The dynamic dungeon-level files in a hackdir (base is empty here)."""
    return sorted(
        f for f in os.listdir(hackdir) if f.startswith(".") and f[1:].isdigit()
    )


def _started(seed=1):
    """A live engine past the welcome ``--More--``.

    The drain matters: an undrained welcome prompt eats the wait keystroke that
    ``goto_depth`` relies on to run its deferred goto, and the level transition
    silently does not happen.
    """
    env = _engine.RawEngine()
    env.start(core=seed, disp=seed)
    env.step(13)  # CR dismisses the welcome --More--
    return env


def _msg(env) -> str:
    return bytes(env.message).split(b"\0")[0].decode("ascii", "replace")


def _game_over(env) -> None:
    """Drive a real ``done()`` game-over and stop at the moment the harness sees it.

    ``clearlocks()`` hangs off ``done()`` itself (end.c:1371), not off any
    particular ``how``, so every game-over — the rollouts' monster kills
    included — wipes the dungeon identically. This uses ``#quit`` because it is
    the only game-over that is deterministic in-process: the observed
    starvation route (``modify(hunger=-400)``) races ``rn2(20 -
    uhunger/10) >= 19`` in ``newuhs`` (eat.c:3003) and faints instead of dying
    about two thirds of the time, and forcing ``done()`` from a ``set_state``
    poke instead of from inside a step SIGFPEs (``done`` -> ``pline`` ->
    window port -> ``jump_fcontext`` out of the main context).

    An ``hp = 0`` poke — what the existing forced-revive tests use — does NOT
    work at all here: it writes ``u.uhp`` directly and never runs ``done()``,
    so ``clearlocks()`` never fires. That is exactly why the harness's own
    simulated-death suite could not have caught this.

    Stops right after the wipe, while ``done`` is still False. That is where
    the forced revive fires (it triggers on ``hitpoints == 0`` in the shaped
    status). Restoring after the game-over sequence has fully drained is a
    separate, unrelated hazard: ``NetHackRL::load_mirror`` aborts on a torn-down
    game.
    """
    for ch in "#quit":
        env.step(ord(ch))
    env.step(13)        # submit the extended command
    env.step(ord("y"))  # "Really quit? [yn]"
    assert _levelfiles(env._hackdir) == [], (
        "game-over did not reach clearlocks(); the fixture is not exercising "
        "the failure mode"
    )


# --------------------------------------------------------------------------- #
# Defect 2: the bundler's filename prefix
# --------------------------------------------------------------------------- #

def test_bundle_captures_level_files_after_a_level_transition():
    """The signature regression: bundling must survive ``s_lock`` being rewritten.

    Pre-fix this asserted 0 bundled files for any game that had changed level.
    """
    env = _started()
    env.goto_depth(3)
    env.goto_depth(2)
    assert _levelfiles(env._hackdir), "test needs level files on disk to bundle"

    h = env.snapshot()
    try:
        n = env.snapshot_levelfile_count(h)
        assert n == len(_levelfiles(env._hackdir)) > 0, (
            f"snapshot bundled {n} level files but the hackdir holds "
            f"{_levelfiles(env._hackdir)}; the bundler's filename prefix is wrong"
        )
    finally:
        env.free_snapshot(h)
        env.end()


def test_restore_reverts_a_level_file_after_a_level_transition():
    """The same defect seen through behaviour rather than a counter.

    The pre-existing multilevel test does this on a FRESH game, where ``s_lock``
    is still the empty base and the prefix match happens to work. Doing it after
    a transition is what caught the bug.
    """
    env = _started(seed=42)
    env.goto_depth(2)
    env.goto_depth(1)  # s_lock is now ".1", not the base ""
    probe = os.path.join(env._hackdir, ".7")
    with open(probe, "wb") as f:
        f.write(b"ORIGINAL")

    h = env.snapshot()
    with open(probe, "wb") as f:
        f.write(b"MUTATED-AND-LONGER")
    env.restore(h)

    with open(probe, "rb") as f:
        assert f.read() == b"ORIGINAL", (
            "restore did not revert a level file that existed at snapshot time"
        )
    env.free_snapshot(h)
    env.end()


def test_restore_drops_level_files_created_after_the_snapshot():
    """Totality in the other direction: no stale futures survive a restore."""
    env = _started(seed=42)
    env.goto_depth(2)
    h = env.snapshot()
    at_snapshot = _levelfiles(env._hackdir)

    env.goto_depth(4)  # writes more level files
    assert set(_levelfiles(env._hackdir)) > set(at_snapshot)

    env.restore(h)
    assert _levelfiles(env._hackdir) == at_snapshot, (
        "after restore the on-disk level set must be exactly the snapshot's"
    )
    env.free_snapshot(h)
    env.end()


# --------------------------------------------------------------------------- #
# Defect 1 + 2 together: the actual rollout failure
# --------------------------------------------------------------------------- #

def test_game_over_wipes_every_level_file_from_disk():
    """The premise of the bug, pinned so it cannot silently change.

    ``done()`` -> ``clearlocks()`` unlinks the whole dungeon. Any restore-based
    revive therefore has to put every level file back; there is no subset that
    is good enough.
    """
    env = _started()
    env.goto_depth(3)
    env.goto_depth(2)
    assert _levelfiles(env._hackdir), "expected level files before the game-over"

    _game_over(env)  # asserts the wipe
    env.end()


def test_forced_revive_after_a_real_death_restores_the_whole_dungeon():
    """THE REPRO. Pre-fix: the stair transition below ends the run as TRICKED.

    Sequence is exactly the rollout's: snapshot (the revive ring snapshots
    after every tool call), real death, restore the ring's newest handle,
    materialize, then use the stairs.
    """
    env = _started()
    env.goto_depth(3)
    env.goto_depth(2)
    before = _levelfiles(env._hackdir)
    h = env.snapshot()

    _game_over(env)  # clearlocks() wipes the dungeon

    env.restore(h)
    env.step(27)  # ESC materializes the restored frame (as the revive does)

    assert _levelfiles(env._hackdir) == before, (
        "forced revive must put the whole dungeon back on disk"
    )
    assert env.missing_level_files() == 0

    # The post-revive stair transition that used to fake a death.
    env.goto_depth(3)
    assert not any(m in _msg(env) for m in TRICKED_FILE_MARKERS), _msg(env)
    assert int(env.blstats[12]) == 3, "stair transition did not happen"
    assert int(env.blstats[10]) > 0, "hero lost HP on a level transition"

    env.goto_depth(1)
    assert int(env.blstats[12]) == 1
    assert not env.done and env.how_done != HOW_TRICKED
    env.free_snapshot(h)
    env.end()


def test_snapshot_taken_after_a_death_still_carries_the_dungeon():
    """Snapshotting the dead game and reviving from it is also whole.

    The revive ring pushes a snapshot after EVERY tool call, including the one
    that kills the hero, so this ordering really occurs.
    """
    env = _started()
    env.goto_depth(2)
    env.goto_depth(3)
    h_alive = env.snapshot()
    alive_files = _levelfiles(env._hackdir)

    _game_over(env)
    h_dead = env.snapshot()  # bundles an empty dungeon: clearlocks() ran
    assert env.snapshot_levelfile_count(h_dead) == 0

    env.restore(h_alive)
    env.step(27)
    assert _levelfiles(env._hackdir) == alive_files
    assert env.missing_level_files() == 0
    env.free_snapshot(h_alive)
    env.free_snapshot(h_dead)
    env.end()


# --------------------------------------------------------------------------- #
# Property 1: a broken checkpoint is an ERROR, never a death
# --------------------------------------------------------------------------- #

def test_missing_level_file_is_detected_before_it_can_fake_a_death():
    """The integrity guard sees the divergence while the hero is still alive."""
    env = _started()
    env.goto_depth(3)
    env.goto_depth(2)
    assert env.missing_level_files() == 0

    victim = os.path.join(env._hackdir, _levelfiles(env._hackdir)[0])
    os.unlink(victim)

    assert env.missing_level_files() == 1
    with pytest.raises(CheckpointIntegrityError) as exc:
        env.check_level_file_integrity("unit test")
    assert "level files are missing" in str(exc.value)
    env.end()


def test_an_undetected_missing_level_file_really_does_present_as_a_death():
    """Why the guard exists, demonstrated rather than asserted from memory.

    Without the guard, the same corruption reaches the agent as ``how_done ==
    TRICKED`` with zeroed blstats — the shape the harness has always scored as
    a death.
    """
    env = _started()
    env.goto_depth(3)
    env.goto_depth(2)
    os.unlink(os.path.join(env._hackdir, ".3"))

    env.goto_depth(3)  # heap says level 3 exists; the file does not
    assert any(m in _msg(env) for m in TRICKED_FILE_MARKERS), _msg(env)

    for _ in range(12):
        if env.done:
            break
        env.step(13)
    assert env.done and env.how_done == HOW_TRICKED, (
        f"expected a TRICKED game-over, got done={env.done} how={env.how_done}"
    )
    assert int(env.blstats[10]) == 0 and int(env.blstats[12]) == 0, (
        "the fake death zeroes HP and Dlvl exactly like a real one"
    )
    env.end()
