"""E16 GATE — checkpoints that survive process death.

This suite exists because this project has been burned by smoke-only
validation, so the central test does NOT fake a fresh process: it saves in one
Python interpreter, lets that interpreter exit, and resumes in a second one
launched from scratch. Nothing but the bytes in the checkpoint directory
crosses the boundary.

What is being defended:

* fidelity at the restore point — status, map, inventory, dlvl;
* the ability to keep PLAYING afterwards, including changing dungeon level,
  which is where a lossy checkpoint fails: the player blob carries the whole
  ``level_info[]`` array (``save_dungeon``, dungeon.c:175), so the resumed heap
  believes in every level the original game visited. If those level files did
  not travel with the checkpoint, the first staircase hands the run to
  ``done(TRICKED)`` — a full-HP game-over that reads downstream as a monster
  kill. That is the exact failure that corrupted three E15 rollouts.
* atomicity — a checkpoint directory appears whole or not at all.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from nethack_harness.checkpoints import (
    META_JSON,
    PREFIX_JSONL,
    STATE_BUNDLE,
    LESSONS_MD,
    CheckpointIntegrityError,
    checkpoint_list,
    checkpoint_meta,
    checkpoint_restore,
    checkpoint_save,
    pack_bundle,
    unpack_bundle,
)

HOW_TRICKED = 12


def _run_in_fresh_process(body: str, tmp_path) -> dict:
    """Execute ``body`` in a brand-new interpreter; return its JSON stdout tail.

    The child inherits this process's PYTHONPATH (the eval cells' layout), so
    it imports the same engine and harness modules — but it shares no memory,
    no engine context, and no hackdir with us.
    """
    script = textwrap.dedent(
        """
        import json, os, sys
        TMP = sys.argv[1]
        def emit(d):
            sys.stdout.write("@@RESULT@@" + json.dumps(d) + "\\n")
        """
    ) + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True, text=True, timeout=600,
        env=dict(os.environ),
    )
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT@@")]
    assert marker, (
        "child process produced no result\n"
        f"--- exit {proc.returncode} ---\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(marker[-1][len("@@RESULT@@"):])


# --------------------------------------------------------------------------- #
# container format
# --------------------------------------------------------------------------- #

def test_bundle_round_trips_and_rejects_truncation():
    sections = {"level": b"L" * 40, "player": b"P" * 10, "levelfiles": b"F" * 7}
    blob = pack_bundle(sections, header_extra={"seed": [1, 1]})
    header, back = unpack_bundle(blob)
    assert back == sections and header["seed"] == [1, 1]

    with pytest.raises(CheckpointIntegrityError):
        unpack_bundle(blob[:-3])          # a torn tail is an error...
    with pytest.raises(CheckpointIntegrityError):
        unpack_bundle(b"not a bundle")    # ...and so is a foreign file


# --------------------------------------------------------------------------- #
# THE GATE: save here, resume in a process that never saw the original game
# --------------------------------------------------------------------------- #

_SAVE_BODY = """
    from nethack_core.engine_env import EngineEnv
    from nethack_core.observations import BLSTATS_IDX
    from nethack_harness.checkpoints import checkpoint_save

    env = EngineEnv()
    env.reset(seeds=(1, 1))
    env.step(13)                      # drain the welcome --More--
    env._engine.goto_depth(3)         # visit a few floors so the checkpoint
    env._engine.goto_depth(2)         # has off-current level files to carry
    for _ in range(5):
        env.step(ord("s"))

    raw = env._engine
    bl = raw.blstats
    fidelity = {
        "status": {k: int(bl[i]) for k, i in BLSTATS_IDX.items()},
        "glyphs": list(map(int, raw.glyphs.reshape(-1))),
        "chars": list(map(int, raw.chars.reshape(-1))),
        "inv_strs": [bytes(r).split(b"\\0")[0].decode("ascii", "replace")
                     for r in raw._inv_strs.reshape(55, 80)],
        "inv_letters": list(map(int, raw._inv_letters)),
    }
    meta = checkpoint_save(env, os.path.join(TMP, "archive", "run", "c1"),
                           name="gate", note="cross-process resume gate",
                           created_by="test_persistent_checkpoint")
    # AFTER the save: nle_save_level writes the current floor to its own
    # <base>.<ledger> file as part of serializing it, so the level-file set the
    # checkpoint captures includes the current level. That is the set the
    # resume must reproduce.
    fidelity["levelfiles"] = sorted(f for f in os.listdir(raw._hackdir)
                                    if f.startswith(".") and f[1:].isdigit())
    json.dump(fidelity, open(os.path.join(TMP, "fidelity.json"), "w"))
    emit({"meta": meta})
"""

_RESUME_BODY = """
    from nethack_core.observations import BLSTATS_IDX
    from nethack_harness.checkpoints import checkpoint_restore
    from nethack_harness.integrity import missing_level_files

    want = json.load(open(os.path.join(TMP, "fidelity.json")))
    env, meta = checkpoint_restore(os.path.join(TMP, "archive", "run", "c1"))
    raw = env._engine
    bl = raw.blstats

    got = {
        "status": {k: int(bl[i]) for k, i in BLSTATS_IDX.items()},
        "glyphs": list(map(int, raw.glyphs.reshape(-1))),
        "chars": list(map(int, raw.chars.reshape(-1))),
        "inv_strs": [bytes(r).split(b"\\0")[0].decode("ascii", "replace")
                     for r in raw._inv_strs.reshape(55, 80)],
        "inv_letters": list(map(int, raw._inv_letters)),
        "levelfiles": sorted(f for f in os.listdir(raw._hackdir)
                             if f.startswith(".") and f[1:].isdigit()),
    }
    diffs = {k: [want["status"][k], got["status"][k]]
             for k in want["status"] if want["status"][k] != got["status"][k]}

    # ---- keep playing: 50+ engine steps spanning a level change ------------
    raw.step(13)                       # drain this game's welcome --More--
    steps, tricked, msgs = 0, False, []
    MARKERS = ("Cannot open file", "Probably someone removed it",
               "Strange, this map is not as I remember it")
    def note():
        m = bytes(raw.message).split(b"\\0")[0].decode("ascii", "replace")
        if m:
            msgs.append(m)
        return any(x in m for x in MARKERS)

    depths = [int(raw.blstats[BLSTATS_IDX["depth"]])]
    for i in range(60):
        raw.step(ord("s"))
        steps += 1
        tricked |= note()
        if i in (20, 40):
            # A real stair transition mid-run, onto a level the ORIGINAL run
            # visited. ESC first: a pending combat --More-- would otherwise eat
            # the wait keystroke goto_depth uses to run its deferred goto, and
            # the transition would silently not happen.
            raw.step(27)
            raw.goto_depth(1 if i == 20 else 3)
            steps += 2
            tricked |= note()
            depths.append(int(raw.blstats[BLSTATS_IDX["depth"]]))

    emit({
        "meta": meta,
        "status_diffs": diffs,
        "glyphs_match": want["glyphs"] == got["glyphs"],
        "chars_match": want["chars"] == got["chars"],
        "inv_match": (want["inv_strs"] == got["inv_strs"]
                      and want["inv_letters"] == got["inv_letters"]),
        "levelfiles_want": want["levelfiles"],
        "levelfiles_got": got["levelfiles"],
        "steps": steps,
        "depths": depths,
        "tricked": tricked,
        "done": bool(raw.done),
        "how_done": int(raw.how_done),
        "hp_after": int(raw.blstats[BLSTATS_IDX["hitpoints"]]),
        "missing_level_files": missing_level_files(env),
        "messages": msgs[-5:],
    })
"""


def test_checkpoint_survives_process_death_and_the_game_keeps_playing(tmp_path):
    """(i) save mid-game, (ii) tear the process down, (iii) resume fresh,
    (iv) play 50+ steps including two level changes."""
    saved = _run_in_fresh_process(_SAVE_BODY, tmp_path)
    ck = tmp_path / "archive" / "run" / "c1"
    assert ck.is_dir(), "checkpoint directory was not created"
    for f in (STATE_BUNDLE, META_JSON, PREFIX_JSONL, LESSONS_MD):
        assert (ck / f).is_file(), f"checkpoint is missing {f}"
    assert saved["meta"]["dlvl"] == 2 and saved["meta"]["name"] == "gate"
    assert (ck / PREFIX_JSONL).read_bytes() == b"", (
        "prefix.jsonl is reserved for the harness and must start empty"
    )

    # (iii) a brand-new interpreter. Nothing but the directory crosses over.
    out = _run_in_fresh_process(_RESUME_BODY, tmp_path)

    # ---- fidelity at the restore point -----------------------------------
    assert out["status_diffs"] == {}, f"status drifted on resume: {out['status_diffs']}"
    assert out["glyphs_match"], "the restored map differs from the saved one"
    assert out["chars_match"], "the restored map glyph characters differ"
    assert out["inv_match"], "the restored inventory differs from the saved one"
    assert out["levelfiles_got"] == out["levelfiles_want"], (
        "the resumed hackdir does not hold the same dungeon level files"
    )
    assert out["missing_level_files"] == 0

    # ---- and it is a real, playable game ---------------------------------
    assert out["steps"] >= 50, out["steps"]
    assert out["depths"] == [2, 1, 3], (
        f"level changes did not happen after resume: {out['depths']}"
    )
    assert not out["tricked"], (
        f"a TRICKED banner fired after resume: {out['messages']}"
    )
    assert out["how_done"] != HOW_TRICKED
    assert out["hp_after"] > 0, "the hero was killed by the resume itself"


def test_visits_counter_increments_across_resumes(tmp_path):
    """meta.json is mutable after the fact, and mutably atomic."""
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.reset(seeds=(1, 1))
    env.step(13)
    env._engine.goto_depth(2)
    ck = tmp_path / "archive" / "run" / "c1"
    meta = checkpoint_save(env, ck, name="v", note="")
    assert meta["visits"] == 0

    _, m1 = checkpoint_restore(ck)
    _, m2 = checkpoint_restore(ck)
    assert m1["visits"] == 1 and m2["visits"] == 2
    assert checkpoint_meta(ck)["visits"] == 2


def test_meta_records_the_full_e16_field_set(tmp_path):
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.reset(seeds=(1, 1))
    env.step(13)
    env._engine.goto_depth(4)
    env._engine.goto_depth(2)
    ck = tmp_path / "archive" / "run" / "c7"
    meta = checkpoint_save(
        env, ck, name="deep", note="try the mines from here",
        parent="3", created_by="agent-a", max_dlvl=4, max_xl=1,
        attempts_from=2, lessons="mines are east\n",
    )
    for field in ("id", "parent", "name", "note", "dlvl", "xl", "hp", "max_hp",
                  "gameturn", "score", "balrog", "balrog_min", "visits",
                  "attempts_from", "created_by", "max_dlvl_reached",
                  "max_xp_level", "balrog_metric"):
        assert field in meta, f"meta.json is missing {field}"
    assert meta["id"] == "7" and meta["parent"] == "3"
    assert meta["dlvl"] == 2
    # THE PAIR IS THE HOUSE METRIC, and this assertion was wrong before.
    #
    # It used to read `balrog > balrog_min > 0`, encoding an invented meaning
    # ("balrog_min is what this checkpoint alone guarantees on resume") on top
    # of `progression_score` -- the analytic proxy whose own module says
    # "DEPRECATED ... do not quote it as BALROG". Two errors at once: the wrong
    # function, and a `min` that was not BALROG's min. `balrog_both` is what
    # nethack_v1's finalize publishes as balrog_pct / balrog_min_pct: max and
    # min over the (Dlvl, XL) axes of the real achievement table. At Dlvl 4 /
    # XL 1 the min is legitimately 0.0 -- an XL-1 character has reached no Xp
    # achievement -- and that zero is the SIGNAL, not a bug: it says this state
    # is deep-but-unlevelled.
    from nethack_harness.prompt.balrog import balrog_both
    hi, lo = balrog_both(4, 1)
    assert (meta["balrog"], meta["balrog_min"]) == (hi, lo)
    assert meta["balrog"] > meta["balrog_min"] == 0.0
    assert meta["max_dlvl_reached"] == 4 and meta["max_xp_level"] == 1
    assert (ck / LESSONS_MD).read_text() == "mines are east\n"
    assert checkpoint_list(ck.parent) == [ck]


# --------------------------------------------------------------------------- #
# atomicity
# --------------------------------------------------------------------------- #

def test_a_crash_mid_write_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """Interrupt a save halfway: the old checkpoint must still be restorable."""
    from nethack_core.engine_env import EngineEnv
    import nethack_harness.checkpoints as ckmod

    env = EngineEnv()
    env.reset(seeds=(1, 1))
    env.step(13)
    env._engine.goto_depth(3)
    env._engine.goto_depth(2)

    root = tmp_path / "archive" / "run"
    good = root / "c1"
    checkpoint_save(env, good, name="good", note="the one that must survive")
    good_bytes = (good / STATE_BUNDLE).read_bytes()
    good_meta = checkpoint_meta(good)

    # Crash after the bundle is written but before the directory is swapped in.
    real_write = ckmod._write_file
    calls = {"n": 0}

    def exploding_write(path, data):
        calls["n"] += 1
        if calls["n"] == 2:  # bundle written, meta.json half-way
            raise OSError("simulated power loss")
        return real_write(path, data)

    monkeypatch.setattr(ckmod, "_write_file", exploding_write)
    with pytest.raises(OSError):
        checkpoint_save(env, root / "c2", name="doomed", note="")
    monkeypatch.undo()

    # The interrupted checkpoint left NO directory of its own...
    assert not (root / "c2").exists()
    assert [p.name for p in checkpoint_list(root)] == ["c1"], (
        "a half-written checkpoint must not be listed as a checkpoint"
    )
    # ...no staging debris survives the failure...
    assert [p.name for p in root.iterdir()] == ["c1"], sorted(
        p.name for p in root.iterdir()
    )
    # ...and the previous checkpoint is byte-identical and still restorable.
    assert (good / STATE_BUNDLE).read_bytes() == good_bytes
    assert checkpoint_meta(good)["name"] == good_meta["name"]
    env2, _ = checkpoint_restore(good)
    from nethack_core.observations import BLSTATS_IDX
    assert int(env2._engine.blstats[BLSTATS_IDX["depth"]]) == 2


def test_overwriting_a_checkpoint_is_refused_by_default(tmp_path):
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.reset(seeds=(1, 1))
    env.step(13)
    ck = tmp_path / "c1"
    checkpoint_save(env, ck, name="first", note="")
    with pytest.raises(FileExistsError):
        checkpoint_save(env, ck, name="second", note="")
    assert checkpoint_meta(ck)["name"] == "first"

    checkpoint_save(env, ck, name="second", note="", overwrite=True)
    assert checkpoint_meta(ck)["name"] == "second"


def test_restoring_a_bundle_without_its_level_files_is_an_error(tmp_path):
    """The lossy-checkpoint trap, refused at the door rather than at a staircase."""
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.reset(seeds=(1, 1))
    env.step(13)
    env._engine.goto_depth(3)
    env._engine.goto_depth(2)
    ck = tmp_path / "c1"
    checkpoint_save(env, ck, name="lossy", note="")

    header, sections = unpack_bundle((ck / STATE_BUNDLE).read_bytes())
    sections.pop("levelfiles")
    (ck / STATE_BUNDLE).write_bytes(
        pack_bundle(sections, header_extra={"seed": header["seed"]})
    )
    with pytest.raises(CheckpointIntegrityError) as exc:
        checkpoint_restore(ck)
    assert "levelfiles" in str(exc.value)


# --------------------------------------------------------------------------- #
# the `save` skill stub
# --------------------------------------------------------------------------- #

def test_save_skill_is_registered_but_published_in_no_tier():
    from nethack_harness.helpers import _build_skill_adapter_callables
    from nethack_harness.tools.skills import list_skills

    assert "save" in list_skills(), "save must be dispatchable"
    for skill_set in ("full", "netplay", "netplay_true", "np_core", "dir8",
                      "move", "balrog80", "np_core,request_map,search,rollback"):
        names = {getattr(a, "__name__", "")
                 for a in _build_skill_adapter_callables(skill_set=skill_set)}
        assert "save" not in names, (
            f"save leaked into the published tool surface for {skill_set!r}; "
            f"publishing it must be a deliberate edit, not a side effect of "
            f"registering it"
        )


def test_save_skill_writes_a_checkpoint_and_names_it(tmp_path):
    from nethack_core.engine_env import EngineEnv
    from nethack_harness.tools.skills import registry, CHECKPOINT_ARCHIVE_ATTR

    env = EngineEnv()
    env.reset(seeds=(1, 1))
    env.step(13)
    env._engine.goto_depth(2)

    # No archive configured -> a clear refusal, never a crash.
    res = registry.call("save", env, None, label="x", note="")
    assert "no checkpoint archive" in res.feedback and res.actions == []

    setattr(env, CHECKPOINT_ARCHIVE_ATTR, str(tmp_path / "archive" / "run"))
    res = registry.call("save", env, None, label="before mines",
                        note="try the branch")
    assert res.actions == [], "save must not consume game time"
    assert "saved checkpoint 1" in res.feedback and "before mines" in res.feedback
    ck = tmp_path / "archive" / "run" / "c1"
    assert checkpoint_meta(ck)["note"] == "try the branch"

    # A second save takes the next free id rather than clobbering the first.
    registry.call("save", env, None, label="second", note="")
    assert {p.name for p in checkpoint_list(tmp_path / "archive" / "run")} == {"c1", "c2"}
