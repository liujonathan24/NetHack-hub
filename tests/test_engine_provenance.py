"""tools/cli_harness_eval/engine_provenance.py + the launch_cell.sh preflight.

These tests never touch the real engine. Every case builds a SYNTHETIC engine
tree in tmp_path -- a git repo with a `third_party/NetHack` submodule-shaped
inner repo, a tracked C source, and a fake `src/build/libnethack.so` whose mtime
is set by hand -- so the verdicts are decided by the fixture, not by whether the
box happens to have a fresh build. That matters here: the real box is *currently*
stale (libnethack.so 2026-07-22 04:26 vs src/src/nle.c 2026-07-31 18:52), which
is exactly the condition under test, and a test that only passes while the box
is broken is worthless.

Order-independence: no module-level state, no shared tmp dir, no event loop, and
nothing that assumes anything about the engine's random starting role.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.cli_harness_eval import engine_provenance as ep  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/cli_harness_eval/launch_cell.sh"

EXPECTED_KEYS = {
    "captured_at",
    "engine_root",
    "engine_head",
    "engine_dirty",
    "submodule_sha",
    "submodule_pinned_sha",
    "submodule_matches_pin",
    "submodule_dirty",
    "so_path",
    "so_mtime",
    "so_mtime_iso",
    "so_size",
    "newest_source_path",
    "newest_source_mtime",
    "newest_source_mtime_iso",
    "so_stale",
    "so_stale_by_seconds",
    "errors",
}

# Two fixed epochs an hour apart. Absolute values, not "now - delta", so a slow
# test run can never reorder them.
T_OLD = 1_700_000_000
T_NEW = 1_700_003_600


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


def _make_engine(root: pathlib.Path, *, so_mtime: float, src_mtime: float):
    """A minimal engine tree: outer repo + inner `third_party/NetHack` repo.

    The inner repo is committed into the outer one as a real gitlink, so
    `git ls-tree HEAD third_party/NetHack` yields the pinned SHA the way the
    production layout does.
    """
    sub = root / "third_party" / "NetHack"
    (sub / "src" / "build").mkdir(parents=True)
    (sub / "src" / "src").mkdir(parents=True)

    src = sub / "src" / "src" / "nle.c"
    src.write_text("int main(void){return 0;}\n")
    (sub / "README.md").write_text("not a build input\n")
    # The build dir is ignored, as it is in the fork -- otherwise the .so itself
    # would show up as an untracked file and every fixture would read "dirty".
    (sub / ".gitignore").write_text("src/build/\n")

    _git_init(sub)
    _git(sub, "add", "-A")
    _git(sub, "commit", "-m", "engine")

    # Written AFTER the commit so it is a build product, not tracked content.
    so = sub / "src" / "build" / "libnethack.so"
    so.write_bytes(b"\x7fELF" + b"\0" * 64)
    os.utime(so, (so_mtime, so_mtime))
    os.utime(src, (src_mtime, src_mtime))

    _git_init(root)
    _git(root, "add", "third_party/NetHack")
    _git(root, "commit", "-m", "pin engine")
    return root, sub, so, src


def _git_init(repo):
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "protocol.file.allow", "always")


# ---------------------------------------------------------------------------
# fingerprint shape
# ---------------------------------------------------------------------------


def test_fingerprint_returns_the_expected_keys(tmp_path):
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    fp = ep.engine_fingerprint(root)
    assert set(fp) == EXPECTED_KEYS
    assert fp["engine_root"] == str(root)
    assert fp["so_path"] == str(so)
    assert fp["so_size"] == so.stat().st_size
    assert fp["so_mtime"] == T_NEW
    assert fp["newest_source_path"] == "src/src/nle.c"
    assert fp["newest_source_mtime"] == T_OLD
    assert len(fp["engine_head"]) == 40
    assert len(fp["submodule_sha"]) == 40
    assert fp["submodule_pinned_sha"] == fp["submodule_sha"]
    assert fp["submodule_matches_pin"] is True
    assert fp["so_stale"] is False
    assert fp["errors"] == []


def test_real_engine_fingerprint_is_shaped_the_same(tmp_path):
    """Against whatever engine this box actually has -- no verdict asserted,
    only that the helper produces the full key set and does not raise. The
    verdict is host state; the contract is not."""
    fp = ep.engine_fingerprint()
    assert set(fp) == EXPECTED_KEYS
    assert isinstance(fp["errors"], list)
    # Whatever it found, the tri-state fields must stay tri-state.
    assert fp["so_stale"] in (True, False, None)
    assert fp["submodule_dirty"] in (True, False, None)


def test_fingerprint_is_json_serialisable(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    json.dumps(ep.engine_fingerprint(root))


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------


def test_detects_a_so_older_than_source(tmp_path):
    """Incident 2: nle.c 2026-07-31 18:52, libnethack.so 2026-07-22 04:26."""
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    fp = ep.engine_fingerprint(root)
    assert fp["so_stale"] is True
    assert fp["so_stale_by_seconds"] == T_NEW - T_OLD
    reasons = ep.staleness_reasons(fp)
    assert any("STALE" in r for r in reasons)
    assert "src/src/nle.c" in " ".join(reasons)
    assert "STALE_SO" in ep.fingerprint_line(fp)


def test_a_newer_non_build_file_does_not_count_as_source(tmp_path):
    """A touched README must not block a launch -- the refusal costs a sweep, so
    only inputs the build actually consumes may trigger it."""
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_OLD)
    readme = sub / "README.md"
    os.utime(readme, (T_NEW, T_NEW))
    fp = ep.engine_fingerprint(root)
    assert fp["newest_source_path"] == "src/src/nle.c"
    assert fp["so_stale"] is False
    assert ep.staleness_reasons(fp) == []


def test_untracked_source_does_not_count(tmp_path):
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_OLD)
    scratch = sub / "src" / "src" / "scratch.c"
    scratch.write_text("/* not part of the engine */\n")
    os.utime(scratch, (T_NEW, T_NEW))
    fp = ep.engine_fingerprint(root)
    assert fp["newest_source_path"] == "src/src/nle.c"
    assert fp["so_stale"] is False


# ---------------------------------------------------------------------------
# submodule state
# ---------------------------------------------------------------------------


def test_detects_a_dirty_submodule(tmp_path):
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    src.write_text("int main(void){return 1;}\n")
    os.utime(src, (T_OLD, T_OLD))  # dirty but NOT newer, to isolate the signal
    fp = ep.engine_fingerprint(root)
    assert fp["submodule_dirty"] is True
    assert fp["so_stale"] is False
    reasons = ep.staleness_reasons(fp)
    assert len(reasons) == 1
    assert "uncommitted changes" in reasons[0]
    assert "DIRTY_SUBMODULE" in ep.fingerprint_line(fp)


def test_detects_a_submodule_off_its_pinned_commit(tmp_path):
    """Incident 1: submodule at 66c84e6, superproject pinning fefd557. Two
    *branched* games then returned byte-identical maps and ~30 tests failed as
    if the engine had logic bugs."""
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    pinned = ep._git(sub, "rev-parse", "HEAD")
    src.write_text("int main(void){return 2;}\n")
    _git(sub, "add", "-A")
    _git(sub, "commit", "-m", "drift")
    os.utime(src, (T_OLD, T_OLD))
    fp = ep.engine_fingerprint(root)
    assert fp["submodule_pinned_sha"] == pinned
    assert fp["submodule_sha"] != pinned
    assert fp["submodule_matches_pin"] is False
    assert fp["submodule_dirty"] is False
    assert any("pins" in r for r in ep.staleness_reasons(fp))
    assert "SUBMODULE_OFF_PIN" in ep.fingerprint_line(fp)


# ---------------------------------------------------------------------------
# degradation: never raise, never block on an unknown
# ---------------------------------------------------------------------------


def test_missing_engine_does_not_raise(tmp_path):
    fp = ep.engine_fingerprint(tmp_path / "does-not-exist")
    assert set(fp) == EXPECTED_KEYS
    assert fp["so_path"] is None
    assert fp["so_stale"] is None
    assert fp["errors"]
    # An unknown verdict must never refuse a launch.
    assert ep.staleness_reasons(fp) == []
    assert ep.check(fp, allow_stale=False) == 0


def test_engine_root_without_a_git_repo_does_not_raise(tmp_path):
    root = tmp_path / "eng"
    (root / "third_party" / "NetHack" / "src" / "build").mkdir(parents=True)
    (root / "third_party" / "NetHack" / "src" / "build" / "libnethack.so").write_bytes(b"x")
    fp = ep.engine_fingerprint(root)
    assert set(fp) == EXPECTED_KEYS
    assert fp["so_path"] is not None
    assert fp["so_stale"] is None
    assert ep.staleness_reasons(fp) == []


def test_an_unreachable_so_does_not_raise(tmp_path):
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    so.unlink()
    so.symlink_to(tmp_path / "nowhere")  # dangling: exists() False, stat() raises
    fp = ep.engine_fingerprint(root)
    assert set(fp) == EXPECTED_KEYS
    assert fp["so_stale"] is None
    assert ep.staleness_reasons(fp) == []


# ---------------------------------------------------------------------------
# check() / the bypass
# ---------------------------------------------------------------------------


def test_check_refuses_a_stale_engine(tmp_path, capsys):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    fp = ep.engine_fingerprint(root)
    assert ep.check(fp, allow_stale=False) == 3
    err = capsys.readouterr().err
    assert "launch refused" in err
    assert ep.BYPASS_ENV in err


def test_check_bypass_returns_zero_but_shouts(tmp_path, capsys):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    fp = ep.engine_fingerprint(root)
    assert ep.check(fp, allow_stale=True) == 0
    err = capsys.readouterr().err
    assert "ALLOW_STALE_ENGINE=1" in err
    assert "NOT reproducible" in err


def test_check_reads_the_bypass_from_the_environment(tmp_path, monkeypatch, capsys):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    fp = ep.engine_fingerprint(root)
    monkeypatch.delenv(ep.BYPASS_ENV, raising=False)
    assert ep.check(fp) == 3
    monkeypatch.setenv(ep.BYPASS_ENV, "1")
    assert ep.check(fp) == 0
    capsys.readouterr()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(args, env_extra=None, cwd=None):
    env = {k: v for k, v in os.environ.items() if k != ep.BYPASS_ENV}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(REPO / "tools/cli_harness_eval/engine_provenance.py"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd or REPO),
    )


def test_cli_check_exit_codes_and_json_dump(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    dump = tmp_path / "fp.json"
    r = _cli(["--engine-root", str(root), "--check", "--json", str(dump)])
    assert r.returncode == 3, r.stderr
    assert "[engine]" in r.stdout
    assert "launch refused" in r.stderr
    fp = json.loads(dump.read_text())
    assert set(fp) == EXPECTED_KEYS
    assert fp["so_stale"] is True

    r = _cli(["--engine-root", str(root), "--check"], env_extra={ep.BYPASS_ENV: "1"})
    assert r.returncode == 0, r.stderr
    assert ep.BYPASS_ENV in r.stderr


def test_cli_engine_root_from_the_environment(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    r = _cli(["--check"], env_extra={"ENGINE_PROVENANCE_ROOT": str(root)})
    assert r.returncode == 3, r.stdout + r.stderr


def test_cli_run_record_appends_an_engine_block(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    results = tmp_path / "results"
    results.mkdir()
    prov = results / "run_provenance.json"
    prov.write_text(json.dumps({"runs": [{"run": "old", "head": "deadbee"}], "cells": []}))
    r = _cli(["--engine-root", str(root), "--quiet", "--run-record", str(prov), "--run", "p9",
              "--driver", "p9.log", "--extra", "n=5"])
    assert r.returncode == 0, r.stderr
    doc = json.loads(prov.read_text())
    assert [e["run"] for e in doc["runs"]] == ["old", "p9"]      # existing entries preserved
    assert doc["cells"] == []
    new = doc["runs"][-1]
    assert new["extra"] == "n=5"
    assert set(new["engine"]) == EXPECTED_KEYS
    assert new["engine"]["submodule_sha"]


def test_run_record_creates_the_file_when_absent(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    prov = tmp_path / "results" / "run_provenance.json"
    ep.record_run(prov, run="fresh", fp=ep.engine_fingerprint(root))
    doc = json.loads(prov.read_text())
    assert len(doc["runs"]) == 1
    assert doc["runs"][0]["run"] == "fresh"


def test_run_record_survives_a_corrupt_file(tmp_path):
    prov = tmp_path / "run_provenance.json"
    prov.write_text("{not json")
    ep.record_run(prov, run="x", fp=ep.engine_fingerprint(tmp_path / "nope"))
    doc = json.loads(prov.read_text())
    assert doc["runs"][0]["run"] == "x"


# ---------------------------------------------------------------------------
# launch_cell.sh preflight
# ---------------------------------------------------------------------------

STUB_EVAL = """#!/usr/bin/env python3
import json, sys, os
with open(os.environ["STUB_EVAL_ARGV_FILE"], "w") as f:
    json.dump(sys.argv[1:], f)
"""


def _launch(tmp_path, engine_root, env_extra=None):
    stub = tmp_path / "bin" / "eval"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(STUB_EVAL)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    argv_file = tmp_path / "argv.json"
    out = tmp_path / "out"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "EVAL_BIN": str(stub),
        "STUB_EVAL_ARGV_FILE": str(argv_file),
        "ENGINE_PROVENANCE_ROOT": str(engine_root),
    }
    env.update(env_extra or {})
    result = subprocess.run(
        ["bash", str(SCRIPT), "claude_code", str(out), "20", "1"],
        capture_output=True,
        text=True,
        env=env,
    )
    argv = json.loads(argv_file.read_text()) if argv_file.exists() else None
    return result, argv, out


def test_launch_cell_refuses_a_stale_engine(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    result, argv, _out = _launch(tmp_path, root)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "STALE" in result.stderr
    assert argv is None, "the eval CLI must not have been reached"


def test_launch_cell_bypass_launches_with_a_loud_warning(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_OLD, src_mtime=T_NEW)
    result, argv, _out = _launch(tmp_path, root, env_extra={ep.BYPASS_ENV: "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert ep.BYPASS_ENV in result.stderr
    assert "NOT reproducible" in result.stderr
    assert argv is not None


def test_launch_cell_records_the_fingerprint_in_the_cell_output(tmp_path):
    root, *_ = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    result, argv, out = _launch(tmp_path, root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert argv is not None
    assert "[engine]" in result.stdout
    fp = json.loads((out / ep.CELL_FILENAME).read_text())
    assert set(fp) == EXPECTED_KEYS
    assert fp["submodule_sha"] == ep._git(root / "third_party" / "NetHack", "rev-parse", "HEAD")
    assert fp["so_stale"] is False


def test_launch_cell_refuses_a_dirty_submodule(tmp_path):
    root, sub, so, src = _make_engine(tmp_path / "eng", so_mtime=T_NEW, src_mtime=T_OLD)
    src.write_text("/* local hack */\n")
    os.utime(src, (T_OLD, T_OLD))
    result, argv, _out = _launch(tmp_path, root)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "uncommitted changes" in result.stderr
    assert argv is None
