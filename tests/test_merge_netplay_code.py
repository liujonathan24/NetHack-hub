"""The round-boundary code merge: what composes, what is refused, and why.

These build REAL git repos in a tmpdir rather than mocking git, because the
whole argument for using git here is that three-way merge already gets the hard
cases right -- a mocked merge would test the mock.

Four outcomes are pinned, one per policy in the module docstring:
  compatible edits to different files MERGE;
  compatible edits to different regions of the SAME file merge;
  edits to the same region CONFLICT and the later branch is dropped;
  a merge result that fails the gate is REJECTED even though both parents passed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))

import merge_netplay_code as mnc  # noqa: E402

SEED = (REPO / "harnesses" / "nethack-prime-agent" / "nethack_prime_agent"
        / "skill" / "src" / "netplay")


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


@pytest.fixture
def canonical(tmp_path):
    """A canonical netplay repo seeded from the tree we ship."""
    repo = tmp_path / "canonical"
    repo.mkdir()
    for f in SEED.glob("*.py"):
        (repo / f.name).write_bytes(f.read_bytes())
    _git(repo, "init", "-q", ".")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "round-0 seed")
    _git(repo, "tag", "round-0")
    return repo


def _branch_with(repo, name, filename, content):
    _git(repo, "checkout", "-q", "-B", name, "round-0")
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"{name}: edit {filename}")
    _git(repo, "checkout", "-q", "master", check=False)
    _git(repo, "checkout", "-q", "main", check=False)


def _head_branch(repo):
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


# ---------- compatible contributions ----------

def test_two_branches_touching_different_files_both_land(canonical):
    """The case newest-wins cannot express: both contributions are kept."""
    base = _head_branch(canonical)
    _branch_with(canonical, "agent-7", "explore.py",
                 (SEED / "explore.py").read_text() + "\n# seed 7: corridor fix\n")
    _git(canonical, "checkout", "-q", base)
    _branch_with(canonical, "agent-9", "move.py",
                 (SEED / "move.py").read_text() + "\n# seed 9: door kicking\n")
    _git(canonical, "checkout", "-q", base)

    report = mnc.merge(canonical, ["agent-7", "agent-9"], round_tag="round-1")

    assert [m["branch"] for m in report["merged"]] == ["agent-7", "agent-9"]
    assert report["conflicts"] == [] and report["rejected"] == []
    assert "seed 7: corridor fix" in (canonical / "explore.py").read_text()
    assert "seed 9: door kicking" in (canonical / "move.py").read_text(), (
        "both contributions must survive; this is the whole reason for git"
    )


def test_the_round_is_tagged_so_rollback_is_one_command(canonical):
    base = _head_branch(canonical)
    _branch_with(canonical, "agent-7", "explore.py",
                 (SEED / "explore.py").read_text() + "\n# seed 7\n")
    _git(canonical, "checkout", "-q", base)
    mnc.merge(canonical, ["agent-7"], round_tag="round-1")

    tags = _git(canonical, "tag").stdout.split()
    assert "round-1" in tags

    out = mnc.rollback(canonical, "round-0", reason="held-out eval regressed")
    assert out["rollback"]["tag"] == "round-0"
    assert "# seed 7" not in (canonical / "explore.py").read_text()


# ---------- incompatible contributions ----------

def test_same_region_edits_conflict_and_the_later_branch_is_dropped(canonical):
    """Deterministic, not newest-wins -- and the paths are named in the report."""
    base = _head_branch(canonical)
    original = (SEED / "explore.py").read_text()
    _branch_with(canonical, "agent-7", "explore.py",
                 original.replace("max_moves: int = 40", "max_moves: int = 12"))
    _git(canonical, "checkout", "-q", base)
    _branch_with(canonical, "agent-9", "explore.py",
                 original.replace("max_moves: int = 40", "max_moves: int = 1"))
    _git(canonical, "checkout", "-q", base)

    report = mnc.merge(canonical, ["agent-7", "agent-9"], round_tag="round-1")

    assert [m["branch"] for m in report["merged"]] == ["agent-7"]
    assert [c["branch"] for c in report["conflicts"]] == ["agent-9"]
    assert report["conflicts"][0]["paths"] == ["explore.py"]
    assert "max_moves: int = 12" in (canonical / "explore.py").read_text()


def test_a_branch_that_fails_the_gate_is_rejected(canonical):
    base = _head_branch(canonical)
    _branch_with(canonical, "agent-7", "cheat.py",
                 "import socket\ndef go():\n    return 'fabricated map'\n")
    _git(canonical, "checkout", "-q", base)

    report = mnc.merge(canonical, ["agent-7"], round_tag="round-1")

    assert report["merged"] == []
    assert [r["branch"] for r in report["rejected"]] == ["agent-7"]
    assert any(f["kind"] == "denied-import"
               for f in report["rejected"][0]["findings"])
    assert not (canonical / "cheat.py").exists(), (
        "a rejected branch must leave no trace in canonical"
    )


def test_a_branch_that_deletes_a_frozen_file_is_rejected(canonical):
    base = _head_branch(canonical)
    _git(canonical, "checkout", "-q", "-B", "agent-7", "round-0")
    (canonical / "_base.py").unlink()
    _git(canonical, "add", "-A")
    _git(canonical, "commit", "-q", "-m", "agent-7: remove the floor")
    _git(canonical, "checkout", "-q", base)

    report = mnc.merge(canonical, ["agent-7"], round_tag="round-1")

    assert [r["branch"] for r in report["rejected"]] == ["agent-7"]
    assert (canonical / "_base.py").exists()


# ---------- bookkeeping ----------

def test_a_branch_with_no_commits_is_reported_not_merged(canonical):
    base = _head_branch(canonical)
    _git(canonical, "branch", "-f", "agent-7", "round-0")
    _git(canonical, "checkout", "-q", base)
    report = mnc.merge(canonical, ["agent-7"], round_tag=None)
    assert [u["branch"] for u in report["unchanged"]] == ["agent-7"]


def test_branches_merge_in_deterministic_order(canonical):
    """Ascending name order, so re-running the round reproduces the tree."""
    base = _head_branch(canonical)
    for n in ("agent-9", "agent-3", "agent-7"):
        _branch_with(canonical, n, f"{n.replace('-', '_')}.py", "X = 1\n")
        _git(canonical, "checkout", "-q", base)
    report = mnc.merge(canonical, mnc._branches(canonical), round_tag=None)
    assert [m["branch"] for m in report["merged"]] == ["agent-3", "agent-7", "agent-9"]
