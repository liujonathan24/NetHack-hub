"""`netplay_code_mode`: what lands in the skill tree, and what survives a rollout.

The three modes are the experiment's arms, and the difference between them is
one branch in `_materialise_netplay`. These tests pin that branch:

  `off`      -- byte-identical to every arm that shipped before this feature.
  `frozen`   -- the whole tree rewritten each rollout ([continual-code-frozen]).
  `mutable`  -- seed composites kept when present, so code accumulates.

The byte-identity test is written so that it CAN fail: SPEC §6.3 records a
previous sync test that could not detect any of four drift mutations including
deleting the whole block, so this one mutates the tree and asserts it is caught.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import types

import pytest
from importlib import resources

from nethack_prime_agent import (
    _NETPLAY_FROZEN_FILES,
    _NETPLAY_SEED_FILES,
    _SKILL_FILES,
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)

SKILL_DIR = "/tmp/vf-prime-agent/skills/nethack"


class _SetupRuntime:
    """Records `write`, and serves `read` from what has been written.

    `read` matters here: "write the seed only if absent" is implemented on top
    of it, so a runtime that cannot read would make every mode look like
    `frozen` and the mutable test would pass for the wrong reason.
    """

    def __init__(self, preexisting: dict[str, bytes] | None = None):
        self.files: dict[str, bytes] = dict(preexisting or {})
        self.written: list[str] = []
        self.commands: list[list[str]] = []

    async def write(self, path, data):
        self.files[path] = data
        self.written.append(path)

    async def read(self, path):
        return self.files[path]          # KeyError when absent, as a real one raises

    async def run(self, argv, env):
        from verifiers.v1.runtimes import ProgramResult

        self.commands.append(argv)
        return ProgramResult(exit_code=0, stdout="prime-agent 0.3.3", stderr="")


def _setup(runtime=None, **overrides):
    """Run `PrimeAgentHarness.setup` and return the runtime it wrote to."""
    overrides.setdefault("sandbox", False)
    cfg = PrimeAgentHarnessConfig(
        id="nethack-prime-agent",
        install_dir="/tmp/vf-prime-agent",
        **overrides,
    )
    harness = PrimeAgentHarness(cfg)
    runtime = runtime if runtime is not None else _SetupRuntime()
    asyncio.run(harness.setup(runtime))
    return runtime


def _skill_tree(runtime):
    return {k: v for k, v in runtime.files.items() if k.startswith(SKILL_DIR)}


# ---------- `off`: the byte-identity contract ----------

def test_off_is_the_default():
    assert PrimeAgentHarnessConfig(id="x").netplay_code_mode == "off"


def test_off_writes_exactly_the_three_files_that_always_shipped():
    tree = _skill_tree(_setup())
    assert set(tree) == {f"{SKILL_DIR}/{n}" for n in _SKILL_FILES}


def test_off_writes_no_netplay_file_at_all():
    tree = _skill_tree(_setup())
    assert not [p for p in tree if "netplay" in p], (
        "netplay_code_mode='off' must leave `base` byte-identical to the arms "
        "that ran before this feature existed"
    )


def test_the_byte_identity_check_can_actually_fail():
    """A drift test that cannot fail is not a test -- SPEC §6.3."""
    tree = _skill_tree(_setup())
    mutated = dict(tree)
    mutated[f"{SKILL_DIR}/src/netplay/explore.py"] = b"# smuggled in\n"
    assert [p for p in mutated if "netplay" in p], (
        "the mutation itself must be visible, or the assertion above is vacuous"
    )
    assert set(mutated) != {f"{SKILL_DIR}/{n}" for n in _SKILL_FILES}


# ---------- `frozen`: the control arm ----------

def test_frozen_materialises_the_whole_tree():
    tree = _skill_tree(_setup(netplay_code_mode="frozen"))
    expected = {f"{SKILL_DIR}/{n}" for n in
                (*_SKILL_FILES, *_NETPLAY_FROZEN_FILES, *_NETPLAY_SEED_FILES)}
    assert set(tree) == expected


def test_frozen_reverts_an_agent_edit():
    """The control accumulates nothing -- that is what makes it a control."""
    edited = {f"{SKILL_DIR}/src/netplay/explore.py": b"# agent's generation 3\n"}
    runtime = _setup(_SetupRuntime(edited), netplay_code_mode="frozen")
    assert runtime.files[f"{SKILL_DIR}/src/netplay/explore.py"] != b"# agent's generation 3\n"


# ---------- `mutable`: the git base-restore treatment ----------
#
# mutable genuinely runs git now, so these drive setup against a REAL temp dir
# with real git rather than the recording runtime. That makes them the
# end-to-end smoke for the persistence model as well as its unit test.

class _RealDirRuntime:
    """A runtime backed by a real directory: writes hit disk, `run` shells out.

    Only the surface `_materialise_netplay` and the teardown use -- write, read,
    run -- is implemented. `run` executes the argv for real, so git actually
    creates commits and branches.
    """

    def __init__(self, root):
        self.root = pathlib.Path(root)

    async def write(self, path, data):
        f = pathlib.Path(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)

    async def read(self, path):
        return pathlib.Path(path).read_bytes()

    async def run(self, argv, env):
        import subprocess
        from verifiers.v1.runtimes import ProgramResult
        proc = subprocess.run(argv, capture_output=True, text=True,
                              env={**os.environ, **(env or {})})
        return ProgramResult(exit_code=proc.returncode, stdout=proc.stdout,
                             stderr=proc.stderr)


def _real_setup(root, **overrides):
    # Default to the mutable git flow -- the whole reason this helper drives a
    # real directory. sandbox=False because the git base-restore is independent
    # of the bwrap sandbox and the recording runtime has no workdir for it.
    overrides.setdefault("sandbox", False)
    overrides.setdefault("netplay_code_mode", "mutable")
    cfg = PrimeAgentHarnessConfig(id="nethack-prime-agent",
                                  install_dir=str(root), **overrides)
    harness = PrimeAgentHarness(cfg)
    runtime = _RealDirRuntime(root)
    asyncio.run(harness.setup(runtime))
    return harness, runtime


def _tree(root):
    return pathlib.Path(root) / "skills" / "nethack" / "src" / "netplay"


def _git(repo, *args):
    import subprocess
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout.strip()


def test_mutable_first_rollout_commits_a_pristine_base(tmp_path):
    """The base has to be the seeds, not the first agent's edits -- every
    rollout of round 1 starts from it."""
    _real_setup(tmp_path)
    tree = _tree(tmp_path)
    assert (tree / ".git").is_dir()
    assert _git(tree, "rev-parse", "--abbrev-ref", "HEAD") == "netplay-canonical"
    assert "round-0" in _git(tree, "tag").split()
    # the committed base is our shipped seed, byte-for-byte
    committed = _git(tree, "show", "netplay-canonical:explore.py")
    assert "YOURS TO EDIT" in committed


def test_mutable_later_rollout_starts_from_base_not_the_previous_edit(tmp_path):
    """Independence: an edit left by one rollout must be gone at the next
    rollout's setup, so branches are independent contributions off one base."""
    harness, runtime = _real_setup(tmp_path)
    tree = _tree(tmp_path)
    # simulate a rollout: edit, and commit it on an agent branch (as teardown would)
    (tree / "explore.py").write_text("# agent generation 1")
    import subprocess
    subprocess.run(["git", "-C", str(tree), "checkout", "-q", "-B", "agent-1"])
    subprocess.run(["git", "-C", str(tree), "add", "-A"])
    subprocess.run(["git", "-C", str(tree), "commit", "-q", "-m", "gen1"])
    subprocess.run(["git", "-C", str(tree), "checkout", "-q", "netplay-canonical"])
    # next rollout's setup must restore explore.py to the pristine base
    asyncio.run(harness.setup(runtime))
    assert "YOURS TO EDIT" in (tree / "explore.py").read_text()
    assert "agent generation 1" not in (tree / "explore.py").read_text()
    # ...but the agent-1 branch still holds the contribution for the merge
    assert "agent generation 1" in _git(tree, "show", "agent-1:explore.py")


def test_mutable_reheals_the_frozen_boundary_on_a_later_rollout(tmp_path):
    """Even if a merged generation ever carried a tampered _base.py past the
    gate, setup rewrites it from the repo on the next rollout."""
    harness, runtime = _real_setup(tmp_path)
    tree = _tree(tmp_path)
    # tamper and COMMIT into base (simulating a bad merge that reached canonical)
    (tree / "_base.py").write_text("# neutered floor")
    import subprocess
    subprocess.run(["git", "-C", str(tree), "commit", "-qam", "tampered base"])
    asyncio.run(harness.setup(runtime))
    assert "# neutered floor" not in (tree / "_base.py").read_text()
    assert "the primitive floor" in (tree / "_base.py").read_text().lower()


def test_mutable_never_rewrites_pyproject(tmp_path):
    """Editing pyproject.toml forces a ~10 min venv rebuild; it is written once."""
    _real_setup(tmp_path)
    pyproj = pathlib.Path(tmp_path) / "skills" / "nethack" / "pyproject.toml"
    assert pyproj.read_bytes() == (
        resources.files("nethack_prime_agent") / "skill" / "pyproject.toml"
    ).read_bytes()


def test_an_unknown_mode_is_refused_rather_than_silently_off(tmp_path):
    with pytest.raises(ValueError, match="netplay_code_mode"):
        _real_setup(tmp_path, netplay_code_mode="mutabel")


# ---------- `pinned`: held-out evaluation ----------

def _seed_pinned_tree(root):
    """Place a full agent tree at the fixed skills path, as eval_frozen would."""
    tree = _tree(root)
    tree.mkdir(parents=True)
    pkg = resources.files("nethack_prime_agent") / "skill" / "src" / "netplay"
    for f in pathlib.Path(str(pkg)).glob("*.py"):
        (tree / f.name).write_bytes(f.read_bytes())
    return tree


def test_pinned_materialises_nothing_under_netplay(tmp_path):
    """The artifact under test must be exactly what the evaluator placed."""
    tree = _seed_pinned_tree(tmp_path)
    (tree / "explore.py").write_text("# the agent's final generation 5")
    before = {f.name: f.read_bytes() for f in tree.glob("*.py")}
    _real_setup(tmp_path, netplay_code_mode="pinned")
    after = {f.name: f.read_bytes() for f in tree.glob("*.py")}
    assert after == before, "pinned mode rewrote the tree it was meant to evaluate"


def test_pinned_without_a_tree_is_refused_not_silently_empty(tmp_path):
    """The failure this mode exists to prevent: evaluating a code arm with no
    code and reporting the number as the arm's."""
    with pytest.raises(ValueError, match="pinned"):
        _real_setup(tmp_path, netplay_code_mode="pinned")


def test_pinned_with_only_base_but_no_composites_is_refused(tmp_path):
    """A tree that imports but has no policies is the silent-empty failure the
    presence check was widened to catch."""
    tree = _tree(tmp_path)
    tree.mkdir(parents=True)
    pkg = resources.files("nethack_prime_agent") / "skill" / "src" / "netplay"
    for n in ("_base.py", "__init__.py"):
        (tree / n).write_bytes((pathlib.Path(str(pkg)) / n).read_bytes())
    with pytest.raises(ValueError, match="incomplete"):
        _real_setup(tmp_path, netplay_code_mode="pinned")


def test_pinned_serves_the_code_document_not_the_baseline(tmp_path):
    """off would also write nothing -- but serve the baseline doc, which
    advertises the retired composites as callable. That is why pinned exists."""
    from nethack_prime_agent import _skill_doc
    pkg = resources.files("nethack_prime_agent") / "skill"
    pinned = _skill_doc(pkg, skill_doc_coords=False, allow_batching=False,
                        netplay_code_mode="pinned")
    off = _skill_doc(pkg, skill_doc_coords=False, allow_batching=False,
                     netplay_code_mode="off")
    assert pinned != off
    flat = " ".join(pinned.decode().split())
    assert "complete set of tools" in flat
    # the retired composites are gone from the surface; kick STAYS (it is a
    # published primitive, since the floor cannot send Ctrl-D)
    text = pinned.decode()
    for retired in ("np_move_to", "np_melee_attack", "np_explore_level"):
        assert retired not in text
    assert "np_kick" in text, "kick is a primitive now and must be documented"


def test_every_mode_is_accounted_for():
    """A new mode must be a deliberate edit here, not an accident elsewhere."""
    import inspect

    from nethack_prime_agent import PrimeAgentHarness

    src = inspect.getsource(PrimeAgentHarness._materialise_netplay)
    for mode in ("off", "frozen", "mutable", "pinned"):
        assert f'"{mode}"' in src or f"'{mode}'" in src
