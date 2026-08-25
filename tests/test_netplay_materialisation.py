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
import types

import pytest

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


# ---------- `mutable`: the treatment ----------

def test_mutable_keeps_an_agent_authored_composite():
    edited = {f"{SKILL_DIR}/src/netplay/explore.py": b"# agent's generation 3\n"}
    runtime = _setup(_SetupRuntime(edited), netplay_code_mode="mutable", sandbox=True)
    assert runtime.files[f"{SKILL_DIR}/src/netplay/explore.py"] == b"# agent's generation 3\n"


def test_mutable_still_seeds_a_composite_that_is_absent():
    """A first round, or a file the agent deleted, must come back seeded."""
    edited = {f"{SKILL_DIR}/src/netplay/explore.py": b"# agent's generation 3\n"}
    runtime = _setup(_SetupRuntime(edited), netplay_code_mode="mutable", sandbox=True)
    assert b"YOURS TO EDIT" in runtime.files[f"{SKILL_DIR}/src/netplay/descend.py"]


def test_mutable_still_restores_the_frozen_boundary():
    """The self-healing rail: tampering with `_base.py` or the shim is undone
    at the start of the next episode even if every gate was bypassed."""
    tampered = {
        f"{SKILL_DIR}/src/netplay/_base.py": b"# neutered\n",
        f"{SKILL_DIR}/src/nethack/__init__.py": b"# neutered\n",
    }
    runtime = _setup(_SetupRuntime(tampered), netplay_code_mode="mutable", sandbox=True)
    for path in (f"{SKILL_DIR}/src/netplay/_base.py",
                 f"{SKILL_DIR}/src/nethack/__init__.py"):
        assert runtime.files[path] != b"# neutered\n", f"{path} must be restored"


def test_mutable_never_rewrites_pyproject():
    """Editing `pyproject.toml` changes `pyprojectHash` and triggers a ~10-min
    kernel-venv rebuild. It is written once, from the repo, and never varies."""
    a = _setup(netplay_code_mode="frozen").files[f"{SKILL_DIR}/pyproject.toml"]
    b = _setup(netplay_code_mode="mutable", sandbox=True).files[f"{SKILL_DIR}/pyproject.toml"]
    assert a == b == _setup().files[f"{SKILL_DIR}/pyproject.toml"]


# ---------- the concurrency guard ----------

def test_mutable_without_a_sandbox_is_refused():
    """One fixed skills path + no per-rollout bind + concurrent seeds = a
    chimeric tree. Fail at setup, not at merge."""
    with pytest.raises(ValueError, match="requires harness.sandbox=true"):
        _setup(netplay_code_mode="mutable", sandbox=False)


def test_frozen_without_a_sandbox_is_allowed():
    """The control writes nothing that has to survive, so sharing is harmless."""
    _setup(netplay_code_mode="frozen", sandbox=False)


def test_an_unknown_mode_is_refused_rather_than_silently_off():
    with pytest.raises(ValueError, match="netplay_code_mode"):
        _setup(netplay_code_mode="mutabel")
