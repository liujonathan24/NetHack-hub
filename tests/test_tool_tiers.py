"""The tier file is the thing that makes this series reproducible, so the parts
that could rot silently are pinned here: the frozen baseline's identity, the
refusal to pool tiers, and the provenance pin that has to survive into each
run's own config.toml.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))

import tiers  # noqa: E402


def test_the_frozen_baseline_is_exactly_e10():
    """[base] is the denominator of every claim in the series. If it drifts, the
    reference numbers in the results doc stop describing it."""
    cfg = tiers.load()
    base = cfg["base"]
    assert base["frozen"] is True
    assert base["skill_set"] == "np_core,request_map,search"
    # The two keys the E9/E10-era launchers dropped, which made "control" cells
    # run fog'd and auto-dismissed.
    assert base["env_args"]["auto_dismiss"] == "false"
    assert base["env_args"]["tune"]["reveal_map"] == 1.0
    assert base["cell"]["seeds"] == [0, 1, 2, 3, 4]
    assert base["cell"]["max_calls"] == 200
    assert base["cell"]["variant"] == "BBOX_MIN"
    # The reference numbers, so a drifted rerun is visible without opening the
    # results doc.
    assert base["reference"]["median_balrog"] == 3.54
    assert base["reference"]["ceiling_dlvl"] == 11


def test_base_env_args_carry_the_full_contract():
    ea = tiers.env_args("base")
    assert ea["skill_set"] == "np_core,request_map,search"
    assert ea["auto_dismiss"] == "false"
    assert ea["tune"] == {"reveal_map": 1.0}


def test_the_pin_rides_env_args_into_the_resolved_config():
    """`load_environment` absorbs unknown keys via **kwargs and the eval CLI
    writes the whole env_args table back out, so these four land literally in
    the run's own config.toml -- no side-car manifest to lose."""
    ea = tiers.env_args("base+human")
    for key in ("tier_stack", "tier_version", "tier_hash", "tier_commit"):
        assert key in ea, f"{key} must ride env_args into config.toml"
    assert ea["tier_stack"] == "base+human"
    assert ea["tier_hash"] == tiers.tier_hash()


def test_the_hash_changes_when_membership_changes(tmp_path):
    """Bumping `version` by hand is cheap and forgettable; the hash is not."""
    src = (REPO / "configs" / "tool_tiers.toml").read_text()
    a = tmp_path / "a.toml"
    a.write_text(src)
    b = tmp_path / "b.toml"
    b.write_text(src.replace('skill_set = "np_core,request_map,search"',
                             'skill_set = "np_core,request_map,search,wiki_lookup"'))
    assert tiers.tier_hash(a) != tiers.tier_hash(b)


def test_human_flags_apply_only_on_a_human_stack():
    assert "descent_gate" not in tiers.env_args("base")
    assert tiers.env_args("base+human")["descent_gate"] == "norm"


def test_continual_store_mounts_only_on_a_continual_stack():
    assert tiers.continual_flags("base") == {}
    assert tiers.continual_flags("base+human") == {}
    flags = tiers.continual_flags("base+human+continual")
    assert flags["CONTINUAL_HARNESS"].endswith("/e13")
    # Single-writer by default: the orchestrator writes, players read.
    assert flags["CONTINUAL_HARNESS_WRITABLE"] == ""


def test_an_undeclared_stack_is_refused_rather_than_composed():
    """Tiers are never pooled ad hoc. A new combination needs a line in the tier
    file, not a shell one-liner."""
    with pytest.raises(SystemExit, match="not an allowed stack"):
        tiers.env_args("base+continual")
    with pytest.raises(SystemExit, match="not an allowed stack"):
        tiers.env_args("continual")


def test_the_continual_arm_declares_what_it_must_be_compared_against():
    cfg = tiers.load()
    required = cfg["stacks"]["required_comparisons"]["base+human+continual"]
    assert set(required) == {"base", "base+human"}
    assert cfg["stacks"]["never_pool"] is True


def test_an_unmeasured_human_member_blocks_the_stack():
    """'additions require a commit AND a measured cell' -- an unmeasured fix is a
    patch, not a tier member."""
    problems = tiers.check("base+human")
    assert any("measured=false" in p for p in problems)


def test_the_unflagged_code_gap_is_reported_not_hidden():
    """PR #29 is unconditional code today, so a `base` cell on this tree is
    really base+netplay_telemetry. The resolver must say so."""
    problems = tiers.check("base")
    assert any("unconditional code" in p for p in problems)
    assert any("netplay_telemetry" in p for p in problems)


def test_the_continual_bounds_match_what_prime_agent_actually_renders():
    """Not our policy -- the scaffold's. formatHarnessStateForPrompt shows 6
    entries per kind at 180 chars; entries past that are invisible."""
    bounds = tiers.load()["continual"]["bounds"]
    assert bounds["max_entries_per_kind"] == 6
    assert bounds["max_content_chars"] == 180


def test_session_persistence_is_scoped_to_the_continual_arm():
    """Relaxing --no-session is a scaffold change, so it belongs to the arm that
    opts into it -- not to [base] or [human]."""
    scaffold = tiers.load()["continual"]["scaffold"]
    assert scaffold["no_session"] is True          # baseline behaviour by default
    assert scaffold["persist_memory_dir"] is False


def test_the_cli_emits_launchable_json():
    out = subprocess.run(
        [sys.executable, str(REPO / "tools/cli_harness_eval/tiers.py"), "env-args", "base+human"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    parsed = json.loads(out.stdout)
    assert parsed["tier_stack"] == "base+human"
