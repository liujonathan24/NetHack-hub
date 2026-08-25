"""The tier registry: what each arm is, and what the new code tiers change.

SPEC §4 step 4 called for extending `tests/test_tool_tiers.py`; no such file
existed, so this is it. What it pins:

  - the flags that define each tier, and the ones the code tiers add;
  - correct ROUTING -- harness-side flags reach only the prime_agent arms,
    env-side ones reach every arm (`tool_tiers.py:101-107`);
  - that `[continual-code]` starts from `[base]`, the discipline
    `[continual]` already follows and the reason the arm is interpretable;
  - that `[continual-code-frozen]` differs from `[continual-code]` in exactly
    one key, since that single difference IS the control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))

import tool_tiers  # noqa: E402

CFG = tool_tiers.load()
PRIME_ARMS = ("prime_agent", "prime_agent_b80")
CODE_TIERS = ("continual-code", "continual-code-frozen")


def _pairs(argv: list[str]) -> dict[str, str]:
    return {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1, 2)
            if argv[i].startswith("--")}


# ---------- the registry ----------

def test_the_code_tiers_are_registered():
    """Tiers are every top-level key but `contract`, so a table is enough."""
    assert set(CODE_TIERS) <= set(tool_tiers.tiers(CFG))


def test_the_code_tier_starts_from_base_not_human():
    """A gain must be attributable to the agent's code, not to the human fixes.
    Same discipline `[continual]` follows, and for the same reason."""
    shared = set(CFG["base"]) & set(CFG["continual-code"])
    assert shared, "the tiers must share the fix-flags to be comparable at all"
    for key in shared:
        assert CFG["continual-code"][key] == CFG["base"][key], (
            f"[continual-code].{key} diverges from [base]; a difference here "
            "confounds the arm with the human-fix arm"
        )


def test_the_control_differs_from_the_treatment_in_exactly_one_key():
    """`netplay_code_mode` alone. If anything else drifts, the control stops
    isolating 'the agent improved its own code' and the arm is uninterpretable."""
    treat, ctrl = CFG["continual-code"], CFG["continual-code-frozen"]
    assert set(treat) == set(ctrl)
    differing = {k for k in treat if treat[k] != ctrl[k]}
    assert differing == {"netplay_code_mode"}, f"unexpected drift: {differing}"
    assert treat["netplay_code_mode"] == "mutable"
    assert ctrl["netplay_code_mode"] == "frozen"


def test_both_code_tiers_retire_the_server_side_composites():
    """The retirement is what forces the agent onto its own `netplay` code."""
    for tier in CODE_TIERS:
        assert CFG[tier]["netplay_composites"] is False


def test_only_the_code_tiers_touch_the_new_keys():
    """Adding these to base/human/continual would change their emitted argv for
    no behavioural reason -- and `netplay_composites` is env-side, so it would
    reach the claude_code arm too."""
    for tier in ("base", "human", "continual"):
        assert "netplay_composites" not in CFG[tier]
        assert "netplay_code_mode" not in CFG[tier]


# ---------- routing ----------

@pytest.mark.parametrize("arm", PRIME_ARMS)
def test_netplay_code_mode_is_routed_harness_side(arm):
    pairs = _pairs(tool_tiers.flags("continual-code", arm))
    assert pairs.get("--harness.netplay_code_mode") == '"mutable"'
    assert "--taskset.env_args.netplay_code_mode" not in pairs, (
        "netplay_code_mode is consumed by PrimeAgentHarnessConfig; sending it "
        "env-side would be a silent no-op"
    )


def test_netplay_code_mode_never_reaches_a_non_prime_arm():
    """`HarnessConfig` is extra='forbid', so `--harness.*` on the claude_code
    arm is a hard ValidationError, not a warning."""
    pairs = _pairs(tool_tiers.flags("continual-code", "claude_code"))
    assert not [k for k in pairs if k.startswith("--harness.netplay")]


@pytest.mark.parametrize("arm", (*PRIME_ARMS, "claude_code"))
def test_netplay_composites_is_routed_env_side_for_every_arm(arm):
    """It gates the np_core keep-set in `helpers.py`, which lives in the env."""
    pairs = _pairs(tool_tiers.flags("continual-code", arm))
    assert pairs.get("--taskset.env_args.netplay_composites") == "false"


def test_the_flag_is_registered_or_it_silently_does_nothing():
    """`tool_flags.configure` IGNORES unknown keys by design, so a flag missing
    from `_DEFAULTS` looks wired up end-to-end and changes nothing."""
    sys.path.insert(0, str(REPO / "environments" / "nethack"))
    from nethack_harness import tool_flags

    assert "netplay_composites" in tool_flags._DEFAULTS
    assert tool_flags._DEFAULTS["netplay_composites"] is True, (
        "the default must preserve the surface every earlier tier ran"
    )


# ---------- provenance ----------

def test_every_cell_records_enough_to_replay_its_surface():
    for tier in CODE_TIERS:
        pairs = _pairs(tool_tiers.flags(tier, "prime_agent"))
        assert pairs["--taskset.env_args.tool_tier"] == tier
        assert pairs["--taskset.env_args.tool_tier_hash"]
        assert pairs["--taskset.env_args.tool_tier_commit"]


def test_the_hash_is_whole_file_so_the_equivalence_note_is_load_bearing():
    """Pins the property that made the churn note necessary. If this ever
    becomes per-tier, the note in tool_tiers.toml can be retired."""
    import hashlib

    path = REPO / "tools" / "cli_harness_eval" / "configs" / "tool_tiers.toml"
    assert tool_tiers.registry_hash(path) == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()[:16]

    text = path.read_text()
    assert "d98f7d9432a2587b" in text, (
        "the pre-change tool_tier_hash must stay recorded, or cells run before "
        "the code tiers cannot be matched to equivalent cells run after"
    )
