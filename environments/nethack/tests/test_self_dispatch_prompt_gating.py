"""Task 18 Step 2: the JOURNAL block and the HINT ladder are redundant
scaffolding for the CLI-agent arms (Claude Code, Prime Agent), which manage
their own reasoning and memory internally, but the control arm must not
drift -- it is the v0 baseline whose fidelity is the strongest claim in this
design.

``format_observation_as_chat`` reads ``state["_self_dispatch"]`` (set from
the env's ``self_dispatch`` constructor flag in ``setup_state``) to gate both
blocks off. The v1 MCP toolset (``nethack_v1.NetHackToolset``) always builds
its v0 env with ``self_dispatch=True`` -- ``self_dispatch=False`` has no v1
path at all (`NetHackToolset.__init__` raises). The control arm calls
``nethack.load_environment`` directly through the v0 legacy bridge and never
sets it, so it defaults to False and keeps both blocks.
"""
from __future__ import annotations

from nethack_core.observations import StructuredObservation
from nethack_harness.memory.journal import Journal
from nethack_harness.prompt.rendering import format_observation_as_chat


def _low_hp_structured() -> StructuredObservation:
    # HP well under the 30% HINT threshold, with nothing else in play, so a
    # HINT ladder entry fires deterministically without needing raw_obs.
    return StructuredObservation(
        map_view="",
        messages=[],
        inventory=[],
        status={
            "x": 5, "y": 5, "hitpoints": 2, "max_hitpoints": 16,
            "armor_class": 6, "depth": 1, "time": 1,
            "experience_level": 1, "gold": 0, "hunger_state": 1,
        },
        character={},
        adjacent={},
    )


def _journal_with_a_pinned_objective() -> Journal:
    journal = Journal()
    journal.pin_objective("Descend as deep into the dungeon as you can.")
    return journal


def test_self_dispatch_observation_has_no_journal_or_hint_block():
    rendered = format_observation_as_chat(
        _low_hp_structured(),
        _journal_with_a_pinned_objective(),
        state={"_self_dispatch": True},
        compact=False,
        include_map=False,
    )
    assert "=== JOURNAL ===" not in rendered, rendered
    assert "=== HINT ===" not in rendered, rendered
    # The objective text itself must not leak in some other form either.
    assert "Descend as deep into the dungeon" not in rendered


def test_control_observation_still_has_journal_and_hint_block():
    for state in ({}, {"_self_dispatch": False}, None):
        rendered = format_observation_as_chat(
            _low_hp_structured(),
            _journal_with_a_pinned_objective(),
            state=state,
            compact=False,
            include_map=False,
        )
        assert "=== JOURNAL ===" in rendered, (state, rendered)
        assert "=== HINT ===" in rendered, (state, rendered)


def test_empty_journal_still_renders_no_block_either_way():
    # An empty journal never renders regardless of self_dispatch -- this just
    # confirms the gate doesn't change that pre-existing behavior.
    empty = Journal()
    for self_dispatch in (True, False):
        rendered = format_observation_as_chat(
            _low_hp_structured(), empty,
            state={"_self_dispatch": self_dispatch},
            compact=False, include_map=False,
        )
        assert "=== JOURNAL ===" not in rendered


def test_load_environment_wires_self_dispatch_onto_the_env():
    """`load_environment(self_dispatch=...)` reaches the env instance attribute
    that `setup_state` copies into `state["_self_dispatch"]` every rollout."""
    from nethack import load_environment

    control_env = load_environment(n_examples=1, max_turns=2, skill_set="netplay")
    assert control_env.self_dispatch is False

    cli_env = load_environment(
        n_examples=1, max_turns=2, skill_set="netplay", self_dispatch=True,
    )
    assert cli_env.self_dispatch is True
