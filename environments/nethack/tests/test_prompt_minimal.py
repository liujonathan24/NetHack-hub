"""Task 18 Step 1: BALROG-minimal SYSTEM_PROMPT, and SYSTEM_PROMPT_VERBOSE
kept as the pre-Task-18 fallback.

research-sota-methods.md: BALROG's entire objective scaffolding is two
sentences and it never discloses the progression metric; NetPlay's default
objective is "Win the game."; we gave MORE goal structure than either, and
NetPlay's own ablation LOST depth (2.60 -> 2.00) when tactical goal text was
added. The trace analyses agree from the other side: `recall`/`pin_objective`
were never called across 1,173 Claude Code calls.

`SYSTEM_PROMPT` is now the stripped-down variant; `SYSTEM_PROMPT_VERBOSE`
preserves the old strategy-heavy prompt byte-for-byte-equivalent so the A/B
is a config flag (`load_environment(verbose_prompt=True)`), not a
`git revert`.
"""
from __future__ import annotations

import re

import pytest

from nethack_harness.helpers import _build_skill_adapter_callables
from nethack_harness.prompt.rendering import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERBOSE,
    render_system_prompt,
)
from nethack_harness.tools.skills import registry

# A generous budget: BALROG's own instruction prompt is a two-sentence goal
# plus an 80-entry action catalogue and runs into the low thousands of
# characters. Ours additionally carries the COORDINATES paragraph, the glyph
# key, the doorway-reading notes and our own (much shorter) action list, but
# must stay well clear of the old ~2.5-3.5k-character strategy-heavy prompt.
MINIMAL_CHAR_BUDGET = 3000

# Text that must NOT survive into the minimal prompt: the STRATEGY PRIMER
# heading + its VISIBLE FEATURES/descend-steps prose, the DESCEND ASAP
# section, the STAY ALIVE sermon (and its three bullets), and the pitfalls
# list — the exact four things task-18-brief.md Step 1 names for deletion.
_STRATEGY_MARKERS = (
    "STRATEGY PRIMER",
    "STRATEGY: DESCEND ASAP",
    "STAY ALIVE",
    "Pitfalls:",
    "Starvation.",
    "Melee swarm",
    "Ranged / approaching",
    "workhorse",
    "pinned as `Objective:`",
)

_BACKTICKED = re.compile(r"`([a-z_][a-z0-9_]*)\s*(?:\(|`)")
SKILL_SETS = ["netplay", "full", "move", "dir8"]
ALL_SKILL_NAMES = set(registry.all_schemas())


def _published(skill_set: str) -> set[str]:
    return {
        getattr(t, "__name__", "") for t in _build_skill_adapter_callables(skill_set)
    } - {""}


def _tools_named_in(text: str) -> set[str]:
    return {m.group(1) for m in _BACKTICKED.finditer(text)} & ALL_SKILL_NAMES


def test_minimal_prompt_is_under_the_character_budget():
    assert len(SYSTEM_PROMPT) < MINIMAL_CHAR_BUDGET, len(SYSTEM_PROMPT)


def test_minimal_prompt_contains_no_strategy_prose():
    for marker in _STRATEGY_MARKERS:
        assert marker not in SYSTEM_PROMPT, f"strategy prose survived: {marker!r}"


def test_minimal_prompt_keeps_the_coordinates_paragraph():
    assert "=== COORDINATES ===" in SYSTEM_PROMPT
    assert "row 0 is the TOP row" in SYSTEM_PROMPT
    assert "Copy coordinates straight across" in SYSTEM_PROMPT


def test_minimal_prompt_keeps_the_glyph_key():
    assert "GLYPH KEY" in SYSTEM_PROMPT
    assert "stairs DOWN" in SYSTEM_PROMPT
    assert "canine" in SYSTEM_PROMPT


def test_minimal_prompt_keeps_the_action_list():
    # The action list is the SKILLS CHEAT SHEET, filtered to the default
    # (full registry) tool set.
    assert "SKILLS CHEAT SHEET" in SYSTEM_PROMPT


def test_minimal_prompt_appends_balrogs_two_sentences_verbatim():
    # balrog/environments/nle/__init__.py:107-131, verbatim.
    assert (
        "Explore the environment to find the stairs down to the next level."
        in SYSTEM_PROMPT
    )
    assert "Your goal is to get as far as possible in the game." in SYSTEM_PROMPT


def test_verbose_prompt_still_has_the_old_strategy_prose():
    # The A/B is a config flag: SYSTEM_PROMPT_VERBOSE is what SYSTEM_PROMPT
    # rendered before Task 18.
    assert "STRATEGY PRIMER" in SYSTEM_PROMPT_VERBOSE
    assert "STRATEGY: DESCEND ASAP" in SYSTEM_PROMPT_VERBOSE
    assert "STAY ALIVE" in SYSTEM_PROMPT_VERBOSE
    assert "Pitfalls:" in SYSTEM_PROMPT_VERBOSE
    assert len(SYSTEM_PROMPT_VERBOSE) > len(SYSTEM_PROMPT)


def test_verbose_prompt_does_not_carry_balrogs_objective_sentences():
    # The two are mutually exclusive: verbose keeps the old pinned-objective
    # pointer instead of BALROG's two sentences.
    assert "Your goal is to get as far as possible in the game." not in SYSTEM_PROMPT_VERBOSE


@pytest.mark.parametrize("skill_set", SKILL_SETS)
@pytest.mark.parametrize("verbose", [False, True])
def test_both_variants_name_only_published_tools(skill_set, verbose):
    # Task 17 Bug 4's generic assertion, re-run against both prompt variants:
    # neither can advertise a tool the active skill_set does not publish.
    avail = _published(skill_set)
    prompt = render_system_prompt(avail, verbose=verbose)
    named = _tools_named_in(prompt)
    assert named <= avail, (
        f"skill_set={skill_set!r} verbose={verbose} advertises unpublished "
        f"tools: {sorted(named - avail)}"
    )


def test_minimal_default_still_renders_a_plain_string():
    assert isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 200
