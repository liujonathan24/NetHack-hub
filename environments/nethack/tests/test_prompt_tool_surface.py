"""The advertised action surface must equal the published action surface.

Under `skill_set="netplay"` — what every arm of the CLI-harness comparison runs
— no `move(direction=...)` adapter is built, yet seven places in the prompt and
the skill feedback told the agent to call it. exp1 measured 232 rejected `move`
attempts under the ASCII encoding.

The assertion below is deliberately generic: it re-derives the published set
from the skill registry for each `skill_set` and requires that the rendered
prompt name nothing outside it, so future drift is caught too.
"""
from __future__ import annotations

import re

import pytest

from nethack_harness.helpers import _build_skill_adapter_callables
from nethack_harness.prompt.rendering import SYSTEM_PROMPT, render_system_prompt
from nethack_harness.tools.skills import registry

_BACKTICKED = re.compile(r"`([a-z_][a-z0-9_]*)\s*(?:\(|`)")

SKILL_SETS = ["netplay", "full", "move", "dir8"]


def published(skill_set: str) -> set[str]:
    return {
        getattr(t, "__name__", "") for t in _build_skill_adapter_callables(skill_set)
    } - {""}


# Every name that is a tool SOMEWHERE. Only these count as "tool mentions", so
# backticked glyphs (`>`), NetHack nouns and ordinary prose are ignored — while
# a name that is a tool in one skill set and absent from another (the `move`
# case) still counts, which is the whole point.
ALL_SKILL_NAMES = set(registry.all_schemas()).union(
    *(published(s) for s in SKILL_SETS)
)


def tools_named_in(text: str) -> set[str]:
    return {m.group(1) for m in _BACKTICKED.finditer(text)} & ALL_SKILL_NAMES


@pytest.mark.parametrize("skill_set", SKILL_SETS)
def test_rendered_system_prompt_names_only_published_tools(skill_set):
    avail = published(skill_set)
    prompt = render_system_prompt(avail)
    named = tools_named_in(prompt)
    assert named <= avail, (
        f"skill_set={skill_set!r} prompt advertises unpublished tools: "
        f"{sorted(named - avail)}"
    )


def test_the_netplay_prompt_does_not_advertise_move():
    prompt = render_system_prompt(published("netplay"))
    assert "`move(" not in prompt
    assert "`move`" not in prompt


def test_default_system_prompt_still_renders():
    """The module constant stays a plain string (the overlay mutates it)."""
    assert isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 200


@pytest.mark.parametrize("skill_set", SKILL_SETS)
def test_rendered_prompt_still_teaches_the_published_movement_tool(skill_set):
    """Gating must not silently strip the prompt down to nothing useful."""
    avail = published(skill_set)
    prompt = render_system_prompt(avail)
    movers = {"move", "move_to", "explore_and_descend", "north"} & avail
    assert movers, f"skill_set={skill_set!r} publishes no movement tool at all"
    assert tools_named_in(prompt) & movers, (
        f"skill_set={skill_set!r} prompt teaches none of its movement tools"
    )


def test_skill_feedback_strings_do_not_advertise_unpublished_move():
    """The `move` mentions in skill feedback are part of the same surface.

    `skills.py` told the agent to `move` to explore on three turns of the
    committed artifact, under a skill set with no `move` tool.
    """
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "nethack_harness" / "tools" / "skills.py"
    ).read_text()
    # Only look at string literals that reach the agent as feedback.
    offenders = [
        line.strip()
        for line in src.splitlines()
        if ("`move`" in line or "`move(" in line) and not line.strip().startswith("#")
    ]
    assert not offenders, "skill feedback advertises `move`: " + " | ".join(offenders)


def test_cli_workspace_primer_is_gated_too():
    """The CLI arms read AGENTS.md, not the chat system message."""
    import inspect

    from tools.cli_harness_eval.workspace import build_workspace

    sig = inspect.signature(build_workspace)
    assert "system_prompt" in sig.parameters, (
        "build_workspace hard-codes the ungated SYSTEM_PROMPT"
    )
