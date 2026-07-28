"""`NLE_LANG` renders BALROG's real observation, and its glyph tables agree with ours.

The converter is stock NLE's; our engine is the fork. If the fork ever renumbers
glyphs, `text_glyphs` keeps producing confident, well-formed, WRONG prose ("gold
piece adjacent southeast" pointing at empty floor) rather than raising. Nothing
downstream would catch that, so the parity assertion below is the guard: it
drives a live fork game and checks the converter's spatial claims land on the
tiles our own `chars` plane says they should.
"""
from __future__ import annotations

import pytest


def _live_obs():
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.seed(0, 0)
    reset = env.reset()
    return reset[0] if isinstance(reset, tuple) else reset


def _rows(obs):
    return ["".join(chr(int(c)) for c in row) for row in obs.chars]


@pytest.mark.slow
def test_converter_agrees_with_our_glyph_plane():
    """Seed 0/0 puts gold adjacent-southeast, the pet adjacent-northwest and a
    grid bug west-northwest of the hero. Assert the converter names each one AND
    that our own map carries the matching glyph at that offset."""
    from nethack_harness.prompt.nle_language import language_obs

    obs = _live_obs()
    text = language_obs(obs)["text_glyphs"]
    rows = _rows(obs)
    x, y = int(obs.blstats[0]), int(obs.blstats[1])

    assert "gold piece adjacent southeast" in text
    assert rows[y + 1][x + 1] == "$", "converter claims gold SE; our map disagrees"

    assert "tame little dog adjacent northwest" in text
    assert rows[y - 1][x - 1] == "d", "converter claims pet NW; our map disagrees"

    assert "grid bug very near westnorthwest" in text
    assert rows[y - 1][x - 2] == "x", "converter claims grid bug WNW; our map disagrees"


@pytest.mark.slow
def test_all_five_channels_are_populated():
    from nethack_harness.prompt.nle_language import language_obs

    o = language_obs(_live_obs())
    for key in ("text_glyphs", "text_blstats", "text_message", "text_inventory", "text_cursor"):
        assert o.get(key), f"{key} empty — the wrapper should populate every channel on reset"


@pytest.mark.slow
def test_variant_renders_language_and_omits_the_ascii_map():
    """NLE_LANG must not append our grid — that would recreate B_ASCII rather
    than reproduce BALROG's input."""
    from nethack_harness.prompt.nle_language import render_language_view

    view = render_language_view(_live_obs())
    assert "surroundings:" in view
    assert "stats:" in view
    assert "=== MAP ===" not in view
    assert "-----" not in view, "an ASCII map frame leaked into the language view"


def test_missing_interpreter_raises_rather_than_falling_back(monkeypatch):
    """A silent fallback would swap the observation under an NLE_LANG cell and
    invalidate the experiment without failing anything."""
    from nethack_harness.prompt import nle_language

    nle_language.shutdown()
    monkeypatch.setenv("NLE_LANG_PYTHON", "/nonexistent/python")
    with pytest.raises(nle_language.LanguageWrapperUnavailable):
        nle_language.language_obs(_live_obs())
    nle_language.shutdown()
