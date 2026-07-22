"""Regression tests for the NETHACK_HARNESS overlay loader (Experiment 2).

These are pure-Python (no engine / NLE needed): they exercise the self-contained
TOML loader and the four overlay seams directly against the shipped
``configs/harness/{baseline,fixes}.toml``.

The key invariant Experiment 2 depends on: ``baseline`` (or unset) is a no-op,
while ``fixes`` produces a *real* configuration difference (system prompt, skill
mask, and reward weights all change). See configs/harness/README.md.
"""
import pathlib
import sys
import types

# harness_overlay.py sits one directory up from this tests/ dir. Insert at the
# front so this repo's copy wins over any other environments/nethack on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import harness_overlay as ho  # noqa: E402

# Guard: make sure we loaded *this* repo's overlay (not a stale sibling clone).
assert str(pathlib.Path(__file__).resolve().parents[1]) in ho.__file__, ho.__file__


def _fake_module(system_prompt="You are playing NetHack.\nDescend and survive."):
    m = types.SimpleNamespace()
    m.SYSTEM_PROMPT = system_prompt
    m._VARIANT_FORMATTERS = {}
    return m


def test_unset_env_is_noop(monkeypatch):
    monkeypatch.delenv(ho._ENV_VAR, raising=False)
    mod = _fake_module()
    before = mod.SYSTEM_PROMPT
    cfg = ho.apply_overlay(mod)
    assert cfg is None
    assert mod.SYSTEM_PROMPT == before


def test_baseline_loads_as_noop():
    cfg = ho._load_cfg("baseline")
    assert cfg.name == "baseline"
    assert cfg.system_prompt.mode == "append"
    assert cfg.system_prompt.text == ""
    assert cfg.tools.enabled == () and cfg.tools.disabled == ()
    assert cfg.rewards == {}
    # resolve_reward_weights returns None (=> vf.Rubric keeps [1.0]*n default).
    assert ho.resolve_reward_weights([], cfg) is None


def test_fixes_loads_expected_overlay():
    cfg = ho._load_cfg("fixes")
    assert cfg.name == "fixes"
    assert cfg.system_prompt.mode == "patch"
    assert cfg.system_prompt.text.strip() != ""
    assert cfg.tools.disabled == ("move",)
    assert cfg.rewards == {"descent": 2.0, "scout": 0.5}


def test_baseline_vs_fixes_system_prompt_differs(monkeypatch):
    base_mod = _fake_module()
    monkeypatch.setenv(ho._ENV_VAR, "baseline")
    ho.apply_overlay(base_mod)

    fix_mod = _fake_module()
    monkeypatch.setenv(ho._ENV_VAR, "fixes")
    ho.apply_overlay(fix_mod)

    assert fix_mod.SYSTEM_PROMPT != base_mod.SYSTEM_PROMPT
    assert "Navigation:" in fix_mod.SYSTEM_PROMPT


def test_fixes_skill_mask_drops_move():
    cfg = ho._load_cfg("fixes")
    fns = [types.SimpleNamespace(__name__=n) for n in ("move", "move_to", "autoexplore")]
    kept = [f.__name__ for f in ho.filter_tool_callables(fns, cfg)]
    assert "move" not in kept
    assert kept == ["move_to", "autoexplore"]


def test_fixes_reward_weights_resolve():
    cfg = ho._load_cfg("fixes")
    funcs = [types.SimpleNamespace(__name__=n) for n in
             ("scout_reward", "descent_reward", "success_reward", "ascension_reward")]
    weights = ho.resolve_reward_weights(funcs, cfg)
    # scout 1.0->0.5, descent 1.0->2.0; unlisted stay at rubric default 1.0.
    assert weights == [0.5, 2.0, 1.0, 1.0]


def test_missing_harness_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        ho._load_cfg("does-not-exist")
