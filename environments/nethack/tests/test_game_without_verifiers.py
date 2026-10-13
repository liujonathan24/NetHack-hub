"""The game box runs without verifiers, and plays the same game byte for byte.

`nethack_toolset.game` (residency-environments, scripts/nethack) builds this env
with `load_environment(..., self_dispatch=True, trace_dir=...)` and then only
calls `setup_state`, `_apply_tool_call` and `_render_obs_text`. Its image
installs neither verifiers nor datasets; `nethack_harness.vf_compat` stands in
for the little of verifiers that path runs. This drives that path in two fresh
interpreters, one with verifiers importable and one with verifiers and datasets
blocked, and requires the same observations, schemas and record.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("verifiers")

ENV_DIR = Path(__file__).resolve().parents[1]

DRIVER = textwrap.dedent(
    """
    import asyncio, json, sys
    if sys.argv[1] == "blocked":
        sys.modules["verifiers"] = None
        sys.modules["datasets"] = None
    from nethack_harness.vf_compat import HAVE_VERIFIERS
    from nethack import load_environment
    from nethack_harness.prompt.content import content_to_text
    from nethack_harness.tools.skills import registry

    CALLS = [
        ("request_map", {}), ("search", {"times": 2}), ("np_explore_level", {}), ("np_rest", {"count": 2}),
        ("np_press_key", {"key": "#pray<ENTER>"}), ("np_press_key", {"key": "n"}),
        ("np_press_key", {"key": "#"}), ("np_press_key", {"key": "<ENTER>"}), ("np_press_key", {"key": "ESC"}),
        ("np_move_to", {"x": 40, "y": 10}), ("np_kick", {"x": 41, "y": 10}),
        ("np_melee_attack", {"x": 41, "y": 10}), ("np_apply", {"item_letter": "a"}),
        ("np_press_key", {"key": "ESC"}), ("np_pray", {}), ("np_press_key", {"key": "y"}),
        ("np_explore_level", {}), ("np_explore_level", {}),
    ]

    async def main():
        env = load_environment(
            variant="BBOX_MIN", map_detail="full", auto_dismiss=False,
            skill_set="np_core,request_map,search", character="Val-hum-neu-fem",
            tune={"reveal_map": 1.0}, modify={"gold": 7}, not_a_knob=1,
            n_examples=1, seed=3, self_dispatch=True, trace_dir=sys.argv[2],
        )
        state = await env.setup_state({"task": {"seed": 3}, "info": {}})
        schemas = registry.all_schemas()
        out = {
            "have_verifiers": HAVE_VERIFIERS,
            "not_a_knob": env.not_a_knob,
            "skills": {n: schemas[n] for n in sorted(env._allowed_skill_names) if n in schemas},
            "turns": [env._render_obs_text(state)],
        }
        for skill, args in CALLS:
            content = await env._apply_tool_call(state, skill, args)
            text = content_to_text(content) if content else env._render_obs_text(state)
            out["turns"].append([text, bool(state.get("terminated")), state.get("max_dlvl_reached")])
        print(json.dumps(out))

    asyncio.run(main())
    """
)

CLOCK = re.compile(rb'"(t_wall|t_mono)": -?[0-9.e+-]+')


def _play(mode: str, trace_dir: Path) -> tuple[dict, bytes]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ENV_DIR), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER, mode, str(trace_dir)],
        cwd=ENV_DIR, env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    (record,) = trace_dir.glob("*.ndjson")
    return json.loads(proc.stdout.splitlines()[-1]), CLOCK.sub(rb'"\1": 0', record.read_bytes())


def test_game_path_is_identical_without_verifiers(tmp_path):
    with_vf, record_with = _play("installed", tmp_path / "installed")
    without_vf, record_without = _play("blocked", tmp_path / "blocked")
    assert with_vf.pop("have_verifiers") is True
    assert without_vf.pop("have_verifiers") is False
    assert with_vf == without_vf
    assert record_with == record_without


def test_verifiers_env_is_unchanged_when_verifiers_is_installed():
    import verifiers

    import nethack
    from nethack_harness import helpers
    from nethack_harness.vf_compat import HAVE_VERIFIERS, vf

    assert HAVE_VERIFIERS and vf is verifiers
    assert issubclass(nethack.NetHackVerifiersEnv, verifiers.StatefulToolEnv)
    assert helpers.ascension_reward.reward_weight == 1000.0
