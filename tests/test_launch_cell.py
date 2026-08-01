"""tools/cli_harness_eval/launch_cell.sh: the one on-ramp every arm's rollout
must go through, so the fixed factors (model, seeds, character, skill_set --
see test_arm_configs.py) can't drift between arms.

These tests never launch a real rollout (no LLM calls, no MCP tool server).
`EVAL_BIN` is pointed at a stub that records its argv and exits 0, so what's
under test is purely the shell script's arm resolution and override
construction -- in particular the `[args]` (control) vs `[taskset]` (CLI
arms) key-shape split, and the JSON-vs-string-typing landmine documented
inline in the script (`--args.max_turns 150` silently becomes the string
"150"; only a `--args '{"max_turns": 150}'` JSON blob round-trips as an int).
"""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/cli_harness_eval/launch_cell.sh"

STUB_EVAL = """#!/usr/bin/env python3
import json, sys, os
with open(os.environ["STUB_EVAL_ARGV_FILE"], "w") as f:
    json.dump(sys.argv[1:], f)
"""


def _make_stub(tmp_path):
    stub = tmp_path / "eval"
    stub.write_text(STUB_EVAL)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


def _run(tmp_path, args, env_extra=None):
    stub = _make_stub(tmp_path)
    argv_file = tmp_path / "argv.json"
    env = dict(**_base_env(), EVAL_BIN=str(stub), STUB_EVAL_ARGV_FILE=str(argv_file))
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env
    )
    argv = json.loads(argv_file.read_text()) if argv_file.exists() else None
    return result, argv


def _base_env():
    import os

    # Keep PATH (for bash/python3) but nothing else -- the script must not
    # depend on ambient PYTHONPATH etc.
    #
    # ALLOW_STALE_ENGINE=1 because these tests are about ARGUMENT CONSTRUCTION,
    # not about the engine: the preflight added in launch_cell.sh reads the real
    # engine on this box, and on a dev box mid-rebuild the .so legitimately lags
    # the source (measured: libnethack.so 2026-07-22 04:26 vs src/src/nle.c
    # 2026-07-31 18:52). Without the bypass, every test here would flip red for
    # a reason none of them is testing. The preflight itself is covered by
    # test_engine_provenance.py against a synthetic engine tree.
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "ALLOW_STALE_ENGINE": "1",
    }


def test_refuses_an_unknown_arm(tmp_path):
    result, argv = _run(tmp_path, ["not_an_arm", str(tmp_path / "out")])
    assert result.returncode == 2
    assert "unknown arm" in result.stderr
    assert argv is None


def test_missing_arguments_prints_usage(tmp_path):
    result, argv = _run(tmp_path, ["control"])
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert argv is None


def test_echoes_the_resolved_settings_before_running(tmp_path):
    out = tmp_path / "out"
    result, _ = _run(tmp_path, ["claude_code", str(out), "20", "1"])
    assert result.returncode == 0, result.stderr
    # Not `splitlines()[0]`: the engine preflight prints its own `[engine] ...`
    # fingerprint line first, and which line comes first is not the contract.
    lines = [ln for ln in result.stdout.strip().splitlines() if "[launch_cell]" in ln]
    assert len(lines) == 1, result.stdout
    line = lines[0]
    assert "arm=claude_code" in line
    assert "max_calls=20" in line
    assert "n=1" in line
    assert str(out) in line


def test_control_arm_overrides_args_as_a_typed_json_blob(tmp_path):
    """The control arm's budget knob lives in the untyped `[args]` dict
    (v0 legacy-bridge kwargs). A bare `--args.max_turns 150` would silently
    become the STRING "150" and crash nethack.py's `self.max_turns > 0`
    check against a real GLM run -- verified live against the pinned
    verifiers CLI (pydantic_config only JSON-decodes a sub-field value
    that starts with `{`/`[`). The script must instead pass one `--args`
    JSON object so `max_turns` round-trips as a real int."""
    out = tmp_path / "out"
    result, argv = _run(tmp_path, ["control", str(out), "37", "5"])
    assert result.returncode == 0, result.stderr
    assert "@" in argv
    cfg_idx = argv.index("@")
    assert argv[cfg_idx + 1].endswith("configs/control.toml")
    assert "--num_tasks" in argv
    assert argv[argv.index("--num_tasks") + 1] == "5"
    assert "--output_dir" in argv
    assert argv[argv.index("--output_dir") + 1] == str(out)
    assert "--args" in argv
    blob = json.loads(argv[argv.index("--args") + 1])
    assert blob["max_turns"] == 37
    assert isinstance(blob["max_turns"], int)
    assert blob["trace_dir"] == str(out / "turns")
    # Must NOT also emit a plain --args.max_turns override (the typing trap).
    assert not any(a.startswith("--args.") for a in argv)


def test_cli_arms_override_taskset_fields_directly(tmp_path):
    """`[taskset]` is a typed sub-model (unlike `[args]`), so a plain dotted
    override coerces to a real int -- verified live. No JSON blob needed."""
    out = tmp_path / "out"
    result, argv = _run(tmp_path, ["prime_agent", str(out), "150", "16"])
    assert result.returncode == 0, result.stderr
    cfg_idx = argv.index("@")
    assert argv[cfg_idx + 1].endswith("configs/prime_agent.toml")
    assert "--taskset.max_skill_calls" in argv
    assert argv[argv.index("--taskset.max_skill_calls") + 1] == "150"
    assert "--taskset.trace_dir" in argv
    assert argv[argv.index("--taskset.trace_dir") + 1] == str(out / "turns")
    assert "--args" not in argv


def test_defaults_max_calls_150_and_n_16(tmp_path):
    out = tmp_path / "out"
    result, argv = _run(tmp_path, ["claude_code", str(out)])
    assert result.returncode == 0, result.stderr
    assert argv[argv.index("--num_tasks") + 1] == "16"
    assert argv[argv.index("--taskset.max_skill_calls") + 1] == "150"


def test_missing_eval_binary_fails_loudly(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "control", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        env={**_base_env(), "EVAL_BIN": str(tmp_path / "no-such-binary")},
    )
    assert result.returncode == 2
    assert "eval binary not found" in result.stderr
