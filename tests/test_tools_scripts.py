"""Put the `tools/` verification scripts under the test runner.

Why this file exists: `tools/encoding_eval/_verify_gate.py` is named at
`docs/superpowers/specs/2026-07-25-cli-harness-eval-design.md` as *the*
netplay-gate acceptance check for this experiment, and `tools/exp1d_obs/verify_1d.py`
is the acceptance check for the observation-visibility sub-experiment. Neither
lives under a directory pytest collects, so when commit `d9fe9ae` dropped the
reserved `task` dataset column both scripts started raising `KeyError: 'task'`
and the suite stayed green at 176/0 (Task 12 part 3, report §16). A one-line
dataset change silently disabled the experiment's own acceptance checks.

These wrappers shell out and assert exit 0, so the scripts run with the suite.
They are deliberately thin: the assertions belong *in* the scripts (they are
also run by hand), and duplicating them here would let the two copies drift.
"""

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = [
    "tools/encoding_eval/_verify_gate.py",
    "tools/exp1d_obs/verify_1d.py",
]


def _env() -> dict:
    """The engine + env packages on `PYTHONPATH`, as the scripts are run by hand.

    `NetHackHarness` (the engine repo) is a sibling checkout, resolved the same
    way the rest of the suite resolves it: from the running interpreter's own
    import path, so this does not hard-code an absolute location.
    """
    import nethack  # noqa: F401  -- proves the env package is importable here

    paths = [str(REPO), str(REPO / "environments" / "nethack"), *sys.path]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    return env


@pytest.mark.parametrize("script", SCRIPTS)
def test_tools_verification_script_exits_zero(script):
    path = REPO / script
    assert path.exists(), f"{script} is named in the design doc but is missing"
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
