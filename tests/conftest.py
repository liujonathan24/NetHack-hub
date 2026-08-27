"""Make the suite test THIS worktree's harness, not the editable install's.

`nethack_prime_agent` is installed editable into the shared eval venv and that
install points at the MAIN checkout, so `import nethack_prime_agent` from a
worktree resolves to the OTHER tree's file. Measured on this branch: `pytest
tests/test_prime_agent_harness.py` reported 7 failures -- including
`ImportError: cannot import name '_within'` -- against code that is present
here and passes, purely because the import resolved elsewhere. The dangerous
direction is the opposite one: a harness change that a worktree BREAKS would
still go green if the installed copy is the one under test.

`tools/cli_harness_eval/run_e13.sh:27` already exports the same path for the
same reason; this is that fix for the test runner.
"""

import pathlib
import sys

_HARNESS = pathlib.Path(__file__).resolve().parents[1] / "harnesses/nethack-prime-agent"
if _HARNESS.is_dir():
    sys.path.insert(0, str(_HARNESS))
