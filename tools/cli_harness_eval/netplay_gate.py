#!/usr/bin/env python3
"""CLI wrapper for the netplay validation gate.

    python netplay_gate.py <tree-dir> [--frozen-reference DIR] [--json]

The gate implementation lives in `nethack_prime_agent/netplay_gate.py` so the
harness (rollout teardown), the round merger, the code orchestrator's
post-check, and this command all run the SAME code. It is loaded HERE by file
path, not `import nethack_prime_agent.netplay_gate`, on purpose: the package's
`__init__` pulls in pydantic/verifiers, but the gate itself is stdlib-only, so
loading the file directly lets the gate run under a bare `python3` (which the
orchestrator's between-rounds post-check relies on).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_IMPL = (pathlib.Path(__file__).resolve().parents[2]
         / "harnesses" / "nethack-prime-agent" / "nethack_prime_agent"
         / "netplay_gate.py")
_spec = importlib.util.spec_from_file_location("netplay_gate_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    sys.exit(_mod.main())
