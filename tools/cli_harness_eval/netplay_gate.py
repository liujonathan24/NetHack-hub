#!/usr/bin/env python3
"""CLI wrapper for the netplay validation gate.

The implementation lives in `nethack_prime_agent.netplay_gate` so that the
harness (rollout teardown), the round-boundary merger, and this command all run
the SAME code. A second copy here would drift, and the copy that drifted would
be the one nobody ran.

    python netplay_gate.py <tree-dir> [--frozen-reference DIR] [--json]
"""

from __future__ import annotations

import sys

from nethack_prime_agent.netplay_gate import main

if __name__ == "__main__":
    sys.exit(main())
