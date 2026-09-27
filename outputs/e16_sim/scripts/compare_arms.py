#!/usr/bin/env python3
"""Cross-arm behaviour table for SIM 3.

The contrast is the evidence. One arm doing what its directive said proves
little on its own -- a Valkyrie on seed 1 fights the adjacent jackal whatever
you tell it. What is load-bearing is that the SAME seed, the SAME tier, the
SAME call budget and the SAME model produce different FIRST ACTIONS and
different action mixes when only the directive block changes, and that the
no-directive control sits where it should.

Reports per arm: first nethack skill call after the opening request_map, the
action mix, the harness's descent metric, and how many served nodes carried a
directive block (0 for the control -- the negative control on the extractor
itself).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compliance import skill_calls  # noqa: E402
from served_bytes import node_text  # noqa: E402

DIRECTIVE_BLOCK_FORMAT = "[ORCHESTRATOR DIRECTIVE for this attempt: {directive}]"


def arm(rollout: Path) -> dict:
    trace = json.loads((rollout / "traces.jsonl").read_text().splitlines()[0])
    calls = skill_calls(trace)
    directive = (rollout / "directive.txt").read_text()
    expected = (DIRECTIVE_BLOCK_FORMAT.format(directive=directive)
                if directive else None)
    served = 0
    if expected:
        for n in trace["nodes"]:
            msg = n.get("message") or {}
            if msg.get("role") in ("system", "user", "tool"):
                if expected in node_text(msg):
                    served += 1
    # The opening request_map is scaffolding every arm does; the first call
    # AFTER it is the first real decision the directive could have moved.
    after = [c for c in calls if c["skill"] != "request_map"]
    mix: dict = {}
    for c in calls:
        mix[c["skill"]] = mix.get(c["skill"], 0) + 1
    m = trace["metrics"]
    return {
        "arm": rollout.name,
        "directive": directive or "(none - control)",
        "served_nodes_with_directive": served,
        "first_decision": (f"{after[0]['skill']}({after[0]['args']})"
                           if after else "(none)"),
        "action_mix": mix,
        "n_attacks": sum(v for k, v in mix.items() if "attack" in k),
        "n_explore": mix.get("np_explore_level", 0),
        "descent_count": m.get("descent_count"),
        "max_dlvl": m.get("max_dlvl_reached"),
        "max_xl": m.get("max_xp_level"),
        "skill_calls": m.get("skill_calls"),
        "died": m.get("died"),
        "stop": trace.get("stop_condition"),
    }


def main(argv) -> int:
    arms = [arm(Path(p)) for p in argv]
    print(f"{'arm':<22} {'served':>6} {'first decision after request_map':<34} "
          f"{'atk':>3} {'exp':>3} {'desc':>4} {'dlvl':>4} {'calls':>5}")
    print("-" * 92)
    for a in arms:
        print(f"{a['arm']:<22} {a['served_nodes_with_directive']:>6} "
              f"{a['first_decision']:<34} {a['n_attacks']:>3} "
              f"{a['n_explore']:>3} {int(a['descent_count'] or 0):>4} "
              f"{int(a['max_dlvl'] or 0):>4} {int(a['skill_calls'] or 0):>5}")
    out = Path(argv[0]).parent.parent / "arm_comparison.json"
    out.write_text(json.dumps(arms, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
