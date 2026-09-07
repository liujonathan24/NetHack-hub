#!/usr/bin/env python3
"""Served-bytes check for the E16 directive, per this program's standing rule.

The rule exists because E15 lost 150 rollouts to a wrong SKILL.md that every
config said was right: "we passed it" is not "the model saw it". So this does
not read `directive.txt`, `e16_args.json`, or config.toml to decide whether the
directive was delivered. It walks `traces.jsonl`'s conversation NODES -- the
messages actually sent to the model -- and reports:

  * the exact node index and role where the directive block first appears,
  * whether the block is byte-identical to
    ``DIRECTIVE_BLOCK_FORMAT.format(directive=<directive.txt>)``,
  * how many served nodes carry it (the design says ONE-SHOT: first
    observation only),
  * every assistant reasoning/text block that mentions the directive, and
  * the ordered list of skill calls, so directive-consistent behaviour can be
    read off the actions rather than off the model's own claims.

Usage: served_bytes.py <rollout-dir> [<rollout-dir> ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIRECTIVE_BLOCK_FORMAT = "[ORCHESTRATOR DIRECTIVE for this attempt: {directive}]"


def node_text(msg: dict) -> str:
    """Everything in this message that was rendered into the prompt."""
    parts = []
    c = msg.get("content")
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
            elif isinstance(b, str):
                parts.append(b)
    if msg.get("reasoning_content"):
        parts.append(str(msg["reasoning_content"]))
    for tc in msg.get("tool_calls") or []:
        parts.append(json.dumps(tc.get("arguments")))
    return "\n".join(parts)


def skill_calls(nodes: list) -> list:
    """The nethack skill calls, in order, as written by the model."""
    out = []
    for i, n in enumerate(nodes):
        msg = n.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        if not n.get("sampled"):
            continue  # the duplicated non-sampled echo of the same message
        for tc in msg.get("tool_calls") or []:
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
            code = (args or {}).get("code") or ""
            for m in re.finditer(r"nethack\.(\w+)\s*\(([^)]*)\)", code):
                out.append({"node": i, "skill": m.group(1),
                            "args": m.group(2).strip()})
    return out


def report(rollout: Path) -> dict:
    trace = json.loads((rollout / "traces.jsonl").read_text().splitlines()[0])
    nodes = trace["nodes"]
    directive = (rollout / "directive.txt").read_text()
    expected = DIRECTIVE_BLOCK_FORMAT.format(directive=directive)

    hits = []
    for i, n in enumerate(nodes):
        msg = n.get("message") or {}
        if msg.get("role") not in ("system", "user", "tool"):
            continue  # only what was SERVED to the model
        txt = node_text(msg)
        if expected in txt:
            idx = txt.index(expected)
            hits.append({"node": i, "role": msg.get("role"),
                         "offset": idx, "node_len": len(txt),
                         "context": txt[max(0, idx - 200): idx + len(expected) + 400]})

    ack = []
    for i, n in enumerate(nodes):
        msg = n.get("message") or {}
        if msg.get("role") != "assistant" or not n.get("sampled"):
            continue
        txt = node_text(msg)
        if re.search(r"directive|ORCHESTRATOR|do not descend|don't descend|"
                     r"not descend|descend as fast|XL ?3|staircase",
                     txt, re.I):
            ack.append({"node": i, "text": txt[:1500]})

    calls = skill_calls(nodes)
    m = trace.get("metrics") or {}
    out = {
        "rollout": str(rollout),
        "directive": directive,
        "expected_block": expected,
        "served_hits": hits,
        "n_served_nodes_with_directive": len(hits),
        "assistant_mentions": ack,
        "skill_calls": calls,
        "metrics": m,
        "stop_condition": trace.get("stop_condition"),
        "n_nodes": len(nodes),
    }
    (rollout / "served_bytes_report.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv) -> int:
    for d in argv:
        r = report(Path(d))
        print("=" * 78)
        print(f"ROLLOUT {r['rollout']}")
        print(f"directive: {r['directive']!r}")
        print(f"SERVED-BYTES: directive block found in "
              f"{r['n_served_nodes_with_directive']} served node(s)")
        for h in r["served_hits"]:
            print(f"  node {h['node']} (role={h['role']}) at char "
                  f"{h['offset']} of {h['node_len']}")
            print("  --- context ---")
            for line in h["context"].splitlines():
                print("  | " + line)
        print(f"\nSKILL CALLS ({len(r['skill_calls'])}):")
        for c in r["skill_calls"]:
            print(f"  node {c['node']:>3}  nethack.{c['skill']}({c['args']})")
        print(f"\nmetrics: descent_count="
              f"{r['metrics'].get('descent_count')} "
              f"max_dlvl={r['metrics'].get('max_dlvl_reached')} "
              f"max_xl={r['metrics'].get('max_xp_level')} "
              f"skill_calls={r['metrics'].get('skill_calls')} "
              f"died={r['metrics'].get('died')}")
        print(f"\nASSISTANT MENTIONS ({len(r['assistant_mentions'])}):")
        for a in r["assistant_mentions"][:4]:
            print(f"  --- node {a['node']} ---")
            for line in a["text"].splitlines()[:14]:
                print("  > " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
