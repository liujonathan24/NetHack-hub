#!/usr/bin/env python3
"""TASK 4, step 3: did the replayed prefix REACH THE MODEL?

This project's standing rule is that "we passed it" is not "the model saw it";
E15 lost 150 rollouts to a wrong SKILL.md that every config said was right. So
this does not read `e16_args.json`, `ledger_text.txt` or the resolved
config.toml to decide anything. It walks `traces.jsonl`'s conversation NODES --
the messages actually sent to the model -- and reports where the checkpoint's
quoted conversation prefix appears in them, if it appears at all.

Three separate questions, answered separately, because they have different
answers and conflating them is how "text replay only" turned into "H4 is
untestable":

  1. Is the prefix BLOCK in the served bytes, byte-identical to what
     `render_prefix` produced?
  2. Is the CANARY -- a string that exists nowhere else in this harness, the
     wiki, the ledger table or the directive -- in the served bytes? This is
     the one that cannot be explained by anything else.
  3. Did the model READ it? Its own reasoning/text is quoted where it refers to
     the prefix. Evidence about attention, reported as such and never as proof
     of delivery, which question 2 already settles on its own.

Usage: prefix_served_bytes.py <rollout-dir> <ledger_text-file> <canary>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PREFIX_HEADER = ("THE PLAN THAT WAS LIVE WHEN THIS STATE WAS SAVED "
                 "(quoted from that session, not your own memory):")


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


def main(argv) -> int:
    rollout = Path(argv[0])
    ledger = Path(argv[1]).read_text()
    canary = argv[2]

    # The exact prefix block the orchestrator built, sliced out of the ledger
    # it assembled -- so the comparison is against bytes we can point at.
    idx = ledger.find(PREFIX_HEADER)
    if idx < 0:
        print("the ledger this run built contains no prefix block at all",
              file=sys.stderr)
        return 2
    prefix_block = ledger[idx:]

    traces = [json.loads(l) for l in
              (rollout / "traces.jsonl").read_text().splitlines() if l.strip()]
    trace = traces[0]
    nodes = trace.get("nodes") or []

    served_hits, canary_hits, model_mentions = [], [], []
    for i, n in enumerate(nodes):
        msg = n.get("message") or {}
        role = msg.get("role")
        text = node_text(msg)
        if not text:
            continue
        is_served = role != "assistant"   # what the model was SENT
        if is_served:
            if prefix_block in text:
                served_hits.append({
                    "node": i, "role": role,
                    "offset_of_prefix_block": text.index(prefix_block),
                    "node_chars": len(text),
                    "byte_identical": True,
                })
            elif PREFIX_HEADER in text:
                served_hits.append({
                    "node": i, "role": role,
                    "offset_of_prefix_block": text.index(PREFIX_HEADER),
                    "node_chars": len(text),
                    "byte_identical": False,
                    "note": "header present but the block is not byte-identical",
                })
            if canary in text:
                canary_hits.append({"node": i, "role": role,
                                    "offset": text.index(canary)})
        elif canary in text or "QUILLFEATHER" in text or "earlier session" in text:
            model_mentions.append({
                "node": i,
                "quote": " ".join(text.split())[:400],
            })

    out = {
        "rollout": str(rollout),
        "canary": canary,
        "prefix_block_chars": len(prefix_block),
        "n_nodes": len(nodes),
        "Q1_prefix_block_in_served_bytes": served_hits,
        "Q2_canary_in_served_bytes": canary_hits,
        "Q3_model_text_referring_to_it": model_mentions,
        "metrics": {k: trace.get("metrics", {}).get(k) for k in
                    ("skill_calls", "max_dlvl_reached", "died",
                     "budget_exhausted")},
        "stop_condition": trace.get("stop_condition"),
    }
    reached = bool(served_hits) and bool(canary_hits)
    out["VERDICT"] = (
        "THE REPLAYED PREFIX REACHES THE MODEL" if reached else
        "THE REPLAYED PREFIX DOES NOT REACH THE MODEL")
    out["verdict_detail"] = (
        f"the prefix block appears byte-identical in "
        f"{sum(1 for h in served_hits if h.get('byte_identical'))} served "
        f"node(s), and the canary -- which exists nowhere else in the harness, "
        f"the wiki, the ledger table or the directive -- appears in "
        f"{len(canary_hits)} served node(s). Delivery is therefore measured, "
        f"not inferred from configuration."
        if reached else
        "no served node carries the prefix; the replay does not reach the "
        "model and H4's prefix half is untestable as built.")
    print(json.dumps(out, indent=2))
    return 0 if reached else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
