"""Join the per-turn NDJSON and the rollout trace on the tool-call correlation id.

THE BARRIER. The LLM-side transcript (`traces.jsonl`: the full message stream)
and the game-side turn NDJSON never referenced each other, so putting "what the
model said" next to "what the game did" required a heuristic join --
`tools/trace_reasoning.py`'s name-walk, which honestly refuses exactly where it
matters most (Prime Agent runs skills inside `ipython`, so one assistant turn
issues many game calls and no tool name ever matches).

The harness now assigns a per-rollout monotonic call id at `_apply_tool_call`
(the single execution path every route shares), stamps it on the turn record
as `tool_results[0]["call_id"]`, and appends a `[call#N]` marker to the RESULT
payload -- the one channel that flows back through the model transcript
verbatim in every scaffold (an MCP tool message, or printed output inside an
ipython block), with **zero changes to the published tool schemas**. See
`nethack_harness.helpers.CALL_ID_MARKER_FORMAT` (the writer half of this
contract; the regex below must stay in sync with it).

Alignment is then exact by construction:

    turn record  <->  transcript message carrying the same `[call#N]`
                 <->  the sampled assistant message that issued the call

An ipython block that issues several calls attributes ALL of their records to
that block's one assistant message -- correct, since that message is the words
that produced them; the ids themselves (monotonic, echoed in output order)
preserve the ordering within the block.

This module is the reader half: pure functions over `(records, turns)` plus a
read-only directory driver that reports coverage:

    python -m tools.trace_align <run_dir>

`tools/trace_reasoning.py` imports the join as its strongest alignment
strategy (mode ``"call_id"``, tried before the name-walk), so
`NetHackTask.finalize`'s live backfill benefits automatically. Old traces
(written before the barrier) carry no ids; the join says so explicitly and the
callers fall back to the name-walk, unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

__all__ = [
    "CALL_ID_MARKER_RE",
    "content_text",
    "marker_call_ids",
    "record_call_id",
    "align_records_to_turns_by_call_id",
    "align_coverage",
    "report_run_dir",
    "main",
]

#: Reader half of `nethack_harness.helpers.CALL_ID_MARKER_FORMAT`
#: (``"[call#{}]"``). Spelled out as a literal so this module stays importable
#: without the env package on the path (same reason `trace_reasoning` treats
#: the engine schema as optional).
CALL_ID_MARKER_RE = re.compile(r"\[call#(\d+)\]")


def content_text(content) -> str:
    """Flatten a message ``content`` value (str, or list of content blocks)
    into searchable text. Unknown shapes flatten to ``""`` -- a missed marker
    is reported as a coverage gap, never an exception."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def marker_call_ids(text) -> list:
    """Every correlation id marked in ``text``, in order of appearance.

    Order matters: an ipython block prints several results in execution order,
    and that order is the sub-ordering of the block's calls.
    """
    return [int(m) for m in CALL_ID_MARKER_RE.findall(text or "")]


def record_call_id(record):
    """The correlation id a turn record carries, or ``None``.

    ``None`` covers both "written before the barrier existed" (no key) and
    "produced by no dispatched call" (the end-of-rollout flush, key explicitly
    null) -- either way there is nothing to join on and the caller must not
    fabricate one.
    """
    results = record.get("tool_results") or []
    if results and isinstance(results[0], dict):
        cid = results[0].get("call_id")
        if cid is not None:
            try:
                return int(cid)
            except (TypeError, ValueError):
                return None
    return None


# --------------------------------------------------------------------------- #
# the join                                                                     #
# --------------------------------------------------------------------------- #
def align_records_to_turns_by_call_id(records: list, turns: list):
    """``(mapping, mode, reason)`` -- the same contract as
    `tools.trace_reasoning.align_records_to_turns`.

    ``turns`` are sampled assistant turns each carrying ``result_call_ids``
    (the ids marked in the transcript messages that FOLLOWED that turn --
    `trace_reasoning.assistant_turns_from_nodes` harvests them). ``mapping[i]``
    is the turn index whose call produced ``records[i]``, or ``None`` for a
    record with no id (legacy record, or the never-dispatched flush).

    ``mode`` is ``"call_id"`` only when every id-bearing record found exactly
    one issuing turn. Anything weaker -- no ids anywhere, an id claimed by two
    turns, an id with no marker in the transcript -- returns no mapping and a
    reason, so the caller falls back rather than trusting a partial join.
    """
    if not records:
        return [], None, "no turn records"
    rec_ids = [record_call_id(r) for r in records]
    id_bearing = [i for i, cid in enumerate(rec_ids) if cid is not None]
    if not id_bearing:
        return [None] * len(records), None, (
            "no record carries a correlation id (trace written before the "
            "call-id barrier existed)"
        )

    id_to_turn: dict = {}
    duplicated: set = set()
    for j, turn in enumerate(turns or []):
        for cid in turn.get("result_call_ids") or []:
            if cid in id_to_turn and id_to_turn[cid] != j:
                duplicated.add(cid)
            else:
                id_to_turn.setdefault(cid, j)
    if duplicated:
        return [None] * len(records), None, (
            f"correlation id(s) {sorted(duplicated)} appear under more than one "
            "assistant turn in the transcript -- the join would be ambiguous, "
            "so nothing is claimed"
        )
    if not id_to_turn:
        return [None] * len(records), None, (
            "the transcript contains no [call#N] markers (result echo disabled "
            "via call_id_in_results=False, or the model calls were not "
            "intercepted)"
        )

    mapping = [id_to_turn.get(cid) if cid is not None else None for cid in rec_ids]
    unmatched = sum(1 for i in id_bearing if mapping[i] is None)
    if unmatched:
        return [None] * len(records), None, (
            f"{unmatched} of {len(id_bearing)} id-bearing records found no "
            "matching [call#N] marker in the transcript"
        )
    return mapping, "call_id", ""


def align_coverage(records: list, turns: list) -> dict:
    """Coverage numbers for one rollout's join (read-only; nothing written)."""
    mapping, mode, reason = align_records_to_turns_by_call_id(records, turns)
    rec_ids = [record_call_id(r) for r in records]
    with_id = sum(1 for cid in rec_ids if cid is not None)
    marked = {cid for turn in (turns or []) for cid in (turn.get("result_call_ids") or [])}
    joined = sum(1 for i, j in enumerate(mapping) if j is not None) if mode else 0
    return {
        "n": len(records),
        "with_id": with_id,
        "without_id": len(records) - with_id,
        "markers_in_transcript": len(marked),
        "joined": joined,
        "mode": mode,
        "reason": reason,
        # The barrier's promise, checked: every id-bearing record joined
        # (total) and no id was claimed twice (unique -- ambiguity refuses
        # the whole join above).
        "total": bool(mode) and joined == with_id and with_id > 0,
    }


# --------------------------------------------------------------------------- #
# directory driver (read-only)                                                 #
# --------------------------------------------------------------------------- #
def report_run_dir(run_dir, warn=None) -> list:
    """Coverage per rollout under ``<run_dir>/<cell>/turns/*.ndjson``, joined
    to ``<cell>/traces.jsonl`` on the seed -- the same pairing the aggregators
    and `trace_reasoning.backfill_run_dir` use. Reports; never writes."""
    from tools.eval_metrics import read_ndjson, select_turn_files
    from tools.trace_reasoning import assistant_turns_from_trace

    warn = warn or (lambda msg: print(msg, file=sys.stderr))
    run_dir = Path(run_dir)
    out = []
    for cell_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        traces_path = cell_dir / "traces.jsonl"
        by_seed = {}
        if traces_path.exists():
            for line in traces_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                trace = json.loads(line)
                seed = ((trace.get("task") or {}).get("data") or {}).get("idx")
                by_seed[seed] = trace
        chosen, dropped = select_turn_files(cell_dir, seeds=set(by_seed) or None)
        for path, reason in dropped:
            warn(f"trace_align: ignoring {path}: {reason}")
        for seed in sorted(chosen):
            records = read_ndjson(chosen[seed])
            trace = by_seed.get(seed)
            turns = assistant_turns_from_trace(trace) if trace is not None else []
            stats = align_coverage(records, turns)
            stats.update({"cell": cell_dir.name, "seed": seed, "path": str(chosen[seed])})
            if not stats["total"]:
                warn(
                    f"trace_align: {cell_dir.name} seed {seed}: join NOT total -- "
                    f"{stats['reason'] or 'records without ids'}"
                )
            out.append(stats)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.trace_align",
        description="Report call-id join coverage between a run's turn traces "
                    "and its rollout transcripts. Read-only.",
    )
    parser.add_argument("run_dir")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if not os.path.isdir(args.run_dir):
        print(f"trace_align: no such run directory: {args.run_dir}", file=sys.stderr)
        return 2
    stats = report_run_dir(args.run_dir)
    if not stats:
        print(
            f"trace_align: found no rollouts under {args.run_dir} -- a cell is a "
            f"SUBDIRECTORY holding turns/*.ndjson (and traces.jsonl).",
            file=sys.stderr,
        )
        return 1
    print(f"{'cell':<16} {'seed':>5} {'turns':>6} {'with_id':>8} "
          f"{'joined':>7} {'total':>6}  mode")
    for s in stats:
        print(f"{s['cell']:<16} {s['seed']:>5} {s['n']:>6} {s['with_id']:>8} "
              f"{s['joined']:>7} {str(s['total']):>6}  {s['mode'] or 'UNALIGNED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
