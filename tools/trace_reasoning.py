"""Recover the agent's own words for each turn of a rollout, and say so
explicitly when they cannot be recovered.

THE DEFECT THIS FIXES (`docs/HARNESS_DEFECTS.md` Sec 3.5: "We can see *what* it
decided, never *why*")
-----------------------------------------------------------------------------
`_write_trace_entry` fills `assistant_message` from the message the harness had
just parsed. That message only exists inside the in-process v0 rollout loop.
The CLI arms (`claude_code`, `prime_agent`) drive the same game over MCP, and
the tool server that writes the trace is a *different process* from the one the
model talks to -- so it has nothing to write. Measured: `assistant_message` is
non-empty on 9 of 29 control-arm turns and on 0 of 12 and 0 of 20 turns of the
two committed CLI reference traces.

WHERE THE WORDS ACTUALLY ARE
----------------------------
Not in the CLI's own stdout. Both harnesses persist their process streams
(`nethack_prime_agent`'s `capture_output` writes `program.stdout.txt`;
verifiers surfaces a harness program's output on a non-zero exit), but those are
ONE blob per rollout of whatever the CLI chose to print under `--print`, with no
per-turn structure and no way to attribute a paragraph to a skill call.

They are in `traces.jsonl`. Every arm's model calls go through the verifiers
interception server -- including Prime Agent's (`tests/test_prime_agent_harness.py`
pins that it routes through the same endpoint) -- which commits each sampled
response as a node on the rollout trace. `nodes[i].message` is the assistant
message itself: `content`, `reasoning_content`, and `tool_calls`. That is a
per-model-turn, structured, arm-independent record of exactly what we are
missing, and it is already being written to disk beside the per-turn NDJSON.

So this module joins the two files. It is deliberately a pure function over
`(records, turns)` plus a thin directory driver, so the same join runs

  * live, from `NetHackTask.finalize`, which holds the `Trace` object; and
  * after the fact, over any completed run directory, including runs whose
    NDJSON was written before this code existed:

        python -m tools.trace_reasoning <run_dir> [--dry-run]

ALIGNMENT, AND WHY IT IS NOT ASSUMED
------------------------------------
A model turn is not always a skill call. The control arm drops parallel calls
past the first; Prime Agent does not call the game tools with the model at all
(its only tool is `ipython`, and the skills are a Python package inside the
kernel), so its assistant turns carry `ipython` calls whose bodies may run zero
or several skills. Guessing a 1:1 mapping there would attach the wrong
paragraph to the wrong move, which is worse than an empty string because it
looks right.

Three strategies are tried, strongest first, and if none holds NOTHING is
written except an explicit `reasoning.available = false` carrying the reason:

  0. `call_id` -- the correlation-id barrier (`tools/trace_align.py`): the
     harness stamps a per-rollout monotonic id on every turn record AND echoes
     `[call#N]` into the result payload, which flows back through the model
     transcript verbatim in every scaffold. When present, the join is exact by
     construction -- and it is the only strategy that survives the
     `ipython`-mediated arms, where one assistant turn issues many game calls.
     (`"call_id"` is an addition to the engine's documented
     `TS.REASONING_ALIGNMENTS` vocabulary -- doc-only; nothing validates the
     value -- noted for the engine repo rather than edited here.)
  1. `tool_call_sequence` -- the assistant turns' game tool-call names, in
     order, are walked against the records' dispatched names. Every record must
     find its turn. This is evidence, not an assumption.
  2. `ordinal` -- used only when the two channels hold the same number of
     entries AND no record exposes a dispatched name to check against.

Validated against ground truth: on `outputs/pilot_reveal` (control arm, 4
rollouts x 99 turns) the recovered text equals the `assistant_message` the
harness wrote inline on 396 of 396 turns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:  # the engine package is the schema's home; the hub imports it
    from nethack_core import trace_schema as TS
except ImportError:  # pragma: no cover - only when the engine is not on the path
    TS = None

from tools.trace_align import (
    align_records_to_turns_by_call_id,
    content_text,
    marker_call_ids,
)

__all__ = [
    "MCP_TOOL_PREFIXES",
    "assistant_turns_from_nodes",
    "assistant_turns_from_trace",
    "align_records_to_turns",
    "backfill_records",
    "backfill_run_dir",
    "normalize_tool_name",
    "record_call_name",
    "verify_alignment",
    "main",
]

#: Claude Code sees the toolset's skills as `mcp__<server>__<skill>`; the trace
#: records the bare skill name because that is what `_apply_tool_call` received.
#: Strip the prefix before comparing the two sides.
MCP_TOOL_PREFIXES = ("mcp__nethack__", "mcp__")

#: Tool names that are the AGENT's own scaffolding rather than a game skill.
#: Prime Agent's whole game interaction happens inside these, which is exactly
#: why its assistant turns cannot be aligned to skill dispatches by name.
NON_GAME_TOOLS = frozenset({"ipython", "python", "bash", "read", "edit", "write"})


def normalize_tool_name(name):
    """The bare skill name a trace record would carry for `name`."""
    if not name:
        return None
    out = str(name)
    for prefix in MCP_TOOL_PREFIXES:
        if out.startswith(prefix):
            out = out[len(prefix):]
            break
    return out


def _call_name(call) -> str | None:
    """`name` out of either tool-call shape (flat, or nested under `function`)."""
    if not isinstance(call, dict):
        fn = getattr(call, "function", None)
        return normalize_tool_name(
            getattr(fn, "name", None) if fn is not None else getattr(call, "name", None)
        )
    fn = call.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return normalize_tool_name(fn["name"])
    return normalize_tool_name(call.get("name"))


def _msg_field(message, key: str) -> str:
    if isinstance(message, dict):
        return message.get(key) or ""
    return getattr(message, key, "") or ""


def assistant_turns_from_nodes(nodes) -> list[dict]:
    """The sampled assistant messages of a rollout, in order.

    `sampled` is the provenance flag verifiers sets on model-produced nodes
    (`v1/trace.py`), so a prompt-supplied assistant message -- a few-shot
    example, a replayed prefix -- can never be mistaken for the agent's own
    reasoning about this turn.
    """
    turns = []
    # First-occurrence rule for marker harvesting. History compaction rewrites
    # older user messages every turn, so `prepare_turn` re-commits the whole
    # rewritten prefix as NEW (unsampled) nodes -- measured on a 10-turn
    # harness rollout: 91 nodes, with `[call#1]`'s message appearing twice,
    # the copy sitting after a much later assistant turn. The `sampled` flag
    # cannot gate these (user/tool nodes are never sampled), but the design
    # guarantees the FIRST appearance of a marker directly follows the turn
    # that issued the call; every later appearance is a replayed prefix.
    seen_call_ids: set = set()
    for node in nodes or []:
        sampled = node.get("sampled") if isinstance(node, dict) else getattr(node, "sampled", None)
        message = node.get("message") if isinstance(node, dict) else getattr(node, "message", None)
        if message is None:
            continue
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if sampled and role == "assistant":
            raw_calls = (message.get("tool_calls") if isinstance(message, dict)
                         else getattr(message, "tool_calls", None)) or []
            names = [n for n in (_call_name(c) for c in raw_calls) if n]
            turns.append({
                "content": _msg_field(message, "content"),
                "reasoning_content": _msg_field(message, "reasoning_content"),
                "tool_names": names,
                "game_tool_names": [n for n in names if n.lower() not in NON_GAME_TOOLS],
                # Correlation ids harvested below from the messages that FOLLOW
                # this turn -- the `[call#N]` markers the harness echoes into
                # every result payload (`tools/trace_align.py`).
                "result_call_ids": [],
            })
            continue
        # Any other message -- a tool result over MCP, the env-response user
        # message on the harness route, ipython's printed output -- may carry
        # `[call#N]` markers. They belong to the assistant turn that PRECEDED
        # the message, which is the turn that issued those calls. Markers seen
        # before any sampled assistant turn are prompt-supplied context (a
        # replayed prefix) and are deliberately dropped, same rule as the
        # `sampled` gate above.
        if turns:
            content = (message.get("content") if isinstance(message, dict)
                       else getattr(message, "content", None))
            ids = [i for i in marker_call_ids(content_text(content))
                   if i not in seen_call_ids]
            if ids:
                seen_call_ids.update(ids)
                turns[-1]["result_call_ids"].extend(ids)
    return turns


def assistant_turns_from_trace(trace) -> list[dict]:
    """`assistant_turns_from_nodes` over one `traces.jsonl` line or `Trace`."""
    nodes = trace.get("nodes") if isinstance(trace, dict) else getattr(trace, "nodes", None)
    return assistant_turns_from_nodes(nodes)


def record_call_name(record: dict) -> str | None:
    """The skill this record dispatched.

    `tool_results[0]["name"]` first: it is route-independent and has been
    populated on every path since schema version 2. `tool_calls[0]["name"]` is
    the version-0/1 fallback -- populated on the harness route always, and on
    the MCP route since the dispatch-route fix in `nethack.py`.
    """
    results = record.get("tool_results")
    if results:
        name = normalize_tool_name((results[0] or {}).get("name"))
        if name:
            return name
    calls = record.get("tool_calls")
    if calls:
        return _call_name(calls[0])
    return None


# --------------------------------------------------------------------------- #
# alignment                                                                    #
# --------------------------------------------------------------------------- #
def align_records_to_turns(records: list[dict], turns: list[dict]):
    """`(mapping, mode, reason)`.

    `mapping[i]` is the index into `turns` of the assistant turn that produced
    `records[i]`, or `None`. `mode` is one of `TS.REASONING_ALIGNMENTS` when a
    mapping was established, else `None` with `reason` explaining what failed.
    Nothing is guessed: a partial match is reported as no match at all, because
    a paragraph attached to the wrong move is worse than no paragraph.
    """
    if not records:
        return [], None, "no turn records"
    if not turns:
        return [None] * len(records), None, (
            "the rollout trace recorded no sampled assistant messages "
            "(the harness's model calls were not intercepted)"
        )

    # Strongest strategy first: the call-id barrier (`tools/trace_align.py`).
    # When it holds it is exact by construction -- the id was assigned by the
    # process that served the call and echoed through the transcript -- and it
    # is the only strategy that survives ipython-mediated arms, where one
    # assistant turn issues many game calls and no name ever matches. Absent
    # ids (pre-barrier traces) or a broken echo fall through to the name-walk
    # below, unchanged.
    by_id, id_mode, id_reason = align_records_to_turns_by_call_id(records, turns)
    if id_mode == "call_id":
        return by_id, "call_id", ""

    rec_names = [record_call_name(r) for r in records]
    if all(n is not None for n in rec_names):
        mapping: list[int | None] = []
        j = 0
        ok = True
        for name in rec_names:
            while j < len(turns) and (
                not turns[j]["game_tool_names"] or turns[j]["game_tool_names"][0] != name
            ):
                j += 1
            if j >= len(turns):
                ok = False
                break
            mapping.append(j)
            j += 1
        if ok and len(mapping) == len(records):
            return mapping, "tool_call_sequence", ""
        matched = len(mapping)
        # Fall through to ordinal, but remember why the strong match failed so
        # an unalignable rollout can say so rather than just "no match".
        seq_reason = (
            f"the dispatched skill sequence could not be walked against the "
            f"model's tool calls (matched {matched} of {len(records)} records "
            f"against {len(turns)} assistant turns)"
        )
    else:
        seq_reason = (
            f"{sum(1 for n in rec_names if n is None)} of {len(records)} records "
            "expose no dispatched skill name to match on"
        )

    if len(turns) == len(records):
        return list(range(len(records))), "ordinal", ""
    return [None] * len(records), None, (
        f"call-id join unavailable ({id_reason}); {seq_reason}; and the two "
        f"channels disagree on length ({len(records)} turn records vs "
        f"{len(turns)} assistant turns), so positional pairing would attach "
        "the wrong message to the wrong move"
    )


# --------------------------------------------------------------------------- #
# backfill                                                                     #
# --------------------------------------------------------------------------- #
def verify_alignment(records: list[dict], turns: list[dict], mapping) -> str:
    """`""` if the mapping is consistent with the text already on the records.

    A free correctness check that costs nothing and catches the one failure mode
    that matters: on the harness route the record ALREADY carries the assistant
    message inline, so if a proposed mapping pairs it with a different message
    the mapping is wrong and every recovered paragraph on that rollout would be
    attributed to the wrong move. Rather than trust the join, we make the
    control arm audit it on every run.
    """
    checked = mismatched = 0
    for i, record in enumerate(records):
        inline = record.get("assistant_message")
        if not inline or mapping[i] is None:
            continue
        checked += 1
        if inline != turns[mapping[i]]["content"]:
            mismatched += 1
    if mismatched:
        return (
            f"the proposed alignment disagrees with the assistant text already on "
            f"the records ({mismatched} of {checked} checked turns pair with a "
            f"different message), so it would attribute reasoning to the wrong move"
        )
    return ""


def backfill_records(records: list[dict], turns: list[dict], *,
                     overwrite: bool = False) -> dict:
    """Attach `reasoning` (and `assistant_message`) to every record in place.

    Returns a stats dict: `{"n", "mode", "reason", "recovered", "already",
    "unavailable"}`. `overwrite=False` leaves an `assistant_message` the harness
    wrote inline exactly as it wrote it -- this is additive repair, not a rewrite
    of a good record. The `reasoning` block is still attached to those records,
    because it carries something the writer never had: the provider's separate
    `reasoning_content` channel, which is non-empty on 306 of the 396
    control-arm turns in `outputs/pilot_reveal` and was being dropped on all of
    them.
    """
    mapping, mode, reason = align_records_to_turns(records, turns)
    if mode is not None:
        conflict = verify_alignment(records, turns, mapping)
        if conflict:
            mapping, mode, reason = [None] * len(records), None, conflict
    stats = {"n": len(records), "mode": mode, "reason": reason,
             "recovered": 0, "already": 0, "unavailable": 0}
    for i, record in enumerate(records):
        turn = turns[mapping[i]] if mapping and mapping[i] is not None else None
        inline = (record.get("assistant_message") or "").strip()
        if inline and not overwrite:
            record["reasoning"] = _reasoning(
                record["assistant_message"],
                turn["reasoning_content"] if turn else "",
                "assistant_message", "inline")
            stats["already"] += 1
            continue
        if turn is None or not (turn["content"] or turn["reasoning_content"]):
            if not (record.get("reasoning") or {}).get("available"):
                record["reasoning"] = _unavailable(
                    reason or ("the model emitted a tool call with no assistant "
                               "text this turn"))
            stats["unavailable"] += 1
            continue
        record["assistant_message"] = turn["content"]
        record["reasoning"] = _reasoning(
            turn["content"], turn["reasoning_content"], "trace_nodes", mode or "ordinal")
        stats["recovered"] += 1
    return stats


def _reasoning(text, reasoning_text, source, alignment) -> dict:
    if TS is not None:
        return TS.reasoning_record(text, source, alignment, reasoning_text=reasoning_text)
    return {"available": True, "source": source, "alignment": alignment,
            "text": text or "", "reasoning_text": reasoning_text or "", "reason": ""}


def _unavailable(reason: str) -> dict:
    if TS is not None:
        return TS.unavailable_reasoning(reason)
    return {"available": False, "source": None, "alignment": None,
            "text": "", "reasoning_text": "", "reason": reason}


# --------------------------------------------------------------------------- #
# directory driver                                                             #
# --------------------------------------------------------------------------- #
def _write_ndjson_atomically(path, records) -> None:
    """Replace `path` only once the whole new file is on disk.

    These files are the primary record of a run that cost real money; a partial
    rewrite from a killed process would destroy it. Write beside, fsync, rename.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".backfill.tmp")
    with tmp.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def backfill_run_dir(run_dir, warn=None, *, dry_run: bool = False,
                     overwrite: bool = False) -> list[dict]:
    """Backfill every rollout under `<run_dir>/<cell>/turns/*.ndjson`.

    Joined to `<run_dir>/<cell>/traces.jsonl` on the seed, using the same
    one-file-per-seed selection the aggregators use (`select_turn_files`), so a
    retried rollout's superseded attempt is reported and skipped rather than
    silently backfilled from the wrong trace.
    """
    from tools.eval_metrics import read_ndjson, select_turn_files

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
            warn(f"trace_reasoning: ignoring {path}: {reason}")
        for seed in sorted(chosen):
            path = chosen[seed]
            records = read_ndjson(path)
            trace = by_seed.get(seed)
            if trace is None:
                warn(
                    f"trace_reasoning: {path} has no matching rollout in "
                    f"{traces_path} -- reasoning left unavailable"
                )
                turns = []
            else:
                turns = assistant_turns_from_trace(trace)
            stats = backfill_records(records, turns, overwrite=overwrite)
            stats.update({"cell": cell_dir.name, "seed": seed, "path": str(path)})
            if stats["mode"] is None and stats["recovered"] == 0 and stats["already"] == 0:
                warn(
                    f"trace_reasoning: {cell_dir.name} seed {seed}: reasoning NOT "
                    f"recoverable -- {stats['reason']}"
                )
            if not dry_run:
                _write_ndjson_atomically(path, records)
            out.append(stats)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.trace_reasoning",
        description="Backfill per-turn agent reasoning into a run's turn traces.",
    )
    parser.add_argument("run_dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change; write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="also replace an assistant_message the harness wrote inline")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if not os.path.isdir(args.run_dir):
        print(f"trace_reasoning: no such run directory: {args.run_dir}", file=sys.stderr)
        return 2
    stats = backfill_run_dir(args.run_dir, dry_run=args.dry_run, overwrite=args.overwrite)
    if not stats:
        print(
            f"trace_reasoning: found no rollouts under {args.run_dir} -- a cell is a "
            f"SUBDIRECTORY holding turns/*.ndjson (and ideally traces.jsonl).",
            file=sys.stderr,
        )
        return 1
    print(f"{'cell':<16} {'seed':>5} {'turns':>6} {'recovered':>10} "
          f"{'inline':>7} {'none':>5}  alignment")
    for s in stats:
        print(f"{s['cell']:<16} {s['seed']:>5} {s['n']:>6} {s['recovered']:>10} "
              f"{s['already']:>7} {s['unavailable']:>5}  {s['mode'] or 'UNALIGNED'}")
    if args.dry_run:
        print("\n(dry run: nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
