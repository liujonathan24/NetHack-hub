"""Per-seed progress for a CLI-harness cell, safe against the aggregation traps.

Usage:  progress.py <cell-dir> [<cell-dir> ...]

Every trap below has cost a real measurement in this project at least once, so
this script exists to stop re-deriving the same fixes in ad-hoc one-liners.
The shared implementations all live in `tools/eval_metrics.py`:

1. ONE ROW PER SEED. A rollout that retried writes a NEW turn file per attempt.
   Globbing `turns/*.ndjson` double-counts: v3_cli/claude_code seed 1 had a
   2-turn stub beside the real 247-turn attempt and was reported as n=6. We key
   on the seed in the filename and keep the longest attempt
   (`eval_metrics.select_turn_files`).

2. TRUST TURN FILES OVER `traces.jsonl` METRICS FOR GAME STATE. A retried
   rollout records only its failed final attempt, so `skill_calls` can read 0
   for a rollout that played thousands of turns (b0_flash/prime_agent: metrics
   said 0, turn files held 2,956 calls to dlvl 2). `traces.jsonl` is still the
   authority for `stop_condition`. And on the v0-legacy path (`running Nx1 v0
   rollouts ... (legacy: nethack)`) `skill_calls` is not written AT ALL, so the
   degeneracy rule runs off `eval_metrics.skill_call_count`'s explicit fallback
   chain and reports UNKNOWN rather than guessing.

3. `tool_calls` WAS EMPTY IN CLI-ARM TURN FILES WRITTEN BEFORE SCHEMA VERSION 3.
   Those arms dispatch over MCP and the env-side record did not populate it;
   `nethack.py:_apply_tool_call` now synthesizes it from the dispatch arguments,
   and every record carries `dispatch_route`. OLDER FILES STILL HAVE IT EMPTY,
   so skill usage is still read from the trace `nodes` -- but ONLY the nodes
   flagged `sampled`, because `nodes` is a cumulative prefix replay of the
   conversation, not a list of calls. Counting every assistant node over-counted
   by ~15x (911 nodes -> "455 skills" for a 30-call rollout) and made every
   skill-adoption percentage garbage.

4. PRIME AGENT NESTS SKILLS INSIDE `ipython`. Counting the outer tool name
   reports 100% `ipython` / 0% everything else. We parse the code payload.

5. BOTH BALROG NUMBERS. The published metric is a `max` over (Dlvl, Xp), so a
   rollout that only levelled up keeps its headline score. Every row carries
   max, min, and the `xp!` flag for `max > 0 and min == 0`.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from tools.eval_metrics import (  # noqa: E402
    balrog_columns,
    degeneracy,
    executed_call_histogram,
    game_turns,
    pace_columns,
    read_ndjson,
    select_turn_files,
    skill_call_count,
)


def _se(v):
    return (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0


def _traces_by_seed(cell):
    """seed -> the rollout's v1 trace dict, from `<cell>/traces.jsonl`."""
    by_seed = {}
    path = os.path.join(cell, "traces.jsonl")
    if not os.path.exists(path):
        return by_seed
    for line in open(path):
        if not line.strip():
            continue
        trace = json.loads(line)
        seed = ((trace.get("task") or {}).get("data") or {}).get("idx")
        by_seed[seed] = trace
    return by_seed


def rows_for_cell(cell):
    """Per-seed rows plus the cell-level skill histogram, all deduped."""
    traces = _traces_by_seed(cell)
    seeds = set(traces) if traces else None
    chosen, dropped = select_turn_files(cell, seeds=seeds)

    rows, skills, no_monster = [], {}, 0
    for seed in sorted(set(chosen) | set(traces)):
        trace = traces.get(seed)
        turn_rows = read_ndjson(chosen[seed]) if seed in chosen else []
        if not turn_rows and trace is None:
            continue

        hist, hist_source = executed_call_histogram(trace=trace, turn_rows=turn_rows)
        for name, n in hist.items():
            skills[name] = skills.get(name, 0) + n

        dlvl = max((r.get("max_dlvl_reached") or 1 for r in turn_rows), default=1)
        xp = max(
            ((r.get("status") or {}).get("experience_level") or 1 for r in turn_rows), default=1
        )
        last = (turn_rows[-1].get("status") or {}) if turn_rows else {}
        no_monster += sum(
            1 for r in turn_rows if "no monster at" in (r.get("rendered_user_message") or "").lower()
        )

        stop = (trace or {}).get("stop_condition") or "RUNNING"
        calls, calls_source = skill_call_count(trace=trace, turn_rows=turn_rows)
        deg, deg_reason = degeneracy(stop, calls)

        row = dict(
            seed=seed,
            turns=len(turn_rows),
            dlvl=dlvl,
            xp=xp,
            hp=last.get("hitpoints"),
            maxhp=last.get("max_hitpoints"),
            stop=stop,
            calls=calls,
            calls_source=calls_source,
            hist_source=hist_source,
            died=last.get("hitpoints") == 0,
            degenerate=deg,
            degenerate_reason=deg_reason,
        )
        row.update(balrog_columns(dlvl, xp))
        row.update(
            pace_columns(dlvl, row["balrog_pct"], game_turns(turn_rows), sum(hist.values()) or None)
        )
        rows.append(row)
    return rows, skills, no_monster, dropped


def report(cell):
    rows, skills, no_monster, dropped = rows_for_cell(cell)
    total_skills = sum(skills.values())
    reveal = skills.get("reveal", 0)
    rollback = skills.get("rollback", 0)

    name = "/".join(cell.rstrip("/").split("/")[-2:])
    pct = (100 * reveal / total_skills) if total_skills else 0.0
    rb_pct = (100 * rollback / total_skills) if total_skills else 0.0
    print(
        f"== {name}: {len(rows)} rollouts | executed skill calls={total_skills} "
        f"reveal={reveal} ({pct:.1f}%) rollback={rollback} ({rb_pct:.1f}%) "
        f"| no-monster={no_monster}"
    )
    for path, reason in dropped:
        print(f"   ! ignored {os.path.basename(path)}: {reason}")
    for r in rows:
        xpflag = " xp!" if r["xp_carried"] else ""
        dpt = r["depth_per_game_turn"]
        dpc = r["depth_per_llm_call"]
        print(
            f"   seed {r['seed']}: {r['turns']:4d}t/{r['game_turns'] or 0:5d}g dlvl={r['dlvl']} "
            f"XL={r['xp']} BALROG={r['balrog_pct']:5.2f}/{r['balrog_min_pct']:.2f}%{xpflag} "
            f"dlvl/gturn={'n/a' if dpt is None else f'{dpt:.5f}'} "
            f"dlvl/call={'n/a' if dpc is None else f'{dpc:.4f}'} "
            f"hp={r['hp']}/{r['maxhp']} {r['stop']} "
            f"calls={r['calls']}({r['calls_source']})"
        )

    deg = [r for r in rows if r["degenerate"] is True]
    unknown = [r for r in rows if r["degenerate"] is None]
    real = [r for r in rows if r["degenerate"] is False]
    if real:
        d = [r["dlvl"] for r in real]
        b = [r["balrog_pct"] for r in real]
        bm = [r["balrog_min_pct"] for r in real]
        c = [r["calls"] for r in real]
        dpt = [r["depth_per_game_turn"] for r in real if r["depth_per_game_turn"] is not None]
        dpc = [r["depth_per_llm_call"] for r in real if r["depth_per_llm_call"] is not None]
        print(
            f"   -> n={len(real)}  dlvl {st.mean(d):.2f}±{_se(d):.2f}  "
            f"BALROG {st.mean(b):.2f}±{_se(b):.2f}% (min {st.mean(bm):.2f}±{_se(bm):.2f}%)  "
            f"died {100 * sum(r['died'] for r in real) / len(real):.0f}%  calls {st.mean(c):.0f}"
        )
        if dpt:
            print(
                f"   -> pace: dlvl/game-turn {st.mean(dpt):.5f}±{_se(dpt):.5f}  "
                f"dlvl/LLM-call {st.mean(dpc):.4f}±{_se(dpc):.4f}"
            )
    print(f"   -> degenerate: {[(r['seed'], r['degenerate_reason']) for r in deg] if deg else 'none'}")
    if unknown:
        print(f"   -> UNKNOWN degeneracy: {[(r['seed'], r['degenerate_reason']) for r in unknown]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        raise SystemExit(2)
    for cell in sys.argv[1:]:
        report(cell)
