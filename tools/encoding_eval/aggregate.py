"""Aggregate encoding-sweep rollouts into the comparison table.

Two entry points, deliberately the same shape as
`tools/cli_harness_eval/aggregate.py`:

  - `aggregate_cells({cell: [sample, ...]})` -- pure, on in-memory samples
    (what `tools/encoding_eval/run.py` drives through its injectable runner);
  - `aggregate_run_dir("<run_dir>")` -- reads a real output directory laid out
    as `<run_dir>/<cell>/{traces.jsonl, turns/<seed>_<pid>_<epoch>.ndjson}`
    (or the older `trace/` subdirectory name). This module previously had NO
    file I/O at all and therefore could not be run over a completed sweep.

Scoring
-------
Progression is the **real BALROG metric** (`nethack_harness.prompt.balrog`),
reported as BOTH numbers:

  - `balrog_pct` -- BALROG's published progression, a `max` over the reached
    `Dlvl:` and `Xp:` achievements;
  - `balrog_min_pct` -- the same table scored with `min`, the discriminator
    between "descended" and "levelled up while stuck";
  - `xp_carried_n` -- rollouts with `max > 0 and min == 0`, whose entire
    headline score came from experience level rather than descent.

It is NOT `progression_score`, the deprecated analytic proxy
`(DL/50)^1.3 * (XL/30)^0.6`, which reads ~2x off the real table and whose own
docstring says "do not quote it as BALROG" -- this module used to call it while
claiming in its docstring to reuse BALROG progression.

Pace
----
Each row also carries the progression SLOPE (`tools/eval_metrics.pace_columns`):
depth gained and BALROG-% per GAME turn (`status.time`) and per LLM call, since
the research question is how fast the agent progresses relative to a human.

Usage:
    PYTHONPATH=.:environments/nethack python -m tools.encoding_eval.aggregate <run_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from tools.eval_metrics import (  # noqa: E402
    balrog_columns,
    game_turns,
    mean_se,
    model_for_cell,
    pace_columns,
    read_ndjson,
    select_turn_files,
)


def _max_dlvl(sample: dict) -> int:
    if sample.get("max_dlvl") is not None:
        return int(sample["max_dlvl"])
    best = 0
    for e in sample.get("trace") or []:
        d = (e.get("status") or {}).get("depth")
        if d is None:
            d = e.get("max_dlvl_reached") or e.get("dlvl")
        if d is not None:
            best = max(best, int(d))
    return best


def _xp(sample: dict) -> int:
    if sample.get("xp_level") is not None:
        return int(sample["xp_level"])
    best = 0
    for e in sample.get("trace") or []:
        x = (e.get("status") or {}).get("experience_level")
        if x is not None:
            best = max(best, int(x))
    return best


def _llm_calls(sample: dict) -> int | None:
    for key in ("skill_calls", "total_tool_calls", "num_turns"):
        v = sample.get(key)
        if v is not None:
            return int(v)
    trace = sample.get("trace")
    return len(trace) if trace else None


def _sample_row(sample: dict) -> dict[str, Any]:
    """Per-rollout progression + pace, from one sample dict."""
    dlvl, xp = _max_dlvl(sample), _xp(sample)
    row = {"max_dlvl": dlvl, "xp_level": xp}
    row.update(balrog_columns(dlvl, xp))
    row.update(
        pace_columns(
            dlvl,
            row["balrog_pct"],
            sample.get("game_turns") or game_turns(sample.get("trace")),
            _llm_calls(sample),
        )
    )
    return row


_PACE_KEYS = (
    "depth_per_game_turn",
    "depth_per_llm_call",
    "balrog_pct_per_game_turn",
    "balrog_pct_per_llm_call",
)


def _progression_block(rows: list[dict]) -> dict[str, Any]:
    """The BALROG + pace columns shared by both entry points."""
    balrog = [r["balrog_pct"] for r in rows]
    balrog_min = [r["balrog_min_pct"] for r in rows]
    b_mu, b_se = mean_se(balrog)
    bm_mu, bm_se = mean_se(balrog_min)
    depths = [r["max_dlvl"] for r in rows]
    d_mu, d_se = mean_se(depths)
    out: dict[str, Any] = {
        "max_dlvl": max(depths) if depths else 0,
        "depth_mean": d_mu,
        "depth_se": d_se,
        "balrog_pct_mean": b_mu,
        "balrog_pct_se": b_se,
        "balrog_pct_max": max(balrog) if balrog else None,
        "balrog_min_pct_mean": bm_mu,
        "balrog_min_pct_se": bm_se,
        "xp_carried_n": sum(1 for r in rows if r["xp_carried"]),
    }
    for k in _PACE_KEYS:
        mu, se = mean_se([r[k] for r in rows])
        out[k + "_mean"] = mu
        out[k + "_se"] = se
        out[k + "_n"] = sum(1 for r in rows if r[k] is not None)
    return out


def aggregate_cells(cells: dict[str, list[dict]]) -> dict[str, Any]:
    """Pure aggregation over in-memory samples, one entry per cell."""
    from tools.eval_instrument import summarize_eval

    rows: dict[str, Any] = {}
    for enc, samples in cells.items():
        summ = summarize_eval(samples)
        per_rollout = [_sample_row(s) for s in samples]
        tokens = [s["tokens_per_turn"] for s in samples if s.get("tokens_per_turn") is not None]
        costs = [s["dollars"] for s in samples if s.get("dollars") is not None]
        row = {
            "n": summ["n"],
            "descent_rate": summ["descent_rate"],
            "ci_lo": summ["ci_lo"],
            "ci_hi": summ["ci_hi"],
            "avg_score": summ["avg_score"],
            "failure_taxonomy": summ["failure_taxonomy"],
            "tokens_per_turn": (sum(tokens) / len(tokens)) if tokens else None,
            "dollars_per_run": (sum(costs) / len(costs)) if costs else None,
        }
        row.update(_progression_block(per_rollout))
        rows[enc] = row
    return {"rows": rows}


# -- path-based entry point ---------------------------------------------------


def _cell_subdir(cell_dir: str) -> str | None:
    """`turns` (current) or `trace` (older sweeps), whichever exists."""
    for name in ("turns", "trace"):
        if os.path.isdir(os.path.join(cell_dir, name)):
            return name
    return None


def discover_cells(run_dir: str) -> list[str]:
    """Every subdirectory of `run_dir` holding a `traces.jsonl` or a per-turn
    trace directory. Unknown names are cells too -- an empty result is a loud
    failure in `main`, never a silently empty table."""
    if not os.path.isdir(run_dir):
        return []
    out = []
    for name in sorted(os.listdir(run_dir)):
        cell = os.path.join(run_dir, name)
        if not os.path.isdir(cell):
            continue
        if os.path.exists(os.path.join(cell, "traces.jsonl")) or _cell_subdir(cell):
            out.append(name)
    return out


def _traces_by_seed(cell_dir: str) -> dict:
    path = os.path.join(cell_dir, "traces.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        seed = ((t.get("task") or {}).get("data") or {}).get("idx")
        out[seed] = t
    return out


def aggregate_run_dir(run_dir: str, warn=None) -> list[dict]:
    """One row per cell, read off disk.

    Turn files go through `eval_metrics.select_turn_files`, so a rollout that
    retried contributes exactly ONE row instead of one per attempt (measured:
    `outputs/encoding_eval/calib_flash_b0_c3_partial/trace` holds 6 files for 5
    seeds and would otherwise report n=6)."""
    warn = warn or (lambda msg: print(msg, file=sys.stderr))
    rows = []
    cells = discover_cells(run_dir)
    if not cells:
        warn(
            f"aggregate: no cells found under {run_dir!r}. A cell is a SUBDIRECTORY "
            f"containing traces.jsonl and/or turns/ (or trace/). Found: "
            f"{sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else 'no such directory'}"
        )
        return rows

    for cell in cells:
        cell_dir = os.path.join(run_dir, cell)
        traces = _traces_by_seed(cell_dir)
        subdir = _cell_subdir(cell_dir) or "turns"
        chosen, dropped = select_turn_files(cell_dir, subdir=subdir, seeds=set(traces) or None)
        for path, reason in dropped:
            warn(f"aggregate: ignoring {path}: {reason}")
        if not chosen:
            warn(f"aggregate: cell {cell!r} has no usable per-turn trace files")
            continue

        per_rollout, died = [], 0
        for seed in sorted(chosen):
            turn_rows = read_ndjson(chosen[seed])
            trace = traces.get(seed) or {}
            sample = {
                "seed": seed,
                "trace": turn_rows,
                "game_turns": game_turns(turn_rows),
                "skill_calls": (trace.get("metrics") or {}).get("skill_calls"),
                "total_tool_calls": (trace.get("metrics") or {}).get("total_tool_calls"),
            }
            if any((r.get("status") or {}).get("hitpoints") == 0 for r in turn_rows):
                died += 1
            per_rollout.append(_sample_row(sample))

        descended = sum(1 for r in per_rollout if r["max_dlvl"] > 1)
        row = {
            "cell": cell,
            "model": model_for_cell(cell_dir),
            "n": len(per_rollout),
            "descent_rate": descended / len(per_rollout) if per_rollout else None,
            "died_pct": 100 * died / len(per_rollout) if per_rollout else None,
        }
        row.update(_progression_block(per_rollout))
        rows.append(row)
    return rows


# -- rendering ----------------------------------------------------------------


def _fmt(x, spec="{:.2f}"):
    return spec.format(x) if x is not None else "n/a"


_MD_COLS = [
    ("n", "n", "{:.0f}"),
    ("descent_rate", "descent rate", "{:.2f}"),
    ("depth_mean", "depth (mean)", "{:.2f}"),
    ("max_dlvl", "max dlvl", "{:.0f}"),
    ("balrog_pct_mean", "BALROG % max", "{:.2f}"),
    ("balrog_min_pct_mean", "BALROG % min", "{:.2f}"),
    ("depth_per_game_turn_mean", "dlvl / game-turn", "{:.5f}"),
    ("depth_per_llm_call_mean", "dlvl / LLM call", "{:.4f}"),
    ("balrog_pct_per_game_turn_mean", "BALROG% / game-turn", "{:.5f}"),
    ("balrog_pct_per_llm_call_mean", "BALROG% / LLM call", "{:.4f}"),
    ("tokens_per_turn", "tokens/turn", "{:.0f}"),
    ("dollars_per_run", "$/run", "{:.4f}"),
]

_NOTE = (
    "BALROG is the real metric (`balrog_progress`), never the deprecated "
    "`progression_score` proxy, and is reported as BOTH the published `max` over the "
    "(Dlvl, Xp) achievement axes and the `min` over the same table; 'xp-carried' "
    "counts rollouts scoring max > 0 with min == 0. Pace columns are slopes: "
    "'game-turn' is in-game `status.time`, not the LLM call count."
)


def table_to_markdown(table) -> str:
    """Render either `aggregate_cells(...)` output or `aggregate_run_dir(...)`
    rows."""
    if isinstance(table, dict):
        items = list(table.get("rows", {}).items())
    else:
        items = [(r.get("cell", "?"), r) for r in table]

    header = ["encoding"] + [label for _, label, _ in _MD_COLS] + ["xp-carried"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for enc, r in items:
        cells = [enc]
        for key, _, spec in _MD_COLS:
            cells.append(_fmt(r.get(key), spec))
        cells.append(f"{r.get('xp_carried_n', 0)}/{r.get('n', 0)}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Notes: " + _NOTE)
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m tools.encoding_eval.aggregate <run_dir>", file=sys.stderr)
        return 2
    run_dir = argv[0]
    if not os.path.isdir(run_dir):
        print(f"aggregate: no such run directory: {run_dir}", file=sys.stderr)
        return 2
    rows = aggregate_run_dir(run_dir)
    if not rows:
        print(
            f"aggregate: produced ZERO rows for {run_dir} -- refusing to write an empty "
            f"table. Check that its subdirectories contain traces.jsonl and/or turns/.",
            file=sys.stderr,
        )
        return 1
    md = table_to_markdown(rows)
    print(md)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(run_dir, "table.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    with open(os.path.join(run_dir, "table.md"), "w") as fh:
        fh.write(md + "\n")
    print(f"\nwrote {os.path.join(run_dir, 'table.md')} and {os.path.join(run_dir, 'table.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
