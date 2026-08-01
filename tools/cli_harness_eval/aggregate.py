"""Aggregate a CLI-harness-eval run's rollouts into the cross-cell results
table (the deliverable of the whole experiment -- see
`docs/superpowers/specs/2026-07-25-cli-harness-eval-design.md`).

Two channels per rollout:

  - the v1 `traces.jsonl` (one JSON line per rollout: `metrics`, `calls` with
    per-call token `usage`, `stop_condition`, `nodes`, ...) -- written by the
    eval CLI to `<OUTDIR>/traces.jsonl` (see `verifiers.v1.cli.output`);
  - the per-turn `turns/<seed>_<pid>_<epoch>.ndjson` (one file per rollout
    ATTEMPT: `status.hitpoints`, `status.time`, `max_dlvl_reached`,
    `t_wall`/`t_mono`, ...) -- written by
    `nethack_harness/helpers.py:_write_trace_entry` into whichever `trace_dir`
    the cell's config points at.

They are joined on the seed (`task.data.idx` on one side, the filename prefix
on the other) so per-rollout rates can be computed; when neither side exposes a
usable seed the channels are paired positionally, and when even that is
impossible they are aggregated independently and `n` is reported per channel.

Column-by-column rationale (each was a prior finding paid for by a review
cycle -- see `.superpowers/sdd/2026-07-25-cli-harness-eval/task-15-brief.md`):

  - **Depth** = mean +/- SE of `max_dlvl_reached`, from the turns trace (the
    primary axis).
  - **BALROG %** = BOTH `balrog_progress` (max, the published metric) and
    `balrog_progress_min`, plus an explicit `xp-carried` count. BALROG's metric
    is a `max` over the Dlvl and Xp achievement axes, so a rollout that only
    ever levelled up keeps its headline score -- `outputs/pilot_reveal/reveal`
    seed 1 scores 2.12% max / 0.00% min at Dlvl 4, XP 1. Never quote one alone.
    Never the deprecated `progression_score` proxy (~2x off).
  - **Pace** = progression SLOPE, because the research question is how FAST the
    agent progresses relative to a human: depth gained and BALROG-% per GAME
    turn (`status.time`, not the number of LLM calls -- one call runs a whole
    pathfinding macro) and per LLM call. Zero/absent denominators report `n/a`,
    never 0 and never infinity; a rollout that genuinely never descended
    reports a real 0.0.
  - **Died** is derived from `hitpoints == 0` in the turns trace, NEVER from
    `metrics.died` -- Task 10's committed artifact shows a rollout dead from
    turn 6 through turn 12 scored `died = 0.0`.
  - **Actions used** normalizes on the MEASURED count, never the nominal
    150-call budget: `metrics.total_tool_calls` for the control arm (a v0
    `ToolEnv` stock metric), `metrics.skill_calls` for the two CLI arms (the
    toolset-side referee, copied onto `trace.metrics` by `NetHackTask.finalize`
    -- which does NOT run on the v0-legacy path, hence
    `eval_metrics.skill_call_count`'s explicit fallback chain). The arms are NOT
    budget-matched: control.toml's `max_turns` caps LM turns while the CLI arms'
    referee grants exactly `max_skill_calls`. State this asymmetry in the table
    notes; do not "fix" it by comparing normalized percentages.
  - **Cost / rollout** is computed from `calls[].usage` against the price table
    FOR THE MODEL THE CELL ACTUALLY RAN (`eval_metrics.PRICE_TABLES`, keyed by
    the `model` in the cell's `config.toml`). A model with no table, or a
    rollout whose calls carry no usage data, reports cost as `None`
    ("unavailable"). This module never estimates tokens from cost or vice versa,
    and never prices a run with another model's table -- it previously reported
    a GLM 5.2 price for a `z-ai/glm-4.7-flash` run.
  - **Seconds / call** is the mean of consecutive `t_wall` (or `t_mono`) deltas,
    split into first-half / second-half so latency growth across a rollout is
    visible instead of averaged away (~17s -> ~37s on the acceptance artifacts).
  - **Post-death drain** counts calls issued strictly after the first zero-HP
    turn -- wasted budget and spend. Should be ~0 after Task 14.

Note on cost availability and the brief's Prime Agent claim: the brief states
Prime Agent's model calls are "not intercepted" and its token split is
therefore permanently unavailable. That is NOT what the evidence in this repo
shows -- `tests/test_prime_agent_harness.py` pins that Prime Agent routes
through the SAME verifiers interception endpoint as the other two arms. So
`rollout_cost` is arm-agnostic: a real number whenever a rollout's `calls`
carry usage AND the model is priced, `None` otherwise.

Usage:
    PYTHONPATH=.:environments/nethack python -m tools.cli_harness_eval.aggregate <run_dir>

`<run_dir>` contains one subdirectory per CELL. The three canonical arm names
(`control/`, `claude_code/`, `prime_agent/`) sort first, but ANY subdirectory
holding a `traces.jsonl` or a `turns/` directory is aggregated -- a sweep laid
out by variant name (`outputs/pilot_reveal/{fog,reveal}`) used to produce an
empty table and exit 0. An empty result is now a loud, non-zero failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from tools.eval_metrics import (  # noqa: E402
    PRICE_TABLE_GLM_5_2,
    PRICE_TABLES,
    balrog_columns,
    degeneracy,
    game_turns,
    mean_se,
    model_for_cell,
    pace_columns,
    price_table_for,
    read_ndjson,
    select_turn_files,
    skill_call_count,
    turn_file_parts,
)

__all__ = [
    "ARMS",
    "ACTIONS_METRIC",
    "PRICE_TABLES",
    "PRICE_TABLE_GLM_5_2",
    "actions_used",
    "aggregate_arm",
    "aggregate_run",
    "discover_cells",
    "post_death_drain",
    "rollout_cost",
    "rollout_from_turns",
    "seconds_per_call_halves",
    "to_markdown",
]

ARMS = ("control", "claude_code", "prime_agent")

ACTIONS_METRIC = {
    "control": "total_tool_calls",
    "claude_code": "skill_calls",
    "prime_agent": "skill_calls",
}

_mean_se = mean_se  # kept: several call sites and older notebooks import it


# -- per-rollout: the turns/*.ndjson channel ---------------------------------


def _hitpoints(turn: dict):
    status = turn.get("status")
    if isinstance(status, dict) and "hitpoints" in status:
        return status["hitpoints"]
    return turn.get("hp")


def _clock_deltas(turns: list[dict]) -> list[float]:
    """Consecutive deltas of `t_mono` if the first turn carries one (added
    alongside `t_wall` by Task 14, immune to a mid-rollout clock step),
    else `t_wall`."""
    key = "t_mono" if turns and turns[0].get("t_mono") is not None else "t_wall"
    vals = [t.get(key) for t in turns if t.get(key) is not None]
    return [vals[i] - vals[i - 1] for i in range(1, len(vals))]


def seconds_per_call_halves(turns: list[dict]):
    """(first_half_mean, second_half_mean) of consecutive-call deltas, so
    latency growth across a rollout is visible instead of averaged away.
    `None, None` if there are fewer than two clock samples."""
    deltas = _clock_deltas(turns)
    if not deltas:
        return None, None
    half = len(deltas) // 2
    if half == 0:
        mu = sum(deltas) / len(deltas)
        return mu, mu
    first, second = deltas[:half], deltas[half:]
    return sum(first) / len(first), sum(second) / len(second)


def post_death_drain(turns: list[dict]) -> int:
    """Calls issued STRICTLY AFTER the first zero-HP turn (the death call
    itself is not drain -- it is the call that ended the character)."""
    death_idx = None
    for i, t in enumerate(turns):
        if _hitpoints(t) == 0:
            death_idx = i
            break
    if death_idx is None:
        return 0
    return len(turns) - death_idx - 1


def rollout_from_turns(turns: list[dict]) -> dict:
    """Per-rollout stats derived purely from the turns/*.ndjson channel."""
    max_dlvl, max_xp, died = 1, 1, False
    for t in turns:
        max_dlvl = max(max_dlvl, t.get("max_dlvl_reached") or t.get("dlvl") or 1)
        status = t.get("status")
        if isinstance(status, dict):
            max_xp = max(max_xp, status.get("experience_level") or 1)
        if _hitpoints(t) == 0:
            died = True
    first_half, second_half = seconds_per_call_halves(turns)
    row = {
        "max_dlvl": max_dlvl,
        "max_xp": max_xp,
        "died": died,
        "n_turns": len(turns),
        "game_turns": game_turns(turns),
        "post_death_drain": post_death_drain(turns),
        "sec_per_call_first_half": first_half,
        "sec_per_call_second_half": second_half,
    }
    row.update(balrog_columns(max_dlvl, max_xp))
    return row


def _read_turns_file(path) -> list[dict]:
    return read_ndjson(path)


# -- per-rollout: the traces.jsonl channel -----------------------------------


def trace_seed(trace: dict):
    """The seed a trace ran, i.e. `task.data.idx`; `None` if absent."""
    return ((trace.get("task") or {}).get("data") or {}).get("idx")


def actions_used_with_source(trace: dict, arm: str = ""):
    """`(measured action count, the field it came from)`.

    The arm's designated metric first -- `total_tool_calls` for the control
    arm's v0 legacy-bridge rollout, `skill_calls` for the two CLI arms'
    toolset-side referee. When it is absent -- which is ALWAYS the case for
    `skill_calls` on the v0-legacy path, where `NetHackTask.finalize` never runs
    -- this falls through to `eval_metrics.skill_call_count`'s explicit chain
    rather than raising KeyError or silently substituting a sentinel. `(None,
    "unavailable")` if even that finds nothing: report, don't synthesize.

    The source is carried into the table so a column headed `skill_calls` can
    never quietly be showing a different field's value."""
    metric = ACTIONS_METRIC.get(arm)
    metrics = trace.get("metrics") or {}
    if metric and metrics.get(metric) is not None:
        return metrics[metric], f"metrics.{metric}"
    return skill_call_count(trace=trace)


def actions_used(trace: dict, arm: str = ""):
    """The measured (never nominal) action count for `arm`; see
    `actions_used_with_source`."""
    return actions_used_with_source(trace, arm)[0]


def _call_cost(call: dict, price: dict):
    usage = call.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        return None
    completion = usage.get("completion_tokens") or 0
    cached = min(usage.get("cached_input_tokens") or 0, prompt)
    fresh = prompt - cached
    return (
        fresh * price["input_per_million"]
        + cached * price["cached_input_per_million"]
        + completion * price["output_per_million"]
    ) / 1_000_000


def rollout_cost(trace: dict, price: dict | None = PRICE_TABLE_GLM_5_2):
    """Sum of `_call_cost` over every call in `trace["calls"]` that carries
    usage. `None` ("unavailable") if `price` is None (the model this cell ran
    is not in `PRICE_TABLES` -- do NOT reach for another model's table), if the
    trace has no calls, or if none of them carry usage. Never estimated from a
    nominal call count or inverted from a different arm's average."""
    if price is None:
        return None
    calls = trace.get("calls") or []
    costs = [c for c in (_call_cost(call, price) for call in calls) if c is not None]
    if not costs:
        return None
    return sum(costs)


# -- joining the two channels --------------------------------------------------


def _pair_channels(traces: list[dict], turn_items: list[tuple]):
    """`[(key, trace_or_None, turn_rows_or_None), ...]`.

    `turn_items` is `[(seed_or_None, rows), ...]`. Joined on the seed when both
    sides expose distinct seeds; else paired positionally when the two channels
    hold the same number of rollouts (the committed acceptance artifacts, whose
    filenames predate the `<seed>_<pid>_<epoch>` convention); else left
    unjoined, so nothing is invented by an accidental mis-pairing."""
    trace_seeds = [trace_seed(t) for t in traces]
    turn_seeds = [s for s, _ in turn_items]
    joinable = (
        traces
        and turn_items
        and all(s is not None for s in trace_seeds)
        and all(s is not None for s in turn_seeds)
        and len(set(trace_seeds)) == len(trace_seeds)
        and len(set(turn_seeds)) == len(turn_seeds)
    )
    if joinable:
        tmap = dict(zip(trace_seeds, traces))
        rmap = dict(turn_items)
        return [(k, tmap.get(k), rmap.get(k)) for k in sorted(set(trace_seeds) | set(turn_seeds))]
    if len(traces) == len(turn_items):
        return [(i, traces[i], turn_items[i][1]) for i in range(len(traces))]
    return [(f"trace{i}", t, None) for i, t in enumerate(traces)] + [
        (f"turns{i}", None, rows) for i, (_, rows) in enumerate(turn_items)
    ]


# -- cell-level aggregate ------------------------------------------------------


def aggregate_arm(
    arm: str,
    traces_jsonl_paths,
    turns_paths,
    price: dict | None = PRICE_TABLE_GLM_5_2,
    model: str | None = None,
) -> dict:
    """Aggregate one cell's rollouts. `traces_jsonl_paths` is one or more
    `traces.jsonl` files (each may hold several rollouts, one per line);
    `turns_paths` is one or more per-turn NDJSON files (one per rollout,
    already de-duplicated by `select_turn_files` when called via
    `aggregate_run`)."""
    traces = []
    for p in traces_jsonl_paths:
        for line in open(p):
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    turn_items = []
    for p in turns_paths:
        parts = turn_file_parts(p)
        turn_items.append((parts[0] if parts else None, _read_turns_file(p)))

    turn_rollouts = [rollout_from_turns(rows) for _, rows in turn_items]

    depths = [r["max_dlvl"] for r in turn_rollouts]
    depth_mean, depth_se = mean_se(depths)

    balrog = [r["balrog_pct"] for r in turn_rollouts]
    balrog_mean, balrog_se = mean_se(balrog)
    balrog_min = [r["balrog_min_pct"] for r in turn_rollouts]
    balrog_min_mean, balrog_min_se = mean_se(balrog_min)
    xp_carried_n = sum(1 for r in turn_rollouts if r["xp_carried"])

    died_flags = [r["died"] for r in turn_rollouts]
    died_pct = 100 * sum(died_flags) / len(died_flags) if died_flags else None

    drains = [r["post_death_drain"] for r in turn_rollouts]
    drain_mean, _ = mean_se(drains)

    sec_first_mean, _ = mean_se([r["sec_per_call_first_half"] for r in turn_rollouts])
    sec_second_mean, _ = mean_se([r["sec_per_call_second_half"] for r in turn_rollouts])

    actions_pairs = [actions_used_with_source(t, arm) for t in traces]
    actions = [a for a, _ in actions_pairs]
    actions_mean, actions_se = mean_se(actions)
    action_sources = sorted({s for a, s in actions_pairs if a is not None})
    actions_source = (
        "/".join(action_sources)
        if action_sources
        else f"metrics.{ACTIONS_METRIC.get(arm, 'skill_calls')}"
    )

    costs = [rollout_cost(t, price) for t in traces]
    cost_available = [c for c in costs if c is not None]
    cost_mean, cost_se = mean_se(cost_available)

    # -- joined, per-rollout: pace + degeneracy ------------------------------
    pace_rows, degenerate, unknown_degeneracy = [], [], []
    for key, trace, rows in _pair_channels(traces, turn_items):
        rows = rows or []
        stats = rollout_from_turns(rows) if rows else None
        calls, calls_source = skill_call_count(trace=trace, turn_rows=rows)
        stop = (trace or {}).get("stop_condition")
        deg, reason = degeneracy(stop, calls)
        if deg is True:
            degenerate.append((key, reason))
        elif deg is None:
            unknown_degeneracy.append((key, reason))
        if stats is None:
            continue
        pace = pace_columns(stats["max_dlvl"], stats["balrog_pct"], stats["game_turns"], calls)
        pace.update({"seed": key, "calls_source": calls_source, "degenerate": deg})
        pace_rows.append(pace)

    pace_means = {}
    for k in (
        "depth_per_game_turn",
        "depth_per_llm_call",
        "balrog_pct_per_game_turn",
        "balrog_pct_per_llm_call",
    ):
        mu, se = mean_se([r[k] for r in pace_rows])
        pace_means[k + "_mean"] = mu
        pace_means[k + "_se"] = se
        pace_means[k + "_n"] = sum(1 for r in pace_rows if r[k] is not None)

    row = {
        "arm": arm,
        "cell": arm,
        "model": model,
        "n": len(turn_rollouts),
        "n_traces": len(traces),
        "depth_mean": depth_mean,
        "depth_se": depth_se,
        "balrog_pct_mean": balrog_mean,
        "balrog_pct_se": balrog_se,
        "balrog_min_pct_mean": balrog_min_mean,
        "balrog_min_pct_se": balrog_min_se,
        "xp_carried_n": xp_carried_n,
        "died_pct": died_pct,
        # The nominal metric this arm is DEFINED to report ...
        "actions_metric_name": ACTIONS_METRIC.get(arm, "skill_calls"),
        # ... and the field the number actually came from, so a `skill_calls`
        # column can never quietly be showing `total_tool_calls`.
        "actions_source": actions_source,
        "actions_mean": actions_mean,
        "actions_se": actions_se,
        "actions_n_available": sum(1 for a in actions if a is not None),
        "cost_mean": cost_mean,
        "cost_se": cost_se,
        "cost_n_available": len(cost_available),
        "cost_n_total": len(costs),
        "cost_priced_model": model if price is not None else None,
        "sec_per_call_first_half_mean": sec_first_mean,
        "sec_per_call_second_half_mean": sec_second_mean,
        "post_death_drain_mean": drain_mean,
        "game_turns_mean": mean_se([r["game_turns"] for r in turn_rollouts])[0],
        "degenerate": degenerate,
        "unknown_degeneracy": unknown_degeneracy,
        "pace_rollouts": pace_rows,
    }
    row.update(pace_means)
    return row


def _fmt(x, spec="{:.2f}"):
    return spec.format(x) if x is not None else "n/a"


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| Cell | n | Depth (mean ± SE) | BALROG % max (mean ± SE) | "
        "BALROG % min (mean ± SE) | xp-carried | Died % | "
        "Actions used (measured, mean ± SE) | Cost/rollout ($) | "
        "dlvl / game-turn | dlvl / LLM call | BALROG% / game-turn | "
        "BALROG% / LLM call | sec/call 1st half → 2nd half | Post-death drain |",
        "|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|",
    ]
    for r in rows:
        depth = f"{_fmt(r['depth_mean'])} ± {_fmt(r['depth_se'])}"
        balrog = f"{_fmt(r['balrog_pct_mean'])} ± {_fmt(r['balrog_pct_se'])}"
        balrog_min = f"{_fmt(r['balrog_min_pct_mean'])} ± {_fmt(r['balrog_min_pct_se'])}"
        xpc = f"{r['xp_carried_n']}/{r['n']}"
        died = f"{_fmt(r['died_pct'], '{:.0f}')}%" if r["died_pct"] is not None else "n/a"
        label = r.get("actions_source") or r["actions_metric_name"]
        actions = (
            f"{label}={_fmt(r['actions_mean'])} ± {_fmt(r['actions_se'])}"
            if r["actions_mean"] is not None
            else f"{label}=unavailable"
        )
        if r["cost_mean"] is not None:
            note = "" if r["cost_n_available"] == r["cost_n_total"] else (
                f" ({r['cost_n_available']}/{r['cost_n_total']} rollouts)"
            )
            cost = f"${_fmt(r['cost_mean'], '{:.4f}')} ± {_fmt(r['cost_se'], '{:.4f}')}{note}"
        elif r.get("model") and r.get("cost_priced_model") is None:
            cost = f"unavailable (no price table for {r['model']})"
        else:
            cost = "unavailable (no token usage in trace)"
        dpt = f"{_fmt(r['depth_per_game_turn_mean'], '{:.5f}')} ± {_fmt(r['depth_per_game_turn_se'], '{:.5f}')}"
        dpc = f"{_fmt(r['depth_per_llm_call_mean'], '{:.4f}')} ± {_fmt(r['depth_per_llm_call_se'], '{:.4f}')}"
        bpt = f"{_fmt(r['balrog_pct_per_game_turn_mean'], '{:.5f}')}"
        bpc = f"{_fmt(r['balrog_pct_per_llm_call_mean'], '{:.4f}')}"
        sec = (
            f"{_fmt(r['sec_per_call_first_half_mean'], '{:.1f}')}s → "
            f"{_fmt(r['sec_per_call_second_half_mean'], '{:.1f}')}s"
        )
        drain = _fmt(r["post_death_drain_mean"], "{:.1f}")
        lines.append(
            f"| {r.get('cell', r['arm'])} | {r['n']} | {depth} | {balrog} | {balrog_min} | "
            f"{xpc} | {died} | {actions} | {cost} | {dpt} | {dpc} | {bpt} | {bpc} | "
            f"{sec} | {drain} |"
        )
    lines.append("")
    lines.append(
        "Notes: BALROG is reported as BOTH its published `max` over the (Dlvl, Xp) "
        "achievement axes AND the `min` over the same table; 'xp-carried' counts "
        "rollouts with max > 0 and min == 0, i.e. whose entire headline score came "
        "from levelling rather than descent. 'Actions used' is the MEASURED count, "
        "never the nominal 150-call budget -- the control arm's `max_turns` caps LM "
        "turns (a no-tool-call turn burns the turn; extra parallel tool calls past "
        "the first are dropped), so it gets *at most* the nominal budget in executed "
        "skills, while the CLI arms' toolset-side referee grants exactly that many. "
        "The arms are therefore NOT budget-matched; do not read a lower control "
        "action count as the model choosing to stop early. Pace columns are slopes: "
        "'game-turn' is in-game `status.time` (one LLM call runs a whole pathfinding "
        "macro, so the two denominators differ by ~10x); depth gained is "
        "`max_dlvl - 1`. Cost uses the price table for the model the cell actually "
        "ran and is 'unavailable' when that model is unpriced -- never another "
        "model's table."
    )
    for r in rows:
        if r["degenerate"]:
            lines.append(
                f"Degenerate in `{r.get('cell', r['arm'])}` (excluded from nothing here -- "
                f"reported, not dropped): {r['degenerate']}"
            )
        if r["unknown_degeneracy"]:
            lines.append(
                f"UNKNOWN degeneracy in `{r.get('cell', r['arm'])}` (skill_calls could not "
                f"be measured): {r['unknown_degeneracy']}"
            )
    return "\n".join(lines)


# -- run-directory layout: <run_dir>/<cell>/{traces.jsonl,turns/*.ndjson} ----


def discover_cells(run_dir: str) -> list[str]:
    """Every subdirectory of `run_dir` that looks like a cell, i.e. holds a
    `traces.jsonl` or a `turns/` directory.

    The three canonical arm names sort first (so an arm-shaped run keeps its
    familiar row order); everything else follows alphabetically. This function
    exists because the aggregator used to iterate a hardcoded
    `("control", "claude_code", "prime_agent")` and emit an EMPTY table with
    exit 0 for a sweep laid out by variant name."""
    if not os.path.isdir(run_dir):
        return []
    found = []
    for name in sorted(os.listdir(run_dir)):
        cell = os.path.join(run_dir, name)
        if not os.path.isdir(cell):
            continue
        if os.path.exists(os.path.join(cell, "traces.jsonl")) or os.path.isdir(
            os.path.join(cell, "turns")
        ):
            found.append(name)
    return [a for a in ARMS if a in found] + [f for f in found if f not in ARMS]


def _cell_paths(run_dir: str, cell: str):
    """`(cell_dir, traces.jsonl or None, [one turn file per seed], dropped)`.

    The turn files come from `eval_metrics.select_turn_files`, NOT from
    `glob('turns/*.ndjson')`: a rollout that retried writes one file per
    attempt, so globbing silently inflates `n` (a 5-seed cell was once reported
    as n=15). See RUNBOOK.md Sec 6."""
    cell_dir = os.path.join(run_dir, cell)
    traces = os.path.join(cell_dir, "traces.jsonl")
    traces = traces if os.path.exists(traces) else None

    seeds = None
    if traces:
        seeds = set()
        for line in open(traces):
            line = line.strip()
            if line:
                s = trace_seed(json.loads(line))
                if s is not None:
                    seeds.add(s)
        seeds = seeds or None

    chosen, dropped = select_turn_files(cell_dir, seeds=seeds)
    return cell_dir, traces, [chosen[s] for s in sorted(chosen)], dropped


def aggregate_run(run_dir: str, warn=None) -> list[dict]:
    """One row per discovered cell. `warn` receives human-readable strings for
    everything ignored (superseded retries, stale seeds) so a caller can print
    them; nothing is dropped silently."""
    warn = warn or (lambda msg: print(msg, file=sys.stderr))
    rows = []
    cells = discover_cells(run_dir)
    if not cells:
        warn(
            f"aggregate: no cells found under {run_dir!r}. A cell is a SUBDIRECTORY "
            f"containing traces.jsonl and/or turns/. Found: "
            f"{sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else 'no such directory'}"
        )
        return rows
    for cell in cells:
        cell_dir, traces_path, turns_paths, dropped = _cell_paths(run_dir, cell)
        for path, reason in dropped:
            warn(f"aggregate: ignoring {path}: {reason}")
        if traces_path is None and not turns_paths:
            warn(f"aggregate: cell {cell!r} has neither traces.jsonl nor usable turn files")
            continue
        model = model_for_cell(cell_dir)
        price = price_table_for(model)
        if price is None:
            warn(
                f"aggregate: cell {cell!r} ran model {model!r}, which is not in "
                f"PRICE_TABLES ({sorted(PRICE_TABLES)}) -- cost reported as unavailable "
                f"rather than priced with another model's table"
            )
        row = aggregate_arm(
            cell, [traces_path] if traces_path else [], turns_paths, price=price, model=model
        )
        rows.append(row)
    return rows


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m tools.cli_harness_eval.aggregate <run_dir>", file=sys.stderr)
        return 2
    run_dir = argv[0]
    if not os.path.isdir(run_dir):
        print(f"aggregate: no such run directory: {run_dir}", file=sys.stderr)
        return 2
    rows = aggregate_run(run_dir)
    if not rows:
        print(
            f"aggregate: produced ZERO rows for {run_dir} -- refusing to write an empty "
            f"table. Check that its subdirectories contain traces.jsonl and/or turns/.",
            file=sys.stderr,
        )
        return 1
    md = to_markdown(rows)
    print(md)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    out_json = os.path.join(run_dir, "table.json")
    out_md = os.path.join(run_dir, "table.md")
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=1)
    with open(out_md, "w") as fh:
        fh.write(md + "\n")
    print(f"\nwrote {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
