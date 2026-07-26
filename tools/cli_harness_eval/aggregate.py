"""Aggregate a CLI-harness-eval arm's rollouts into the cross-arm results
table (the deliverable of the whole experiment -- see
`docs/superpowers/specs/2026-07-25-cli-harness-eval-design.md`).

Two channels per rollout, read independently (same shape as
`tools/encoding_eval/aggregate_run.py`'s cell/results split) rather than
joined by id:

  - the v1 `traces.jsonl` (one JSON line per rollout: `metrics`, `calls` with
    per-call token `usage`, `stop_condition`, ...) -- written by the eval CLI
    to `<OUTDIR>/traces.jsonl` (see `verifiers.v1.cli.output`);
  - the per-turn `turns/*.ndjson` (one file per rollout: `status.hitpoints`,
    `max_dlvl_reached`, `t_wall`/`t_mono`, ...) -- written by
    `nethack_harness/helpers.py:_write_trace_entry` into whichever
    `trace_dir` the arm's config points at.

Column-by-column rationale (each was a prior finding paid for by a review
cycle -- see `.superpowers/sdd/2026-07-25-cli-harness-eval/task-15-brief.md`):

  - **Depth** = mean +/- SE of `max_dlvl_reached`, from the turns trace (the
    primary axis).
  - **BALROG %** = `nethack_harness.prompt.balrog.balrog_progress(depth, xp)`,
    the real BALROG progression metric, not the deprecated analytic proxy.
  - **Died** is derived from `hitpoints == 0` in the turns trace, NEVER from
    `metrics.died` -- Task 10's committed artifact shows a rollout dead from
    turn 6 through turn 12 scored `died = 0.0` (`_apply_tool_call` kept
    accepting calls against a corpse; Task 14 added live termination, but
    historical traces and any path that fix misses still need the
    trace-derived value).
  - **Actions used** normalizes on the MEASURED count, never the nominal
    150-call budget: `metrics.total_tool_calls` for the control arm (a v0
    `ToolEnv` stock metric -- `verifiers.envs.tool_env.ToolEnv.total_tool_calls`,
    preserved verbatim into the v1 trace by `legacy.rollout_output_to_trace`),
    `metrics.skill_calls` for the two CLI arms (the toolset-side referee,
    copied onto `trace.metrics` by `NetHackTask.finalize`). The arms are NOT
    budget-matched: control.toml's `max_turns` caps LM turns (a turn can pass
    with 0 skills executed, or drop every parallel call past the first), so
    control gets *at most* `max_turns` executed skills while the CLI arms'
    toolset-side referee grants exactly `max_skill_calls`. State this
    asymmetry in the table notes; do not "fix" it by comparing normalized
    percentages of a shared nominal budget.
  - **Cost / rollout** is computed from `calls[].usage` (`prompt_tokens`,
    `completion_tokens`, `cached_input_tokens`) against one GLM 5.2 price
    table (`PRICE_TABLE_GLM_5_2` below). A rollout whose calls carry no usage
    data at all reports cost as `None` ("unavailable") -- this module never
    estimates tokens from cost or vice versa (one equation, two unknowns;
    prompt caching breaks it further). See the module docstring note below on
    why this is NOT hardcoded per-arm.
  - **Seconds / call** is the mean of consecutive `t_wall` (or `t_mono`, once
    present -- Task 14 added it alongside `t_wall`, immune to a mid-rollout
    clock step) deltas, split into first-half / second-half so latency growth
    across a rollout is visible instead of averaged away. Measured on the
    committed acceptance artifacts: ~17s -> ~37s per call.
  - **Post-death drain** counts calls issued strictly after the first
    zero-HP turn -- wasted budget and spend. Should be ~0 after Task 14; a
    nonzero value is the regression signal.

Note on cost availability and the brief's Prime Agent claim: the brief states
Prime Agent's model calls are "not intercepted" and its token split is
therefore permanently unavailable. That is NOT what the evidence in this repo
shows: `tests/test_prime_agent_harness.py::test_models_json_routes_the_agent_through_interception_not_a_vendor`
pins that Prime Agent routes through the SAME verifiers interception endpoint
as the other two arms, and the committed `task10_prime_agent_seed0.traces.jsonl`
carries real per-call `usage` (see `task-15-report.md` Sec on this). Rather
than hardcode an arm-name exclusion that would silently contradict the data,
`rollout_cost` below is arm-agnostic: it reports a real number whenever a
rollout's `calls` carry usage, and `None` ("unavailable") whenever they do
not -- which reproduces the brief's fallback behavior automatically if a
future run's routing changes, without asserting a fact this codebase's own
tests contradict today.

Usage:
    PYTHONPATH=.:environments/nethack python -m tools.cli_harness_eval.aggregate <run_dir>

`<run_dir>` is expected to contain one subdirectory per arm
(`control/`, `claude_code/`, `prime_agent/`), each with `traces.jsonl` (from
`--output_dir`) and `turns/*.ndjson` (from the arm's `trace_dir`, which
`launch_cell.sh` points at `<arm_outdir>/turns`).
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "environments", "nethack")
)
from nethack_harness.prompt.balrog import balrog_progress  # noqa: E402

ARMS = ("control", "claude_code", "prime_agent")

ACTIONS_METRIC = {
    "control": "total_tool_calls",
    "claude_code": "skill_calls",
    "prime_agent": "skill_calls",
}

# $/1M tokens for z-ai/glm-5.2. Sourced from public aggregator pricing pages
# (Requesty, SiliconFlow) citing "official Z.ai rates" as of 2026-07; NOT
# verified against an actual Prime Inference invoice for this project's
# account, which may carry a different reseller markup. Treat as the
# best-available placeholder and correct it here (one place) if a real
# invoice disagrees -- every cost figure in the table derives from this one
# table, per the brief's "one GLM 5.2 price table" requirement.
PRICE_TABLE_GLM_5_2 = {
    "input_per_million": 1.40,
    "cached_input_per_million": 0.26,
    "output_per_million": 4.40,
}


def _mean_se(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var / n)


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
    return {
        "max_dlvl": max_dlvl,
        "max_xp": max_xp,
        "died": died,
        "n_turns": len(turns),
        "post_death_drain": post_death_drain(turns),
        "sec_per_call_first_half": first_half,
        "sec_per_call_second_half": second_half,
    }


def _read_turns_file(path) -> list[dict]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


# -- per-rollout: the traces.jsonl channel -----------------------------------


def actions_used(trace: dict, arm: str):
    """The measured (never nominal) action count for `arm`: `total_tool_calls`
    for the control arm's v0 legacy-bridge rollout, `skill_calls` for the two
    CLI arms' toolset-side referee. `None` if the trace does not carry the
    field -- report, don't synthesize."""
    metric = ACTIONS_METRIC[arm]
    metrics = trace.get("metrics") or {}
    return metrics.get(metric)


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


def rollout_cost(trace: dict, price: dict = PRICE_TABLE_GLM_5_2):
    """Sum of `_call_cost` over every call in `trace["calls"]` that carries
    usage. `None` ("unavailable") if the trace has no calls, or none of them
    carry usage -- never estimated from a nominal call count or inverted from
    a different arm's average."""
    calls = trace.get("calls") or []
    costs = [c for c in (_call_cost(call, price) for call in calls) if c is not None]
    if not costs:
        return None
    return sum(costs)


# -- arm-level aggregate -------------------------------------------------------


def aggregate_arm(arm: str, traces_jsonl_paths, turns_paths, price=PRICE_TABLE_GLM_5_2) -> dict:
    """Aggregate one arm's rollouts. `traces_jsonl_paths` is one or more
    `traces.jsonl` files (each may hold several rollouts, one per line);
    `turns_paths` is one or more per-turn NDJSON files (one per rollout).
    The two channels are aggregated independently, exactly like
    `tools/encoding_eval/aggregate_run.py`'s cell/results split -- they are
    not joined by rollout id, so `n` below is reported per channel."""
    traces = []
    for p in traces_jsonl_paths:
        for line in open(p):
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    turn_rollouts = [rollout_from_turns(_read_turns_file(p)) for p in turns_paths]

    depths = [r["max_dlvl"] for r in turn_rollouts]
    depth_mean, depth_se = _mean_se(depths)

    balrog = [100 * balrog_progress(r["max_dlvl"], r["max_xp"]) for r in turn_rollouts]
    balrog_mean, balrog_se = _mean_se(balrog)

    died_flags = [r["died"] for r in turn_rollouts]
    died_pct = 100 * sum(died_flags) / len(died_flags) if died_flags else None

    drains = [r["post_death_drain"] for r in turn_rollouts]
    drain_mean, _ = _mean_se(drains)

    sec_first_mean, _ = _mean_se([r["sec_per_call_first_half"] for r in turn_rollouts])
    sec_second_mean, _ = _mean_se([r["sec_per_call_second_half"] for r in turn_rollouts])

    actions = [actions_used(t, arm) for t in traces]
    actions_mean, actions_se = _mean_se(actions)

    costs = [rollout_cost(t, price) for t in traces]
    cost_available = [c for c in costs if c is not None]
    cost_mean, cost_se = _mean_se(cost_available)

    return {
        "arm": arm,
        "n": len(turn_rollouts),
        "n_traces": len(traces),
        "depth_mean": depth_mean,
        "depth_se": depth_se,
        "balrog_pct_mean": balrog_mean,
        "balrog_pct_se": balrog_se,
        "died_pct": died_pct,
        "actions_metric_name": ACTIONS_METRIC[arm],
        "actions_mean": actions_mean,
        "actions_se": actions_se,
        "actions_n_available": sum(1 for a in actions if a is not None),
        "cost_mean": cost_mean,
        "cost_se": cost_se,
        "cost_n_available": len(cost_available),
        "cost_n_total": len(costs),
        "sec_per_call_first_half_mean": sec_first_mean,
        "sec_per_call_second_half_mean": sec_second_mean,
        "post_death_drain_mean": drain_mean,
    }


def _fmt(x, spec="{:.2f}"):
    return spec.format(x) if x is not None else "n/a"


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| Arm | n | Depth (mean ± SE) | BALROG % (mean ± SE) | Died % | "
        "Actions used (measured, mean ± SE) | Cost/rollout ($) | "
        "sec/call 1st half → 2nd half | Post-death drain |",
        "|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|",
    ]
    for r in rows:
        depth = f"{_fmt(r['depth_mean'])} ± {_fmt(r['depth_se'])}"
        balrog = f"{_fmt(r['balrog_pct_mean'])} ± {_fmt(r['balrog_pct_se'])}"
        died = f"{_fmt(r['died_pct'], '{:.0f}')}%" if r["died_pct"] is not None else "n/a"
        actions = (
            f"{r['actions_metric_name']}={_fmt(r['actions_mean'])} ± {_fmt(r['actions_se'])}"
            if r["actions_mean"] is not None
            else f"{r['actions_metric_name']}=unavailable"
        )
        if r["cost_mean"] is not None:
            note = "" if r["cost_n_available"] == r["cost_n_total"] else (
                f" ({r['cost_n_available']}/{r['cost_n_total']} rollouts)"
            )
            cost = f"${_fmt(r['cost_mean'], '{:.4f}')} ± {_fmt(r['cost_se'], '{:.4f}')}{note}"
        else:
            cost = "unavailable (no token usage in trace)"
        sec = f"{_fmt(r['sec_per_call_first_half_mean'], '{:.1f}')}s → {_fmt(r['sec_per_call_second_half_mean'], '{:.1f}')}s"
        drain = _fmt(r["post_death_drain_mean"], "{:.1f}")
        lines.append(
            f"| {r['arm']} | {r['n']} | {depth} | {balrog} | {died} | {actions} | "
            f"{cost} | {sec} | {drain} |"
        )
    lines.append("")
    lines.append(
        "Notes: 'Actions used' is the MEASURED count, never the nominal 150-call budget "
        "-- the control arm's `max_turns` caps LM turns (a no-tool-call turn burns the "
        "turn; extra parallel tool calls past the first are dropped), so it gets *at "
        "most* the nominal budget in executed skills, while the CLI arms' toolset-side "
        "referee grants exactly that many. The arms are therefore NOT budget-matched; "
        "do not read a lower control action count as the model choosing to stop early."
    )
    return "\n".join(lines)


# -- run-directory layout: <run_dir>/<arm>/{traces.jsonl,turns/*.ndjson} ----


def _arm_dir_paths(run_dir: str, arm: str):
    arm_dir = os.path.join(run_dir, arm)
    traces = os.path.join(arm_dir, "traces.jsonl")
    turns = sorted(glob.glob(os.path.join(arm_dir, "turns", "*.ndjson")))
    return arm_dir, traces, turns


def aggregate_run(run_dir: str) -> list[dict]:
    rows = []
    for arm in ARMS:
        arm_dir, traces_path, turns_paths = _arm_dir_paths(run_dir, arm)
        if not os.path.isdir(arm_dir) or not os.path.exists(traces_path):
            continue
        rows.append(aggregate_arm(arm, [traces_path], turns_paths))
    return rows


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/cli_harness_eval/run1"
    rows = aggregate_run(run_dir)
    md = to_markdown(rows)
    print(md)
    out_json = os.path.join(run_dir, "table.json")
    out_md = os.path.join(run_dir, "table.md")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_json, "w"), indent=1)
    open(out_md, "w").write(md + "\n")
    print(f"\nwrote {out_md} and {out_json}")
