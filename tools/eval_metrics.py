"""Shared scoring/aggregation primitives for every NetHack eval aggregator.

This module exists because the same four mistakes kept being re-derived
independently in `tools/cli_harness_eval/progress.py`,
`tools/cli_harness_eval/aggregate.py` and `tools/encoding_eval/aggregate.py`,
and each one silently changed a published number:

1. **Executed calls are NOT `len(trace["nodes"])`.** A verifiers v1 trace's
   `nodes` list is a CUMULATIVE PREFIX REPLAY of the conversation: after `k`
   sampled turns it contains every prefix `1..k`, so an assistant message that
   issued one tool call on turn 3 reappears in the replay of turns 4, 5, ... .
   Measured on `outputs/trace_probe/B0_reveal`: 911 nodes, 455 assistant nodes
   carrying `tool_calls`, for a rollout that emitted **30** calls. Exactly the
   nodes with `sampled == True` are the newly-sampled ones, and counting only
   those reproduces the engine-side referee counters in `metrics` to the call
   (`np_move_to`: 23 sampled == `metrics.np_move_to_calls` 23). See
   `executed_call_histogram`.

2. **`skill_calls` does not exist on the v0-legacy path.** `eval.log` reads
   `running Nx1 v0 rollouts ... (legacy: nethack)`, so v1's
   `NetHackTask.finalize` never runs and never writes `skill_calls`,
   `max_dlvl_reached`, `died`, ... . `metrics['skill_calls']` raises KeyError
   and `metrics.get('skill_calls', 999)` marks every rollout non-degenerate.
   `skill_call_count` below makes the fallback chain explicit and returns the
   source it used, so an aggregate can say WHERE its number came from.

3. **One row per seed, joined on the PID in the filename.** Turn files are
   named `<seed>_<pid>_<epoch>.ndjson` (`nethack_harness/helpers.py`
   `_write_trace_entry`), one per rollout ATTEMPT. Globbing `turns/*.ndjson`
   double-counts retries and silently inflates `n` from a stale directory.
   `select_turn_files` performs the documented join (RUNBOOK.md Sec 6,
   docs/experiments/exp2-cli-harness-results.md trap 1).

4. **BALROG must always be reported as the (max, min) pair.** The published
   metric is a `max` over the Dlvl / Xp achievement axes, so a rollout that
   only ever levelled up keeps its headline score. `balrog_columns` emits both
   plus the explicit `xp_carried` flag.

Everything here is pure except `select_turn_files` / `read_ndjson` /
`model_for_cell`, which are the file-layout join and are kept in one place for
the same reason.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import re
import sys
from typing import Any, Iterable

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "environments", "nethack")
)
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402

__all__ = [
    "PRICE_TABLES",
    "PRICE_TABLE_GLM_5_2",
    "balrog_columns",
    "degeneracy",
    "executed_call_histogram",
    "game_turns",
    "mean_se",
    "model_for_cell",
    "pace_columns",
    "price_table_for",
    "read_ndjson",
    "select_turn_files",
    "skill_call_count",
    "turn_file_parts",
]


# ---------------------------------------------------------------------------
# small stats helper (was duplicated verbatim in three aggregators)
# ---------------------------------------------------------------------------


def mean_se(xs: Iterable[Any]) -> tuple[float | None, float | None]:
    """`(mean, standard error)` over the non-`None` values; `(None, None)` if
    there are none. Never returns NaN -- a missing measurement is `None`, so a
    caller cannot accidentally format it as a number."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var / n)


# ---------------------------------------------------------------------------
# BALROG: always the (max, min) pair, never one alone
# ---------------------------------------------------------------------------


def balrog_columns(max_dlvl: int, xp_level: int, **kw) -> dict[str, Any]:
    """The three BALROG columns every table this repo emits must carry.

    `balrog_pct` is BALROG's published metric (a `max` over the Dlvl / Xp
    achievement axes); `balrog_min_pct` is the same table scored with `min`,
    which is the discriminator between "descended" and "levelled up while
    stuck"; `xp_carried` is `max > 0 and min == 0`, i.e. the whole headline
    score came from experience level and not from descent at all.

    Real example from `outputs/pilot_reveal/reveal` seed 1 (Dlvl 4 at XP 1):
    2.12 max / 0.00 min / xp_carried=True.
    """
    hi, lo = balrog_both(max_dlvl, xp_level, **kw)
    return {
        "balrog_pct": 100.0 * hi,
        "balrog_min_pct": 100.0 * lo,
        "xp_carried": hi > 0.0 and lo == 0.0,
    }


# ---------------------------------------------------------------------------
# executed tool calls: dedupe the cumulative prefix replay
# ---------------------------------------------------------------------------

# Prime Agent nests real skills inside an `ipython` call
# (`await nethack.np_move_to(...)`), so counting the outer tool name reports
# 100% `ipython` and 0% everything else.
_SKILL_IN_CODE_RE = re.compile(
    r"\b(reveal|rollback|np_[a-z_]+|explore_and_descend|move_to|descend)\s*\("
)

_METRIC_CALL_RE = re.compile(r"^(?P<name>.+)_calls$")


def _tool_call_names(tool_calls) -> list[str]:
    names = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or (call.get("function") or {}).get("name") or ""
        name = name.replace("mcp__nethack__", "")
        if name == "ipython":
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args).get("code", "")
                except ValueError:
                    pass
            names.extend(_SKILL_IN_CODE_RE.findall(str(args)))
        elif name:
            names.append(name)
    return names


def _histogram_from_turn_rows(turn_rows) -> collections.Counter:
    hist: collections.Counter = collections.Counter()
    for row in turn_rows or []:
        for name in _tool_call_names(row.get("tool_calls")):
            hist[name] += 1
    return hist


def _histogram_from_sampled_nodes(trace) -> collections.Counter:
    """Count ONLY nodes flagged `sampled` -- `nodes` is a cumulative prefix
    replay, so every other assistant node is a repeat of an earlier turn."""
    hist: collections.Counter = collections.Counter()
    for node in (trace or {}).get("nodes") or []:
        if not node.get("sampled"):
            continue
        msg = node.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        for name in _tool_call_names(msg.get("tool_calls")):
            hist[name] += 1
    return hist


def _histogram_from_metrics(trace) -> collections.Counter:
    """The engine-side referee counters (`np_move_to_calls`, `reveal_calls`,
    ...) that the v0 legacy bridge copies onto `trace.metrics`."""
    hist: collections.Counter = collections.Counter()
    for key, value in ((trace or {}).get("metrics") or {}).items():
        m = _METRIC_CALL_RE.match(key)
        if not m:
            continue
        name = m.group("name")
        if name in ("skill", "total_tool", "tool"):
            continue
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n:
            hist[name] += n
    return hist


def executed_call_histogram(trace=None, turn_rows=None) -> tuple[collections.Counter, str]:
    """`(Counter of executed skill calls, source label)` for ONE rollout.

    Source precedence, most authoritative first:

      1. `turns/*.ndjson` `tool_calls` -- what the environment actually applied.
         Authoritative for the control / v0-legacy arm. (Empty for the CLI arms,
         which dispatch over MCP: the env-side record never populates it.)
      2. `trace["nodes"]` restricted to `sampled == True` -- the newly sampled
         assistant message of each turn, i.e. the prefix replay deduped. This is
         what the CLI arms have to use, and it reproduces (3) exactly.
      3. `trace["metrics"]`'s `<skill>_calls` referee counters.

    NEVER count every assistant node: that over-counts by ~15x on a 30-turn
    rollout and grows quadratically with rollout length.
    """
    hist = _histogram_from_turn_rows(turn_rows)
    if hist:
        return hist, "turns.tool_calls"
    hist = _histogram_from_sampled_nodes(trace)
    if hist:
        return hist, "trace.nodes[sampled]"
    hist = _histogram_from_metrics(trace)
    if hist:
        return hist, "trace.metrics.*_calls"
    return collections.Counter(), "unavailable"


# ---------------------------------------------------------------------------
# skill_calls: the explicit fallback chain (v1 writes it, v0-legacy does not)
# ---------------------------------------------------------------------------


def skill_call_count(trace=None, turn_rows=None) -> tuple[int | None, str]:
    """`(number of executed skill calls, source label)` for ONE rollout.

    `metrics['skill_calls']` only exists on the v1 path (`NetHackTask.finalize`).
    Under `running Nx1 v0 rollouts ... (legacy: nethack)` it is absent, so a
    literal subscript raises KeyError and `.get('skill_calls', 999)` silently
    marks the rollout non-degenerate. The chain below is explicit and reports
    which link produced the value; `(None, "unavailable")` when no link does --
    never a sentinel that happens to pass the degeneracy rule.
    """
    metrics = (trace or {}).get("metrics") or {}
    for key in ("skill_calls", "total_tool_calls"):
        value = metrics.get(key)
        if value is not None:
            try:
                return int(value), f"metrics.{key}"
            except (TypeError, ValueError):
                pass
    hist, source = executed_call_histogram(trace=trace, turn_rows=turn_rows)
    if hist:
        return sum(hist.values()), source
    if turn_rows:
        # A turn file with rows but no tool_calls at all still proves the
        # rollout took that many environment steps.
        return len(turn_rows), "len(turns)"
    return None, "unavailable"


# ---------------------------------------------------------------------------
# degeneracy rule (docs/experiments/exp2-cli-harness-results.md Sec 4)
# ---------------------------------------------------------------------------


def degeneracy(stop_condition, skill_calls) -> tuple[bool | None, str]:
    """`(is_degenerate, reason)`; `is_degenerate is None` means UNKNOWN.

    Rule: degenerate if `stop_condition == "error"`, or `skill_calls < 40`, or
    (`stop_condition == "agent_completed"` and `skill_calls < 80`).

    An unknown `skill_calls` returns `None` rather than `False`: "we could not
    measure it" and "it is fine" are different statements, and collapsing them
    is exactly what `metrics.get('skill_calls', 999)` did.
    """
    if stop_condition == "error":
        return True, "stop_condition=error"
    if skill_calls is None:
        return None, "skill_calls unavailable"
    if skill_calls < 40:
        return True, f"skill_calls={skill_calls} < 40"
    if stop_condition == "agent_completed" and skill_calls < 80:
        return True, f"agent_completed with skill_calls={skill_calls} < 80"
    return False, ""


# ---------------------------------------------------------------------------
# progression pace: how FAST the agent descends
# ---------------------------------------------------------------------------


def game_turns(turn_rows) -> int | None:
    """In-game elapsed turns for a rollout = max `status.time` over its turn
    rows. NOT `len(turn_rows)`: one LLM call runs a whole pathfinding macro, so
    a 99-call rollout can span 936 game turns. `None` if no row carries it."""
    times = [
        (row.get("status") or {}).get("time")
        for row in turn_rows or []
        if isinstance(row.get("status"), dict)
    ]
    times = [t for t in times if t is not None]
    return max(times) if times else None


def pace_columns(
    max_dlvl: int,
    balrog_pct: float | None,
    n_game_turns: int | None,
    n_llm_calls: int | None,
) -> dict[str, Any]:
    """Progression SLOPE for one rollout -- the research question is how fast
    the agent descends, so depth alone is not enough.

    Depth gained is `max_dlvl - 1` (every rollout starts on Dlvl 1; using
    `max_dlvl` itself would credit a rollout that never moved with a positive
    slope). Rates are `None` -- unmeasurable -- when the denominator is zero or
    missing, and `0.0` -- a real measurement -- when the agent simply never
    descended.
    """
    gained = max(0, int(max_dlvl) - 1)

    def _rate(numerator, denominator):
        if numerator is None or not denominator or denominator <= 0:
            return None
        return numerator / denominator

    return {
        "depth_gained": gained,
        "game_turns": n_game_turns,
        "llm_calls": n_llm_calls,
        "depth_per_game_turn": _rate(gained, n_game_turns),
        "depth_per_llm_call": _rate(gained, n_llm_calls),
        "balrog_pct_per_game_turn": _rate(balrog_pct, n_game_turns),
        "balrog_pct_per_llm_call": _rate(balrog_pct, n_llm_calls),
    }


PACE_KEYS = (
    "depth_per_game_turn",
    "depth_per_llm_call",
    "balrog_pct_per_game_turn",
    "balrog_pct_per_llm_call",
)


# ---------------------------------------------------------------------------
# one row per seed: the PID join over turns/<seed>_<pid>_<epoch>.ndjson
# ---------------------------------------------------------------------------


_TURN_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<pid>\d+)_(?P<ts>\d+)\.ndjson$")


def turn_file_parts(path) -> tuple[int, int, int] | None:
    """`(seed, pid, epoch)` parsed out of a `<seed>_<pid>_<epoch>.ndjson` turn
    filename, or `None` if the name does not match that shape."""
    m = _TURN_NAME_RE.match(os.path.basename(str(path)))
    if not m:
        return None
    return int(m.group("seed")), int(m.group("pid")), int(m.group("ts"))


def read_ndjson(path) -> list[dict]:
    """Every well-formed JSON line of an NDJSON file; malformed lines skipped
    (a killed rollout can leave a half-written final line)."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def select_turn_files(cell_dir, subdir: str = "turns", seeds=None):
    """The documented PID join: exactly ONE turn file per seed.

    `turns/` is SHARED per cell and a rollout that retried writes a new
    `<seed>_<pid>_<epoch>.ndjson` per attempt, so `glob('turns/*.ndjson')`
    reports one row per ATTEMPT. That is how a 5-seed cell was once reported as
    n=15 (`best_fixed` held files from two earlier attempts) and how a 2-turn
    stub beside a real 247-turn attempt was counted as its own rollout.

    Selection, per seed: keep the attempt with the most turn rows; break ties on
    the later `(pid, epoch)`. A partial stub can therefore never outvote the
    real attempt, and the result is provably one row per seed.

    `seeds`, when given (the seed set from `traces.jsonl`), restricts the result
    -- a stale directory holding seeds this run never scheduled cannot inflate
    `n`.

    Returns `(chosen, dropped)` where `chosen` is `{seed: path}` and `dropped`
    is a list of `(path, reason)` for everything not selected, so the caller can
    say out loud what it ignored.
    """
    paths = sorted(glob.glob(os.path.join(str(cell_dir), subdir, "*.ndjson")))
    by_seed: dict[int, list] = collections.defaultdict(list)
    dropped: list[tuple[str, str]] = []

    for path in paths:
        parts = turn_file_parts(path)
        if parts is None:
            dropped.append((path, "filename is not <seed>_<pid>_<epoch>.ndjson"))
            continue
        seed, pid, ts = parts
        if seeds is not None and seed not in seeds:
            dropped.append((path, f"seed {seed} is not in traces.jsonl (stale directory?)"))
            continue
        rows = read_ndjson(path)
        if not rows:
            dropped.append((path, "no turn rows"))
            continue
        by_seed[seed].append((len(rows), pid, ts, path))

    chosen: dict[int, str] = {}
    for seed, attempts in sorted(by_seed.items()):
        attempts.sort(key=lambda a: (a[0], a[2], a[1]))
        n_rows, pid, ts, path = attempts[-1]
        chosen[seed] = path
        for other in attempts[:-1]:
            dropped.append(
                (
                    other[3],
                    f"superseded retry for seed {seed}: {other[0]} rows vs {n_rows} "
                    f"in pid {pid} attempt {ts}",
                )
            )
    return chosen, dropped


# ---------------------------------------------------------------------------
# model-aware pricing: never misprice, report unavailable instead
# ---------------------------------------------------------------------------

# $/1M tokens, keyed by the `model` string in the cell's `config.toml`.
#
# Sourced from public aggregator pricing pages (Requesty, SiliconFlow) citing
# "official Z.ai rates" as of 2026-07; NOT verified against an actual Prime
# Inference invoice for this project's account, which may carry a different
# reseller markup. Correct them HERE (one place) if a real invoice disagrees.
#
# A model absent from this table gets NO price: `price_table_for` returns None
# and every cost derived from it reports "unavailable". This is deliberate --
# the previous code applied the GLM 5.2 table unconditionally and reported a
# GLM 5.2 price for a `z-ai/glm-4.7-flash` run (`outputs/trace_probe`). A wrong
# cost is worse than a missing one.
PRICE_TABLES: dict[str, dict[str, float]] = {
    "z-ai/glm-5.2": {
        "input_per_million": 1.40,
        "cached_input_per_million": 0.26,
        "output_per_million": 4.40,
    },
}

#: Back-compat alias for the one table that was hardcoded module-wide.
PRICE_TABLE_GLM_5_2 = PRICE_TABLES["z-ai/glm-5.2"]


def price_table_for(model: str | None) -> dict[str, float] | None:
    """The $/1M-token table for `model`, or `None` when the model is unknown
    (including `None`). Callers must report cost as unavailable on `None`, not
    fall back to another model's table."""
    if not model:
        return None
    return PRICE_TABLES.get(model) or PRICE_TABLES.get(model.strip())


_MODEL_TOML_RE = re.compile(r"^\s*model\s*=\s*[\"'](?P<model>[^\"']+)[\"']", re.M)
_MODEL_LOG_RE = re.compile(r"rollouts on (?P<model>\S+)")


def model_for_cell(cell_dir) -> str | None:
    """The model a cell actually ran, read from the run's own artifacts:
    `config.toml`'s `model = "..."` first, then `eval.log`'s
    `running Nx1 v0 rollouts on <model> (legacy: nethack)`. `None` if neither
    is present -- which makes cost unavailable rather than mispriced."""
    cfg = os.path.join(str(cell_dir), "config.toml")
    if os.path.exists(cfg):
        with open(cfg) as fh:
            m = _MODEL_TOML_RE.search(fh.read())
        if m:
            return m.group("model")
    log = os.path.join(str(cell_dir), "eval.log")
    if os.path.exists(log):
        # eval.log can be tens of MB; the banner is in the first few lines.
        with open(log, errors="replace") as fh:
            head = fh.read(1 << 16)
        m = _MODEL_LOG_RE.search(head)
        if m:
            return m.group("model")
    return None
