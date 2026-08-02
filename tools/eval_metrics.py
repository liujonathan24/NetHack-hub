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
import time
from typing import Any, Iterable

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "environments", "nethack")
)
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402

__all__ = [
    "PRICE_TABLES",
    "PRICE_TABLE_GLM_5_2",
    "PRICE_CACHE_PATH",
    "PRICE_ENDPOINT",
    "refresh_price_tables",
    "TABLE_PROVENANCE_NAME",
    "balrog_columns",
    "concurrent_rollout_files",
    "degeneracy",
    "executed_call_histogram",
    "game_turns",
    "mean_se",
    "model_for_cell",
    "pace_columns",
    "price_table_for",
    "read_ndjson",
    "rollout_clock_span",
    "select_turn_files",
    "skill_call_count",
    "table_paths",
    "turn_file_parts",
    "write_table",
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
# THESE ARE THE PROVIDER'S OWN NUMBERS, read from Prime Inference's `/models`
# endpoint (`pricing.input_usd_per_mtok` / `output_usd_per_mtok`) and refreshed
# by `refresh_price_tables()`. The previous values here were $1.40/$4.40, taken
# from public aggregator pages citing "official Z.ai rates"; Prime actually
# charges $1.68/$5.28, so every cost this module produced was 20% low before
# any other error compounded on top. Do not re-source these from a third-party
# pricing page -- the only authority for what this account is billed is the
# endpoint the account calls.
#
# `cached_input_per_million` DELIBERATELY EQUALS `input_per_million`. Prime's
# `/models` payload exposes exactly two prices per model and no cached-read
# discount, so a cache hit is billed at the full input rate until Prime
# publishes otherwise. The old 0.26 here was an OpenAI-style 0.19x guess and,
# combined with the sign error in `_call_cost` (see aggregate.py), it hid the
# single largest line item in the run: cache reads were 52% of all input.
#
# A model absent from this table gets NO price: `price_table_for` returns None
# and every cost derived from it reports "unavailable". This is deliberate --
# the previous code applied the GLM 5.2 table unconditionally and reported a
# GLM 5.2 price for a `z-ai/glm-4.7-flash` run (`outputs/trace_probe`). A wrong
# cost is worse than a missing one.
PRICE_TABLES: dict[str, dict[str, float]] = {
    "z-ai/glm-5.2": {
        "input_per_million": 1.68,
        "cached_input_per_million": 1.68,
        "output_per_million": 5.28,
    },
    "z-ai/glm-5.1": {
        "input_per_million": 1.75,
        "cached_input_per_million": 1.75,
        "output_per_million": 5.50,
    },
    "z-ai/glm-5": {
        "input_per_million": 1.20,
        "cached_input_per_million": 1.20,
        "output_per_million": 3.50,
    },
    "z-ai/glm-4.7": {
        "input_per_million": 0.60,
        "cached_input_per_million": 0.60,
        "output_per_million": 2.65,
    },
    "z-ai/glm-4.7-flash": {
        "input_per_million": 0.10,
        "cached_input_per_million": 0.10,
        "output_per_million": 0.43,
    },
}

#: Back-compat alias for the one table that was hardcoded module-wide.
PRICE_TABLE_GLM_5_2 = PRICE_TABLES["z-ai/glm-5.2"]

#: Where `refresh_price_tables` caches the provider's reply, so a sweep on a box
#: with no outbound network still prices correctly from the last fetch.
PRICE_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "model_prices.json"
)

PRICE_ENDPOINT = "https://api.pinference.ai/api/v1/models"


def _price_table_from_models_entry(entry: dict) -> dict[str, float] | None:
    """One `/models` record -> a price table, or None when it carries no
    pricing (some entries do not)."""
    pricing = entry.get("pricing") or {}
    inp = pricing.get("input_usd_per_mtok")
    out = pricing.get("output_usd_per_mtok")
    if inp is None or out is None:
        return None
    # No `cached_input_usd_per_mtok` exists in this payload. Read a cache hit at
    # the full input rate rather than inventing a discount -- see the note on
    # PRICE_TABLES. If Prime ever publishes one, honour it here and nowhere else.
    cached = pricing.get("cached_input_usd_per_mtok", inp)
    return {
        "input_per_million": float(inp),
        "cached_input_per_million": float(cached),
        "output_per_million": float(out),
    }


def refresh_price_tables(api_key: str | None = None, timeout: float = 20.0,
                         cache_path: str | None = None) -> dict[str, dict[str, float]]:
    """Fetch live prices from Prime's `/models` and merge them into
    `PRICE_TABLES`, caching the result.

    Returns the tables that were merged in (empty on any failure). NEVER
    raises: a budget number computed from the committed constants is worth more
    than a traceback, and the constants above are the last known-good fetch.
    Falls back to the on-disk cache when the endpoint cannot be reached.
    """
    cache_path = cache_path or PRICE_CACHE_PATH
    key = api_key or os.environ.get("PI_API_KEY") or os.environ.get("PRIME_API_KEY")
    fetched: dict[str, dict[str, float]] = {}
    try:
        import urllib.request

        req = urllib.request.Request(PRICE_ENDPOINT)
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        # Cloudflare in front of the endpoint 403s the default
        # `Python-urllib/3.x` agent while serving the identical curl request
        # 200. Without this the refresh always failed and always fell back --
        # silently correct, but never actually refreshing.
        req.add_header("User-Agent", "nethack-hub-eval/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for entry in payload.get("data") or []:
            model = entry.get("id")
            table = _price_table_from_models_entry(entry)
            if model and table:
                fetched[model] = table
    except Exception:
        fetched = {}
    if fetched:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as fh:
                json.dump(fetched, fh, indent=2, sort_keys=True)
        except OSError:
            pass
    elif os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                fetched = json.load(fh)
        except (OSError, ValueError):
            fetched = {}
    PRICE_TABLES.update(fetched)
    return fetched


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


# ---------------------------------------------------------------------------
# 5. Two aggregators, one output filename
# ---------------------------------------------------------------------------
#: Sidecar naming whichever aggregator last wrote `<run_dir>/table.{md,json}`.
TABLE_PROVENANCE_NAME = "table.provenance.json"


def table_paths(run_dir, producer: str) -> dict:
    """Every path `write_table` touches for `producer`, as a dict.

    `producer` is the aggregator's own short name (`cli_harness_eval`,
    `encoding_eval`, ...). Exposed separately so tests and callers can name the
    files without reproducing the convention.
    """
    run_dir = str(run_dir)
    return {
        "primary_md": os.path.join(run_dir, f"table.{producer}.md"),
        "primary_json": os.path.join(run_dir, f"table.{producer}.json"),
        "canonical_md": os.path.join(run_dir, "table.md"),
        "canonical_json": os.path.join(run_dir, "table.json"),
        "provenance": os.path.join(run_dir, TABLE_PROVENANCE_NAME),
    }


def _read_provenance(path):
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_table(run_dir, rows, markdown: str, *, producer: str, warn=None) -> dict:
    """Write one aggregator's results without ever destroying another's.

    THE DEFECT. `tools/cli_harness_eval/aggregate.py` and
    `tools/encoding_eval/aggregate.py` (and `encoding_eval/aggregate_run.py`)
    each wrote `<run_dir>/table.md` + `<run_dir>/table.json`. They compute
    DIFFERENT tables from the same directory -- the CLI one has cost, pace,
    post-death drain and per-arm action sources; the encoding one has token/turn
    and descent rate -- so running both over one run directory silently replaced
    the first result with the second. Nothing warned, and the files look
    identical from the outside.

    THE FIX, in two parts:

    * Each producer ALWAYS writes its own `table.<producer>.{md,json}`. Those
      names cannot collide, so a result can no longer be lost, whatever order
      the aggregators run in.
    * `table.{md,json}` is kept, because RUNBOOK Sec 6, `docs/experiments/*` and
      several shell scripts refer to it by name. It is a copy of the
      last-written table, stamped: the markdown carries a leading HTML comment
      naming its producer and pointing at the un-clobberable copy, and
      `table.provenance.json` records the same thing machine-readably. When the
      canonical file is about to be overwritten by a DIFFERENT producer, `warn`
      is called with a loud message naming the file that still holds the other
      producer's result.

    `table.json`'s shape is deliberately unchanged (the callers pass their own
    rows straight through) -- provenance goes in the sidecar, not in the payload,
    so `json.load(open("table.json"))` keeps returning what it always returned.

    Returns the `table_paths` dict, plus `"overwrote"` naming the displaced
    producer (or None).
    """
    warn = warn or (lambda msg: print(msg, file=sys.stderr))
    paths = table_paths(run_dir, producer)
    os.makedirs(str(run_dir), exist_ok=True)

    previous = _read_provenance(paths["provenance"])
    displaced = None
    if previous and previous.get("producer") and previous["producer"] != producer:
        displaced = previous["producer"]
        warn(
            f"aggregate: {paths['canonical_md']} and {paths['canonical_json']} were "
            f"last written by {displaced!r}; overwriting them with {producer!r}'s "
            f"table. {displaced!r}'s result is NOT lost -- it is preserved at "
            f"{os.path.join(str(run_dir), f'table.{displaced}.md')} "
            f"(and .json). Read the per-producer files, not table.md, when a run "
            f"directory is aggregated by more than one tool."
        )

    stamp = (
        f"<!-- generated by tools/{producer}/aggregate.py -- this is a copy of "
        f"table.{producer}.md, which no other aggregator writes. "
        f"See {TABLE_PROVENANCE_NAME}. -->\n"
    )
    body = markdown if markdown.endswith("\n") else markdown + "\n"
    for key in ("primary_md", "canonical_md"):
        with open(paths[key], "w") as fh:
            fh.write(stamp + body)
    for key in ("primary_json", "canonical_json"):
        with open(paths[key], "w") as fh:
            json.dump(rows, fh, indent=1)
    with open(paths["provenance"], "w") as fh:
        json.dump(
            {
                "producer": producer,
                "written_at": time.time(),
                "run_dir": os.path.abspath(str(run_dir)),
                "primary_md": os.path.basename(paths["primary_md"]),
                "primary_json": os.path.basename(paths["primary_json"]),
                "previous_producer": displaced,
                "argv": list(sys.argv),
            },
            fh,
            indent=1,
        )
    paths["overwrote"] = displaced
    return paths


# ---------------------------------------------------------------------------
# 6. `sec/call` is only meaningful for a rollout that had the process to itself
# ---------------------------------------------------------------------------
def rollout_clock_span(rows):
    """`(start, end, key)` of a rollout's per-turn timestamps, or `None`.

    `key` is `"t_mono"` when the records carry one, else `"t_wall"`. Both are
    stamped by the writer at the END of each LM turn.
    """
    if not rows:
        return None
    key = "t_mono" if rows[0].get("t_mono") is not None else "t_wall"
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    if len(vals) < 2:
        return None
    return min(vals), max(vals), key


def concurrent_rollout_files(turn_items) -> dict:
    """`{path: [paths it overlapped in time inside the same process]}`.

    WHY THIS EXISTS. `t_mono` is `time.monotonic()`, which is PROCESS-WIDE. The
    eval CLI runs a cell's seeds concurrently in ONE process, so while rollout A
    is between its turn `k` and turn `k+1` the event loop is also serving B: A's
    delta is A's LLM round-trip PLUS whatever share of the process B took. The
    `sec/call` column is a mean of those deltas, so for a concurrent cell it is
    not A's latency and not B's -- it is an artifact of how the two interleaved.
    Measured on `outputs/pilot_reveal/fog` (both seeds, one process, pid 33959):
    `1.3s -> 11.5s`, whose second half is dominated by one seed that ran 7.6x
    longer than the other in the same interpreter.

    Detection is exact rather than heuristic. Turn files are named
    `<seed>_<pid>_<epoch>.ndjson`, so the PID groups rollouts by process -- which
    is also the only scope in which two `t_mono` values are comparable at all --
    and two rollouts in one process are concurrent exactly when their
    `[first turn, last turn]` spans overlap.

    `turn_items` is `[(path, rows), ...]`.
    """
    by_pid: dict[int, list] = collections.defaultdict(list)
    for path, rows in turn_items:
        parts = turn_file_parts(path)
        span = rollout_clock_span(rows)
        if parts is None or span is None:
            continue
        by_pid[parts[1]].append((str(path), span))

    overlaps: dict[str, list] = {}
    for entries in by_pid.values():
        for i, (path_a, (a0, a1, key_a)) in enumerate(entries):
            for path_b, (b0, b1, key_b) in entries[i + 1:]:
                if key_a != key_b or a0 > b1 or b0 > a1:
                    continue
                overlaps.setdefault(path_a, []).append(path_b)
                overlaps.setdefault(path_b, []).append(path_a)
    return overlaps
