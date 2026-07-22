"""Cross-model x harness-version sweep for Experiment 2 (harness modifications).

The claim under test (Experiment 2): progression gains come from the *harness
architecture* (skills / prompt / navigation / memory / typed interface), not
from one model's internal quirks. To show that, we run the **same** harness
version across a diverse model set {Gemini, GLM, GPT-5.5, ...} and check that
the BALROG-style progression signal moves together across models -- and that
switching harness version moves every model in the same direction.

This module mirrors ``tools/encoding_eval`` (pure aggregation + an injectable
runner seam) but sweeps the ``(model, harness_version)`` matrix instead of the
``(encoding, model)`` one.

Layers
------
- ``build_matrix(models, harness_versions)`` -> list of cells.
- ``run_sweep(matrix, runner)`` -> aggregated table (progression mean/stdev per
  cell, descent rate + Wilson CI, and a per-harness cross-model spread that is
  the actual robustness statistic). ``runner(cell) -> list[sample]`` is the
  injectable seam; tests / the ``--demo`` path pass a synthetic runner so the
  module is exercisable with **no** model calls or engine build.
- ``make_prime_runner(...)`` -> a real ``runner`` that drives ``prime eval run``
  with ``NETHACK_HARNESS=<version>`` set (the ``harness_overlay.py`` seam) and a
  per-model provider/endpoint. ``dry_run=True`` prints the argv without spending.

The harness-version toggle
--------------------------
``environments/nethack/harness_overlay.py`` reads ``NETHACK_HARNESS=<name>`` in
``load_environment`` and overlays system prompt / per-step formatter / enabled
skills / reward weights. Setting the env var per cell is how one cell runs the
"baseline" harness and another runs the "fixed" harness against the *same*
model. (NOTE: the overlay's config loader -- ``tools.launchpad.core.harness`` --
is not yet vendored into this repo, so today an unknown ``NETHACK_HARNESS`` name
is a logged no-op. Until that loader lands, ``--harness-versions`` cells differ
only in label. See ``docs/experiments/exp2_harness_modifications.md`` -> "What
remains".)

Provider / endpoint mapping (best-effort; override per cell with ``provider``):
  gemini-*      -> Google Generative Language OpenAI-compatible endpoint
                   (GEMINI_API_KEY) -- reachable locally on this machine.
  z-ai/glm-*    -> Prime Inference (PI_API_KEY) -- team-billed, reachable.
  gpt-*, o1*    -> OpenAI (OPENAI_API_KEY) -- key NOT set here; flag before use.
This mirrors ``approaches/voyager/reverse_curriculum_sweep.py::_endpoint_for``.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# Repo layout: this file is tools/harness_sweep.py
_REPO = Path(__file__).resolve().parents[1]
_ENV_DIR = _REPO / "environments" / "nethack"

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_PRIME_BASE = "https://api.pinference.ai/api/v1"
_OPENAI_BASE = "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Provider / endpoint resolution (pure)
# ---------------------------------------------------------------------------
def endpoint_for(model: str) -> dict:
    """Map a model id to ``{provider, base_url, key_env, key_present}``.

    Prefix-based, matching the wiring already used elsewhere in the repo. The
    caller can override by putting ``provider`` on the cell.
    """
    m = model.lower()
    if m.startswith("gemini"):
        return {"provider": "gemini", "base_url": _GEMINI_BASE,
                "key_env": "GEMINI_API_KEY",
                "key_present": bool(os.environ.get("GEMINI_API_KEY"))}
    if m.startswith("z-ai/") or "glm" in m:
        return {"provider": "prime", "base_url": _PRIME_BASE,
                "key_env": "PI_API_KEY",
                "key_present": bool(os.environ.get("PI_API_KEY"))}
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return {"provider": "openai", "base_url": _OPENAI_BASE,
                "key_env": "OPENAI_API_KEY",
                "key_present": bool(os.environ.get("OPENAI_API_KEY"))}
    # Fallback: assume Prime Inference (serves the open-weight zoo).
    return {"provider": "prime", "base_url": _PRIME_BASE,
            "key_env": "PI_API_KEY",
            "key_present": bool(os.environ.get("PI_API_KEY"))}


# ---------------------------------------------------------------------------
# Matrix construction (pure)
# ---------------------------------------------------------------------------
def build_matrix(models: list[str], harness_versions: list[str]) -> list[dict]:
    """Full model x harness-version grid as a flat list of cells."""
    return [{"model": mdl, "harness": hv}
            for hv in harness_versions for mdl in models]


def cell_key(cell: dict) -> str:
    return f"{cell['harness']}::{cell['model']}"


# ---------------------------------------------------------------------------
# Per-sample progression (pure) -- reuses the shipped BALROG metric
# ---------------------------------------------------------------------------
def _progression_score(max_dlvl: int, xp_level: int) -> float:
    """BALROG P(ascend) proxy. Prefers the shipped module; falls back to an
    inline copy of the same calibrated formula so this tool works even when
    ``nethack_harness`` is not importable (e.g. engine not built)."""
    try:
        if str(_ENV_DIR) not in sys.path:
            sys.path.insert(0, str(_ENV_DIR))
        from nethack_harness.prompt.balrog import progression_score  # pure module
        return progression_score(max_dlvl, xp_level)
    except Exception:
        dl = max(0.0, float(max_dlvl)) / 50.0
        xl = max(0.0, float(xp_level)) / 30.0
        return max(0.0, min(1.0, (dl ** 1.3) * (xl ** 0.6)))


def _sample_progression(sample: dict) -> float:
    """P(ascend) proxy for one rollout from its deepest (DL, XL)."""
    max_dlvl = sample.get("max_dlvl")
    xp = sample.get("xp_level")
    if max_dlvl is None or xp is None:
        best_dl, best_xp = 0, 0
        for e in sample.get("trace") or []:
            st = e.get("status") or {}
            if st.get("depth") is not None:
                best_dl = max(best_dl, int(st["depth"]))
            if st.get("experience_level") is not None:
                best_xp = max(best_xp, int(st["experience_level"]))
        max_dlvl = best_dl if max_dlvl is None else max_dlvl
        xp = best_xp if xp is None else xp
    return _progression_score(int(max_dlvl or 0), int(xp or 1))


def _descended(sample: dict) -> bool:
    if sample.get("descent_reward") is not None:
        return float(sample["descent_reward"]) >= 1.0
    return int(sample.get("max_dlvl") or 0) >= 2


def _wilson(k: int, n: int) -> tuple[float, float]:
    try:
        from tools.eval_instrument import wilson_ci  # reuse if importable
        return wilson_ci(k, n)
    except Exception:
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
        return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------
def aggregate_cells(cells: dict[str, list[dict]]) -> dict[str, Any]:
    """Per-cell progression mean/stdev + descent rate/CI, plus a per-harness
    cross-model spread (the robustness statistic)."""
    rows: dict[str, Any] = {}
    for key, samples in cells.items():
        n = len(samples)
        progs = [_sample_progression(s) for s in samples]
        k = sum(1 for s in samples if _descended(s))
        lo, hi = _wilson(k, n)
        harness, model = key.split("::", 1)
        rows[key] = {
            "harness": harness,
            "model": model,
            "n": n,
            "progression_mean": (sum(progs) / n) if n else 0.0,
            "progression_stdev": (statistics.pstdev(progs) if n > 1 else 0.0),
            "progression_max": (max(progs) if progs else 0.0),
            "descent_rate": (k / n) if n else 0.0,
            "ci_lo": lo,
            "ci_hi": hi,
        }

    # Cross-model spread per harness version: if the harness (not the model) is
    # what drives progression, the per-model means should cluster -> low spread,
    # and the mean should shift when the harness version changes.
    per_harness: dict[str, Any] = {}
    by_h: dict[str, list[float]] = {}
    for r in rows.values():
        by_h.setdefault(r["harness"], []).append(r["progression_mean"])
    for h, means in by_h.items():
        per_harness[h] = {
            "models": len(means),
            "progression_mean_across_models": (sum(means) / len(means)) if means else 0.0,
            "progression_spread_stdev": (statistics.pstdev(means) if len(means) > 1 else 0.0),
            "progression_min": min(means) if means else 0.0,
            "progression_max": max(means) if means else 0.0,
        }
    return {"rows": rows, "per_harness": per_harness}


def table_to_markdown(table: dict) -> str:
    lines = ["### Per-cell (model x harness)", "",
             "| harness | model | n | progression_mean | progression_stdev | descent_rate | 95% CI |",
             "|---|---|---|---|---|---|---|"]
    for r in table["rows"].values():
        lines.append(
            f"| {r['harness']} | {r['model']} | {r['n']} | "
            f"{r['progression_mean']:.4f} | {r['progression_stdev']:.4f} | "
            f"{r['descent_rate']:.2f} | [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] |")
    lines += ["", "### Per-harness cross-model robustness", "",
              "| harness | #models | mean(progression) | spread(stdev) | min | max |",
              "|---|---|---|---|---|---|"]
    for h, s in table["per_harness"].items():
        lines.append(
            f"| {h} | {s['models']} | {s['progression_mean_across_models']:.4f} | "
            f"{s['progression_spread_stdev']:.4f} | {s['progression_min']:.4f} | "
            f"{s['progression_max']:.4f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration (injectable runner seam)
# ---------------------------------------------------------------------------
def _default_runner(cell: dict) -> list[dict]:
    raise NotImplementedError(
        "default runner needs model access; pass make_prime_runner(...) or a stub")


def run_sweep(matrix: list[dict], *,
              runner: Callable[[dict], list[dict]] = _default_runner) -> dict[str, Any]:
    cells: dict[str, list[dict]] = {}
    for cell in matrix:
        cells[cell_key(cell)] = runner(cell)
    return aggregate_cells(cells)


# ---------------------------------------------------------------------------
# Real runner: prime eval run + NETHACK_HARNESS overlay (dry-run capable)
# ---------------------------------------------------------------------------
def build_command(cell: dict, *, run_dir: Path, num_examples: int,
                  rollouts_per_example: int, max_tokens: int, max_turns: int,
                  tier: str, n_examples: int) -> tuple[list[str], dict, Path]:
    """Return (argv, env, trace_dir) for one cell. Pure -- no side effects."""
    model = cell["model"]
    harness = cell["harness"]
    ep = endpoint_for(model)
    provider = cell.get("provider") or ep["provider"]
    trace_dir = run_dir / cell_key(cell).replace("/", "_").replace("::", "__")
    env_args = {
        "trace_dir": str(trace_dir),
        "max_turns": max_turns,
        "tier": tier,
        "n_examples": n_examples,
    }
    argv = [
        "prime", "eval", "run", "nethack",
        "--model", model,
        "--provider", provider,
        "--num-examples", str(num_examples),
        "--rollouts-per-example", str(rollouts_per_example),
        "--max-concurrent", "1",
        "--max-tokens", str(max_tokens),
        "--env-args", json.dumps(env_args),
        "--output-dir", str(trace_dir / "eval_out"),
        "--save-results",
        "--disable-tui",
    ]
    proc_env = dict(os.environ)
    proc_env["PYTHONPATH"] = os.pathsep.join(
        [str(_ENV_DIR)] + ([proc_env["PYTHONPATH"]] if proc_env.get("PYTHONPATH") else []))
    proc_env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    # The harness-version toggle: consumed by harness_overlay.apply_overlay.
    if harness and harness not in ("default", "baseline", "none"):
        proc_env["NETHACK_HARNESS"] = harness
    else:
        proc_env.pop("NETHACK_HARNESS", None)
    return argv, proc_env, trace_dir


def _load_traces(trace_dir: Path) -> list[list[dict]]:
    out = []
    for f in sorted(trace_dir.glob("*.ndjson")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if rows:
            out.append(rows)
    return out


def _samples_from_trace_dir(trace_dir: Path) -> list[dict]:
    samples = []
    res = sorted((trace_dir / "eval_out").rglob("results.jsonl"))
    if res:
        rows = [json.loads(l) for l in res[-1].read_text().splitlines() if l.strip()]
        traces = _load_traces(trace_dir)
        for i, r in enumerate(rows):
            if i < len(traces):
                r["trace"] = traces[i]
                r["max_dlvl"] = max((int((e.get("status") or {}).get("depth", 0) or 0)
                                     for e in traces[i]), default=0)
            samples.append(r)
        return samples
    for trace in _load_traces(trace_dir):
        max_dlvl = max((int((e.get("status") or {}).get("depth", 0) or 0) for e in trace),
                       default=0)
        samples.append({"trace": trace, "max_dlvl": max_dlvl,
                        "descent_reward": 1.0 if max_dlvl >= 2 else 0.0})
    return samples


def make_prime_runner(*, run_dir: str | Path, num_examples: int = 1,
                      rollouts_per_example: int = 3, max_tokens: int = 1024,
                      max_turns: int = 200, tier: str = "corridor_explore",
                      n_examples: int = 8, timeout_s: int = 1800,
                      dry_run: bool = False) -> Callable[[dict], list[dict]]:
    """Build ``runner(cell) -> list[sample]`` that drives ``prime eval run``.

    ``dry_run=True`` prints the argv + resolved provider/key status and returns
    whatever traces already exist (usually none) -- no spend, no engine needed.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def runner(cell: dict) -> list[dict]:
        argv, proc_env, trace_dir = build_command(
            cell, run_dir=run_dir, num_examples=num_examples,
            rollouts_per_example=rollouts_per_example, max_tokens=max_tokens,
            max_turns=max_turns, tier=tier, n_examples=n_examples)
        trace_dir.mkdir(parents=True, exist_ok=True)
        ep = endpoint_for(cell["model"])
        if dry_run:
            print(f"[dry-run] {cell_key(cell)}")
            print(f"          provider={cell.get('provider') or ep['provider']} "
                  f"key_env={ep['key_env']} key_present={ep['key_present']} "
                  f"NETHACK_HARNESS={proc_env.get('NETHACK_HARNESS', '(unset)')}")
            print("          " + " ".join(argv))
            if not ep["key_present"]:
                print(f"          !! {ep['key_env']} is NOT set -- this cell would fail")
            return _samples_from_trace_dir(trace_dir)
        with (trace_dir / "run.log").open("w") as lf:
            subprocess.run(argv, env=proc_env, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=timeout_s, check=False)
        return _samples_from_trace_dir(trace_dir)

    return runner


# ---------------------------------------------------------------------------
# Demo / self-test: synthetic runner, no model calls, proves aggregation.
# ---------------------------------------------------------------------------
def _demo_runner_factory():
    """Synthetic samples: harness 'fixes' descends deeper than 'baseline',
    consistently across models -- the shape Experiment 2 wants to confirm."""
    depth_by_harness = {"baseline": (2, 2), "fixes": (7, 6)}

    def runner(cell: dict) -> list[dict]:
        base_dl, base_xp = depth_by_harness.get(cell["harness"], (2, 2))
        out = []
        for r in range(3):  # 3 rollouts / cell
            jitter = (hash((cell["model"], r)) % 3)  # tiny per-model variation
            dl = base_dl + jitter - 1
            out.append({"max_dlvl": max(1, dl), "xp_level": base_xp,
                        "descent_reward": 1.0 if dl >= 2 else 0.0})
        return out
    return runner


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+",
                   default=["gemini-2.5-flash", "z-ai/glm-5", "gpt-5.5"],
                   help="model ids to sweep (default: the Exp-2 trio)")
    p.add_argument("--harness-versions", nargs="+", default=["baseline", "fixes"],
                   help="NETHACK_HARNESS overlay names (baseline = no overlay)")
    p.add_argument("--run-dir", default="outputs/harness_sweep")
    p.add_argument("--num-examples", type=int, default=1)
    p.add_argument("--rollouts-per-example", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-turns", type=int, default=200)
    p.add_argument("--tier", default="corridor_explore")
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="print the prime-eval commands + key status; no spend")
    p.add_argument("--demo", action="store_true",
                   help="run with a synthetic runner (no model calls) and print the table")
    args = p.parse_args(argv)

    matrix = build_matrix(args.models, args.harness_versions)
    if args.demo:
        table = run_sweep(matrix, runner=_demo_runner_factory())
    else:
        runner = make_prime_runner(
            run_dir=args.run_dir, num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example, max_tokens=args.max_tokens,
            max_turns=args.max_turns, tier=args.tier, n_examples=args.n_examples,
            dry_run=args.dry_run)
        table = run_sweep(matrix, runner=runner)

    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps(table, indent=2, default=str))
    md = table_to_markdown(table)
    (out / "table.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
