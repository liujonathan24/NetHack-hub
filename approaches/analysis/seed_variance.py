"""Cross-seed variance report for the six-floor curriculum (Experiment 3).

Why this exists
---------------
Every exploration architecture we throw at the compressed six-floor curriculum
(Go-Explore, Voyager, the continual-harness loop) ends up trying to build the
same thing: a *navigation primitive* that walks the hero to the next staircase
and descends/climbs. That primitive **inevitably gets stuck** — it plateaus a
couple of floors in, and *where* it plateaus swings wildly from seed to seed
because NetHack's level layout is procedurally generated. The consequence is
**extremely high cross-seed variance** and therefore **low coverage of the
gameplay-trace space**: no single run comes close to touching the whole tour.

This module quantifies exactly that. It runs (or aggregates) the six-floor
curriculum across N seeds, then reports per-seed depth + coverage rolled up into
mean / std / max / min, the coefficient of variation (the headline "variance"
number), and the "nav component gets stuck" signal (the iteration at which each
seed's depth stops improving, and the modal stall floor).

Backends (``--source``)
-----------------------
- ``mock``     : synthesize deterministic per-seed results with NO engine and
                 NO API. This is the ``--dry-run`` default; it exists so the whole
                 aggregation + reporting pipeline can be smoke-tested for free and
                 reproducibly. The synthetic numbers deliberately reproduce the
                 empirically observed shape (most seeds stall at floor 1-2, one
                 occasionally reaches 3, coverage swings by ~3x).
- ``aggregate``: read existing per-seed result JSONs (as written by
                 ``approaches/go_explore/curriculum_go_explore.py`` or
                 ``approaches/voyager/*``) from ``--results-dir`` and compute the
                 variance report over whatever is on disk. NO engine, NO API.
- ``go-explore``: run the KEYLESS curriculum Go-Explore driver
                 (``run_curriculum_go_explore``) live, once per seed. Uses the
                 real NetHack engine on CPU but makes NO API call. Import of the
                 engine is deferred to this backend so the mock/aggregate paths
                 stay dependency-free.

All three feed the same normaliser + aggregator, so the report shape is
identical regardless of source.

Run
---
    # free smoke test (no engine, no API):
    python -m approaches.analysis.seed_variance --dry-run --seeds 19 2 9 7 42

    # aggregate whatever real runs already exist on disk:
    python -m approaches.analysis.seed_variance --source aggregate \
        --results-dir outputs/curriculum_experiments/go_explore

    # live keyless six-floor Go-Explore sweep (engine on CPU, no API):
    python -m approaches.analysis.seed_variance --source go-explore \
        --seeds 19 2 9 --iterations 400 --explore-steps 40 \
        --out outputs/curriculum_experiments/seed_variance
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

# The six-floor curriculum is a 6-down / 6-up tour; max tour progress is 11
# (descend 1..6 = 6, then climb 6..1 = +5). See docs/CURRICULUM.md.
MAX_FLOOR = 6
MAX_PROGRESS = 11


# --------------------------------------------------------------------------- #
# normalised per-seed result
# --------------------------------------------------------------------------- #
@dataclass
class SeedResult:
    """One curriculum run, normalised across backends.

    coverage proxies
    ----------------
    - ``n_cells``       : distinct Go-Explore archive cells reached — a direct
                          proxy for how much of the (floor, position, progress)
                          state space the run touched, i.e. gameplay-trace
                          coverage. 0/absent for backends that do not archive.
    - ``tour_coverage`` : fraction of the 11-step tour reached =
                          progress / MAX_PROGRESS, where progress descends 1..6
                          then adds floors climbed back. A layout-agnostic
                          "how far along the curriculum did this seed get" score.
    - ``stall_iter``    : iteration index at which ``deepest_floor`` last
                          improved (None if unknown). The gap between this and
                          the total iteration count is how long the nav component
                          sat stuck — the "gets stuck" signal, per seed.
    """

    seed: int
    deepest_floor: int
    climbed_back: int = 0
    reached_bottom: bool = False
    n_cells: int = 0
    iterations: int = 0
    stall_iter: Optional[int] = None
    source: str = ""

    @property
    def progress(self) -> int:
        if self.deepest_floor <= 0:
            return 0
        if self.reached_bottom:
            return MAX_FLOOR + self.climbed_back
        return self.deepest_floor

    @property
    def tour_coverage(self) -> float:
        return round(self.progress / MAX_PROGRESS, 4)


def _stall_iter_from_timeseries(timeseries: list) -> Optional[int]:
    """Iteration at which ``deepest_floor`` last increased. Everything after that
    is the nav component stuck at its ceiling."""
    if not timeseries:
        return None
    last_improve = None
    best = None
    for row in timeseries:
        d = row.get("deepest_floor")
        it = row.get("iter", row.get("turn"))
        if d is None or it is None:
            continue
        if best is None or d > best:
            best = d
            last_improve = it
    return last_improve


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def mock_backend(seed: int, *, iterations: int, explore_steps: int) -> SeedResult:
    """Deterministic synthetic result — NO engine, NO API.

    Reproduces the empirically observed shape: the navigation primitive stalls
    shallow (floor 1-2, occasionally 3), coverage (n_cells) swings by roughly 3x
    across seeds, and the plateau begins early. The numbers are a stable function
    of ``seed`` so the report is byte-reproducible across runs.
    """
    h = (seed * 2654435761) & 0xFFFFFFFF          # cheap deterministic hash
    # deepest floor: mostly 1-2, ~1-in-6 seeds sneak to 3. This is the "nav gets
    # stuck shallow" ceiling, and it varies seed-to-seed (procedural layout).
    deepest = 1 + (h % 6 == 0) + (h % 2)          # 1, 2, or 3
    deepest = min(deepest, 3)
    reached_bottom = False
    climbed_back = 0
    # coverage: distinct archived cells. High-variance across seeds (30..120ish).
    n_cells = 30 + (h % 90)
    # the nav component plateaus early: stall between ~8% and ~35% of the run.
    stall_iter = max(1, int(iterations * (0.08 + (h % 27) / 100.0)))
    # a compact synthetic timeseries so downstream stall logic exercises real code.
    timeseries = []
    cur = 1
    for it in range(1, iterations + 1):
        if it <= stall_iter and cur < deepest and it % max(1, stall_iter // deepest) == 0:
            cur += 1
        cur = min(cur, deepest)
        cells = min(n_cells, int(n_cells * min(1.0, it / max(1, stall_iter * 1.5))))
        timeseries.append({"iter": it, "deepest_floor": cur, "cells": cells})
    return SeedResult(
        seed=seed, deepest_floor=deepest, climbed_back=climbed_back,
        reached_bottom=reached_bottom, n_cells=n_cells, iterations=iterations,
        stall_iter=_stall_iter_from_timeseries(timeseries), source="mock",
    )


def go_explore_backend(seed: int, *, iterations: int, explore_steps: int) -> SeedResult:
    """Run the KEYLESS six-floor Go-Explore curriculum live (engine on CPU, no API).

    Imports the engine + driver lazily so ``mock``/``aggregate`` never pay for a
    libnethack load. Requires ``NLE_LIB_PATH`` set and the engine + env dirs on
    ``PYTHONPATH`` (see module docstring / docs/experiments/exp3_*.md).
    """
    _ensure_engine_on_path()
    from approaches.go_explore.curriculum_go_explore import (  # noqa: E402
        run_curriculum_go_explore,
    )

    res = run_curriculum_go_explore(
        iterations=iterations, explore_steps=explore_steps, seed=seed, verbose=False,
    )
    return SeedResult(
        seed=seed,
        deepest_floor=res.deepest_floor,
        climbed_back=res.climbed_back,
        reached_bottom=res.reached_bottom,
        n_cells=res.n_cells,
        iterations=res.iterations,
        stall_iter=_stall_iter_from_timeseries(res.timeseries),
        source="go-explore",
    )


def aggregate_backend(results_dir: pathlib.Path) -> list[SeedResult]:
    """Load per-seed result JSONs already on disk. NO engine, NO API.

    Accepts either the per-seed files (``*_seed<N>.json`` with ``deepest_floor``
    etc., optionally a ``timeseries``) or a ``*_summary.json`` list of per-seed
    dicts. Silently skips files it cannot parse into a seed record.
    """
    out: list[SeedResult] = []
    seen: set[int] = set()
    files = sorted(results_dir.glob("*.json"))
    # Prefer detailed per-seed files (they carry a timeseries -> stall_iter);
    # fall back to summary files for any seed not yet covered.
    per_seed = [f for f in files if "summary" not in f.name]
    summaries = [f for f in files if "summary" in f.name]
    for f in per_seed + summaries:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        records = data if isinstance(data, list) else [data]
        for rec in records:
            if not isinstance(rec, dict) or "seed" not in rec:
                continue
            seed = int(rec["seed"])
            if seed in seen or "deepest_floor" not in rec:
                continue
            seen.add(seed)
            out.append(SeedResult(
                seed=seed,
                deepest_floor=int(rec.get("deepest_floor", 0)),
                climbed_back=int(rec.get("climbed_back", 0)),
                reached_bottom=bool(rec.get("reached_bottom", False)),
                n_cells=int(rec.get("n_cells", 0) or 0),
                iterations=int(rec.get("iterations", 0) or 0),
                stall_iter=_stall_iter_from_timeseries(rec.get("timeseries", [])),
                source=f"aggregate:{f.name}",
            ))
    return sorted(out, key=lambda r: r.seed)


def _ensure_engine_on_path() -> None:
    """Put engine + env dirs on sys.path if an env var points at the engine root.

    Honours ``NETHACK_ENGINE_ROOT`` (or infers from ``NLE_LIB_PATH``). A no-op if
    the engine is already importable, so this never fights an existing PYTHONPATH.
    """
    try:
        import nethack_core.curriculum_engine_env  # noqa: F401
        return
    except Exception:
        pass
    root = None
    import os
    if os.environ.get("NETHACK_ENGINE_ROOT"):
        root = pathlib.Path(os.environ["NETHACK_ENGINE_ROOT"])
    elif os.environ.get("NLE_LIB_PATH"):
        # .../third_party/NetHack/src/build/libnethack.so -> engine root
        p = pathlib.Path(os.environ["NLE_LIB_PATH"]).resolve()
        for parent in p.parents:
            if (parent / "nethack_core").is_dir():
                root = parent
                break
    if root and (root / "nethack_core").is_dir():
        sys.path.insert(0, str(root))
    # env dir (for the go_explore driver's relative imports).
    hub_root = pathlib.Path(__file__).resolve().parents[2]
    env_dir = hub_root / "environments" / "nethack"
    if env_dir.is_dir():
        sys.path.insert(0, str(env_dir))
    sys.path.insert(0, str(hub_root))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
@dataclass
class VarianceReport:
    n_seeds: int
    seeds: list
    depth_mean: float
    depth_std: float
    depth_max: int
    depth_min: int
    depth_range: int
    depth_cv: float                 # coefficient of variation = std / mean
    coverage_cells_mean: float
    coverage_cells_std: float
    coverage_cells_max: int
    coverage_cells_min: int
    tour_coverage_mean: float
    tour_coverage_max: float
    frac_reached_bottom: float
    modal_stall_floor: int          # the floor the nav component most often stalls at
    mean_stall_fraction: float      # mean(stall_iter / iterations): how early it stalls
    stuck_signal: str               # human-readable diagnosis
    per_seed: list = field(default_factory=list)


def _std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def aggregate(results: list[SeedResult]) -> VarianceReport:
    if not results:
        raise ValueError("no seed results to aggregate")
    depths = [r.deepest_floor for r in results]
    cells = [r.n_cells for r in results]
    tour = [r.tour_coverage for r in results]
    depth_mean = statistics.fmean(depths)
    depth_std = _std([float(d) for d in depths])
    cv = (depth_std / depth_mean) if depth_mean else 0.0

    # modal stall floor: the floor at which the nav primitive most often plateaus.
    modal_floor = statistics.mode(depths) if depths else 0
    # how early (as a fraction of the run) the plateau sets in, averaged over seeds
    # that reported a timeseries.
    fracs = [r.stall_iter / r.iterations
             for r in results if r.stall_iter and r.iterations]
    mean_stall_fraction = statistics.fmean(fracs) if fracs else 0.0

    frac_bottom = sum(1 for r in results if r.reached_bottom) / len(results)

    signal = (
        f"nav component stalls at floor {modal_floor}/{MAX_FLOOR} for most seeds; "
        f"depth swings {min(depths)}->{max(depths)} across {len(results)} seeds "
        f"(std={depth_std:.2f}, CV={cv:.2f}); "
        f"{frac_bottom * 100:.0f}% of seeds reach the bottom (floor {MAX_FLOOR}); "
        f"plateau sets in at ~{mean_stall_fraction * 100:.0f}% of the run. "
        "High CV + low reached-bottom + early plateau == the nav primitive gets "
        "stuck and coverage of the gameplay-trace space stays low."
    )

    return VarianceReport(
        n_seeds=len(results),
        seeds=[r.seed for r in results],
        depth_mean=round(depth_mean, 3),
        depth_std=round(depth_std, 3),
        depth_max=max(depths),
        depth_min=min(depths),
        depth_range=max(depths) - min(depths),
        depth_cv=round(cv, 3),
        coverage_cells_mean=round(statistics.fmean(cells), 2),
        coverage_cells_std=round(_std([float(c) for c in cells]), 2),
        coverage_cells_max=max(cells),
        coverage_cells_min=min(cells),
        tour_coverage_mean=round(statistics.fmean(tour), 4),
        tour_coverage_max=max(tour),
        frac_reached_bottom=round(frac_bottom, 3),
        modal_stall_floor=modal_floor,
        mean_stall_fraction=round(mean_stall_fraction, 3),
        stuck_signal=signal,
        per_seed=[asdict(r) | {"tour_coverage": r.tour_coverage,
                               "progress": r.progress} for r in results],
    )


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_table(results: list[SeedResult], report: VarianceReport) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append("SIX-FLOOR CURRICULUM — CROSS-SEED VARIANCE (Experiment 3)")
    lines.append("=" * 74)
    lines.append(f"{'seed':>6} {'deepest':>8} {'climbed':>8} {'bottom':>7} "
                 f"{'cells':>7} {'tour_cov':>9} {'stall@it':>9}")
    lines.append("-" * 74)
    for r in sorted(results, key=lambda x: x.seed):
        lines.append(
            f"{r.seed:>6} {r.deepest_floor:>8} {r.climbed_back:>8} "
            f"{('yes' if r.reached_bottom else 'no'):>7} {r.n_cells:>7} "
            f"{r.tour_coverage:>9.3f} {str(r.stall_iter):>9}")
    lines.append("-" * 74)
    lines.append(
        f"depth   mean={report.depth_mean:.2f}  std={report.depth_std:.2f}  "
        f"max={report.depth_max}  min={report.depth_min}  "
        f"range={report.depth_range}  CV={report.depth_cv:.2f}")
    lines.append(
        f"cells   mean={report.coverage_cells_mean:.1f}  "
        f"std={report.coverage_cells_std:.1f}  "
        f"max={report.coverage_cells_max}  min={report.coverage_cells_min}")
    lines.append(
        f"tour    mean_cov={report.tour_coverage_mean:.3f}  "
        f"max_cov={report.tour_coverage_max:.3f}  "
        f"reached_bottom={report.frac_reached_bottom * 100:.0f}%")
    lines.append("=" * 74)
    lines.append("NAV-STUCK SIGNAL:")
    for chunk in _wrap(report.stuck_signal, 72):
        lines.append("  " + chunk)
    lines.append("=" * 74)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_sweep(
    seeds: list[int],
    backend: Callable[[int], SeedResult],
) -> list[SeedResult]:
    results = []
    for s in seeds:
        results.append(backend(s))
    return results


def _make_backend(args) -> Callable[[int], SeedResult]:
    if args.source == "mock":
        return lambda s: mock_backend(
            s, iterations=args.iterations, explore_steps=args.explore_steps)
    if args.source == "go-explore":
        return lambda s: go_explore_backend(
            s, iterations=args.iterations, explore_steps=args.explore_steps)
    raise ValueError(f"backend {args.source!r} is not a per-seed callable")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m approaches.analysis.seed_variance",
        description="Cross-seed variance report over the six-floor curriculum, "
                    "quantifying the 'nav component gets stuck -> high variance "
                    "-> low coverage' failure mode (Experiment 3).",
    )
    p.add_argument("--source", choices=("mock", "aggregate", "go-explore"),
                   default="mock",
                   help="mock: synthetic, no engine/API (default; forced by "
                        "--dry-run). aggregate: read existing result JSONs. "
                        "go-explore: run keyless six-floor Go-Explore live "
                        "(engine on CPU, no API).")
    p.add_argument("--dry-run", action="store_true",
                   help="force the mock backend (NO engine, NO API). Smoke test.")
    p.add_argument("--seeds", type=int, nargs="+", default=[19, 2, 9, 7, 42],
                   help="seeds to sweep (mock / go-explore backends).")
    p.add_argument("--iterations", type=int, default=400,
                   help="Go-Explore iterations per seed (go-explore backend).")
    p.add_argument("--explore-steps", dest="explore_steps", type=int, default=40,
                   help="explore steps per return (go-explore backend).")
    p.add_argument("--results-dir", type=pathlib.Path, default=None,
                   help="directory of existing result JSONs (aggregate backend).")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="write the variance report JSON here.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        args.source = "mock"

    if args.source == "aggregate":
        if not args.results_dir or not args.results_dir.is_dir():
            print("ERROR: --source aggregate requires --results-dir <dir>",
                  file=sys.stderr)
            return 2
        results = aggregate_backend(args.results_dir)
        if not results:
            print(f"ERROR: no parseable seed results under {args.results_dir}",
                  file=sys.stderr)
            return 1
    else:
        results = run_sweep(args.seeds, _make_backend(args))

    report = aggregate(results)
    print(render_table(results, report))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"report": asdict(report),
                   "results": [asdict(r) for r in results]}
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nvariance report written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
