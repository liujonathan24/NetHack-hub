"""The experiments tab — one uniform entrypoint for every experiment.

    python -m experiments.run <name> [--smoke | --real] [-- ...passthrough]

Each experiment is DEFINED here (name -> how to run) but the PLUMBING lives in
shared infra (experiments.common) + the engine + the per-experiment runner
modules. Every experiment runs POST-MONOLITH: this box orchestrates the env, the
model is called remotely from Prime Inference (see experiments.common).

Modes:
  --smoke  (default)  free / keyless / dry — proves the wiring, ~$0.
  --real              actually calls the model (costs budget); cheap defaults.

Experiments (see docs/experiments/*.md for the full plans):
  encoding    Exp 1  encoding ablations (ASCII/JSON/TOON/IMG vs baselines)
  harness     Exp 2  harness modifications across frontier models
  continual   Exp 3  continual-harness optimization loop
  explore     Exp 3  go-explore / branch exploration (keyless)
  variance    Exp 3  cross-seed variance report on the 6-floor curriculum
  ablations   —      level-modification ablations (vision/health/doors/luck)
"""
from __future__ import annotations

import argparse
import sys

from experiments import common


def _smoke(args) -> int:
    return 0 if args.mode == "smoke" else 1


def run_continual(args) -> int:
    """Exp 3 — continual-harness loop. Runs post-monolith (engine is external;
    the loop's immutable-game guard is satisfied trivially since there is no
    engine dir in this repo). Smoke = --dry-run (no API)."""
    from approaches.continuous_harness import loop
    common.apply_post_monolith_env()
    argv = [
        "--iterations", "1",
        "--policy", common.CHEAP_MODEL, "--teacher", common.CHEAP_TEACHER,
        "--tier", "corridor_explore", "--n-seeds", "1", "--max-turns", "2",
        "--proposer", "fallback", "--verbose",
        "--out", args.out or "/tmp/exp_continual",
        "--env-sh", args.env_sh or "/tmp/ch_env.sh",
    ]
    if args.mode == "smoke":
        argv.append("--dry-run")
    argv += args.passthrough
    return int(bool(loop.run_loop(loop.build_parser().parse_args(argv)).get("error")))


def run_explore(args) -> int:
    """Exp 3 — go-explore (KEYLESS: no model, exercises snapshot/branch). Always
    free, so --smoke and --real both do a real (short) run."""
    from approaches.go_explore.go_explore import run_go_explore
    common.apply_post_monolith_env()
    iters = 3 if args.mode == "smoke" else 50
    result, env = run_go_explore(iterations=iters, explore_steps=4, seed=2, verbose=True)
    try:
        print(f"[explore] cells={result.n_cells} max_depth={result.max_depth}")
    finally:
        env.close()
    return 0


def run_encoding(args) -> int:
    """Exp 1 — encoding ablations via the prime runner. Smoke = dry aggregate
    (no paid eval); real = one cheap B1 cell."""
    from tools.encoding_eval.prime_runner import make_runner
    from tools.encoding_eval.run import run_matrix
    common.apply_post_monolith_env()
    run_dir = args.out or "/tmp/exp_encoding"
    matrix = {"encodings": [{"variant": "B1", "map_detail": None}], "models": [common.CHEAP_MODEL]}
    runner = make_runner(run_dir=run_dir, num_examples=1, max_turns=2, max_tokens=2048,
                         tier="corridor_explore", n_examples=1, dry_run=(args.mode == "smoke"))
    table = run_matrix(matrix, runner=runner)
    print(table)
    return 0


def _delegated(module: str, pr: str):
    def _run(args) -> int:
        try:
            mod = __import__(module, fromlist=["main"])
        except ModuleNotFoundError:
            print(f"[{args.name}] runner '{module}' not on this branch yet — lands with {pr}.",
                  file=sys.stderr)
            return 2
        common.apply_post_monolith_env()
        return int(mod.main(args.passthrough) or 0) if hasattr(mod, "main") else 0
    return _run


EXPERIMENTS = {
    "encoding": run_encoding,
    "continual": run_continual,
    "explore": run_explore,
    "variance": _delegated("approaches.analysis.seed_variance", "PR #6"),
    "ablations": _delegated("tools.ablation_sweep", "PR #5"),
    "harness": _delegated("tools.harness_sweep", "PR #7"),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.run", description="The experiments tab.")
    p.add_argument("name", choices=sorted(EXPERIMENTS), help="which experiment")
    m = p.add_mutually_exclusive_group()
    m.add_argument("--smoke", dest="mode", action="store_const", const="smoke", default="smoke")
    m.add_argument("--real", dest="mode", action="store_const", const="real")
    p.add_argument("--out", default=None)
    p.add_argument("--env-sh", default=None)
    p.add_argument("passthrough", nargs="*", help="args forwarded to the runner")
    args = p.parse_args(argv)
    print(f"[experiments] {args.name} (mode={args.mode}, post-monolith)")
    return EXPERIMENTS[args.name](args)


if __name__ == "__main__":
    raise SystemExit(main())
