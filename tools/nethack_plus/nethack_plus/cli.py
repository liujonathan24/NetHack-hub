"""nethack-plus: one command per NetHack experiment on Prime.

    nethack-plus single  -n 5 -r 15 --skills netcore,codehack --full-vision
    nethack-plus explore -n 5 -r 1 -a 15 -p 3 --skills netcore,codehack --full-vision
    nethack-plus status  runs/<name>
    nethack-plus summary runs/<name>
    nethack-plus resume  runs/<name>

A run subcommand writes one vf-eval config per skill library and runs vf-eval on
it (the libraries side by side). Each library's run lands in
`<out>/<name>/<library>/`: its config, the vf-eval run (`vf/`), the tool
servers' per-call progress (`progress/`), explore checkpoints (`archive/`) and
the log. Inference and sandboxes are Prime's, so it needs `prime login` or
`$PRIME_API_KEY`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from . import prime, report, served

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
#: Tokens per game at 500 calls (uncached input, cached input, output), measured on
#: deepseek-v4.1-flash-1017 games with the map revealed; the estimate scales them by
#: the model's prices, so it is a guide for other models and settings.
PROFILES = {
    "netcore": (42_306, 3_095_603, 23_014),
    "codehack": (154_464, 28_313_088, 99_145),
}
#: Prime sandboxes a game holds at once (the agent's box and the game's box).
BOXES_PER_GAME = 2


def tune_pairs(values: list[str], kind=float) -> dict:
    pairs = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected knob=value, got {item!r}")
        pairs[key.strip()] = kind(value)
    return pairs


def run_config(args, library: str, arm_dir: Path, concurrency: int) -> dict:
    """The vf-eval config for one skill library."""
    tune = {"reveal_map": 1.0 if args.full_vision else 0.0}
    if args.damage is not None:
        tune["dmg_to_player_scale"] = args.damage
    if args.xp is not None:
        tune["xp_gain_scale"] = args.xp
    tune.update(tune_pairs(args.tune))
    taskset = {
        "id": "nethack",
        "num_seeds": args.seeds,
        "seed_base": args.seed_base,
        "max_calls": args.max_calls,
        "agent_timeout": args.timeout,
        "character": None if args.character == "random" else args.character,
        "tune": tune,
        "modify": tune_pairs(args.modify, int),
        "skill_library": library,
        "editable_skills": False,
        "mode": args.mode,
    }
    if args.mode == "explore":
        taskset.update(
            attempts=args.attempts,
            parallel=args.parallel,
            checkpoint_every=args.every,
            chooser=args.chooser,
            archive_dir=str(arm_dir / "archive"),
        )
    # An explore run plays a round's attempts side by side: each is one agent of the run's episode.
    agents = args.parallel if args.mode == "explore" else 1
    return {
        "model": args.model,
        "num_rollouts": args.reps,
        "max_concurrent": concurrency,
        "push": args.push,
        "output_dir": str(arm_dir / "vf"),
        "run": {"name": f"{args.name}-{library}", "dir": "run"},
        "sampling": {"max_tokens": args.max_tokens},
        "env": {
            "taskset": taskset,
            "agent": {"network_notice": False, "harness": {"id": args.harness}, "runtime": {"type": "prime"}},
            "interception": {"multiplex": min(concurrency * agents, 16)},
            "max_concurrent_agents": agents,
        },
    }


def estimate(args, libraries: list[str]) -> dict:
    price = prime.prices(args.model)
    games = args.seeds * args.reps * (args.attempts if args.mode == "explore" else 1)
    scale = args.max_calls / 500
    per_game = {
        lib: sum(n * scale * rate for n, rate in zip(PROFILES[lib], (price["input"], price["cached"], price["output"])))
        / 1e6
        for lib in libraries
    }
    return {
        "games_per_library": games,
        "dollars_per_game": per_game,
        "dollars": sum(games * cost for cost in per_game.values()),
        "upper_bound": args.mode == "explore",
    }


def print_estimate(args, libraries: list[str], est: dict, balance: float) -> None:
    every = "at most " if est["upper_bound"] else ""
    print(
        f"{args.mode}: {args.seeds} seeds x {args.reps} reps"
        + (f" x {args.attempts} attempts" if args.mode == "explore" else "")
        + f" = {every}{est['games_per_library']} games per library, libraries {', '.join(libraries)}, model {args.model}"
    )
    for lib, cost in est["dollars_per_game"].items():
        print(f"  {lib}: about ${cost:.2f} per game")
    print(f"estimate: {every}${est['dollars']:.2f} in inference (from a token profile; the wallet is the billed truth)")
    print(f"wallet balance: ${balance:.2f}")


def launch(args) -> int:
    libraries = [lib.strip() for lib in args.skills.split(",") if lib.strip()]
    if args.mode == "single" and args.attempts_given:
        raise SystemExit("-a/--attempts applies to explore only")
    key = prime.api_key()
    balance = prime.balance()
    est = estimate(args, libraries)
    print_estimate(args, libraries, est, balance)
    if args.estimate:
        return 0
    args.name = args.name or f"{args.mode}-{args.model.split('/')[-1]}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    run = Path(args.out) / args.name
    if run.exists():
        raise SystemExit(f"{run} exists: pick another --name, or `nethack-plus resume {run}`")
    concurrency = max(1, args.concurrency // len(libraries))
    configs = {}
    for library in libraries:
        arm = run / library
        (arm / "progress").mkdir(parents=True)
        configs[library] = run_config(args, library, arm.resolve(), concurrency)
        (arm / "config.json").write_text(json.dumps(configs[library], indent=2))
    manifest = {
        "argv": sys.argv,
        "mode": args.mode,
        "model": args.model,
        "seeds": args.seeds,
        "seed_base": args.seed_base,
        "reps": args.reps,
        "attempts": args.attempts if args.mode == "explore" else 1,
        "parallel": args.parallel if args.mode == "explore" else 1,
        "libraries": libraries,
        "estimate": est,
        "balance_before": balance,
        "started": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"run directory: {run}")
    agents = args.parallel if args.mode == "explore" else 1
    peak = BOXES_PER_GAME * agents * min(concurrency, args.seeds * args.reps) * len(libraries)
    print(f"sandboxes at peak: about {peak} (Prime has capped us near 56)")
    if args.dry_run:
        print("dry run: configs written, vf-eval not started")
        return 0
    return supervise(run, libraries, key, ["@", "config.json"])


def supervise(run: Path, libraries: list[str], key: str, vf_args: list[str]) -> int:
    """Run vf-eval for each library side by side, each logging to `<library>/vf-eval.log`."""
    vf_eval = Path(sys.executable).with_name("vf-eval")
    processes = {}
    for library in libraries:
        arm = (run / library).resolve()
        env = {
            **os.environ,
            "PRIME_API_KEY": key,
            "PRIME_DISABLE_VERSION_CHECK": "1",
            "NETHACK_PROGRESS_DIR": str(arm / "progress"),
        }
        log = open(arm / "vf-eval.log", "a")
        processes[library] = subprocess.Popen(
            [str(vf_eval), *vf_args, "--no-rich"], cwd=arm, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        print(f"{library}: vf-eval pid {processes[library].pid}, log {arm / 'vf-eval.log'}")
    print(f"watch with: nethack-plus status {run}")
    codes = {library: process.wait() for library, process in processes.items()}
    for library, code in codes.items():
        print(f"{library}: vf-eval exited {code}")
    print(report.summary(run))
    return max(codes.values())


def resume(args) -> int:
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    libraries = [lib for lib in manifest["libraries"] if (run / lib / "vf" / "run" / "configs").exists()]
    if not libraries:
        raise SystemExit(f"{run}: no vf-eval run to resume")
    # vf-eval re-reads the run's own resolved config, so the run continues exactly as it started.
    return supervise(run, libraries, prime.api_key(), ["@", "vf/run/configs/resolved/eval.json", "--resume", "True"])


def run_args(parser: argparse.ArgumentParser, mode: str) -> None:
    parser.set_defaults(mode=mode)
    parser.add_argument("-n", "--seeds", type=int, default=5, help="seeds, from --seed-base (default 5)")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("-r", "--reps", type=int, default=1, help="independent runs per seed (default 1)")
    if mode == "explore":
        parser.add_argument("-a", "--attempts", type=int, default=10, help="most attempts per run (default 10)")
        parser.add_argument(
            "-p", "--parallel", type=int, default=3, help="attempts of a run played at once, in rounds (default 3)"
        )
        parser.add_argument("--every", type=int, default=20, help="checkpoint every N calls (default 20)")
        parser.add_argument(
            "--chooser",
            choices=("model", "rule", "blind"),
            default="model",
            help="who picks the next checkpoint: the player's model (default), or a control arm",
        )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Prime inference model (default {DEFAULT_MODEL})")
    parser.add_argument(
        "--skills", default="netcore", help="skill library, or a comma list run side by side: netcore,codehack"
    )
    parser.add_argument("--full-vision", action="store_true", help="reveal the whole map (off: fog of war)")
    parser.add_argument("--max-calls", type=int, default=500, help="skill calls per game (explore: per attempt)")
    parser.add_argument("--timeout", type=float, default=14400.0, help="seconds per game (explore: per attempt)")
    parser.add_argument("-c", "--concurrency", type=int, default=20, help="runs at once, split across libraries")
    parser.add_argument("--max-tokens", type=int, default=32768, help="output cap per model turn")
    parser.add_argument("--character", default="Val-hum-neu-fem", help="role-race-alignment-gender, or random")
    parser.add_argument("--damage", type=float, help="monsters deal X times their normal damage")
    parser.add_argument("--xp", type=float, help="experience at X times the normal rate")
    parser.add_argument("--tune", action="append", default=[], metavar="KNOB=X", help="any engine knob (repeatable)")
    parser.add_argument(
        "--modify",
        action="append",
        default=[],
        metavar="FIELD=N",
        help="starting state: hp, max_hp, gold, xp_level, hunger",
    )
    parser.add_argument("--harness", default="nethack")
    parser.add_argument("-o", "--out", default="runs", help="directory for run directories (default runs/)")
    parser.add_argument("--name", help="run name (default <mode>-<model>-<time>)")
    parser.add_argument("--push", action="store_true", help="upload results to Prime Evals (off by default)")
    parser.add_argument("--estimate", action="store_true", help="print the estimate and the balance, then exit")
    parser.add_argument("--dry-run", action="store_true", help="write the configs, do not start vf-eval")


def main(argv: list[str] | None = None) -> int:
    # A run is long and is often started with its output in a file; print each line as it comes.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="nethack-plus", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("single", "explore"):
        run_args(sub.add_parser(mode, help=f"run {mode} mode"), mode)
    for name, help_text in (
        ("status", "live games of a run"),
        ("summary", "results of a run"),
        ("resume", "re-run a run's missing or failed runs"),
        ("served", "what the model was served, per library, and how the libraries differ"),
    ):
        sub.add_parser(name, help=help_text).add_argument("run", help="the run directory")
    args = parser.parse_args(argv)
    if args.command in ("single", "explore"):
        args.attempts_given = "-a" in (argv or sys.argv) or "--attempts" in (argv or sys.argv)
        args.attempts = getattr(args, "attempts", 1)
        args.every = getattr(args, "every", 20)
        args.parallel = getattr(args, "parallel", 1)
        args.chooser = getattr(args, "chooser", "model")
        unknown = set(args.skills.split(",")) - set(PROFILES)
        if unknown:
            raise SystemExit(f"unknown skill library {sorted(unknown)}; choose from {sorted(PROFILES)}")
        return launch(args)
    if args.command == "status":
        print(report.status(Path(args.run)))
    elif args.command == "summary":
        print(report.summary(Path(args.run)))
    elif args.command == "served":
        print(served.compare(Path(args.run)))
    elif args.command == "resume":
        return resume(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
