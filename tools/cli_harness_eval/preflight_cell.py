#!/usr/bin/env python3
"""Read back what a cell ACTUALLY resolved to, and refuse the batch if it lies.

Mandatory before any paid cell: resolve the config, mock-play one seed, then run
this over the mock's output directory. It exists because every expensive mistake
in this series has been a config that claimed one thing and ran another --
`tune.reveal_map` and `auto_dismiss` silently dropped so "control" cells ran
fog'd, a stale harness package rejecting a flag, a trace_dir resolved relative to
a workdir that teardown deletes. None of those were visible in the launch
command; all of them were visible in the resolved config.toml and the first turn
record.

    preflight_cell.py <run_dir> --stack base+human [--tier-hash <hash>]

Exit 0 = the cell ran what it said. Non-zero = do not spend the batch.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

# The np_core surface, exactly. 10 tools. A cell that publishes 9 or 11 is not
# the baseline, whatever its config says.
NP_CORE = {
    "np_move_to", "np_explore_level", "np_melee_attack", "np_pickup",
    "np_descend", "np_rest", "np_apply", "np_press_key",
    "request_map", "search",
}


def fail(msg: str, problems: list[str]) -> None:
    problems.append(msg)


def check_config(run_dir: Path, stack: str, tier_hash: str | None, problems: list[str]) -> dict:
    cfg_path = run_dir / "config.toml"
    if not cfg_path.exists():
        fail(f"no config.toml in {run_dir} -- the cell never resolved", problems)
        return {}
    cfg = tomllib.load(open(cfg_path, "rb"))
    ts = cfg.get("taskset", {})
    ea = ts.get("env_args", {})

    # The baseline contract, by name. These are the two keys whose absence
    # produced fog'd "control" cells.
    if ea.get("auto_dismiss") not in ("false", False):
        fail(f"auto_dismiss is {ea.get('auto_dismiss')!r}, expected \"false\"", problems)
    # Accept both 1.0 and "1.0": launch_cell.sh's ENV_ARGS flattener delivers
    # every leaf through the CLI as text, so the resolved config records the
    # string even though the loader coerces it. Measured on
    # outputs/e10_baseline/, which demonstrably ran with full vision.
    reveal = (ea.get("tune") or {}).get("reveal_map")
    try:
        reveal_ok = reveal is not None and float(reveal) == 1.0
    except (TypeError, ValueError):
        reveal_ok = False
    if not reveal_ok:
        fail(f"tune.reveal_map is {reveal!r}, expected 1.0 (cell would run fog'd)", problems)
    if ea.get("skill_set") != "np_core,request_map,search":
        fail(f"skill_set is {ea.get('skill_set')!r}, not the np_core surface", problems)
    if ts.get("variant") not in (None, "BBOX_MIN"):
        fail(f"variant is {ts.get('variant')!r}, expected BBOX_MIN", problems)

    # The provenance pin must have survived into the artifact.
    if ea.get("tool_tier") != stack:
        fail(f"tool_tier pin is {ea.get('tier_stack')!r}, expected {stack!r}", problems)
    if tier_hash and ea.get("tool_tier_hash") != tier_hash:
        fail(
            f"tool_tier_hash pin is {ea.get('tier_hash')!r}, expected {tier_hash!r} -- "
            "the tier file changed between resolve and launch", problems,
        )
    commit = str(ea.get("tool_tier_commit", ""))
    if commit.endswith("-dirty"):
        fail(
            f"tool_tier_commit is {commit!r}: the tree has uncommitted changes, so this "
            "run is not replayable from any commit", problems,
        )
    return cfg


def check_mock_play(run_dir: Path, problems: list[str]) -> None:
    turns = sorted((run_dir / "turns").glob("*.ndjson")) if (run_dir / "turns").is_dir() else []
    if not turns:
        fail("no turns/*.ndjson -- the mock play produced no record", problems)
        return
    first = None
    for line in open(turns[0]):
        first = json.loads(line)
        break
    if first is None:
        fail(f"{turns[0].name} is empty -- the mock never took a turn", problems)
        return

    # Full vision: with reveal_map=1.0 the first observation shows the whole
    # level, not a torch-lit blob. Measured as "the map has more than one room's
    # worth of revealed terrain".
    grid = first.get("raw_grid") or ""
    revealed = sum(1 for ch in str(grid) if ch not in " \n\x00")
    if revealed and revealed < 200:
        fail(
            f"first observation shows only {revealed} revealed glyphs -- looks "
            "fog'd, so reveal_map did not take effect despite the config",
            problems,
        )

    # The published surface, as the model actually sees it.
    tools = first.get("tools") or first.get("allowed_skills")
    if isinstance(tools, list) and tools:
        names = {t.get("name") if isinstance(t, dict) else str(t) for t in tools}
        if names != NP_CORE:
            missing, extra = NP_CORE - names, names - NP_CORE
            fail(f"tool surface differs: missing={sorted(missing)} extra={sorted(extra)}", problems)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--tier-hash", default=None)
    ap.add_argument("--config-only", action="store_true",
                    help="skip the mock-play assertions (config read-back only)")
    args = ap.parse_args()

    problems: list[str] = []
    check_config(args.run_dir, args.stack, args.tier_hash, problems)
    if not args.config_only:
        check_mock_play(args.run_dir, problems)

    if problems:
        print(f"PREFLIGHT FAILED for {args.run_dir} ({len(problems)} problems):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("Do not launch the batch.", file=sys.stderr)
        return 1
    print(f"preflight OK: {args.run_dir} ran the {args.stack} stack it declared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
