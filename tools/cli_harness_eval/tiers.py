#!/usr/bin/env python3
"""Resolve a tier stack from `configs/tool_tiers.toml` into launch arguments.

Tool inclusion is config, not a branch checkout. A cell names a stack --
`base`, `base+human`, `base+human+continual` -- and this module turns it into
the `ENV_ARGS` JSON and harness flags `launch_cell.sh` takes, plus the
provenance pin (tier-file sha256, tier-file version, code commit) that rides
`env_args` into the run's own resolved `config.toml`.

Why the pin travels in `env_args`: `load_environment` absorbs unknown keys via
`**kwargs`, and the eval CLI writes the whole `[taskset.env_args]` table back
out. So `tier_hash`/`tier_commit` land literally in the artifact that describes
the run, and a result can be replayed from that file alone -- no separate
manifest to lose.

Usage:
    tiers.py env-args base+human            # ENV_ARGS JSON for launch_cell.sh
    tiers.py flags    base+human+continual  # extra CONTINUAL_* env for the cell
    tiers.py check    base                  # refuse a stack this tree cannot honour
    tiers.py show                           # the resolved table, for eyeballing
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIER_FILE = REPO / "configs" / "tool_tiers.toml"


def load(path: Path = TIER_FILE) -> dict:
    return tomllib.load(open(path, "rb"))


def tier_hash(path: Path = TIER_FILE) -> str:
    """sha256 of the tier file. Bumping `version` without changing membership is
    cheap and human; this catches the reverse -- membership edited, version
    forgotten."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        # A dirty tree is not a pin. Say so in the artifact rather than quietly
        # recording a commit the run did not actually use.
        return f"{rev}-dirty" if dirty else rev
    except Exception:
        return "unknown"


def parse_stack(stack: str, cfg: dict) -> list[str]:
    allowed = cfg["stacks"]["allowed"]
    if stack not in allowed:
        raise SystemExit(
            f"tiers: {stack!r} is not an allowed stack. Allowed: {allowed}.\n"
            "  Tiers are never pooled ad hoc -- a new combination needs a line in\n"
            "  configs/tool_tiers.toml [stacks], not a shell one-liner."
        )
    return stack.split("+")


def env_args(stack: str, cfg: dict | None = None, *, path: Path = TIER_FILE) -> dict:
    """The ENV_ARGS dict for this stack, with the provenance pin folded in."""
    cfg = cfg or load(path)
    tiers = parse_stack(stack, cfg)

    out: dict = {"skill_set": cfg["base"]["skill_set"]}
    out.update(cfg["base"].get("env_args", {}))

    if "human" in tiers:
        for member in cfg["human"].get("members", []):
            # Only flag-shaped members can be switched on through config; a
            # `code` member is present or absent with the checkout, which is the
            # gap [human.gap] documents.
            if member.get("kind") == "flag":
                out.update(member.get("env_args", {}))

    # The pin. Unknown to load_environment (absorbed by **kwargs), written back
    # into the resolved config.toml by the eval CLI.
    out["tier_stack"] = stack
    out["tier_version"] = cfg["version"]
    out["tier_hash"] = tier_hash(path)
    out["tier_commit"] = code_commit()
    return out


def continual_flags(stack: str, cfg: dict | None = None) -> dict:
    """Shell env for the [continual] tier: the shared store and its write mode.
    Empty for stacks that do not include it."""
    cfg = cfg or load()
    if "continual" not in parse_stack(stack, cfg):
        return {}
    mech = cfg["continual"]["mechanism"]
    return {
        "CONTINUAL_HARNESS": mech["harness_continual_harness_dir"],
        "CONTINUAL_HARNESS_WRITABLE": "1" if mech["harness_continual_harness_writable"] else "",
    }


def check(stack: str, cfg: dict | None = None) -> list[str]:
    """Reasons this tree cannot honestly run `stack`. Empty means it can."""
    cfg = cfg or load()
    tiers = parse_stack(stack, cfg)
    problems: list[str] = []

    gap = cfg.get("human", {}).get("gap", {})
    unflagged = gap.get("unflagged", [])
    blocked = gap.get("blocks_tier_stack", [])
    if stack in blocked and unflagged:
        problems.append(
            f"stack {stack!r} cannot be produced by config on this tree: "
            f"{', '.join(unflagged)} is unconditional code, so it is present "
            "whether or not the tier declares it. Check out "
            f"{cfg['base']['code_pin']['branch']} at "
            f"{'/'.join(cfg['base']['code_pin']['honesty_pass'])} to run a true "
            "base cell, or gate it behind a flag first (see [human.gap])."
        )

    if "human" in tiers:
        for m in cfg["human"].get("members", []):
            if not m.get("measured"):
                problems.append(
                    f"human member {m['id']!r} has measured=false: it is a patch, "
                    "not a tier member, until a cell has measured it against base."
                )
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    cfg = load()

    if cmd == "show":
        print(f"tier file : {TIER_FILE}")
        print(f"version   : {cfg['version']}  hash: {tier_hash()}")
        print(f"code      : {code_commit()}")
        print(f"stacks    : {cfg['stacks']['allowed']}")
        for stack in cfg["stacks"]["allowed"]:
            problems = check(stack, cfg)
            mark = "OK " if not problems else "BLOCKED"
            print(f"\n[{mark}] {stack}")
            print("  env_args:", json.dumps(env_args(stack, cfg)))
            flags = continual_flags(stack, cfg)
            if flags:
                print("  flags   :", flags)
            for p in problems:
                print("  !!", p)
        return 0

    stack = sys.argv[2] if len(sys.argv) > 2 else "base"
    if cmd == "env-args":
        print(json.dumps(env_args(stack, cfg)))
        return 0
    if cmd == "flags":
        for k, v in continual_flags(stack, cfg).items():
            print(f"{k}={v}")
        return 0
    if cmd == "check":
        problems = check(stack, cfg)
        for p in problems:
            print(f"tiers: {p}", file=sys.stderr)
        return 1 if problems else 0

    print(f"tiers: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
