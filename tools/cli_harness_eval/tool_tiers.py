#!/usr/bin/env python3
"""Expand a tool tier into eval-CLI overrides. Single source of truth.

`launch_cell.sh` used to duplicate the tier mapping in bash "because the
launcher must not parse TOML", with a test pinning the two in sync. That test
could not detect any of the four drift mutations tried against it -- deleting
the whole TOOL_TIER block still passed -- because it only grepped for flag
names. Duplication plus a substring test is worse than no duplication: the
launcher already runs Python for the engine preflight and the ENV_ARGS
flattener, so it can read the TOML directly and the drift class disappears.

    tool_tiers.py flags <tier> [--arm ARM]   # override flags, one per line
    tool_tiers.py contract                   # the cell contract, as JSON
    tool_tiers.py show                       # everything, for eyeballing

Every tier also emits its provenance -- tier name, registry sha256, code commit
-- through `env_args`, which `load_environment` absorbs via `**kwargs` and the
eval CLI writes back into the run's own `config.toml`. A cell is then replayable
from its output artifact alone, which is what `docs/CONTINUAL_HARNESS_BASELINE.md`
asks for and the env-var mapping never recorded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIER_FILE = REPO / "tools" / "cli_harness_eval" / "configs" / "tool_tiers.toml"

# Flags consumed by the harness process (PrimeAgentHarnessConfig), not the env.
# The doc is materialized by the harness, which never imports the env-side flag
# registry -- so these must go to `--harness.*`, and only to an arm whose config
# actually declares them. `HarnessConfig` is `extra="forbid"`, so sending
# `--harness.skill_doc_coords` to the claude_code or control arm is a hard
# ValidationError; the previous block was not arm-guarded.
_HARNESS_SIDE = {"skill_doc_coords"}
_PRIME_AGENT_ARMS = {"prime_agent", "prime_agent_b80"}


def _cli_value(v) -> str:
    """A scalar as the eval CLI expects it: strings RAW (json.dumps would ship
    the quotes into the config -- a tier key `descent_gate = "norm"` must reach
    the env as norm, not "norm"), everything else JSON so 1.0/true coerce.
    Same contract as launch_cell.sh's ENV_ARGS flattener. The existing tiers
    carry only booleans, which json.dumps identically -- byte-identical output
    for [base]/[human]/[continual]."""
    return v if isinstance(v, str) else json.dumps(v)


def load(path: Path = TIER_FILE) -> dict:
    return tomllib.load(open(path, "rb"))


def registry_hash(path: Path = TIER_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def code_commit() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        # A dirty tree is not a pin; record that rather than a commit the run
        # did not actually use.
        return f"{rev}-dirty" if dirty else (rev or "unknown")
    except Exception:
        return "unknown"


def tiers(cfg: dict) -> list[str]:
    return [k for k in cfg if k not in ("contract",)]


def contract_for(tier: str | None, cfg: dict | None = None,
                 *, path: Path = TIER_FILE) -> dict:
    """The cell contract as `tier` runs it: the global [contract] with the
    tier's own [<tier>.contract] overrides merged over it (one level deep for
    sub-tables such as `tune`, so an experiment tier can add a generation knob
    without restating -- or accidentally dropping -- reveal_map). No tier, or a
    tier without overrides, returns the global contract unchanged: every
    pre-existing caller behaves byte-identically."""
    cfg = cfg or load(path)
    base = dict(cfg["contract"])
    if tier and tier in cfg:
        for k, v in (cfg[tier].get("contract") or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
    return base


def flags(tier: str, arm: str = "prime_agent", cfg: dict | None = None,
          *, path: Path = TIER_FILE) -> list[str]:
    """The eval-CLI override flags for `tier` on `arm`, ready to splice into argv."""
    cfg = cfg or load(path)
    if tier not in tiers(cfg):
        raise SystemExit(
            f"tool_tiers: unknown tier {tier!r} (have: {', '.join(tiers(cfg))}). "
            "A new tier needs a section in tool_tiers.toml, not a launcher edit."
        )
    contract = contract_for(tier, cfg)
    out: list[str] = [
        "--taskset.env_args.skill_set", contract["skill_set"],
        "--taskset.env_args.auto_dismiss", contract["auto_dismiss"],
    ]
    # EVERY tune leaf, not just reveal_map: an experiment tier's contract can
    # add generation knobs (locked_door, room_density), and dotted leaves merge
    # into the config's table where a whole-object override would replace it.
    for k in sorted(contract["tune"]):
        out += [f"--taskset.env_args.tune.{k}", json.dumps(contract["tune"][k])]
    out += [
        # model / character / task_spec were declared in the contract but never
        # emitted -- they only happened to match configs/prime_agent.toml, so a
        # change to that file would have moved the "baseline" without touching
        # the registry that claims to define it. Emit them, so the contract is
        # enforced rather than coincidental.
        "--model", contract["model"],
        "--taskset.character", contract["character"],
        "--taskset.task_spec", contract["task_spec"],
        "--taskset.max_parallel_skill_calls", json.dumps(contract["max_parallel_skill_calls"]),
    ]
    # Harness-side contract factors: same arm restriction as the tier's own
    # harness flags -- only the prime_agent harness declares them.
    if arm in _PRIME_AGENT_ARMS:
        out += [
            "--harness.allow_batching", json.dumps(contract["allow_batching"]),
            "--harness.max_relaunches", json.dumps(contract["max_relaunches"]),
        ]
    for name, value in cfg[tier].items():
        if name == "contract":
            continue  # per-tier contract overrides, consumed by contract_for()
        if name in _HARNESS_SIDE:
            # Silently skipping would give the wrong doc; the launcher refuses
            # the combination instead (see launch_cell.sh).
            if arm in _PRIME_AGENT_ARMS:
                out += [f"--harness.{name}", json.dumps(value)]
        elif isinstance(value, dict):
            # Flatten to dotted leaves: a dict value would reach the CLI as one
            # JSON blob and REPLACE the whole sub-table (dropping e.g.
            # tune.reveal_map); dotted scalars merge, same as the launcher's
            # ENV_ARGS flattener.
            for k in sorted(value):
                out += [f"--taskset.env_args.{name}.{k}", _cli_value(value[k])]
        else:
            out += [f"--taskset.env_args.{name}", _cli_value(value)]
    # Provenance, into the run's own config.toml.
    out += [
        "--taskset.env_args.tool_tier", tier,
        "--taskset.env_args.tool_tier_hash", registry_hash(path),
        "--taskset.env_args.tool_tier_commit", code_commit(),
    ]
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    cfg = load()

    if cmd == "contract":
        # `contract [tier]`: with a tier, its [<tier>.contract] overrides are
        # merged in (variant, skill_set, tune knobs); bare `contract` is the
        # global table, unchanged for existing callers.
        print(json.dumps(contract_for(sys.argv[2] if len(sys.argv) > 2 else None, cfg)))
        return 0
    if cmd == "show":
        print(f"registry : {TIER_FILE}")
        print(f"hash     : {registry_hash()}   commit: {code_commit()}")
        print(f"contract : {json.dumps({k: v for k, v in cfg['contract'].items() if not isinstance(v, dict)})}")
        print(f"tune     : {cfg['contract']['tune']}")
        for t in tiers(cfg):
            print(f"\n[{t}] {cfg[t]}")
            print("  prime_agent:", " ".join(flags(t, "prime_agent", cfg)))
            print("  claude_code:", " ".join(flags(t, "claude_code", cfg)))
        return 0
    if cmd == "flags":
        tier = sys.argv[2] if len(sys.argv) > 2 else "base"
        arm = "prime_agent"
        if "--arm" in sys.argv:
            arm = sys.argv[sys.argv.index("--arm") + 1]
        for f in flags(tier, arm, cfg):
            print(f)
        return 0
    if cmd == "harness-side":
        # For the launcher's arm guard.
        for name in sorted(_HARNESS_SIDE):
            print(name)
        return 0

    print(f"tool_tiers: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
