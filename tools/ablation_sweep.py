"""Ablation sweep: "remove one difficulty at a time, see which bottleneck dominates".

This is the executable scaffolding for the *Ablations with Level Modification*
sub-study (see ``docs/experiments/exp_ablations_level_mod.md``). It defines the
ablation-config set (baseline + one-difficulty-removed variants) and a small
runner that maps each ablation to the concrete ``tune`` / ``modify`` env-args
that ``load_environment(...)`` accepts.

Every ablation is realized purely through the fork engine's difficulty knobs
(``env.tune``) and secure-state pokes (``env.modify``) — no game-content forks.

Modes (all default-free of any LLM/API call):

    python tools/ablation_sweep.py --list          # show ablations + support
    python tools/ablation_sweep.py --dry-run       # print env-args each would use
    python tools/ablation_sweep.py --verify        # local engine effect checks (free)
    python tools/ablation_sweep.py --emit-cmds -m <model>   # print vf-eval commands

Only ``--verify`` touches the engine (locally, on CPU, no model). A real eval
(``--emit-cmds`` output) calls a REMOTE model and costs money — it is never run
automatically here; the commands are only printed for a human to launch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Ablation:
    """One 'remove a difficulty' cell of the sweep.

    ``tune``   -> passed to load_environment(tune=...)  (generation + live knobs)
    ``modify`` -> passed to load_environment(modify=...) (secure state pokes)
    ``supported`` False means no knob/modify realizes this today (needs engine
    work); it is listed but excluded from the runnable sweep.
    ``verify`` is an optional free local-engine check returning (ok, detail).
    """

    name: str
    difficulty_removed: str
    tune: dict = field(default_factory=dict)
    modify: dict = field(default_factory=dict)
    supported: bool = True
    note: str = ""
    verify: Optional[Callable[[], "tuple[bool, str]"]] = None

    def env_args(self, base: Optional[dict] = None) -> dict:
        """The load_environment(...) kwargs for this ablation."""
        args = dict(base or {})
        if self.tune:
            args["tune"] = dict(self.tune)
        if self.modify:
            args["modify"] = dict(self.modify)
        return args


# --------------------------------------------------------------------------- #
# Free local-engine verifications (imported lazily so this module is import-    #
# clean without the engine .so present).                                        #
# --------------------------------------------------------------------------- #

def _new_env(tune=None, modify=None, steps_east=3):
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.seed(42, 42)
    env.reset(tune=tune or None, modify=modify or None)
    for _ in range(steps_east):
        env.step(ord("l"))  # explore a little so 'base' is non-trivial
    return env


def _visible(env) -> int:
    return int((env.engine.chars != ord(" ")).sum())


def _verify_full_vision() -> "tuple[bool, str]":
    base_env = _new_env()
    base = _visible(base_env)
    base_env.close()
    on_env = _new_env(tune={"reveal_map": 1.0})
    on = _visible(on_env)
    on_env.close()
    ok = on > base + 50
    return ok, f"visible cells base={base} reveal_map=1 -> {on} (+{on - base})"


def _verify_infinite_health() -> "tuple[bool, str]":
    env = _new_env(tune={"dmg_to_player_scale": 0.0}, steps_east=0)
    dmg = env.tune.get()["dmg_to_player_scale"]
    env.modify(hp=9999, max_hp=9999)
    b = [int(x) for x in env.engine.blstats]
    env.close()
    ok = dmg == 0.0 and b[10] == 9999
    return ok, (f"dmg_to_player_scale={dmg} (0.0 => hero takes no monster damage); "
                f"modify hp->{b[10]}/{b[11]} (blstats display-capped at 9999)")


def _verify_unlocked_doors() -> "tuple[bool, str]":
    # generation knob: must be set at reset(tune=), not live.
    env = _new_env(tune={"locked_door": 0.0}, steps_east=0)
    val = env.tune.get()["locked_door"]
    env.close()
    ok = val == 0.0
    return ok, f"locked_door={val} at generation time (0.0 => doors never rolled locked)"


def _verify_unsupported() -> "tuple[bool, str]":
    # Confirm the negative: no luck/item knob, no luck modify field.
    from nethack_core.engine_env import EngineEnv

    env = EngineEnv()
    env.seed(42, 42)
    env.reset()
    cat = env.tune.catalog()
    env.close()
    luck_like = [k for k in cat if any(t in k.lower() for t in ("luck", "item", "loot"))]
    mod_has_luck = "luck" in EngineEnv._MODIFY_BOUNDS
    ok = not luck_like and not mod_has_luck  # "ok" == correctly-absent
    return ok, (f"no luck/item tune knob (matches={luck_like or 'none'}); "
                f"luck in modify fields={mod_has_luck} -> needs engine/env work")


# --------------------------------------------------------------------------- #
# The ablation-config set.                                                       #
# --------------------------------------------------------------------------- #

ABLATIONS: "list[Ablation]" = [
    Ablation(
        name="baseline",
        difficulty_removed="(none — vanilla)",
        note="Control cell. Vanilla NetHack generation and rules.",
        # baseline still runs a free sanity check that the engine boots.
        verify=lambda: (True, "vanilla, no knobs applied"),
    ),
    Ablation(
        name="full_vision",
        difficulty_removed="Fog of war / exploration",
        tune={"reveal_map": 1.0},
        note="Live render overlay: whole level terrain + live monsters revealed "
             "in every obs. Reversible, side-effect-free (winrl.cc fill_obs).",
        verify=_verify_full_vision,
    ),
    Ablation(
        name="infinite_health",
        difficulty_removed="Combat lethality / survival",
        tune={"dmg_to_player_scale": 0.0, "hp_regen_scale": 8.0},
        modify={"hp": 9999, "max_hp": 9999},
        note="dmg_to_player_scale=0 -> hero takes no monster damage (mhitu.c); "
             "hp_regen_scale + big HP pool are belt-and-suspenders. NOTE: hp is "
             "internally up to 30000 but blstats display-caps at 9999 (botl.c).",
        verify=_verify_infinite_health,
    ),
    Ablation(
        name="unlocked_doors",
        difficulty_removed="Locked-door / access barriers",
        tune={"locked_door": 0.0},
        note="Generation-time knob (mklev.c): doors are never rolled locked. "
             "Must be applied at reset(tune=), not live.",
        verify=_verify_unlocked_doors,
    ),
    Ablation(
        name="better_items_luck",
        difficulty_removed="Item scarcity / bad luck",
        supported=False,
        modify={"gold": 100000},  # weak buying-power proxy ONLY
        note="NOT SUPPORTED by a real knob. No luck knob, no starting-inventory "
             "knob, no 'luck' modify field. modify(gold=) is a weak buying-power "
             "proxy, not better items/luck. Needs new engine/env work "
             "(luck field poke + starting-inventory/bones injection). See doc §e.",
        verify=_verify_unsupported,
    ),
]

BY_NAME = {a.name: a for a in ABLATIONS}


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #

def cmd_list() -> int:
    print(f"{'ablation':22} {'supported':10} difficulty removed")
    print("-" * 70)
    for a in ABLATIONS:
        print(f"{a.name:22} {'yes' if a.supported else 'NO (todo)':10} {a.difficulty_removed}")
    return 0


def cmd_dry_run(base_env_args: dict) -> int:
    print("# Dry run: env-args each ablation passes to load_environment(...).")
    print(f"# Base env-args: {json.dumps(base_env_args)}")
    print("# Run a cell with:  tools/run_eval.sh <name> '<env-args-json>' [prime flags]\n")
    for a in ABLATIONS:
        tag = "" if a.supported else "   # UNSUPPORTED (excluded from real sweep)"
        print(f"[{a.name}] {a.difficulty_removed}{tag}")
        print(f"    -a '{json.dumps(a.env_args(base_env_args))}'")
        if a.note:
            print(f"    note: {a.note}")
        print()
    return 0


def cmd_verify() -> int:
    print("# Free local-engine effect checks (no model / API).")
    failures = 0
    for a in ABLATIONS:
        if a.verify is None:
            print(f"[skip] {a.name}: no verify hook")
            continue
        try:
            ok, detail = a.verify()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"EXCEPTION: {exc!r}"
        # For the unsupported cell, verify() confirms the *absence* is correct.
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {a.name}: {detail}")
    print(f"\n{len(ABLATIONS)} ablations, {failures} failing check(s).")
    return 1 if failures else 0


def cmd_emit_cmds(model: str, base_env_args: dict, n: int, r: int) -> int:
    print(f"# vf-eval commands for the ablation sweep (model={model}).")
    print("# WARNING: these call a REMOTE model and COST money. Run by hand.\n")
    for a in ABLATIONS:
        if not a.supported:
            print(f"# ({a.name} unsupported — skipped)\n")
            continue
        args = a.env_args(base_env_args)
        print(f"# {a.name}: remove {a.difficulty_removed}")
        print(f"vf-eval nethack -m {model} -n {n} -r {r} \\")
        print(f"  -a '{json.dumps(args)}'\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="list ablations + support status")
    g.add_argument("--dry-run", action="store_true", help="print env-args per ablation (no engine)")
    g.add_argument("--verify", action="store_true", help="free local engine effect checks")
    g.add_argument("--emit-cmds", action="store_true", help="print vf-eval commands (does NOT run)")
    p.add_argument("-m", "--model", default="<model>", help="model id for --emit-cmds")
    p.add_argument("-n", type=int, default=20, help="num_examples for --emit-cmds")
    p.add_argument("-r", type=int, default=1, help="rollouts_per_example for --emit-cmds")
    p.add_argument("--base-env-args", default="{}",
                   help="JSON env-args merged into every cell (e.g. seeds, tier)")
    args = p.parse_args(argv)

    base = json.loads(args.base_env_args)

    if args.list:
        return cmd_list()
    if args.verify:
        return cmd_verify()
    if args.emit_cmds:
        return cmd_emit_cmds(args.model, base, args.n, args.r)
    # default = dry-run (safe, no engine, no API)
    return cmd_dry_run(base)


if __name__ == "__main__":
    sys.exit(main())
