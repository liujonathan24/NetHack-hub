"""Runtime feature flags for the post-baseline tool fixes.

Why this exists: the E10 baseline (`docs/E10_E11_RESULTS.md`) is the denominator
of every claim in this series, and it was measured on the harness as it stood at
the honesty pass. The NetPlay audit fixes that landed afterwards
(`c1a0bec`, `aee5c43`, `ea8cc15`, PR #29) were *unconditional code*: once merged,
every cell got them, so "run the baseline" silently meant "run the baseline plus
three undated improvements" and the published reference numbers stopped
describing anything reproducible.

Each of those commits is now one flag, **default OFF**, so:

    all flags off  ==  the tree behaves as it did when the baseline was measured
    all flags on   ==  current HEAD behaviour

That is what lets `configs/tool_tiers.toml` express `[base]` and `[human]` as
configuration instead of a branch checkout. Same pattern as `descent_gate`:
present in the code, off unless a cell asks for it.

Flags are set once per process from `load_environment`'s kwargs (they arrive as
strings through `ENV_ARGS`) and read at the point of behaviour. The vendored
NetPlay code imports this lazily, inside the functions, so no import cycle and
no vendor-to-harness dependency at module load.
"""
from __future__ import annotations

from typing import Any

# Commit-mapped, one flag per landed change, so a cell can name exactly which
# improvements it is running and a result can be replayed from that name.
_DEFAULTS: dict[str, bool] = {
    # c1a0bec -- interruption severity filter, move_to progress telemetry,
    # np_rest delivering its full count, the added RawKeyPress keys.
    "netplay_telemetry": False,
    # c1a0bec (stale-target report) + ea8cc15 (nearest-monster hint on the
    # no-monster-here path). One behaviour split across two commits; gating half
    # of it would produce a state that never existed, so they share a flag.
    "melee_hints": False,
    # aee5c43 -- SKILL.md stating the MAP coordinate frame explicitly.
    # DECLARED BUT NOT YET CONSUMED: that commit is local to another worktree
    # and is not on origin/exp/e8-planning-guidance, so there is nothing here to
    # gate. The name is reserved so configs/tool_tiers.toml can list all three
    # members now; wiring it is a one-line strip in PrimeAgentHarness (the
    # `allow_batching` pattern) once the commit lands on the remote.
    "skill_doc_coords": False,
}

_flags: dict[str, bool] = dict(_DEFAULTS)


def _coerce(value: Any) -> bool:
    """env_args arrive as strings through the CLI, so "false"/"0"/"off" must not
    read as True the way a non-empty string otherwise would."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "false", "0", "no", "off")


def configure(**kwargs: Any) -> dict[str, bool]:
    """Set flags from `load_environment` kwargs. Unknown keys are ignored (the
    env takes many kwargs that are not flags); returns the resolved state so the
    caller can record it."""
    for name in _DEFAULTS:
        if name in kwargs:
            _flags[name] = _coerce(kwargs[name])
    return dict(_flags)


def enabled(name: str) -> bool:
    return _flags.get(name, _DEFAULTS.get(name, False))


def snapshot() -> dict[str, bool]:
    return dict(_flags)


def reset() -> None:
    """Restore defaults. For tests -- the flags are process-global."""
    _flags.clear()
    _flags.update(_DEFAULTS)
