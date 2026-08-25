"""Your own NetHack policies. This package is YOURS to edit.

`netplay` is ordinary Python on the kernel's `sys.path`, living beside the
frozen `nethack` shim. Read it, rewrite it, add to it -- what you leave here
persists into the next episode, and into every later round of this experiment.

    import netplay
    print(await netplay.explore())     # your explore policy
    print(netplay.check())             # compile-check after every edit

NAME COLLISION, on purpose: `netplay.explore` and `netplay.descend` are the
FUNCTIONS, not the modules of the same name -- the policy call is what you want
99 times out of 100, so it wins the short name. To reach a module object use
`netplay.module("explore")`; to edit the file, open it by path.

What is yours          `explore.py`, `descend.py`, `fight.py`, `survive.py`,
                       and any new module you add beside them.
What is frozen         `_base.py` (the primitive floor) and this file. Both are
                       restored from the repo at the start of every episode, so
                       edits to them are silently undone -- do not spend a turn
                       on it.

The rule that matters: `_base` is the only way to touch the game. A policy that
returns an observation it did not get from `_base` is fabricating, and the
harness reconciles every call against the server's own record.
"""

# Maintainer note (not model-facing): import failures are recorded rather than
# raised. `netplay` must stay importable even when a composite the agent just
# edited does not compile -- otherwise `netplay.check()`, the one tool for
# diagnosing the breakage, becomes unreachable too, and a single bad edit
# bricks every later rollout (SPEC §6.1). Failures are loud in `check()` and in
# `import_errors()`; they are not silent, they are just not fatal.

from __future__ import annotations

import warnings as _warnings

from netplay._base import (  # noqa: F401  -- re-exported for the agent
    apply,
    call_count,
    check,
    features,
    grid,
    kick,
    monsters,
    position,
    pray,
    press,
    rest,
    screen,
    search,
    status,
    tile,
)

_import_errors: dict[str, str] = {}
_modules: dict[str, object] = {}

for _name in ("explore", "descend", "fight", "survive"):
    try:
        _mod = __import__(f"netplay.{_name}", fromlist=["*"])
    except Exception as _exc:  # noqa: BLE001 -- see maintainer note
        _import_errors[_name] = f"{type(_exc).__name__}: {_exc}"
        continue
    _modules[_name] = _mod
    # Bound AFTER the module is recorded, because two of these functions share
    # a name with their own module (`explore`, `descend`) and the function is
    # the one worth the short name. `module()` below is the unambiguous route.
    for _attr in getattr(_mod, "__all__", ()):
        globals()[_attr] = getattr(_mod, _attr)

if _import_errors:
    _warnings.warn(
        "netplay: these modules failed to import and their policies are "
        "UNAVAILABLE: "
        + "; ".join(f"{k} ({v})" for k, v in sorted(_import_errors.items()))
        + " -- call netplay.check() for detail, then fix the file.",
        RuntimeWarning,
        stacklevel=2,
    )


def import_errors() -> dict[str, str]:
    """Modules that failed to import this episode, as {module: error}."""
    return dict(_import_errors)


def module(name: str):
    """The submodule object called `name`, e.g. `module("explore")`.

    Use this rather than `netplay.explore`, which is the FUNCTION: two of the
    policy functions share a name with the module that defines them.
    """
    if name not in _modules:
        raise KeyError(
            f"netplay has no loaded module {name!r}; loaded: "
            f"{sorted(_modules)}; failed: {sorted(_import_errors)}"
        )
    return _modules[name]
