"""Play NetHack. Every game action is an async call on this module.

Read `SKILL.md` in this skill's directory FIRST -- it lists every tool with its
exact arguments and how to descend.

Quick start::

    import nethack
    print(await nethack.request_map())               # see the full map (free)
    print(await nethack.np_move_to(x=54, y=5))       # pathfind to a tile
    print(await nethack.np_press_key(key=">"))       # descend on a > staircase

Every tool is async (always `await`), and the returned string is the rendered
observation -- print it and read it before deciding the next call.
"""

# Maintainer notes (not model-facing): `url` is deliberately unset. The tool
# server binds an OS-assigned port per rollout, so the endpoint is written into
# `mcpServers.nethack.url` in the per-rollout `settings.json` and resolved at
# call time through the host (`rlm.host_request("mcp.config", ...)`, see
# `rlm/mcp_base.py:192-204`). Hard-coding a URL here would pin the first
# rollout's port into every later one.

# `bearer_token_env` exists only to satisfy `McpIntegration`'s credential gate:
# `_resolve_token` raises `NotEnabled` when it finds no token, and the verifiers
# tool server does not authenticate MCP at all. The harness sets the variable to
# a per-rollout random value.

from __future__ import annotations

from rlm import McpIntegration

# `NetHack` is deliberately NOT exported. pydoc filters module `help()` through
# `__all__`, and documenting the class re-advertises `list_tools` as an
# inherited method -- the one call we withhold below. The model never needs the
# class; it calls tools on the module.
__all__ = ["nethack"]


class NetHack(McpIntegration):
    server = "nethack"
    url = None
    bearer_token_env = "NETHACK_MCP_TOKEN"


nethack = NetHack()


# Names the kernel bootstrap probes to decide whether a module is a *callable*
# skill. Forwarding them would make `getattr(module, "run")` return an MCP tool
# stub, the module would be wrapped as callable, and `await nethack.<tool>()`
# dispatch would break. Same rule as the built-in `linear` / `notion` packages.
_RESERVED = {"run", "__wrapped__", "__call__"}

# Discovery is WITHHELD, not merely discouraged. SKILL.md is the authoritative
# API reference, and the JSON schemas the server publishes are empty -- so a
# `list_tools()` round-trip returns less than the document the agent already
# has, while costing a turn. Telling the model not to call it left the call
# available and put the idea in its head; removing it does neither.
#
# Only MODULE-level access is withheld. `McpIntegration` still discovers and
# binds tools on the instance internally, which is how `await nethack.<tool>()`
# works at all -- blocking that would break the game.
_WITHHELD = {"list_tools"}


def __getattr__(name: str):
    # Forward bare module access (`import nethack; await nethack.search()`) to
    # the instance, so the model never has to know about the class.
    if name.startswith("_") or name in _RESERVED:
        raise AttributeError(name)
    if name in _WITHHELD:
        raise AttributeError(
            f"nethack.{name} is not available. SKILL.md in this skill's "
            "directory is the complete tool reference -- every tool, with its "
            "exact arguments."
        )
    return getattr(nethack, name)
