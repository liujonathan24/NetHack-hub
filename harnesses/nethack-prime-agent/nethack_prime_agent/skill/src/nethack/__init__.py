"""Play NetHack. Every game action is an async call on this module.

Read `SKILL.md` in this skill's directory FIRST -- it lists every tool with its
exact arguments and how to descend. Do not spend calls probing: `help()` and the
tool JSON schemas will not tell you more than SKILL.md does.

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

__all__ = ["NetHack", "nethack"]


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


def __getattr__(name: str):
    # Forward bare module access (`import nethack; await nethack.search()`) to
    # the instance, so the model never has to know about the class.
    if name.startswith("_") or name in _RESERVED:
        raise AttributeError(name)
    return getattr(nethack, name)
