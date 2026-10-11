"""Environment adapters: how a game is started fresh and restored.

An adapter gives each player its own game. ``fresh`` starts a new game for an
attempt. ``restore`` rebuilds the game a checkpoint saved. Both return the
MCP servers the player must use (the spawn's ``mcp_servers``), so the game
server can key one game per bearer token, and the (possibly new) env handle.

The adapter is named by ``EXPLORE_ADAPTER`` (``"module:Class"``) or by the
run directory's ``explore.json`` (``{"adapter": "module:Class"}``).
``ScriptedAdapter`` is the no-game adapter the machine tests run with.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

ADAPTER_ENV = "EXPLORE_ADAPTER"
CONFIG_FILE = "explore.json"


class EnvFork(Protocol):
    """The two operations the explore loop needs from an environment."""

    def fresh(self, *, attempt: int, run_dir: Path) -> dict[str, Any]:
        """Start a new game for ``attempt``. Returns
        ``{"env_handle": ..., "mcp_servers": [...]}``."""

    def restore(self, env_handle: Any, *, attempt: int, run_dir: Path) -> dict[str, Any]:
        """Rebuild the game ``env_handle`` names, for ``attempt``. Returns
        ``{"env_handle": ..., "mcp_servers": [...]}``."""

    def settle(self, *, attempt: int, run_dir: Path) -> dict[str, Any]:
        """Wait until ``attempt``'s game is over, then stop it and record the
        outcome. Optional: an adapter without it is assumed to end attempts
        itself."""


class ScriptedAdapter:
    """No game at all: handles are echoed, and no MCP servers are needed.
    Records every call in ``run_dir/adapter_calls.jsonl`` so a test can
    check what the machine asked for."""

    name = "scripted"

    def _record(self, run_dir: Path, event: dict[str, Any]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "adapter_calls.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def fresh(self, *, attempt: int, run_dir: Path) -> dict[str, Any]:
        handle = {"scripted": True, "attempt": attempt}
        self._record(run_dir, {"op": "fresh", "attempt": attempt})
        return {"env_handle": handle, "mcp_servers": []}

    def restore(self, env_handle: Any, *, attempt: int, run_dir: Path) -> dict[str, Any]:
        self._record(run_dir, {"op": "restore", "attempt": attempt, "env_handle": env_handle})
        return {"env_handle": env_handle, "mcp_servers": []}

    def settle(self, *, attempt: int, run_dir: Path) -> dict[str, Any]:
        self._record(run_dir, {"op": "settle", "attempt": attempt})
        return {}


def read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / CONFIG_FILE
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def load_adapter(spec: str | None = None, *, run_dir: Path | None = None) -> EnvFork:
    """Resolve the adapter: ``spec``, then ``$EXPLORE_ADAPTER``, then
    ``explore.json``; ``"scripted"`` names the test adapter."""
    if spec is None:
        spec = os.environ.get(ADAPTER_ENV) or None
    if spec is None and run_dir is not None:
        configured = read_config(run_dir).get("adapter")
        spec = configured if isinstance(configured, str) else None
    if not spec or spec == "scripted":
        return ScriptedAdapter()
    module_name, _, class_name = spec.rpartition(":")
    if not module_name or not class_name:
        raise ValueError(f"adapter must be 'module:Class', '/path/to/file.py:Class', or 'scripted', got {spec!r}")
    if module_name.endswith(".py") or "/" in module_name:
        # A file path: the kernel venv has no PYTHONPATH into the repo, and
        # it must not (the env's `nethack.py` would shadow the `nethack`
        # skill package), so the adapter loads from its file.
        import importlib.util

        path = Path(module_name).expanduser()
        loader_spec = importlib.util.spec_from_file_location(f"explore_adapter_{abs(hash(str(path)))}", path)
        if loader_spec is None or loader_spec.loader is None:
            raise ImportError(f"cannot load adapter module from {path}")
        module = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, class_name)
    return factory()
