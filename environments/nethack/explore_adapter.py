"""The NetHack adapter for the explore skill: one game server per attempt.

`fresh` boots a new game for an attempt; `restore` stops the previous
attempt's server (which writes that attempt's outcome) and boots a server
restored from the chosen checkpoint. Both return the MCP server list the
player spawns with, so each player talks to its own game.

Configuration is `explore.json` in the run directory:

    {
      "adapter": "explore_adapter:NetHackAdapter",
      "python": "/path/to/venv/bin/python",        # has verifiers, mcp, aiohttp, the engine
      "pythonpath": ["/root/NetHack-engine", ".../skills/explore/src"],
      "env_dir": ".../environments/nethack",        # where explore_server.py lives
      "seed": 0, "character": "Val-hum-neu-fem", "task_spec": "full_nle",
      "skill_set": "np_core,request_map,search,save",
      "max_skill_calls": 150, "auto_checkpoint_every_n_calls": 20,
      "archive": ".../run/archive"                  # default: <run_dir>/archive
    }
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
BOOT_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 90.0
#: settle(): how long a game may make no progress before the attempt is
#: declared over anyway, and the hard cap on waiting.
SETTLE_QUIET_S = 180.0
SETTLE_TIMEOUT_S = 3600.0


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class NetHackAdapter:
    name = "nethack"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config
        # The detached servers this adapter started: kept so Python does not
        # warn about a dropped Popen handle while the server deliberately
        # outlives this process.
        self.processes: list[subprocess.Popen] = []

    def config(self, run_dir: pathlib.Path) -> dict[str, Any]:
        config = dict(self._config or _read_json(run_dir / "explore.json"))
        config.setdefault("python", sys.executable)
        config.setdefault("pythonpath", [])
        config.setdefault("env_dir", str(HERE))
        config.setdefault("archive", str(run_dir / "archive"))
        config.setdefault("seed", 0)
        config.setdefault("character", "Val-hum-neu-fem")
        config.setdefault("task_spec", "full_nle")
        config.setdefault("skill_set", "np_core,request_map,search,save")
        config.setdefault("max_skill_calls", 150)
        config.setdefault("auto_checkpoint_every_n_calls", 20)
        return config

    # -- the EnvFork protocol ---------------------------------------------------

    def fresh(self, *, attempt: int, run_dir: pathlib.Path) -> dict[str, Any]:
        run_dir = pathlib.Path(run_dir)
        self.finalize(run_dir, attempt - 1, reason="next attempt started")
        return self._boot(run_dir, attempt, resume=None, checkpoint_from=None)

    def restore(self, env_handle: Any, *, attempt: int, run_dir: pathlib.Path) -> dict[str, Any]:
        run_dir = pathlib.Path(run_dir)
        if not isinstance(env_handle, dict) or not env_handle.get("checkpoint_dir"):
            raise ValueError(f"env_handle must be {{'checkpoint_dir', 'id'}}, got {env_handle!r}")
        directory = pathlib.Path(env_handle["checkpoint_dir"])
        if not (directory / "state.bundle").is_file():
            raise FileNotFoundError(f"checkpoint {directory} has no state.bundle")
        self.finalize(run_dir, attempt - 1, reason="next attempt selected")
        return self._boot(run_dir, attempt, resume=str(directory), checkpoint_from=str(env_handle.get("id") or directory.name))

    def settle(self, *, attempt: int, run_dir: pathlib.Path, quiet_s: float | None = None, timeout_s: float | None = None) -> dict[str, Any]:
        """Wait until attempt ``attempt``'s GAME is over, then stop its server
        (which writes the outcome). Over means: died, ascended, terminated,
        or the call budget exhausted; or no game call for ``quiet_s``; or
        ``timeout_s`` elapsed. A player's turn can end while its game calls
        keep running in the kernel, so the orchestrator must not read the
        ledger before this returns. Returns the final state (empty when the
        server was already gone)."""
        run_dir = pathlib.Path(run_dir)
        config = self.config(run_dir)
        quiet_s = float(quiet_s if quiet_s is not None else config.get("settle_quiet_s", SETTLE_QUIET_S))
        timeout_s = float(timeout_s if timeout_s is not None else config.get("settle_timeout_s", SETTLE_TIMEOUT_S))
        ready = _read_json(self.ready_path(run_dir, attempt))
        pid = ready.get("pid")
        if not isinstance(pid, int) or not _alive(pid):
            return {}
        state_path = self.ready_path(run_dir, attempt).with_suffix(".state.json")
        started = time.time()
        last_calls, last_change = -1, time.time()
        reason = "settle timeout"
        while True:
            state = _read_json(state_path)
            calls = int(state.get("skill_calls") or 0)
            if calls != last_calls:
                last_calls, last_change = calls, time.time()
            if state.get("died") or state.get("ascended") or state.get("terminated") or state.get("budget_exhausted"):
                reason = "game over"
                break
            if time.time() - last_change > quiet_s:
                reason = f"no game call for {int(quiet_s)} s"
                break
            if time.time() - started > timeout_s:
                break
            if not _alive(pid):
                return state
            time.sleep(2.0)
        self.finalize(run_dir, attempt, reason=reason)
        return state

    # -- servers -----------------------------------------------------------------

    def ready_path(self, run_dir: pathlib.Path, attempt: int) -> pathlib.Path:
        return run_dir / "servers" / f"attempt-{attempt}.json"

    def finalize(self, run_dir: pathlib.Path, attempt: int, *, reason: str) -> bool:
        """Stop attempt ``attempt``'s server, if it is still up, and wait for
        it to write its outcome. Returns whether a server was stopped."""
        if attempt < 0:
            return False
        ready = _read_json(self.ready_path(run_dir, attempt))
        pid = ready.get("pid")
        if not isinstance(pid, int) or not _alive(pid):
            return False
        stop_path = pathlib.Path(str(self.ready_path(run_dir, attempt)) + ".stop")
        stop_path.write_text(reason + "\n", encoding="utf-8")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + STOP_TIMEOUT_S
        done = self.ready_path(run_dir, attempt).with_suffix(".done.json")
        while time.time() < deadline:
            if done.exists() or not _alive(pid):
                return True
            time.sleep(0.5)
        with_sig = signal.SIGKILL
        try:
            os.kill(pid, with_sig)
        except ProcessLookupError:
            pass
        return True

    def _boot(self, run_dir: pathlib.Path, attempt: int, *, resume: str | None, checkpoint_from: str | None) -> dict[str, Any]:
        config = self.config(run_dir)
        ready_path = self.ready_path(run_dir, attempt)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        for stale in (ready_path, ready_path.with_suffix(".done.json"), pathlib.Path(str(ready_path) + ".stop")):
            if stale.exists():
                stale.unlink()
        command = [
            str(config["python"]), "-m", "explore_server",
            "--run-dir", str(run_dir),
            "--attempt", str(attempt),
            "--archive", str(config["archive"]),
            "--ready", str(ready_path),
            "--seed", str(config["seed"]),
            "--character", str(config["character"]),
            "--task-spec", str(config["task_spec"]),
            "--skill-set", str(config["skill_set"]),
            "--max-skill-calls", str(config["max_skill_calls"]),
            "--every", str(config["auto_checkpoint_every_n_calls"]),
        ]
        if resume:
            command += ["--resume-checkpoint", resume]
        if checkpoint_from:
            command += ["--checkpoint-from", checkpoint_from]
        env = dict(os.environ)
        extra = [str(p) for p in config.get("pythonpath") or []]
        env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["EXPLORE_RUN_DIR"] = str(run_dir)
        log = open(ready_path.with_suffix(".server.log"), "ab")
        # Detached: the server must outlive the kernel that selected the
        # checkpoint, because the player (another process) uses it.
        process = subprocess.Popen(command, cwd=str(config["env_dir"]), env=env, stdout=log, stderr=log, start_new_session=True)
        self.processes.append(process)
        log.close()
        deadline = time.time() + float(config.get("boot_timeout_s", BOOT_TIMEOUT_S))
        while time.time() < deadline:
            ready = _read_json(ready_path)
            if ready.get("mcp_servers"):
                return {
                    "env_handle": {"checkpoint_dir": resume, "id": checkpoint_from} if resume else {"checkpoint_dir": None, "id": None},
                    "mcp_servers": ready["mcp_servers"],
                    "server": {"pid": ready.get("pid"), "url": ready.get("url"), "attempt": attempt},
                }
            if process.poll() is not None:
                tail = ready_path.with_suffix(".server.log").read_bytes()[-2000:].decode(errors="replace")
                raise RuntimeError(f"explore_server for attempt {attempt} exited (rc={process.returncode}):\n{tail}")
            time.sleep(0.5)
        raise TimeoutError(f"explore_server for attempt {attempt} did not become ready in {BOOT_TIMEOUT_S:.0f}s")
