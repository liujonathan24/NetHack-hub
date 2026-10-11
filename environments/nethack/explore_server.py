"""One NetHack game server for one explore attempt.

The explore machine gives each player its own game. This module boots one:
the real `python -m nethack_v1` MCP tool server, launched the way verifiers
launches it (`VF_CONFIG`, the `/state` + `/task` channel, `MCP_PORT_FILE`),
with the state channel served from this process and backed by a real
rollout `Trace`. A restore (`--resume-checkpoint`) rides `env_args`, so the
environment itself rebuilds the game from the checkpoint directory
(`nethack.py:924`), and every attempt writes into the shared archive
(`env_args.checkpoint_archive`).

While the player plays, this process mirrors the archive into the explore
ledger: a checkpoint directory that appears after this server started is
this attempt's, and its `meta.json` numbers become a ledger `checkpoint`
record (`env_handle` = the directory). When told to stop (SIGTERM, or a
`<ready file>.stop` request), it writes the attempt's `outcome` from the
trace state and shuts the game server down.

    python -m explore_server --run-dir R --attempt N --archive A --seed S \
        [--resume-checkpoint DIR --checkpoint-from cK] --ready R/servers/attempt-N.json

The ready file carries the MCP server list the player spawns with.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pathlib
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nethack_v1 as m  # noqa: E402
from nethack_harness.checkpoints import CheckpointIntegrityError, checkpoint_list, checkpoint_meta  # noqa: E402

#: The meta.json fields that become the ledger score (all harness-computed).
SCORE_FIELDS = (
    "dlvl", "xl", "hp", "max_hp", "hunger_state", "gameturn", "score",
    "balrog", "balrog_min", "max_dlvl_reached", "max_xp_level",
    "dungeon_number", "level_number",
)


def build_task_and_trace(**config_overrides):
    """A `(task, trace)` pair built the way `Rollout.run` builds them (the
    `tests/_mcp_server_harness.py` recipe)."""
    from verifiers.v1.state import state_cls
    from verifiers.v1.trace import Trace, TraceTask

    config = {
        "task_spec": "full_nle",
        "n_examples": 1,
        "explicit_seeds": [0],
        "character": "Val-hum-neu-fem",
        "env_args": {"skill_set": "netplay"},
        "seed_workspace": False,
    }
    config.update(config_overrides)
    task = m.load_taskset(m.NetHackTasksetConfig(**config)).select(1)[0]
    trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        state=state_cls(type(task))(),
    )
    return task, trace


class StateChannel:
    """The `/state` + `/task` channel a launched tool server needs, backed by
    a real Trace (the `InterceptionServer` contract, bearer-authenticated)."""

    def __init__(self, trace, task_data, secret: str):
        self.trace = trace
        self.task_data = task_data
        self.secret = secret

    def app(self):
        from aiohttp import web
        from pydantic import TypeAdapter

        adapter = TypeAdapter(m.NetHackState)

        def _auth(request):
            if request.headers.get("Authorization") != f"Bearer {self.secret}":
                raise web.HTTPUnauthorized()

        async def state_get(request):
            _auth(request)
            return web.Response(body=adapter.dump_json(self.trace.state), content_type="application/json", charset="utf-8")

        async def state_put(request):
            _auth(request)
            self.trace.state = m.NetHackState.model_validate_json(await request.read())
            return web.json_response({"ok": True})

        async def task_get(request):
            _auth(request)
            data = self.task_data
            return web.json_response({"cls": f"{type(data).__module__}:{type(data).__qualname__}", "task": data.model_dump_json()})

        app = web.Application()
        app.router.add_get("/state", state_get)
        app.router.add_put("/state", state_put)
        app.router.add_get("/task", task_get)
        return app


def atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def meta_score(meta: dict) -> dict:
    out = {}
    for key in SCORE_FIELDS:
        value = meta.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    return out


class Server:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = pathlib.Path(args.run_dir)
        self.archive = pathlib.Path(args.archive)
        self.ready_path = pathlib.Path(args.ready)
        self.stop_path = pathlib.Path(str(self.ready_path) + ".stop")
        self.started_at = time.time()
        self.mirrored: set[str] = set()
        self.proc: subprocess.Popen | None = None
        self.trace = None
        self.stop_event = asyncio.Event()
        self.stop_reason = "server stopped"

    # -- the ledger ------------------------------------------------------------

    def _ledger(self):
        from explore.ledger import Ledger

        return Ledger(self.run_dir)

    def mirror_archive(self) -> None:
        """Append this attempt's new checkpoints to the ledger."""
        ledger = self._ledger()
        known = {checkpoint.id for checkpoint in ledger.checkpoints()}
        for directory in checkpoint_list(self.archive):
            if directory.name in self.mirrored:
                continue
            try:
                meta = checkpoint_meta(directory)
            except CheckpointIntegrityError:
                continue
            created = float(meta.get("created_at") or 0.0)
            if created < self.started_at:
                self.mirrored.add(directory.name)  # another attempt's checkpoint
                continue
            checkpoint_id = str(meta.get("id") or directory.name)
            self.mirrored.add(directory.name)
            if checkpoint_id in known:
                continue
            ledger.add_checkpoint(
                checkpoint_id=checkpoint_id,
                attempt=self.args.attempt,
                score=meta_score(meta),
                env_handle={"checkpoint_dir": str(directory), "id": checkpoint_id},
                call=meta.get("call") if isinstance(meta.get("call"), int) else None,
                parent=self.args.checkpoint_from,
                label=str(meta.get("name") or ""),
                note=str(meta.get("note") or ""),
                now=created,
            )

    def write_outcome(self) -> None:
        state = self.trace.state if self.trace is not None else None
        final = {}
        if state is not None:
            final = {"max_dlvl_reached": int(state.max_dlvl_reached), "max_xp_level": int(state.max_xp_level), "skill_calls": int(state.skill_calls)}
        if state is not None and state.died:
            result, reason = "died", ""
        elif state is not None and state.budget_exhausted:
            result, reason = "budget", "call budget exhausted"
        elif state is not None and state.ascended:
            result, reason = "alive", "ascended"
        else:
            result, reason = "censored", self.stop_reason
        self._ledger().add_outcome(
            attempt=self.args.attempt,
            result=result,
            checkpoint_from=self.args.checkpoint_from,
            reason=reason,
            final_score=final,
        )

    # -- the game server -------------------------------------------------------

    async def boot(self) -> dict:
        from aiohttp import web

        env_args = {
            "skill_set": self.args.skill_set,
            "checkpoint_archive": str(self.archive),
            "auto_checkpoint": True,
            "auto_checkpoint_every_n_calls": int(self.args.every),
        }
        if self.args.resume_checkpoint:
            env_args["resume_checkpoint"] = self.args.resume_checkpoint
        task, trace = build_task_and_trace(
            task_spec=self.args.task_spec,
            explicit_seeds=[int(self.args.seed)],
            character=self.args.character,
            max_skill_calls=int(self.args.max_skill_calls),
            env_args=env_args,
        )
        self.trace = trace
        secret = secrets.token_hex(16)
        channel = StateChannel(trace, task.data, secret)
        runner = web.AppRunner(channel.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.runner = runner
        state_url = f"http://127.0.0.1:{runner.addresses[0][1]}/state"

        toolset = task.tool_servers()[0]
        port_file = pathlib.Path(tempfile.gettempdir()) / f"explore-port-{uuid.uuid4().hex}"
        env = {
            **os.environ,
            "VF_CONFIG": toolset.config.model_dump_json(),
            "VF_STATE_URL": state_url,
            "VF_STATE_SECRET": secret,
            "MCP_PORT_FILE": str(port_file),
        }
        log_path = self.ready_path.with_suffix(".game.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.game_log = open(log_path, "ab")
        self.proc = subprocess.Popen([sys.executable, "-m", "nethack_v1"], env=env, stdout=self.game_log, stderr=self.game_log, cwd=str(HERE))
        port = None
        for _ in range(300):
            if port_file.exists() and port_file.read_text().strip().isdigit():
                port = int(port_file.read_text().strip())
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"game server exited before binding (rc={self.proc.returncode}); see {log_path}")
            await asyncio.sleep(1)
        if port is None:
            raise RuntimeError(f"game server never reported a port; see {log_path}")
        url = f"http://127.0.0.1:{port}/mcp"
        import urllib.error
        import urllib.request

        for _ in range(300):
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"game server exited in setup (rc={self.proc.returncode}); see {log_path}")
                await asyncio.sleep(1)
        else:
            raise RuntimeError(f"game server never listened; see {log_path}")
        token = secrets.token_hex(16)
        ready = {
            "attempt": self.args.attempt,
            "url": url,
            "token": token,
            "pid": os.getpid(),
            "game_pid": self.proc.pid,
            "archive": str(self.archive),
            "resume_checkpoint": self.args.resume_checkpoint,
            "checkpoint_from": self.args.checkpoint_from,
            "started_at": self.started_at,
            "mcp_servers": [
                {"type": "http", "name": "nethack", "url": url, "headers": {"Authorization": f"Bearer {token}"}}
            ],
        }
        atomic_write(self.ready_path, json.dumps(ready, indent=2) + "\n")
        return ready

    def log(self, message: str) -> None:
        print(f"[explore_server attempt {self.args.attempt} {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)

    def request_stop(self, reason: str) -> None:
        self.log(f"stop requested: {reason}")
        self.stop_reason = reason
        self.stop_event.set()

    async def serve(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, lambda s=signum: self.request_stop(f"signal {s}"))
        try:
            await self.boot()
            self.log("ready; serving")
            while not self.stop_event.is_set():
                with contextlib.suppress(Exception):
                    self.mirror_archive()
                if self.stop_path.exists():
                    reason = self.stop_path.read_text(encoding="utf-8").strip() or "stop requested"
                    self.request_stop(reason)
                    break
                if self.proc is not None and self.proc.poll() is not None:
                    self.request_stop(f"game server exited (rc={self.proc.returncode})")
                    break
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.stop_event.wait(), timeout=float(self.args.poll))
        except asyncio.CancelledError:
            # The interpreter cancels the main task when a signal lands while
            # the loop waits (measured: the CancelledError arrives before the
            # registered handler runs). The stop request, when the adapter
            # made one, is in the stop file; a bare signal keeps its name.
            self.log("serve loop cancelled")
        except BaseException as error:  # noqa: BLE001 - logged, then re-raised
            self.log(f"serve loop left by {type(error).__name__}: {error!r}")
            raise
        finally:
            if not self.stop_event.is_set() and self.stop_path.exists():
                with contextlib.suppress(OSError):
                    self.stop_reason = self.stop_path.read_text(encoding="utf-8").strip() or self.stop_reason
            self.log(f"shutting down: {self.stop_reason}")
            try:
                self.mirror_archive()
            except Exception as error:  # noqa: BLE001 - logged, never fatal at shutdown
                self.log(f"mirror failed: {error!r}")
            try:
                self.write_outcome()
                self.log("outcome written")
            except Exception as error:  # noqa: BLE001
                self.log(f"outcome failed: {error!r}")
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except Exception:
                    self.proc.kill()
            with contextlib.suppress(Exception):
                await self.runner.cleanup()
            with contextlib.suppress(Exception):
                self.game_log.close()
            with contextlib.suppress(Exception):
                atomic_write(self.ready_path.with_suffix(".done.json"), json.dumps({"attempt": self.args.attempt, "reason": self.stop_reason, "ended_at": time.time()}) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ready", required=True, help="the ready file to write (JSON)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--character", default="Val-hum-neu-fem")
    parser.add_argument("--task-spec", default="full_nle")
    parser.add_argument("--skill-set", default="np_core,request_map,search,save")
    parser.add_argument("--max-skill-calls", type=int, default=150)
    parser.add_argument("--every", type=int, default=20, help="auto-checkpoint every N calls")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--checkpoint-from", default=None, help="the ledger id of the checkpoint this attempt resumes")
    parser.add_argument("--poll", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    asyncio.run(Server(args).serve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
