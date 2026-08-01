"""Boot `python -m nethack_v1` as a real MCP tool server, for tests that need one.

Not a test module (the leading underscore keeps pytest from collecting it). Two
test files need the same rig — the cross-route trace-equivalence check and the
call-budget concurrency check — and duplicating ~90 lines of process/port/channel
plumbing between them is how the two copies drift.

What this stands up is deliberately the *real* thing on both sides of the wire:

* the server is launched exactly as `serve_in_runtime` launches it
  (`v1/mcp/launch.py:186-215`): `[sys.executable, "-m", <module>]` with the
  config in `VF_CONFIG` and the channel coordinates in `VF_STATE_URL` /
  `VF_STATE_SECRET`, reporting its port through `MCP_PORT_FILE`;
* `MiniInterception` implements the same `/state` + `/task` contract as
  `InterceptionServer` (`v1/interception/server.py:745-795`) — bearer auth, the
  state as the typed model's JSON, the task as `{"cls": ..., "task": ...}` — and
  its backing store is a real `Trace.state`, so what the tools push is what
  scoring later reads;
* the client is a real `mcp.ClientSession` over streamable HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack_v1 as m  # noqa: E402

SECRET = "vf-test-secret"


class MiniInterception:
    """The two channels a launched tool server needs, backed by a real Trace."""

    def __init__(self, trace, task_data):
        self.trace = trace
        self.task_data = task_data
        self.gets = 0
        self.puts = 0

    def app(self):
        from aiohttp import web
        from pydantic import TypeAdapter

        adapter = TypeAdapter(m.NetHackState)

        def _auth(request):
            assert request.headers.get("Authorization") == f"Bearer {SECRET}"

        async def state_get(request):
            _auth(request)
            self.gets += 1
            return web.Response(
                body=adapter.dump_json(self.trace.state),
                content_type="application/json",
                charset="utf-8",
            )

        async def state_put(request):
            _auth(request)
            self.puts += 1
            self.trace.state = m.NetHackState.model_validate_json(await request.read())
            return web.json_response({"ok": True})

        async def task_get(request):
            _auth(request)
            data = self.task_data
            return web.json_response(
                {
                    "cls": f"{type(data).__module__}:{type(data).__qualname__}",
                    "task": data.model_dump_json(),
                }
            )

        app = web.Application()
        app.router.add_get("/state", state_get)
        app.router.add_put("/state", state_put)
        app.router.add_get("/task", task_get)
        return app


def _log_tail(log, proc, why: str) -> str:
    log.flush()
    tail = pathlib.Path(log.name).read_bytes()[-3000:].decode(errors="replace")
    return f"{why} (rc={proc.returncode})\n--- tool server log ---\n{tail}"


@contextlib.asynccontextmanager
async def booted_toolset(task, trace):
    """Yield `(mcp.ClientSession, MiniInterception)` for a live tool server.

    The server owns a real per-rollout engine, so this costs an engine boot
    (~5-10 s). Tear-down kills the process and the channel.
    """
    from aiohttp import web
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    mini = MiniInterception(trace, task.data)
    runner = web.AppRunner(mini.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    state_url = f"http://127.0.0.1:{runner.addresses[0][1]}/state"

    toolset = task.tool_servers()[0]
    port_file = pathlib.Path(tempfile.gettempdir()) / f"vf-port-{uuid.uuid4().hex}"
    env = {
        **os.environ,
        "VF_CONFIG": toolset.config.model_dump_json(),
        "VF_STATE_URL": state_url,
        "VF_STATE_SECRET": SECRET,
        "MCP_PORT_FILE": str(port_file),
    }
    log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "nethack_v1"], env=env, stdout=log, stderr=log
    )
    try:
        port = None
        for _ in range(300):
            if port_file.exists() and port_file.read_text().strip().isdigit():
                port = int(port_file.read_text().strip())
                break
            assert proc.poll() is None, _log_tail(log, proc, "server exited before binding")
            await asyncio.sleep(1)
        assert port, _log_tail(log, proc, "server never reported a port")
        url = f"http://127.0.0.1:{port}/mcp"

        # The port file is written BEFORE setup (`v1/mcp/server.py:243-246`), so
        # the socket is bound but not listening yet; verifiers' own `_PROBE`
        # (`v1/mcp/launch.py:52-63`) polls the same way.
        import urllib.error
        import urllib.request

        for _ in range(300):
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                assert proc.poll() is None, _log_tail(log, proc, "server exited in setup")
                await asyncio.sleep(1)
        else:
            raise AssertionError(_log_tail(log, proc, "server never listened"))

        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session, mini
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()
        with contextlib.suppress(Exception):
            pathlib.Path(log.name).unlink()
        await runner.cleanup()


def build_task_and_trace(**config_overrides):
    """A `(task, trace)` pair built the way `Rollout.run` builds them."""
    from verifiers.v1.state import state_cls
    from verifiers.v1.trace import Trace, TraceTask

    config = {
        "task_spec": "full_nle",
        "n_examples": 1,
        "explicit_seeds": [0],
        "character": "Val-hum-neu-fem",
        "env_args": {"skill_set": "netplay"},
        # Nothing runs an agent in these tests, so there is no workspace to seed.
        "seed_workspace": False,
    }
    config.update(config_overrides)
    task = m.load_taskset(m.NetHackTasksetConfig(**config)).select(1)[0]
    trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        state=state_cls(type(task))(),
    )
    return task, trace
