"""The explore adapter against a REAL game server: boot, save, stop, restore.

Costs two engine boots (~5-10 s each). Needs the eval venv with the engine on
PYTHONPATH (the `launch_cell.sh` recipe) and the explore skill importable;
skipped otherwise. Run with:

    PYTHONPATH=tools/pycompat:/root/NetHack-engine:.:environments/nethack:skills/explore/src \
      python -m unittest environments.nethack.tests.test_explore_server
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import unittest
from tempfile import TemporaryDirectory

HERE = pathlib.Path(__file__).resolve().parent
ENV_DIR = HERE.parent
REPO = ENV_DIR.parents[1]
sys.path.insert(0, str(ENV_DIR))
sys.path.insert(0, str(REPO / "skills" / "explore" / "src"))

try:
    import mcp  # noqa: F401
    import nethack_core  # noqa: F401
    import verifiers.v1  # noqa: F401

    HAVE_STACK = True
except Exception:  # noqa: BLE001 - any missing piece skips the test
    HAVE_STACK = False

if HAVE_STACK:
    import explore_adapter
    from explore.ledger import Ledger


async def call_tool(url: str, name: str, arguments: dict) -> str:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return "\n".join(part.text for part in result.content if getattr(part, "text", None))


async def tool_names(url: str) -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            return [tool.name for tool in listed.tools]


def pick(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name == suffix or name.endswith("_" + suffix) or name.endswith("." + suffix)]
    if not matches:
        raise AssertionError(f"no tool ending in {suffix!r} among {names}")
    return matches[0]


@unittest.skipUnless(HAVE_STACK, "the eval stack (engine + verifiers + mcp) is not importable")
class ExploreServerTest(unittest.TestCase):
    def test_fresh_save_stop_restore(self) -> None:
        with TemporaryDirectory() as temp:
            run_dir = pathlib.Path(temp)
            (run_dir / "explore.json").write_text(
                json.dumps(
                    {
                        "adapter": "explore_adapter:NetHackAdapter",
                        "python": sys.executable,
                        "pythonpath": [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p],
                        "env_dir": str(ENV_DIR),
                        "seed": 0,
                        "skill_set": "np_core,request_map,search,save",
                        "max_skill_calls": 40,
                        "auto_checkpoint_every_n_calls": 1000,
                        "boot_timeout_s": 240,
                    }
                ),
                encoding="utf-8",
            )
            adapter = explore_adapter.NetHackAdapter()
            ledger = Ledger(run_dir)

            # Attempt 0: a fresh game. Save one checkpoint by hand, then stop.
            fresh = adapter.fresh(attempt=0, run_dir=run_dir)
            servers = fresh["mcp_servers"]
            self.assertEqual(servers[0]["name"], "nethack")
            self.assertTrue(servers[0]["headers"]["Authorization"].startswith("Bearer "))
            url = servers[0]["url"]
            names = asyncio.run(tool_names(url))
            observation = asyncio.run(call_tool(url, pick(names, "request_map"), {}))
            self.assertIn("[call#", observation)
            # The server publishes the live game state beside its ready file.
            import time as _time

            state_path = run_dir / "servers" / "attempt-0.state.json"
            for _ in range(20):
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if state.get("skill_calls", 0) >= 1:
                    break
                _time.sleep(0.5)
            self.assertGreaterEqual(state["skill_calls"], 1, state)
            self.assertFalse(state["budget_exhausted"])
            saved = asyncio.run(call_tool(url, pick(names, "save"), {"label": "smoke", "note": "before the stop"}))
            self.assertNotIn("Error", saved[:200], saved)
            # The mirror runs every 2 s inside the server; finalize flushes it.
            self.assertTrue(adapter.finalize(run_dir, 0, reason="smoke stop"))
            checkpoints = ledger.checkpoints()
            self.assertEqual(len(checkpoints), 1, ledger.records())
            saved_checkpoint = checkpoints[0]
            self.assertEqual(saved_checkpoint.attempt, 0)
            self.assertEqual(saved_checkpoint.label, "smoke")
            self.assertIn("dlvl", saved_checkpoint.score)
            self.assertIsNotNone(saved_checkpoint.call)
            handle = saved_checkpoint.env_handle
            self.assertTrue(pathlib.Path(handle["checkpoint_dir"]).joinpath("state.bundle").is_file())
            outcomes = ledger.outcomes()
            self.assertEqual([(o.attempt, o.result, o.reason) for o in outcomes], [(0, "censored", "smoke stop")])

            # Attempt 1: restored from that checkpoint, on its own server.
            restored = adapter.restore(handle, attempt=1, run_dir=run_dir)
            url2 = restored["mcp_servers"][0]["url"]
            self.assertNotEqual(url2, url, "each attempt has its own game server")
            names2 = asyncio.run(tool_names(url2))
            observation2 = asyncio.run(call_tool(url2, pick(names2, "request_map"), {}))
            self.assertIn("Dlvl", observation2)
            # settle(): a quiet game is declared over and its outcome recorded.
            state = adapter.settle(attempt=1, run_dir=run_dir, quiet_s=1.0, timeout_s=60.0)
            self.assertIn("skill_calls", state)
            outcomes = ledger.outcomes()
            self.assertEqual(outcomes[-1].attempt, 1)
            self.assertIn("no game call", outcomes[-1].reason)
            self.assertEqual(outcomes[-1].checkpoint_from, saved_checkpoint.id)
            self.assertEqual(restored["env_handle"]["id"], saved_checkpoint.id)


if __name__ == "__main__":
    unittest.main()
