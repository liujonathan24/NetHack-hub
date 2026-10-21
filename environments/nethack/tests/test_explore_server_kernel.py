"""The game server stores, with each checkpoint it mirrors, the kernel its
player had BEFORE the checkpoint (explore.checkpoint_kernel). No engine
boot: the archive and the daemon's spawn ledger are written by hand.

    PYTHONPATH=tools/pycompat:/root/NetHack-engine:.:environments/nethack:skills/explore/src \
      python -m unittest environments.nethack.tests.test_explore_server_kernel
"""

from __future__ import annotations

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
    import explore_server

    HAVE_STACK = True
except Exception:  # noqa: BLE001 - any missing piece skips the test
    HAVE_STACK = False


def write_snapshot(artifact_dir: pathlib.Path, payload: bytes, at: float) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "kernel-state.dill").write_bytes(payload)
    (artifact_dir / "kernel-state.json").write_text("{}")
    for name in ("kernel-state.dill", "kernel-state.json"):
        os.utime(artifact_dir / name, (at, at))


@unittest.skipUnless(HAVE_STACK, "explore_server (the eval stack) is not importable")
class CheckpointKernelTest(unittest.TestCase):
    def test_a_mirrored_checkpoint_keeps_the_kernel_from_before_it(self) -> None:
        from explore.ledger import Ledger
        from nethack_harness.explore_checkpoints import CHECKPOINT_FILE, META_FILE

        with TemporaryDirectory() as temp:
            run = pathlib.Path(temp)
            agent = run / "agent"
            root = agent / "sessions" / "root.jsonl"
            opener = agent / "session-artifacts" / "root" / "sub-op" / "op-1.jsonl"
            opener.parent.mkdir(parents=True)
            opener.write_text(json.dumps({"type": "session", "id": "op-1", "parentSession": str(root)}) + "\n")
            (agent / "rlm-ledger").mkdir(parents=True)
            (agent / "rlm-ledger" / "x.jsonl").write_text(
                json.dumps({"op": "spawn", "depth": 1, "parent": str(root), "name": "ab12cd-explore-opener", "child": str(opener)}) + "\n"
            )
            (run / "explore.json").write_text(json.dumps({"agent_dir": str(agent)}))
            (run / "root_session.json").write_text(json.dumps({"sessionFile": str(root)}))
            live = agent / "session-artifacts" / "root" / "session-artifacts" / "op-1"
            archive = run / "archive"
            server = explore_server.Server(
                explore_server.parse_args(
                    ["--run-dir", str(run), "--attempt", "0", "--archive", str(archive), "--ready", str(run / "servers" / "attempt-0.json")]
                )
            )
            server.started_at = 0.0
            write_snapshot(live, b"after cell 1", 1_000)
            server.watch_kernel()
            # Cell 2 wrote checkpoint c3 at t=2000, and its own snapshot
            # (t=3000) landed before the archive was mirrored.
            write_snapshot(live, b"after cell 2", 3_000)
            checkpoint = archive / "c3"
            checkpoint.mkdir(parents=True)
            (checkpoint / CHECKPOINT_FILE).write_bytes(b"x")
            (checkpoint / META_FILE).write_text(json.dumps({"id": "c3", "created_at": 2_000.0, "call": 20, "dlvl": 2}))
            server.mirror_archive()
            self.assertEqual([c.id for c in Ledger(run).checkpoints()], ["c3"])
            stored = run / "checkpoint-kernels" / "c3"
            self.assertEqual((stored / "kernel-state.dill").read_bytes(), b"after cell 1")
            self.assertEqual(json.loads((stored / "about.json").read_text())["session_file"], str(opener))


if __name__ == "__main__":
    unittest.main()
