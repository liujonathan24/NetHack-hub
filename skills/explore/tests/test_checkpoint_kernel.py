"""A checkpoint keeps the kernel its player had when it was written: the
game server's snapshot buffer, the per-checkpoint store, and the lookup of
the attempt's player in the daemon's spawn ledger."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from explore.checkpoint_kernel import (  # noqa: E402
    KernelBuffer,
    attempt_player_session,
    checkpoint_kernel_dir,
    store_for_checkpoint,
)


def write_snapshot(artifact_dir: Path, payload: bytes, at: float) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "kernel-state.dill").write_bytes(payload)
    (artifact_dir / "kernel-state.json").write_text(json.dumps({"savedNames": [payload.decode()]}))
    for name in ("kernel-state.dill", "kernel-state.json"):
        os.utime(artifact_dir / name, (at, at))


class KernelBufferTest(unittest.TestCase):
    def test_a_checkpoint_gets_the_newest_snapshot_written_before_it(self) -> None:
        with TemporaryDirectory() as temp:
            run, live = Path(temp), Path(temp) / "player-artifacts"
            buffer = KernelBuffer(run / "servers" / "attempt-1.kernels", keep=3)
            write_snapshot(live, b"after cell 1", 1_000)
            self.assertTrue(buffer.watch(live))
            self.assertFalse(buffer.watch(live))  # unchanged: no new version
            write_snapshot(live, b"after cell 2", 2_000)
            self.assertTrue(buffer.watch(live))
            # The checkpoint at t=1500 was written during cell 2: it gets the
            # kernel after cell 1, never the one after cell 2.
            self.assertTrue(store_for_checkpoint(run, "c4", buffer, 1_500.0, "/s/player.jsonl"))
            stored = checkpoint_kernel_dir(run, "c4")
            self.assertEqual((stored / "kernel-state.dill").read_bytes(), b"after cell 1")
            about = json.loads((stored / "about.json").read_text())
            self.assertEqual((about["ok"], about["snapshot_at"]), (True, 1_000.0))
            # Nothing before the first snapshot: no kernel, and why.
            self.assertFalse(store_for_checkpoint(run, "c1", buffer, 500.0, "/s/player.jsonl"))
            about = json.loads((checkpoint_kernel_dir(run, "c1") / "about.json").read_text())
            self.assertEqual(about["reason"], "no kernel snapshot written before the checkpoint")
            self.assertFalse((checkpoint_kernel_dir(run, "c1") / "kernel-state.dill").exists())
            # Unknown player: no kernel.
            self.assertFalse(store_for_checkpoint(run, "c2", None, 3_000.0, None))
            # Old versions are pruned to `keep`.
            for at in (3_000, 4_000, 5_000):
                write_snapshot(live, f"v{at}".encode(), at)
                buffer.watch(live)
            self.assertEqual([stamp // 1_000_000_000 for stamp, _ in buffer.versions()], [3_000, 4_000, 5_000])


class AttemptPlayerTest(unittest.TestCase):
    def test_the_attempts_player_comes_from_the_daemons_spawn_ledger(self) -> None:
        with TemporaryDirectory() as temp:
            run = Path(temp)
            agent = run / "agent"
            (agent / "rlm-ledger").mkdir(parents=True)
            root = agent / "sessions" / "root.jsonl"
            (run / "explore.json").write_text(json.dumps({"agent_dir": str(agent)}))
            (run / "root_session.json").write_text(json.dumps({"sessionFile": str(root)}))
            kids = agent / "session-artifacts" / "root"

            def child(name: str, parent_session: str | None) -> str:
                path = kids / name / f"{name}.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"type": "session", "id": name, "parentSession": parent_session}) + "\n")
                return str(path)

            opener = child("op", str(root))
            player = child("p1", str(run / "handoff" / "attempt-1-1.jsonl"))
            respawn = child("p1b", str(run / "handoff" / "attempt-1-1.jsonl"))
            orchestrator = child("or", str(root))
            records = [
                ("explore-opener", opener),
                ("explore-orchestrator", orchestrator),
                ("explore-player", player),
                ("explore-player", respawn),
            ]
            (agent / "rlm-ledger" / "x.jsonl").write_text(
                "".join(
                    json.dumps({"op": "spawn", "depth": 1, "parent": str(root), "name": f"ab12cd-{name}", "child": path}) + "\n"
                    for name, path in records
                )
            )
            self.assertEqual(attempt_player_session(run, 0), opener)
            self.assertEqual(attempt_player_session(run, 1), respawn)  # the latest spawn plays
            self.assertIsNone(attempt_player_session(run, 2))


if __name__ == "__main__":
    unittest.main()
