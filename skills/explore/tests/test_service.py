"""The explore service authorizes by the daemon's spawn ledger, not a token."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import explore  # noqa: E402
from explore import service  # noqa: E402


class ServiceRolesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.run = base / "run"
        self.agent = base / "agent"
        (self.run / "handoff").mkdir(parents=True)
        (self.agent / "rlm-ledger").mkdir(parents=True)
        self.root_file = str(self.agent / "sessions" / "root-0.jsonl")
        (self.run / "explore.json").write_text(json.dumps({"agent_dir": str(self.agent)}), encoding="utf-8")
        (self.run / "root_session.json").write_text(json.dumps({"sessionFile": self.root_file}), encoding="utf-8")
        art = self.agent / "session-artifacts" / "root-0"
        self.files = {}
        records = []
        for child, sid, name, parent, depth in [
            ("sub-1", "s-opener", "ab-explore-opener", self.root_file, 1),
            ("sub-2", "s-orch", "ab-explore-orchestrator", self.root_file, 1),
            ("sub-3", "s-player", "ab-explore-player", self.root_file, 1),
            # The factory's retry of a round that bound no decision.
            ("sub-5", "s-orch-retry", "ab-explore-orchestrator-a2", self.root_file, 1),
            # Later rounds (re-entries), and a retry of one.
            ("sub-6", "s-orch-r2", "ab-explore-orchestrator-i1", self.root_file, 1),
            ("sub-7", "s-orch-r2-retry", "ab-explore-orchestrator-i1-a2", self.root_file, 1),
            # A player's own spawn, named to look like the orchestrator.
            ("sub-4", "s-fake", "ab-explore-orchestrator", str(art / "sub-3" / "s-player.jsonl"), 2),
        ]:
            f = art / child / f"{sid}.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            self.files[sid] = f
            records.append({"op": "spawn", "childId": child, "parent": parent, "child": str(f), "depth": depth, "name": name})
        self.files["s-opener"].write_text(json.dumps({"type": "session"}) + "\n", encoding="utf-8")
        self.files["s-player"].write_text(
            json.dumps({"type": "session", "parentSession": str(self.run / "handoff" / "attempt-1-2.jsonl")}) + "\n", encoding="utf-8"
        )
        (self.agent / "rlm-ledger" / "x.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        self.session_dir = lambda sid: str(art / "session-artifacts" / sid)
        service.ROLE_WAIT_S = 0.0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_roles_come_from_the_daemon_spawn_ledger(self) -> None:
        self.assertEqual(service.caller_role(self.run, self.session_dir("s-orch")), "orchestrator")
        for sid in ("s-orch-retry", "s-orch-r2", "s-orch-r2-retry"):
            self.assertEqual(service.caller_role(self.run, self.session_dir(sid)), "orchestrator")
        self.assertEqual(service.caller_role(self.run, self.session_dir("s-player")), "player")
        self.assertIsNone(service.caller_role(self.run, self.session_dir("s-fake")))  # depth 2, not the root's child
        self.assertIsNone(service.caller_role(self.run, self.session_dir("nobody")))
        self.assertIsNone(service.caller_role(self.run, ""))

    def test_only_the_orchestrator_may_call(self) -> None:
        for caller in ("s-player", "s-fake", "nobody"):
            with self.assertRaises(explore.NotTheOrchestrator):
                service.dispatch(self.run, "leaderboard", {}, self.session_dir(caller))
        text = service.dispatch(self.run, "leaderboard", {}, self.session_dir("s-orch"))
        self.assertIsInstance(text, str)

    def test_players_map_to_attempts_by_their_handoff(self) -> None:
        service.sync_players(self.run)
        ledger = explore.Ledger(self.run)
        self.assertEqual(ledger.player_session(0), str(self.files["s-opener"]))
        self.assertEqual(ledger.player_session(1), str(self.files["s-player"]))


if __name__ == "__main__":
    unittest.main()
