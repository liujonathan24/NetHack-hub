"""Gate 3: the explore machine runs end to end without an LLM.

The upstream factory executor drives the real MACHINE.md with a fake host
(upstream's test double) whose children are scripted: the opener and the
players answer like players, and each orchestrator answer is computed from
the ledger with ``explore.select`` the way the real orchestrator would. The
fake host also writes a synthetic session file per child, so the session
cut and the fork chain are checked for real.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
RUNTIME = Path(os.environ.get("PRIME_AGENT_RUNTIME", "/root/nld/pa-upstream/prime-agent-runtime"))
HAVE_RUNTIME = (RUNTIME / "src" / "rlm" / "factory.py").is_file() and (RUNTIME / "test" / "test_factory.py").is_file()
if HAVE_RUNTIME:
    sys.path.insert(0, str(RUNTIME / "src"))
    sys.path.insert(0, str(RUNTIME))
    import rlm as rlm_module  # noqa: E402
    from rlm import factory as factory_module  # noqa: E402
    from rlm.factory import FactoryExecutor  # noqa: E402
    from rlm.harness import HarnessState  # noqa: E402
    from test import test_factory as tf  # noqa: E402

import explore  # noqa: E402
from explore.ledger import Ledger  # noqa: E402

MACHINES_DIR = HERE.parent / "machines"


def write_session(path: Path, *, calls: range) -> dict[int, str]:
    """A player session with one tool result per call; returns call -> row id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "session", "version": 3, "id": path.stem, "cwd": str(path.parent)})]
    ids: dict[int, str] = {}
    parent = None
    index = 0
    for call in calls:
        for row in (
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": f"move {call}"}], "timestamp": 1000 * call}},
            {"type": "message", "message": {"role": "toolResult", "content": [{"type": "text", "text": f"[call#{call}] observation"}], "timestamp": 1000 * call + 1}},
        ):
            row = {"id": f"r{index}", "parentId": parent, **row}
            parent = row["id"]
            index += 1
            if row["message"]["role"] == "toolResult":
                ids[call] = row["id"]
            lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids


if HAVE_RUNTIME:

    class ExploreHost(tf.FakeHost):
        """The fake host with an environment behind it.

        Spawn replies carry a real session file. When a player child is
        collected, the "environment" writes one checkpoint (at call 3 of
        that player's five calls) and the attempt's outcome, then the
        player answers its summary. When an orchestrator child is
        collected, its answer is computed from the ledger: select the
        newest checkpoint with a directive, or stop after ``rounds``
        selections.
        """

        def __init__(self, *, run_dir: Path, rounds: int) -> None:
            super().__init__()
            self.run_dir = run_dir
            self.rounds = rounds
            self.selections = 0
            self.session_ids: dict[str, dict[int, str]] = {}
            self.attempt_of_child: dict[str, int] = {}

        def _node(self, name: str) -> str:
            return tf._node_of_name(name)

        async def __call__(self, request_type: str, payload=None):
            payload = payload or {}
            if request_type == "rlm.run":
                reply = await super().__call__(request_type, payload)
                child = reply["rlm_child_id"]
                node = self._node(payload["kwargs"]["name"])
                session = self.run_dir / "sessions" / f"{child}.jsonl"
                ledger = Ledger(self.run_dir)
                if node in ("explore-opener", "explore-player"):
                    attempt = ledger.attempts_started() if node == "explore-player" else 0
                    self.attempt_of_child[child] = attempt
                    self.session_ids[child] = write_session(session, calls=range(1, 6))
                    # The root's watcher would record this from the status rows.
                    ledger.add_player(attempt=attempt, child=child, session_file=str(session))
                else:
                    write_session(session, calls=range(0, 0))
                reply["session_file"] = str(session)
                return reply
            if request_type == "rlm.collect":
                for target in payload.get("targets") or []:
                    child = self.children.get(target)
                    if child is None or target in self.child_outcomes:
                        continue
                    node = child["node"]
                    if node in ("explore-opener", "explore-player"):
                        attempt = self.attempt_of_child[target]
                        ledger = Ledger(self.run_dir)
                        explore.checkpoint(
                            attempt=attempt,
                            score={"dlvl": attempt + 1, "xl": 1},
                            env_handle={"bundle": f"a{attempt}"},
                            call=3,
                            parent=None,
                            label=f"attempt {attempt} save",
                            run_dir=self.run_dir,
                        )
                        explore.outcome(attempt=attempt, result="died", checkpoint_from=None, run_dir=self.run_dir)
                        self.child_outcomes[target] = {
                            "status": "done",
                            "answer": '```json\n{"summary": {"outcome": "died", "note": "ant"}}\n```',
                        }
                    elif node == "explore-orchestrator":
                        if self.selections >= self.rounds:
                            result = explore.stop("enough rounds")
                        else:
                            newest = Ledger(self.run_dir).checkpoints()[-1]
                            result = explore.select(
                                newest.id,
                                f"Go deeper from {newest.id}." if self.selections else 'Go deeper from c0. Avoid the "mines" door.',
                                run_dir=self.run_dir,
                                adapter=explore.ScriptedAdapter(),
                            )
                            self.selections += 1
                        self.child_outcomes[target] = {"status": "done", "answer": "prose first\n" + explore.answer(result)}
            return await super().__call__(request_type, payload)


@unittest.skipUnless(HAVE_RUNTIME, "upstream runtime and its test double are not available")
class ExploreMachineRunTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.run_dir = root / "run"
        self.run_dir.mkdir()
        agent_dir = root / "agent"
        agent_dir.mkdir()
        (agent_dir / "settings.json").write_text(json.dumps({"factory": {"enabled": True}}), encoding="utf-8")
        self.env = patch.dict(
            os.environ,
            {
                "PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir),
                "PRIME_AGENT_MACHINES_DIR": str(MACHINES_DIR),
                "EXPLORE_RUN_DIR": str(self.run_dir),
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.harness = HarnessState(root / "harness_state.json")
        self.clock = tf.FakeClock()
        self.host = ExploreHost(run_dir=self.run_dir, rounds=2)
        self.executor = FactoryExecutor(now=self.clock, sleep=tf.ClockSleep(self.clock), harness=self.harness)
        previous = factory_module._DEFAULT_EXECUTOR
        factory_module._DEFAULT_EXECUTOR = self.executor
        self.addCleanup(lambda: setattr(factory_module, "_DEFAULT_EXECUTOR", previous))
        patcher = patch.object(rlm_module, "host_request", self.host)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def settle(self, run_id: str) -> dict:
        for _ in range(50_000):
            status = await rlm_module.rlm.factory.status(run_id)
            if status["state"] not in ("running", "stopping"):
                return status
            await asyncio.sleep(0)
        raise AssertionError("the run never settled")

    def test_two_rounds_then_stop(self) -> None:
        async def body() -> None:
            result = await rlm_module.rlm.factory.run("explore")
            status = await self.settle(result["run_id"])
            self.assertEqual(status["state"], "done", status.get("pause_reason"))
            nodes = {node["id"]: node for node in status["nodes"]}
            # opener once; orchestrator three times (select, select, stop); player twice.
            self.assertEqual(nodes["opener"]["entries_used"], 1)
            self.assertEqual(nodes["orchestrator"]["entries_used"], 3)
            self.assertEqual(nodes["player"]["entries_used"], 2)
            fired = [(e["from"], e["to"]) for e in status["events"] if e["kind"] == "transition_fired"]
            self.assertEqual(
                fired,
                [
                    ("opener", "orchestrator"),
                    ("orchestrator", "player"),
                    ("player", "orchestrator"),
                    ("orchestrator", "player"),
                    ("player", "orchestrator"),
                ],
            )
            # The orchestrator forks ITSELF from round 2 on.
            orch = self.host.spawn_calls("explore-orchestrator")
            self.assertEqual(len(orch), 3)
            self.assertNotIn("fork_from", orch[0]["kwargs"])
            self.assertEqual(orch[1]["kwargs"]["fork_from"], {"session_file": str(self.run_dir / "sessions" / "child-2.jsonl")})
            self.assertEqual(orch[2]["kwargs"]["fork_from"], {"session_file": str(self.run_dir / "sessions" / "child-4.jsonl")})
            # Each player forks the checkpoint's player at the call-3 row and
            # gets the directive block in its prompt.
            players = self.host.spawn_calls("explore-player")
            self.assertEqual(len(players), 2)
            opener_ids = self.host.session_ids["child-1"]
            # Players fork one-time handoff copies (handoff/attempt-<N>.jsonl)
            # of the checkpoint's player conversation.
            self.assertEqual(
                players[0]["kwargs"]["fork_from"],
                {"session_file": str(self.run_dir / "handoff" / "attempt-1-1.jsonl"), "entry_id": opener_ids[3]},
            )
            player1_ids = self.host.session_ids["child-3"]
            self.assertEqual(
                players[1]["kwargs"]["fork_from"],
                {"session_file": str(self.run_dir / "handoff" / "attempt-2-1.jsonl"), "entry_id": player1_ids[3]},
            )
            self.assertEqual(players[0]["kwargs"]["mcp_servers"], [])
            # The directive reaches the player verbatim: no quotes, no escapes.
            self.assertTrue(
                players[0]["prompt"].startswith('[Strategy for this attempt: Go deeper from c0. Avoid the "mines" door.]\n'),
                players[0]["prompt"][:160],
            )
            # A player sees nothing but the directive and the resume note: no
            # leaderboard, no other attempt's numbers, no path, no token.
            openers = self.host.spawn_calls("explore-opener")
            self.assertEqual(len(openers), 1)
            for opener in openers:
                for word in ("orchestrator", "other agents", "parent"):
                    self.assertNotIn(word, opener["prompt"].lower())
            for prompt in (p["prompt"] for p in players):
                self.assertNotIn("## Inputs", prompt)
                self.assertNotIn("session_ref", prompt)
                # Nor anything that says other agents exist (the user's rule:
                # a child does not know it is one, or that others exist).
                for word in ("checkpoint", "orchestrator", "other agents", "parent"):
                    self.assertNotIn(word, prompt.lower())
                self.assertNotIn(str(self.run_dir), prompt)
                self.assertNotIn("Dlvl", prompt)
            # The ledger: three attempts, three checkpoints, two selections.
            ledger = Ledger(self.run_dir)
            self.assertEqual(ledger.attempts_started(), 3)
            self.assertEqual([c.id for c in ledger.checkpoints()], ["c0", "c1", "c2"])
            selections = [r for r in ledger.records() if r.get("kind") == "selection"]
            self.assertEqual([s["checkpoint"] for s in selections], ["c0", "c1"])
            self.assertTrue(all(s["memory_ok"] for s in selections))

        asyncio.run(body())


class WatchPlayersTest(unittest.TestCase):
    def test_watch_records_each_player_once(self) -> None:
        snapshots = [
            {
                "state": "running",
                "changed": True,
                "nodes": [
                    {"id": "opener", "instances": [{"index": 0, "child": "sub-a", "session_file": "/s/a.jsonl"}]},
                    {"id": "orchestrator", "instances": [{"index": 0, "child": "sub-o", "session_file": "/s/o.jsonl"}]},
                ],
            },
            {
                "state": "done",
                "changed": True,
                "nodes": [
                    {"id": "opener", "instances": [{"index": 0, "child": "sub-a", "session_file": "/s/a.jsonl"}]},
                    {"id": "player", "instances": [{"index": 0, "child": "sub-p", "session_file": "/s/p.jsonl"}]},
                ],
            },
        ]

        class FakeFactory:
            async def watch(self, run_id, timeout):
                return snapshots.pop(0)

        class FakeRlm:
            factory = FakeFactory()

        with TemporaryDirectory() as temp:
            import types

            fake_module = types.SimpleNamespace(rlm=FakeRlm())
            with patch.dict(sys.modules, {"rlm": fake_module}):
                asyncio.run(explore.watch_players("run-1", run_dir=temp, poll_seconds=0))
            players = Ledger(temp).players()
            self.assertEqual([(p.attempt, p.child, p.session_file) for p in players], [(0, "sub-a", "/s/a.jsonl"), (1, "sub-p", "/s/p.jsonl")])

    def test_watch_players_resumes_a_pause_and_reports_the_end_in_service_mode(self) -> None:
        # The root ends its turn after starting the run, so its kernel's
        # watcher resumes a paused run and reports the end to the service.
        snapshots = [{"state": "paused"}, {"state": "running"}, {"state": "done"}]
        calls: list[tuple] = []

        class FakeFactory:
            async def watch(self, run_id, timeout):
                return snapshots.pop(0)

            async def resume(self, run_id):
                calls.append(("resume", run_id))

            async def status(self, run_id):
                return {"state": "done", "usage": {"turns": 3}, "events": [{"seq": 1}]}

        class FakeRlm:
            factory = FakeFactory()

        import types

        fake_module = types.SimpleNamespace(rlm=FakeRlm())
        remote = lambda call, **kwargs: calls.append((call, kwargs))  # noqa: E731
        with patch.dict(sys.modules, {"rlm": fake_module}), patch.dict(
            os.environ, {"EXPLORE_SERVICE_URL": "http://127.0.0.1:9"}
        ), patch.object(explore, "_remote", remote):
            asyncio.run(explore.watch_players("run-1", poll_seconds=0))
        self.assertEqual(
            calls,
            [
                ("resume", "run-1"),
                (
                    "run_ended",
                    {
                        "run_id": "run-1",
                        "state": "done",
                        "resumes": 1,
                        "pause_reason": None,
                        "usage": {"turns": 3},
                        "events": [{"seq": 1}],
                    },
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
