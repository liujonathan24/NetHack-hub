"""The explore skill: ledger integrity, the session cut, select(), the fixed
rule, and the explore machine against the upstream factory validator."""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import explore  # noqa: E402
from explore.ledger import Ledger, clean_score  # noqa: E402
from explore.session_cut import cut_session  # noqa: E402

#: The upstream runtime (for the machine validator). Overridable; the
#: machine test skips when it is absent.
RUNTIME_SRC = Path(os.environ.get("PRIME_AGENT_RUNTIME_SRC", "/root/nld/pa-upstream/prime-agent-runtime/src"))


def session_file(directory: Path, rows: list[dict]) -> Path:
    """A session file in the daemon's JSONL shape: a header, then rows with
    ids and parentIds."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "player.jsonl"
    lines = [json.dumps({"type": "session", "version": 3, "id": "sess-1", "cwd": str(directory)})]
    parent = None
    for index, row in enumerate(rows):
        row = {"id": f"r{index}", "parentId": parent, **row}
        parent = row["id"]
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def tool_result(text: str, *, ms: int) -> dict:
    return {
        "type": "message",
        "message": {"role": "toolResult", "content": [{"type": "text", "text": text}], "timestamp": ms},
    }


def assistant(text: str, *, ms: int) -> dict:
    return {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": text}], "timestamp": ms}}


class LedgerTest(unittest.TestCase):
    def test_numbers_only_cross_into_scores(self) -> None:
        # Integrity rule 1: a model cannot write a number into the record.
        self.assertEqual(clean_score({"dlvl": 5, "hp": 3.5, "name": "Dlvl 30", "ok": True, 7: 1}), {"dlvl": 5, "hp": 3.5})
        self.assertEqual(clean_score("dlvl: 99"), {})

    def test_round_trip_and_leaderboard(self) -> None:
        with TemporaryDirectory() as temp:
            ledger = Ledger(temp)
            c0 = ledger.add_checkpoint(attempt=0, score={"dlvl": 1, "xl": 1}, env_handle="h0", call=20, parent=None, label="start", now=1.0)
            c1 = ledger.add_checkpoint(attempt=0, score={"dlvl": 3, "xl": 2, "note": "deep"}, env_handle="h1", call=40, parent="c0", now=2.0)
            ledger.add_player(attempt=0, child="sub-a", session_file="/tmp/a.jsonl", now=0.5)
            ledger.add_outcome(attempt=0, result="died", checkpoint_from=None, reason="ant", now=3.0)
            ledger.add_player(attempt=1, child="sub-b", session_file="/tmp/b.jsonl", now=3.5)
            ledger.add_outcome(attempt=1, result="censored", checkpoint_from="c1", reason="wall clock", now=4.0)
            self.assertEqual([c.id for c in ledger.checkpoints()], ["c0", "c1"])
            self.assertEqual(ledger.checkpoint("c1").score, {"dlvl": 3, "xl": 2})
            self.assertEqual(ledger.player_session(0), "/tmp/a.jsonl")
            self.assertEqual(ledger.attempts_started(), 2)
            self.assertEqual(ledger.attempts_from("c1"), 1)
            self.assertEqual(ledger.next_checkpoint_id(), "c2")
            text = explore.leaderboard(run_dir=temp)
            self.assertIn("c1 | c0 | 0 | 3 | 2 | 1 | censored", text)
            self.assertIn("2 checkpoints . 2 attempts started . 2 finished", text)
            with self.assertRaises(ValueError):
                ledger.add_outcome(attempt=2, result="lost", checkpoint_from=None)
            self.assertEqual(c0.id, "c0")
            self.assertEqual(c1.parent, "c0")


class SessionCutTest(unittest.TestCase):
    def test_marker_cut_after_the_call(self) -> None:
        with TemporaryDirectory() as temp:
            path = session_file(
                Path(temp),
                [
                    assistant("look", ms=1000),
                    tool_result("[call#1] you see a room", ms=1100),
                    assistant("move", ms=2000),
                    tool_result("[call#2] you moved", ms=2100),
                    assistant("descend", ms=3000),
                    tool_result("[call#3] you are on Dlvl 2", ms=3100),
                ],
            )
            cut = cut_session(path, 2)
            self.assertTrue(cut.ok, cut.reason)
            self.assertEqual((cut.entry_id, cut.method, cut.position, cut.lag_calls), ("r3", "marker", "after_call", 0))
            self.assertEqual(cut.fork_source(), {"session_file": str(path), "entry_id": "r3"})

    def test_batch_cell_cuts_before_it_and_reports_the_lag(self) -> None:
        # One cell ran calls 4..6: the save happened at 5, so the transcript
        # must stop at the previous result (call 3) and lag by 2 calls.
        with TemporaryDirectory() as temp:
            path = session_file(
                Path(temp),
                [
                    tool_result("[call#3] room", ms=1000),
                    tool_result("[call#4] a\n[call#5] b\n[call#6] c", ms=2000),
                ],
            )
            cut = cut_session(path, 5)
            self.assertTrue(cut.ok, cut.reason)
            self.assertEqual((cut.entry_id, cut.position, cut.transcript_end_call, cut.lag_calls), ("r0", "before_batch", 3, 2))
            # Call 4 sits in the same batch: the same cut, lagging by one.
            fourth = cut_session(path, 4)
            self.assertEqual((fourth.ok, fourth.entry_id, fourth.lag_calls), (True, "r0", 1))
            # A batch that is the FIRST cell has no earlier exchange to cut at.
            first_batch = session_file(Path(temp) / "first", [tool_result("[call#1] a\n[call#2] b", ms=1000)])
            first = cut_session(first_batch, 1)
            self.assertFalse(first.ok)
            self.assertIn("no earlier exchange", first.reason)

    def test_timestamp_fallback_and_failures(self) -> None:
        with TemporaryDirectory() as temp:
            path = session_file(Path(temp), [tool_result("quiet", ms=5000), tool_result("still quiet", ms=9000)])
            cut = cut_session(path, 7, call_times={7: 6.0})
            self.assertTrue(cut.ok, cut.reason)
            self.assertEqual((cut.entry_id, cut.method), ("r1", "timestamp"))
            self.assertFalse(cut_session(path, 7).ok)
            self.assertFalse(cut_session(path, None).ok)
            self.assertFalse(cut_session(Path(temp) / "ghost.jsonl", 1).ok)


class SelectTest(unittest.TestCase):
    def test_select_builds_the_decision_and_directive_ports(self) -> None:
        with TemporaryDirectory() as temp:
            run = Path(temp)
            path = session_file(run, [tool_result("[call#9] saved here", ms=1000), tool_result("[call#10] later", ms=2000)])
            ledger = Ledger(run)
            ledger.add_player(attempt=0, child="sub-a", session_file=str(path))
            ledger.add_checkpoint(attempt=0, score={"dlvl": 4}, env_handle={"bundle": "c0"}, call=9, parent=None)
            ledger.add_outcome(attempt=0, result="died", checkpoint_from=None)
            result = explore.select("c0", "Find the downstairs before you fight.", run_dir=run, adapter=explore.ScriptedAdapter())
            nxt = result["decision"]["next"]
            self.assertEqual(nxt["checkpoint"], "c0")
            self.assertEqual(nxt["attempt"], 1)
            # The player forks a one-time handoff COPY of the conversation.
            handoff = Path(nxt["session_ref"]["session_file"])
            self.assertEqual(nxt["session_ref"]["entry_id"], "r0")
            self.assertEqual((handoff.parent.name, handoff.name), ("handoff", f"attempt-{nxt['attempt']}-1.jsonl"))
            self.assertEqual(handoff.read_bytes(), path.read_bytes())
            self.assertEqual(nxt["mcp_servers"], [])
            selection = [r for r in ledger.records() if r.get("kind") == "selection"][-1]
            self.assertEqual(selection["directive"], "Find the downstairs before you fight.")
            self.assertTrue(selection["memory"]["ok"])
            self.assertNotIn("memory", nxt)
            self.assertNotIn("directive", nxt)
            self.assertNotIn("env_handle", nxt)
            self.assertEqual(selection_env := [r for r in ledger.records() if r.get("kind") == "selection"][-1]["env_handle"], {"bundle": "c0"})
            self.assertTrue(json.loads(explore.answer(result).strip("`\n").removeprefix("json\n"))["decision"]["next"]["session_ref"])
            self.assertLess(len(explore.answer(result)), 900)
            self.assertEqual(result["directive"], "[ORCHESTRATOR DIRECTIVE for this attempt: Find the downstairs before you fight.]")
            self.assertTrue(explore.answer(result).startswith("```json\n"))
            self.assertEqual([r["kind"] for r in ledger.records()][-1], "selection")
            calls = (run / "adapter_calls.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(calls[-1])["op"], "restore")
            # Blind arms get the neutral directive; a bad directive is refused.
            blind = explore.select("c0", "ignored", blind=True, run_dir=run, adapter=explore.ScriptedAdapter())
            self.assertEqual(blind["directive"], "[ORCHESTRATOR DIRECTIVE for this attempt: Continue playing.]")
            self.assertEqual([r for r in ledger.records() if r.get("kind") == "selection"][-1]["directive"], "Continue playing.")
            with self.assertRaisesRegex(ValueError, "names a key"):
                explore.select("c0", "Press > to descend.", run_dir=run, adapter=explore.ScriptedAdapter())
            with self.assertRaisesRegex(ValueError, "unknown checkpoint"):
                explore.select("c9", "go", run_dir=run, adapter=explore.ScriptedAdapter())
            # No recorded player: the player starts with fresh memory, and says why.
            ledger.add_checkpoint(attempt=3, score={"dlvl": 1}, env_handle="h", call=1, parent=None)
            fresh = explore.select("c1", "Explore.", run_dir=run, adapter=explore.ScriptedAdapter())
            self.assertIsNone(fresh["decision"]["next"]["session_ref"])
            selection = [r for r in ledger.records() if r.get("kind") == "selection"][-1]
            self.assertIn("no player session recorded", selection["memory"]["reason"])

    def test_the_handoff_carries_the_source_players_kernel_snapshot(self) -> None:
        with TemporaryDirectory() as temp:
            run = Path(temp)
            # The daemon's layout: agent/sessions/<file>, agent/session-artifacts/<header id>/.
            path = session_file(run / "agent" / "sessions", [tool_result("[call#9] saved here", ms=1000)])
            source_dir = run / "agent" / "session-artifacts" / "sess-1"
            source_dir.mkdir(parents=True)
            for name, data in (("kernel-state.dill", b"\x80names"), ("kernel-state.json", b"{}")):
                (source_dir / name).write_bytes(data)
                os.utime(source_dir / name, (1_000, 1_000))
            ledger = Ledger(run)
            ledger.add_player(attempt=0, child="sub-a", session_file=str(path))
            ledger.add_checkpoint(attempt=0, score={"dlvl": 4}, env_handle={"bundle": "c0"}, call=9, parent=None)
            ledger.add_outcome(attempt=0, result="died", checkpoint_from=None)
            nxt = explore.select("c0", "Go down.", run_dir=run, adapter=explore.ScriptedAdapter())["decision"]["next"]
            handoff = Path(nxt["session_ref"]["session_file"])
            # Prime Agent looks for the copy's snapshot by the copy's location.
            copy_dir = explore.session_artifact_dir(handoff)
            self.assertEqual(copy_dir, run / "session-artifacts" / "sess-1")
            self.assertEqual((copy_dir / "kernel-state.dill").read_bytes(), b"\x80names")
            # File times are kept, so the daemon's freshness check stays honest.
            self.assertEqual((copy_dir / "kernel-state.json").stat().st_mtime, 1_000)
            self.assertTrue([r for r in ledger.records() if r.get("kind") == "selection"][-1]["kernel_snapshot"])
            # A source with no snapshot hands off none.
            shutil.rmtree(copy_dir)
            shutil.rmtree(source_dir)
            explore.select("c0", "Go down.", run_dir=run, adapter=explore.ScriptedAdapter())
            self.assertFalse(copy_dir.exists())
            self.assertFalse([r for r in ledger.records() if r.get("kind") == "selection"][-1]["kernel_snapshot"])
            self.assertEqual(explore.stop("done")["decision"], {"stop": True, "reason": "done"})
            self.assertNotIn("next", explore.stop()["decision"])

    def test_a_later_select_supersedes_the_earlier_one(self) -> None:
        class Recording(explore.ScriptedAdapter):
            def __init__(self):
                super().__init__()
                self.finalized = []

            def finalize(self, run_dir, attempt, *, reason):
                self.finalized.append((attempt, reason))
                return True

        with TemporaryDirectory() as temp:
            run = Path(temp)
            session = run / "opener.jsonl"
            session.write_text(json.dumps({"type": "session", "id": "s"}) + "\n" + json.dumps(
                {"type": "message", "id": "r0", "message": {"role": "toolResult", "content": [{"type": "text", "text": "[call#1]"}]}}
            ) + "\n", encoding="utf-8")
            ledger = Ledger(run)
            ledger.add_player(attempt=0, child="c", session_file=str(session))
            ledger.add_checkpoint(attempt=0, score={"dlvl": 1}, env_handle="h", call=1, parent=None)
            adapter = Recording()
            first = explore.select("c0", "Explore.", run_dir=run, adapter=adapter)
            second = explore.select("c0", "Descend.", run_dir=run, adapter=adapter)
            self.assertEqual(first["decision"]["next"]["attempt"], second["decision"]["next"]["attempt"])
            self.assertEqual(adapter.finalized, [(1, "superseded by a later select")])
            first_handoff = Path(first["decision"]["next"]["session_ref"]["session_file"])
            second_handoff = Path(second["decision"]["next"]["session_ref"]["session_file"])
            self.assertFalse(first_handoff.exists())  # a stale block now fails at spawn
            self.assertTrue(second_handoff.exists())

    def test_no_key_in_any_prompt_or_signature(self) -> None:
        # The explore service authorizes the orchestrator by its session
        # (test_service.py); no call takes a key and no prompt carries one.
        import inspect

        for function in (explore.leaderboard, explore.rows, explore.select, explore.stop):
            self.assertNotIn("token", inspect.signature(function).parameters)
        machine = (HERE.parent / "machines" / "explore" / "MACHINE.md").read_text(encoding="utf-8")
        self.assertNotIn("TOKEN", machine)
        self.assertNotIn("token=", machine)

    def test_leaderboard_and_select_settle_the_previous_attempt_first(self) -> None:
        class RecordingAdapter(explore.ScriptedAdapter):
            settled: list[int] = []

            def settle(self, *, attempt, run_dir):
                self.settled.append(attempt)
                Ledger(run_dir).add_outcome(attempt=attempt, result="budget", checkpoint_from=None, reason="settled by adapter")
                return {"budget_exhausted": True}

        with TemporaryDirectory() as temp:
            run = Path(temp)
            ledger = Ledger(run)
            ledger.add_player(attempt=0, child="sub-a", session_file=None)
            ledger.add_checkpoint(attempt=0, score={"dlvl": 2}, env_handle="h", call=5, parent=None)
            adapter = RecordingAdapter()
            text = explore.leaderboard(run_dir=run, adapter=adapter)
            self.assertEqual(adapter.settled, [0])
            self.assertIn("1 finished", text)  # the settled outcome counts as finished
            self.assertEqual([(o.attempt, o.result) for o in ledger.outcomes()], [(0, "budget")])
            # Already settled: no second settle on the next read or on select.
            explore.rows(run_dir=run, adapter=adapter)
            explore.select("c0", "Go.", run_dir=run, adapter=adapter)
            self.assertEqual(adapter.settled, [0])
            self.assertEqual(explore.stop("x")["decision"]["stop"], True)

    def test_fixed_rule_takes_the_previous_attempts_latest_checkpoint(self) -> None:
        with TemporaryDirectory() as temp:
            run = Path(temp)
            self.assertIsNone(explore.select_fixed(run_dir=run))
            ledger = Ledger(run)
            ledger.add_checkpoint(attempt=0, score={}, env_handle=None, call=1, parent=None, now=1.0)
            ledger.add_checkpoint(attempt=0, score={}, env_handle=None, call=2, parent="c0", now=2.0)
            ledger.add_outcome(attempt=0, result="died", checkpoint_from=None)
            self.assertEqual(explore.select_fixed(run_dir=run), "c1")
            ledger.add_outcome(attempt=1, result="died", checkpoint_from="c1")  # wrote nothing
            self.assertEqual(explore.select_fixed(run_dir=run), "c1")

    def test_directive_rules(self) -> None:
        self.assertEqual(explore.validate_directive("Go down. Avoid the mines."), [])
        self.assertIn("3 sentences", explore.validate_directive("A. B. C.")[0])
        self.assertIn("characters", explore.validate_directive("x" * 301)[0])
        self.assertIn("key", explore.validate_directive("Type the letter a to apply.")[0])


@unittest.skipUnless((RUNTIME_SRC / "rlm" / "factory.py").is_file(), "upstream runtime not available")
class MachineTest(unittest.TestCase):
    def test_explore_machine_validates_under_the_upstream_factory(self) -> None:
        sys.path.insert(0, str(RUNTIME_SRC))
        from rlm.factory import parse_machine_file, validate_factory_spec  # noqa: E402

        machine_file = HERE.parent / "machines" / "explore" / "MACHINE.md"
        machine, errors = parse_machine_file(machine_file.read_text(encoding="utf-8"), source=str(machine_file))
        self.assertEqual(errors, [])
        self.assertEqual(machine.name, "explore")
        spec = machine.spec
        self.assertEqual(validate_factory_spec(spec), [])
        states = {state["id"]: state for state in spec["states"]}
        self.assertEqual(states["orchestrator"]["subagent"]["fork_from"], "self")
        self.assertEqual(states["player"]["subagent"]["fork_from"], {"input": "decision", "path": "next.session_ref"})
        self.assertEqual(states["player"]["subagent"]["mcp_servers"], {"input": "decision", "path": "next.mcp_servers"})
        self.assertEqual(
            [(t["from"], t["to"]) for t in spec["transitions"]],
            [("opener", "orchestrator"), ("player", "orchestrator"), ("orchestrator", "player")],
        )


if __name__ == "__main__":
    unittest.main()
