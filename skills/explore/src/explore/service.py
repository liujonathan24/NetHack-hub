"""The explore run's control service: the ledger, archive and game servers,
kept OUTSIDE the agents' sandbox.

Every agent kernel of a sandboxed run sees only its workspace and its own
session directory, so it cannot read the ledger, the archive, another
player's conversation, or our code. The orchestrator still needs
``leaderboard()`` and ``select()``; with ``EXPLORE_SERVICE_URL`` set, the
``explore`` package sends those calls here (``explore._remote``), and this
process runs them against the run directory with the real adapter.

Authorization is the harness's, never a key in a prompt: the client sends
its kernel's private session directory (``RLM_SESSION_DIR``), and this
service answers only the session the DAEMON's spawn ledger records as the
root's ``explore-orchestrator`` child.

Run by the driver::

    python -m explore.service --run-dir R --port-file R/service.json
"""

from __future__ import annotations

import argparse
import json
import re
import os
import pathlib
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

#: The calls a kernel may make, all orchestrator-only. Anything else is refused.
CALLS = ("leaderboard", "rows", "select", "select_fixed")
# Factory child names (factory._child_name): "<run prefix>-explore-<state>",
# then "-i<N>" on a state's re-entry and "-a<N>" on a retry of a round that
# bound no decision, in that order.
ROLE_NAME = re.compile(r"-explore-(orchestrator|player|opener)(?:-i\d+)?(?:-a\d+)?$")
#: How long a caller the spawn ledger does not know yet is waited for.
ROLE_WAIT_S = 30.0

# One call at a time: select() boots and finalizes game servers and appends to
# the ledger; the ledger is append-only but a selection must see the settle of
# the attempt before it.
_LOCK = Lock()


def handoff_attempt(session_file: str | None) -> int | None:
    """The attempt a forked player belongs to: its session header's
    ``parentSession`` is the handoff copy ``handoff/attempt-<N>-<k>.jsonl``
    that select() wrote for exactly that attempt (k-th selection)."""
    if not session_file:
        return None
    try:
        with open(session_file, encoding="utf-8") as handle:
            header = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError):
        return None
    parent = pathlib.Path(str(header.get("parentSession") or ""))
    if parent.parent.name == "handoff" and parent.stem.startswith("attempt-"):
        try:
            return int(parent.stem.split("-")[1])
        except ValueError:
            return None
    return None


def spawn_records(run_dir: pathlib.Path) -> list[dict]:
    """The root's depth-1 children, from the DAEMON's spawn ledger
    (``<agent>/rlm-ledger/*.jsonl``). Only the daemon writes it, outside the
    agents' sandbox, so it is the authority on who is which role: a kernel
    can neither write it nor see it."""
    config = json.loads((run_dir / "explore.json").read_text(encoding="utf-8"))
    agent_dir = pathlib.Path(config["agent_dir"])
    root_file = json.loads((run_dir / "root_session.json").read_text(encoding="utf-8"))["sessionFile"]
    records = []
    for ledger_file in sorted((agent_dir / "rlm-ledger").glob("*.jsonl")):
        for line in ledger_file.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("op") == "spawn" and record.get("depth") == 1 and record.get("parent") == root_file:
                records.append(record)
    return records


def caller_role(run_dir: pathlib.Path, caller: str) -> str | None:
    """``orchestrator``, ``player`` or None for the kernel whose private
    session directory is ``caller``. A child kernel's directory is named by
    its session id, which is its spawn record's session file stem."""
    session_id = pathlib.Path(caller or "/").name
    if not session_id:
        return None
    for record in spawn_records(run_dir):
        if pathlib.Path(str(record.get("child") or "")).stem != session_id:
            continue
        match = ROLE_NAME.search(str(record.get("name") or ""))
        if match:
            return "orchestrator" if match.group(1) == "orchestrator" else "player"
    return None


def sync_players(run_dir: pathlib.Path) -> None:
    """Record each player's session file in the explore ledger from the spawn
    ledger: the opener is attempt 0, a player is the attempt its handoff names."""
    import explore

    ledger = explore.Ledger(run_dir)
    for record in spawn_records(run_dir):
        match = ROLE_NAME.search(str(record.get("name") or ""))
        child_file = str(record.get("child") or "")
        if match is None or match.group(1) == "orchestrator":
            continue
        attempt = 0 if match.group(1) == "opener" else handoff_attempt(child_file)
        if attempt is None or ledger.player_session(attempt) is not None:
            continue
        ledger.add_player(attempt=attempt, child=str(record.get("childId") or ""), session_file=child_file)


def dispatch(run_dir: pathlib.Path, call: str, kwargs: dict[str, Any], caller: str) -> Any:
    import time as _time

    import explore

    deadline = _time.time() + ROLE_WAIT_S
    role = caller_role(run_dir, caller)
    while role is None and _time.time() < deadline:
        _time.sleep(1.0)
        role = caller_role(run_dir, caller)
    if role != "orchestrator":
        raise explore.NotTheOrchestrator(
            f"explore.{call} is for the orchestrator only; a player never reads the leaderboard or selects a checkpoint"
        )
    sync_players(run_dir)
    kwargs.pop("token", None)
    function = getattr(explore, call)
    return function(**{**kwargs, "run_dir": run_dir})


def make_handler(run_dir: pathlib.Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: D401 - quiet; errors go to the reply
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                call = body.get("call")
                if call not in CALLS:
                    raise PermissionError(f"explore service: {call!r} is not a remote call")
                with _LOCK:
                    result = dispatch(run_dir, call, dict(body.get("kwargs") or {}), str(body.get("caller") or ""))
                reply = {"ok": True, "result": result}
            except Exception as error:  # noqa: BLE001 - the kernel re-raises it
                reply = {
                    "ok": False,
                    "type": type(error).__name__,
                    "error": str(error),
                    "trace": traceback.format_exc(limit=3),
                }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--port", type=int, default=0, help="bind this port (a restart keeps the URL the daemon already has)")
    args = parser.parse_args(argv)
    run_dir = pathlib.Path(args.run_dir).resolve()
    os.environ["EXPLORE_RUN_DIR"] = str(run_dir)
    os.environ.pop("EXPLORE_SERVICE_URL", None)  # this process IS the service
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(run_dir))
    port_file = pathlib.Path(args.port_file)
    tmp = port_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid()}))
    tmp.replace(port_file)
    print(f"explore service on 127.0.0.1:{server.server_port} for {run_dir}", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
