"""The explore run's control service: the ledger, archive and game servers,
kept OUTSIDE the agents' sandbox.

Every agent kernel of a sandboxed run sees only its workspace and its own
session directory, so it cannot read the ledger, the archive, another
player's conversation, or our code. The orchestrator still needs
``leaderboard()`` and ``select()``; with ``EXPLORE_SERVICE_URL`` set, the
``explore`` package sends those calls here (``explore._remote``), and this
process runs them against the run directory with the real adapter. The
orchestrator token gates them exactly as in-process.

Run by the driver::

    python -m explore.service --run-dir R --port-file R/service.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

#: The calls a kernel may make. Anything else is refused.
CALLS = ("leaderboard", "rows", "select", "select_fixed", "record_player")

# One call at a time: select() boots and finalizes game servers and appends to
# the ledger; the ledger is append-only but a selection must see the settle of
# the attempt before it.
_LOCK = Lock()


def handoff_attempt(session_file: str | None) -> int | None:
    """The attempt a forked player belongs to: its session header's
    ``parentSession`` is the handoff copy ``handoff/attempt-<N>.jsonl``
    that select() wrote for exactly that attempt."""
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
            return int(parent.stem.split("-", 1)[1])
        except ValueError:
            return None
    return None


def dispatch(run_dir: pathlib.Path, call: str, kwargs: dict[str, Any]) -> Any:
    import explore

    if call == "record_player":
        ledger = explore.Ledger(run_dir)
        attempt = kwargs.get("attempt")
        if kwargs.get("state") != "opener":
            attempt = handoff_attempt(kwargs.get("session_file"))
        if attempt is None:
            return None  # a player that never forked from a handoff is no attempt
        ledger.add_player(attempt=int(attempt), child=str(kwargs["child"]), session_file=kwargs.get("session_file"))
        return int(attempt)
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
                    result = dispatch(run_dir, call, dict(body.get("kwargs") or {}))
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
    args = parser.parse_args(argv)
    run_dir = pathlib.Path(args.run_dir).resolve()
    os.environ["EXPLORE_RUN_DIR"] = str(run_dir)
    os.environ.pop("EXPLORE_SERVICE_URL", None)  # this process IS the service
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(run_dir))
    port_file = pathlib.Path(args.port_file)
    tmp = port_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid()}))
    tmp.replace(port_file)
    print(f"explore service on 127.0.0.1:{server.server_port} for {run_dir}", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
