#!/usr/bin/env python3
"""Drive one Prime Agent orchestrator round and capture it as evidence.

Standalone on purpose. ``tools/cli_harness_eval/e16_session.py`` is the
production module and was being written concurrently with this verification;
this simulation must not edit it and must not break if it churns, so the launch
is reproduced here from the same primitives: ``--print`` (no ``--no-session``),
``--resume <header id>``, a dedicated ``PRIME_AGENT_CODING_AGENT_DIR``, and a
FIXED cwd across rounds.

Two things this script learned the hard way, both now baked in:

* ``TMPDIR`` MUST be private. The orchestrator runs unsandboxed, so it shares
  ``/tmp/prime-agent-0/daemon.sock`` with every other prime-agent on the box.
  With another tenant's two wedged processes on that socket, a resumed round
  hung for 900 s and then 500 s; with ``TMPDIR=/root/nld/.pa-e16-sim-tmp`` the
  identical call returned in 3.5 s. Players get a private socket from bwrap
  ``--tmpfs /tmp``; the orchestrator does not.
* ``--mode json`` is OPTIONAL here and defaults OFF. json mode is how
  ``e16_session.py`` reads the session header id, but on this box a resumed
  ``--mode json`` call hung on a prompt that succeeded in text ``--print``
  mode 30 s later, same session and same cwd. The header id is therefore also
  recoverable from the session file, and text mode is the safe default.

Session records are the primary evidence: they carry the prompt, the thinking,
every tool call, and the per-message token usage and USD cost.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parent.parent
AGENT_DIR = Path(os.environ.get("E16_SIM_AGENT_DIR",
                                "/root/nld/.prime-agent-e16-sim"))
SESSIONS = AGENT_DIR / "sessions"
PRIVATE_TMPDIR = Path(os.environ.get("E16_SIM_TMPDIR",
                                     "/root/nld/.pa-e16-sim-tmp"))
MODEL = os.environ.get("E16_MODEL", "z-ai/glm-5.2")
PROVIDER = os.environ.get("E16_PROVIDER", "prime-inference")


def _snapshot() -> dict:
    if not SESSIONS.is_dir():
        return {}
    return {p.name: p.stat().st_size for p in SESSIONS.glob("*.jsonl")}


def run_round(tag: str, prompt: str, cwd: Path, resume: str | None = None,
              timeout: int = 900, json_mode: bool = False) -> dict:
    cwd.mkdir(parents=True, exist_ok=True)
    PRIVATE_TMPDIR.mkdir(parents=True, exist_ok=True)
    before = _snapshot()

    argv = ["prime-agent", "--print"]
    if json_mode:
        argv += ["--mode", "json"]
    argv += ["--offline", "--provider", PROVIDER, "--model", MODEL]
    if resume:
        argv += ["--resume", resume]
    argv += ["--", prompt]

    env = dict(os.environ)
    env["PRIME_AGENT_CODING_AGENT_DIR"] = str(AGENT_DIR)
    env["TMPDIR"] = str(PRIVATE_TMPDIR)
    t0 = time.time()
    proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True,
                          text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    wall = time.time() - t0
    after = _snapshot()

    touched = [n for n, sz in after.items() if before.get(n) != sz]
    new_records, session_file, growth, header_id = [], None, None, None
    if touched:
        session_file = touched[0]
        path = SESSIONS / session_file
        prior = before.get(session_file, 0)
        tail = path.read_bytes()[prior:].decode(errors="replace")
        for line in tail.splitlines():
            line = line.strip()
            if line:
                try:
                    new_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        growth = {"before": prior, "after": after[session_file]}
        # The header id is the FIRST record of the file and is NOT the
        # filename stem (session-manager mints a fresh uuid for the header).
        first = path.read_text(errors="replace").splitlines()[0]
        try:
            header_id = json.loads(first).get("id")
        except json.JSONDecodeError:
            pass

    usage = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
             "cost_usd": 0.0, "assistant_messages": 0}
    for rec in new_records:
        u = (rec.get("message") or {}).get("usage")
        if not u:
            continue
        usage["assistant_messages"] += 1
        for k in ("input", "output", "cacheRead", "cacheWrite"):
            usage[k] += int(u.get(k) or 0)
        usage["cost_usd"] += float((u.get("cost") or {}).get("total") or 0.0)

    record = {
        "tag": tag, "argv": argv, "cwd": str(cwd), "resume": resume,
        "model": MODEL, "provider": PROVIDER, "tmpdir": str(PRIVATE_TMPDIR),
        "prompt": prompt, "stdout": proc.stdout, "stderr": proc.stderr,
        "exit_code": proc.returncode, "wall_s": round(wall, 2),
        "session_header_id": header_id, "session_file": session_file,
        "session_file_growth": growth, "records": new_records, "usage": usage,
    }
    out = SIM / "transcripts" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"[{tag}] exit={proc.returncode} wall={wall:.1f}s "
          f"session={header_id} cost=${usage['cost_usd']:.4f} "
          f"in={usage['input']} out={usage['output']} "
          f"cacheRead={usage['cacheRead']}", file=sys.stderr)
    return record


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--json-mode", action="store_true")
    a = ap.parse_args()
    rec = run_round(a.tag, Path(a.prompt_file).read_text(), Path(a.cwd),
                    resume=a.resume, json_mode=a.json_mode)
    print(rec["stdout"])
