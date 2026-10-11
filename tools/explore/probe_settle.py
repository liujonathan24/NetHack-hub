#!/usr/bin/env python3
"""Probe: does rlm.collect settle a child while its kernel cell still runs?

Starts a private daemon, creates a root session, and prompts it to spawn ONE
child whose only job is to run a 200 s sleep in one cell and then answer
"done". The root collects the child and prints how long collect waited. A
collect that returns long before ~200 s reproduces the premature settle the
explore runs hit. Run with the same --agent-dir you logged in with.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_explore import run_daemon_session  # noqa: E402

PROMPT_SPAWN = """Run exactly this Python cell and nothing else, then answer with the printed child id:

h = await rlm.spawn("Run exactly one Python cell: `import time; time.sleep(200); print('slept')`. Wait for it to finish. Then answer with the single word: done", name="sleeper")
SLEEPER = h
print(h.rlm_child_id)

Do not spawn anything else. Do not collect. Do not install packages.
"""

PROMPT_COLLECT = """Run exactly this Python cell and nothing else, then print its output as your final answer, verbatim:

import time
t0 = time.time()
results = await rlm.collect([SLEEPER], timeout_ms=600000)
print("collect returned after", round(time.time() - t0), "seconds")
print("status:", results[0].status, "settled:", results[0].settled, "answer:", results[0].answer_preview)

Do not spawn anything. Do not retry. Do not install packages.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--prime-agent", required=True)
    parser.add_argument("--provider", default="prime-inference")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    args = parser.parse_args()
    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = pathlib.Path(args.agent_dir).resolve()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir), "PRIME_AGENT_KERNEL_VENV": str(agent_dir / "kernel-venv")})
    return run_two_prompts(prime_agent=args.prime_agent, run_dir=run_dir, agent_dir=agent_dir, env=env, provider=args.provider, model=args.model)


def run_two_prompts(*, prime_agent, run_dir, agent_dir, env, provider, model) -> int:
    import json
    import subprocess
    import time

    from run_explore import DaemonClient

    socket_path = run_dir / "daemon.sock"
    if socket_path.exists():
        socket_path.unlink()
    log = open(run_dir / "daemon.log", "ab")
    daemon = subprocess.Popen([prime_agent, "--mode", "daemon", "--daemon-socket", str(socket_path)], cwd=str(run_dir), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    try:
        client = DaemonClient(socket_path)
        config = {"cwd": str(run_dir), "sessionDir": str(agent_dir / "sessions"), "executionMode": "print", "provider": provider, "model": model, "thinking": "off"}
        created = client.command({"type": "create", "lifecycle": "resident", "config": config})
        sid = (created.get("data") or {}).get("activeSessionId")
        print("root", sid, flush=True)
        t0 = time.time()
        client.command({"type": "prompt", "activeSessionId": sid, "message": PROMPT_SPAWN, "source": "rpc"})
        client.command({"type": "wait_for_headless_completion", "activeSessionId": sid, "waitForRlmQuiescence": False}, timeout_s=900)
        last = client.command({"type": "get_last_assistant_text", "activeSessionId": sid})
        print(f"[{round(time.time()-t0)}s] spawn turn done:", json.dumps(last.get("data"))[:300], flush=True)
        time.sleep(25)
        print(f"[{round(time.time()-t0)}s] sending collect prompt (child should be mid-sleep)", flush=True)
        client.command({"type": "prompt", "activeSessionId": sid, "message": PROMPT_COLLECT, "source": "rpc"})
        client.command({"type": "wait_for_headless_completion", "activeSessionId": sid, "waitForRlmQuiescence": True}, timeout_s=1800)
        last = client.command({"type": "get_last_assistant_text", "activeSessionId": sid})
        print(f"[{round(time.time()-t0)}s] collect turn done:", json.dumps(last.get("data"))[:600], flush=True)
        client.command({"type": "kill", "activeSessionId": sid})
        client.close()
        return 0
    finally:
        try:
            os.killpg(daemon.pid, 15)
        except ProcessLookupError:
            pass
        try:
            daemon.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(daemon.pid, 9)
        log.close()


if __name__ == "__main__":
    sys.exit(main())
