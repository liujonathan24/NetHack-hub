#!/usr/bin/env python3
"""Run Prime-Agent-Explore on NetHack with the Rust prime-agent and the explore machine.

One command sets up a run directory and starts the root session:

    tools/explore/run_explore.py --run-dir runs/explore-s0 --seed 0 --attempts 10 \
        --prime-agent /root/nld/pa-upstream/target/release/prime-agent \
        --python /root/nld/.venv-e1315/bin/python -- --provider prime-inference --model zai/glm-5.2

What it does, in order:
1. writes <run-dir>/explore.json (the adapter config) and the archive dir;
2. boots the attempt-0 game server through the adapter (the opener plays it);
3. writes a private agent dir: settings.json with the attempt-0 MCP server,
   `factory.enabled`, and symlinked skills (explore, nethack); the explore
   machine library is pointed at through PRIME_AGENT_MACHINES_DIR;
4. runs `prime-agent --print` with a prompt that calls `explore.run("explore")`
   and watches the run to its end.

The root session is daemon-backed: a Rust `--print` session runs in process
with no RLM child runtime, so `rlm.spawn` (and therefore the factory) refuses
there. The driver starts `prime-agent --mode daemon` on a private socket with
this run's environment, then speaks the daemon's JSONL protocol: `create` a
resident root session, `prompt` it, `wait_for_headless_completion` with RLM
quiescence, read the last assistant text, `kill` the session, stop the daemon.
`--mode print` keeps the old in-process path (no subagents; useful only to
check the setup).

The model's credentials are prime-agent's own (run `prime-agent` once with the
same agent dir and `/login` first; this script does not manage them).
"""
from __future__ import annotations


DAEMON_PROTOCOL = {"name": "prime-agent.daemon", "version": 7}


class DaemonClient:
    """A minimal JSONL client for the prime-agent daemon socket."""

    def __init__(self, socket_path: pathlib.Path, *, connect_timeout_s: float = 60.0) -> None:
        deadline = time.time() + connect_timeout_s
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(str(socket_path))
                break
            except OSError:
                self.sock.close()
                if time.time() > deadline:
                    raise
                time.sleep(0.5)
        self.reader = self.sock.makefile("r", encoding="utf-8")
        self.hello = json.loads(self.reader.readline())

    def command(self, command: dict, *, timeout_s: float | None = 120.0) -> dict:
        request_id = uuid.uuid4().hex[:12]
        envelope = {"type": "command", "id": request_id, "protocol": DAEMON_PROTOCOL, "command": command}
        self.sock.sendall((json.dumps(envelope) + "\n").encode("utf-8"))
        self.sock.settimeout(timeout_s)
        while True:
            line = self.reader.readline()
            if not line:
                raise RuntimeError("the daemon closed the connection")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id:
                if message.get("success") is False:
                    raise RuntimeError(f"daemon {command['type']} failed: {message.get('error')} {message.get('errorInfo') or ''}")
                return message

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def run_daemon_session(*, prime_agent: str, run_dir: pathlib.Path, agent_dir: pathlib.Path, env: dict, provider: str | None, model: str | None, thinking: str | None, prompt: str, wait_hours: float) -> int:
    """Start a private daemon, create and prompt a root session, wait for it to finish."""
    socket_path = run_dir / "daemon.sock"
    if socket_path.exists():
        socket_path.unlink()
    daemon_log = open(run_dir / "daemon.log", "ab")
    daemon = subprocess.Popen(
        [prime_agent, "--mode", "daemon", "--daemon-socket", str(socket_path)],
        cwd=str(run_dir), env=env, stdin=subprocess.DEVNULL, stdout=daemon_log, stderr=daemon_log, start_new_session=True,
    )
    exit_code = 1
    try:
        client = DaemonClient(socket_path)
        print(f"daemon up: {client.hello.get('type')} pid {daemon.pid}", flush=True)
        config = {"cwd": str(run_dir), "sessionDir": str(agent_dir / "sessions"), "executionMode": "print"}
        if provider:
            config["provider"] = provider
        if model:
            config["model"] = model
        if thinking:
            config["thinking"] = thinking
        created = client.command({"type": "create", "lifecycle": "resident", "config": config})
        data = created.get("data") or {}
        session_id = data.get("activeSessionId") or data.get("id")
        if not session_id:
            raise RuntimeError(f"create returned no session id: {created}")
        print(f"root session: {session_id} file {data.get('sessionFile')}", flush=True)
        (run_dir / "root_session.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        client.command({"type": "prompt", "activeSessionId": session_id, "message": prompt, "source": "rpc"})
        print("prompt admitted; waiting for the run to end ...", flush=True)
        client.command(
            {"type": "wait_for_headless_completion", "activeSessionId": session_id, "waitForRlmQuiescence": True},
            timeout_s=wait_hours * 3600,
        )
        last = client.command({"type": "get_last_assistant_text", "activeSessionId": session_id})
        text = (last.get("data") or {}).get("text") if isinstance(last.get("data"), dict) else last.get("data")
        (run_dir / "root.log").write_text(json.dumps(text, indent=2) if not isinstance(text, str) else text, encoding="utf-8")
        print("--- root's final answer ---", flush=True)
        print(text if isinstance(text, str) else json.dumps(text, indent=2), flush=True)
        try:
            client.command({"type": "kill", "activeSessionId": session_id})
        except RuntimeError as error:
            print(f"kill: {error}", flush=True)
        client.close()
        exit_code = 0
    finally:
        try:
            os.killpg(daemon.pid, 15)
        except ProcessLookupError:
            pass
        try:
            daemon.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(daemon.pid, 9)
        daemon_log.close()
    return exit_code


import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENV_DIR = REPO / "environments" / "nethack"
SKILLS_DIR = REPO / "skills"
NETHACK_SKILL = REPO / "harnesses" / "nethack-prime-agent" / "nethack_prime_agent" / "skill"
ENGINE = pathlib.Path(os.environ.get("NETHACK_ENGINE", "/root/NetHack-engine"))

ROOT_PROMPT = """You run a checkpointed-exploration experiment. Do exactly this and nothing else.

1. Run `result = await explore.run("explore")` and print `result["run_id"]`. Run it ONCE.
2. Then loop: `snap = await rlm.factory.watch(result["run_id"], 60)`; print one line
   with `snap["state"]` and the `entries_used` of each node in `snap["nodes"]`.
   Stop looping when `snap["state"]` is not "running" and not "stopping".
   If it is "paused", print `snap["pause_reason"]` and run
   `await rlm.factory.resume(result["run_id"])`, then keep looping.
3. When the run has ended, print `explore.leaderboard(100, token="{{EXPLORE_TOKEN}}")`
   and the run's `rlm.factory.status(result["run_id"])["usage"]` and `["events"]`. Then stop.

Rules. Never start a second run. Never install, upgrade, or remove Python
packages. Never edit files. Do not try to repair anything: if something fails,
print the error and the run's status and stop.
"""


def write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def link(target: pathlib.Path, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.symlink_to(target, target_is_directory=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=30, help="N attempts; the machine's run limits are set from it")
    parser.add_argument("--character", default="Val-hum-neu-fem")
    parser.add_argument("--skill-set", default="np_core,request_map,search,save")
    parser.add_argument("--max-skill-calls", type=int, default=150)
    parser.add_argument("--every", type=int, default=20, help="auto-checkpoint every N calls")
    parser.add_argument("--prime-agent", default="prime-agent", help="the Rust prime-agent binary with fork_from support")
    parser.add_argument("--python", default=sys.executable, help="the eval venv python (verifiers, mcp, aiohttp)")
    parser.add_argument("--agent-dir", default=None, help="a prime-agent agent dir you already logged in with (its auth.json and settings are kept); default <run-dir>/agent")
    parser.add_argument("--provider", default=None, help="model provider for the root and its children (e.g. prime-inference)")
    parser.add_argument("--model", default=None, help="model id (e.g. deepseek/deepseek-v4-flash)")
    parser.add_argument("--thinking", default=None, help="off, minimal, low, medium, high")
    parser.add_argument("--mode", choices=("daemon", "print"), default="daemon", help="daemon (subagents work) or the in-process --print path (no subagents)")
    parser.add_argument("--wait-hours", type=float, default=6.0, help="how long to wait for the run to finish")
    parser.add_argument("--dry-run", action="store_true", help="set everything up and print the command; do not start prime-agent")
    parser.add_argument("prime_agent_args", nargs="*", help="arguments after -- go to prime-agent")
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pythonpath = [str(REPO / "tools" / "pycompat"), str(ENGINE), str(REPO), str(ENV_DIR), str(SKILLS_DIR / "explore" / "src")]
    write_json(
        run_dir / "explore.json",
        {
            "adapter": f"{ENV_DIR / 'explore_adapter.py'}:NetHackAdapter",
            "python": args.python,
            "pythonpath": pythonpath,
            "env_dir": str(ENV_DIR),
            "archive": str(run_dir / "archive"),
            "seed": args.seed,
            "character": args.character,
            "task_spec": "full_nle",
            "skill_set": args.skill_set,
            "max_skill_calls": args.max_skill_calls,
            "auto_checkpoint_every_n_calls": args.every,
            "attempts": args.attempts,
            # The orchestrator-only token: gates explore.leaderboard() and
            # explore.select(); the machine copy below carries it in the
            # orchestrator's prompt, and only there.
            "orchestrator_token": uuid.uuid4().hex,
        },
    )
    orchestrator_token = json.loads((run_dir / "explore.json").read_text(encoding="utf-8"))["orchestrator_token"]

    # The machine with this run's attempt count (a copy of the library machine).
    machines_dir = run_dir / "machines"
    source = SKILLS_DIR / "explore" / "machines" / "explore" / "MACHINE.md"
    text = source.read_text(encoding="utf-8")
    text = text.replace('"max_entries": 29', f'"max_entries": {max(1, args.attempts - 1)}')
    text = text.replace('"max_transitions": 64', f'"max_transitions": {2 * args.attempts + 2}')
    text = text.replace("{{EXPLORE_TOKEN}}", orchestrator_token)
    (machines_dir / "explore").mkdir(parents=True, exist_ok=True)
    (machines_dir / "explore" / "MACHINE.md").write_text(text, encoding="utf-8")

    # Attempt 0's game server, which the opener plays through the agent dir's MCP config.
    sys.path.insert(0, str(ENV_DIR))
    sys.path.insert(0, str(SKILLS_DIR / "explore" / "src"))
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    import explore_adapter  # noqa: E402

    adapter = explore_adapter.NetHackAdapter()
    print(f"booting the attempt-0 game server in {run_dir} ...", flush=True)
    opener = adapter.fresh(attempt=0, run_dir=run_dir)
    server = opener["mcp_servers"][0]
    print(f"attempt 0: {server['url']}", flush=True)

    agent_dir = pathlib.Path(args.agent_dir).resolve() if args.agent_dir else run_dir / "agent"
    link(SKILLS_DIR / "explore", agent_dir / "skills" / "explore")
    link(NETHACK_SKILL, agent_dir / "skills" / "nethack")
    # Merge into the existing settings (a logged-in agent dir keeps its
    # provider, model, and auth); this run's keys win.
    settings = {}
    if (agent_dir / "settings.json").is_file():
        try:
            settings = json.loads((agent_dir / "settings.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            settings = {}
    # The Rust binary writes every known key, unset ones as null: drop the
    # nulls so the merge below sees only real values.
    settings = {key: value for key, value in settings.items() if value is not None}
    settings.update(
        {
            "onboardingShown": True,
            "quietStartup": True,
            "factory": {"enabled": True},
            "autoRefine": {"enabled": False, "compact": False},
        }
    )
    servers = settings.get("mcpServers") if isinstance(settings.get("mcpServers"), dict) else {}
    servers["nethack"] = {"type": "http", "url": server["url"], "bearerTokenEnvVar": "NETHACK_MCP_TOKEN"}
    settings["mcpServers"] = servers
    write_json(agent_dir / "settings.json", settings)
    env = dict(os.environ)
    env.update(
        {
            "PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir),
            # The Rust binary's kernel venv must never be the shared
            # ~/.prime/agent/kernel-venv the 0.3.3 install and other runs use
            # (it would rebuild it with its own runtime); one venv per run.
            "PRIME_AGENT_KERNEL_VENV": str((agent_dir / "kernel-venv") if args.agent_dir else (run_dir / "kernel-venv")),
            "PRIME_AGENT_MACHINES_DIR": str(machines_dir),
            "EXPLORE_RUN_DIR": str(run_dir),
            # The adapter loads from its file: the kernel venv must NOT get a
            # PYTHONPATH into the repo (the env's nethack.py would shadow the
            # `nethack` skill package). The game servers get their PYTHONPATH
            # from explore.json through the adapter.
            "EXPLORE_ADAPTER": f"{ENV_DIR / 'explore_adapter.py'}:NetHackAdapter",
            "NETHACK_MCP_TOKEN": server["headers"]["Authorization"].split(" ", 1)[1],
            # Diagnostic: the daemon logs each RLM child settle verdict with
            # the state snapshot it rests on (an opt-in trace in our fork).
            "PRIME_AGENT_RLM_SETTLE_TRACE": "1",
        }
    )
    env.pop("PYTHONPATH", None)
    model_flags = []
    for flag, value in (("--provider", args.provider), ("--model", args.model), ("--thinking", args.thinking)):
        if value:
            model_flags += [flag, value]
    # The Rust CLI takes the print-mode message right after --print (no `--` arm).
    command = [args.prime_agent, "--print", ROOT_PROMPT.replace("{{EXPLORE_TOKEN}}", orchestrator_token), *model_flags, *args.prime_agent_args]
    (run_dir / "command.json").write_text(json.dumps({"mode": args.mode, "print_command": command, "env": {k: env[k] for k in ("PRIME_AGENT_CODING_AGENT_DIR", "PRIME_AGENT_KERNEL_VENV", "PRIME_AGENT_MACHINES_DIR", "EXPLORE_RUN_DIR", "EXPLORE_ADAPTER")}}, indent=2) + "\n", encoding="utf-8")
    if args.mode == "print":
        print("prime-agent command:", " ".join([command[0], "--print", "<prompt>", *command[3:]]), flush=True)
    else:
        print(f"daemon mode: socket {run_dir / 'daemon.sock'}, model {args.provider}/{args.model}", flush=True)
    if args.dry_run:
        return 0
    if args.mode == "print":
        with open(run_dir / "root.log", "ab") as log:
            completed = subprocess.run(command, cwd=str(run_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"prime-agent exited {completed.returncode}; see {run_dir / 'root.log'} and {run_dir / 'ledger.jsonl'}")
        return completed.returncode
    code = run_daemon_session(
        prime_agent=args.prime_agent, run_dir=run_dir, agent_dir=agent_dir, env=env,
        provider=args.provider, model=args.model, thinking=args.thinking,
        prompt=ROOT_PROMPT.replace("{{EXPLORE_TOKEN}}", orchestrator_token), wait_hours=args.wait_hours,
    )
    print(f"run ended ({code}); see {run_dir / 'ledger.jsonl'} and {run_dir / 'root.log'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
