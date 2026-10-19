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


def run_daemon_session(*, prime_agent: str, run_dir: pathlib.Path, agent_dir: pathlib.Path, env: dict, provider: str | None, model: str | None, thinking: str | None, prompt: str, wait_hours: float, cwd: pathlib.Path | None = None) -> int:
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
        config = {"cwd": str(cwd or run_dir), "sessionDir": str(agent_dir / "sessions"), "executionMode": "print"}
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
import contextlib
import json
import re
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
3. When the run has ended, print the run's `rlm.factory.status(result["run_id"])["usage"]`
   and `["events"]`. Then stop.

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


# The keys of a control [taskset] that describe ITS run, not the game.
_RUN_ONLY_TASKSET_KEYS = ("trace_dir", "explicit_seeds", "n_examples", "seed", "max_skill_calls")

# What the factory adds to the control's task prompt for an RLM child.
OPENER_RULES = (
    "Use the game skill: await every game call in your own cell and read its observation before the next "
    "call. Do not run game calls in background tasks, and do not message other agents."
)


def control_taskset(config_toml: pathlib.Path) -> dict:
    import tomllib

    taskset = dict(tomllib.loads(config_toml.read_text(encoding="utf-8"))["taskset"])
    for key in _RUN_ONLY_TASKSET_KEYS:
        taskset.pop(key, None)
    env_args = dict(taskset.get("env_args") or {})
    env_args.pop("explicit_seeds", None)
    taskset["env_args"] = env_args
    return taskset


def control_opener_machine(text: str, task_prompt: str) -> str:
    """The opener's prompt is the control's task prompt, then the RLM rules."""
    match = re.search(r"```machine-spec\n(.*?)\n```", text, re.S)
    spec = json.loads(match.group(1))
    for state in spec["states"]:
        if state.get("entry"):
            state["subagent"]["prompt"] = f"{task_prompt}\n\n{OPENER_RULES}"
    return text[: match.start(1)] + json.dumps(spec, indent=2) + text[match.end(1) :]


def start_service(run_dir: pathlib.Path, *, python: str, pythonpath: list[str]) -> dict:
    """Start the run's explore service and wait for its URL."""
    port_file = run_dir / "service.json"
    port_file.unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.pop("EXPLORE_SERVICE_URL", None)
    log = open(run_dir / "service.log", "ab")
    process = subprocess.Popen(
        [python, "-m", "explore.service", "--run-dir", str(run_dir), "--port-file", str(port_file)],
        cwd=str(run_dir), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
    )
    for _ in range(300):
        if port_file.exists():
            return {"url": json.loads(port_file.read_text())["url"], "process": process}
        if process.poll() is not None:
            raise RuntimeError(f"explore service exited (rc={process.returncode}); see {run_dir / 'service.log'}")
        time.sleep(0.1)
    process.kill()
    raise RuntimeError("explore service never wrote its port file")


def build_kernel_venv(prime_agent: str, agent_dir: pathlib.Path, venv: pathlib.Path, env: dict, run_dir: pathlib.Path) -> None:
    """Build the kernel venv BEFORE the sandbox wraps it: a throwaway daemon's
    session create bootstraps it (runtime, default packages, the skills as
    editable installs). No model call is made."""
    marker = "_editable_impl_prime_agent_skill_nethack.pth"
    if any(venv.glob(f"lib/python*/site-packages/{marker}")):
        return
    socket_path = run_dir / "bootstrap.sock"
    socket_path.unlink(missing_ok=True)
    log = open(run_dir / "bootstrap.log", "ab")
    daemon = subprocess.Popen(
        [prime_agent, "--mode", "daemon", "--daemon-socket", str(socket_path)],
        cwd=str(run_dir), env={k: v for k, v in env.items() if k != "PRIME_AGENT_KERNEL_PYTHON"},
        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
    )
    try:
        client = DaemonClient(socket_path)
        client.command({"type": "create", "config": {"cwd": str(run_dir), "sessionDir": str(run_dir / "bootstrap-sessions"), "executionMode": "print"}})
        deadline = time.time() + 600
        while time.time() < deadline:
            if any(venv.glob(f"lib/python*/site-packages/{marker}")):
                time.sleep(2.0)
                return
            time.sleep(2.0)
        raise RuntimeError(f"the kernel venv was not built in 600 s; see {run_dir / 'bootstrap.log'}")
    finally:
        daemon.terminate()
        socket_path.unlink(missing_ok=True)


def write_kernel_sandbox(run_dir: pathlib.Path, venv: pathlib.Path, agent_dir: pathlib.Path, workspace: pathlib.Path, machines_dir: pathlib.Path) -> pathlib.Path:
    """The kernel interpreter, wrapped in one bwrap per kernel process.

    Prime Agent runs every agent action in its python kernel (`python_repl`;
    `bash()` spawns from the kernel too), so this is the whole sandbox. A
    kernel sees: system files, the venv and the code it imports (read-only),
    the skills' files (read-only), the workspace, and its OWN session
    directory. It does not see the run directory, the agent dir (other
    sessions, the shared memory store), /root/nld, or any other repo. The
    root kernel (RLM_DEPTH=0) also sees the machines dir: it starts the
    factory. Network stays shared (the game's MCP server and the explore
    service are on localhost)."""
    site = next(venv.glob("lib/python*/site-packages"))
    code_dirs: set[str] = set()
    for pth in site.glob("*.pth"):
        for line in pth.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("/") and pathlib.Path(line).is_dir():
                code_dirs.add(line)
    base = None
    cfg = venv / "pyvenv.cfg"
    for line in cfg.read_text().splitlines():
        if line.startswith("home ="):
            base = str(pathlib.Path(line.split("=", 1)[1].strip()).parent)
    skills = []
    for skill in sorted((agent_dir / "skills").iterdir()):
        skills.append((str(skill.resolve()), str(skill)))
    ro = [str(venv)] + ([base] if base else []) + sorted(code_dirs)
    lines = [
        "#!/bin/bash",
        "# Generated by run_explore.py: one bwrap per kernel (see write_kernel_sandbox).",
        "set -e",
        f'REAL="{venv}/bin/python"',
        # A private pid namespace, or /proc/<pid>/root of any outside process
        # would lead straight back to the host filesystem. The kernel watches
        # its owner's pid to exit with it; inside the namespace that owner is
        # bwrap's init (pid 1), which --die-with-parent ties to the worker.
        "args=(--die-with-parent --unshare-pid --unshare-ipc --unshare-uts --setenv PRIME_AGENT_KERNEL_OWNER_PID 1",
        "  --ro-bind /usr /usr --ro-bind-try /bin /bin --ro-bind-try /sbin /sbin",
        "  --ro-bind-try /lib /lib --ro-bind-try /lib64 /lib64 --ro-bind-try /lib32 /lib32",
        "  --ro-bind /etc /etc --proc /proc --dev /dev --tmpfs /tmp --tmpfs /root --tmpfs /run)",
    ]
    for path in ro:
        lines.append(f'args+=(--ro-bind "{path}" "{path}")')
    for target, where in skills:
        lines.append(f'args+=(--ro-bind "{target}" "{where}")')
    # settings.json is read-only and holds no secret (the game server's URL
    # and the NAME of its token variable): the runtime reads factory.enabled
    # and similar switches from it.
    lines.append(f'args+=(--ro-bind "{agent_dir / "settings.json"}" "{agent_dir / "settings.json"}")')
    lines += [
        f'args+=(--bind "{workspace}" "{workspace}" --chdir "{workspace}")',
        'if [ -n "$RLM_SESSION_DIR" ]; then mkdir -p "$RLM_SESSION_DIR"; args+=(--bind "$RLM_SESSION_DIR" "$RLM_SESSION_DIR"); fi',
        'if [ -n "$RLM_HARNESS_STATE_DIR" ]; then mkdir -p "$RLM_HARNESS_STATE_DIR"; args+=(--bind "$RLM_HARNESS_STATE_DIR" "$RLM_HARNESS_STATE_DIR"); fi',
        f'if [ "${{RLM_DEPTH:-0}}" = "0" ]; then args+=(--ro-bind "{machines_dir}" "{machines_dir}"); fi',
        'exec bwrap "${args[@]}" "$REAL" "$@"',
    ]
    wrapper = run_dir / "kernel-python"
    wrapper.write_text("\n".join(lines) + "\n", encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def single_life_machine(text: str) -> str:
    """The machine with the opener only: the same first life a PAE run plays
    as attempt 1 (same prompt, same seed), with no state after it."""
    match = re.search(r"```machine-spec\n(.*?)\n```", text, re.S)
    spec = json.loads(match.group(1))
    spec["states"] = [state for state in spec["states"] if state.get("entry")]
    spec["transitions"] = []
    spec["run"]["max_transitions"] = 2
    return text[: match.start(1)] + json.dumps(spec, indent=2) + text[match.end(1) :]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=30, help="N attempts; the machine's run limits are set from it")
    parser.add_argument("--character", default="Val-hum-neu-fem")
    parser.add_argument("--skill-set", default=None, help="default: the control config's skill set, else np_core,request_map,search,save")
    parser.add_argument("--control-config", default=None, help="the base arm's config.toml: its [taskset] game knobs, its workspace (AGENTS.md system prompt), its task prompt, and its SKILL.md bytes are served")
    parser.add_argument("--max-skill-calls", type=int, default=150)
    parser.add_argument("--every", type=int, default=20, help="auto-checkpoint every N calls")
    parser.add_argument("--nle-lib", default=None, help="the libnethack.so every game server loads (NLE_LIB_PATH)")
    parser.add_argument("--wall-clock-s", type=float, default=7200.0, help="per-attempt wall clock (the single-life rollout timeout); 0 = none")
    parser.add_argument("--single-life", action="store_true", help="the opener only: one life, no orchestrator, no checkpoint resumes (the base arm)")
    parser.add_argument("--prime-agent", default="prime-agent", help="the Rust prime-agent binary with fork_from support")
    parser.add_argument("--python", default=sys.executable, help="the eval venv python (verifiers, mcp, aiohttp)")
    parser.add_argument("--agent-dir", default=None, help="a prime-agent agent dir you already logged in with (its auth.json and settings are kept); default <run-dir>/agent")
    parser.add_argument("--provider", default=None, help="model provider for the root and its children (e.g. prime-inference)")
    parser.add_argument("--model", default=None, help="model id (e.g. deepseek/deepseek-v4-flash)")
    parser.add_argument("--thinking", default=None, help="off, minimal, low, medium, high")
    parser.add_argument("--mode", choices=("daemon", "print"), default="daemon", help="daemon (subagents work) or the in-process --print path (no subagents)")
    parser.add_argument("--wait-hours", type=float, default=6.0, help="how long to wait for the run to finish")
    parser.add_argument("--context-window", type=int, default=None, help="the model's REAL context window in tokens, when Prime Agent's catalog overstates it (Qwen/Qwen3.5-9B on Prime: catalog 262144, served 65536); compaction is computed against it")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="with --context-window: the reply budget, so prompt + reply fits the window")
    parser.add_argument("--compaction-threshold-percent", type=float, default=80.0, help="with --context-window: compact at this share of the window")
    parser.add_argument("--no-sandbox", dest="sandbox", action="store_false", help="run the agents' kernels unsandboxed (default: one bwrap per kernel, which sees only the workspace and its own session)")
    parser.add_argument("--dry-run", action="store_true", help="set everything up and print the command; do not start prime-agent")
    parser.add_argument("prime_agent_args", nargs="*", help="arguments after -- go to prime-agent")
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    taskset_json, workspace = None, None
    if args.control_config:
        taskset = control_taskset(pathlib.Path(args.control_config))
        if args.skill_set is None:
            args.skill_set = taskset.get("env_args", {}).get("skill_set")
        taskset_json = run_dir / "taskset.json"
        write_json(taskset_json, taskset)
    # The agents work here and nowhere else: a sandboxed kernel sees this
    # directory, its own session directory, and read-only code, never the run
    # directory (ledger, archive, servers) or the agent dir.
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if args.skill_set is None:
        args.skill_set = "np_core,request_map,search"
    pythonpath = [str(REPO / "tools" / "pycompat"), str(ENGINE), str(REPO), str(ENV_DIR), str(SKILLS_DIR / "explore" / "src")]
    write_json(
        run_dir / "explore.json",
        {
            "adapter": f"{ENV_DIR / 'explore_adapter.py'}:NetHackAdapter",
            "python": args.python,
            "pythonpath": pythonpath,
            "env_dir": str(ENV_DIR),
            "archive": str(run_dir / "archive"),
            # The service reads roles from the daemon's spawn ledger here.
            "agent_dir": str(pathlib.Path(args.agent_dir).resolve() if args.agent_dir else run_dir / "agent"),
            "seed": args.seed,
            "character": args.character,
            "task_spec": "full_nle",
            "skill_set": args.skill_set,
            "max_skill_calls": args.max_skill_calls,
            "auto_checkpoint_every_n_calls": args.every,
            "wall_clock_s": args.wall_clock_s,
            "nle_lib_path": str(pathlib.Path(args.nle_lib).resolve()) if args.nle_lib else None,
            "taskset_json": str(taskset_json) if taskset_json else None,
            "workspace": str(workspace) if workspace else None,
            "single_life": args.single_life,
            "attempts": args.attempts,
        },
    )

    # The machine with this run's attempt count (a copy of the library machine).
    machines_dir = run_dir / "machines"
    source = SKILLS_DIR / "explore" / "machines" / "explore" / "MACHINE.md"
    text = source.read_text(encoding="utf-8")
    text = text.replace('"max_entries": 29', f'"max_entries": {max(1, args.attempts - 1)}')
    text = text.replace('"max_transitions": 64', f'"max_transitions": {2 * args.attempts + 2}')
    if args.single_life:
        text = single_life_machine(text)
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
    if taskset_json is not None:
        task = json.loads((run_dir / "servers" / "task.json").read_text(encoding="utf-8"))
        machine = machines_dir / "explore" / "MACHINE.md"
        machine.write_text(control_opener_machine(machine.read_text(encoding="utf-8"), task["prompt"]), encoding="utf-8")

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
    if args.context_window:
        # Prime Agent compacts against the model's catalog window; a provider
        # that serves less overflows before compaction ever runs. Override the
        # window (and cap the reply) for this model in this run's agent dir.
        provider_id = args.provider or "prime-inference"
        models_path = agent_dir / "models.json"
        models = json.loads(models_path.read_text(encoding="utf-8")) if models_path.is_file() else {}
        # A FULL definition, not modelOverrides: Prime's hosted models reach
        # the registry through a path that overrides never touch (measured:
        # `prime-agent model list` showed no Qwen/Qwen3.5-9B at all, so the
        # session fell back to a default window). A custom model wins on
        # provider+id.
        defined = models.setdefault("providers", {}).setdefault(provider_id, {}).setdefault("models", [])
        defined[:] = [m for m in defined if m.get("id") != args.model]
        defined.append({
            "id": args.model, "name": f"{args.model} ({args.context_window // 1024}K served)",
            "api": "openai-completions", "baseUrl": "https://api.pinference.ai/api/v1",
            "reasoning": True, "input": ["text"],
            "contextWindow": int(args.context_window), "maxTokens": int(args.max_output_tokens),
        })
        write_json(models_path, models)
        settings.setdefault("compaction", {}).update({"enabled": True, "thresholdPercent": float(args.compaction_threshold_percent)})
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
    # The run's explore service: the ledger, archive and game servers stay
    # outside the agents' sandbox; kernels reach them only through it.
    service = start_service(run_dir, python=args.python, pythonpath=pythonpath)
    env["EXPLORE_SERVICE_URL"] = service["url"]
    if args.sandbox:
        venv = pathlib.Path(env["PRIME_AGENT_KERNEL_VENV"])
        build_kernel_venv(args.prime_agent, agent_dir, venv, env, run_dir)
        env["PRIME_AGENT_KERNEL_PYTHON"] = str(write_kernel_sandbox(run_dir, venv, agent_dir, workspace, machines_dir))
    model_flags = []
    for flag, value in (("--provider", args.provider), ("--model", args.model), ("--thinking", args.thinking)):
        if value:
            model_flags += [flag, value]
    # The Rust CLI takes the print-mode message right after --print (no `--` arm).
    command = [args.prime_agent, "--print", ROOT_PROMPT, *model_flags, *args.prime_agent_args]
    (run_dir / "command.json").write_text(json.dumps({"mode": args.mode, "print_command": command, "env": {k: env[k] for k in ("PRIME_AGENT_CODING_AGENT_DIR", "PRIME_AGENT_KERNEL_VENV", "PRIME_AGENT_MACHINES_DIR", "EXPLORE_RUN_DIR", "EXPLORE_ADAPTER")}}, indent=2) + "\n", encoding="utf-8")
    if args.mode == "print":
        print("prime-agent command:", " ".join([command[0], "--print", "<prompt>", *command[3:]]), flush=True)
    else:
        print(f"daemon mode: socket {run_dir / 'daemon.sock'}, model {args.provider}/{args.model}", flush=True)
    if args.dry_run:
        service["process"].terminate()
        return 0
    if args.mode == "print":
        with open(run_dir / "root.log", "ab") as log:
            completed = subprocess.run(command, cwd=str(run_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"prime-agent exited {completed.returncode}; see {run_dir / 'root.log'} and {run_dir / 'ledger.jsonl'}")
        return completed.returncode
    code = run_daemon_session(
        prime_agent=args.prime_agent, run_dir=run_dir, agent_dir=agent_dir, env=env,
        provider=args.provider, model=args.model, thinking=args.thinking,
        prompt=ROOT_PROMPT, wait_hours=args.wait_hours,
        cwd=workspace,
    )
    service["process"].terminate()
    # The final leaderboard, read here, outside the sandbox: no kernel needs
    # ledger access after the orchestrator's last round.
    import explore  # noqa: E402

    with contextlib.suppress(Exception):
        print("--- final leaderboard ---\n" + explore.leaderboard(100, run_dir=run_dir), flush=True)
    print(f"run ended ({code}); see {run_dir / 'ledger.jsonl'} and {run_dir / 'root.log'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
