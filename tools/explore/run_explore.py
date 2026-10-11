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

Arguments after `--` go to prime-agent verbatim (provider, model, thinking).
The model's credentials are prime-agent's own (run `prime-agent` once and
`/login` first; this script does not manage them).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENV_DIR = REPO / "environments" / "nethack"
SKILLS_DIR = REPO / "skills"
NETHACK_SKILL = REPO / "harnesses" / "nethack-prime-agent" / "nethack_prime_agent" / "skill"
ENGINE = pathlib.Path(os.environ.get("NETHACK_ENGINE", "/root/NetHack-engine"))

ROOT_PROMPT = """You run a checkpointed-exploration experiment. Do exactly this and nothing else.

1. Run `result = await explore.run("explore")` and print `result["run_id"]`.
2. Then loop: `snap = await rlm.factory.watch(result["run_id"], 60)`; print one line
   with `snap["state"]` and the `entries_used` of each node in `snap["nodes"]`.
   Stop looping when `snap["state"]` is not "running" and not "stopping".
   If it is "paused", print `snap["pause_reason"]` and run
   `await rlm.factory.resume(result["run_id"])`, then keep looping.
3. When the run has ended, print `explore.leaderboard(100)` and the run's
   `rlm.factory.status(result["run_id"])["usage"]`. Then stop.
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
    parser.add_argument("--dry-run", action="store_true", help="set everything up and print the command; do not start prime-agent")
    parser.add_argument("prime_agent_args", nargs="*", help="arguments after -- go to prime-agent")
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pythonpath = [str(REPO / "tools" / "pycompat"), str(ENGINE), str(REPO), str(ENV_DIR), str(SKILLS_DIR / "explore" / "src")]
    write_json(
        run_dir / "explore.json",
        {
            "adapter": "explore_adapter:NetHackAdapter",
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
        },
    )

    # The machine with this run's attempt count (a copy of the library machine).
    machines_dir = run_dir / "machines"
    source = SKILLS_DIR / "explore" / "machines" / "explore" / "MACHINE.md"
    text = source.read_text(encoding="utf-8")
    text = text.replace('"max_entries": 29', f'"max_entries": {max(1, args.attempts - 1)}')
    text = text.replace('"max_transitions": 64', f'"max_transitions": {2 * args.attempts + 2}')
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
            "EXPLORE_ADAPTER": "explore_adapter:NetHackAdapter",
            "NETHACK_MCP_TOKEN": server["headers"]["Authorization"].split(" ", 1)[1],
            "PYTHONPATH": os.environ["PYTHONPATH"],
        }
    )
    # The Rust CLI takes the print-mode message right after --print (no `--` arm).
    command = [args.prime_agent, "--print", ROOT_PROMPT, *args.prime_agent_args]
    (run_dir / "command.json").write_text(json.dumps({"command": command, "env": {k: env[k] for k in ("PRIME_AGENT_CODING_AGENT_DIR", "PRIME_AGENT_KERNEL_VENV", "PRIME_AGENT_MACHINES_DIR", "EXPLORE_RUN_DIR", "EXPLORE_ADAPTER", "PYTHONPATH")}}, indent=2) + "\n", encoding="utf-8")
    print("prime-agent command:", " ".join([command[0], "--print", "<prompt>", *command[3:]]), flush=True)
    if args.dry_run:
        return 0
    with open(run_dir / "root.log", "ab") as log:
        completed = subprocess.run(command, cwd=str(run_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"prime-agent exited {completed.returncode}; see {run_dir / 'root.log'} and {run_dir / 'ledger.jsonl'}")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
