"""The generated kernel wrapper (run_explore.write_kernel_sandbox), run with
its final `exec bwrap` replaced by a print of the bwrap arguments: each
session gets its own workspace copy, seeded from the pristine template and
bound at the workspace's path, and only the root's kernel sees the machines
dir."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import run_explore  # noqa: E402


class KernelWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        base = pathlib.Path(self.temp.name)
        self.run = base / "run"
        self.agent = self.run / "agent"
        (self.agent / "skills").mkdir(parents=True)
        (self.agent / "settings.json").write_text("{}")
        venv = base / "venv"
        (venv / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
        self.workspace = self.run / "workspace"
        (self.workspace / "memory").mkdir(parents=True)
        (self.workspace / "AGENTS.md").write_text("system prompt\n")
        self.template = self.run / "workspace-template"
        subprocess.run(["cp", "-a", str(self.workspace), str(self.template)], check=True)
        wrapper = run_explore.write_kernel_sandbox(
            self.run, venv, self.agent, self.workspace, self.agent / "machines", memory_gb=0, template=self.template
        )
        text = wrapper.read_text().replace('exec bwrap "${args[@]}" "$REAL" "$@"', 'printf "%s\\n" "${args[@]}"')
        self.script = base / "wrapper-printing"
        self.script.write_text(text)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def start(self, session_dir: pathlib.Path | None) -> list[str]:
        env = {key: value for key, value in os.environ.items() if not key.startswith("RLM_")}
        if session_dir is not None:
            env["RLM_SESSION_DIR"] = str(session_dir)
        out = subprocess.run(["bash", str(self.script)], env=env, check=True, capture_output=True, text=True).stdout
        return out.splitlines()

    def bound_at(self, args: list[str], target: pathlib.Path) -> str:
        for index, arg in enumerate(args):
            if arg == "--bind" and args[index + 2] == str(target):
                return args[index + 1]
        raise AssertionError(f"nothing bound at {target}: {args}")

    def test_each_session_works_in_its_own_copy_of_the_pristine_workspace(self) -> None:
        artifacts = self.agent / "session-artifacts"
        root, player = artifacts / "root-1", artifacts / "root-1" / "session-artifacts" / "player-1"
        root_args, player_args = self.start(root), self.start(player)
        own_root = pathlib.Path(self.bound_at(root_args, self.workspace))
        own_player = pathlib.Path(self.bound_at(player_args, self.workspace))
        self.assertEqual((own_root, own_player), (self.run / "workspaces" / "root-1", self.run / "workspaces" / "player-1"))
        self.assertIn("--chdir", player_args)
        self.assertEqual(player_args[player_args.index("--chdir") + 1], str(self.workspace))
        # Seeded from the template; one session's writes stay its own.
        self.assertEqual((own_player / "AGENTS.md").read_text(), "system prompt\n")
        (own_player / "memory" / "notes.md").write_text("DIED on Dlvl 8\n")
        (self.template / "memory" / "late.md").write_text("never seeded\n")
        other = self.start(artifacts / "root-1" / "session-artifacts" / "player-2")
        own_other = pathlib.Path(self.bound_at(other, self.workspace))
        self.assertFalse((own_other / "memory" / "notes.md").exists())
        self.assertFalse((self.workspace / "memory" / "notes.md").exists())
        # A restart of the same session finds its copy, not a fresh one.
        self.start(player)
        self.assertEqual((own_player / "memory" / "notes.md").read_text(), "DIED on Dlvl 8\n")
        # No session dir (the kernel venv's setup probes): no workspace at all.
        anonymous = self.start(None)
        self.assertNotIn(str(self.workspace), anonymous)
        self.assertEqual(anonymous[anonymous.index("--chdir") + 1], "/tmp")
        self.assertEqual(sorted(path.name for path in (self.run / "workspaces").iterdir()), ["player-1", "player-2", "root-1"])

    def test_only_the_roots_kernel_sees_the_machines_dir(self) -> None:
        machines = str(self.agent / "machines")
        artifacts = self.agent / "session-artifacts"
        self.assertIn(machines, self.start(artifacts / "root-1"))
        self.assertNotIn(machines, self.start(artifacts / "root-1" / "session-artifacts" / "player-1"))
        self.assertNotIn(machines, self.start(None))


if __name__ == "__main__":
    unittest.main()
