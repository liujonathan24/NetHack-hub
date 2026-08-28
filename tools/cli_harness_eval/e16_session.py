#!/usr/bin/env python3
"""The E16 orchestrator's own Prime Agent conversation — one session, N rounds.

WHY A SESSION AND NOT N INDEPENDENT CALLS
-----------------------------------------
The v3 design's orchestrator was a scripted function and genuinely free. The
current design makes it an LM that opens with a planning discussion, then
chooses each round's checkpoint and writes the player a directive. That only
beats the scripted selector if it ACCUMULATES: "c12 looked promising twice and
both attempts died the same way" is a thing you can only say if round 9
remembers rounds 3 and 6. Re-prompting a fresh model with a rendered ledger
every round throws exactly that away, and would make the LLM arm a slower,
more expensive version of the scripted one.

E13's orchestrator is the opposite case and its reasoning does not transfer:
there, "nothing needs to survive in its head -- the store IS the state", so a
fresh `--print` process per round was right. Here the accumulated judgement IS
the treatment.

THE MECHANISM (verified against prime-agent 0.3.3 on this box)
--------------------------------------------------------------
Players stay exactly as they are. ``nethack_prime_agent`` builds their argv
with ``--print --no-session --offline`` (installed package,
``__init__.py:648-655``) and that is correct: each rollout is an independent
measurement and a shared session would be cross-contamination. Its own
docstring is explicit that a relaunch keeps the game but not the conversation
-- "only the agent's own conversation is gone" (``__init__.py:67-70``). This
module does not touch that path.

The orchestrator gets a separate one:

* ``--no-session`` is OMITTED. It is checked first in ``createSessionManager``
  (``dist/main.js:274-323``) and returns ``SessionManager.inMemory()``
  unconditionally, so passing it alongside ``--resume`` would silently discard
  the resume. That is the single flag standing between the player path and a
  persistent conversation.
* ``--print --mode json`` on the DISCOVERY round ONLY, and text ``--print``
  on every resumed round. Text ``--print`` never reveals the session id
  (``dist/modes/print-mode.js:80-95`` writes assistant text only), while json
  mode emits the session header as stdout line 1 (``print-mode.js:60-65``,
  ``docs/json.md:60-63``):
  ``{"type":"session","version":3,"id":"...","cwd":"..."}``. So json mode is
  how the id is discovered on round 1.

  IT IS NOT USED AFTER THAT, and that is a measured decision rather than a
  preference. The E16 model-in-the-loop sims found ``--mode json`` +
  ``--resume`` HANGING on a prompt that the identical text ``--print`` call --
  same session, same cwd -- answered 30 s later
  (``outputs/e16_sim/scripts/run_pa.py``, which consequently defaults json mode
  OFF). A hang is worse than an error by the width of a whole run's budget, so
  the mode that hangs is used exactly once, on the cheapest call, and never
  again.

  Everything json mode was doing on later rounds is recovered from the SESSION
  FILE instead, which is strictly more information than stdout carried:
  ``sessions/<file>.jsonl``'s first record is the header (``id`` AND ``cwd``),
  and the records appended by a round carry per-message ``usage`` with the
  provider's own ``cost.total`` in USD. So continuity is still CHECKED on every
  round -- a resumed round must touch the same file and that file's header id
  must still be ours, or the conversation forked and the run says so -- and the
  orchestrator's spend stops being a price-table estimate.
* ``--resume <id>`` from round 2. Explicit id, never ``-c/--continue``: the id
  goes into ``provenance.json`` so the conversation this run had is nameable
  afterwards, and a stray second session in the same directory cannot be
  picked up by accident. Never a BARE ``--resume`` -- with no argument it opens
  the interactive session picker (``main.js:306-318``), which in a
  non-interactive driver hangs.
* THE ID IS NOT THE FILENAME. ``sessions/<uuid>.jsonl``'s stem and the header's
  ``id`` disagree in all three sessions on this box, because ``setSessionFile``
  on an empty file mints a fresh uuid and then restores the old path
  (``dist/core/session-manager.js:736-746``); ``--resume`` matches the HEADER
  id (``session-resolver.js:31-52``). Discovering the id by "newest file in
  sessions/" is therefore a trap, and this module does not do it.
* A DEDICATED agent-state dir (``PRIME_AGENT_CODING_AGENT_DIR`` -- the name is
  built from the package name in ``dist/config.js:380-385``; there is no
  ``PRIME_AGENT_DIR`` and no ``--agent-dir``). Never ``/root/.prime/agent``,
  which holds ``daemon-workers/`` and ``session-leases/`` that a booting
  experiment has been measured reaping out from under another experiment's
  live rollouts.
* A FIXED cwd across rounds, ASSERTED rather than intended. ``--resume`` on a
  session whose recorded ``cwd`` matches resolves as ``"local"`` and opens the
  file directly; a different cwd takes the ``"global"`` branch, which asks
  "Fork this session into current directory?" through an interactive confirm
  (``main.js:311-318``). In a non-interactive driver that confirm HANGS
  FOREVER -- measured in the E16 sims. Two independent defences, because one
  silent hang costs the run:
  (1) the session's cwd is read out of the session file's header the moment the
      id is discovered, written to ``provenance.json``, and compared with the
      process's actual cwd BEFORE every resumed launch -- a mismatch raises
      :class:`SessionCwdMismatch` and no subprocess is started at all;
  (2) every launch gets ``stdin=DEVNULL`` and a hard timeout that kills the
      whole process GROUP, so an interactive prompt that slips past (1) reads
      EOF instead of blocking, and anything that still blocks dies at the
      deadline with :class:`RoundTimeout` naming the command.

* A PRIVATE ``TMPDIR``. The orchestrator runs UNSANDBOXED, so it shares
  ``/tmp/prime-agent-0/daemon.sock`` with every other prime-agent on this box.
  Measured in the sims: with another tenant's wedged processes on that socket a
  resumed round hung for 900 s and then 500 s; the identical call with
  ``TMPDIR`` pointed at a private directory returned in 3.5 s. Players do not
  need this -- bwrap ``--tmpfs /tmp`` already gives each rollout its own socket
  -- and this is why the variable is set on the ORCHESTRATOR's own env dict and
  never exported: an exported ``TMPDIR`` leaks into sandboxed rollouts where
  the path is not bound, and pointing it at a bind-mounted directory collapses
  concurrent rollouts onto one daemon socket again.

WHAT IS PROVEN AND WHAT IS NOT
------------------------------
Proven here, with no inference spent: ``--resume`` is honoured in ``--print``
mode and resolves BEFORE any provider call --
``prime-agent --print --resume 00000000-0000-0000-0000-000000000000 -- x``
exits 1 with "No session found matching ...". ``--mode json`` and
``--session-dir`` parse. Not proven without spending inference: that a resumed
session's history actually reaches the model.
:meth:`PrimeAgentSession.probe_resume_carries_history` is the two-call check
that settles it, its result is written into ``provenance.json`` as
``session_resume_verified``, and :class:`ReplaySession` is the fallback that
does not need the feature at all. Which one a run used is recorded, because
"the orchestrator remembered" is a claim about the run.

NOT RESIDENT
------------
Each round is a fresh short-lived process that resumes a file. This matters
operationally: the daemon-wedge recipe in this tree runs
``pkill -9 -f prime-agent`` before every cell, which would kill a resident
orchestrator between rounds. A resumed session survives it.

EVERY ROUND IS LOGGED
---------------------
Prompt, reply, argv, session id, exit code, wall clock and token usage go to
``orchestrator_rounds.jsonl``. The orchestrator's narration is evidence about
the orchestrator; it is never a measurement of the game.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

#: Default binary. Overridable, because the harness pins a version and a box
#: may carry more than one.
PRIME_AGENT_BIN = os.environ.get("PRIME_AGENT_BIN", "prime-agent")

#: The agent-state dir env var, built in `dist/config.js:380-385` from the
#: package name. NOT `PRIME_AGENT_DIR` (does not exist) and NOT `HOME`.
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"

#: Where an agent dir keeps conversations (`config.js:501-506`).
SESSIONS_SUBDIR = "sessions"

#: When ``--mode json`` is passed. `discovery_only` is the DEFAULT and the
#: measured-safe one: json mode is the only way to learn a session id, and it
#: was measured HANGING when combined with `--resume`, so it runs exactly once
#: on the round that has nothing to resume. `always` restores the pre-hardening
#: behaviour for anyone who wants to re-measure it; `never` is for a session
#: whose id is already known.
JSON_MODE_DISCOVERY_ONLY = "discovery_only"
JSON_MODE_ALWAYS = "always"
JSON_MODE_NEVER = "never"
JSON_MODE_POLICIES = frozenset({JSON_MODE_DISCOVERY_ONLY, JSON_MODE_ALWAYS,
                                JSON_MODE_NEVER})

#: E13's lesson, reused: `--model <pattern>` alone matched openrouter's catalog
#: entry first and died with "No API key found for openrouter" even though
#: settings.json names prime-inference. The provider is pinned, not inferred.
DEFAULT_PROVIDER = os.environ.get("ORCH_PROVIDER", "prime-inference")
DEFAULT_MODEL = os.environ.get("ORCH_MODEL", "z-ai/glm-5.2")

#: Files copied from the operator's agent dir into the orchestrator's private
#: one. Credentials for prime-inference live in ~/.prime/config.json, OUTSIDE
#: the agent dir, so a private agent dir does not lose them (E13 established
#: this).
_SEED_FILES = ("settings.json", "auth.json", "models.json")


@dataclass
class RoundResult:
    """One orchestrator turn."""

    text: str = ""
    session_id: str = ""
    exit_code: int = 0
    wall_s: float = 0.0
    usage: dict = field(default_factory=dict)
    spend_usd: float = 0.0
    stderr: str = ""
    argv: list = field(default_factory=list)
    error: str = ""
    #: Set when the session header id changed between rounds, i.e. the
    #: conversation forked and continuity was NOT what it claimed to be.
    continuity_broken: bool = False
    #: The call passed its hard deadline and its process group was killed. A
    #: distinct field from `error` because a timeout is the failure this whole
    #: module was hardened against, and it must be countable in the summary
    #: rather than buried in an error string.
    timed_out: bool = False
    #: ``--mode json`` was on for this call. True only on the discovery round.
    json_mode: bool = False
    #: The cwd the session file's header records, read from disk. The value
    #: every later round's cwd is asserted against.
    session_cwd: str = ""
    #: Which ``sessions/*.jsonl`` this round appended to.
    session_file: str = ""
    #: USD the provider itself reported for this round (session-file
    #: ``usage.cost.total``), as opposed to `spend_usd`'s price-table estimate.
    #: 0.0 when the session file carried no cost, which is reported as
    #: unavailable rather than as free.
    cost_usd_reported: float = 0.0


class RoundTimeout(RuntimeError):
    """One orchestrator call passed its deadline and was killed.

    Exists so a hang becomes an ERROR with a message naming the command and the
    deadline, instead of a process the run waits on forever. The three E16
    silent hangs (resume from a foreign cwd, the shared daemon socket, json
    mode + resume) all present identically to a driver: nothing happens, no
    output, no exit. A run that can only be rescued by a human noticing is not
    a run that can be launched with $700 behind it.
    """


class SessionCwdMismatch(RuntimeError):
    """A resumed round was about to launch from a cwd the session was not
    recorded in.

    Raised BEFORE the subprocess exists, because the failure mode on the other
    side of that launch is prime-agent's interactive "Fork this session into
    current directory?" confirm, which hangs a non-interactive driver forever.
    """


#: A runner executes ``(argv, env, cwd, timeout)`` and returns
#: ``(stdout, stderr, returncode)``. Injected so the whole session layer is
#: testable with no inference: the dry run passes a scripted runner and still
#: exercises argv assembly, id discovery, continuity checking, parsing and
#: logging.
Runner = Callable[[list, dict, str, float], tuple]


def _subprocess_runner(argv, env, cwd, timeout_s) -> tuple:
    """Run one orchestrator call under a HARD deadline, with no stdin.

    ``subprocess.run(timeout=...)`` is not enough on its own for either half of
    what this needs:

    * It kills only the direct child. prime-agent spawns a daemon worker, so a
      wedged call can leave the reaped parent's children holding the pipes and
      ``communicate()`` blocks past the deadline anyway. ``start_new_session``
      puts the call in its own process GROUP and the deadline kills the group.
    * It inherits stdin. Every interactive prompt prime-agent can raise -- the
      session picker on a bare ``--resume``, the "Fork this session into
      current directory?" confirm on a cwd mismatch -- blocks on a read that
      never returns when stdin is a terminal or an idle pipe. ``DEVNULL`` turns
      each of those into an immediate EOF.
    """
    proc = subprocess.Popen(
        argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except Exception:
            stdout, stderr = "", ""
        raise RoundTimeout(
            f"orchestrator call exceeded its {timeout_s:g}s deadline and was "
            f"killed (process group {proc.pid}). Command: "
            f"{' '.join(str(a) for a in argv[:8])} ... "
            f"[{len(argv)} argv items]. cwd={cwd}. "
            f"TMPDIR={env.get('TMPDIR', '(inherited)')}. "
            f"Last stderr: {(stderr or '')[-400:]!r}"
        ) from None
    return stdout, stderr, proc.returncode


def _kill_group(proc) -> None:
    """SIGKILL the call's whole process group; never raise."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=5)
            return
        except Exception:
            continue
    try:
        proc.kill()
    except Exception:
        pass


class SessionBase:
    """Common bookkeeping for both continuity strategies."""

    kind = "base"

    def __init__(self, *, work_dir, log_path=None, summary_cap: int = 1500,
                 context_chars: int = 400_000):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else self.work_dir / "rounds.jsonl"
        self.summary_cap = summary_cap
        self.context_chars = context_chars
        self.rounds = 0
        self.sent_chars = 0
        self.spend_usd = 0.0
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0,
                            "cached_input_tokens": 0}
        self.usage_available = False
        self.session_ids: list = []
        self.compactions = 0
        self.continuity_breaks = 0

    # -- context bounding -------------------------------------------------- #

    def cap_summary(self, text: str) -> str:
        """Trim player prose to the orchestrator's per-round allowance.

        The orchestrator's context is the only thing in this design that grows
        monotonically with run length, and an uncapped player session's account
        of its own game is the largest single contributor. The truncation is
        marked in the text, and the full report survives in the attempt record
        and in the checkpoint's ``lessons.md`` either way -- so the cap costs
        the orchestrator detail, never the dataset.
        """
        text = (text or "").strip()
        if len(text) <= self.summary_cap:
            return text
        return text[:max(0, self.summary_cap - 20)].rstrip() + " [...truncated]"

    def needs_compaction(self) -> bool:
        return self.sent_chars >= self.context_chars

    def budget_lines(self) -> dict:
        return {
            "rounds": self.rounds,
            "sent_chars": self.sent_chars,
            "compactions": self.compactions,
            "session_ids": list(self.session_ids),
            "continuity_breaks": self.continuity_breaks,
            "usage": dict(self.usage_total),
            "usage_available": self.usage_available,
            "spend_usd": round(self.spend_usd, 6),
            # The hardening, made countable. A run whose orchestrator timed out
            # four times had four rounds decided by the scripted fallback, and
            # that must be readable from the summary rather than from the log.
            "timeouts": getattr(self, "timeouts", 0),
            "cwd_asserts": getattr(self, "cwd_asserts", 0),
            "session_cwd": getattr(self, "session_cwd", ""),
            "tmpdir": str(getattr(self, "tmpdir", "")),
            "json_mode_policy": getattr(self, "json_mode_policy", ""),
            "cost_usd_reported": round(getattr(self, "cost_usd_reported", 0.0), 6),
        }

    # -- logging ----------------------------------------------------------- #

    def _log(self, kind: str, prompt: str, res: RoundResult) -> None:
        rec = {
            "round": self.rounds, "kind": kind, "session_kind": self.kind,
            "session_id": res.session_id, "exit_code": res.exit_code,
            "wall_s": round(res.wall_s, 2), "usage": res.usage,
            "spend_usd": round(res.spend_usd, 6), "error": res.error,
            "continuity_broken": res.continuity_broken,
            "timed_out": res.timed_out, "json_mode": res.json_mode,
            "session_cwd": res.session_cwd, "session_file": res.session_file,
            "cost_usd_reported": round(res.cost_usd_reported, 6),
            "argv": res.argv, "prompt": prompt, "reply": res.text,
            "sent_chars_cumulative": self.sent_chars, "t": time.time(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def seed_agent_dir(dest, src=None) -> list:
    """Give the orchestrator a private agent dir with the operator's settings.

    Copies ``settings.json``/``auth.json``/``models.json`` so ``--provider
    prime-inference`` resolves, without the orchestrator ever writing into
    ``~/.prime/agent`` -- which is the directory whose ``daemon-workers/`` and
    ``session-leases/`` a booting experiment was measured reaping from another
    experiment's live rollouts.
    """
    dest = Path(dest)
    src = Path(src or (Path.home() / ".prime" / "agent"))
    (dest / SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    copied = []
    for name in _SEED_FILES:
        s = src / name
        if s.is_file():
            shutil.copy2(s, dest / name)
            copied.append(name)
    return copied


class PrimeAgentSession(SessionBase):
    """A resumable Prime Agent conversation, driven one round at a time."""

    kind = "prime_agent_session"

    def __init__(self, *, work_dir, agent_dir, log_path=None,
                 binary: str = PRIME_AGENT_BIN,
                 model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER,
                 timeout_s: float = 900.0, runner: Optional[Runner] = None,
                 env: Optional[dict] = None, extra_args: Optional[list] = None,
                 summary_cap: int = 1500, context_chars: int = 400_000,
                 kernel_venv: str = "", tmpdir=None,
                 json_mode_policy: str = JSON_MODE_DISCOVERY_ONLY):
        super().__init__(work_dir=work_dir, log_path=log_path,
                         summary_cap=summary_cap, context_chars=context_chars)
        self.agent_dir = Path(agent_dir)
        (self.agent_dir / SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
        self.binary = binary
        self.model = model
        self.provider = provider
        self.timeout_s = timeout_s
        self.runner = runner or _subprocess_runner
        self.base_env = env
        self.extra_args = list(extra_args or [])
        self.kernel_venv = kernel_venv
        if json_mode_policy not in JSON_MODE_POLICIES:
            raise ValueError(f"json_mode_policy must be one of "
                             f"{sorted(JSON_MODE_POLICIES)}, not "
                             f"{json_mode_policy!r}")
        self.json_mode_policy = json_mode_policy
        #: PRIVATE temp dir. Defaults under the agent dir, which is already
        #: private per-run, so the daemon socket cannot be shared with another
        #: tenant even if the caller forgets to pass one.
        self.tmpdir = Path(tmpdir) if tmpdir else (self.agent_dir / "tmp")
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.session_id: str = ""
        #: The cwd the session's own header records. Discovered on round 1 and
        #: asserted before every resumed launch.
        self.session_cwd: str = ""
        self.session_file: str = ""
        self.timeouts = 0
        self.cwd_asserts = 0
        #: Provider-reported USD, summed from the session file. Kept beside the
        #: price-table estimate rather than replacing it, so a divergence
        #: between what we model and what we are billed is visible.
        self.cost_usd_reported = 0.0

    # -- the fixed cwd ----------------------------------------------------- #

    @property
    def launch_cwd(self) -> str:
        """The one directory every round of this session launches from."""
        return str(self.work_dir.resolve())

    def assert_cwd(self) -> None:
        """Refuse a resumed launch from anywhere but the session's own cwd.

        The check that costs nothing and saves the run. A ``--resume`` whose
        cwd does not match the session's recorded one takes prime-agent's
        "global" branch and asks an interactive confirm that a non-interactive
        driver never answers -- so this raises instead, BEFORE the subprocess
        is created. Called on every round, not only after a suspicious one:
        the mismatch that kills a run is the one nobody suspected.
        """
        self.cwd_asserts += 1
        if not self.session_cwd:
            return  # nothing to resume yet; round 1 is where cwd is recorded
        actual = self.launch_cwd
        if os.path.realpath(actual) != os.path.realpath(self.session_cwd):
            raise SessionCwdMismatch(
                f"REFUSING to resume session {self.session_id} from the wrong "
                f"directory. The session's own header records "
                f"cwd={self.session_cwd!r}; this process would launch from "
                f"{actual!r}. prime-agent resolves that as a foreign session "
                f"and asks an interactive 'Fork this session into current "
                f"directory?' confirm, which hangs a non-interactive driver "
                f"forever. Re-run the orchestrator from {self.session_cwd!r}."
            )

    # -- argv -------------------------------------------------------------- #

    def use_json_mode(self) -> bool:
        """Whether THIS round runs ``--mode json``.

        Default policy ``discovery_only``: json mode exactly once, on the round
        that has no session id yet and therefore no other way to learn one.
        Every resumed round runs plain text ``--print``, because json mode +
        ``--resume`` was measured hanging where the same text call succeeded.
        """
        if self.json_mode_policy == JSON_MODE_ALWAYS:
            return True
        if self.json_mode_policy == JSON_MODE_NEVER:
            return False
        return not self.session_id

    def build_argv(self, prompt: str) -> list:
        """The orchestrator's command line. See the module docstring for why.

        ``--`` before the prompt, matching the harness (``__init__.py:670``):
        without it a prompt that begins with a dash is parsed as a flag.
        """
        argv = [self.binary, "--print"]
        if self.use_json_mode():
            argv += ["--mode", "json"]
        argv += ["--provider", self.provider, "--model", self.model]
        if self.session_id:
            argv += ["--resume", self.session_id]
        argv += self.extra_args
        argv += ["--", prompt]
        return argv

    def _env(self) -> dict:
        env = dict(self.base_env if self.base_env is not None else os.environ)
        env[ENV_AGENT_DIR] = str(self.agent_dir)
        env.setdefault("PI_SKIP_VERSION_CHECK", "1")
        env.setdefault("PI_TELEMETRY", "0")
        # PRIVATE DAEMON SOCKET. Set on THIS dict only -- never exported, never
        # written into os.environ -- because the player rollouts inherit the
        # real environment and a TMPDIR they cannot see inside their bwrap
        # sandbox breaks them. See the module docstring: 900s -> 3.5s.
        env["TMPDIR"] = str(self.tmpdir)
        if self.kernel_venv:
            # ORCH_KERNEL_VENV's lesson: the PRIME_ name leaks into sandboxed
            # rollouts where the path is not bound, so the ORCHESTRATOR sets it
            # for itself here rather than exporting it into the world.
            env["PRIME_AGENT_KERNEL_VENV"] = self.kernel_venv
        return env

    # -- the session file, which is where the truth lives ------------------ #

    @property
    def sessions_dir(self) -> Path:
        return self.agent_dir / SESSIONS_SUBDIR

    def _snapshot(self) -> dict:
        """``{filename: size}`` for every session file, before a round."""
        if not self.sessions_dir.is_dir():
            return {}
        out = {}
        for p in self.sessions_dir.glob("*.jsonl"):
            try:
                out[p.name] = p.stat().st_size
            except OSError:
                continue
        return out

    def _scan_session_files(self, before: dict) -> dict:
        """What one round did to the session files on disk.

        Returns ``{"file", "header_id", "cwd", "usage", "cost_usd",
        "records"}``. This is how a TEXT-mode round recovers everything json
        mode used to print: the header id (first record of the file, and NOT
        the filename stem -- ``setSessionFile`` mints a fresh uuid for the
        header and keeps the old path), the cwd the session was opened in, and
        the per-message ``usage``/``cost`` the provider reported.
        """
        out = {"file": "", "header_id": "", "cwd": "", "records": [],
               "usage": {}, "cost_usd": 0.0}
        after = self._snapshot()
        touched = sorted(n for n, sz in after.items() if before.get(n) != sz)
        if not touched:
            return out
        # Prefer OUR file when several moved (another tenant should not be
        # writing here at all, but the private agent dir is a policy, not a
        # kernel guarantee).
        name = touched[0]
        if self.session_file and self.session_file in touched:
            name = self.session_file
        path = self.sessions_dir / name
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return out
        lines = text.splitlines()
        if not lines:
            return out
        out["file"] = name
        try:
            head = json.loads(lines[0])
            if isinstance(head, dict) and head.get("type") == "session":
                out["header_id"] = str(head.get("id") or "")
                out["cwd"] = str(head.get("cwd") or "")
        except json.JSONDecodeError:
            pass
        # Only the records this round appended.
        prior = before.get(name, 0)
        tail = text[prior:] if prior and prior <= len(text) else (
            "" if prior else text)
        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "cached_input_tokens": 0}
        found = False
        cost = 0.0
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out["records"].append(rec)
            u = (rec.get("message") or {}).get("usage") if isinstance(rec, dict) else None
            if not isinstance(u, dict):
                continue
            found = True
            usage["prompt_tokens"] += int(u.get("input") or u.get("prompt_tokens") or 0)
            usage["completion_tokens"] += int(u.get("output") or u.get("completion_tokens") or 0)
            usage["cached_input_tokens"] += int(
                u.get("cacheRead") or u.get("cached_input_tokens") or 0)
            c = u.get("cost")
            if isinstance(c, dict):
                cost += float(c.get("total") or 0.0)
        out["usage"] = usage if found else {}
        out["cost_usd"] = cost
        return out

    # -- one round --------------------------------------------------------- #

    def ask(self, prompt: str, *, kind: str = "round") -> RoundResult:
        # GUARD FIRST, LAUNCH SECOND. The cwd assertion runs before the round
        # counter moves and before any subprocess exists, because the thing it
        # prevents is unkillable-by-timeout in the worst case and always
        # invisible: a driver waiting on a confirm nobody will type.
        self.assert_cwd()
        self.rounds += 1
        self.sent_chars += len(prompt)
        json_mode = self.use_json_mode()
        argv = self.build_argv(prompt)
        before = self._snapshot()
        t0 = time.time()
        res = RoundResult(argv=argv, json_mode=json_mode)
        try:
            stdout, stderr, rc = self.runner(argv, self._env(),
                                             self.launch_cwd, self.timeout_s)
        except RoundTimeout as exc:
            self.timeouts += 1
            res.timed_out = True
            res.error = f"RoundTimeout: {exc}"
            res.exit_code = -1
            res.wall_s = time.time() - t0
            res.session_id = self.session_id
            self._log(kind, prompt, res)
            return res
        except subprocess.TimeoutExpired as exc:
            # An injected runner that used subprocess.run's own timeout.
            self.timeouts += 1
            res.timed_out = True
            res.error = (f"RoundTimeout: orchestrator call exceeded its "
                         f"{self.timeout_s:g}s deadline ({exc})")
            res.exit_code = -1
            res.wall_s = time.time() - t0
            res.session_id = self.session_id
            self._log(kind, prompt, res)
            return res
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            res.exit_code = -1
            res.wall_s = time.time() - t0
            res.session_id = self.session_id
            self._log(kind, prompt, res)
            return res
        res.wall_s = time.time() - t0
        res.exit_code = rc
        res.stderr = stderr or ""
        header, text, usage = parse_json_mode_stdout(stdout or "")
        res.text = text
        res.usage = usage
        if rc != 0 and not res.error:
            res.error = f"prime-agent exited {rc}: {(stderr or '')[:300]}"

        # THE SESSION FILE IS THE AUTHORITY. json mode's stdout header agrees
        # with it on the discovery round; on a text-mode round it is the only
        # source, and it carries the cwd and the provider's own cost besides.
        scan = self._scan_session_files(before)
        res.session_file = scan["file"]
        res.session_cwd = scan["cwd"]
        if not usage and scan["usage"]:
            usage = scan["usage"]
            res.usage = usage
        res.cost_usd_reported = float(scan["cost_usd"] or 0.0)
        self.cost_usd_reported += res.cost_usd_reported

        new_id = (header or {}).get("id") or scan["header_id"] or ""
        if new_id:
            if not self.session_id:
                self.session_id = new_id
                self.session_ids.append(new_id)
                # RECORD THE CWD THE SESSION WAS BORN IN. Everything after this
                # round is asserted against it.
                self.session_cwd = scan["cwd"] or self.launch_cwd
                self.session_file = scan["file"]
            elif new_id != self.session_id:
                # The conversation forked. Recorded loudly: a run that believes
                # it had one long conversation and actually had two is making a
                # false claim about its own treatment.
                res.continuity_broken = True
                self.continuity_breaks += 1
                self.session_id = new_id
                self.session_ids.append(new_id)
                self.session_file = scan["file"] or self.session_file
        elif self.session_id and scan["file"] and self.session_file \
                and scan["file"] != self.session_file:
            # No header id recoverable, but a DIFFERENT file grew. In text mode
            # that is the only shape a silent fork can take, and reporting a
            # continuity we did not verify is the failure this module exists to
            # avoid.
            res.continuity_broken = True
            self.continuity_breaks += 1
            self.session_file = scan["file"]
        res.session_id = self.session_id
        res.spend_usd = self._price(usage)
        self.spend_usd += res.spend_usd
        if usage:
            self.usage_available = True
            for k in self.usage_total:
                self.usage_total[k] += int(usage.get(k) or 0)
        self._log(kind, prompt, res)
        return res

    # -- spend ------------------------------------------------------------- #

    def _price(self, usage: dict) -> float:
        """USD for one round's usage. 0.0 when usage is UNAVAILABLE, never
        estimated -- ``usage_available`` in the summary says which it was. A
        fabricated orchestrator spend is worse than an unknown one, because the
        budget's second line would then be fiction."""
        if not usage:
            return 0.0
        try:
            from aggregate import PRICE_TABLES, PRICE_TABLE_GLM_5_2
            price = PRICE_TABLES.get(self.model) or PRICE_TABLE_GLM_5_2
        except Exception:
            return 0.0
        return (int(usage.get("prompt_tokens") or 0) * price["input_per_million"]
                + int(usage.get("cached_input_tokens") or 0) * price["cached_input_per_million"]
                + int(usage.get("completion_tokens") or 0) * price["output_per_million"]
                ) / 1_000_000

    # -- the feasibility probe --------------------------------------------- #

    PROBE_ROUND_1 = ("Remember this codeword exactly and reply with only the "
                     "word OK: {codeword}")
    PROBE_ROUND_2 = ("What codeword did I ask you to remember? Reply with only "
                     "that word.")

    def probe_resume_carries_history(self, codeword: str = "ZORKMID7") -> dict:
        """Prove (or disprove) that ``--resume`` carries history under ``--print``.

        Round 1 states a codeword in a fresh session; round 2 resumes it by id
        and asks for the codeword back. If it comes back, history carries and
        this class is usable. If it does not, :class:`ReplaySession` is the
        fallback -- which is why the answer is measured and written into
        provenance instead of assumed.

        COSTS TWO SMALL MODEL CALLS, and is the only thing in the E16 build
        that must call a model at all. Static reading has already established
        everything it can: ``--resume`` IS honoured under ``--print`` (a bogus
        id exits 1 before any provider call). What it cannot establish is
        whether the resumed history reaches the completion.
        """
        r1 = self.ask(self.PROBE_ROUND_1.format(codeword=codeword), kind="probe1")
        r2 = self.ask(self.PROBE_ROUND_2, kind="probe2")
        carried = codeword.lower() in (r2.text or "").lower()
        return {
            "verified": bool(carried and not r1.error and not r2.error
                             and not r2.continuity_broken),
            "carried_history": carried,
            "session_id_round1": r1.session_id,
            "session_id_round2": r2.session_id,
            "same_session": bool(r1.session_id and r1.session_id == r2.session_id),
            "round1_error": r1.error, "round2_error": r2.error,
            "round2_reply": (r2.text or "")[:400],
            "argv_round2": r2.argv,
        }


def parse_json_mode_stdout(stdout: str) -> tuple:
    """``(session_header, assistant_text, usage)`` from ``--mode json`` output.

    Line 1 is the session header (``docs/json.md:60-63``); the rest are records
    whose shapes this parser deliberately treats loosely, because it is reading
    a third-party protocol that has already changed version (the session file
    header says ``"version":3``). Anything it cannot recognise is skipped
    rather than guessed at, and unavailable usage comes back as ``{}`` so the
    caller reports it as unavailable instead of as zero.
    """
    header = None
    texts: list = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_input_tokens": 0}
    found_usage = False
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # Text that is not a record: keep it, so a mode that degrades to
            # plain text still yields the reply instead of nothing.
            texts.append(line)
            continue
        if not isinstance(rec, dict):
            continue
        if header is None and rec.get("type") == "session":
            header = rec
            continue
        u = _find_usage(rec)
        if u:
            found_usage = True
            for dst, srcs in (("prompt_tokens", ("prompt_tokens", "input_tokens")),
                              ("completion_tokens", ("completion_tokens", "output_tokens")),
                              ("cached_input_tokens", ("cached_input_tokens",
                                                       "cache_read_input_tokens",
                                                       "cached_tokens"))):
                for s in srcs:
                    if u.get(s) is not None:
                        usage[dst] += int(u[s] or 0)
                        break
        txt = _assistant_text(rec)
        if txt:
            texts.append(txt)
    return header, "\n".join(texts).strip(), (usage if found_usage else {})


def _assistant_text(rec: dict) -> str:
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else None
    if msg is None and rec.get("role"):
        msg = rec
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content
                 if isinstance(c, dict) and c.get("type") in (None, "text")]
        return "".join(p for p in parts if p)
    return ""


def _find_usage(rec) -> Optional[dict]:
    """Any ``usage``-shaped dict inside one record, at any depth."""
    if isinstance(rec, dict):
        u = rec.get("usage")
        if isinstance(u, dict):
            return u
        for v in rec.values():
            got = _find_usage(v)
            if got:
                return got
    elif isinstance(rec, list):
        for v in rec:
            got = _find_usage(v)
            if got:
                return got
    return None


class ReplaySession(SessionBase):
    """Fallback: keep the conversation ourselves and replay it as the prompt.

    Used when the probe says ``--resume`` does not carry history under
    ``--print``, or when a run deliberately wants continuity that does not
    depend on a third-party feature.

    ITS LIMITS, STATED. This is NOT a resumed session: the model reads a
    transcript of a conversation rather than having had it; there is no
    server-side prompt cache to inherit, so it is MORE expensive per round, not
    less; and the replay is text this module composed, not the exact bytes of
    the original turns. It is continuity of CONTENT, not of session. A run that
    used it says so in provenance.
    """

    kind = "replay"

    def __init__(self, *, work_dir, ask_fn, log_path=None,
                 summary_cap: int = 1500, context_chars: int = 400_000):
        super().__init__(work_dir=work_dir, log_path=log_path,
                         summary_cap=summary_cap, context_chars=context_chars)
        self.ask_fn = ask_fn
        self.history: list = []

    def ask(self, prompt: str, *, kind: str = "round") -> RoundResult:
        self.rounds += 1
        replayed = self.render_history() + prompt
        self.sent_chars += len(replayed)
        t0 = time.time()
        res = RoundResult()
        try:
            res.text = self.ask_fn(replayed) or ""
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
        res.wall_s = time.time() - t0
        self.history.append(("user", prompt))
        self.history.append(("assistant", res.text))
        self._log(kind, replayed, res)
        return res

    def render_history(self) -> str:
        if not self.history:
            return ""
        parts = ["=== CONVERSATION SO FAR (replayed; you said these things) ==="]
        for role, text in self.history:
            parts.append(f"[{role}] {text}")
        parts.append("=== END OF CONVERSATION SO FAR ===\n")
        return "\n".join(parts) + "\n"

    def compact(self, summary: str) -> None:
        self.history = [("assistant", summary)]
        self.compactions += 1


# --------------------------------------------------------------------------- #
# parsing the orchestrator's decision
# --------------------------------------------------------------------------- #

#: The orchestrator answers each round with a JSON object. Prose around it is
#: fine and is kept as the round's narration.
_JSON_BLOCK = re.compile(r"\{[^{}]*\"checkpoint\"[^{}]*\}", re.S)


@dataclass
class Decision:
    """What the orchestrator decided, AFTER validation against the archive."""

    checkpoint_id: Optional[str] = None
    directive: str = ""
    rationale: str = ""
    raw: str = ""
    parsed: bool = False
    valid: bool = False
    fallback_reason: str = ""


def parse_decision(text: str, valid_ids) -> Decision:
    """Extract ``{"checkpoint": ..., "directive": ...}`` and VALIDATE it.

    Validation is the integrity-critical half. The orchestrator is a language
    model: it can name a checkpoint that does not exist, invent an id, or
    answer in prose. None of those may become a silent no-op or a wrong resume,
    so an unparseable or unknown id yields ``valid=False`` with a reason and the
    caller falls back to the scripted selector -- recorded as a fallback, never
    as an LLM choice.

    Note what does NOT cross from model text into the run: any number. The
    orchestrator may write "c12 is at Dlvl 14"; the ledger's Dlvl for c12 comes
    from ``meta.json``, which the harness computed from engine blstats. The only
    things that cross are an ID (checked against the archive) and a DIRECTIVE
    (carried as text, shown to the player, never measured).
    """
    d = Decision(raw=text or "")
    valid_ids = {str(v) for v in valid_ids}
    if not (text or "").strip():
        d.fallback_reason = "empty orchestrator reply"
        return d
    obj = None
    for match in _JSON_BLOCK.finditer(text):
        try:
            obj = json.loads(match.group(0))
            break
        except json.JSONDecodeError:
            continue
    if obj is None:
        m = re.search(r"checkpoint\W{0,4}(c?\d+)", text, re.I)
        if m:
            obj = {"checkpoint": m.group(1)}
            dm = re.search(r"directive\W{0,4}(.+)", text, re.I)
            if dm:
                obj["directive"] = dm.group(1).strip().strip('"').strip()
    if not isinstance(obj, dict):
        d.fallback_reason = "no checkpoint decision found in reply"
        return d
    d.parsed = True
    d.directive = str(obj.get("directive") or "").strip()
    d.rationale = str(obj.get("rationale") or "").strip()
    raw_id = str(obj.get("checkpoint") or "").strip()
    ident = raw_id[1:] if (raw_id[:1].lower() == "c" and raw_id[1:].isdigit()) else raw_id
    if ident in valid_ids:
        d.checkpoint_id = ident
        d.valid = True
    else:
        d.fallback_reason = (f"orchestrator chose checkpoint {raw_id!r}, which is "
                             f"not in the archive")
    return d


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "Decision",
    "ENV_AGENT_DIR",
    "JSON_MODE_ALWAYS",
    "JSON_MODE_DISCOVERY_ONLY",
    "JSON_MODE_NEVER",
    "JSON_MODE_POLICIES",
    "PRIME_AGENT_BIN",
    "PrimeAgentSession",
    "ReplaySession",
    "RoundResult",
    "RoundTimeout",
    "SessionBase",
    "SessionCwdMismatch",
    "parse_decision",
    "parse_json_mode_stdout",
    "seed_agent_dir",
]
