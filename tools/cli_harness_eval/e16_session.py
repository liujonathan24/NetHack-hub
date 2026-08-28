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

#: How much of a round's raw stdout is kept on the record. Generous, because
#: the thing it exists to preserve is the evidence for an extraction failure,
#: and one pilot round's stdout was 2.6 MB of it.
RAW_STDOUT_CAP = 200_000


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
    #: The subprocess's RAW stdout, capped. Kept because `text` is the output
    #: of a parser, and every failure this module has actually had was a
    #: disagreement between the two: the E16 pilot's round-1 directive was in
    #: stdout and not in `text`, and its opening plan was 11,478 chars in
    #: stdout and 2,656,190 chars in `text`. When extraction fails, the bytes
    #: it failed on are the evidence, so they go on the record.
    raw_stdout: str = ""


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

    def __init__(self, *, work_dir, log_path=None, summary_cap: int = 1500):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else self.work_dir / "rounds.jsonl"
        self.summary_cap = summary_cap
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

    # `needs_compaction` IS GONE, and with it the 400k-sent-chars trigger and
    # the hand-off prompt it fired. Compaction is prime-agent's own: see
    # `configure_native_compaction`, which writes the CLI's `settings.compaction`
    # so its threshold check (`shouldCompact`: contextTokens > contextWindow -
    # reserveTokens) fires at a hard 128K-token bound. `compactions` below is
    # kept only as a counter for compactions we can OBSERVE, and is no longer
    # something this module causes.

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
            # THE TWO LENGTHS, ALWAYS. `reply` is a parser's output; when it
            # disagrees with the bytes it was parsed from, that disagreement is
            # the whole story, and reading it out of the log should not require
            # re-running the parser. The pilot's round-1 stdout was 1,104 chars
            # and its `reply` 678; its opening stdout was ~2.7 MB and its
            # `reply` 2,656,190. Either number alone hides both failures.
            "reply_chars": len(res.text or ""),
            "raw_stdout_chars": len(res.raw_stdout or ""),
            "sent_chars_cumulative": self.sent_chars, "t": time.time(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


#: THE HARD CONTEXT BOUND for the orchestrator's conversation, in tokens.
#: 128K. Compaction fires when the session's context exceeds it.
CONTEXT_LIMIT_TOKENS = int(os.environ.get("ORCH_CONTEXT_LIMIT_TOKENS",
                                          128 * 1024))

#: Tokens of recent conversation prime-agent's compactor keeps verbatim. Its
#: own default; named here so the written settings are complete rather than
#: half-specified.
COMPACTION_KEEP_RECENT_TOKENS = 20_000


def model_context_window(model: str = DEFAULT_MODEL,
                         provider: str = DEFAULT_PROVIDER,
                         binary: str = PRIME_AGENT_BIN) -> Optional[int]:
    """The EXACT context window prime-agent will use for ``provider/model``.

    Read out of the CLI's own bundled model catalog, because the number has to
    be exact and no interface exposes it exactly. ``prime-agent model list``
    prints it rounded ("1.0M"), and rounding DOWN a 1048576-token window to
    1000000 would push the compaction threshold ~48K tokens above the bound we
    are trying to enforce -- the one direction of error that matters, since the
    trigger is ``contextWindow - reserveTokens``.

    Returns ``None`` when the catalog cannot be read or the model is not in it.
    A caller that gets ``None`` must say so rather than guess: an over-estimate
    of the window compacts late, and an under-estimate can make the threshold
    negative and compact on every single turn.
    """
    try:
        real = Path(shutil.which(binary) or binary).resolve()
    except Exception:
        return None
    bundle = real.parent
    if not bundle.is_dir():
        return None
    # The catalog is `"<provider>": { ... "<model id>": { ... contextWindow: N }`
    # in one of the bundle chunks. Anchored on the model id AND the provider
    # field inside its own object, so a same-named model under another provider
    # (there are several `glm-5.2`s) cannot answer for this one.
    key = re.compile(r'"' + re.escape(model) + r'":\s*\{')
    for js in sorted(bundle.glob("*.js")):
        try:
            text = js.read_text(errors="replace")
        except OSError:
            continue
        if model not in text:
            continue
        for m in key.finditer(text):
            body = _brace_body(text, m.end() - 1)
            if body is None or f'provider: "{provider}"' not in body:
                continue
            cw = re.search(r'contextWindow:\s*([0-9][0-9_.]*(?:e[0-9]+)?)', body)
            if not cw:
                continue
            try:
                return int(float(cw.group(1).replace("_", "")))
            except ValueError:
                continue
    return None


def _brace_body(text: str, open_idx: int, limit: int = 8000) -> Optional[str]:
    """The text between ``text[open_idx] == '{'`` and its matching ``}``.

    Brace-matched rather than regex-terminated because a model entry contains a
    nested ``cost: { ... }``, and any non-greedy "up to the next closing brace"
    pattern stops inside it -- before ``contextWindow``, which is the one field
    this is read for.
    """
    if open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    for i in range(open_idx, min(len(text), open_idx + limit)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return None


def configure_native_compaction(agent_dir, *, model: str = DEFAULT_MODEL,
                                provider: str = DEFAULT_PROVIDER,
                                binary: str = PRIME_AGENT_BIN,
                                limit_tokens: int = CONTEXT_LIMIT_TOKENS
                                ) -> dict:
    """Make prime-agent compact THIS session at a hard ``limit_tokens`` bound.

    THE CLI'S OWN COMPACTION, not ours. prime-agent has native threshold
    compaction: ``settings.compaction = {enabled, reserveTokens,
    keepRecentTokens}``, and its check is

        shouldCompact(contextTokens, contextWindow, s) =
            s.enabled and contextWindow > 0 and
            contextTokens > contextWindow - s.reserveTokens

    so the only knob that sets an absolute bound is ``reserveTokens``. Setting
    it to ``contextWindow - limit_tokens`` makes the threshold exactly
    ``limit_tokens``. Nothing here writes a summarization prompt: what the
    compactor keeps and how it summarizes is the CLI's business.

    Written into the orchestrator's PRIVATE agent dir, never the operator's.

    Returns the record of what was done, including the failure cases, for
    provenance. When the window cannot be discovered, or is already at or below
    the bound, the CLI's own defaults are left alone and ``enforced`` is False
    -- a run that could not enforce its bound has to say so rather than write a
    reserve computed from a guessed window.
    """
    agent_dir = Path(agent_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "settings.json"
    try:
        settings = json.loads(path.read_text())
        if not isinstance(settings, dict):
            settings = {}
    except (OSError, json.JSONDecodeError):
        settings = {}

    cw = model_context_window(model, provider, binary)
    rec = {
        "mechanism": "prime-agent native threshold compaction "
                     "(settings.compaction), configured -- no summarization "
                     "prompt of ours",
        "model": model, "provider": provider,
        "context_window_tokens": cw,
        "limit_tokens": int(limit_tokens),
        "settings_path": str(path),
    }
    if cw is None:
        rec.update(enforced=False,
                   why="prime-agent's model catalog did not yield an exact "
                       "contextWindow for this provider/model, and a reserve "
                       "computed from a guessed window would enforce the wrong "
                       "bound. The CLI's default compaction (enabled, reserve "
                       "16384) is left in place.")
        return rec
    if cw <= limit_tokens:
        rec.update(enforced=False, reserve_tokens=None,
                   why=f"the model's own context window ({cw} tokens) is at or "
                       f"below the {limit_tokens}-token bound, so the bound "
                       f"cannot bind. The CLI's default compaction is left in "
                       f"place.")
        return rec

    reserve = int(cw) - int(limit_tokens)
    settings["compaction"] = {
        "enabled": True,
        "reserveTokens": reserve,
        "keepRecentTokens": COMPACTION_KEEP_RECENT_TOKENS,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    rec.update(enforced=True, reserve_tokens=reserve,
               keep_recent_tokens=COMPACTION_KEEP_RECENT_TOKENS,
               compacts_when="context tokens exceed "
                             f"{cw} - {reserve} = {limit_tokens}")
    return rec


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
                 summary_cap: int = 1500,
                 kernel_venv: str = "", tmpdir=None,
                 json_mode_policy: str = JSON_MODE_DISCOVERY_ONLY):
        super().__init__(work_dir=work_dir, log_path=log_path,
                         summary_cap=summary_cap)
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
        # PARSED ACCORDING TO THE MODE THAT WAS ACTUALLY REQUESTED. Handing
        # plain `--print` output to the json record parser is what deleted the
        # pilot's directive line -- see `parse_round_stdout`.
        header, text, usage = parse_round_stdout(stdout or "", json_mode)
        res.text = text
        res.raw_stdout = (stdout or "")[:RAW_STDOUT_CAP]
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


#: WHERE ASSISTANT TEXT MAY BE READ FROM IN ``--mode json``, narrowest first,
#: and ONLY ONE OF THEM IS EVER READ.
#:
#: Everything else in that stream is a partial: ``message_update`` re-emits the
#: WHOLE message on every text delta (``docs/json.md``, "Output Format"). And
#: even the complete ones overlap -- ``turn_end`` repeats the message
#: ``message_end`` just delivered, ``agent_end`` repeats the whole conversation
#: -- so a parser that reads more than one tier double-counts BOTH the text and
#: the usage that rides with it. The first tier carrying any assistant text
#: wins; the rest are ignored. ``loose`` is the session-file/legacy shape,
#: which has no event types at all, plus any line that was not JSON.
#:
#: MEASURED, not feared. The E16 GE-wiki pilot's opening round produced ONE
#: 11,478-char plan -- the session file has it once, at
#: ``sessions/01a04847-…jsonl`` record 44 -- and the old parser, which
#: concatenated the text of every record, turned it into 2,656,190 chars: 533
#: cumulative prefixes of the same plan, 589 unique lines out of 18,088. The
#: run read that as the model stuck in a repetition loop and threw the opening
#: plan away. The model was fine; the parser was not.
_TEXT_TIERS = ("message_end", "turn_end", "agent_end", "loose")

_USAGE_FIELDS = (
    ("prompt_tokens", ("prompt_tokens", "input_tokens")),
    ("completion_tokens", ("completion_tokens", "output_tokens")),
    ("cached_input_tokens", ("cached_input_tokens", "cache_read_input_tokens",
                             "cached_tokens")),
)


def _collapse_stream_texts(texts: list) -> list:
    """Drop cumulative prefixes and exact repeats from a streamed text list.

    A BACKSTOP, deliberately kept even though `_TEXT_TIERS` already excludes
    the partials that cause the blow-up. The protocol this
    reads is third-party and has already changed version once; if a future
    shape streams partials under a type this module does not know, the failure
    should be a slightly odd reply, not a 2.6 MB one that reads as a model
    defect.
    """
    out: list = []
    for t in texts:
        if not t:
            continue
        if out:
            last = out[-1]
            if t.startswith(last):      # this chunk supersedes the last one
                out[-1] = t
                continue
            if last.startswith(t):      # a shorter re-emission of the same text
                continue
        if t in out:
            continue
        out.append(t)
    return out


def _accumulate_usage(into: dict, rec) -> bool:
    u = _find_usage(rec)
    if not u:
        return False
    for dst, srcs in _USAGE_FIELDS:
        for s in srcs:
            if u.get(s) is not None:
                into[dst] += int(u[s] or 0)
                break
    return True


def parse_round_stdout(stdout: str, json_mode: bool) -> tuple:
    """``(header, assistant_text, usage)`` for ONE round, given its mode.

    THE BUG THIS CLOSES, and it is the one that cost the E16 pilot its central
    mechanism. Plain ``--print`` writes assistant text and nothing else
    (``dist/modes/print-mode.js:80-95``) -- there are no records to parse. Every
    round after the discovery round runs in that mode (see
    :meth:`PrimeAgentSession.use_json_mode`), and this module used to hand that
    plain text to the json-mode record parser anyway. The parser reads
    LINE BY LINE and treats any line that is a JSON object as a protocol record;
    a record with no assistant role and no usage is skipped as unrecognised.

    The orchestrator's round prompt asks for exactly that shape: "a short
    rationale and then EXACTLY this JSON object ON ITS OWN LINE". So the
    decision line -- ``{"checkpoint": "1", "directive": "…"}`` -- was parsed as
    a record, recognised as nothing, and DELETED, leaving only the rationale
    prose. `parse_decision` then found no decision, and the run fell back to
    the scripted selector with no directive. Measured in the pilot: the model's
    reply was 1,104 chars ending in a well-formed decision object; the round
    log recorded 678 chars ending at the blank line before it, and attempt 1
    was launched with "(orchestrator produced no directive for this attempt)".

    A directive-carrying experiment that silently serves no directive is its
    own control. Text mode is therefore parsed as text.
    """
    if not json_mode:
        return None, (stdout or "").strip(), {}
    return parse_json_mode_stdout(stdout)


def parse_json_mode_stdout(stdout: str) -> tuple:
    """``(session_header, assistant_text, usage)`` from ``--mode json`` output.

    Line 1 is the session header (``docs/json.md:60-63``); the rest are records
    whose shapes this parser deliberately treats loosely, because it is reading
    a third-party protocol that has already changed version (the session file
    header says ``"version":3``). Anything it cannot recognise is skipped
    rather than guessed at, and unavailable usage comes back as ``{}`` so the
    caller reports it as unavailable instead of as zero.

    Assistant text is taken from the NARROWEST tier of records that carries
    any: terminal message events first, then ``agent_end``'s batch, then --
    for the session-file/legacy shape, which has no event types at all -- every
    record. Usage is summed over the SAME tier, because the streaming partials
    repeat their message's usage as well as its text.
    """
    header = None
    texts = {t: [] for t in _TEXT_TIERS}
    tier_usage = {t: {f: 0 for f, _ in _USAGE_FIELDS} for t in _TEXT_TIERS}
    tier_found = {t: False for t in _TEXT_TIERS}
    all_usage = {f: 0 for f, _ in _USAGE_FIELDS}
    all_found = False
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # Text that is not a record: keep it, so a mode that degrades to
            # plain text still yields the reply instead of nothing.
            texts["loose"].append(line)
            continue
        if not isinstance(rec, dict):
            continue
        if header is None and rec.get("type") == "session":
            header = rec
            continue
        rtype = rec.get("type")
        tier = rtype if rtype in _TEXT_TIERS else "loose"
        if tier == "agent_end":
            msgs = rec.get("messages")
            got = ([_assistant_text({"message": m}) for m in msgs]
                   if isinstance(msgs, list) else [_assistant_text(rec)])
        else:
            got = [_assistant_text(rec)]
        texts[tier].extend(t for t in got if t)
        if _accumulate_usage(tier_usage[tier], rec):
            tier_found[tier] = True
        if _accumulate_usage(all_usage, rec):
            all_found = True

    for tier in _TEXT_TIERS:
        collapsed = _collapse_stream_texts(texts[tier])
        if not collapsed:
            continue
        usage = tier_usage[tier] if tier_found[tier] else (
            all_usage if all_found else {})
        return header, "\n".join(collapsed).strip(), usage
    return header, "", (all_usage if all_found else {})


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
                 summary_cap: int = 1500):
        super().__init__(work_dir=work_dir, log_path=log_path,
                         summary_cap=summary_cap)
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
    "CONTEXT_LIMIT_TOKENS",
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
    "RAW_STDOUT_CAP",
    "ReplaySession",
    "RoundResult",
    "RoundTimeout",
    "SessionBase",
    "SessionCwdMismatch",
    "configure_native_compaction",
    "model_context_window",
    "parse_decision",
    "parse_json_mode_stdout",
    "parse_round_stdout",
    "seed_agent_dir",
]
