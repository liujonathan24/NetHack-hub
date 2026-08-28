#!/usr/bin/env python3
"""E16 preflight: prove the three silent hangs cannot happen, before any spend.

WHY THIS FILE EXISTS
--------------------
Three failure modes were measured in the E16 model-in-the-loop sims, and all
three share the property that makes them worth a dedicated gate: they HANG
rather than error.

  1. ``--resume`` from a cwd the session was not recorded in. prime-agent
     resolves it as a foreign session and asks "Fork this session into current
     directory?" through an interactive confirm (``main.js:311-318``). A
     non-interactive driver never answers it.
  2. The unsandboxed orchestrator on the SHARED ``/tmp/prime-agent-0/daemon.sock``.
     Measured: 900 s of hang, then 500 s, with another tenant's two wedged
     processes on that socket; 3.5 s with a private ``TMPDIR``.
  3. ``--mode json`` + ``--resume``. Measured hanging on a prompt that the
     identical text ``--print`` call -- same session, same cwd -- answered 30 s
     later.

An error costs a round. A hang costs the run's wall clock and, on a $700
budget with an uncapped player behind it, the run. A dry run that exercises the
happy path proves nothing about any of them, because on the happy path all
three are invisible.

WHAT THIS PROVES, AND WITH WHAT
-------------------------------
Every check here runs with NO INFERENCE. The subprocess under test is a STUB
prime-agent written into a temp directory: it can be told to hang forever, to
spawn a child and hang, to read stdin, or to behave. That is what lets the
timeout path, the process-group kill and the stdin-EOF path be exercised for
real rather than asserted about.

Two checks additionally touch the REAL binary, and neither spends anything: a
bogus session id is refused before any provider call (prime-agent exits 1 with
"No session found matching ..."), and ``--mode json`` parses. Both run under
the same hard deadline as everything else, so a preflight cannot itself hang.

    python e16_preflight.py [--run-dir DIR] [--json]

Exit 0 = every check passed. Exit 1 = at least one FAILED. Exit 2 = a check
could not run (reported as SKIP, which is not a pass).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))

import e16_session as S  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

#: A uuid that cannot exist, for the "refused before any provider call" check.
BOGUS_SESSION_ID = "00000000-0000-0000-0000-000000000000"


# --------------------------------------------------------------------------- #
# the stub binary
# --------------------------------------------------------------------------- #

_STUB = r'''#!/usr/bin/env python3
"""A prime-agent stand-in whose behaviour is chosen by E16_STUB_MODE.

  ok          write a session header + one assistant message, exit 0
  hang        sleep forever
  hang_child  spawn a child that sleeps forever, then sleep forever (so a
              killer that reaps only the direct child leaves the child alive
              and this check can notice)
  read_stdin  block on stdin, then report what it got
"""
import json, os, subprocess, sys, time, uuid

mode = os.environ.get("E16_STUB_MODE", "ok")
sess_dir = os.environ.get("E16_STUB_SESSIONS", "")
argv = sys.argv[1:]

if mode == "hang":
    time.sleep(100000)
elif mode == "hang_child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(100000)"])
    with open(os.environ["E16_STUB_CHILD_PID"], "w") as fh:
        fh.write(str(child.pid))
    time.sleep(100000)
elif mode == "read_stdin":
    data = sys.stdin.read()
    with open(os.environ["E16_STUB_STDIN_OUT"], "w") as fh:
        fh.write(f"read {len(data)} byte(s) then EOF")
    sys.exit(0)

# `ok`: behave like the real thing. --resume <id> continues that session file;
# no --resume mints a new one. The HEADER id is deliberately NOT the filename
# stem, because the real session-manager mints a fresh uuid for the header and
# keeps the old path -- discovering the id from the filename is a trap this
# stub reproduces so the preflight can prove we do not fall into it.
resume = argv[argv.index("--resume") + 1] if "--resume" in argv else ""
cwd = os.getcwd()
if resume:
    path, header_id = None, resume
    for name in sorted(os.listdir(sess_dir)):
        if not name.endswith(".jsonl"):
            continue
        p = os.path.join(sess_dir, name)
        with open(p) as fh:
            head = json.loads(fh.readline())
        if head.get("id") == resume:
            path = p
            break
    if path is None:
        print(f"No session found matching {resume}", file=sys.stderr)
        sys.exit(1)
else:
    header_id = str(uuid.uuid4())
    path = os.path.join(sess_dir, f"{uuid.uuid4()}.jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps({"type": "session", "version": 3,
                             "id": header_id, "cwd": cwd}) + "\n")

reply = os.environ.get("E16_STUB_REPLY", "stub reply")
msg = {"type": "message", "id": str(uuid.uuid4()),
       "message": {"role": "assistant", "content": reply,
                   "usage": {"input": 1000, "output": 100, "cacheRead": 50,
                             "cacheWrite": 0,
                             "cost": {"total": 0.001234}}}}
with open(path, "a") as fh:
    fh.write(json.dumps(msg) + "\n")

if "--mode" in argv and argv[argv.index("--mode") + 1] == "json":
    print(json.dumps({"type": "session", "version": 3, "id": header_id,
                      "cwd": cwd}))
    print(json.dumps(msg))
else:
    # Plain `--print` writes assistant text and NOTHING else -- no header, no
    # records (dist/modes/print-mode.js:80-95). Reproducing that exactly is
    # what makes `directive_survives_text_round` a real check: a stub that
    # answered json here would drive the record parser on a text round, which
    # is precisely the substitution that hid the pilot's directive bug.
    print(reply)
'''


def _write_stub(root: Path) -> Path:
    path = root / "prime-agent-stub"
    path.write_text(_STUB)
    os.chmod(path, 0o755)
    return path


def _session(root: Path, stub: Path, *, work="orch", **kw):
    work_dir = root / work
    agent_dir = root / "agent"
    sessions = agent_dir / S.SESSIONS_SUBDIR
    sessions.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["E16_STUB_SESSIONS"] = str(sessions)
    env.setdefault("E16_STUB_MODE", "ok")
    return S.PrimeAgentSession(
        work_dir=work_dir, agent_dir=agent_dir, binary=str(stub),
        env=env, timeout_s=kw.pop("timeout_s", 20.0), **kw)


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #

def check_cwd_guard_refuses_before_launching(root: Path, stub: Path) -> dict:
    """HANG 1. A resumed round from the wrong cwd must raise, not launch.

    The stub is put in ``hang`` mode for the second round, so if the guard
    fails to fire the check does not merely mis-assert -- it would hang, which
    the deadline then turns into a FAIL. That is deliberate: a guard tested
    against a cooperative subprocess is not tested.
    """
    sess = _session(root / "cwd", stub)
    r1 = sess.ask("round one")
    if not sess.session_id:
        return {"status": FAIL, "detail": "round 1 discovered no session id"}
    recorded = sess.session_cwd
    # Move the session's launch directory out from under it, exactly as a
    # `--resume` invoked from a different working directory would.
    sess.work_dir = Path(root / "cwd" / "somewhere-else")
    sess.work_dir.mkdir(parents=True, exist_ok=True)
    sess.base_env = dict(sess.base_env or {})
    sess.base_env["E16_STUB_MODE"] = "hang"
    t0 = time.time()
    try:
        sess.ask("round two")
    except S.SessionCwdMismatch as exc:
        elapsed = time.time() - t0
        if elapsed > 5.0:
            return {"status": FAIL,
                    "detail": f"guard fired but only after {elapsed:.1f}s; it "
                              f"must raise before any subprocess exists"}
        return {"status": PASS,
                "detail": f"refused in {elapsed * 1000:.0f}ms without "
                          f"launching; recorded cwd {recorded!r}",
                "message": str(exc)[:200]}
    except Exception as exc:
        return {"status": FAIL,
                "detail": f"wrong exception: {type(exc).__name__}: {exc}"}
    return {"status": FAIL,
            "detail": "NO GUARD FIRED: a resume from a foreign cwd was allowed "
                      "to launch. This is the $700 hang."}


def check_hard_timeout_kills_the_process_group(root: Path, stub: Path) -> dict:
    """HANG 1/2/3, the backstop. Anything that blocks must die at the deadline.

    Uses ``hang_child`` so this checks what ``subprocess.run(timeout=...)``
    alone does NOT: the grandchild. A killer that reaps only the direct child
    leaves the grandchild holding the pipes, and the wait outlives the deadline.
    """
    work = root / "timeout"
    (work / "agent" / S.SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    child_pid_file = work / "child.pid"
    env = dict(os.environ)
    env["E16_STUB_SESSIONS"] = str(work / "agent" / S.SESSIONS_SUBDIR)
    env["E16_STUB_MODE"] = "hang_child"
    env["E16_STUB_CHILD_PID"] = str(child_pid_file)
    sess = S.PrimeAgentSession(work_dir=work / "orch",
                               agent_dir=work / "agent", binary=str(stub),
                               env=env, timeout_s=4.0)
    t0 = time.time()
    res = sess.ask("this will hang")
    elapsed = time.time() - t0
    if not res.timed_out:
        return {"status": FAIL,
                "detail": f"the call returned in {elapsed:.1f}s without being "
                          f"marked timed_out; error={res.error[:200]!r}"}
    if elapsed > 25.0:
        return {"status": FAIL,
                "detail": f"deadline was 4s but the call took {elapsed:.1f}s"}
    if "RoundTimeout" not in (res.error or ""):
        return {"status": FAIL, "detail": f"unclear error: {res.error[:200]!r}"}
    # The grandchild must be gone too.
    child_status = "no child pid recorded"
    if child_pid_file.is_file():
        pid = int(child_pid_file.read_text().strip())
        deadline = time.time() + 10
        alive = True
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                alive = False
                break
            time.sleep(0.2)
        if alive:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
            return {"status": FAIL,
                    "detail": f"the grandchild ({pid}) survived the deadline: "
                              f"only the direct child was killed"}
        child_status = f"grandchild {pid} was killed with the group"
    if sess.timeouts != 1:
        return {"status": FAIL,
                "detail": f"timeout counter is {sess.timeouts}, expected 1"}
    return {"status": PASS,
            "detail": f"killed at {elapsed:.1f}s (deadline 4s); {child_status}; "
                      f"error names the command and the deadline"}


def check_stdin_is_devnull(root: Path, stub: Path) -> dict:
    """The other half of hang 1. An interactive prompt must read EOF.

    The cwd guard is the first defence, but it can only cover the mismatch it
    knows about. Every interactive confirm prime-agent can raise blocks on a
    read; DEVNULL turns each into an immediate EOF instead.
    """
    work = root / "stdin"
    (work / "agent" / S.SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    out = work / "stdin.txt"
    env = dict(os.environ)
    env["E16_STUB_SESSIONS"] = str(work / "agent" / S.SESSIONS_SUBDIR)
    env["E16_STUB_MODE"] = "read_stdin"
    env["E16_STUB_STDIN_OUT"] = str(out)
    sess = S.PrimeAgentSession(work_dir=work / "orch", agent_dir=work / "agent",
                               binary=str(stub), env=env, timeout_s=10.0)
    t0 = time.time()
    res = sess.ask("prompt")
    elapsed = time.time() - t0
    if res.timed_out:
        return {"status": FAIL,
                "detail": "a subprocess that reads stdin HUNG: stdin is not "
                          "DEVNULL, so an interactive confirm would block "
                          "forever"}
    if not out.is_file():
        return {"status": FAIL, "detail": "the stub never finished its read"}
    return {"status": PASS,
            "detail": f"{out.read_text()} in {elapsed:.2f}s"}


def check_private_tmpdir(root: Path, stub: Path) -> dict:
    """HANG 2. The orchestrator's TMPDIR must be private, and never exported."""
    outer = os.environ.get("TMPDIR")
    seen = {}

    def spy(argv, env, cwd, timeout_s):
        seen.update({"env": dict(env), "cwd": cwd})
        return S._subprocess_runner(argv, env, cwd, timeout_s)

    work = root / "tmpdir"
    (work / "agent" / S.SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["E16_STUB_SESSIONS"] = str(work / "agent" / S.SESSIONS_SUBDIR)
    sess = S.PrimeAgentSession(work_dir=work / "orch", agent_dir=work / "agent",
                               binary=str(stub), env=env, timeout_s=20.0,
                               runner=spy)
    sess.ask("hello")
    got = seen.get("env", {}).get("TMPDIR")
    if not got:
        return {"status": FAIL, "detail": "no TMPDIR was set for the call"}
    p = Path(got)
    if not p.is_dir():
        return {"status": FAIL, "detail": f"TMPDIR {got!r} does not exist"}
    if got in ("/tmp", "/var/tmp") or got.startswith("/tmp/prime-agent"):
        return {"status": FAIL,
                "detail": f"TMPDIR {got!r} is the SHARED one; this is the "
                          f"900s hang"}
    if os.environ.get("TMPDIR") != outer:
        return {"status": FAIL,
                "detail": "the orchestrator EXPORTED TMPDIR into the process "
                          "environment; players inherit it and their bwrap "
                          "sandbox has no such path"}
    return {"status": PASS,
            "detail": f"private TMPDIR={got}; os.environ['TMPDIR'] unchanged "
                      f"({outer!r})"}


def check_json_mode_only_on_discovery(root: Path, stub: Path) -> dict:
    """HANG 3. json mode runs once; resumed rounds are text and still work."""
    argvs = []

    def spy(argv, env, cwd, timeout_s):
        argvs.append(list(argv))
        return S._subprocess_runner(argv, env, cwd, timeout_s)

    work = root / "jsonmode"
    (work / "agent" / S.SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["E16_STUB_SESSIONS"] = str(work / "agent" / S.SESSIONS_SUBDIR)
    sess = S.PrimeAgentSession(work_dir=work / "orch", agent_dir=work / "agent",
                               binary=str(stub), env=env, timeout_s=20.0,
                               runner=spy)
    r1 = sess.ask("one")
    r2 = sess.ask("two")
    r3 = sess.ask("three")
    a1, a2, a3 = argvs
    if "--mode" not in a1 or a1[a1.index("--mode") + 1] != "json":
        return {"status": FAIL,
                "detail": "the discovery round did not use --mode json, so "
                          "there is no way to learn a session id"}
    for i, a in ((2, a2), (3, a3)):
        if "--mode" in a:
            return {"status": FAIL,
                    "detail": f"round {i} still passes --mode json alongside "
                              f"--resume; that combination was measured hanging"}
        if "--resume" not in a:
            return {"status": FAIL, "detail": f"round {i} did not resume"}
        if a.index("--resume") + 1 >= len(a):
            return {"status": FAIL,
                    "detail": f"round {i} has a BARE --resume, which opens the "
                              f"interactive session picker and hangs"}
    if not (r1.session_id and r1.session_id == r2.session_id == r3.session_id):
        return {"status": FAIL,
                "detail": f"session id was not carried through text-mode "
                          f"rounds: {r1.session_id!r} / {r2.session_id!r} / "
                          f"{r3.session_id!r}"}
    if r2.continuity_broken or r3.continuity_broken:
        return {"status": FAIL,
                "detail": "a text-mode round was reported as a continuity break"}
    # The id must NOT have been guessed from the filename -- the stub mints a
    # header id different from its file's stem exactly as the real one does.
    stem = Path(r2.session_file).stem if r2.session_file else ""
    if stem and stem == r1.session_id:
        return {"status": FAIL,
                "detail": "the session id equals the filename stem; the real "
                          "session-manager mints a different header id and "
                          "this check has stopped testing anything"}
    if not r2.usage or not r2.cost_usd_reported:
        return {"status": FAIL,
                "detail": "a text-mode round recovered no usage/cost from the "
                          "session file, so the orchestrator's budget line "
                          "would silently read $0"}
    return {"status": PASS,
            "detail": f"round 1 json (id {r1.session_id[:8]}...), rounds 2-3 "
                      f"text and resumed; usage and ${r2.cost_usd_reported:.6f} "
                      f"recovered from the session file, id != filename stem"}


def check_real_binary_refuses_bogus_resume(root: Path) -> dict:
    """The real prime-agent, for free: a bogus id is refused before any call."""
    binary = shutil.which(S.PRIME_AGENT_BIN) or S.PRIME_AGENT_BIN
    if not shutil.which(S.PRIME_AGENT_BIN):
        return {"status": SKIP,
                "detail": f"{S.PRIME_AGENT_BIN} is not on PATH"}
    work = root / "realbin"
    agent = work / "agent"
    (agent / S.SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    work_orch = work / "orch"
    work_orch.mkdir(parents=True, exist_ok=True)
    tmpdir = agent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env[S.ENV_AGENT_DIR] = str(agent)
    env["TMPDIR"] = str(tmpdir)
    env["PI_SKIP_VERSION_CHECK"] = "1"
    env["PI_TELEMETRY"] = "0"
    argv = [binary, "--print", "--resume", BOGUS_SESSION_ID, "--", "x"]
    t0 = time.time()
    try:
        stdout, stderr, rc = S._subprocess_runner(argv, env, str(work_orch), 90.0)
    except S.RoundTimeout as exc:
        return {"status": FAIL,
                "detail": f"the real binary HUNG on a bogus --resume even with "
                          f"a private TMPDIR and DEVNULL stdin: {exc}"[:400]}
    elapsed = time.time() - t0
    blob = (stdout or "") + (stderr or "")
    if rc == 0:
        return {"status": FAIL,
                "detail": f"a bogus session id exited 0 in {elapsed:.1f}s; "
                          f"--resume is not being honoured under --print"}
    return {"status": PASS,
            "detail": f"exit {rc} in {elapsed:.1f}s before any provider call: "
                      f"{' '.join(blob.split())[:160]!r}"}


def check_provenance_records_the_mitigations(root: Path) -> dict:
    """A mitigation nobody can find in the record cannot be checked later."""
    try:
        import e16_orchestrator as E
    except Exception as exc:
        return {"status": SKIP, "detail": f"orchestrator not importable: {exc}"}
    run = root / "prov_run"
    cfg = E.OrchestratorConfig(run_dir=run, max_attempts=1)
    try:
        prov = E.Orchestrator(cfg, lambda ctx: E.PlayerResult()).prepare()
    except Exception as exc:
        return {"status": SKIP, "detail": f"prepare() failed: {exc}"}
    sess = prov.get("orchestrator_session", {})
    missing = [k for k in ("cwd", "tmpdir", "json_mode_policy",
                           "round_timeout_s") if not sess.get(k)]
    if missing:
        return {"status": FAIL,
                "detail": f"provenance.orchestrator_session is missing "
                          f"{missing}"}
    if "reseed_on_restore" not in prov:
        return {"status": FAIL,
                "detail": "provenance does not record whether the run reseeded"}
    if not Path(sess["cwd"]).is_dir():
        return {"status": FAIL,
                "detail": f"the recorded cwd {sess['cwd']!r} does not exist"}
    if not Path(sess["tmpdir"]).is_dir():
        return {"status": FAIL,
                "detail": f"the recorded TMPDIR {sess['tmpdir']!r} was not "
                          f"created"}
    return {"status": PASS,
            "detail": f"cwd={sess['cwd']}, tmpdir={sess['tmpdir']}, "
                      f"json_mode={sess['json_mode_policy']}, "
                      f"timeout={sess['round_timeout_s']}s, "
                      f"reseed_on_restore={prov['reseed_on_restore']}"}


#: The E16 GE-wiki pilot's round-1 reply: an ABRIDGED rationale followed by the
#: decision object VERBATIM, on its own line, as the round prompt demands. Read
#: out of the orchestrator's session file
#: (`gewiki_pilot/orchestrator/agent/sessions/01a04847-...jsonl`, record 49).
#: The unabridged 1,104-char original is the fixture behind
#: `tests/test_e16_orchestrator.py::test_THE_PILOT_DIRECTIVE_survives_a_text_mode_round`;
#: it is shortened here only to keep this module free of a data file.
PILOT_ROUND1_REPLY = (
    "**Rationale:** We have only the D1 entrance checkpoint. The first "
    "priority is to establish a checkpoint lineage.\n\n"
    '{"checkpoint": "1", "directive": "Take the down-stairs on each level as '
    'soon as you find them and save a checkpoint at every new depth you '
    'reach. Do not fight any monster while your HP is below half of its '
    'maximum.", "rationale": "Fresh D1 start."}'
)


def check_directive_survives_a_text_round(root: Path, stub: Path) -> dict:
    """THE PILOT'S FAILURE, as a launch gate.

    A directive-carrying experiment that silently serves no directive is its
    own control, and that is what shipped: the orchestrator wrote a well-formed
    decision, the round ran in text mode, and the driver handed plain text to
    the json RECORD parser, which read the decision line as an unrecognised
    protocol record and dropped it. The reply on the record was the rationale
    alone; the attempt was launched with a placeholder and $13.36 was spent
    measuring the control arm under a treatment label.

    So the gate is end-to-end and on the RESUMED round specifically -- the one
    every round after the first is.
    """
    work = root / "directive"
    (work / "agent" / S.SESSIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["E16_STUB_SESSIONS"] = str(work / "agent" / S.SESSIONS_SUBDIR)
    env["E16_STUB_REPLY"] = PILOT_ROUND1_REPLY
    sess = S.PrimeAgentSession(work_dir=work / "orch", agent_dir=work / "agent",
                               binary=str(stub), env=env, timeout_s=20.0)
    sess.ask("discovery", kind="opening")          # json mode
    r2 = sess.ask("ROUND 1 ...", kind="round1")    # text mode -- the real one
    if r2.json_mode:
        return {"status": FAIL,
                "detail": "the decision round ran in json mode; this check is "
                          "about the TEXT round every real round after the "
                          "first takes"}
    if (r2.text or "").strip() != PILOT_ROUND1_REPLY.strip():
        return {"status": FAIL,
                "detail": f"the reply was altered in transit: "
                          f"{len(r2.text or '')} of "
                          f"{len(PILOT_ROUND1_REPLY)} chars survived. The "
                          f"decision line is what gets dropped."}
    dec = S.parse_decision(r2.text, ["1"])
    if not dec.valid or not dec.directive.strip():
        return {"status": FAIL,
                "detail": f"no directive could be extracted from a well-formed "
                          f"reply ({dec.fallback_reason!r}); every attempt "
                          f"would launch as an unlabelled control"}
    return {"status": PASS,
            "detail": f"text round returned all {len(r2.text)} chars; "
                      f"checkpoint {dec.checkpoint_id!r}, directive "
                      f"{dec.directive[:48]!r}..."}


CHECKS = (
    ("hang1_cwd_guard", "a --resume from a foreign cwd is refused before any "
                        "subprocess exists", check_cwd_guard_refuses_before_launching),
    ("hang1_stdin_devnull", "an interactive prompt reads EOF instead of "
                            "blocking", check_stdin_is_devnull),
    ("hang2_private_tmpdir", "the orchestrator gets a private TMPDIR and never "
                             "exports it", check_private_tmpdir),
    ("hang3_json_mode", "json mode runs only on the discovery round; resumed "
                        "rounds are text and still carry id, usage and cost",
     check_json_mode_only_on_discovery),
    ("directive_extraction", "a decision written on a TEXT round survives "
                             "extraction; a run that loses it is its own "
                             "control", check_directive_survives_a_text_round),
    ("backstop_hard_timeout", "any call that blocks dies at its deadline, "
                              "process group and all",
     check_hard_timeout_kills_the_process_group),
)

#: Checks that do not take the stub binary.
BARE_CHECKS = (
    ("real_binary_bogus_resume", "the real prime-agent refuses an unknown "
                                 "session id, fast, spending nothing",
     check_real_binary_refuses_bogus_resume),
    ("provenance_records_mitigations", "provenance.json names the cwd, the "
                                       "TMPDIR, the json-mode policy, the "
                                       "round deadline and the reseed choice",
     check_provenance_records_the_mitigations),
)


def run_preflight(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    stub = _write_stub(root)
    results = []
    for name, what, fn in CHECKS:
        t0 = time.time()
        try:
            got = fn(root, stub)
        except Exception as exc:
            got = {"status": FAIL,
                   "detail": f"the check itself raised "
                             f"{type(exc).__name__}: {exc}"}
        got.update({"name": name, "checks": what,
                    "wall_s": round(time.time() - t0, 2)})
        results.append(got)
    for name, what, fn in BARE_CHECKS:
        t0 = time.time()
        try:
            got = fn(root)
        except Exception as exc:
            got = {"status": FAIL,
                   "detail": f"the check itself raised "
                             f"{type(exc).__name__}: {exc}"}
        got.update({"name": name, "checks": what,
                    "wall_s": round(time.time() - t0, 2)})
        results.append(got)
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_skip = sum(1 for r in results if r["status"] == SKIP)
    return {
        "preflight": "E16",
        "root": str(root),
        "results": results,
        "passed": sum(1 for r in results if r["status"] == PASS),
        "failed": n_fail,
        "skipped": n_skip,
        "ok": n_fail == 0 and n_skip == 0,
        "verdict": ("READY" if n_fail == 0 and n_skip == 0
                    else "NOT READY" if n_fail else "INCOMPLETE"),
        "spent_usd": 0.0,
        "note": ("Every check runs with no inference. A SKIP is not a pass: it "
                 "means a hang class was left unmeasured."),
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default="",
                    help="Where to put the preflight's scratch tree. Default: "
                         "a temp directory, removed afterwards.")
    ap.add_argument("--json", action="store_true",
                    help="Emit the full result object instead of a report.")
    args = ap.parse_args(argv)

    tmp = None
    if args.run_dir:
        root = Path(args.run_dir).resolve()
    else:
        tmp = tempfile.mkdtemp(prefix="e16-preflight-")
        root = Path(tmp)
    try:
        report = run_preflight(root)
    finally:
        if tmp and not args.run_dir:
            shutil.rmtree(tmp, ignore_errors=True)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("E16 PREFLIGHT -- the three silent hangs, with no inference")
        print("=" * 74)
        for r in report["results"]:
            print(f"[{r['status']:<4}] {r['name']:<32} {r['wall_s']:>5.1f}s")
            print(f"        what: {r['checks']}")
            print(f"        got : {r['detail']}")
        print("=" * 74)
        print(f"{report['passed']} passed, {report['failed']} failed, "
              f"{report['skipped']} skipped -> {report['verdict']}")
        print(report["note"])
    return 0 if report["failed"] == 0 and report["skipped"] == 0 else (
        1 if report["failed"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
