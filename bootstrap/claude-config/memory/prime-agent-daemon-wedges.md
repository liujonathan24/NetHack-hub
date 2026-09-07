---
name: prime-agent-daemon-wedges
description: A wedged prime-agent supervisor daemon silently poisons every later rollout (agent hangs at boot, 0 model calls); fix is `prime-agent shutdown --force`
metadata:
  node_type: memory
  type: project
  originSessionId: 66c47de3-b72f-4f4b-a231-ab1b3d96599c
---

Symptom (seen 2026-08-19): every Prime Agent rollout ends in HarnessError with
turns=0 and **model.duration=0.0** — the agent process launches, the tool server
and interception come up, but the agent never makes a single model call and is
killed at the ~300s watchdog timeout (exit -15, "<no output>"). Reproduces across
retries and is NOT caused by config (record_step_frames, auto_dismiss, etc. were
red herrings — a no-frames control hung identically).

Root cause: a stale/wedged `prime-agent` supervisor daemon holding
`/tmp/prime-agent-0/daemon.sock`. New agents connecting to the dead socket hang.
`prime-agent doctor --json` LIES — it reported the daemon "status: current" while
`prime-agent shutdown --force` had to "force-killed unresponsive background
service". Trust the force-kill, not doctor.

Fix (2026-08-19 FINAL, in order — the built-in tools are the cure, manual
socket surgery makes it worse):
  1. prime-agent doctor --fix     # reaps stale sockets/state properly
  2. prime-agent status           # should say "No background services found"
  3. verify: intercept provider reaches a dummy endpoint (or plain
     `prime-agent --print --offline "OK"` prints OK)
  If doctor can't recover: prime-agent shutdown --force, THEN doctor --fix
  again. Do NOT hand-rm /tmp/prime-agent-0 or kill -9 the daemon alone —
  that leaves supervisor-owner state that blocks a fresh daemon ("Timed out
  waiting for daemon to start") and doctor is what cleans it.
  Hygiene: ls -1dt /tmp/vf-prime-agent/agent-* | tail -n +6 | xargs -r rm -rf
Verify with: prime-agent --print --no-session --offline "Reply with OK" (must print OK in <60s).

Run the shutdown between sweeps and whenever boot-hang HarnessErrors appear. NOTE:
this box is NOT the ephemeral Prime sandbox — it has been up since Aug 1, so
daemons and /tmp accumulate for weeks (contrast [[prime-intellect-sandbox-is-ephemeral]]).
The harness docstring flags the same class of failure (DaemonSocketClosedError,
"a stale supervisor poisoning a later rollout"). See [[nethack-experiment-setup]],
[[prime-agent-rebuilds-dead-game]].


**2026-08-19 update — the daemon degrades under SUSTAINED load, so clear PER-CELL, not per-session.** A fresh daemon serves ONE concurrent 5-seed cell fine (NPCORE_v3), but the NEXT cell launched against that same warm daemon hits accumulated worker/journal corruption and every rollout boot-hangs (seen: E8a = 25 HarnessErrors, 15 watchdog kills, 0 turns). `prime-agent status` still reports the daemon 'current' while it is dead — do not trust it. Fix: run the full clear (kill procs; rm -rf /tmp/prime-agent-0 + forkserver dirs; mv aside /root/.prime/agent/daemon-workers AND session-leases; verify `--print OK`) as a pre-launch step before EACH cell in a multi-cell sweep, not once at the top.