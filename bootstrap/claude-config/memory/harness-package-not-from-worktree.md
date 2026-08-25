---
name: harness-package-not-from-worktree
description: NetHack-hub worktrees pick up env code but NOT the prime-agent harness package, which is an editable install from the main checkout
metadata:
  type: project
---

In NetHack-hub, a git worktree gets you the *environment* code (launch_cell.sh
puts `$REPO` on PYTHONPATH) but NOT the Prime Agent harness. `nethack_prime_agent`
is an editable install in `/root/NetHack-hub/.venv-cli-eval` whose `.pth` points
at `/root/NetHack-hub/harnesses/nethack-prime-agent` — the main checkout.

**Why:** the RUNBOOK's "working-tree edits take effect without reinstalling" rule
covers the env only. Harness edits in a worktree are silently ignored.

**How to apply:** when launching a cell from a worktree with harness changes,
export `PYTHONPATH=$REPO/harnesses/nethack-prime-agent` first (PYTHONPATH beats
the .pth-added site-packages entry). Symptom if you forget: the eval CLI rejects
your new `--harness.<field>` flag with "Extra inputs are not permitted", which
reads like a typo in the flag rather than a stale package. Do NOT fix it by
reinstalling editable from the worktree — that changes the shared venv other
agents' runs depend on.

Related: [[work-in-own-worktree]], [[nethack-experiment-setup]].
