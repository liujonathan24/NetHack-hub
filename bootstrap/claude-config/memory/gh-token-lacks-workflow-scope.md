---
name: gh-token-lacks-workflow-scope
description: "gh CLI token now has workflow scope (granted 2026-08-05); the device flow needs the browser page open before the command runs"
metadata: 
  node_type: memory
  type: project
  originSessionId: db2eec92-b154-4ace-89f0-c4dddd763fa8
  modified: 2026-08-05T16:41:20.171Z
---

The `gh` CLI on this machine is authenticated as `liujonathan24`. As of
2026-08-05 its scopes are `gist, read:org, repo, workflow` — pushes touching
`.github/workflows/` now succeed. Before that date the missing `workflow` scope
caused `! [remote rejected] ... (refusing to allow an OAuth App to create or
update workflow '<file>.yml' without 'workflow' scope)`.

**Why:** GitHub gates workflow-file writes behind a separate OAuth scope, and
the rejection happens at push time — the local commit succeeds, so it only
surfaces after the work is done.

**How to apply:** If that rejection ever reappears (a re-login can drop the
scope), run `gh auth refresh -h github.com -s workflow`. It is a device flow:
it prints a one-time code and waits, so open <https://github.com/login/device>
in the browser *first* — a first attempt here died with `context deadline
exceeded` because the code went unentered. The command blocks past the 120s
foreground timeout, so run it with `run_in_background: true` and poll the output
file for the code rather than letting the harness background it mid-prompt.
Related: [[nethack-experiment-setup]].
