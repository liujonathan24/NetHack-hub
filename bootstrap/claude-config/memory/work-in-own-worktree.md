---
name: work-in-own-worktree
description: Always create and work in your own git worktree, never the shared checkouts other agents are using
metadata:
  type: feedback
---

Do not work directly in the existing NetHack-hub checkouts (`/root/nld/blog-wt`,
`/root/nld/hub-eval`, `/root/NetHack-hub` itself). Create a dedicated worktree —
e.g. `git worktree add -b claude/<topic> /root/NetHack-hub/.claude/worktrees/<name>
origin/<branch>`, then EnterWorktree with that path.

**Why:** several agents run against the same repo at once; sharing a checkout means
stepping on each other's index, HEAD, and generated files. A branch can also only be
checked out in one worktree, so a private branch is required.

**How to apply:** at the start of any repo work, make your own worktree on a
`claude/…` branch off the upstream branch, set `user.name`/`user.email` there
([[commits-authored-as-jonathan]]), and push to the real branch only when asked.
See [[blog-nethack-e7-handoff]].
