---
name: commits-authored-as-jonathan
description: "Git commits must be authored as Jonathan Liu (jl0796@princeton.edu), never as Claude"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 66c47de3-b72f-4f4b-a231-ab1b3d96599c
  modified: 2026-08-16T05:34:42.050Z
---

Commits pushed to Jonathan's repos must carry `user.name="Jonathan Liu"`, `user.email="jl0796@princeton.edu"` as the git author — not Claude/noreply@anthropic.com.

**Why:** A docs commit authored as "Claude" on NetHack-hub main (2026-08-16) had to be rewritten and force-pushed on request: "it has to be a commit by liujonathan24 not claude". The repo attribution is user-facing (advisor, collaborators).

**How to apply:** `git -c user.name="Jonathan Liu" -c user.email="jl0796@princeton.edu" commit ...` for every commit, including ones made by subagents (put it in their briefs). The Claude-Session/Co-Authored-By trailers in the message body are acceptable; the author field is what matters. Note force-pushes are blocked by the permission classifier — if history rewriting is needed, prepare the commit and hand the user the push command to run with `!`.
