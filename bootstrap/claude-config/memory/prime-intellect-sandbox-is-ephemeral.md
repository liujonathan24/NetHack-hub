---
name: prime-intellect-sandbox-is-ephemeral
description: Work happens on a Prime Intellect CPU sandbox with no persistent volume — nothing survives deletion
metadata: 
  node_type: memory
  type: project
  originSessionId: 35175e07-0ed1-4861-8be8-4956823e570c
  modified: 2026-08-01T02:29:13.865Z
---

Work runs on a **Prime Intellect CPU sandbox**: one ext4 root disk
(`/dev/vda3`), no persistent volume mounted. `/workspace`, `/data` and
`/ephemeral` exist but are plain directories on that same disk — none of them
are separate mounts, so none confer durability. Default sandbox lifetime is
240 min (`--timeout-minutes`) plus an idle timeout.

**How to apply:** treat every build artifact, venv and output dir as disposable.
Prefer a committed rebuild script over hand-run setup steps, and push branches or
`prime sandbox download` results before the box goes away. Prime's real
persistent storage (Storage tab → Create Disk) is documented for *instances* and
*clusters*, not sandboxes.

See [[nethack-experiment-setup]].
