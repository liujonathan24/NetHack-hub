---
name: nethack-experiment-setup
description: "Standing decisions for running NetHack harness experiments (inference via Prime, sandbox off, pinned hub branch)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 35175e07-0ed1-4861-8be8-4956823e570c
  modified: 2026-08-02T01:23:53.302Z
---

Three-repo NetHack experiment stack: `NetHack-engine` (layer 1, engine +
`third_party/NetHack` fork submodule), `NetHack-hub` (layer 2, env + harness +
eval tooling), `NetHack` (standalone fork clone — a decoy; the build reads the
submodule, not this).

Standing decisions as of 2026-08-01:

- **Inference goes through Prime Intellect only** — `api.pinference.ai`, key
  `PI_API_KEY` (CLI arms read `PRIME_API_KEY`). Do not add per-vendor keys
  (OpenAI/Gemini/Anthropic). Gemini models must route via the `prime-team`
  registry entry so `X-Prime-Team-ID` is sent, or calls return
  `insufficient_funds`.
- **bwrap sandbox ON** for the `prime_agent` arms (reversed 2026-08-02, commit
  `bc50ab7`). It needs `--ro-bind /run /run` (resolv.conf symlinks into /run —
  without it DNS dies as a generic "Connection error.") and `~/.prime` bound
  (credentials + kernel venv). Keep `prime_agent.toml` and
  `prime_agent_b80.toml` identical or the b80 comparison drifts.
- **Hub runs on branch `exp/cli-harness-eval`, not `main`.** `main` is missing
  ~4,400 lines of harness repairs and both handoff docs; running it reproduces
  pre-repair numbers.

`prime-agent` (the arm-3 agent scaffold) is **not on npm** — `npm install -g
prime-agent` 404s. It installs from Prime's own script:
`curl -fsSL https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh | sh`,
which lands v0.3.3 at `/usr/bin/prime-agent`. Distinct from `prime`, the Prime
Intellect CLI (`uv tool install prime`). Its harness does not self-install it, so
this is manual on every fresh box.

The handoff refers to a `RUNBOOK.md` that never existed in any commit of any of
the three repos. One was reconstructed at `NetHack-hub/RUNBOOK.md`, alongside
`setup_sandbox.sh` (full rebuild from a bare box). The real docs are
`docs/HARNESS_DEFECTS.md` and `docs/experiments/exp2-cli-harness-results.md`.

See [[prime-intellect-sandbox-is-ephemeral]].
