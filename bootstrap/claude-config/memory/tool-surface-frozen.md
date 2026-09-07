---
name: tool-surface-frozen
description: Published tool/function definitions in the NetHack harness must never change for logging/observability; use result-side channels
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 66c47de3-b72f-4f4b-a231-ab1b3d96599c
  modified: 2026-08-16T06:16:34.031Z
---

Never change the published tool/function schemas the model sees (names, parameters, descriptions) to add logging, observability, or metadata channels. PR #24's first design (a `reasoning` parameter on every tool) was explicitly rejected: "the definitions of the functions should not change."

**Why:** The project's core comparison holds the tool surface fixed across scaffolds ([[nethack-experiment-setup]]). Schema changes perturb model behavior and token spend on exactly the surface being measured, invalidating parity and token-matched cells.

**How to apply:** Observability data must ride server-side channels: assign correlation IDs at `_apply_tool_call`, embed them in the tool RESULT payload (which flows back through every scaffold's transcript verbatim), and recover the model's reasoning by joining `traces.jsonl` to turn records on that ID — never by asking the model to supply extra arguments. When reviewing any harness PR, check the published inputSchema is byte-identical to main.
