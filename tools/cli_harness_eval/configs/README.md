# Arm configs: the runtime determination and the MCP-exposure account

Two configs, one environment package. `control.toml` runs arm 0 through verifiers 0.2.1's
**v0 legacy bridge** (no port in its path); `claude_code.toml` runs arm 1 as a real CLI
coding agent driving the same game over **MCP**. Everything below was established by
execution against the pinned `verifiers==0.2.1` in `.venv-cli-eval`, not by reading docs.

Run either with the v1 CLI (`eval`, not the v0 `vf-eval`):

```bash
cd .../.worktrees/cli-harness-eval
ENG=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness
PYTHONPATH="$ENG:$PWD:$PWD/environments/nethack" \
  .venv-cli-eval/bin/eval @ tools/cli_harness_eval/configs/claude_code.toml
```

`PYTHONPATH` is load-bearing for the CLI arms: the tool server is launched as
`[sys.executable, "-m", "nethack_v1"]` with the parent's environment inherited
(`v1/mcp/launch.py:186-215`), so the engine repo, the repo root and
`environments/nethack` must all be importable in that child.

---

## 1. Runtime: subprocess, no container needed (brief step 1, answer **(a)**)

`grep -rn NEEDS_CONTAINER` over the installed 0.2.1 tree returns the flag in exactly one
place that can gate a run: **`v1/task.py:227`, `NEEDS_CONTAINER: ClassVar[bool] = False`**
on `Task`. `validate_pairing` (`v1/env.py:242-248`) refuses the subprocess runtime only
when `task_cls.NEEDS_CONTAINER` is true. There is **no `Harness.NEEDS_CONTAINER`** in this
release — upstream PR #2102 (merged 2026-07-22), which makes every third-party harness
refuse the bare subprocess runtime, is *not* in 0.2.1.

`NetHackTask` does not set `NEEDS_CONTAINER`, so `claude_code` + subprocess is a legal
pairing here. The only consequence is a warning (`v1/env.py:265-272`, "Harness
'claude_code' is running in the subprocess runtime on the local system").

**Proven by running it**, not just by reading: a full Claude Code rollout completed on the
subprocess runtime (see §4). No container runtime was needed, and no Prime remote sandbox
(which would cost credits) was used.

This machine has neither Docker nor Podman — only `apptainer`/`singularity`, which
`v1/runtimes/` has no adapter for (`subprocess`, `docker`, `modal`, `prime`). So if a
future verifiers bump lands the harness-level `NEEDS_CONTAINER`, **this arm stops working
on this machine** and the only options are a Prime remote sandbox (paid) or pinning 0.2.1.
That is the single largest upgrade risk in this experiment.

## 2. How the toolset is exposed over MCP

* **`TOOL_PREFIX = "nethack"`** (`nethack_v1.NetHackToolset`). `ServerBase.server_name`
  (`v1/mcp/server.py:218-224`) turns it into the MCP server name, which
  `ClaudeCodeHarness` writes as the `mcpServers` key (`claude_code/harness.py:88-97`).
* **URL shape**: `http://127.0.0.1:<os-assigned-port>/mcp`, streamable HTTP,
  `json_response=True`, `stateless_http=True` (`server.py:255-283`). The harness receives
  it as `mcp_urls == {"nethack": "http://127.0.0.1:<port>"}` and writes
  `.vf-claude/mcp.json` = `{"mcpServers": {"nethack": {"type": "http", "url": ...}}}`,
  then passes `--mcp-config .vf-claude/mcp.json --strict-mcp-config` so Claude Code sees
  *only* this server (no ambient user/global MCP config).
* **Advertised names, verified from the trace of a real rollout** (`trace.tools`):

  ```
  mcp__nethack__add_note        mcp__nethack__explore_and_descend  mcp__nethack__quaff
  mcp__nethack__attack          mcp__nethack__kick                 mcp__nethack__read
  mcp__nethack__descend         mcp__nethack__move_to              mcp__nethack__recall
  mcp__nethack__eat             mcp__nethack__pickup               mcp__nethack__search
  mcp__nethack__engrave_elbereth mcp__nethack__pin_objective       mcp__nethack__throw
                                                                   mcp__nethack__wiki_lookup
                                                                   mcp__nethack__wiki_search
  ```

  18 tools. **`mcp__nethack__move` is absent** (the netplay gate) and
  `mcp__nethack__explore_and_descend` is present.

  Note the naming is harness-family specific: verifiers' own `null`/`bash` programs flatten
  the same tools to `nethack_search` (`harnesses/null/program.py:77-92`), while Claude Code,
  a native MCP client, applies `mcp__<server>__<tool>`. Do not hard-code either literal
  form into a system prompt.
* **A withheld name is rejected at the MCP layer**, not at the engine: calling `move`
  against a booted server returns `isError=True, "Unknown tool: move"`. The v0
  `_apply_tool_call` re-check (`nethack.py:754`) is the second layer, and it refuses
  without stepping the engine.
* **State** rides the interception `/state` channel (`server.py:190-216`): pull before each
  call, publish on a `contextvars.ContextVar`, push back if changed. `Trace.state` is
  `exclude=True` (`v1/trace.py:374`) and never reaches `traces.jsonl`, so
  `NetHackTask.finalize` copies the referee's bookkeeping (`skill_calls`, `moves_executed`,
  `max_dlvl_reached`, …) onto `trace.metrics`, where a run aggregator can see it.

## 3. Model routing: Anthropic dialect, and where the billing header comes from

Claude Code speaks the **Anthropic Messages** API. The interception server registers
`/v1/messages` (`v1/dialects/anthropic.py:272`) and the eval client is a **relay**: it
forwards the program's native JSON to `base_url + "/v1/messages"` with `x-api-key` auth
(`v1/clients/eval.py`). `ClaudeCodeHarness.launch` strips the `/v1` off a pinference
`base_url` first (`claude_code/harness.py:60-63`), so the upstream is
`https://api.pinference.ai/api/v1/messages`. Prime Inference serves that route — confirmed
directly with curl for both `z-ai/glm-5.2` and `google/gemini-3-flash-preview`.

**`configs/endpoints.toml` is not consulted on this path.** It is the v0 `vf-eval`
registry. The v1 CLI resolves the endpoint from `EvalClientConfig`, which falls back to the
active Prime CLI config (`v1/clients/config.py:37-59`): the base URL, the API key **and**
`X-Prime-Team-ID` are read from `~/.prime/config.json` automatically for a pinference base
URL. A `--dry-run` of `claude_code.toml` shows the resolved header in the saved config.

So the "GLM 5.2 bills a $0 personal balance" hazard does **not** apply to the v1 route, and
`endpoints.toml` was left unchanged. It *is* real for anything still on `vf-eval`: verified
by curl, `z-ai/glm-5.2` without `X-Prime-Team-ID` returns HTTP 402 `insufficient_funds`,
with the header it returns 200.

## 4. Acceptance: a Claude Code agent actually played

One rollout, `explicit_seeds = [0]`, `max_skill_calls = 20`, `Val-hum-neu-fem`,
`task_spec = "full_nle"`, `z-ai/glm-5.2`. Evidence is in
`.superpowers/sdd/2026-07-25-cli-harness-eval/task-13-report.md` §4; headline: 21
`mcp__nethack__*` calls and **zero** non-MCP tool calls, rendered observations returned as
tool results, `skill_calls` accumulated to the cap over the `/state` channel
(`stop_condition = "call_budget_exhausted"`), `moves_executed = 0`, and the agent descended
a floor (`descent_reward = 10.0`).

## 5. Caveats you should know before spending a 16-seed run

1. **The subprocess runtime gives Claude Code a real host shell.** `--bare` trims its tool
   set to Bash / Edit / Read (no WebFetch, no WebSearch), which is the intended
   capability-match for the control arm's wiki + Journal — but the runtime workdir is a
   plain `/tmp/<trace-id>` on this host and nothing confines the agent to it. Accepted
   deliberately here because there is no container runtime available; re-evaluate if this
   ever runs somewhere with Docker.
2. **`APPENDS_SYSTEM_PROMPT` differs across harness families.** `ClaudeCodeHarness` is
   `True` (`--append-system-prompt`), so the NetHack spec prompt is a real system message.
   Any future arm on a `False` harness (codex, mini-swe-agent) folds it into the user
   message — a cross-arm confound that must be recorded per arm.
3. **`codex` is not an option**: `SUPPORTS_MCP = False  # TODO` (`codex/harness.py:47`), and
   `validate_pairing` rejects the pairing at construction.
4. **Claude Code is invoked one-shot** (`--print --no-session-persistence`) and is not in
   verifiers' ACP-resume test matrix. A rollout is one process; there is no mid-episode
   resume.
5. **The workspace is rebuilt per rollout** (`NetHackTask.setup` -> `build_workspace`),
   which wipes `memory/`. That is deliberate: a CLI arm must not carry notes across seeds
   when the control arm's Journal cannot.
