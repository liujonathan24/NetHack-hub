# Arm configs: the runtime determination and the MCP-exposure account

Three configs, one environment package. `control.toml` runs arm 0 through verifiers 0.2.1's
**v0 legacy bridge** (no port in its path); `claude_code.toml` runs arm 1 as a real CLI
coding agent driving the same game over **MCP**; `prime_agent.toml` runs arm 2 as a second CLI
agent over the same MCP server but with a **structurally different tool surface** (§6).
Everything below was established by execution against the pinned `verifiers==0.2.1` in
`.venv-cli-eval`, not by reading docs.

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
* **Advertised names, verified from the trace of a real rollout, pre-Task-18** (`trace.tools`):

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

  **Task 18 Step 2** withholds `add_note`, `recall` and `pin_objective` from this toolset
  specifically (`NetHackToolset.tool_functions`, `nethack_v1.py`) — redundant scaffolding for a CLI
  agent that manages its own reasoning/memory internally; the trace analyses found `recall` and
  `pin_objective` called **zero** times across 1,173 Claude Code calls. **15 tools** are advertised
  now. The shared `skill_set="netplay"` resolution the control arm uses is untouched, so the control
  arm's own tool surface (through the v0 legacy bridge, not this class) still includes all three.

  Note the naming is harness-family specific: verifiers' own `null`/`bash` programs flatten
  the same tools to `nethack_search` (`harnesses/null/program.py:77-92`), while Claude Code,
  a native MCP client, applies `mcp__<server>__<tool>`. Do not hard-code either literal
  form into a system prompt.
* **A withheld name is rejected at the MCP layer**, not at the engine: calling `move`
  against a booted server returns `isError=True, "Unknown tool: move"`. The v0
  `_apply_tool_call` re-check (`nethack.py:779`) is the second layer, and it refuses
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
`task_spec = "full_nle"`, `z-ai/glm-5.2`: 21 `mcp__nethack__*` calls and **zero** non-MCP
tool calls, rendered observations returned as tool results, `skill_calls` accumulated to the
cap over the `/state` channel (`stop_condition = "call_budget_exhausted"`),
`moves_executed = 0`, and the agent descended two floors.

**The run itself is committed** — trace, per-turn NDJSON and resolved config — under
[`../acceptance/`](../acceptance/README.md). Narrative in
`.superpowers/sdd/2026-07-25-cli-harness-eval/task-13-report.md` §4.

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
6. **`/tmp/vf-scripts` is a shared-node landmine, and it already went off here.**
   `Runtime.prepare_uv_script` (`v1/runtimes/base.py:172`) writes its prepared script to a
   **hard-coded, non-per-user** path, `/tmp/vf-scripts/{sha256}.py`. On this cluster node that
   directory belongs to someone else:

   ```
   $ ls -ld /tmp/vf-scripts
   drwxr-xr-x. 2 rf7382 orfe 81 Jul 25 17:35 /tmp/vf-scripts

   HarnessError: harness setup: PermissionError: [Errno 13] Permission denied:
     '/tmp/vf-scripts/a849337f8e4e87e3...py.9190ec7e....tmp'
   ```

   So **every harness that goes through `prepare_uv_script` is dead here** — `null`, `bash`,
   and therefore this taskset's own default `NetHackHarness`. `claude_code` escapes only
   because it installs to `/tmp/vf-claude-code-<version>` instead
   (`claude_code/harness.py:13-14`) — the **same design**, so if another user's job creates
   that directory first, this arm breaks identically. Two consequences:
   * there is no fallback harness available if `claude_code` misbehaves mid-run;
   * on a fresh node, check `ls -ld /tmp/vf-scripts /tmp/vf-claude-code-2.1.214` before
     launching, and move nodes if either is foreign — the paths are not overridable by
     config or environment.

   This is also why `test_cross_route_trace_equivalence.py` cannot make the two routes' agent
   loops shape-comparable: the `null` harness that would have done it cannot start.

   The Prime Agent arm inherits the same hazard: its skill package lives at
   `/tmp/vf-prime-agent` — but unlike the two above, that path **is** overridable
   (`harness.install_dir` in `prime_agent.toml`). Its own daemon sockets go under
   `/tmp/prime-agent-<uid>/`, which is per-user by construction.
7. **The v0 `vf-eval` endpoint registry (`configs/endpoints.toml`) is on neither arm's path**
   (§3). Editing it will have no effect on these runs.
8. **The action budget is not exactly equal across the arms** — the CLI arm gets exactly 150
   executed skills, the control arm at most 150 (a v0 turn can pass without executing a
   skill, and extra parallel tool calls in one turn are dropped). See the `max_turns` note in
   `control.toml`; Task 11 must normalize on the measured counts.

---

## 6. Arm 2 (`prime_agent.toml`): the same server, a different *kind* of tool surface

This is the finding the third arm exists to produce, and it must be stated before any table is
read: **arm 1 and arm 2 do not see the toolset the same way.**

| | arm 1 — `claude_code` | arm 2 — `nethack-prime-agent` |
|---|---|---|
| what the model is offered | 18 agent tools, `mcp__nethack__<skill>` | **one** agent tool, `ipython` |
| how a skill is invoked | a native tool call | Python inside that tool: `await nethack.explore_and_descend()` |
| tool schemas in the prompt | yes, all 18 with JSON Schema | no — the model must run `await nethack.list_tools()` / `help(...)` to discover them |
| who parses the arguments | the MCP client | the model, as Python source |
| observation | the tool result | whatever the model chooses to `print()` |

Prime Agent does this on purpose ("Consistent with Prime Agent's single-tool design, MCP
integrations are **not** exposed as new agent tools" — `prime-agent/docs/mcp-integrations.md`),
and only remote `"http"` MCP servers are wired to its kernel at all, which is exactly what
verifiers serves. So the difference is a property of the scaffold under test. Three
consequences for reading results:

* **A discovery cost is charged to arm 2 and not to arm 1.** Arm 1's agent starts knowing all
  18 tool names and their schemas; arm 2's has to spend calls finding them out. Some of arm
  2's budget therefore buys information that arm 1 gets free.
* **Arm 2 can compose.** Loops, `asyncio.gather`, and post-processing of an observation are
  available to it inside a single tool call. The toolset-side referee still counts every
  executed skill, and `NetHackToolset._with_state` serializes them, so a `gather` cannot
  overrun the budget — but it *can* spend it faster than one call per model turn.
* **`num_turns` is even less comparable here** than §6.3 of the Task 13 report already said
  for arm 1: one Prime Agent turn can execute several skills.

### 6.1 Model routing: a custom `models.json` provider, not an env var

Prime Agent has no `<VENDOR>_BASE_URL` escape hatch. The route to verifiers' interception is
the one verifiers' own bundled `pi` harness uses (`v1/harnesses/pi/harness.py:186-199`): a
custom provider written into the per-rollout `models.json`

```jsonc
{"providers": {"intercept": {
  "baseUrl": "<the endpoint verifiers hands the harness>",   // ends in /v1
  "api": "openai-completions",                                // -> POST <baseUrl>/chat/completions
  "apiKey": "PRIME_AGENT_INTERCEPT_KEY",                      // an env-var NAME; the secret never hits disk
  "models": [{"id": "z-ai/glm-5.2", ...}]}}}
```

selected with `--provider intercept --model z-ai/glm-5.2`. Interception's chat dialect
registers exactly `/v1/chat/completions` (`v1/dialects/chat.py:316`), so the shapes match.
`--provider` + `--model` pins the provider deterministically (`resolveCliModel`: an explicit
`--provider` filters the candidate set before pattern matching), and `defaultProvider` /
`defaultModel` in `settings.json` pin the fallback path too — a built-in provider must never
win, or the arm would silently run a different model.

The whole config directory is redirected per rollout with `PRIME_AGENT_CODING_AGENT_DIR`, so
nothing reads or writes the operator's `~/.prime/agent` settings, models or credentials.

### 6.2 What Prime Agent needs on the host (and does *not* install)

Unlike `claude_code`, this harness **does not install the CLI**. Prime Agent is an npm global
(`npm install -g prime-agent`), not a downloadable release tarball, so the harness verifies it
and fails loudly instead. What must exist on the host:

* `prime-agent` on PATH (or `harness.binary` = an absolute path), plus `node` and `uv` —
  `harness.path_prepend` puts them there for the runtime;
* a writable `~/.prime/agent/kernel-venv`. **The harness writes into it**: Prime Agent installs
  every Python skill `--editable` into that shared venv at kernel setup, so the first run
  materializes `prime-agent-skill-nethack` and pulls the `mcp` SDK into it. That is a real
  side effect on the operator's home directory, and it is why `harness.install_dir` is a
  *fixed* path — Prime Agent keys the venv on the set of skill paths, so a per-rollout path
  would rebuild the venv (a `uv` install) on every seed.
* Note this venv build happens even under `--offline` / `PI_OFFLINE=1`, which only disable
  *startup* network operations (update and package-update checks).

### 6.3 Acceptance: a Prime Agent agent actually played

One rollout, seed 0, `max_skill_calls = 12`, `z-ai/glm-5.2`, subprocess runtime:
`skill_calls = 12` (the cap), `budget_exhausted = 1`, `stop_condition = "call_budget_exhausted"`,
`moves_executed = 0`, `errors: []`, `max_dlvl_reached = 2` — the agent descended a floor. It took
**18 model turns for 12 executed skills** (arm 1 runs ~1 skill per turn), spending the difference
on discovery: reading `SKILL.md` and calling `await nethack.list_tools()`. The executed sequence
was `explore_and_descend` ×7, `eat`, `pray`, `move_to`, `search`, `attack`, plus a 13th call
(`descend`) refused by the referee.

**The run is committed** under [`../acceptance/`](../acceptance/README.md), whose "What it shows"
section carries the per-call table and two scoring findings Task 11 must handle: the character
**died at call 6** and was still scored `died = 0` (the last seven calls drained into a tombstone
screen), and **`attack` executes the low-level `move` primitive**, so the gate's honest phrasing
is "no unrestricted `move` tool was published" rather than "the low-level `move` never executes".
Note also that this agent never attempted `move`, so arm 2's `move` gate is asserted **by
construction** here, not measured.

Three operational facts the run established, all of which bite a 16-seed launch:

1. **`trace_dir` must be ABSOLUTE.** A relative value is resolved in the *tool server's* runtime
   workdir (`/tmp/vf-<id>`), which is deleted at teardown — the per-turn NDJSON vanishes without
   an error. Measured: the same rollout produced no file with a relative `trace_dir` and 12 lines
   with an absolute one. **All three arm configs currently carry relative values.** Task 11 should
   either absolutize them or pass `--taskset.trace-dir "$PWD/..."` on the command line.
2. **PYTHONPATH must not reach the agent.** `environments/nethack/nethack.py` shadows the skill
   package (the kernel import name is the `mcpServers` key, fixed to `nethack` by `TOOL_PREFIX`).
   The harness now launches through `sh -c 'unset PYTHONPATH; exec "$@"'`. Before that fix the
   agent could not reach the game at all and — with a real host shell in the kernel — spent 131
   model turns reading this repository's own source instead, executing 2 skills. That is caveat
   §5.1 made concrete: **the subprocess runtime lets a CLI agent read the code that scores it.**
3. **Do not delete the per-rollout config dir.** Prime Agent's supervisor is shared
   (`<tmpdir()>/prime-agent-<uid>/daemon.sock`) and keeps live state under each rollout's config
   directory; removing it kills the supervisor and the *next* rollout dies with
   `DaemonSocketClosedError`. The harness therefore leaves `install_dir/agent-<trace-id>/` behind
   (small, no secrets). Clear it between runs, and check `prime-agent doctor` if a run starts
   failing at launch.

---

## 7. Known unenforced constraint: Prime Agent's `ipython` has no denylist (Task 18 Step 4)

Unlike `claude_code.toml` (§2, §5), **there is no `disabled_tools` for `prime_agent.toml` that
does anything.** Prime Agent's only agent tool is `ipython` — a full Python interpreter running in
the harness's own subprocess runtime, with no tool-level allowlist/denylist mechanism at all (§6:
"Consistent with Prime Agent's single-tool design, MCP integrations are not exposed as new agent
tools" — that quote is about *skills*, not about constraining what the interpreter itself can
import or call). In the run1 acceptance rollout the agent reached `glob('/scratch/**',
recursive=True)` from inside `ipython` and printed the leaked MCP bearer token
(`NETHACK_MCP_TOKEN`) into its own trace — a live demonstration that nothing in this arm's config
stops a Python `import os; os.environ` or an arbitrary filesystem walk. `PrimeAgentHarnessConfig`
(`harnesses/nethack-prime-agent/nethack_prime_agent/__init__.py`) does the honest thing when asked
to pretend otherwise: passing `disabled_tools` in the TOML makes `launch()` raise a loud
`ValueError` instead of silently ignoring the setting and running an arm that differs from what the
config says (`tests/test_prime_agent_harness.py::test_disabled_tools_is_refused_rather_than_silently_ignored`).
**There is no config on either side of this — verifiers' `HarnessConfig.disabled_tools` or Prime
Agent's own settings — that closes this off. Do not read the absence of a `disabled_tools` line in
`prime_agent.toml` as "nothing to clamp"; it means clamping is not offered.**

The real fix is sandboxing the *runtime*, not the tool surface. **This is now done — see §7.1.**
The paragraph that used to sit here observed that the pinned verifiers 0.2.1 has no apptainer
runtime adapter (true: `v1/runtimes/RuntimeConfig` is a closed discriminated union over
`subprocess | docker | modal | prime`, with no plugin hook) and concluded the arm therefore had to
run unconfined. That conclusion was wrong — the container does not have to be introduced at the
*runtime* layer at all.

### 7.1 The sandbox: `tools/cli_harness_eval/sandbox/prime-agent-apptainer` (Task 19 Part C)

`PrimeAgentHarnessConfig.binary` is injected into argv and only ever `exec`'d, so pointing it at a
wrapper that `exec`s `apptainer exec … prime-agent "$@"` containerizes the arm with **no patch to
verifiers and no new runtime adapter**. That is what the wrapper is; its header carries the full
rationale and the exact flag set.

```toml
[harness]
binary = "<repo>/tools/cli_harness_eval/sandbox/prime-agent-apptainer"
```

What the mount namespace looks like from inside:

| path | state | why |
|---|---|---|
| `/scratch`, `/projects`, `/tigress` | **absent** | `--no-mount bind-paths` drops the site block at `/etc/apptainer/apptainer.conf:440-446` |
| `$HOME` | **absent** | `--home /tmp/vf-prime-agent/sandbox-home:…` replaces the home bind *and* sets `HOME` |
| the launching cwd | **absent** | `--no-mount cwd`; without it, launching from a checkout on `/scratch` silently re-binds `/scratch` (measured) |
| `/tmp` | present | not optional — the rollout workspace `/tmp/vf-<id>`, `install_dir`, and Prime Agent's per-user daemon socket `/tmp/prime-agent-<uid>` all live there |
| `/opt/node` | present, read-only | the node + `prime-agent@0.3.3` install, bound from wherever it lives on the host. The one path re-exposed out of `/scratch`, at a different mount point |
| host loopback | reachable | no `--net`, so the container shares the host network namespace and both the MCP server and the interception endpoint stay on `127.0.0.1` with no port mapping |

**Demonstrated, not asserted.** The agent was handed exactly the run-1 offending operations as its
own `ipython` tool call; this is what came back in its tool result, captured off the wire from the
following request body:

```
GLOB_SCRATCH_STAR: []
GLOB_SCRATCH_RECURSIVE: []
GLOB_HOME: []
GLOB_PROJECTS: []
LISTDIR_SCRATCH: FileNotFoundError [Errno 2] No such file or directory: '/scratch'
OPEN_SCRATCH: FileNotFoundError [Errno 2] No such file or directory: '/scratch/gpfs/…/README.md'
```

The identical probe against the unsandboxed binary returns the real directory listings, so the
container — not the probe — is what closes it. A full `--print` session was also driven end to end
through the wrapper (1175 model round-trips, IPython kernel executing inside the container, model
traffic reaching a host-bound `127.0.0.1` listener), so this is not a `--version` smoke test.

**What it is not.** Apptainer runs as the invoking user, and this is a mount namespace, not a
security boundary against a determined adversary: `/tmp` is shared with the host, `/opt/node` is
readable, and the network is the host's. It is a hard stop on the specific failure this experiment
hit — an agent walking a shared multi-user GPFS filesystem — and nothing more.

**Environment knobs**, all overridable, defaults are this cluster's: `VF_SANDBOX_IMAGE`
(`docker://python:3.12-slim`), `VF_SANDBOX_NODE_DIR`, `VF_SANDBOX_UV`, `VF_SANDBOX_HOME`,
`APPTAINER_CACHEDIR` (defaults to `/tmp/apptainer-cache-<uid>`, keeping SIF blobs off the home
quota).

### 7.2 The wired-in default: `bwrap`, via `harness.sandbox` (no wrapper binary needed)

§7.1's apptainer wrapper requires pointing `harness.binary` at
`tools/cli_harness_eval/sandbox/prime-agent-apptainer` by hand. There is a second, simpler
mechanism now built directly into `PrimeAgentHarness.launch`: `harness.sandbox = true` (set in
`prime_agent.toml`) prepends a `bwrap` (bubblewrap) argv prefix to the CLI invocation itself —
`PrimeAgentHarness._sandbox_prefix` in
`harnesses/nethack-prime-agent/nethack_prime_agent/__init__.py`. No wrapper script, no image, no
`apptainer` dependency; `bwrap` (0.6.3 here, `/usr/bin/bwrap`) is a single static-ish binary that
just builds a mount namespace and `exec`s.

The base recipe — verified working on this machine before any extra binds were added:

```bash
bwrap --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
      --ro-bind /bin /bin --ro-bind /etc /etc \
      --proc /proc --dev /dev --tmpfs /tmp \
      --bind <WORKSPACE> /work --chdir /work \
      --unshare-pid --die-with-parent \
      <COMMAND...>
```

confirmed from inside: `cwd=/work`, workspace files readable/writable, `os.path.exists('/scratch')`
is `False`, opening a repo file raises `FileNotFoundError`, and (no `--unshare-net`) a loopback HTTP
request still succeeds — required, since the agent reaches both the MCP tool server and the
interception model endpoint on `127.0.0.1`.

**What `_sandbox_prefix` adds on top, and why each one is unavoidable:**

* **`path_prepend`'s entries, one level above `bin/`, read-only.** `prime-agent` and `uv` have to
  resolve on PATH inside the sandbox too, and on this cluster both live under `/scratch` (an nvm
  install for the former, `~/.local/bin` for the latter isn't under `/scratch` but is bound the
  same way for uniformity). Binding only the `bin/` directory runs `prime-agent --version` but
  breaks the CLI at the first real command — resolving `harness.binary` from
  `.../v24.13.1/bin/prime-agent` is a symlink to
  `.../v24.13.1/lib/node_modules/prime-agent/dist/bundle/cli.js`, so the whole version root
  (`bin/` + `lib/` + …) has to come back, not just the symlink's directory. **Measured
  consequence:** because that root sits under `/scratch`, re-exposing it forces `bwrap` to
  materialize the intermediate path as empty stub directories all the way from `/scratch` down to
  the bound leaf — `os.path.exists('/scratch')` is `True` under the harness's actual prefix, not
  `False` as in the base recipe above. Nothing else is reachable through that stub: no sibling of
  the bound leaf, no other user's data, no repo file. `tests/test_sandbox_isolation.py` proves both
  claims separately — the unmodified base recipe (fully absent `/scratch`) and the harness's
  extended prefix (present-but-empty-except-the-one-leaf `/scratch`).
* **`install_dir` (default `/tmp/vf-prime-agent`), read-write.** Carries the skill package and
  every rollout's `models.json`/`settings.json`/`auth.json` (§6.1); Prime Agent has to read and
  write it. Bound after `--tmpfs /tmp` in the argv (order matters — `bwrap` applies binds in
  sequence, and a bind nested under an earlier `--tmpfs` needs to come later to attach on top of
  it).
* **The rollout workspace**, i.e. the `subprocess` runtime's own workdir (`runtime.workdir`,
  `/tmp/vf-<runtime-name>`) — the counterpart of the base recipe's `<WORKSPACE>`/`/work`, except
  bound at its real path and `--chdir`'d there rather than remapped to `/work`, since nothing else
  in the launch argv needs to know about a `/work` alias.

**Two things this does NOT cover, left as known gaps, not fixed here:**

1. **`$HOME` is never bound.** Prime Agent's IPython kernel-venv bootstrap writes under
   `~/.prime/agent/kernel-venv` (§6.2) — a real path on this operator's *actual* home
   (`/home/<user>`, not under `/scratch`), but the sandbox namespace has no `/home` at all. A
   sandboxed rollout cannot find or reuse that venv.
2. **`/tmp` is a fresh, per-rollout `tmpfs`, not the host's.** §6.3's "the daemon socket is left
   shared on purpose" no longer holds under `sandbox = true`: Prime Agent's per-user daemon socket
   (`<tmpdir()>/prime-agent-<uid>/…`) is resolved *inside* the sandbox's own empty `/tmp`, so each
   sandboxed rollout gets an isolated supervisor rather than attaching to a shared one.

Both follow directly from taking the base recipe as given rather than redesigning it (the task this
was built against was explicit: *use the proven recipe, don't redesign it*). Together they mean a
sandboxed Prime Agent rollout has not been driven end-to-end against a real model in this pass —
only the isolation properties themselves (workspace read/write, `/scratch` unreachable content,
repo file unreadable, loopback network) were proven, via `tests/test_sandbox_isolation.py` and the
ad hoc `prime-agent --version` smoke below. Before spending a paid sandboxed run, re-verify the
kernel-venv path survives (it may simply rebuild every rollout, which is slow but not unsafe) and
budget for a fresh daemon per rollout.

**Ad hoc smoke, not part of the committed test suite (machine-specific paths):** with the nvm node
root and `~/.local` re-bound as above, `prime-agent --version` inside the sandbox printed `0.3.3`
(matching the unsandboxed CLI) and `ls /scratch` inside returned only the single synthetic path
segment needed to reach the bound leaf — no other file anywhere under `/scratch` was listable or
openable.

### 7.3 The Claude Code arm: no sandbox toggle, and why

Arm 1 does **not** get either treatment above. Both §7.1's wrapper-binary approach and §7.2's
in-harness `bwrap` prefix rely on the same hook: the harness controls what ends up in argv[0].
Verifiers' bundled `ClaudeCodeHarness` hard-codes its binary path (`CLAUDE_BIN`,
`verifiers/v1/harnesses/claude_code/harness.py`) and exposes no `binary` field and no other
command-prefix hook — there is no way to reach it without patching the installed `verifiers`
package, which must stay stock (`claude_code.toml` records this decision at `[harness]` directly).
The route to close this gap, if it's ever needed, is the one this repo already uses for arm 2: a
small external harness plugin subclassing `ClaudeCodeHarness` and overriding `launch` to prepend
either wrapper to argv. Until that exists, arm 1's only clamp remains `disabled_tools = ["Bash"]`
(Task 18) — a tool-surface denylist, not filesystem isolation, and weaker than arm 2's: Read, Edit
and the MCP client itself still run against the real, unconfined host filesystem. It is at least a
*denylist*, which is more than Prime Agent's bare `ipython` offers on its own (§7 above).

---

## 8. Implementation-specific: the stray `^M`, and how we stop it (Task 19 Part B)

Run 1 carried `Unknown command '^M'.` on NetHack's top line for 31 turns of one Prime Agent seed
and 17 of another. It is worth knowing about because **it is ours, not NetHack's**, and because one
agent built a false theory on top of it and wrote that theory into its own notes: *"When move_to
fails to advance turn, call engrave_elbereth to unstick the game (clears ^M from input buffer)"* —
then burned an 18-turn streak acting on it.

**Cause.** `engrave_elbereth` emitted `E - E l b e r e t h \r \r`: a first CR to submit the
engraving text and a second, unconditional one to "close any prompt". When no prompt was open, that
second CR reached NetHack's command dispatcher, which has no binding for byte 13, so `rhack()`
takes its bad-command branch and `visctrl()` renders the byte as `^M`. Every one of the 48
occurrences across run 1 was originated by that exact twelve-byte action list. It *looked* like a
`move_to` bug only because NetHack's top line is not cleared until something repaints it, so the
one stray CR sat in the observation of every following turn.

**Two fixes, both shipped.**

1. *Source.* `engrave_elbereth` no longer emits the second CR. Closing a prompt was never its job:
   `env_response` already runs a guarded auto-dismiss loop after every skill that presses CR only
   while a `--More--` is actually up.
2. *Funnel.* Every skill's keystrokes pass through one loop in `env_response` (and one in
   `TypedNetHackInterface.step`). Both now drop a CR that would land in command context, making a
   stray CR a no-op that can never again surface to the agent. The discriminator is the engine's own
   `misc` observation — `(in_yn_function, in_getlin, xwaitingforspace)` — and a CR is stray iff all
   three are clear. Measured against a live engine: idle `(0,0,0)`, item prompt `(1,0,0)`,
   `--More--` `(0,1,1)`, getlin `(0,1,0)`, menu `(0,0,1)`. It fails open: an unreadable observation
   means "a prompt might be open", and the CR is sent unchanged, so the CRs that `descend`,
   `ascend` and `pray` legitimately need are untouched.

`state["cr_swallowed_total"]` counts how often the guard fired; a non-zero value means some skill is
still emitting keystrokes it does not need. Tests:
`environments/nethack/tests/test_stray_carriage_return.py`.

---

## 9. Implementation-specific: `harness.context_window` (Task 19 Part A)

**Do not remove `context_window` from `prime_agent.toml`.** A custom provider's model definition
that omits `contextWindow` is normalized to `128e3` by Prime Agent, and its session host then ends
the agent loop the first time context exceeds `contextWindow - reserveTokens` = 111,616 tokens
(`_shouldStopForThresholdCompaction` → `shouldStopAfterTurn`). Under `--print` nothing resumes it:
the CLI exits 0 having printed nothing, and verifiers can only record a clean exit as
`agent_completed`. That is what ended three of five run-1 rollouts mid-action, on a tool call that
never received a response, at 107-130 of 400 skill calls and 34-43 minutes of a 7200 s budget.

It is set to `1048576`, the value Prime Agent's own bundled catalog carries for `z-ai/glm-5.2`.
**Change it whenever you change `model`.** Leaving it at `0` keeps the old wire format and logs a
loud warning rather than guessing a window on your behalf.

The CLI's stdout/stderr are now persisted to `install_dir/agent-<trace id>/program.{stdout,stderr}.txt`
(`harness.capture_output`, on by default). Verifiers only surfaces a harness program's output on a
non-zero exit, so an exit-0 truncation like this one otherwise leaves nothing at all to read.
