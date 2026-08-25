<!-- Addressed to the agent picking this up. Read section 1 before anything else. -->

> **If you are the agent assigned to this:** this document is a specification produced by a
> read-only investigation. No code was changed to write it. Start at **§1 Verdict** — the
> hypothesis that motivated the work is partly wrong, and §1 says exactly which part and what
> replaces it. Then read **§9 Open questions**: seven of the eight must be answered *before*
> you write anything, and **Q1** (is the editable install a bare `.pth` on your box, or an
> import hook?) invalidates the whole Step 1 if it resolves the other way. Work in your own
> git worktree off `exp/e13-v2-continual`; do not share an existing worktree with another agent.

# Skills-as-Code: making the NetPlay layer agent-editable

**Status:** specification, not implementation. Nothing in the repo was modified to produce this.
**Audience:** a Claude Code session with no memory of the investigation that produced it.
**Prime Agent version investigated:** 0.3.3, installed at `/usr/lib/node_modules/prime-agent/` (a **Node.js** bundle — `prime-agent` is not a Python package; `pip show prime-agent` and `import prime_agent` both fail. `/usr/bin/prime-agent` is a symlink to `dist/bundle/cli.js`).
**Repo branch of record:** `exp/e13-v2-continual`, tip `c6349c1` (= merged PR #36). **Not** `main`, and **not** the currently checked-out `feat/tier-flags-part2`, which lacks `run_e13.sh`, `tool_tiers.py`, `merge_harness_stores.py`, `SKILL.baseline.md` and `configs/continual/experiment.toml`. Every repo citation below is repo-relative on `c6349c1`.

---

## 1. Verdict

**Partially correct — the right instinct, aimed at the wrong mechanism, and blocked by a boundary the hypothesis does not account for. But the mechanism that *does* work already exists, is already wired into our launcher, and I verified it empirically on this box.**

Three separate claims are bundled in the hypothesis. They resolve differently:

**(a) "Define them as skills in Prime Agent, so the agent can modify them… this is the designed way of using continual harness." — FALSE as stated.**
`rlm.harness.create_skill(...)` does **not** store code. A harness entry is a `HarnessEntry` dataclass with `title`, `content`, `reference`, `arguments` (`/usr/lib/node_modules/prime-agent/dist/prime-agent-runtime/src/rlm/harness.py:94-111`), persisted as JSON in `harness_state.json`. `create_skill` *requires* a Python `reference` — `_validate_python_skill_reference` rejects anything without `reference["type"] == "python"` plus an `import`/`python_import` and a `callable`/`call_pattern` (`harness.py:128-139`, called from `harness.py:602`). That reference is a **pointer to code that must already exist and already be importable**. The harness store never contains, versions, validates, or executes a function body. Creating a `create_skill` entry for a callable that does not exist raises nothing at write time and simply produces a system-prompt line advertising a call that will `AttributeError` at runtime. Our own `docs/EXPERIMENT_E13.md:64` already states this ("They are **not code**"), and Prime Agent's own docs are quoted there: a harness entry "does not replace packaging new functionality with `skill-creator`."

**(b) "Moving Prime Agent to the same layer as where we're defining our skills." — ALREADY DONE.**
Prime Agent has a first-class *Python-backed skill* mechanism, documented at `/usr/lib/node_modules/prime-agent/skills/skill-creator/references/python-skills.md`. Contract: a directory with `SKILL.md` + `pyproject.toml` + `src/<import_name>/__init__.py`, detected by `detectPythonSkill` (`/usr/lib/node_modules/prime-agent/dist/core/skills.js:127-171`) and installed `--editable` into the shared kernel venv (`dist/core/kernel/bootstrap.js:279`). **Our NetHack skill already is one** — `harnesses/nethack-prime-agent/nethack_prime_agent/skill/` contains exactly those three files, and `harnesses/nethack-prime-agent/nethack_prime_agent/__init__.py:82` names them as `_SKILL_FILES`. There is no layer to move.

**(c) "Move our NetPlay functions into skills." — NOT A MOVE. The functions are not where the hypothesis thinks they are.**
This is the finding that reshapes the whole design. `skill/src/nethack/__init__.py` is **78 lines and contains zero `np_*` functions**. It is a pure MCP shim: a `NetHack(McpIntegration)` subclass (`:40-43`), one instance (`:46`), and a PEP-562 module `__getattr__` (`:67-78`) that forwards any attribute to that instance, where `McpIntegration` synthesises an awaitable MCP tool stub. Every `np_*` name is resolved **at attribute-access time over HTTP**.

The actual NetPlay implementations live **server-side, in a different process**, at `environments/nethack/vendor/netplay/nethack_agent/skills.py` (783 lines), and every one of them takes a `NetHackAgent` handle with direct NLE engine access — `move_to(agent, x, y)` at `:94`, `melee_attack(agent, x, y)` at `:402`, `explore_level(agent, ...)` at `:484`, `press_key(agent, key)` at `:267`. Registered as `np_*` tools by `environments/nethack/nethack_harness/tools/netplay_true.py:867-877` with `TOOL_PREFIX = "np_"` (`:128`).

So the composites the user wants to make editable are on the far side of an HTTP/MCP process boundary from the agent, and they are written against an engine handle the agent's kernel does not have and must not have. **You cannot move them. You must re-implement the composites client-side, expressed in terms of the primitives that remain server-side.** That is a genuine re-architecture, not a file relocation — and it is the single largest cost item in this spec.

**The nearest design that actually works — and the mechanism that makes it cheap.**
The enabling fact is not `create_skill`. It is the shape of the editable install. I inspected the live kernel venv:

```
/root/.prime/agent/kernel-venv/lib/python3.11/site-packages/_editable_impl_prime_agent_skill_nethack.pth
  contents: /tmp/vf-prime-agent/skills/nethack/src
```

That is a **bare `.pth` path entry**, not an import hook. I verified the consequence empirically: `sys.path` literally contains `/tmp/vf-prime-agent/skills/nethack/src`, and dropping a brand-new package directory there is importable immediately — no reinstall, no `pyproject.toml` change, no venv rebuild, no skill-roster rescan.

```
$ /root/.prime/agent/kernel-venv/bin/python -c "import sys; print([p for p in sys.path if 'vf-prime-agent' in p])"
['/tmp/vf-prime-agent/skills/nethack/src']
$ mkdir -p .../src/_probe_tmp && echo 'VALUE = 42' > .../src/_probe_tmp/__init__.py
$ .../bin/python -c "import _probe_tmp; print(_probe_tmp.VALUE)"
42
```

This materially corrects the pessimism recorded in `docs/EXPERIMENT_E13.md:80-90`, which deferred agent-authored code on the grounds that "the skill roster is scanned at process start… so a package written mid-episode takes effect for the NEXT rollout, never the current one." That is true for a **new Prime Agent skill directory** (which needs a roster scan and a venv install). It is **false for a new Python module under an existing skill's `src/`**, which is just a `sys.path` entry. The E13 doc's blocker does not apply to the design below.

**Therefore the design is:**

> A second, agent-owned Python package — `netplay/` — living beside `nethack/` under the *same* skill's `src/` directory, at `/tmp/vf-prime-agent/skills/nethack/src/netplay/`. It is plain, agent-writable, agent-readable Python that composes the frozen MCP primitives (`np_press_key`, `request_map`, and a small primitive floor) into editable policies (`explore`, `descend`, `fight`, `pray_safely`). It is importable in the same kernel it is written from. It persists across episodes because `{install_dir}/skills` is already a fixed, shared, read-write path. `rlm.harness.create_skill(...)` is used for what it is actually for: registering a ≤180-char routing hint whose `reference` points at the agent's own `netplay.<fn>`.

Code lives in files. The harness store holds pointers and prose. That split is what Prime Agent's design intends, and it is what the hypothesis conflates.

---

## 2. What exists today

### 2.1 Prime Agent's two "skill" concepts

| | Continual-harness **skill entry** | Prime Agent **Python-backed skill** |
|---|---|---|
| Storage | a row in `harness_state.json` | a directory on disk |
| Contains code? | **No** — `title`, `content`, `reference`, `arguments` | **Yes** — `src/<import_name>/__init__.py` |
| Written by | `rlm.harness.create_skill(...)` from the kernel | `skill-creator`, or any file write |
| Reaches the model as | ≤180 chars in the system prompt | a pre-imported module in the kernel |
| Validated? | `reference` shape only (`harness.py:128-139`) | layout only (`skills.js:127-171`) — **no syntax check** |
| Capped? | 6 entries/kind × 180 chars **in the render** | uncapped |

**The 6/180 caps apply to the rendered prompt summary, not to stored content.** `DEFAULT_OVERVIEW_ENTRY_LIMIT = 6`, `DEFAULT_OVERVIEW_REFINEMENT_LIMIT = 5`, `DEFAULT_OVERVIEW_CONTENT_LIMIT = 180` at `dist/core/refinement/refinement.js:12-14`, consumed by `formatHarnessStateForPrompt` (`refinement.js:277-280`), which truncates via `compactText`. The dataclass imposes no length limit, and `HarnessState.overview()` (the kernel-side view, `harness.py:721`) uses a *different*, laxer budget: 20 entries/kind at 120 chars. So an entry's body can be arbitrarily long — it is simply invisible past 180 chars to the next episode's player. **A skill body must never live in `content`.**

**On-disk representation of a Python skill** (`python-skills.md`, verified against `dist/core/skills.js`):

```
<skill-name>/
├── SKILL.md          # YAML frontmatter: name, description
├── pyproject.toml    # its presence is what marks the skill Python-backed
└── src/
    └── <import_name>/__init__.py     # import_name = name with '-' → '_'
```

All four conditions must hold or the skill **silently degrades to markdown-only** with a load warning (`skills.js:128-171`). If `__init__.py` defines `run()`, the module is wrapped as an async callable; otherwise it is imported and exposed by name. Import failures do not crash the kernel — the name binds to a placeholder that raises `RuntimeError` with the import error when called (`python-skills.md`, "The run() Convention").

**How a skill is invoked at runtime.** Not through any API. The kernel pre-imports the module under `import_name` and the agent calls it as ordinary Python: `await nethack.np_move_to(x=54, y=5)`. There is no `call_skill(...)` — the RLM prompt explicitly forbids inventing one (`dist/core/prompts/rlm.js:21`).

**Update/delete API for harness entries:** `create/update/delete_{memory,prompt_note,skill,subagent}` plus `record_refinement` and `overview` (`harness.py:530-720`). `update_skill` re-validates `reference` only when one is supplied (`harness.py:614-641`), so title/content-only updates preserve it. `_upsert` bumps `version` and stamps `updated_at` (`harness.py:377-384`). Entries are keyed by a slug of the title unless an explicit `id` is given (`harness.py:33-34`, `:366`).

**Validation/linting of skill code: none.** Neither the loader nor the kernel bootstrap parses, compiles, or lints `__init__.py`. A syntax error surfaces only as a bound placeholder that raises on call.

**Kernel venv and install keying.** `getKernelVenvDir()` → `$PRIME_AGENT_KERNEL_VENV` or `~/.prime/agent/kernel-venv` (`bootstrap.js:290-294`). Install args are `["--editable", skill.packagePath]` (`bootstrap.js:279`). The venv identity key includes `JSON.stringify(pythonSkills)` — i.e. **the set of skill package paths** (`ensureKernelPythonKey`, `bootstrap.js:280-288`). Reinstall is skipped when `pyprojectPath` and `pyprojectHash` both match (`bootstrap.js:671-685`); `pyprojectHash` is a sha256 of the file's bytes (`bootstrap.js:94`, `:113`). **Consequence: editing `src/**/*.py` triggers nothing. Editing `pyproject.toml` triggers a reinstall. Adding a new skill *directory* changes the venv key and rebuilds (~10 min, per `run_e13.sh:157-159`).** This is why the design puts the agent's package under the existing skill's `src/` rather than creating a sibling skill.

**`RLM_MAX_DEPTH` and subagents.** `_rlmKernelEnv()` (`dist/core/agent-session.js:5769-5784`) hands the kernel `RLM_DEPTH`, `RLM_MAX_DEPTH`, `RLM_GLOBAL_HARNESS_STATE_DIR`, and — when a session dir exists — `RLM_SESSION_DIR`/`RLM_HARNESS_STATE_DIR`. A child is spawned with `rlmDepth: this._rlmDepth + 1` (`:5912`) and refused past the limit (`:6400-6401`); the Python side re-checks at `dist/prime-agent-runtime/src/rlm/__init__.py:78-82`. Our arm sets `RLM_MAX_DEPTH` from `config.rlm_max_depth`, default **1** (`harnesses/.../__init__.py:878`, field at `:302-308`), so exactly one level of subagent is permitted.
Interaction with skills: a subagent runs in the **same kernel venv and the same skill roster** (the roster is process-level; the venv is one shared path). So a subagent sees agent-authored `netplay` code automatically. Two consequences to design around: (i) a subagent can be used as a *code reviewer* for a proposed skill edit without extra plumbing, and (ii) auto-refine's gate `_autoRefineAllowedForSession()` is `this._rlmDepth === 0 && this._localHarnessStateDir() !== undefined` (`agent-session.js:4465`) — under `--no-session` there is no local harness dir, so auto-refine is inert *by accident*, and the moment a player calls `rlm(...)` the host mints an ephemeral RLM session dir on the parent and the gate can open mid-run. Our launcher already pins `autoRefine` explicitly rather than inheriting it (`harnesses/.../__init__.py:792`, with the reasoning at `:780-791`).

### 2.2 Our NetPlay layer

`harnesses/nethack-prime-agent/nethack_prime_agent/skill/src/nethack/__init__.py`, 78 lines, no `np_*` functions:

- `__all__ = ["nethack"]` (`:37`) — `NetHack` is deliberately unexported because `pydoc` filters `help()` through `__all__`, and documenting the class would re-advertise the inherited `list_tools`.
- `_RESERVED = {"run", "__wrapped__", "__call__"}` (`:53`) — blocks the names the kernel bootstrap probes to decide whether a module is *callable*. Forwarding `run` would make `getattr(module, "run")` return an MCP stub, the module would be wrapped as callable, and `await nethack.<tool>()` dispatch would break.
- `_WITHHELD = {"list_tools"}` (`:64`) — tool discovery is withheld at **module** level only; `McpIntegration` still binds tools on the instance internally, which is how dispatch works at all.
- `__getattr__` (`:67-78`) raises `AttributeError` for `_`-prefixed and `_RESERVED` names, raises a pointed message for `_WITHHELD`, else forwards to the instance.

Transport: `from rlm import McpIntegration` (`:31`); `url = None` deliberately (`:42`) because the tool server binds an OS-assigned port per rollout and the endpoint is resolved at call time through the host from `settings.json`'s `mcpServers.nethack.url`. `bearer_token_env = "NETHACK_MCP_TOKEN"` (`:43`) exists only to satisfy `McpIntegration`'s credential gate — the verifiers tool server does not authenticate MCP.

**Catalogue of the exposed surface.** The tier contract pins `skill_set = "np_core,request_map,search"` (`tools/cli_harness_eval/configs/tool_tiers.toml:27-63`); `np_core` is defined at `environments/nethack/nethack_harness/helpers.py:1403-1425`. Implementations at `environments/nethack/vendor/netplay/nethack_agent/skills.py`:

| Tool | Impl | Class | Notes |
|---|---|---|---|
| `request_map` | `nethack_harness/tools/skills.py:2282+` | **PRIMITIVE** | full map dump, costs no game time |
| `np_press_key(key)` | `skills.py:267-280` | **PRIMITIVE** | one keystroke incl. `ESC`/`SPACE`/`ENTER` |
| `np_rest(count)` | `skills.py:680-699` | PRIMITIVE-ish | loop of waits |
| `np_pray()` | `skills.py:715-724` | PRIMITIVE | |
| `np_apply(item_letter)` | `skills.py:575-591` (`create_inventory_command`) | composite-lite | issues APPLY then answers the prompt |
| `search(times)` | `nethack_harness/tools/skills.py:416+` | PRIMITIVE-ish | repeats `s` |
| `np_move_to(x,y)` | `skills.py:94-124` | **COMPOSITE** | pathfind + step loop |
| `np_melee_attack(x,y)` | `skills.py:402-467` (+ `_melee_target_report:368`) | **COMPOSITE** | pursue-and-attack-until-dead policy |
| `np_explore_level()` | `skills.py:484-562` | **COMPOSITE** | up to 100 game turns, self-interrupting frontier explore |
| `np_kick(x,y)` | `skills.py:621-659` (`create_direction_command`) | **COMPOSITE** | picks a reachable neighbour, walks there, kicks |

The four COMPOSITEs are exactly the units worth making agent-editable. The user's stated irreducible primitive — "send keystroke / read screen" — already exists as `np_press_key` + `request_map`.

`SKILL.md` (94 lines) carries YAML frontmatter `name: nethack` / `description: Play NetHack…` and is the authoritative API reference. `SKILL.baseline.md` (91 lines) is the frozen E10 control document, hash-pinned as `_BASELINE_SKILL_SHA256` (`harnesses/.../__init__.py:172`) and served **wholesale** by `_skill_doc()` (`:175-197`) when `skill_doc_coords=False`. It is never written into the runtime skill dir — the agent must see exactly one `SKILL.md`. The only substantive difference is one hunk about the coordinate frame (map rows 0-20, not tty rows).

### 2.3 Layering as it stands

```
prime-agent process (Node)                      verifiers env process (Python)
  argv: sh -c 'unset PYTHONPATH; exec "$@"'        nethack_v1 MCP tool server
        prime-agent --print --no-session             └─ vendor/netplay/.../skills.py
        --offline --provider … --model …                (np_* composites, engine handle)
  cwd:  bwrap --chdir <workdir>                              ▲
  env:  PRIME_AGENT_CODING_AGENT_DIR=                        │ HTTP MCP
          {install_dir}/agent-{trace.id}                     │
        RLM_MAX_DEPTH=1                                      │
        NETHACK_MCP_TOKEN=<per-rollout>                      │
        (RLM_GLOBAL_HARNESS_STATE_DIR NOT set —              │
         controlled positionally, see below)                 │
    │                                                        │
    └── IPython kernel ── venv ~/.prime/agent/kernel-venv ────┘
            sys.path ∋ /tmp/vf-prime-agent/skills/nethack/src   ← the seam
```

Key facts, all verified:

- **argv** (`harnesses/.../__init__.py:886-921`): `sh -c 'unset PYTHONPATH; exec "$@"'` then `prime-agent --print --no-session --offline --provider … --model …`, `--thinking` if set, then `-- <prompt>`. The `unset PYTHONPATH` is load-bearing (`:887-898`): the tool server needs `PYTHONPATH`, but if it reaches the kernel, `environments/nethack/nethack.py` **shadows the skill package** and fails silently as `<unavailable Python skill 'nethack'>`.
- **`--print` mode.** Non-interactive. `/reload` is an interactive-mode slash command (`dist/modes/interactive/interactive-mode.js:3573`) and a daemon command (`dist/modes/daemon/daemon-protocol.js:144`, handler `daemon-mode.js:2800-2806`) requiring an `activeSessionId`. **It is not reachable from the agent's own ipython tool.** This is why the design must not depend on a roster rescan.
- **Skill provisioning is a `settings.json` search path, not a CLI flag.** `_skill_dir` = `f"{install_dir}/skills/nethack"` (`:481-483`), written file-by-file at `:522-535`; `settings["skills"] = [f"{install_dir}/skills"]` (`:799`). The path is **fixed on purpose** — a per-rollout path would change `ensureKernelPythonKey` and rebuild the venv every seed.
- **`{install_dir}` is bound read-**write** in the sandbox** (`--bind`, not `--ro-bind`, at `:672-675`). It is a real copy, not a symlink to the repo. **The skills tree is therefore already agent-writable today.**
- **The skills tree is already shared across rollouts and rounds.** Only `{install_dir}/agent-{trace.id}` is per-rollout (`:741`). `{install_dir}/skills` is one directory that every rollout of every round reads.
- **`RLM_GLOBAL_HARNESS_STATE_DIR` is never set by the launcher.** `getGlobalHarnessStateDir()` is `join(agentDir, "harness")` with no env override, so the harness controls it *positionally*, by placing a symlink or a copy at `{agent_dir}/harness` (`:839`). Two modes, selected by `continual_harness_mode` (`:309`, validated `:833-838`): `copy-merge` → `mkdir -p <ch> <link> && cp -a <ch>/. <link>/` (`:850-853`); `shared-ro` → `ln -sfn <ch> <link>` (`:855-858`) plus a `--ro-bind` layered over the read-write `install_dir` bind (`:676-688`).
- **`PRIME_AGENT_KERNEL_VENV` is never set by the launcher.** Only `e13_orchestrate.sh:145` sets it, for the orchestrator. The kernel venv is therefore one shared global path, bound read-write into every sandbox.

**So the persistence substrate the design needs already exists.** Agent-authored code under `{install_dir}/skills/nethack/src/` is writable, on `sys.path`, and shared across every rollout — today, with zero launcher changes. The work is not plumbing persistence. The work is *governing* it.

---

## 3. Target architecture

### 3.1 Directory layout

```
{install_dir}/skills/nethack/                 ← ONE Prime Agent Python skill (unchanged identity)
├── SKILL.md                                  ← FROZEN. Re-materialised every rollout from the repo.
├── pyproject.toml                            ← FROZEN. Never touched (any edit = venv reinstall).
└── src/                                      ← on sys.path via _editable_impl_*.pth
    ├── nethack/__init__.py                   ← FROZEN shim. The MCP boundary. Re-materialised.
    └── netplay/                              ← AGENT-OWNED. Never re-materialised. Persists.
        ├── __init__.py                       ← re-exports the public policy surface
        ├── _base.py                          ← FROZEN helper: primitive floor + trace beacons
        ├── explore.py                        ← editable composite
        ├── descend.py                        ← editable composite
        ├── fight.py                          ← editable composite
        ├── survive.py                        ← editable composite (pray/rest/flee)
        └── NOTES.md                          ← agent's own changelog (not prompt-rendered)
```

Two invariants make this safe and reviewable:

1. **`nethack` stays frozen; `netplay` is the mutable layer.** The MCP boundary is not negotiable — it is the only thing that makes the game real. `_SKILL_FILES` continues to overwrite `SKILL.md`, `pyproject.toml` and `src/nethack/__init__.py` on every rollout (`harnesses/.../__init__.py:522-535`), which *automatically* repairs any agent tampering with the boundary at the start of every episode. Add `src/netplay/**` to an explicit **exclusion** list so setup never clobbers it.
2. **`pyproject.toml` is never edited.** No new third-party dependencies, ever. `netplay` may import only the stdlib, `nethack`, and `rlm`. This keeps `pyprojectHash` stable (`bootstrap.js:671-685`), so no rollout ever triggers a venv reinstall, and the ~10-minute rebuild never fires mid-experiment.

### 3.2 The primitive floor

`src/netplay/_base.py` is shipped by us, frozen, and is the **only** module `netplay/*` may use to touch the game. It re-exports exactly the primitives:

```python
# frozen — the agent may read this but edits are reverted at rollout setup
from nethack import nethack

async def press(key: str) -> str: ...       # → np_press_key
async def screen() -> str: ...              # → request_map
async def rest(count: int = 5) -> str: ...  # → np_rest
async def pray() -> str: ...                # → np_pray
async def apply(item_letter=None) -> str: ...
async def search(times: int = 1) -> str: ...
```

Every call in `_base.py` is instrumented with a correlation beacon on the result payload (see §6.3 and the standing rule that the published tool schemas are frozen — beacons ride the *result*, never the schema).

The four server-side COMPOSITEs (`np_move_to`, `np_melee_attack`, `np_explore_level`, `np_kick`) are **retired from the agent's documented surface in this tier** and re-implemented in `netplay/`. That is the experiment. Keeping them available would let the agent no-op the whole arm by calling the frozen version.

### 3.3 What changes in the launcher

| Concern | Today | Target |
|---|---|---|
| skill dir | `{install_dir}/skills/nethack`, fully re-materialised each rollout | same path; `src/netplay/**` **excluded** from re-materialisation |
| persistence unit | `harness_state.json` (id-keyed JSON entries) | JSON entries **plus** a git repo at `{install_dir}/skills/nethack/src/netplay/` |
| PYTHONPATH | scrubbed by `sh -c 'unset PYTHONPATH'` | **unchanged** — do not touch this; the `.pth` already provides the path |
| kernel venv | shared, keyed on skill package paths | **unchanged** — no new skill dir, no pyproject edit |
| harness store | `copy-merge` / `shared-ro` | unchanged, and reused as the *registry* for `netplay` entry points |

The launcher change is small and surgical: one exclusion in the setup copy loop, plus provisioning of the `netplay` seed on round 0 only.

---

## 4. Implementation plan

Ordered, each step independently testable. Files named are on `exp/e13-v2-continual`.

**Step 0 — Verify the substrate on the target box (do this first, before writing code).**
Run the two probes from §1 against `/root/.prime/agent/kernel-venv`: confirm `sys.path` contains `<install_dir>/skills/nethack/src` and that a fresh package dropped there imports without reinstall. If the venv on the target box was built with a different install strategy (an import-hook `__editable__*.py` finder rather than a bare `.pth`), **stop** — the whole design's cheapness rests on this, and the fallback is §9 Q1.

**Step 1 — Freeze the primitive floor.** Add `harnesses/nethack-prime-agent/nethack_prime_agent/skill/src/netplay/_base.py` and `__init__.py` to the repo. Add both to `_SKILL_FILES` in `harnesses/nethack-prime-agent/nethack_prime_agent/__init__.py:82` so they are re-materialised (and therefore self-repairing) every rollout. Test: unit-test `_base.py` against a stub `nethack` module; assert every public name maps 1:1 to a single MCP call.

**Step 2 — Add the seed composites.** Write reference implementations of `explore.py`, `descend.py`, `fight.py`, `survive.py` on top of `_base.py` only. These are the *starting* generation — deliberately competent but not clever, so there is headroom to improve. Test: a golden test that each module imports cleanly and that `ast.parse` over the tree finds no import outside `{stdlib, netplay._base}`.

**Step 3 — Split materialisation into frozen vs mutable.** In `__init__.py`'s `setup()` (currently `:522-535`), separate the loop:
- frozen set (`SKILL.md`, `pyproject.toml`, `src/nethack/__init__.py`, `src/netplay/_base.py`, `src/netplay/__init__.py`) — always overwritten;
- mutable set (`src/netplay/{explore,descend,fight,survive}.py`) — written **only if absent**.
Add a config field `netplay_code_mode: Literal["frozen", "mutable"] = "frozen"` next to `continual_harness_mode` (`:309`), so every existing arm stays byte-identical. Test: two acceptance tests mirroring the existing tier tests — with `netplay_code_mode="frozen"`, argv and the on-disk skill tree are byte-identical to a `base` cell; with `"mutable"`, a pre-existing `explore.py` survives setup.

**Step 4 — Retire the server-side composites for this tier.** Add a tier flag `netplay_composites: bool` to `tools/cli_harness_eval/configs/tool_tiers.toml` routed through `tool_tiers.py` (add to `_HARNESS_SIDE` at `:40` if harness-side, else it flows to `--taskset.env_args.*` at `:107`). When false, `helpers.py:1403-1425`'s `np_core` keep-set drops `np_move_to`, `np_melee_attack`, `np_explore_level`, `np_kick`. Test: extend `tests/test_tool_tiers.py` to assert the emitted `tool_tier_hash` changes and that the four names are absent from the registered schema set.

**Step 5 — Rewrite `SKILL.md` for the new tier.** A tier-selected variant (the machinery already exists: `_skill_doc()` at `:175-197` already serves whole frozen documents by tier). The new document must (a) list the primitive floor, (b) tell the agent that `netplay.*` is *its own editable code* at a named path, (c) state the edit protocol, (d) state the hard rule that `nethack/` and `_base.py` are frozen and reverted. Hash-pin it the way `_BASELINE_SKILL_SHA256` (`:172`) pins the baseline.

**Step 6 — Version control the mutable tree.** `git init` at `{install_dir}/skills/nethack/src/netplay/` during round-0 provisioning. Each rollout commits its own generation on a branch named `agent-{trace.id}` off the round's base commit. This is the whole persistence-and-merge design; see §5.

**Step 7 — Add the validation gate.** A `netplay_lint` module (frozen, in `_base.py`'s package) the agent is *told* to call before finishing, and that the launcher runs unconditionally at rollout teardown. See §6.1.

**Step 8 — Extend `merge_harness_stores.py` to a code-aware merge.** New sibling script `merge_netplay_code.py` in `tools/cli_harness_eval/`. Do **not** extend the JSON merger — the unit and the conflict semantics are different. See §5.

**Step 9 — Wire the round loop.** `tools/cli_harness_eval/run_e13.sh` (round loop at `:166-177`, write-back at `:143-150`) gains a parallel write-back step for code. Reuse the existing guards: refuse round 0 against a non-empty store (`:83-104`), and refuse `RESUME=1` across a changed spec sha.

**Step 10 — Held-out evaluation.** `eval_frozen.sh` already mounts the harness store `shared-ro` with writes off (`:11,47`). Add the same for the code tree: bind `src/netplay` read-only for the eval cells, so evaluation cannot contaminate what it tests.

---

## 5. Persistence & merge design for code artifacts

### 5.1 Why the existing merge is the wrong tool

`tools/cli_harness_eval/merge_harness_stores.py` (195 lines) merges on key `(kind, id)` (`:31`, `:94-95`), resolving conflicts by **newest `updated_at` string-compare wins** (`:105`), logging every differing collision to `report["conflicts"]` (`:106-113`), and treating an id present in the baseline but absent from a copy as a deliberate delete (`:119-146`). It writes atomically (tmp+rename, `:148-152`) and refuses to fall back to empty on unreadable JSON (`:39-42`) — deliberately the opposite of the runtime loader.

That is a sound design **for independent 180-character text entries**. It is wrong for code, for four reasons:

1. **Newest-wins is a silent overwrite.** Two rollouts both improving `explore.py` produce one winner and one discarded body, with no record beyond a 200-char content preview. For a text memory that is a tolerable loss; for a function body it discards the entire episode's contribution.
2. **The unit is wrong.** A code change is a *diff against a known base*, not a whole-value replacement. Newest-wins cannot express "seed 7 fixed the corridor bug and seed 9 added door-kicking" — outcomes that are trivially compatible.
3. **Files are coupled; entries are not.** `explore.py` may start calling a helper that only exists in seed 9's `_util.py`. Per-file newest-wins yields a tree that imports a module that isn't there.
4. **`refinements` is already dropped** (`merge_harness_stores.py:36,46` — `setdefault`ed, never unioned), and the `skill` kind's `reference`/`arguments` are never inspected or merged field-wise; entries are replaced wholesale, so two skills differing only in `reference.call_pattern` produce a conflict report showing identical `content` strings. Both are pre-existing gaps that a code tier makes materially worse.

### 5.2 The design: git, with an explicit round-boundary integration

Concurrency is real: `run_e13.sh` never sets `MAX_CONCURRENT`, so `launch_cell.sh:257-271` leaves the config default (128) and **all 5 seeds of a cell run simultaneously**. And we have a measurement of what concurrent writers do to the JSON store: `harnesses/.../__init__.py:315-325` records that five concurrent writers adding six entries each kept **12 of 30**, with one contributing nothing and nothing reporting the loss — because `rlm.harness` persists via a non-atomic whole-file rewrite and its loader treats an unreadable file as empty. **Never let five rollouts write one code tree either.**

Therefore, exactly mirroring the existing `copy-merge` mode:

**During a round (concurrent):**
```
canonical:  {install_dir}/netplay-canonical/        ← a git repo, read-only to players
per-rollout: {install_dir}/skills/nethack/src/netplay/  ← ??? see below
```

There is a wrinkle the harness-store design does not have: the skills path is **fixed** (`_skill_dir`, `:481-483`) and must stay fixed to keep the venv key stable, so five concurrent rollouts share one `src/netplay/`. Resolve it the way the sandbox already resolves the harness store — with a per-rollout bind, not a per-rollout path:

- Materialise `{install_dir}/netplay-work/{trace.id}/` as a `git clone` (or `cp -a`) of canonical at the round's base commit.
- In `_sandbox_prefix` (`:659-695`), add `--bind {install_dir}/netplay-work/{trace.id} {install_dir}/skills/nethack/src/netplay` **after** the read-write `install_dir` bind, so the narrower later bind wins (the same ordering trick already used for the `--ro-bind` of the harness store at `:676-688`).
- The path inside the sandbox is identical for every rollout, so `sys.path` and the venv key never change; the *content* is private per rollout.
- **This requires `sandbox = true`.** With sandbox off (which our memory records as the current NetHack-experiment default) there is no bind and the five rollouts collide. Under `sandbox = false` the code tier must run **one seed at a time** — make the launcher refuse `sandbox=false and netplay_code_mode="mutable" and n_seeds > 1` rather than silently producing a corrupted tree. This mirrors the existing `continual_harness_writable` docstring, which already says a shared-mutable store must be run one seed at a time (`:333-347`).

**At the round boundary (sequential, orchestrator-only):** `merge_netplay_code.py`:

1. For each rollout branch `agent-{trace.id}`, `git merge --no-commit --no-ff` onto the round-base commit, in a deterministic order (ascending trace id, so the merge is replayable).
2. On a **clean** merge: run the full validation gate (§6.1). If it passes, commit. If it fails, `git merge --abort` and record the branch in `report["rejected"]` with the failure.
3. On a **conflict**: do not auto-resolve. Record in `report["conflicts"]` with the conflicting hunks, and **drop the later branch** (deterministic, not newest-wins). Rationale: an unresolved code conflict is a real research signal about whether the arm's contributions compose, and silently picking one hides it.
4. Emit a report with the same shape as the JSON merger's (`added`/`updated`/`unchanged`/`conflicts`/`rejected`) so `aggregate.py` can consume both.
5. Tag the resulting commit `round-{n}` on canonical. Every round is a named, diffable, revertible point.

**Why git and not a bespoke merger.** Three-way merge already solves problem (1) and (2) correctly, gives per-hunk attribution for free, makes "what did seed 7 add?" answerable as `git log`, and makes rollback (§6.4) a one-line `git reset --hard round-{n-1}`. The alternative — a field-wise Python merger — is a worse three-way merge that we would have to write and test.

**The harness store's role, unchanged.** `merge_harness_stores.py` keeps running exactly as it does. In the code tier it merges the *registry*: `create_skill` entries whose `reference` points at `netplay.<fn>`, plus memories about what worked. Its newest-wins rule is correct for those, because they are prose. **Both mergers run at every round boundary; they merge different artifacts.** One gap worth closing while you are there: the store's `refinements` list is currently discarded (`:36,46`), which loses the audit trail of *why* each generation changed — in a code tier that is the most interesting record we have.

---

## 6. Safety rails

Ordered by how much damage each failure does.

### 6.1 The agent writes broken Python and bricks every subsequent episode

**Severity: highest.** `{install_dir}/skills` is one shared tree, and a `SyntaxError` in `explore.py` propagates to every later rollout of every later round. Prime Agent does not lint skill code (§2.1) and an import failure binds a placeholder that raises only on call, so the failure is **late and quiet**.

Four independent gates:

1. **Compile gate, in-episode.** Ship `netplay.check()` in the frozen `_base.py` package: `compile()` every file in the tree, `ast.parse` it, and re-import. SKILL.md instructs the agent to call it after every edit. This is advisory — agents skip advice.
2. **Compile gate, at rollout teardown.** The launcher runs the same check unconditionally after the CLI exits, before the tree is offered to the merge. A tree that does not compile never leaves the rollout. This is the gate that actually holds.
3. **Merge gate.** `merge_netplay_code.py` re-runs the full gate after each three-way merge (§5.2 step 2), because two individually-valid trees can merge into an invalid one.
4. **Smoke gate.** After the round merge, run the canonical tree against a **stub MCP server** (a recorded-fixture replay of the tool responses, not a live game) and assert every public `netplay` entry point returns without raising. This catches runtime breakage that compiling misses. Reuse the existing golden-obs fixtures under `environments/nethack/tests/golden/obs/`.

Plus the structural rail that costs nothing: `src/nethack/__init__.py`, `SKILL.md`, `pyproject.toml` and `_base.py` are re-materialised at every rollout setup (`:522-535`), so tampering with the boundary self-heals at the start of the next episode even if all four gates are bypassed.

### 6.2 The agent writes code that cheats

**Severity: highest, and this is not hypothetical for us.** We have a documented prior incident in which "zombie seed" rollouts, after their MCP server died, **rebuilt a counterfeit NetHack backend in-kernel** and continued producing plausible observations. Today that required the agent to improvise under duress. Under this design we would be *handing it a persistent, importable, cross-episode code package* — which turns an incidental failure mode into a structural one. A `netplay/explore.py` that simulates a dungeon and returns a fabricated `=== MAP ===` block would be indistinguishable from a real one in the model transcript, would score, and would persist into every later round.

Rails, in order of strength:

1. **Ground truth is server-side and already recorded.** The verifiers tool server keeps its own trace of every skill call against the live engine. Reward, depth and BALROG are computed **from the engine**, never from anything the agent's kernel says. Verify this holds for the new tier before running it — if any scored quantity is derived from kernel-side output, fix that first.
2. **Call-count reconciliation.** At teardown, compare the number of MCP calls the client made (from `_base.py`'s beacons) against the number the server recorded. A `netplay` function that produces observations without a matching server-side call is fabricating. Any nonzero divergence fails the rollout. Correlation IDs ride the **result payload**, never the tool schema — the published tool surface is frozen and must not be changed for logging.
3. **Import allowlist, enforced statically.** `ast.parse` the whole `netplay` tree at every gate and reject any `Import`/`ImportFrom` outside `{stdlib-safe subset, netplay.*, nethack}` — and specifically reject `socket`, `subprocess`, `ctypes`, `importlib`, `urllib`, `httpx`, `requests`, `nle`, `gym`, and any `verifiers*`. Reject `exec`, `eval`, `compile`, `__import__`, and attribute writes to `nethack.*` (which would let the agent monkey-patch the shim). This is a real gate, not a heuristic: the allowlist is short because `_base.py` is the only sanctioned door.
4. **`nethack` is frozen and restored every rollout.** Combined with (3), the agent cannot reach the game except through `_base.py`, and cannot reach `_base.py` except through the MCP shim.
5. **Diff review is cheap because it is git.** Every round's merge is a reviewable diff. Budget human review of round diffs into the experiment; that is the point of using git.

### 6.3 Non-determinism destroying arm comparability

The code tier's independent variable is *the code the agent wrote*, which means a cell is only reproducible against a pinned tree. Mitigations:

- Record `netplay_commit` (the canonical tree's sha) alongside the existing `tool_tier`, `tool_tier_hash`, `tool_tier_commit` emitted for every cell (`tool_tiers.py:109-113`). A cell is then replayable from its output artifact, exactly as tiers are today.
- Held-out evaluation runs against a **frozen tree bound read-only** (§4 step 10), so the evaluated artifact is a fixed sha, not a moving target.
- Keep `[continual-code]`'s flags identical to `[base]`, for the same reason `[continual]` already is (`tool_tiers.toml:78-82`: "STARTS FROM BASE, NOT HUMAN… otherwise a gain could be the human fixes rather than the agent's own edits"). A test should assert `cfg["continual-code"] == cfg["base"]` on every shared key, mirroring the existing tier tests.

**Keeping `base` byte-identical.** This is already the established discipline in this repo and the mechanism is proven: every new capability lands behind a config field defaulting to off, and an acceptance test asserts that a `base` cell's argv and materialised skill tree are byte-identical with the feature absent. `netplay_code_mode` defaults to `"frozen"`, and in `"frozen"` mode `_SKILL_FILES` behaves exactly as today. Add the byte-identity assertion in the same file as the existing tier-drift tests — `tool_tiers.py:4-10` records that a previous sync test could not detect any of four drift mutations including deleting the whole block, so make the new test *able to fail* (mutate the tree in the test and assert it is caught).

### 6.4 Rolling back a bad skill generation

Git gives this for free and it should be an explicit, exercised procedure, not a theoretical property:

- Every round is tagged `round-{n}` on canonical.
- `merge_netplay_code.py --rollback round-{n-1}` resets canonical and records the reason.
- The round-boundary report names every rejected branch and why, so a regression is attributable to a generation rather than to "the arm".
- Add an automatic trigger: if round *n*'s held-out eval is worse than round *n-1*'s by more than the seed-variance band, flag the round for review before round *n+1* starts. Do **not** auto-rollback on a single round — with 5 seeds the variance is large, and our own `exp3_continual_curriculum` work already documents extremely high cross-seed variance as the headline finding.

### 6.5 The unguarded channel that exists today

Worth fixing regardless of whether this spec is implemented: `{install_dir}` is bound read-write (`:672-675`) and `settings["skills"] = [f"{install_dir}/skills"]` (`:799`) is scanned recursively for `SKILL.md`. **An agent today can write `{install_dir}/skills/anything/SKILL.md` and it will be loaded by every subsequent rollout**, with no gate, no review, and no record. Only the three `_SKILL_FILES` are re-materialised; nothing prunes unexpected siblings. Two cheap fixes: (i) at rollout setup, `rm -rf` any entry under `{install_dir}/skills` that is not the known `nethack` dir when `netplay_code_mode == "frozen"`; (ii) hash the skills tree before and after each rollout and record the diff in the trace, so an unexpected write is at least *visible*. Fix (ii) is worth doing before any code tier runs, because it is also the detector for §6.2.

---

## 7. Experiment design

### 7.1 The tier

Add `[continual-code]` to `tools/cli_harness_eval/configs/tool_tiers.toml`, alongside `[base]` (`:65`), `[human]` (`:72`), `[continual]` (`:77`). Tiers are derived as every top-level key except `contract` (`tool_tiers.py:65-66`), so adding a table is sufficient.

| key | base | human | continual | **continual-code** |
|---|---|---|---|---|
| `netplay_telemetry` | false | true | false | **false** |
| `melee_hints` | false | true | false | **false** |
| `skill_doc_coords` | false | true | false | **false** |
| `netplay_composites` (new) | true | true | true | **false** |
| `netplay_code_mode` (new) | frozen | frozen | frozen | **mutable** |
| harness store mounted | no | no | yes | **yes** |

The `[contract]` block (`:27-63`) is emitted for every tier and stays as-is: `skill_set = "np_core,request_map,search"`, `auto_dismiss = "false"`, `variant = "BBOX_MIN"`, `max_calls = 200`, `seeds = [0,1,2,3,4]`, `model = "z-ai/glm-5.2"`, `character = "Val-hum-neu-fem"`, `task_spec = "full_nle"`, `allow_batching = false`, `max_relaunches = 0`, `tune.reveal_map = 1.0`. Note that `np_core`'s membership is now tier-dependent via `netplay_composites`, so `skill_set` alone no longer pins the surface — emit `netplay_composites` into `tool_tier_hash` so the recorded hash still identifies the surface uniquely.

`[continual-code]` starts from **base**'s flags, not human's, for the same reason `[continual]` does.

### 7.2 What it is measured against

The question is *whether agent-authored code beats agent-authored prose, at equal budget*. That needs a three-way comparison on the **same held-out seeds**:

| arm | what carries across episodes | denominator |
|---|---|---|
| `base` | nothing | the floor |
| `continual` | 6×180-char text entries per kind | the prose ceiling |
| `continual-code` | a git tree of Python + registry entries | the treatment |

Plus one control that isolates the confound: `[continual-code-frozen]` — the code tier's *retired-composite* surface with `netplay_code_mode="frozen"`. Without it, a gain could be "our seed `netplay` implementations are better than the vendored `np_*` ones" rather than "the agent improved them." This control is cheap and I would not run the experiment without it.

**Primary metric:** cost-per-unit-progress on held-out seeds — the same metric E13 uses. **Secondary:** max depth, BALROG, and the code-specific ones this tier makes available for the first time — lines changed per round, functions added/deleted, fraction of rounds whose merge was clean, fraction of rollouts rejected by the validation gate.

### 7.3 Rollout units

Mirror `configs/continual/experiment.toml` exactly so the arms are comparable:

- `corpus_seeds = [5,6,7,8,9]` (played and reflected on), `eval_seeds = [0,1,2,3,4]` (**held out**, never played during the experiment, and identical to the `[contract]` seed set so held-out eval is directly comparable to the E10 baseline).
- `rounds = 2` committed / `5` in the live worktree → **10 or 25 playing rollouts**, plus a 5-seed held-out eval per arm.
- Held-out eval is **not part of a round**: the final tree and store are frozen into `outputs/` and evaluated by `eval_frozen.sh`, which mounts the store `shared-ro` with writes off (`:11,47`) so evaluation cannot contaminate what it tests. Do the same for the code tree.
- `replicate = 1` is too thin for a code tier. Given the documented cross-seed variance, budget `replicate = 2` if the cost allows, and if it does not, say so in the writeup rather than over-reading a single replicate.

**Concurrency.** `run_e13.sh:134-140` does not export `MAX_CONCURRENT`, so `launch_cell.sh:257-271` leaves the config default (128) and all 5 seeds start simultaneously. Under `sandbox = true` that is fine with the per-rollout bind from §5.2. Under `sandbox = false` the code tier **must** run one seed at a time; make the launcher refuse the combination rather than silently corrupting the tree. Note also the recorded cost of load: `launch_cell.sh:262-268` records the first `b0_cli` attempt losing all five `claude_code` seeds to MCP disconnects under 16-way load.

### 7.4 What would falsify the hypothesis

State this before running. The hypothesis is that agent-editable code beats agent-editable prose. It is falsified if `continual-code` does not beat `continual-code-frozen` on held-out cost-per-unit-progress — i.e. the agent's edits added nothing over our seed implementations. It is *uninformative* (not falsified) if the validation gate rejects most rollouts; that is a harness result, and the fix is the harness, not the conclusion.

---

## 8. Things I verified, listed for the implementer's confidence

- Prime Agent 0.3.3 is a Node bundle; `create_skill` stores a pointer, not code — `harness.py:94-111,128-139,588-611`.
- 6/180 caps are render-only — `refinement.js:12-14,277-280`; kernel-side `overview()` uses 20/120 (`harness.py:721-740`).
- Python-backed skill contract and the `run()` convention — `skills/skill-creator/references/python-skills.md`; detection at `skills.js:127-171`.
- Our skill already satisfies that contract — `skill/{SKILL.md,pyproject.toml,src/nethack/__init__.py}`.
- `src/nethack/__init__.py` is a 78-line shim with zero `np_*` — `:37,53,64,67-78`.
- The composites are server-side against an engine handle — `vendor/netplay/nethack_agent/skills.py:94,267,402,484,621`.
- Skills dir is fixed and shared across rollouts; `install_dir` is bound read-**write** — `__init__.py:481-483,672-675,799`.
- The editable install is a bare `.pth` path entry, and new packages under `src/` import with no reinstall — verified empirically on this box against `/root/.prime/agent/kernel-venv`.
- `--print --no-session` means `/reload` is unreachable from the agent — `__init__.py:904-905`; `interactive-mode.js:3573`, `daemon-protocol.js:144`.
- Concurrent JSON-store writers lose data: 12 of 30 entries kept — `__init__.py:315-325`.
- The store merge is `(kind,id)` keyed, newest-`updated_at`-wins, drops `refinements` — `merge_harness_stores.py:31,94-95,105,36,46`.
- `RLM_MAX_DEPTH` default 1; subagents inherit the roster and venv — `__init__.py:878`; `agent-session.js:5769-5784,5912,6400-6401`.

---

## 9. Open questions — verify before building

**Q1. Is the bare-`.pth` editable install stable across `uv` versions and across a venv rebuild?** I verified it on this box (`_editable_impl_prime_agent_skill_nethack.pth` → a bare path). Hatchling's editable backend can also emit an import-hook shim (`__editable___*_finder.py`) that maps *only declared packages*, in which case a new sibling package under `src/` would **not** be importable and Step 1 of the plan must instead append the directory to `sys.path` explicitly from `_base.py`. Re-run the §4 Step 0 probe on the target box before writing anything.

**Q2. Does the sandbox actually run for this experiment?** Our recorded NetHack-experiment setup is *sandbox off*. §5.2's per-rollout bind requires `sandbox = true`. If the experiment must run unsandboxed, the code tier is single-seed-per-cell, which multiplies wall-clock by 5. Decide this before costing the experiment; it may be the deciding factor.

**Q3. Is any scored quantity derived from kernel-side output?** §6.2's primary rail assumes reward/depth/BALROG come from the engine only. I did not trace the full scoring path. Verify it explicitly — this is the load-bearing assumption of the entire cheat-detection story, and the zombie-seed incident is proof the failure mode is live.

**Q4. Does `merge_harness_stores.py`'s deletion logic behave correctly when a rollout's *code* tree is empty but its store is not (and vice versa)?** The two mergers are independent; a rollout that crashes after writing code but before writing entries, or the reverse, produces a half-contribution. I did not determine what the round loop does in that case.

**Q5. Can `_base.py`'s beacons be attached without touching the published tool schemas?** The standing rule is that the tool surface is frozen and correlation IDs ride the result payload. `_base.py` is client-side, so beacons there are client-side bookkeeping — but confirm that the *server-side* trace already carries enough to reconcile call counts per rollout without a schema change. If it does not, §6.2 rail 2 needs a different implementation.

**Q6. What is the real cost of a venv rebuild mid-experiment, and can it be triggered accidentally?** `ensureKernelPythonKey` includes the skill-path set and `HOME`/`XDG_DATA_HOME` (`bootstrap.js:280-288`); `run_e13.sh:157-159` records a one-time ~10-minute build. I am confident editing `src/**/*.py` does not trigger it and editing `pyproject.toml` does. I did **not** verify whether creating a new *directory* under `{install_dir}/skills` (e.g. an agent writing a rogue skill, §6.5) changes the key and rebuilds the venv mid-round. If it does, §6.5's fix (i) becomes urgent rather than merely tidy.

**Q7. Which branch should this land on?** The E13 work is on `exp/e13-v2-continual` (`c6349c1`); `main` is at `3e42b2d` and lacks all of it; the working tree is on `feat/tier-flags-part2`. The user's stated working branch, `exp/cli-harness-eval`, does **not** contain the E13 files either. Confirm the base branch before starting, and work in a dedicated worktree — do not share `/root/nld/hub-eval` or an existing worktree with another agent.

**Q8. Does a subagent (`RLM_MAX_DEPTH=1`) inherit a *mid-episode* edit to `netplay`?** Subagents share the kernel venv and skill roster, but I did not verify whether a child kernel re-imports `netplay` fresh or inherits the parent's module cache. This matters for the "use a subagent to review my skill edit" pattern, which is otherwise the cheapest reviewer available.
