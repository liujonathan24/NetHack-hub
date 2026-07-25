# CLI-agent harness comparison — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a general-purpose coding-agent scaffold (Codex, Prime Agent) plays NetHack better than the purpose-built harness, holding game, action surface, seeds, and model fixed.

**Architecture:** One verifiers-v1 taskset and one toolset, consumed by three harnesses. The toolset gains a `self_dispatch` mode so MCP-driven CLI agents get self-contained tool calls (execute + return observation), while the control arm keeps today's harness-driven `env_response` loop byte-identical. CLI arms run in a sandbox seeded with the filesystem equivalents of the harness's prompt/tool affordances.

**Tech Stack:** Python 3.12, verifiers v1 (`vf.Taskset` / `vf.Toolset` / `vf.Harness`), MCP over streamable HTTP, Prime Inference (GLM 5.2), pytest.

**Spec:** `docs/superpowers/specs/2026-07-25-cli-harness-eval-design.md`

## Preconditions

**Do not start until the experiments migration lands in hub `main`.** Tasks 1–3 restore source
fixes that migration may also touch; Task 11 depends on files it brings over.

Verify before Task 1, and rebase `exp/cli-harness-eval` onto hub `main` first:

```bash
git -C /scratch/gpfs/ZHUANGL/jl0796/NetHack-hub fetch origin
git log --oneline -3 origin/main
ls tools/encoding_eval/          # expect: _verify_gate.py, aggregate_run.py, launch_cell.sh, _smoke.sh
ls docs/experiments/             # expect: exp1-results.md, exp1-trace-assumptions.md
```

If those files are present, re-check whether Tasks 1–3 are still needed:

```bash
grep -c allowed_skill_names environments/nethack/nethack.py                    # 0 => Task 1 needed
grep -c _standard_tier environments/nethack/nethack.py                         # 0 => Task 2 needed
ls environments/nethack/nethack_harness/prompt/balrog_achievements.json        # absent => Task 3 needed
```

### RESOLVED 2026-07-25 — Tasks 1–3 are NO-OPS

PR #16 ("Experiment 1 + real BALROG metric + 1b/1c/1d infrastructure") merged to `main` as
`1fbbc3a` and brought all three source fixes plus the eval tooling. Verified on `1fbbc3a`:

| check | result | task |
|---|---|---|
| `grep -c allowed_skill_names nethack.py` | **7** | Task 1 — skip |
| `grep -c _standard_tier nethack.py` | **2** | Task 2 — skip |
| `balrog_achievements.json` | **present**; `balrog.py` 53 → 102 lines, exports `balrog_progress` | Task 3 — skip |
| `tests/test_balrog_progress.py` | asserts the 12.56% DL10/XL6 anchor | Task 3 — skip |
| `tools/encoding_eval/` | `_verify_gate.py`, `aggregate_run.py`, `launch_cell.sh`, `_smoke.sh` present | Task 11 unblocked |

**Still required:** Task 11, Step 1. `configs/endpoints.toml` line 41 keeps `z-ai/glm-5.2` under
`pinference-glm` (no team header); the funded `prime-team` block at line 50 lists Gemini only.
GLM will bill the $0 personal balance and **hang**.

**Execution starts at Task 4.**

## Global Constraints

Every task inherits these. Values copied verbatim from the spec.

- Model: `z-ai/glm-5.2`, identical for every arm, via the verifiers v1 intercepted endpoint.
- Seeds: `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`, pinned, identical per arm.
- Character: `Val-hum-neu-fem`. Task spec: `full_nle`, uncapped.
- Action surface: `skill_set="netplay"` (15 skills), same registry and gate for every arm.
- Budget: **150 skill calls**, counted toolset-side, not `max_turns`.
- Early stop counts as terminal; report `turns_used`.
- `memory/` is wiped per rollout. No `map/` is ever seeded.
- MCP transport is **streamable HTTP** (Prime Agent supports no stdio). Engine runs outside the agent sandbox.
- Repo: NetHack-hub, branch `exp/cli-harness-eval`. Author commits as Jonathan Liu; no AI attribution trailers.
- `nethack_core` is an external dependency (`nethack-core @ git+https://github.com/liujonathan24/NetHack-engine.git`); never vendor it.

## File Structure

| path | responsibility |
|---|---|
| `environments/nethack/nethack.py` | Tasks 1, 2, 4 — gate, per-turn scrub, `_apply_tool_call` seam |
| `environments/nethack/nethack_harness/prompt/balrog.py` (+ `balrog_achievements.json`) | Task 3 — real BALROG scorer |
| `environments/nethack/nethack_v1.py` | Tasks 5, 6 — `self_dispatch`, `obs_mode`, call-count referee |
| `tools/cli_harness_eval/workspace.py` | Task 7 — seeded-workspace builder |
| `vendor/verifiers_1985/` | Task 8 — vendored external-harness loader patch |
| `tools/cli_harness_eval/configs/*.toml` | Tasks 9, 10 — per-arm configs |
| `harnesses/nethack-prime-agent/` | Task 10 — installable external harness distribution |
| `tools/cli_harness_eval/launch_cell.sh`, `aggregate_run.py` | Task 11 — launcher + aggregation |

---

### Task 1: Restore the tool gate

Without this a self-dispatching toolset executes any tool name Codex emits, including the
withheld `move`. This is the single most load-bearing fix in the plan.

**Files:**
- Modify: `environments/nethack/nethack.py` (constructor kwargs; `env_response` after `env = state["env"]`; `load_environment` tool-callable block)
- Test: `environments/nethack/tests/test_tool_gate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `NetHackVerifiersEnv.__init__(..., allowed_skill_names: Optional[set] = None)`, attribute `self._allowed_skill_names: set[str]`. Task 4 and Task 5 both rely on the gate running inside `_apply_tool_call`.

- [ ] **Step 1: Write the failing test**

```python
# environments/nethack/tests/test_tool_gate.py
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nethack as m


def test_netplay_env_exposes_no_low_level_move():
    env = m.load_environment(task_spec="full_nle", skill_set="netplay", n_examples=1)
    assert "move" not in env._allowed_skill_names
    assert "explore_and_descend" in env._allowed_skill_names


def test_gate_defaults_to_off_for_backcompat():
    env = m.load_environment(task_spec="full_nle", n_examples=1)
    env._allowed_skill_names = set()
    assert not env._allowed_skill_names  # empty set disables the gate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_tool_gate.py -v`
Expected: FAIL with `AttributeError: 'NetHackVerifiersEnv' object has no attribute '_allowed_skill_names'`

- [ ] **Step 3: Add the constructor kwarg**

In `NetHackVerifiersEnv.__init__`, after the `setup_character: Optional[str] = None,` parameter:

```python
        # Names of the skills actually exposed to the model this rollout (the
        # tool schema it was given). Tool calls for anything NOT in this set are
        # rejected in env_response instead of being dispatched against the full
        # registry — otherwise a hallucinated `move(direction=…)` executes even
        # under the netplay set (which withholds it), leaking the low-level
        # primitive into the effective action surface and confounding
        # cross-encoding comparisons. None/empty disables the gate (back-compat).
        allowed_skill_names: Optional[set] = None,
```

and in the body, after `self._setup_character = setup_character`:

```python
        self._allowed_skill_names = set(allowed_skill_names or ())
```

- [ ] **Step 4: Wire it in `load_environment`**

Immediately after the `interface` if/elif/else that builds `tool_callables`:

```python
    # The exact set of tool names offered to the model — used to gate
    # hallucinated tool calls in env_response (see allowed_skill_names).
    _allowed_skill_names = {getattr(t, "__name__", "") for t in tool_callables} - {""}
```

and add to the `NetHackVerifiersEnv(...)` call, next to `setup_character=character,`:

```python
        allowed_skill_names=_allowed_skill_names,
```

- [ ] **Step 5: Add the rejection branch in `env_response`**

Immediately after `env: NetHackCoreEnv = state["env"]` and **before** the `_DIR_BIND` dir8 rebind:

```python
        # Gate hallucinated tool calls: reject any skill the model was NOT given
        # in its tool schema this rollout, instead of dispatching it against the
        # full registry. Checked on the ORIGINAL name, before the dir8 rebind
        # below (dir8's compass tools ARE in the exposed set). No NLE step is
        # consumed.
        if self._allowed_skill_names and skill_name not in self._allowed_skill_names:
            avail = ", ".join(sorted(self._allowed_skill_names))
            obs_text = self.spec.turn_template(
                state["structured_obs"], state["journal"], state,
                compact=self.compact_obs,
                journal_max_chars=self.journal_render_max_chars,
            )
            content = compose_user_content(
                obs_text,
                [f"[Tool {skill_name!r} is not available. Call one of: {avail}]"],
            )
            return [vf.UserMessage(role="user", content=content)]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_tool_gate.py -v`
Expected: 2 passed

- [ ] **Step 7: Run the ported gate verifier end-to-end**

Run: `PYTHONPATH=.:environments/nethack python tools/encoding_eval/_verify_gate.py`
Expected: reports `move executed = 0`; exit 0

- [ ] **Step 8: Commit**

```bash
git add environments/nethack/nethack.py environments/nethack/tests/test_tool_gate.py
git commit -m "fix(nethack): gate hallucinated tool calls to the exposed skill set"
```

---

### Task 2: Restore the per-turn intro-banner scrub

The setup-time scrub is not enough: every `env.step` re-carries the banner in the unexplored top
tty rows, and standard tiers never repaint them, so it bleeds into the rendered MAP — biasing
ASCII/tty renders while leaving JSON/TOON clean.

**Files:**
- Modify: `environments/nethack/nethack.py` (`setup_state` after `state["meta"] = meta`; `env_response` just before the `obs_text = self.spec.turn_template(...)` render at ~line 1072)
- Test: `environments/nethack/tests/test_banner_scrub.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `state["_standard_tier"]: bool`, read by the render path in `env_response` and (after Task 4) by `_apply_tool_call`.

- [ ] **Step 1: Write the failing test**

```python
# environments/nethack/tests/test_banner_scrub.py
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nethack as m

BANNER = "NetHack, Copyright 1985-2020"


def test_standard_tier_flag_is_set_at_setup():
    spec = m.GAME_SPECS["full_nle"]
    assert spec.nle_task != "engine"   # standard tier => scrubbing applies


def test_scrub_removes_banner_rows():
    import numpy as np
    tty = np.full((24, 80), ord(" "), dtype=np.uint8)
    for i, ch in enumerate(BANNER):
        tty[0, i] = ord(ch)

    class _Obs:
        pass
    obs = _Obs()
    obs.tty_chars = tty
    m._scrub_intro_banner(obs)
    row0 = "".join(chr(c) for c in obs.tty_chars[0]).strip()
    assert "Copyright" not in row0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_banner_scrub.py -v`
Expected: `test_scrub_removes_banner_rows` FAILS if the scrub does not clear row 0; `test_standard_tier_flag_is_set_at_setup` should already pass (it asserts on the existing `GameSpec`).

- [ ] **Step 3: Set the flag in `setup_state`**

After `state["meta"] = meta`:

```python
        # Standard NLE tiers (full_nle etc.) paint the intro/copyright banner
        # over the top tty rows on every step; the engine-driven CurriculumEnv
        # does not, and owns its own map — so per-turn banner scrubbing is gated
        # to the standard tiers only (see the render path in env_response).
        state["_standard_tier"] = spec.nle_task != "engine"
```

- [ ] **Step 4: Re-scrub before the render in `env_response`**

Immediately before `# Build the per-turn user message from the spec's turn template.`:

```python
        # Re-scrub the intro/copyright banner before rendering. Each env.step
        # re-carries the banner in the still-unexplored top tty rows, and in the
        # right-offset-map standard tiers it is never repainted by gameplay — so
        # the one-time scrub in setup_state is not enough. Gated to standard
        # tiers; defensive (never raises).
        if state.get("_standard_tier", True):
            _scrub_intro_banner(state["raw_obs"])
            state["structured_obs"] = shape_observation(state["raw_obs"], state["character"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_banner_scrub.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add environments/nethack/nethack.py environments/nethack/tests/test_banner_scrub.py
git commit -m "fix(nethack): re-scrub intro banner every turn before map render"
```

---

### Task 3: Restore the real BALROG scorer

Hub `balrog.py` is a 53-line self-described *"smooth proxy… approximation"*. The real metric
reproduces `balrog-ai/BALROG` from a vendored achievements table. They disagree materially: at
DL10/XL6 the real metric gives **12.56%**, the proxy **6.4%**.

**Files:**
- Create: `environments/nethack/nethack_harness/prompt/balrog_achievements.json` (2,959 bytes, copied verbatim)
- Modify: `environments/nethack/nethack_harness/prompt/balrog.py`
- Modify: `environments/nethack/pyproject.toml` (package the JSON)
- Test: `environments/nethack/tests/test_balrog_real.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `balrog_progress(max_dlvl: int, xp_level: int, *, ascended: bool = False) -> float` returning 0.0–1.0. `progression_score` / `progression_tier` keep their existing signatures. Task 11's aggregation imports `balrog_progress`.

- [ ] **Step 1: Copy the achievements table and the scorer from engine history**

```bash
ENG=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness
P=environments/nethack/nethack_harness/prompt
git -C "$ENG" show 9baa515^:$P/balrog_achievements.json > $P/balrog_achievements.json
git -C "$ENG" show 9baa515^:$P/balrog.py > $P/balrog.py
wc -c $P/balrog_achievements.json   # expect 2959
```

- [ ] **Step 2: Write the anchor test**

```python
# environments/nethack/tests/test_balrog_real.py
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nethack_harness.prompt.balrog import balrog_progress


def test_matches_published_anchor_dl10_xl6():
    """BALROG's published anchor: DL10 / XL6 == 12.56%."""
    assert round(balrog_progress(10, 6) * 100, 2) == 12.56


def test_monotonic_in_depth():
    assert balrog_progress(6, 1) >= balrog_progress(2, 1)


def test_bounded():
    assert 0.0 <= balrog_progress(1, 1) <= 1.0
    assert 0.0 <= balrog_progress(50, 30) <= 1.0
```

- [ ] **Step 3: Run the test**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_balrog_real.py -v`
Expected: 3 passed. If `test_matches_published_anchor_dl10_xl6` fails, the JSON did not copy — re-run Step 1 and check the byte count.

- [ ] **Step 4: Package the JSON in the wheel**

In `environments/nethack/pyproject.toml`, add to `[tool.hatch.build] include`:

```toml
"nethack_harness/prompt/balrog_achievements.json",
```

- [ ] **Step 5: Verify the package data resolves from an installed layout**

Run: `PYTHONPATH=environments/nethack python -c "from nethack_harness.prompt.balrog import balrog_progress; print(round(balrog_progress(10,6)*100,2))"`
Expected: `12.56`

- [ ] **Step 6: Commit**

```bash
git add environments/nethack/nethack_harness/prompt/balrog.py \
        environments/nethack/nethack_harness/prompt/balrog_achievements.json \
        environments/nethack/pyproject.toml \
        environments/nethack/tests/test_balrog_real.py
git commit -m "feat(nethack): real BALROG progression scorer + vendored achievements table"
```

---

### Task 4: Extract the `_apply_tool_call` seam

`env_response` currently does two jobs: decide *what* the model called, and *apply* it. CLI arms
need only the second half, callable directly. This is a pure refactor — arm 0 behavior must not
change.

**Files:**
- Modify: `environments/nethack/nethack.py` (`env_response`, ~lines 568–1110)
- Test: `environments/nethack/tests/test_apply_tool_call.py`

**Interfaces:**
- Consumes: `self._allowed_skill_names` (Task 1), `state["_standard_tier"]` (Task 2).
- Produces: `async def _apply_tool_call(self, state: vf.State, skill_name: str, skill_args: dict) -> MessageContent` — executes one skill against the engine, mutates `state`, writes the trace entry, and returns the composed user content. Task 5 calls this directly.

- [ ] **Step 1: Write the equivalence test**

```python
# environments/nethack/tests/test_apply_tool_call.py
import asyncio, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nethack as m
import verifiers as vf


def _env_and_state():
    env = m.load_environment(task_spec="full_nle", skill_set="netplay",
                             n_examples=1, explicit_seeds=[0],
                             character="Val-hum-neu-fem")
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(env.setup_state(state))
    return env, state


def test_apply_tool_call_returns_rendered_observation():
    env, state = _env_and_state()
    content = asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    text = m.content_to_text(content) if hasattr(m, "content_to_text") else str(content)
    assert "=== STATUS ===" in text


def test_gate_still_rejects_withheld_move_through_the_seam():
    env, state = _env_and_state()
    content = asyncio.run(env._apply_tool_call(state, "move", {"direction": "N"}))
    text = m.content_to_text(content) if hasattr(m, "content_to_text") else str(content)
    assert "is not available" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_apply_tool_call.py -v`
Expected: FAIL with `AttributeError: 'NetHackVerifiersEnv' object has no attribute '_apply_tool_call'`

- [ ] **Step 3: Move the body into the new method**

Cut everything in `env_response` from the gate branch (Task 1, Step 5) through the trace write, and
paste it into a new method. Replace the final `return [vf.UserMessage(role="user", content=content)]`
statements inside the moved body with bare `return content`.

```python
    async def _apply_tool_call(self, state: vf.State, skill_name: str, skill_args: dict):
        """Execute one skill against the engine and return the rendered observation.

        This is the whole `env_response` body minus tool-call parsing: the gate,
        journal short-circuit, registry dispatch, engine stepping, menu drain,
        terminal detection, reward bookkeeping, banner re-scrub, render, and
        trace write. Split out so MCP-driven CLI harnesses (which must return the
        observation from the tool call itself) share one code path with the
        native harness.
        """
        env: NetHackCoreEnv = state["env"]
        ...  # moved body, unchanged
        return content
```

- [ ] **Step 4: Reduce `env_response` to parse + delegate**

```python
    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        skill_name, skill_args = self._parse_tool_call(messages, state)
        content = await self._apply_tool_call(state, skill_name, skill_args)
        return [vf.UserMessage(role="user", content=content)]
```

Extract the parsing preamble (everything before `env = state["env"]`) into
`_parse_tool_call(self, messages, state) -> tuple[str, dict]`, returning the same
`(skill_name, skill_args)` the original code computed, including its malformed-JSON recovery.

- [ ] **Step 5: Run the new tests**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_apply_tool_call.py -v`
Expected: 2 passed

- [ ] **Step 6: Run the full env suite for regressions**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/ -x -q`
Expected: no new failures versus the pre-refactor baseline. Record the baseline first with
`git stash && pytest ... -q | tail -3 && git stash pop` if unsure.

- [ ] **Step 7: Run golden parity**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_golden_parity.py -v`
Expected: PASS — this is the guard that the refactor did not move behavior.

- [ ] **Step 8: Commit**

```bash
git add environments/nethack/nethack.py environments/nethack/tests/test_apply_tool_call.py
git commit -m "refactor(nethack): extract _apply_tool_call seam from env_response"
```

---

### Task 5: `self_dispatch` and `obs_mode` on the toolset

**Files:**
- Modify: `environments/nethack/nethack_v1.py` (`_build_toolset`, `load_taskset`, `NetHackTasksetConfig`)
- Test: `environments/nethack/tests/test_toolset_self_dispatch.py`

**Interfaces:**
- Consumes: `NetHackVerifiersEnv._apply_tool_call` (Task 4).
- Produces: `_build_toolset(v0env, *, self_dispatch: bool = False, obs_mode: str = "push") -> vf.Toolset`; config fields `NetHackTasksetConfig.self_dispatch: bool = False` and `.obs_mode: str = "push"`.

- [ ] **Step 1: Write the failing test**

```python
# environments/nethack/tests/test_toolset_self_dispatch.py
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nethack_v1 as m


def test_default_toolset_is_harness_driven():
    ts = m.load_taskset({"task_spec": "full_nle", "n_examples": 1})
    assert ts.toolsets[0].scope == "rollout"


def test_self_dispatch_tools_are_wrapped():
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True,
                                 env_args={"skill_set": "netplay"})
    ts = m.load_taskset(cfg)
    names = {t.__name__ for t in ts.toolsets[0].tools}
    assert "explore_and_descend" in names
    assert "move" not in names          # netplay withholds it
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_toolset_self_dispatch.py -v`
Expected: FAIL — `NetHackTasksetConfig` forbids the extra field `self_dispatch` (`extra="forbid"`).

- [ ] **Step 3: Add the config fields**

In `NetHackTasksetConfig`, after `trace_dir`:

```python
    # When True, each tool executes the skill and RETURNS the rendered
    # observation, so MCP-driven CLI agents get self-contained calls. False =
    # the v0/harness-driven loop where env_response applies the previous call.
    self_dispatch: bool = False
    # "push" = every tool result carries the observation; "on_demand" = terse
    # feedback only, map gated behind an explicit look() call.
    obs_mode: str = "push"
```

- [ ] **Step 4: Implement the wrapper in `_build_toolset`**

```python
def _build_toolset(v0env, *, self_dispatch: bool = False,
                   obs_mode: str = "push") -> vf.Toolset:
    ...  # existing _nethack_setup / _nethack_cleanup unchanged

    tools = list(v0env.tools)
    if self_dispatch:
        tools = [_self_dispatching(v0env, t, obs_mode) for t in tools]

    return vf.Toolset(
        tools=tools,
        setups=[_nethack_setup],
        cleanups=[_nethack_cleanup],
        scope="rollout",
    )


def _self_dispatching(v0env, tool, obs_mode: str):
    """Wrap a schema-only v0 tool adapter into one that actually executes.

    The v0 adapters carry the JSON schema (name/signature/docstring) but do not
    touch the engine — dispatch lives in env_response. CLI harnesses call tools
    over MCP and must get the result back from the call itself, so we bind the
    adapter's identity to `_apply_tool_call`.
    """
    import functools

    @functools.wraps(tool)
    async def _run(state, **kwargs):
        content = await v0env._apply_tool_call(state, tool.__name__, kwargs)
        if obs_mode == "on_demand" and tool.__name__ != "look":
            return _terse(content)
        return content

    return _run


def _terse(content) -> str:
    """Strip everything but MESSAGES + the trailing feedback line."""
    from nethack_harness.prompt.content import content_to_text
    text = content_to_text(content)
    keep, emit = ("=== MESSAGES ===", "["), []
    for line in text.splitlines():
        if line.startswith(keep) or line.startswith(emit):
            emit.append(line)
    return "\n".join(emit) or "(no message)"
```

- [ ] **Step 5: Thread the flags through `load_taskset`**

```python
        toolsets=[_build_toolset(v0env, self_dispatch=cfg.self_dispatch,
                                 obs_mode=cfg.obs_mode)],
```

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_toolset_self_dispatch.py environments/nethack/tests/test_v1_taskset.py -v`
Expected: all passed — `test_v1_taskset.py` proves arm 0 is unaffected.

- [ ] **Step 7: Commit**

```bash
git add environments/nethack/nethack_v1.py environments/nethack/tests/test_toolset_self_dispatch.py
git commit -m "feat(nethack_v1): self-dispatching toolset mode for MCP-driven CLI harnesses"
```

---

### Task 6: Toolset-side 150-call referee

`max_turns` only binds harnesses we control. The cap must live where every arm passes through.

**Files:**
- Modify: `environments/nethack/nethack_v1.py`
- Test: `environments/nethack/tests/test_call_budget.py`

**Interfaces:**
- Consumes: `_self_dispatching(v0env, tool, obs_mode)` (Task 5).
- Produces: config field `NetHackTasksetConfig.max_skill_calls: int = 150`; state key `state["skill_calls"]: int`; terminal reason `"call_budget_exhausted"`. **Widens Task 5's signature to `_self_dispatching(v0env, tool, obs_mode, budget)`** — update the call site in `_build_toolset` in the same commit.

- [ ] **Step 1: Write the failing test**

```python
# environments/nethack/tests/test_call_budget.py
import asyncio, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import nethack_v1 as m
import verifiers as vf


def test_budget_refuses_past_the_cap():
    cfg = m.NetHackTasksetConfig(task_spec="full_nle", n_examples=1,
                                 self_dispatch=True, max_skill_calls=3,
                                 explicit_seeds=[0],
                                 env_args={"skill_set": "netplay"})
    ts = m.load_taskset(cfg)
    tool = next(t for t in ts.toolsets[0].tools if t.__name__ == "search")
    state = vf.State({"task": {"tier": "full_nle", "seed": 0}, "info": {}})
    asyncio.run(ts.toolsets[0].setups[0](None, state))

    for _ in range(3):
        asyncio.run(tool(state, times=1))
    assert state["skill_calls"] == 3

    out = asyncio.run(tool(state, times=1))
    assert "budget" in str(out).lower()
    assert state["skill_calls"] == 3          # refused calls do not count
    assert state["terminated"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_call_budget.py -v`
Expected: FAIL — `max_skill_calls` is not a config field.

- [ ] **Step 3: Add the config field**

```python
    # Hard per-rollout budget on executed skill calls. Enforced toolset-side so
    # it binds every harness, including CLI agents whose internal loop we do not
    # control. One call == one v0 LM turn.
    max_skill_calls: int = 150
```

- [ ] **Step 4: Enforce it in the wrapper**

Inside `_run` in `_self_dispatching`, before dispatch:

```python
        used = int(state.get("skill_calls", 0))
        if budget > 0 and used >= budget:
            state["terminated"] = True
            state["stop_reason"] = "call_budget_exhausted"
            return f"[Call budget exhausted: {budget} skill calls used. The episode is over.]"
        state["skill_calls"] = used + 1
```

Thread `budget` into `_self_dispatching(v0env, tool, obs_mode, budget)` and pass
`cfg.max_skill_calls` from `_build_toolset`.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=environments/nethack pytest environments/nethack/tests/test_call_budget.py -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add environments/nethack/nethack_v1.py environments/nethack/tests/test_call_budget.py
git commit -m "feat(nethack_v1): toolset-side skill-call budget as the cross-arm referee"
```

---

### Task 7: Seeded-workspace builder

**Files:**
- Create: `tools/cli_harness_eval/__init__.py`, `tools/cli_harness_eval/workspace.py`
- Test: `tests/test_cli_workspace.py`

**Interfaces:**
- Consumes: `nethack_harness.prompt.rendering.SYSTEM_PROMPT`, `environments/nethack/wiki/snapshot.json`.
- Produces: `build_workspace(dest: Path, *, objective: str) -> Path`, creating `AGENTS.md`, `CLAUDE.md`, `wiki/*.md`, `memory/objective.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_workspace.py
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.cli_harness_eval.workspace import build_workspace


def test_builds_all_three_affordances(tmp_path):
    ws = build_workspace(tmp_path / "ws", objective="Descend as deep as you can.")
    assert (ws / "AGENTS.md").exists()
    assert (ws / "CLAUDE.md").read_text() == (ws / "AGENTS.md").read_text()
    assert len(list((ws / "wiki").glob("*.md"))) >= 100
    assert "Descend" in (ws / "memory" / "objective.md").read_text()


def test_agents_md_carries_the_system_prompt(tmp_path):
    ws = build_workspace(tmp_path / "ws", objective="x")
    assert "STRATEGY: DESCEND ASAP" in (ws / "AGENTS.md").read_text()


def test_no_map_is_seeded(tmp_path):
    ws = build_workspace(tmp_path / "ws", objective="x")
    assert not (ws / "map").exists()
    assert not list(ws.glob("*map*"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_workspace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.cli_harness_eval.workspace'`

- [ ] **Step 3: Implement the builder**

```python
# tools/cli_harness_eval/workspace.py
"""Build the sandbox workspace handed to CLI-agent arms.

Each item is the filesystem counterpart of an affordance the native harness
supplies through prompt and tools, so the arms are capability-matched:
  AGENTS.md   <- SYSTEM_PROMPT
  wiki/       <- wiki_lookup / wiki_search
  memory/     <- Journal (objective + notes)
No map is seeded: it would duplicate the observation channel and destroy the
"does the scaffold build its own map notes?" measurement.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

_ENV = Path(__file__).resolve().parents[2] / "environments" / "nethack"
_SNAPSHOT = _ENV / "wiki" / "snapshot.json"
# Filenames each CLI looks for; all get the same bytes.
_PROMPT_FILES = ("AGENTS.md", "CLAUDE.md")


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "page"


def build_workspace(dest: Path, *, objective: str) -> Path:
    import sys
    sys.path.insert(0, str(_ENV))
    from nethack_harness.prompt.rendering import SYSTEM_PROMPT

    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)           # memory/ is wiped per rollout
    (dest / "wiki").mkdir(parents=True)
    (dest / "memory").mkdir()

    primer = SYSTEM_PROMPT + (
        "\n\n=== WORKSPACE ===\n"
        "`wiki/` holds NetHack reference pages (read-only) — grep it.\n"
        "`memory/` is yours: keep notes there across turns. `memory/objective.md`"
        " is your goal.\n"
    )
    for name in _PROMPT_FILES:
        (dest / name).write_text(primer)

    pages = json.loads(_SNAPSHOT.read_text())
    for page in pages:
        path = dest / "wiki" / f"{_slug(page['title'])}.md"
        path.write_text(f"# {page['title']}\n\n{page['body']}\n")
        path.chmod(0o444)

    (dest / "memory" / "objective.md").write_text(f"# Objective\n\n{objective}\n")
    return dest
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli_workspace.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tools/cli_harness_eval/__init__.py tools/cli_harness_eval/workspace.py tests/test_cli_workspace.py
git commit -m "feat(cli-eval): seeded-workspace builder (AGENTS.md, wiki/, memory/)"
```

---

### Task 8: Vendor the external-harness loader and pin verifiers

**Files:**
- Create: `vendor/verifiers_1985/README.md`, `vendor/verifiers_1985/loaders.patch`
- Modify: `environments/nethack/pyproject.toml` (verifiers floor)
- Test: `tests/test_vendored_loader.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed verifiers whose `verifiers.v1.loaders.harness_class(<id>)` resolves an external top-level module. Task 10 depends on it.

- [ ] **Step 1: Discover which verifiers version ships the Codex harness**

```bash
pip index versions verifiers 2>/dev/null | head -3
python - <<'PY'
import importlib, pkgutil
m = importlib.import_module("verifiers.v1.packages.harnesses")
print(sorted(n for _, n, _ in pkgutil.iter_modules(m.__path__)))
PY
```

Record the installed version and the module list. If `codex` is absent, upgrade
(`uv pip install -U verifiers`) and re-run until it appears. Write the resolved version into
`environments/nethack/pyproject.toml` as the new floor, replacing `verifiers>=0.1.14`.

- [ ] **Step 2: Fetch the PR patch**

```bash
mkdir -p vendor/verifiers_1985
gh pr diff 1985 --repo PrimeIntellect-ai/verifiers > vendor/verifiers_1985/loaders.patch
grep -c "^diff --git" vendor/verifiers_1985/loaders.patch     # expect 12
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_vendored_loader.py
import pytest


def test_external_harness_ids_resolve():
    from verifiers.v1.loaders import import_harness
    with pytest.raises(ModuleNotFoundError) as e:
        import_harness("definitely-not-installed-harness")
    # The patched loader reports BOTH import candidates.
    assert "verifiers.v1.harnesses." in str(e.value)
    assert "definitely_not_installed_harness" in str(e.value)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_vendored_loader.py -v`
Expected: FAIL — unpatched verifiers reports only one candidate (or `import_harness` is absent).

- [ ] **Step 5: Apply the patch to the installed package**

```bash
SITE=$(python -c "import verifiers, pathlib; print(pathlib.Path(verifiers.__file__).parent.parent)")
git apply --directory="$(basename "$SITE")" --exclude='tests/*' --exclude='docs/*' \
    -p1 vendor/verifiers_1985/loaders.patch || \
  echo "patch did not apply cleanly — reconcile against the installed version, then re-run"
```

Record the exact command that worked in `vendor/verifiers_1985/README.md`, along with the
verifiers version it was applied to, so the run is reproducible.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_vendored_loader.py -v`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add vendor/verifiers_1985/ environments/nethack/pyproject.toml tests/test_vendored_loader.py
git commit -m "chore(cli-eval): vendor verifiers #1985 external-harness loader; pin version"
```

---

### Task 9: Codex arm

**Files:**
- Create: `tools/cli_harness_eval/configs/codex.toml`, `tools/cli_harness_eval/configs/control.toml`
- Test: `tests/test_arm_configs.py`

**Interfaces:**
- Consumes: Tasks 5–8.
- Produces: two loadable `SingleAgentEnvConfig` TOMLs. Task 11's launcher reads them by name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arm_configs.py
import pathlib, tomllib
CFG = pathlib.Path(__file__).resolve().parents[1] / "tools" / "cli_harness_eval" / "configs"


def test_every_arm_pins_the_same_fixed_factors():
    arms = {p.stem: tomllib.loads(p.read_text()) for p in CFG.glob("*.toml")}
    assert arms, "no arm configs found"
    seeds = {tuple(a["taskset"]["explicit_seeds"]) for a in arms.values()}
    models = {a["agent"]["harness"]["model"] for a in arms.values()}
    chars = {a["taskset"]["env_args"]["character"] for a in arms.values()}
    assert len(seeds) == 1 and len(models) == 1 and len(chars) == 1
    assert models == {"z-ai/glm-5.2"}


def test_cli_arms_self_dispatch_and_control_does_not():
    ctl = tomllib.loads((CFG / "control.toml").read_text())
    cdx = tomllib.loads((CFG / "codex.toml").read_text())
    assert ctl["taskset"]["self_dispatch"] is False
    assert cdx["taskset"]["self_dispatch"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_arm_configs.py -v`
Expected: FAIL with `AssertionError: no arm configs found`

- [ ] **Step 3: Write `control.toml`**

```toml
# Arm 0 — the control: our harness, native tools, harness-driven loop.
[taskset]
id = "nethack"
task_spec = "full_nle"
self_dispatch = false
max_skill_calls = 150
explicit_seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
n_examples = 16

[taskset.env_args]
skill_set = "netplay"
character = "Val-hum-neu-fem"
compact_obs = false

[agent.harness]
id = "nethack"
model = "z-ai/glm-5.2"
```

- [ ] **Step 4: Write `codex.toml`**

```toml
# Arm 1 — Codex CLI, driving the same toolset over MCP.
[taskset]
id = "nethack"
task_spec = "full_nle"
self_dispatch = true
obs_mode = "push"
max_skill_calls = 150
explicit_seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
n_examples = 16

[taskset.env_args]
skill_set = "netplay"
character = "Val-hum-neu-fem"
compact_obs = false

[agent.harness]
id = "codex"
model = "z-ai/glm-5.2"

[agent.harness.sandbox]
scope = "rollout"
network_access = true
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_arm_configs.py -v`
Expected: 2 passed

- [ ] **Step 6: Confirm how the toolset is exposed over MCP, and where the URL comes from**

The framework — not us — serves toolsets to CLI harnesses over MCP; the PR fixture shows
`async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls)`. Determine the concrete
contract for the pinned version before writing any harness that consumes it:

```bash
python - <<'PY'
import inspect
from verifiers.v1.packages.harnesses.cli import CLIHarness
from verifiers.v1.utils import mcp_proxy_utils as mp
print(inspect.signature(CLIHarness.__init__))
print([n for n in dir(mp) if not n.startswith("_")])
PY
```

Record in `tools/cli_harness_eval/configs/README.md`: how a `Toolset` becomes an MCP endpoint,
whether the URL arrives via `launch(mcp_urls=...)` or an env var in the sandbox, and the exact
name. Task 10's `mcp_url` parameter must match what you find here.

- [ ] **Step 7: Commit**

```bash
git add tools/cli_harness_eval/configs/ tests/test_arm_configs.py
git commit -m "feat(cli-eval): control and Codex arm configs"
```

---

### Task 10: Prime Agent external harness

Prime Agent supports **only** remote `"http"` MCP servers, and exposes them as a Python skill in
its IPython kernel rather than as agent tools (`docs/MCP_INTEGRATIONS.md`).

**Files:**
- Create: `harnesses/nethack-prime-agent/pyproject.toml`
- Create: `harnesses/nethack-prime-agent/nethack_prime_agent/__init__.py`
- Create: `tools/cli_harness_eval/configs/prime_agent.toml`
- Test: `tests/test_prime_agent_harness.py`

**Interfaces:**
- Consumes: Task 8's patched loader; `CLIHarness` from the pinned verifiers.
- Produces: an installable distribution named `nethack-prime-agent` whose module `nethack_prime_agent` exports exactly one `Harness` subclass, `PrimeAgentHarness`, resolvable by `harness.id = "nethack-prime-agent"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prime_agent_harness.py
def test_module_exports_exactly_one_harness():
    import nethack_prime_agent as m
    from verifiers.v1 import Harness
    exported = [getattr(m, n) for n in m.__all__]
    harnesses = [o for o in exported if isinstance(o, type) and issubclass(o, Harness)]
    assert len(harnesses) == 1
    assert harnesses[0].__name__ == "PrimeAgentHarness"


def test_id_resolves_through_the_loader():
    from verifiers.v1.loaders import harness_class
    assert harness_class("nethack-prime-agent").__name__ == "PrimeAgentHarness"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prime_agent_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nethack_prime_agent'`

- [ ] **Step 3: Write the distribution metadata**

```toml
# harnesses/nethack-prime-agent/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nethack-prime-agent"
version = "0.1.0"
requires-python = ">=3.11"

[tool.hatch.build.targets.wheel]
packages = ["nethack_prime_agent"]
```

- [ ] **Step 4: Write the harness**

```python
# harnesses/nethack-prime-agent/nethack_prime_agent/__init__.py
"""Prime Agent harness: drives the NetHack toolset through Prime Agent's
kernel-side MCP integration.

Prime Agent supports only remote "http" MCP servers, and surfaces them as a
Python skill imported into its IPython kernel rather than as agent tools — so
the agent calls `await nethack.explore_and_descend(...)`. Same server, same 15
tools; only the integration path differs from Codex.
"""
import json

from verifiers.v1 import HarnessConfig
from verifiers.v1.packages.harnesses.cli import CLIHarness

__all__ = ["PrimeAgentHarness", "PrimeAgentHarnessConfig"]

_SETTINGS = "/app/.prime/agent/settings.json"


class PrimeAgentHarnessConfig(HarnessConfig):
    mcp_bearer_env: str = "NETHACK_MCP_TOKEN"
    agent_workdir: str = "/app"


class PrimeAgentHarness(CLIHarness):
    # Config specialization. PR #1985 documents the generic form
    # `Harness[SomeConfig]`; in verifiers 0.1.14 `CLIHarness` is NOT subscriptable
    # and the specialization is declared with this ClassVar instead. Check which
    # form the pinned version accepts (Task 8, Step 1) and use that one — if
    # `CLIHarness[PrimeAgentHarnessConfig]` imports cleanly, prefer it and drop
    # this line.
    config_type = PrimeAgentHarnessConfig

    def __init__(self, *, mcp_url: str = "", **kwargs):
        settings = json.dumps(
            {
                "mcpServers": {
                    "nethack": {
                        "type": "http",
                        "url": mcp_url,
                        "bearerTokenEnvVar": "NETHACK_MCP_TOKEN",
                    }
                }
            },
            indent=2,
        )
        super().__init__(
            command="prime-agent -p --model $VF_MODEL @AGENTS.md 'Play NetHack. "
                    "Use the nethack skill; descend as deep as you can.'",
            files={_SETTINGS: settings},
            **kwargs,
        )
```

- [ ] **Step 5: Install it editable and run the tests**

Run:
```bash
uv pip install -e harnesses/nethack-prime-agent
pytest tests/test_prime_agent_harness.py -v
```
Expected: 2 passed. If `test_id_resolves_through_the_loader` fails, Task 8's patch is not applied
to the active interpreter — re-run Task 8, Step 5.

- [ ] **Step 6: Write `prime_agent.toml`**

Copy `codex.toml` verbatim and change only the harness block:

```toml
[agent.harness]
id = "nethack-prime-agent"
model = "z-ai/glm-5.2"
```

- [ ] **Step 7: Commit**

```bash
git add harnesses/ tools/cli_harness_eval/configs/prime_agent.toml tests/test_prime_agent_harness.py
git commit -m "feat(cli-eval): Prime Agent external harness plugin"
```

---

### Task 11: Launcher, aggregation, and the GLM billing fix

**Files:**
- Create: `tools/cli_harness_eval/launch_cell.sh`
- Modify: `configs/endpoints.toml` (GLM team billing)
- Modify: `tools/encoding_eval/aggregate_run.py` → extend for `turns_used`
- Test: `tests/test_cli_aggregate.py`

**Interfaces:**
- Consumes: Tasks 3, 9, 10.
- Produces: `launch_cell.sh <ARM> <OUTDIR> [MAX_CALLS] [N]`; `outputs/cli_harness_eval/<run>/<arm>/table.md`.

- [ ] **Step 1: Fix GLM team billing first**

`z-ai/glm-5.2` currently sits under `pinference-glm`, which carries no `X-Prime-Team-ID`, so it
bills the $0 personal balance and **hangs** rather than erroring. Add the model to the funded
block in `configs/endpoints.toml`:

```toml
[[endpoints]]
id   = "prime-team"
url  = "https://api.pinference.ai/api/v1"
key  = "PI_API_KEY"
headers = { "X-Prime-Team-ID" = "cmotasmp5005ppyp07dcoh50u" }
models = ["google/gemini-3-flash-preview", "google/gemini-3.1-pro-preview",
          "google/gemini-3.5-flash", "z-ai/glm-5.2"]
```

- [ ] **Step 2: Verify billing before anything else runs**

```bash
curl -s https://api.pinference.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $PI_API_KEY" \
  -H "X-Prime-Team-ID: cmotasmp5005ppyp07dcoh50u" \
  -H 'Content-Type: application/json' \
  -d '{"model":"z-ai/glm-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | head -c 300
```
Expected: a completion. If you see `insufficient_funds`, stop — every later step will hang, not fail.

- [ ] **Step 3: Write the launcher**

```bash
#!/usr/bin/env bash
# Launch ONE arm of the CLI-harness comparison.
# Every arm must go through THIS script so the fixed factors never drift.
#   tools/cli_harness_eval/launch_cell.sh <ARM> <OUTDIR> [MAX_CALLS] [N]
set -euo pipefail

ARM="${1:?usage: launch_cell.sh <ARM> <OUTDIR> [MAX_CALLS] [N]}"
OUTDIR="${2:?usage: launch_cell.sh <ARM> <OUTDIR> [MAX_CALLS] [N]}"
MAX_CALLS="${3:-150}"
N="${4:-16}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH=".:environments/nethack"
export PI_API_KEY="${PI_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")}"
export NETHACK_MCP_TOKEN="${NETHACK_MCP_TOKEN:?set a bearer token for the MCP server}"

CFG="tools/cli_harness_eval/configs/${ARM}.toml"
[ -f "$CFG" ] || { echo "no such arm: $ARM ($CFG)"; exit 2; }

mkdir -p "$OUTDIR/trace"
echo "[launch_cell] arm=${ARM} max_calls=${MAX_CALLS} n=${N} out=${OUTDIR}"

exec .venv/bin/vf eval --config "$CFG" \
  --set "taskset.max_skill_calls=${MAX_CALLS}" \
  --set "taskset.n_examples=${N}" \
  --set "taskset.trace_dir=${OUTDIR}/trace" \
  --output-dir "$OUTDIR"
```

Then `chmod +x tools/cli_harness_eval/launch_cell.sh`.

- [ ] **Step 4: Write the aggregation test**

```python
# tests/test_cli_aggregate.py
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.encoding_eval.aggregate_run import _trace_rollouts


def test_turns_used_is_read_from_traces(tmp_path):
    cell = tmp_path / "codex"
    (cell / "trace").mkdir(parents=True)
    lines = [
        {"max_dlvl_reached": 1, "skill_calls": 1, "status": {"hitpoints": 12, "experience_level": 1}},
        {"max_dlvl_reached": 3, "skill_calls": 2, "status": {"hitpoints": 0, "experience_level": 2}},
    ]
    (cell / "trace" / "r0.ndjson").write_text("\n".join(json.dumps(x) for x in lines))
    rows = _trace_rollouts(str(cell))
    assert rows[0][0] == 3      # max dlvl
    assert rows[0][1] is True   # died
```

- [ ] **Step 5: Run test to verify it fails, then extend `_trace_rollouts`**

Run: `pytest tests/test_cli_aggregate.py -v` — expect FAIL if `skill_calls` is unhandled.

In `tools/encoding_eval/aggregate_run.py`, change `_trace_rollouts` to carry the call count:

```python
def _trace_rollouts(cell_dir):
    """Per rollout: (max_dlvl, died, max_xp_level, turns_used)."""
    out = []
    for f in sorted(glob.glob(os.path.join(cell_dir, "trace", "*.ndjson"))):
        md, died, mx, calls = 1, False, 1, 0
        for line in open(f):
            try:
                t = json.loads(line)
            except ValueError:
                continue
            md = max(md, t.get("max_dlvl_reached") or t.get("dlvl") or 1)
            calls = max(calls, int(t.get("skill_calls") or 0))
            st = t.get("status")
            if isinstance(st, dict):
                if st.get("hitpoints") == 0:
                    died = True
                mx = max(mx, st.get("experience_level") or 1)
        out.append((md, died, mx, calls))
    return out
```

Then update every unpacking site of `_trace_rollouts(...)` from 3-tuples to 4-tuples, and add a
`turns_used` column (mean of the fourth element) to the table renderer next to `Alive@cap`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_cli_aggregate.py -v`
Expected: 1 passed

- [ ] **Step 7: Smoke every arm before fanning out**

```bash
for arm in control codex prime_agent; do
  tools/cli_harness_eval/launch_cell.sh "$arm" "outputs/cli_harness_eval/smoke/$arm" 20 1
done
```
Expected per arm: engine reached, workspace seeded (CLI arms), observation rendered, `skill_calls > 0`,
`move` executed = 0. **All three must pass before any n=16 run.**

- [ ] **Step 8: Commit**

```bash
git add tools/cli_harness_eval/launch_cell.sh configs/endpoints.toml \
        tools/encoding_eval/aggregate_run.py tests/test_cli_aggregate.py
git commit -m "feat(cli-eval): arm launcher, turns_used aggregation, GLM team billing"
```

---

## Run procedure (after Task 11)

1. Smoke all three arms at n=1 / 20 calls. Any failure stops the fan-out.
2. Fan out: `launch_cell.sh <arm> outputs/cli_harness_eval/run1/<arm> 150 16` per arm.
3. Aggregate: `PYTHONPATH=.:environments/nethack python tools/encoding_eval/aggregate_run.py outputs/cli_harness_eval/run1`.
4. Verify: independent pass over all 48 traces against the A1–A16 contract in
   `docs/experiments/exp1-trace-assumptions.md`, plus `move` executed = 0 in every trace.
   For the CLI arms additionally audit for **out-of-band state reads** (spec §9.4): grep each
   agent's shell transcript for reads outside the workspace — any hit on the engine, level files,
   or repo invalidates that rollout. The workspace is the only legitimate filesystem.
5. Report paired per-seed deltas and an exact sign test across the 16 seeds. Marginal SEs at n=16
   are wide; never quote them alone.
