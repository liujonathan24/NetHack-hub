"""Arm-config parity: `control.toml`, `claude_code.toml` and `prime_agent.toml`
must pin the identical experiment (model, seeds, character, task, skill set) --
otherwise a difference in the results table could be a confound instead of a
scaffold effect.

Key-shape gotcha, load-bearing for every assertion below: the control arm runs
through verifiers 0.2.x's v0 legacy bridge, so its knobs live under `[args]`
(kwargs forwarded to `load_environment`); the two CLI arms run through the v1
taskset, so theirs live under `[taskset]` (+ `[taskset.env_args]` for
`skill_set`, which is an env-side kwarg, not a taskset field). A parity check
that reads only one shape silently compares nothing for the other two arms --
see `tools/cli_harness_eval/configs/README.md` Sec 10 (this is the exact bug
class `test_prime_agent_harness.py::test_the_three_arms_pin_the_same_experiment`
was written to catch, and also very nearly repeated -- see the trace_dir test
below).
"""

from __future__ import annotations

import pathlib
import tomllib

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "tools/cli_harness_eval/configs"
ARM_FILES = ("control.toml", "claude_code.toml", "prime_agent.toml")


def _load(name: str) -> dict:
    return tomllib.loads((CONFIGS / name).read_text())


def _knobs(name: str) -> dict:
    """Normalize the two key shapes into one dict of the fields every arm
    must pin identically."""
    cfg = _load(name)
    if name == "control.toml":
        block = cfg["args"]
        skill_set = block["skill_set"]
    else:
        block = cfg["taskset"]
        skill_set = cfg["taskset"]["env_args"]["skill_set"]
    return {
        "model": cfg["model"],
        "task_spec": block["task_spec"],
        "character": block["character"],
        "explicit_seeds": list(block["explicit_seeds"]),
        "skill_set": skill_set,
        "trace_dir": block["trace_dir"],
    }


def test_all_three_arms_pin_the_identical_experiment():
    knobs = {name: _knobs(name) for name in ARM_FILES}
    for field in ("model", "task_spec", "character", "explicit_seeds", "skill_set"):
        values = {name: k[field] for name, k in knobs.items()}
        distinct = {tuple(v) if isinstance(v, list) else v for v in values.values()}
        assert len(distinct) == 1, (field, values)

    control = knobs["control.toml"]
    assert control["model"] == "z-ai/glm-5.2"
    assert control["explicit_seeds"] == list(range(16))
    assert control["character"] == "Val-hum-neu-fem"
    assert control["task_spec"] == "full_nle"
    # The skill set is an EXPERIMENT VARIABLE — 18-tool `netplay` (ours) vs the 31
    # vendored upstream skills in `netplay_true`. Pin it to a known set rather than
    # one literal; the property that must hold is that all three arms AGREE, which
    # the parity loop above already enforces across every arm.
    assert control["skill_set"] in ("netplay", "netplay_true")


def test_claude_code_clamps_bash_but_keeps_read():
    """Task 18 Step 3: the arm made ZERO Bash/Edit/Read calls across all five
    run1 rollouts (analysis-claude-code.md §3), so disallowing Bash costs
    nothing measurable and removes the out-of-band host-shell path. Read stays
    enabled -- it is the capability match for the control arm's wiki tools
    (`memory/`/`wiki/` in the seeded workspace, tools/cli_harness_eval/
    workspace.py). Checked against the exact argv-building expression
    `ClaudeCodeHarness.launch` uses so this fails if that logic ever changes
    shape (verifiers/v1/harnesses/claude_code/harness.py)."""
    from verifiers.v1.harnesses.claude_code.harness import ClaudeCodeHarnessConfig

    harness_cfg = ClaudeCodeHarnessConfig.model_validate(_load("claude_code.toml")["harness"])

    argv_fragment = [
        arg
        for tool in harness_cfg.disabled_tools or []
        for arg in ("--disallowedTools", tool)
    ]
    assert "--disallowedTools" in argv_fragment
    assert argv_fragment[argv_fragment.index("--disallowedTools") + 1] == "Bash"
    assert "Read" not in argv_fragment


def test_every_trace_dir_is_absolute_and_mutually_distinct():
    """Absolute: a relative `trace_dir` resolves inside a rollout's own
    ephemeral runtime workdir and is silently discarded at teardown (measured
    in Task 10/13; see the `trace_dir` notes in each config). Distinct: three
    arms writing into the same directory would clobber each other's per-turn
    NDJSON.

    Checked pairwise with a set, not the `a != b != c` chained-comparison
    pattern (`test_prime_agent_harness.py`'s original version of this check)
    -- that only evaluates `(a != b) and (b != c)` and never compares `a`
    against `c`, so it would pass silently even if the control and
    claude_code arms shared a directory.
    """
    dirs = {name: _knobs(name)["trace_dir"] for name in ARM_FILES}
    for name, trace_dir in dirs.items():
        assert pathlib.PurePosixPath(trace_dir).is_absolute(), (name, trace_dir)
    assert len(set(dirs.values())) == len(dirs), dirs
