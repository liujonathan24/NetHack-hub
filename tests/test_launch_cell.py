"""tools/cli_harness_eval/launch_cell.sh: the one on-ramp every arm's rollout
must go through, so the fixed factors (model, seeds, character, skill_set --
see test_arm_configs.py) can't drift between arms.

These tests never launch a real rollout (no LLM calls, no MCP tool server).
`EVAL_BIN` is pointed at a stub that records its argv and exits 0, so what's
under test is purely the shell script's arm resolution and override
construction -- in particular the `[args]` (control) vs `[taskset]` (CLI
arms) key-shape split, and the JSON-vs-string-typing landmine documented
inline in the script (`--args.max_turns 150` silently becomes the string
"150"; only a `--args '{"max_turns": 150}'` JSON blob round-trips as an int).
"""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/cli_harness_eval/launch_cell.sh"

STUB_EVAL = """#!/usr/bin/env python3
import json, sys, os
with open(os.environ["STUB_EVAL_ARGV_FILE"], "w") as f:
    json.dump(sys.argv[1:], f)
"""


def _make_stub(tmp_path):
    stub = tmp_path / "eval"
    stub.write_text(STUB_EVAL)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


def _run(tmp_path, args, env_extra=None):
    stub = _make_stub(tmp_path)
    argv_file = tmp_path / "argv.json"
    env = dict(**_base_env(), EVAL_BIN=str(stub), STUB_EVAL_ARGV_FILE=str(argv_file))
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env
    )
    argv = json.loads(argv_file.read_text()) if argv_file.exists() else None
    return result, argv


def _base_env():
    import os

    # Keep PATH (for bash/python3) but nothing else -- the script must not
    # depend on ambient PYTHONPATH etc.
    #
    # ALLOW_STALE_ENGINE=1 because these tests are about ARGUMENT CONSTRUCTION,
    # not about the engine: the preflight added in launch_cell.sh reads the real
    # engine on this box, and on a dev box mid-rebuild the .so legitimately lags
    # the source (measured: libnethack.so 2026-07-22 04:26 vs src/src/nle.c
    # 2026-07-31 18:52). Without the bypass, every test here would flip red for
    # a reason none of them is testing. The preflight itself is covered by
    # test_engine_provenance.py against a synthetic engine tree.
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "ALLOW_STALE_ENGINE": "1",
    }


def test_refuses_an_unknown_arm(tmp_path):
    result, argv = _run(tmp_path, ["not_an_arm", str(tmp_path / "out")])
    assert result.returncode == 2
    assert "unknown arm" in result.stderr
    assert argv is None


def test_missing_arguments_prints_usage(tmp_path):
    result, argv = _run(tmp_path, ["control"])
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert argv is None


def test_echoes_the_resolved_settings_before_running(tmp_path):
    out = tmp_path / "out"
    result, _ = _run(tmp_path, ["claude_code", str(out), "20", "1"])
    assert result.returncode == 0, result.stderr
    # Not `splitlines()[0]`: the engine preflight prints its own `[engine] ...`
    # fingerprint line first, and which line comes first is not the contract.
    lines = [ln for ln in result.stdout.strip().splitlines() if "[launch_cell]" in ln]
    assert len(lines) == 1, result.stdout
    line = lines[0]
    assert "arm=claude_code" in line
    assert "max_calls=20" in line
    assert "n=1" in line
    assert str(out) in line


def test_control_arm_overrides_args_as_a_typed_json_blob(tmp_path):
    """The control arm's budget knob lives in the untyped `[args]` dict
    (v0 legacy-bridge kwargs). A bare `--args.max_turns 150` would silently
    become the STRING "150" and crash nethack.py's `self.max_turns > 0`
    check against a real GLM run -- verified live against the pinned
    verifiers CLI (pydantic_config only JSON-decodes a sub-field value
    that starts with `{`/`[`). The script must instead pass one `--args`
    JSON object so `max_turns` round-trips as a real int."""
    out = tmp_path / "out"
    result, argv = _run(tmp_path, ["control", str(out), "37", "5"])
    assert result.returncode == 0, result.stderr
    assert "@" in argv
    cfg_idx = argv.index("@")
    assert argv[cfg_idx + 1].endswith("configs/control.toml")
    assert "--num_tasks" in argv
    assert argv[argv.index("--num_tasks") + 1] == "5"
    assert "--output_dir" in argv
    assert argv[argv.index("--output_dir") + 1] == str(out)
    assert "--args" in argv
    blob = json.loads(argv[argv.index("--args") + 1])
    assert blob["max_turns"] == 37
    assert isinstance(blob["max_turns"], int)
    assert blob["trace_dir"] == str(out / "turns")
    # Must NOT also emit a plain --args.max_turns override (the typing trap).
    assert not any(a.startswith("--args.") for a in argv)


def test_cli_arms_override_taskset_fields_directly(tmp_path):
    """`[taskset]` is a typed sub-model (unlike `[args]`), so a plain dotted
    override coerces to a real int -- verified live. No JSON blob needed."""
    out = tmp_path / "out"
    result, argv = _run(tmp_path, ["prime_agent", str(out), "150", "16"])
    assert result.returncode == 0, result.stderr
    cfg_idx = argv.index("@")
    assert argv[cfg_idx + 1].endswith("configs/prime_agent.toml")
    assert "--taskset.max_skill_calls" in argv
    assert argv[argv.index("--taskset.max_skill_calls") + 1] == "150"
    assert "--taskset.trace_dir" in argv
    assert argv[argv.index("--taskset.trace_dir") + 1] == str(out / "turns")
    assert "--args" not in argv


def test_defaults_max_calls_150_and_n_16(tmp_path):
    out = tmp_path / "out"
    result, argv = _run(tmp_path, ["claude_code", str(out)])
    assert result.returncode == 0, result.stderr
    assert argv[argv.index("--num_tasks") + 1] == "16"
    assert argv[argv.index("--taskset.max_skill_calls") + 1] == "150"


def test_missing_eval_binary_fails_loudly(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "control", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        env={**_base_env(), "EVAL_BIN": str(tmp_path / "no-such-binary")},
    )
    assert result.returncode == 2
    assert "eval binary not found" in result.stderr


# -- TOOL_TIER: what the launcher ACTUALLY emits -----------------------------
#
# The previous pin for this was a substring test over the two files' text. It
# could not detect: the human branch emitting `false`, the tier values being
# swapped in the TOML, the harness flag being rerouted to `--taskset.env_args.*`
# (a documented no-op), or the entire TOOL_TIER block being deleted. All four
# mutations passed it. These tests run the launcher and read its argv, so a
# mapping that does not reach the eval CLI cannot pass.

import tomllib

TIERS = REPO / "tools/cli_harness_eval/configs/tool_tiers.toml"


def _tier_cfg():
    return tomllib.load(open(TIERS, "rb"))


def _tier_env():
    """The launcher resolves its Python as EVAL_BIN's sibling `./python`, which
    in production is the venv interpreter. The stub EVAL_BIN has no sibling, so
    PY_BIN would fall back to the system python3 -- 3.10 here, no tomllib. Point
    it at the interpreter running the tests, which is the same venv."""
    import sys
    return {"PY_BIN": sys.executable}


def _pairs(argv):
    """argv as {flag: value} -- every override is emitted as two elements."""
    return {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}


def test_tool_tier_base_emits_the_e10_contract_not_nothing(tmp_path):
    """`TOOL_TIER=base` used to expand to nothing, so a cell claiming the E10
    baseline inherited configs/prime_agent.toml: netplay_true instead of
    np_core, B0 instead of BBOX_MIN, 150 calls, 16 seeds. Wrong on four factors
    while calling itself the baseline."""
    contract = _tier_cfg()["contract"]
    result, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                        {"TOOL_TIER": "base", **_tier_env()})
    assert argv is not None, result.stderr
    got = _pairs(argv)
    assert got["--taskset.env_args.skill_set"] == contract["skill_set"]
    assert got["--taskset.env_args.auto_dismiss"] == contract["auto_dismiss"]
    assert float(got["--taskset.env_args.tune.reveal_map"]) == contract["tune"]["reveal_map"]
    assert got["--taskset.variant"] == contract["variant"]
    assert json.loads(got["--taskset.env_args.explicit_seeds"]) == contract["seeds"]


def test_the_whole_contract_is_emitted_not_half_of_it(tmp_path):
    """model / character / task_spec were declared in the registry but never
    emitted -- they only happened to match configs/prime_agent.toml, so an edit
    to that file would have moved the baseline without touching the registry
    that defines it. Found by adversarial review of the human tier."""
    contract = _tier_cfg()["contract"]
    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                   {"TOOL_TIER": "base", **_tier_env()})
    got = _pairs(argv)
    assert got["--model"] == contract["model"]
    assert got["--taskset.character"] == contract["character"]
    assert got["--taskset.task_spec"] == contract["task_spec"]


def test_tool_tier_base_is_not_a_no_op(tmp_path):
    """The regression that made the tier meaningless: `base` produced argv
    byte-identical to leaving TOOL_TIER unset."""
    _, with_tier = _run(tmp_path, ["prime_agent", str(tmp_path / "a"), "200", "5"],
                        {"TOOL_TIER": "base", **_tier_env()})
    _, without = _run(tmp_path, ["prime_agent", str(tmp_path / "b"), "200", "5"])
    assert with_tier != without


def test_each_tier_emits_its_own_flag_values(tmp_path):
    """Reads the expected values FROM the TOML, so swapping [base] and [human]
    in the registry flips what these assertions require -- the mutation the old
    substring test could not see."""
    cfg = _tier_cfg()
    for tier in ("base", "human"):
        _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / tier), "200", "5"],
                       {"TOOL_TIER": tier, **_tier_env()})
        got = _pairs(argv)
        for flag, value in cfg[tier].items():
            key = ("--harness." if flag == "skill_doc_coords"
                   else "--taskset.env_args.") + flag
            assert key in got, f"{tier}: {flag} never reached the CLI"
            assert json.loads(got[key]) is value, f"{tier}: {flag} emitted {got[key]}"


def test_the_harness_side_flag_is_not_rerouted_to_env_args(tmp_path):
    """`skill_doc_coords` is consumed by the harness process, which never
    imports the env flag registry -- sending it through env_args is a
    documented no-op, i.e. a silently wrong doc."""
    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                   {"TOOL_TIER": "human", **_tier_env()})
    assert "--harness.skill_doc_coords" in argv
    assert "--taskset.env_args.skill_doc_coords" not in argv


def test_the_tier_records_its_own_provenance(tmp_path):
    """A result has to be replayable from its output config.toml alone."""
    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                   {"TOOL_TIER": "base", **_tier_env()})
    got = _pairs(argv)
    assert got["--taskset.env_args.tool_tier"] == "base"
    assert len(got["--taskset.env_args.tool_tier_hash"]) == 16
    assert got["--taskset.env_args.tool_tier_commit"]


def test_tool_tier_refuses_to_be_combined_with_env_args(tmp_path):
    """Both write the same dotted paths and the CLI resolves duplicates
    last-wins silently: ENV_ARGS='{"netplay_telemetry":true}' with TOOL_TIER=base
    produced a cell labelled base running a human-tier fix."""
    result, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                        {"TOOL_TIER": "base", "ENV_ARGS": '{"netplay_telemetry":true}', **_tier_env()})
    assert result.returncode == 2
    assert argv is None, "the eval binary must not have been reached"
    assert "cannot be combined" in result.stderr


def test_tool_tier_is_refused_on_arms_whose_harness_lacks_the_field(tmp_path):
    """HarnessConfig is extra='forbid' and only the prime_agent harness declares
    skill_doc_coords, so this was a raw pydantic ValidationError before."""
    result, _ = _run(tmp_path, ["claude_code", str(tmp_path / "out"), "200", "5"],
                     {"TOOL_TIER": "human", **_tier_env()})
    assert result.returncode == 2
    assert "prime_agent" in result.stderr


def test_a_contradicting_variant_or_budget_is_refused(tmp_path):
    """The contract owns the encoding and the budget; the reference numbers
    describe them. A caller passing different ones is running a different
    experiment."""
    result, _ = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                     {"TOOL_TIER": "base", "VARIANT": "B0", **_tier_env()})
    assert result.returncode == 2 and "contradicts the tier contract" in result.stderr

    result, _ = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "150", "5"],
                     {"TOOL_TIER": "base", **_tier_env()})
    assert result.returncode == 2 and "MAX_CALLS" in result.stderr


def test_an_unknown_tier_is_refused(tmp_path):
    result, _ = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                     {"TOOL_TIER": "continual+human", **_tier_env()})
    assert result.returncode != 0


# -- E13: one store per continual experiment ---------------------------------
#
# Several continual experiments run side by side (same base surface, different
# reflection instructions). Everything that could let two of them be confused on
# disk, or let one start from another's leftovers, is pinned here.


def test_the_continual_store_requires_a_run_id(tmp_path):
    """An unlabelled continual cell cannot be told apart from another
    experiment's once it is on disk."""
    result, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                        {"CONTINUAL_HARNESS": "/tmp/vf-prime-agent/continual-harness/x",
                         **_tier_env()})
    assert result.returncode == 2
    assert argv is None
    assert "CONTINUAL_RUN_ID" in result.stderr


def test_the_run_id_and_prompt_hash_reach_the_config(tmp_path):
    """Two runs that differ only in the orchestrator's reflection prompt are
    otherwise identical on disk, so the prompt hash has to be in the artifact."""
    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                   {"TOOL_TIER": "continual",
                    "CONTINUAL_HARNESS": "/tmp/vf-prime-agent/continual-harness/reflect-terse",
                    "CONTINUAL_RUN_ID": "reflect-terse",
                    "CONTINUAL_PROMPT_SHA": "abc123def456",
                    **_tier_env()})
    got = _pairs(argv)
    assert got["--taskset.env_args.continual_run_id"] == "reflect-terse"
    assert got["--taskset.env_args.continual_prompt_sha"] == "abc123def456"
    assert got["--harness.continual_harness_dir"].endswith("/reflect-terse")
    assert got["--taskset.env_args.tool_tier"] == "continual"


def test_the_continual_tier_starts_from_the_base_surface(tmp_path):
    """The arm's independent variable is the store the agent writes, so its
    floor must be the same floor the denominator uses. If [continual] inherited
    [human], a gain could be the hand-engineered fixes instead."""
    cfg = _tier_cfg()
    assert cfg["continual"] == cfg["base"], "continual must carry base's flags"
    _, cont = _run(tmp_path, ["prime_agent", str(tmp_path / "c"), "200", "5"],
                   {"TOOL_TIER": "continual", **_tier_env()})
    _, base = _run(tmp_path, ["prime_agent", str(tmp_path / "b"), "200", "5"],
                   {"TOOL_TIER": "base", **_tier_env()})
    # Ignore the tier label itself and the per-cell paths, which necessarily
    # differ; everything that decides BEHAVIOUR must match.
    def strip(d):
        return {k: v for k, v in d.items()
                if "tool_tier" not in k and k not in ("--output_dir", "--taskset.trace_dir")}
    assert strip(_pairs(cont)) == strip(_pairs(base)), (
        "a continual cell with no store mounted must be a base cell"
    )


def test_a_continual_store_is_refused_on_arms_that_have_none(tmp_path):
    result, _ = _run(tmp_path, ["claude_code", str(tmp_path / "out"), "200", "5"],
                     {"CONTINUAL_HARNESS": "/tmp/x", "CONTINUAL_RUN_ID": "r",
                      **_tier_env()})
    assert result.returncode == 2 and "prime_agent" in result.stderr


def test_seeds_can_be_overridden_and_the_override_is_recorded(tmp_path):
    """E13's reflection corpus runs seeds 5-9 while evaluation stays on the
    contract's 0-4. The override cannot go through ENV_ARGS (refused alongside
    TOOL_TIER), and a silent replacement of the contract's seeds would make the
    artifact a lie."""
    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                   {"TOOL_TIER": "base", "SEEDS": "[5,6,7,8,9]", **_tier_env()})
    got = _pairs(argv)
    assert json.loads(got["--taskset.env_args.explicit_seeds"]) == [5, 6, 7, 8, 9]
    assert got["--taskset.env_args.seeds_overridden"] == "true"

    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out2"), "200", "5"],
                   {"TOOL_TIER": "base", **_tier_env()})
    assert "--taskset.env_args.seeds_overridden" not in _pairs(argv)


def test_batching_cannot_defeat_the_frozen_baseline_document(tmp_path):
    """The hole adversarial review found: `_skill_doc` hash-checks the FIXTURE,
    then `allow_batching` strips the no-batch rule from the SERVED bytes -- so a
    cell labelled tool_tier=base served 4753 bytes where E10 served 4829, with
    no error. Refused at the launcher, and again in the harness."""
    result, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "5"],
                        {"TOOL_TIER": "base", "ALLOW_BATCHING": "true", **_tier_env()})
    assert result.returncode == 2
    assert argv is None
    assert "ALLOW_BATCHING" in result.stderr

    import nethack_prime_agent as hp
    import pathlib
    import pytest

    pkg = pathlib.Path(hp.__file__).parent / "skill"
    with pytest.raises(RuntimeError, match="frozen E10 baseline document"):
        hp._skill_doc(pkg, skill_doc_coords=False, allow_batching=True)


def test_the_harness_side_contract_factors_are_pinned(tmp_path):
    """max_relaunches is called 'ARM SYMMETRY, load-bearing' in prime_agent.toml
    and is ABSENT from prime_agent_b80.toml, where it defaults to 5 -- so a b80
    cell claiming [base] silently got five extra chances to finish its budget."""
    contract = _tier_cfg()["contract"]
    for arm in ("prime_agent", "prime_agent_b80"):
        _, argv = _run(tmp_path, [arm, str(tmp_path / arm), "200", "5"],
                       {"TOOL_TIER": "base", **_tier_env()})
        got = _pairs(argv)
        assert json.loads(got["--harness.max_relaunches"]) == contract["max_relaunches"]
        assert json.loads(got["--harness.allow_batching"]) == contract["allow_batching"]
        assert json.loads(got["--taskset.max_parallel_skill_calls"]) == \
            contract["max_parallel_skill_calls"]


def test_a_seed_count_that_contradicts_the_contract_is_refused(tmp_path):
    """`MAX_CALLS` got a contradiction check and `N` did not, so `... 200 1`
    produced a 1-seed cell labelled base."""
    result, _ = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "200", "1"],
                     {"TOOL_TIER": "base", **_tier_env()})
    assert result.returncode == 2 and "seeds" in result.stderr

    # A preflight mock play is allowed to be short, and says so in the artifact.
    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "pf"), "3", "1"],
                   {"TOOL_TIER": "base", "TIER_SHORT_BUDGET": "1", **_tier_env()})
    assert _pairs(argv)["--taskset.env_args.tier_short_budget"] == "true"


# --------------------------------------------------------------------------- #
# E16_ARGS: the whitelisted knob, and the multi-line value that broke it
# --------------------------------------------------------------------------- #

#: What an E16 attempt actually sends. `ledger_text` is the rendered archive
#: table plus the checkpoint's lessons plus its quoted conversation prefix, so
#: it is MULTI-LINE by construction -- there is no E16 attempt whose ledger is
#: one line.
_E16_LEDGER = (
    "CHECKPOINT ARCHIVE (2 saved state(s); 1 on the Dlvl/XL/score frontier).\n"
    "  id  Dlvl  XL   HP     turn   score  attempts  name / why it was saved\n"
    "->   2     1   1  13/16      4      0         0  fountain room\n"
    "\n"
    "THE PLAN THAT WAS LIVE WHEN THIS STATE WAS SAVED (quoted from that "
    "session, not your own memory):\n"
    "(assistant) Plan: the east corridor is a dead end.\n"
    "(user) You move west. There is a fountain here."
)


def test_e16_args_survives_a_multi_line_ledger_as_one_argv_item(tmp_path):
    """THE BUG THAT WOULD HAVE KILLED THE RUN, as a regression test.

    `mapfile -t` over newline-separated records splits ONE multi-line value into
    one array element per LINE. Every line after the first then reaches the eval
    CLI as a bare positional argument, and the run dies at boot with
    "Unrecognized arguments: id Dlvl XL HP turn score ...".

    Nothing upstream caught it. The 4-attempt no-inference dry run uses a stub
    player and never goes through this script; the model-in-the-loop sims passed
    only a single-line `directive`. The first real resumed rollout hit it
    immediately, which is to say: EVERY E16 attempt would have.
    """
    e16 = json.dumps({
        "resume_checkpoint": str(tmp_path / "archive" / "c2"),
        "checkpoint_archive": str(tmp_path / "archive"),
        "wiki_dir": str(tmp_path / "wiki"),
        "ledger_text": _E16_LEDGER,
        "fidelity_log": str(tmp_path / "fid.jsonl"),
        "directive": "Follow the plan quoted from the earlier session.",
    })
    result, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "12", "1"],
                        {"TOOL_TIER": "e16_gewiki", "SEEDS": "[1]",
                         "TIER_SHORT_BUDGET": "1", "E16_ARGS": e16,
                         **_tier_env()})
    assert result.returncode == 0, result.stderr[-2000:]
    got = _pairs(argv)
    # ONE argv item, byte-identical, newlines and all.
    assert got["--taskset.env_args.ledger_text"] == _E16_LEDGER
    assert "\n" in got["--taskset.env_args.ledger_text"]
    # And no fragment of it leaked out as a bare positional.
    for stray in ("id", "Dlvl", "(user)", "fountain"):
        assert stray not in argv, (
            f"{stray!r} reached the eval CLI as a bare positional argument; "
            f"the multi-line value was split")
    assert got["--taskset.env_args.directive"] == \
        "Follow the plan quoted from the earlier session."


def test_e16_args_still_refuses_a_key_no_tier_could_ever_write(tmp_path):
    """The whitelist is the reason this knob is allowed next to TOOL_TIER."""
    result, _ = _run(tmp_path, ["prime_agent", str(tmp_path / "out"), "12", "1"],
                     {"TOOL_TIER": "e16_gewiki", "SEEDS": "[1]",
                      "TIER_SHORT_BUDGET": "1",
                      "E16_ARGS": json.dumps({"skill_set": "full"}),
                      **_tier_env()})
    assert result.returncode == 2
    assert "not allowed" in result.stderr


def test_e16_args_carries_the_reseed_pair(tmp_path):
    """`--reseed` reaches the player as a scalar, and is ABSENT by default.

    Absent matters as much as present: an attempt that was not asked to reseed
    must be handed no reseed argument at all, rather than a zero pair, which
    would be a third stochastic semantics no provenance field describes.
    """
    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "on"), "12", "1"],
                   {"TOOL_TIER": "e16_gewiki", "SEEDS": "[1]",
                    "TIER_SHORT_BUDGET": "1",
                    "E16_ARGS": json.dumps({"reseed": "[424242, 99]"}),
                    **_tier_env()})
    assert _pairs(argv)["--taskset.env_args.reseed"] == "[424242, 99]"

    _, argv = _run(tmp_path, ["prime_agent", str(tmp_path / "off"), "12", "1"],
                   {"TOOL_TIER": "e16_gewiki", "SEEDS": "[1]",
                    "TIER_SHORT_BUDGET": "1",
                    "E16_ARGS": json.dumps({"directive": "descend"}),
                    **_tier_env()})
    assert "--taskset.env_args.reseed" not in _pairs(argv)
