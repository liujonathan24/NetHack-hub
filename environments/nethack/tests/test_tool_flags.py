"""The post-baseline tool fixes must be reconstructible, both ways.

The E10 baseline is the denominator of every claim in this series. It was
measured before PR #29, so unless those fixes can be switched OFF, "run the
baseline" silently means "run the baseline plus undated improvements" and the
published reference numbers describe nothing reproducible.

These tests pin the contract: default OFF, string-coerced, and each flag
actually changing the behaviour it names.
"""
from __future__ import annotations

import pathlib

import pytest

from nethack_harness import tool_flags


@pytest.fixture(autouse=True)
def _clean_flags():
    tool_flags.reset()
    yield
    tool_flags.reset()


def test_every_flag_defaults_off_so_base_is_the_default():
    snap = tool_flags.snapshot()
    assert snap == {
        "netplay_telemetry": False,
        "melee_hints": False,
        "skill_doc_coords": False,
    }, "a flag defaulting ON would make the baseline unreachable from config"


def test_env_args_strings_coerce_the_way_the_other_knobs_do():
    """env_args arrive as strings through the CLI, so "false" must not read as
    True the way a non-empty string otherwise would -- the bug `describe_args`
    and `reflect` both carry explicit coercion for."""
    tool_flags.configure(netplay_telemetry="false", melee_hints="0")
    assert tool_flags.enabled("netplay_telemetry") is False
    assert tool_flags.enabled("melee_hints") is False

    tool_flags.configure(netplay_telemetry="true", melee_hints="1")
    assert tool_flags.enabled("netplay_telemetry") is True
    assert tool_flags.enabled("melee_hints") is True

    for falsey in ("", "off", "no", "False", None):
        tool_flags.configure(netplay_telemetry=falsey)
        assert tool_flags.enabled("netplay_telemetry") is False, falsey


def test_unrelated_kwargs_are_ignored():
    """`configure(**kwargs)` is handed the env's whole kwarg bag."""
    tool_flags.configure(skill_set="np_core", max_turns=200, tune={"reveal_map": 1.0})
    assert tool_flags.snapshot() == {
        "netplay_telemetry": False, "melee_hints": False, "skill_doc_coords": False,
    }


def test_melee_report_is_the_bare_coordinate_when_the_flag_is_off():
    """The stale-target report (c1a0bec) is the whole point of `melee_hints`."""
    from netplay.nethack_agent.skills import _melee_target_report

    off = _melee_target_report(agent=None, target_glyph=42, tx=3, ty=4)
    assert off == "Unable to reach the target at (3, 4)."
    assert "matching monster" not in off


def test_melee_report_names_the_new_position_when_the_flag_is_on():
    from netplay.nethack_agent.skills import _melee_target_report

    class _Pos:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Level:
        def get_monsters(self):
            return [(42, _Pos(9, 9)), (42, _Pos(5, 5)), (7, _Pos(3, 4))]

    class _Agent:
        current_level = _Level()

    tool_flags.configure(melee_hints=True)
    on = _melee_target_report(_Agent(), 42, 3, 4)
    # Nearest same-glyph monster by Manhattan distance, and never a claimed
    # identity -- glyph values cannot distinguish two of the same species.
    assert "(5, 5)" in on
    assert "a matching monster" in on


def test_the_added_keys_are_unavailable_until_the_flag_is_on():
    """`-` is the one that matters: it is "your fingers" in the engrave prompt,
    i.e. the only path to dust-engraving Elbereth. The baseline could not press
    it, so a baseline cell must not be able to either."""
    from netplay.nethack_utils.nle_wrapper import RawKeyPress

    with pytest.raises(ValueError, match="Cannot parse the given key"):
        RawKeyPress.parse("-")

    tool_flags.configure(netplay_telemetry=True)
    assert RawKeyPress.parse("-") == RawKeyPress.KEYPRESS_MINUS


def test_the_gate_covers_every_key_the_commit_added():
    from netplay.nethack_utils.nle_wrapper import RawKeyPress, _POST_BASELINE_KEYS

    for key in "-']{}|~":
        with pytest.raises(ValueError):
            RawKeyPress.parse(key)
    assert len(_POST_BASELINE_KEYS) == 6 + 1

    # Keys that predate the baseline must stay available with everything off.
    for key in "abz>%`":
        assert RawKeyPress.parse(key) == RawKeyPress(ord(key))


def test_keys_that_predate_the_baseline_are_never_gated():
    from netplay.nethack_utils.nle_wrapper import RawKeyPress

    assert RawKeyPress.parse("enter") == RawKeyPress.KEYPRESS_ENTER
    assert RawKeyPress.parse("esc") == RawKeyPress.KEYPRESS_ESC


# --------------------------------------------------------------------------- #
# ea8cc15: the nearest-monster hint on the no-monster-here path (same flag)    #
# --------------------------------------------------------------------------- #
def _drive_no_monster_failure():
    """Run melee_attack against an empty tile and return the failure text."""
    from netplay.nethack_agent.skills import melee_attack

    class _Pos:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Level:
        def get_monster_glyph(self, x, y):
            return None                      # nothing at the target tile

        def get_monsters(self):
            return [(42, _Pos(10, 12)), (7, _Pos(2, 2))]

    class _Agent:
        current_level = _Level()

        def waiting_for_popup(self):
            return False          # @fail_on_popup gate

    steps = list(melee_attack(_Agent(), 3, 4))
    assert len(steps) == 1 and "fail" in str(steps[0].status).lower()
    return steps[0].thoughts


def test_no_monster_failure_is_bare_when_the_flag_is_off():
    """ea8cc15 shares `melee_hints` with the stale-target report: one
    behaviour split across two commits (see tool_flags._DEFAULTS)."""
    text = _drive_no_monster_failure()
    assert text == "There is no monster at (3,4)."


def test_no_monster_failure_names_the_nearest_monster_when_on():
    tool_flags.configure(melee_hints=True)
    text = _drive_no_monster_failure()
    # nearest by Manhattan distance to the requested tile, either glyph --
    # there is no target glyph to match on this path.
    assert "Nearest visible monster is at (2, 2)." in text


# --------------------------------------------------------------------------- #
# aee5c43 / skill_doc_coords: the SKILL.md coordinate-frame note               #
# --------------------------------------------------------------------------- #
def test_skill_doc_serves_the_baseline_wording_when_off():
    """OFF must be byte-exact baseline: the note is swapped back, not deleted."""
    import importlib
    hp = importlib.import_module("nethack_prime_agent")

    shipped = (importlib.resources.files("nethack_prime_agent") / "skill" / "SKILL.md").read_bytes()
    assert hp._COORD_FRAME_NOTE in shipped.decode()   # ON-state is the file itself

    off = hp._restore_baseline_coord_note(shipped).decode()
    assert hp._COORD_FRAME_NOTE not in off
    assert hp._COORD_FRAME_BASELINE in off


def test_skill_doc_strip_fails_loudly_when_the_note_drifts():
    import importlib
    hp = importlib.import_module("nethack_prime_agent")

    with pytest.raises(RuntimeError, match="stale"):
        hp._restore_baseline_coord_note(b"# NetHack\nsome other doc\n")


def test_the_launcher_mapping_is_pinned_elsewhere_and_why():
    """The launcher no longer duplicates the tier mapping, so there is nothing
    here to keep in sync.

    The old test grepped both files for flag NAMES. It passed when the human
    branch emitted `false`, when [base]/[human] were swapped in the TOML, when
    the harness flag was rerouted to a documented no-op path, and when the
    entire TOOL_TIER block was deleted -- four mutations, four passes. The
    mapping now lives only in tools/cli_harness_eval/tool_tiers.py, which
    launch_cell.sh calls, and it is pinned by tests that RUN the launcher and
    read its argv: see tests/test_launch_cell.py::test_each_tier_emits_its_own_
    flag_values and friends.
    """
    launcher = (
        pathlib.Path(__file__).resolve().parents[3]
        / "tools/cli_harness_eval/launch_cell.sh"
    ).read_text()
    # Prose is fine (the refusal messages name the flags); an EMISSION is the
    # duplication. Strip comments and echoes, then look for the flag names.
    code = "\n".join(
        line for line in launcher.splitlines()
        if not line.lstrip().startswith("#") and "echo " not in line
    )
    for flag in ("netplay_telemetry", "melee_hints", "skill_doc_coords"):
        assert flag not in code, (
            f"{flag} is emitted by the launcher again -- the duplication that "
            "the substring pin could not police. Keep the mapping in the registry."
        )

def test_the_baseline_skill_doc_is_byte_identical_to_what_e10_served():
    """The load-bearing claim. Previously nothing pinned it: the assertion was
    `_COORD_FRAME_BASELINE in served`, trivially true for any value of that
    constant, and a substring check rather than a byte comparison. Now the
    baseline doc is a frozen FILE and this is its hash -- so an edit anywhere in
    it, not just in the coordinate note, fails."""
    import hashlib
    import pathlib

    import nethack_prime_agent as hp

    pkg = pathlib.Path(hp.__file__).parent / "skill"
    served = hp._skill_doc(pkg, skill_doc_coords=False, allow_batching=False)
    # `git show aee5c43^:harnesses/.../skill/SKILL.md` -- the exact bytes every
    # rollout in outputs/e10_baseline/ was served.
    assert hashlib.sha256(served).hexdigest() == (
        "39f34ad07961a27cb440ced0ff53ec3001df6172b2813dfb33e7d344af3d10aa"
    )


def test_the_human_tier_serves_the_annotated_doc():
    import pathlib

    import nethack_prime_agent as hp

    pkg = pathlib.Path(hp.__file__).parent / "skill"
    on = hp._skill_doc(pkg, skill_doc_coords=True, allow_batching=False)
    off = hp._skill_doc(pkg, skill_doc_coords=False, allow_batching=False)
    assert on != off
    assert b"Do\nNOT count rows from the raw terminal screen" in on
    assert b"NOT count rows from the raw terminal screen" not in off


def test_editing_the_frozen_baseline_doc_fails_the_run(tmp_path):
    """The fixture IS the definition of [base]; a silent edit would redefine
    every baseline cell."""
    import pathlib
    import shutil

    import nethack_prime_agent as hp
    import pytest

    real = pathlib.Path(hp.__file__).parent / "skill"
    fake = tmp_path / "skill"
    shutil.copytree(real, fake)
    (fake / "SKILL.baseline.md").write_bytes(b"# not the baseline\n")
    with pytest.raises(RuntimeError, match="has been edited"):
        hp._skill_doc(fake, skill_doc_coords=False, allow_batching=False)


def test_the_harness_flag_defaults_off():
    """Neutering check: flipping this default made [base] cells serve the
    annotated doc, and every existing test still passed."""
    from nethack_prime_agent import PrimeAgentHarnessConfig

    assert PrimeAgentHarnessConfig(id="x").skill_doc_coords is False


def test_allow_batching_still_works_on_both_tier_documents():
    """`_NO_BATCH_RULE` went stale when the honesty pass rewrote SKILL.md, so
    allow_batching=True raised RuntimeError -- the feature was dead on both
    documents and only a red test recorded it."""
    import pathlib

    import nethack_prime_agent as hp

    pkg = pathlib.Path(hp.__file__).parent / "skill"
    for coords in (False, True):
        plain = hp._skill_doc(pkg, skill_doc_coords=coords, allow_batching=False)
        batched = hp._skill_doc(pkg, skill_doc_coords=coords, allow_batching=True)
        assert hp._NO_BATCH_RULE in plain.decode()
        assert hp._NO_BATCH_RULE not in batched.decode()
        assert len(batched) < len(plain)


def _lost_track_agent():
    """Minimal agent that drives melee_attack to the `not found` branch: the
    target is present at (x,y), one reachable neighbour exists, the step lands
    on it, and the monster is then gone from (x,y) and every neighbour."""
    class _Blstats:
        x, y = 4, 4

    class _Level:
        def __init__(self):
            self.calls = 0

        def get_monster_glyph(self, x, y):
            # Present for the initial lookup, gone from every later query.
            self.calls += 1
            return 42 if self.calls == 1 else None

        def get_neighbors(self, x, y):
            return [(4, 4)]

        def get_monsters(self):
            return []

    class _Agent:
        blstats = _Blstats()

        def __init__(self):
            self.current_level = _Level()

        def get_path_to(self, x, y, **kw):
            return [(x, y)]

        def distance_to(self, x, y, **kw):
            return 0

        def step(self, action, **kw):
            from netplay.core.agent_base import Step, StepStatus, ThoughtType
            # `running` has no classmethod factory; construct it directly so the
            # driver keeps pulling the generator.
            return Step(status=StepStatus.running, thoughts="stepped",
                        thought_type=ThoughtType.System, step_data=None)

        def waiting_for_popup(self):
            # melee_attack is wrapped in @fail_on_popup, which probes this
            # before and after every yielded step.
            return False

    return _Agent()


def _drive_lost_track(monkeypatch):
    """Run melee_attack to the `not found` branch and return the last Step.

    `get_move_towards_action` is stubbed to a non-WAIT sentinel so the skill
    takes the "walk toward the target" path rather than the adjacent-kill path;
    real pathfinding is not what these two tests are about.
    """
    from netplay.nethack_agent import skills

    monkeypatch.setattr(skills, "get_move_towards_action",
                        lambda *a, **k: "MOVE_SENTINEL")
    agent = _lost_track_agent()
    last = None
    for step in skills.melee_attack(agent, 5, 4):
        last = step
    return last


def test_lost_track_message_is_bare_when_melee_hints_is_off(monkeypatch):
    """The one ungated hunk in c1a0bec. `_melee_target_report` self-gates its
    BODY, and this path concatenated the result unconditionally -- so [base]
    emitted `Lost track of the target. Unable to reach the target at (x, y).`
    where the baseline emitted `Lost track of the target`, including a
    coordinate the baseline never printed here. Drives the real generator, so
    replacing the gate with `if True` fails this."""
    step = _drive_lost_track(monkeypatch)
    assert step is not None
    assert "Lost track of the target" in str(step.thoughts)
    assert "Unable to reach" not in str(step.thoughts), (
        "the stale-target suffix leaked into a [base] cell"
    )


def test_lost_track_message_carries_the_report_when_melee_hints_is_on(monkeypatch):
    tool_flags.configure(melee_hints=True)
    step = _drive_lost_track(monkeypatch)
    assert "Lost track of the target." in str(step.thoughts)
    assert "Unable to reach" in str(step.thoughts)

def test_load_environment_wires_env_args_into_the_flag_registry():
    """Structural pin: replacing `configure(**kwargs)` with `pass` left all 694
    tests passing, i.e. nothing proved a cell could turn a flag on at all.

    A full `load_environment` call needs the engine, so this asserts the call
    survives in the source of the function that receives the cell's env_args.
    Crude, but it fails on the exact neuter that nothing else caught."""
    import inspect

    import nethack

    src = inspect.getsource(nethack.load_environment)
    assert "tool_flags" in src, "load_environment no longer imports the flag registry"
    assert "configure(**kwargs)" in src, (
        "load_environment no longer passes the cell's env_args to the flag "
        "registry -- every flag would be stuck at its default"
    )


# -- no model-facing string may disclose the call budget ---------------------


def test_no_served_prompt_or_skill_doc_mentions_the_call_budget():
    """The agent is meant to be playing NetHack, not playing a budgeted eval.

    Two strings were shaping play. `SKILL.md` told it "There is a hard budget of
    skill calls... Spend calls on progress, not probing", which is strategic
    instruction to economise; the prompt tail told it the episode ends when "you
    run out of calls". Downstream evidence that this mattered: a reflection pass
    over one game wrote `Budget is limited - spend on descent, not looting` into
    the continual store as a LEARNED LESSON, where it would have been served to
    every later player.

    This asserts on the served text, not on source comments -- explaining why the
    budget is hidden is fine; telling the model about it is not.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3]
    banned = re.compile(r"budget|run out of calls|call limit|spend calls", re.I)

    docs = [
        root / "harnesses/nethack-prime-agent/nethack_prime_agent/skill/SKILL.md",
        root / "harnesses/nethack-prime-agent/nethack_prime_agent/skill/SKILL.baseline.md",
    ]
    for doc in docs:
        offending = [ln for ln in doc.read_text().splitlines() if banned.search(ln)]
        assert not offending, f"{doc.name} discloses the call budget: {offending}"

    # The prompt tails, taken from the module rather than re-typed here.
    from nethack_harness.prompt import rendering

    for name in ("_PROMPT_TAIL", "_PROMPT_TAIL_MINIMAL"):
        text = getattr(rendering, name)
        assert not banned.search(text), f"{name} discloses the call budget: {text!r}"
        # The clause this replaced exists for a reason -- without it 15-25% of
        # rollouts ended early with the model declaring itself finished. Keep the
        # anti-give-up half.
        # Normalise: the clause is line-wrapped differently in the two tails.
        assert "never because you stopped" in " ".join(text.split()), name


def test_the_end_of_episode_message_does_not_state_the_budget_size():
    """The only place the NUMBER ever reached a model. Post-hoc, so it cannot
    shape play within a game -- but a reflection pass reads it out of the trace
    and turns it into advice."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3]
    src = (root / "environments/nethack/nethack_v1.py").read_text()
    i = src.index("budget_exhausted = True")
    window = src[i : i + 600]
    returned = [ln for ln in window.splitlines() if '"[' in ln or 'f"[' in ln]
    assert returned, "could not find the returned end-of-episode message"
    for ln in returned:
        assert "{budget}" not in ln and "skill calls used" not in ln, ln


def test_tool_discovery_is_withheld_not_merely_discouraged():
    """SKILL.md is the authoritative reference and the server's JSON schemas are
    empty, so a `list_tools()` round-trip returns LESS than the document the
    agent already has, while costing a turn.

    Telling the model not to call it left the call available and put the idea in
    its head. This asserts the call is gone and the docs no longer name it --
    while the instance's internal discovery, which is how `await
    nethack.<tool>()` resolves at all, still works.
    """
    import pathlib
    import re
    import sys
    import types

    root = pathlib.Path(__file__).resolve().parents[3]
    pkg = root / "harnesses/nethack-prime-agent/nethack_prime_agent/skill/src"

    # Stub `rlm`: the real one lives in Prime Agent's kernel venv.
    stub = types.ModuleType("rlm")

    class _McpIntegration:
        def __getattr__(self, n):
            return lambda *a, **k: n

        def list_tools(self):
            return ["np_move_to", "search"]

    stub.McpIntegration = _McpIntegration
    sys.modules.setdefault("rlm", stub)
    sys.path.insert(0, str(pkg))
    try:
        sys.modules.pop("nethack", None)
        import nethack

        with pytest.raises(AttributeError, match="not available"):
            nethack.list_tools

        # The game must still work: tools resolve, and the instance still
        # discovers internally.
        assert callable(nethack.np_move_to)
        assert nethack.nethack.list_tools() == ["np_move_to", "search"]

        # `help()` is a Python builtin we cannot remove, but the module must not
        # advertise the withheld call through it -- pydoc filters on __all__,
        # and exporting the class re-surfaced `list_tools` as inherited.
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            help(nethack)
        assert "list_tools" not in buf.getvalue()
    finally:
        sys.path.remove(str(pkg))
        sys.modules.pop("nethack", None)

    # And neither document names either call.
    banned = re.compile(r"list_tools|help\(\)", re.I)
    for name in ("SKILL.md", "SKILL.baseline.md"):
        doc = root / "harnesses/nethack-prime-agent/nethack_prime_agent/skill" / name
        hits = [ln for ln in doc.read_text().splitlines() if banned.search(ln)]
        assert not hits, f"{name} still names a withheld call: {hits}"
