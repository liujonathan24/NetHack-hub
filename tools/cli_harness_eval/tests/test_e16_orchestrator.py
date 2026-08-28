"""E16 step 3 gate: the orchestrator loop, its selector, and its bookkeeping.

NO INFERENCE IS SPENT HERE, and that is a design property rather than a
limitation of the tests. The player is injected (``launcher(ctx) ->
PlayerResult``) and the orchestrator's LM session is injected behind a
``runner`` seam, so the whole loop -- select, restore, play, save, ingest,
select again -- runs against a real engine and a real archive with scripted
stand-ins at exactly two points: the model that plays, and the model that
chooses.

The end-to-end test is the one that matters. It seeds an archive, runs four
attempts through the real ``Orchestrator``, and then asserts on what is ON
DISK: the checkpoints, their meta, the lessons, the attempt records, and the
summary. A loop that only passes because its mocks agree with it is the failure
mode this suite is written against.
"""
import json
import os
import random
import time
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))
sys.path.insert(0, str(REPO / "environments" / "nethack"))

import e16_orchestrator as E  # noqa: E402
import e16_session as S  # noqa: E402
from nethack_harness.checkpoints import (  # noqa: E402
    checkpoint_list, checkpoint_meta, checkpoint_restore, checkpoint_save,
)

WIKI_SRC = Path("/root/nld/e15-wiki/configs/continual/wiki")

#: The archive the interrupted mechcheck attempt left behind: 24 real
#: checkpoints from ONE trajectory, D1 -> D8, XL 1 -> 4. Read-only, and every
#: test that uses it copies it first. It is the evidence for the defect this
#: section is about: the Pareto frontier over (Dlvl, XL, score) collapses to
#: exactly ONE of those 24, because a single trajectory rises on all three
#: objectives at once.
REAL_ARCHIVE = Path("/root/nld/e16_runs/mechcheck/archive")
needs_real_archive = pytest.mark.skipif(
    not (REAL_ARCHIVE / "c24" / "meta.json").is_file(),
    reason="the mechcheck 24-checkpoint archive is not on this box")


def copy_real_archive(dest: Path) -> Path:
    """A WRITABLE copy of the real archive. The original is never touched."""
    import shutil
    shutil.copytree(REAL_ARCHIVE, dest)
    return dest


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

def mkrow(ident, dlvl=1, xl=1, score=0, attempts=0, balrog_min=0.0,
          name="", note="", created_by="save", dungeon=0):
    return E.Row(id=str(ident), path=Path(f"/nowhere/c{ident}"), dlvl=dlvl,
                 xl=xl, score=score, attempts_from=attempts,
                 balrog_min=balrog_min, balrog=balrog_min, name=name,
                 note=note, created_by=created_by, dungeon_number=dungeon,
                 hp=10, max_hp=10, gameturn=100)


def cfg_for(tmp_path, **kw):
    kw.setdefault("selector", "scripted")
    return E.OrchestratorConfig(run_dir=Path(tmp_path), wiki_src=WIKI_SRC, **kw)


# --------------------------------------------------------------------------- #
# 1. selector determinism
# --------------------------------------------------------------------------- #

def test_selector_is_deterministic_and_records_the_probabilities_it_used():
    rows = [mkrow(1, dlvl=1, score=0, balrog_min=0.001),
            mkrow(2, dlvl=5, score=50, balrog_min=0.012, attempts=3),
            mkrow(3, dlvl=5, score=90, balrog_min=0.012),
            mkrow(4, dlvl=3, score=10, balrog_min=0.005)]
    cfg = E.OrchestratorConfig(run_dir=Path("/tmp/x"))

    a = E.select(rows, cfg, random.Random(7))
    b = E.select(rows, cfg, random.Random(7))
    assert a.chosen_id == b.chosen_id
    assert a.draw == b.draw
    assert [c["prob"] for c in a.candidates] == [c["prob"] for c in b.candidates]

    # A DIFFERENT seed is allowed to differ; what must not differ is the
    # distribution, which is a function of the ledger alone.
    c = E.select(rows, cfg, random.Random(99))
    assert [x["prob"] for x in a.candidates] == [x["prob"] for x in c.candidates]

    # The record is complete enough to reproduce the draw by hand.
    assert abs(sum(x["prob"] for x in a.candidates) - 1.0) < 1e-9
    got = a.to_json()
    for key in ("chosen_id", "candidates", "weights", "temperature",
                "rng_seed", "draw", "n_rows"):
        assert key in got
    for cand in got["candidates"]:
        for key in ("id", "norm_balrog_min", "novelty", "norm_attempts_from",
                    "weighted_score", "prob"):
            assert key in cand


def test_selector_only_considers_the_pareto_frontier():
    # c1 is dominated by c2 on all three objectives, so it must not be a
    # candidate at all -- not merely improbable.
    rows = [mkrow(1, dlvl=1, xl=1, score=0), mkrow(2, dlvl=5, xl=4, score=90)]
    sel = E.select(rows, E.OrchestratorConfig(run_dir=Path("/tmp/x")),
                   random.Random(0))
    assert [c["id"] for c in sel.candidates] == ["2"]
    assert sel.chosen_id == "2"


def test_attempts_from_discount_moves_probability_away_from_a_reused_state():
    rows = [mkrow(1, dlvl=5, xl=3, score=10, attempts=0),
            mkrow(2, dlvl=5, xl=3, score=10, attempts=9)]
    cfg = E.OrchestratorConfig(run_dir=Path("/tmp/x"))
    sel = E.select(rows, cfg, random.Random(0))
    p = {c["id"]: c["prob"] for c in sel.candidates}
    assert p["1"] > p["2"], "the state tried nine times must be discounted"


# --------------------------------------------------------------------------- #
# 2. budget
# --------------------------------------------------------------------------- #

def test_budget_refuses_at_the_ceiling_and_keeps_two_lines():
    b = E.Budget(ceiling_usd=100.0, min_headroom_usd=5.0)
    b.add_player(60.0)
    b.add_orchestrator(20.0)
    assert b.spent_usd == 80.0
    assert b.can_launch()
    b.check()  # does not raise

    b.add_orchestrator(16.0)          # 96 spent, 4 left, below headroom
    assert not b.can_launch()
    with pytest.raises(E.BudgetExceeded) as exc:
        b.check()
    # The refusal names BOTH lines: "we are out of money" and "the orchestrator
    # ate a fifth of it" are different findings.
    assert "players $60.00" in str(exc.value)
    assert "orchestrator $36.00" in str(exc.value)
    lines = b.lines()
    assert lines["player_usd"] == 60.0 and lines["orchestrator_usd"] == 36.0
    assert lines["total_usd"] == 96.0


def test_orchestrator_stops_at_the_ceiling_without_launching(tmp_path):
    cfg = cfg_for(tmp_path, budget_ceiling_usd=10.0, min_headroom_usd=5.0)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)
    launched = []

    def never(ctx):
        launched.append(ctx)
        return E.PlayerResult(died=True)

    orch = E.Orchestrator(cfg, never)
    orch.budget.add_player(8.0)          # 2 left, headroom is 5
    orch.run(max_attempts=3)
    assert launched == [], "a launch happened past the ceiling"
    assert orch.stop_reason == E.STOP_BUDGET


# --------------------------------------------------------------------------- #
# 3. ledger rendering: 0 / 1 / many
# --------------------------------------------------------------------------- #

def test_ledger_renders_empty_one_and_many():
    empty = E.render_ledger([])
    assert "empty" in empty.lower()

    one = E.render_ledger([mkrow(1, dlvl=3, xl=2, score=40, name="entrance",
                                 note="fresh game")])
    assert "entrance" in one and "fresh game" in one and " 3 " in one

    # MANY: every id is still a choice. The old rendering showed the Pareto
    # frontier plus named saves and capped the result; these 30 rows are a
    # totally ordered chain, so that cap used to be a filter with one survivor.
    rows = [mkrow(i, dlvl=i, xl=i, score=i * 10, name=f"n{i}",
                  created_by="save") for i in range(1, 31)]
    many, cands = E.build_ledger(rows)
    assert "measured by the harness" in many
    for r in rows:
        assert f"c{r.id} " in many or f"c{r.id}\n" in many, \
            f"c{r.id} is not choosable from the rendered ledger"
    assert {c["id"] for c in cands} == {r.id for r in rows}
    assert "not listed" not in many


def test_a_large_archive_stays_legible_without_hiding_any_state():
    """Past the detail budget states go COMPACT, never missing.

    "Group or paginate, but never make a state unchoosable" -- so the overflow
    still carries id, branch, depth, XL, HP, turn, score and kind; what it
    drops is the free text.
    """
    rows = [mkrow(i, dlvl=1 + i % 12, xl=1 + i % 5, score=i, name=f"n{i}",
                  created_by="auto", note=f"automatic checkpoint: tick {i}")
            for i in range(1, 201)]
    cfg = E.OrchestratorConfig(run_dir=Path("/tmp/x"), ledger_max_rows=25)
    text, cands = E.build_ledger(rows, cfg)
    assert len(cands) == 200
    assert {c["shown_as"] for c in cands} == {"detail", "compact"}
    assert sum(1 for c in cands if c["shown_as"] == "detail") <= 25 * 1
    for r in rows:
        assert f"c{r.id} " in text, f"c{r.id} vanished from a 200-row ledger"
    assert "EQUALLY CHOOSABLE" in text


def test_ledger_shows_the_directive_next_to_the_outcome():
    # Point 4 of the directive contract: the orchestrator's feedback about its
    # OWN instructions is the pairing, not the outcome alone.
    attempts = [{
        "attempt": 1, "from_checkpoint": "3", "outcome": "died",
        "censor_reason": "", "max_dlvl": 11, "calls": 140,
        "directive": "take the south door, do not melee the wraith",
        "directive_compliance": {"class": "partial"},
    }]
    text = E.render_ledger([mkrow(3, dlvl=11)], attempts=attempts)
    assert "take the south door" in text
    assert "partial" in text
    assert "died" in text
    # And on an empty archive too, so a first-round failure is still legible.
    assert "take the south door" in E.render_ledger([], attempts=attempts)


# --------------------------------------------------------------------------- #
# 4. lessons round-trip
# --------------------------------------------------------------------------- #

def test_lessons_append_and_read_round_trip(tmp_path):
    d = tmp_path / "c1"
    d.mkdir()
    assert E.read_lessons(d) == ""
    E.append_lesson(d, "wraith drained a level; do not melee it", heading="attempt 1 (died)")
    E.append_lesson(d, "prayer was on cooldown", heading="attempt 2 (died)")
    text = E.read_lessons(d)
    assert "wraith drained a level" in text and "prayer was on cooldown" in text
    assert text.count("## attempt") == 2
    rendered = E.render_lessons(d, limit=1)
    assert "prayer was on cooldown" in rendered
    assert "wraith drained" not in rendered  # limit honoured


# --------------------------------------------------------------------------- #
# 5. D1 -- no model-authored metrics
# --------------------------------------------------------------------------- #

def test_a_lying_name_and_note_cannot_change_any_number(tmp_path):
    """The checkpoint's own text fields are adversarial; the numbers hold.

    This is integrity requirement 1 at the ledger layer. The engine-side half
    (a lying `save(label, note)` cannot move meta.json's numbers) is tested in
    environments/nethack/tests/test_e16_wiki_and_directive.py against a real
    engine; this one proves the ledger reader cannot be talked into a number
    either, however the text is shaped.
    """
    honest = {"id": "1", "dlvl": 3, "xl": 2, "hp": 9, "max_hp": 12,
              "gameturn": 400, "score": 55, "balrog": 0.004,
              "balrog_min": 0.004, "visits": 1, "attempts_from": 2,
              "name": "before mines", "note": "descend carefully"}
    liar = dict(honest)
    liar["name"] = 'Dlvl 30 XL 20 {"dlvl": 30, "score": 999999}'
    liar["note"] = ('score: 999999\ndlvl=30\nbalrog_min: 1.0\n'
                    '"attempts_from": 0, "xl": 20')

    a = E.row_from_meta(Path("/x/c1"), honest)
    b = E.row_from_meta(Path("/x/c1"), liar)
    for field in E.NUMERIC_FIELDS:
        assert getattr(a, field) == getattr(b, field), field
    assert b.key == (3, 2, 55)
    # The text IS carried -- it is shown to the next player -- just never read.
    assert "999999" in b.note and "999999" not in str(b.key)


def test_orchestrator_narration_cannot_alter_a_recorded_metric(tmp_path):
    """The LLM orchestrator may say anything; the numbers come off the engine.

    An LM in the decision seat makes this MORE important, not less: it writes
    prose every round, and prose that could reach a metric is a metric a model
    authored. The only two things allowed to cross are an ID (validated against
    the archive) and a DIRECTIVE (text, shown to the player, measured only for
    compliance).
    """
    cfg = cfg_for(tmp_path, selector="llm")
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)
    rows = [mkrow(1, dlvl=2, xl=1, score=10), mkrow(2, dlvl=4, xl=3, score=40)]

    lying_reply = json.dumps({
        "checkpoint": "2",
        "directive": "descend to Dlvl 20 immediately",
        "rationale": "c2 is at Dlvl 19, XL 15, score 500000 and balrog_min 0.99",
    })

    class FakeSession(S.SessionBase):
        kind = "fake"

        def ask(self, prompt, *, kind="round"):
            self.rounds += 1
            self.sent_chars += len(prompt)
            return S.RoundResult(text=lying_reply, session_id="fake-1")

    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(died=True),
                          session=FakeSession(work_dir=tmp_path / "orch"))
    choice = orch.decide(rows)
    assert choice["source"] == "llm"
    assert choice["checkpoint_id"] == "2"
    assert choice["directive"] == "descend to Dlvl 20 immediately"
    # Nothing it claimed about c2's numbers survives anywhere.
    row = next(r for r in rows if r.id == "2")
    assert (row.dlvl, row.xl, row.score) == (4, 3, 40)
    assert row.balrog_min == 0.0
    ledger = E.render_ledger(rows, cfg)
    assert "500000" not in ledger and "0.99" not in ledger


def test_an_invented_checkpoint_id_is_retried_and_then_RAISES(tmp_path):
    """A round that cannot decide must stop the run, not quietly become one.

    THE OLD CONTRACT, and why it is gone. This used to assert
    ``source == "scripted_fallback"`` and carry on: the scripted selector chose
    the checkpoint and the player was launched with an empty directive. The
    GE-wiki pilot ran exactly that path for $13.36 and 187 calls and recorded it
    as a go_explore attempt; what it actually measured was the no-directive
    control. The fallback is still COMPUTED and still recorded -- it is what the
    ablation would have done -- but it no longer gets to launch a player.
    """
    cfg = cfg_for(tmp_path, selector="llm", directive_retries=2)
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    rows = [mkrow(1, dlvl=2, xl=1, score=10)]

    class FakeSession(S.SessionBase):
        kind = "fake"

        def __init__(self, **kw):
            super().__init__(**kw)
            self.asked = []

        def ask(self, prompt, *, kind="round"):
            self.asked.append(kind)
            return S.RoundResult(text='{"checkpoint": "c99", "directive": "go"}')

    sess = FakeSession(work_dir=tmp_path / "orch")
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(), session=sess)
    with pytest.raises(E.DirectiveExtractionFailed) as exc:
        orch.decide(rows)
    assert "not in the archive" in str(exc.value)
    # BOUNDED: the first round plus exactly `directive_retries` retries.
    assert sess.asked == ["round1", "round1_retry1", "round1_retry2"]
    assert orch.llm_fallbacks == 1
    failed = json.loads(
        (cfg.orchestrator_dir / "failed_decision_attempt1.json").read_text())
    assert len(failed["attempts"]) == 3
    assert all(a["valid"] is False for a in failed["attempts"])
    # The raw bytes of every attempt are on disk, because the reason a reply
    # could not be parsed is only answerable from the reply.
    raw = cfg.orchestrator_dir / "raw"
    assert (raw / "decide_a1_attempt1.reply.txt").is_file()
    assert (raw / "decide_a1_attempt3.reply.txt").is_file()


def test_a_bad_first_round_is_recovered_by_the_bounded_retry(tmp_path):
    """The retry exists so a recoverable round does not cost the run."""
    cfg = cfg_for(tmp_path, selector="llm", directive_retries=2)
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    rows = [mkrow(1, dlvl=2, xl=1, score=10)]

    class FlakySession(S.SessionBase):
        kind = "fake"

        def __init__(self, **kw):
            super().__init__(**kw)
            self.n = 0

        def ask(self, prompt, *, kind="round"):
            self.n += 1
            if self.n == 1:
                return S.RoundResult(text="I think c1 looks promising.\n")
            return S.RoundResult(text=(
                'Choosing c1.\n{"checkpoint": "1", "directive": "descend to '
                'D3 and save at each new depth", "rationale": "lineage"}'))

    sess = FlakySession(work_dir=tmp_path / "orch")
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(), session=sess)
    choice = orch.decide(rows)
    assert choice["source"] == "llm"
    assert choice["checkpoint_id"] == "1"
    assert "descend to D3" in choice["directive"]
    assert sess.n == 2
    assert orch.llm_fallbacks == 0
    assert [a["valid"] for a in choice["decision"]["attempts"]] == [False, True]


def test_an_empty_directive_can_never_be_served_as_a_treatment(tmp_path):
    """Defence in depth: the launch path refuses too, not only `decide`."""
    cfg = cfg_for(tmp_path, selector="llm")
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult())
    with pytest.raises(E.DirectiveExtractionFailed) as exc:
        orch._launch_one([], "1", None, "", {}, pair_id=None,
                         pair_role=E.ROLE_SOLO)
    assert "empty directive" in str(exc.value)


def test_parse_decision_accepts_prose_around_json_and_rejects_prose_alone():
    ok = E.__dict__ and S.parse_decision(
        'I think c3 is best because it is healthy.\n'
        '{"checkpoint": "c3", "directive": "clear the east rooms", '
        '"rationale": "healthy and unexplored"}', ["1", "3"])
    assert ok.valid and ok.checkpoint_id == "3"
    assert ok.directive == "clear the east rooms"

    bare = S.parse_decision("Let us resume checkpoint c1 and be careful.", ["1"])
    assert bare.valid and bare.checkpoint_id == "1"

    nothing = S.parse_decision("I am not sure which one to pick.", ["1"])
    assert not nothing.valid and nothing.fallback_reason

    empty = S.parse_decision("", ["1"])
    assert not empty.valid and "empty" in empty.fallback_reason


# --------------------------------------------------------------------------- #
# 6. censoring
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("result,expected,reason", [
    (E.PlayerResult(died=True, stop_condition="died"), E.OUTCOME_DIED, ""),
    (E.PlayerResult(ascended=True), E.OUTCOME_ASCENDED, ""),
    (E.PlayerResult(stop_condition="harness_timeout"),
     E.OUTCOME_CENSORED, E.CENSOR_WALL_CLOCK),
    (E.PlayerResult(stop_condition="agent_completed"),
     E.OUTCOME_CENSORED, E.CENSOR_EMPTY_COMPLETION),
    (E.PlayerResult(stop_condition="error"),
     E.OUTCOME_CENSORED, E.CENSOR_HARNESS_ERROR),
    (E.PlayerResult(stop_condition="ProviderError"),
     E.OUTCOME_CENSORED, E.CENSOR_HARNESS_ERROR),
    (E.PlayerResult(stop_condition="max_turns_reached"),
     E.OUTCOME_CENSORED, E.CENSOR_BUDGET_STOP),
    (E.PlayerResult(error="CheckpointIntegrityError: dungeon has holes"),
     E.OUTCOME_CENSORED, E.CENSOR_INTEGRITY),
    (E.PlayerResult(stop_condition=""), E.OUTCOME_CENSORED, E.CENSOR_UNKNOWN),
])
def test_censoring_is_classified_and_never_collapsed_into_died(result, expected, reason):
    got, got_reason = E.classify_outcome(result)
    assert (got, got_reason) == (expected, reason)


def test_a_timed_out_but_deep_attempt_is_censored_not_a_death():
    """E15's exact bug: a live game the wall clock stopped, counted as a death.

    A death is a completed observation of how far a life got. A censoring is a
    LOWER BOUND on a life that was still going. Pooling them biases the death
    rate up and the depth distribution down at the same time.
    """
    r = E.PlayerResult(died=False, stop_condition="harness_timeout", max_dlvl=11)
    outcome, reason = E.classify_outcome(r)
    assert outcome == E.OUTCOME_CENSORED and reason == E.CENSOR_WALL_CLOCK
    assert outcome != E.OUTCOME_DIED


# --------------------------------------------------------------------------- #
# 7. directive compliance
# --------------------------------------------------------------------------- #

def test_directive_compliance_rubric():
    calls_south = [{"name": "np_move_to", "args": {"x": 40, "y": 12,
                                                   "why": "south door"}},
                   {"name": "np_press_key", "args": {"key": "j"}}]
    got = E.classify_directive_compliance("take the south door", calls_south, "died")
    assert got["class"] == E.COMPLY_FOLLOWED
    assert "south" in got["matched"]

    got = E.classify_directive_compliance(
        "take the south door", [{"name": "np_move_to", "args": {"x": 1, "y": 1}}] * 12,
        "died")
    assert got["class"] == E.COMPLY_IGNORED

    # A prohibition that was broken is `violated`, NOT `ignored`: the
    # instruction was reachable and was overridden, which is a different fact.
    got = E.classify_directive_compliance(
        "do not melee the wraith",
        [{"name": "np_melee_attack", "args": {"target": "wraith"}}], "died")
    assert got["class"] == E.COMPLY_VIOLATED
    assert "wraith" in got["violated"]

    # Respected prohibition, inside the window.
    got = E.classify_directive_compliance(
        "do not melee the wraith",
        [{"name": "np_move_to", "args": {"x": 3, "y": 3}}], "died")
    assert got["class"] == E.COMPLY_FOLLOWED

    # Died before it could act.
    got = E.classify_directive_compliance("clear the east rooms", [], "died")
    assert got["class"] == E.COMPLY_PREVENTED

    # The control mode.
    got = E.classify_directive_compliance("", [{"name": "search"}], "died")
    assert got["class"] == E.COMPLY_NONE

    # The rubric names itself in its own output, so a reader never mistakes it
    # for a semantic judgement.
    assert got["rubric"] == E.RUBRIC_NAME
    assert "per-clause" in got["rubric"]


def test_compliance_scores_prohibition_and_goal_separately(tmp_path):
    """The finding that forced the per-clause structure, as a regression test.

    A directive that mixes a prohibition with an aspiration, scored on ONE
    enum, collapses to `prevented` when the attempt runs out of calls -- and
    that erases the fact that the prohibition was fully and checkably obeyed.
    A prohibition is scoreable at any budget; a goal is not.
    """
    directive = ("Do NOT descend this attempt. Explore this level and reach "
                 "XL 3 before taking any staircase.")
    calls = [{"name": "np_melee_attack", "args": {"x": 27, "y": 8}}] * 7
    got = E.classify_directive_compliance(
        directive, calls, E.OUTCOME_CENSORED,
        metrics={"descent_count": 0, "budget_exhausted": True, "max_xp_level": 1})

    # The half with evidence is not erased by the half without.
    assert got["by_clause_kind"][E.KIND_PROHIBITION]["label"] == E.COMPLY_FOLLOWED
    assert got["by_clause_kind"][E.KIND_GOAL]["label"] != E.COMPLY_FOLLOWED
    # The headline is DERIVED from those two, and both survive in the record.
    assert got["headline"] == got["class"]
    assert got["clauses"], "per-clause verdicts must always be kept"
    assert any(c["basis"] == "metric" for c in got["clauses"]), (
        "a descent clause must be scored against the harness metric, not "
        "against whether the model typed a matching word")
    assert "prohibition clauses ->" in got["rationale"]


def test_a_zero_call_attempt_still_records_per_clause_verdicts():
    got = E.classify_directive_compliance("clear the east rooms", [], "died")
    assert got["class"] == E.COMPLY_PREVENTED
    # Never `followed`: a fast death must not look like obedience.
    assert got["class"] != E.COMPLY_FOLLOWED
    assert got["by_clause_kind"], (
        "a zero-call attempt is the one case that would otherwise have no "
        "auditable rubric output at all")
    assert all(c["result"] == E.CLAUSE_NO_EVIDENCE for c in got["clauses"])


def test_tool_prohibitions_are_linted_as_unscoreable():
    """The sims' instrumental-action case: 'do not explore' cannot be scored.

    The player explored SEVEN times in order to reach the staircase the same
    directive told it to take. A checker that stays off the model's prose
    cannot tell that from disobedience, so the fix is upstream -- prohibit
    outcomes, not tools -- and the lint is what makes that reachable.
    """
    warns = E.lint_directive(
        "Descend as fast as possible. Do not explore, do not fight anything.")
    codes = [w["code"] for w in warns]
    assert "tool_prohibition" in codes
    hit = next(w for w in warns if w["code"] == "tool_prohibition")
    assert "explore" in hit["tokens"]
    assert "UNSCOREABLE" in hit["message"]
    assert "do not clear the level" in hit["message"], (
        "the warning must name the better phrasing, or it is a complaint "
        "rather than a fix")

    # An OUTCOME prohibition is not warned about.
    assert not [w for w in E.lint_directive(
        "Reach the down staircase. Do not clear the level.")
        if w["code"] == "tool_prohibition"]

    # A prohibition-only directive is satisfied by dying on turn 2.
    assert "no_goal_clause" in [
        w["code"] for w in E.lint_directive("do not fight anything")]


def test_instrumental_ambiguity_is_recorded_in_the_rationale():
    """A tool-prohibition broken while the goal was unmet AND being pursued."""
    got = E.classify_directive_compliance(
        "Descend as fast as possible. Do not explore, do not fight anything.",
        [{"name": "np_move_to", "args": {"x": 8, "y": 4, "why": "descend"}}]
        + [{"name": "np_explore_level", "args": {}}] * 7,
        E.OUTCOME_CENSORED,
        metrics={"descent_count": 0, "budget_exhausted": True})
    assert got["instrumental_ambiguity"] is True
    assert "INSTRUMENTAL-ACTION AMBIGUITY" in got["rationale"]
    assert "MAY UNDERSTATE compliance" in got["rationale"]
    assert "prohibit outcomes, not tools" in got["rationale"]


def test_directive_kind_separates_a_prohibition_from_its_own_verb():
    """`descend` and `no_descend` share every content token.

    Only the clause kind separates them, and getting this wrong would pair a
    "descend now" directive with a "do not descend" control.
    """
    assert "no_descend" in E.classify_directive_kind("do not descend this level")
    assert "descend" == E.classify_directive_kind("descend to D10 immediately")
    assert E.classify_directive_kind("") == "none"


# --------------------------------------------------------------------------- #
# 8. the orchestrator's session mechanism (no inference: injected runner)
# --------------------------------------------------------------------------- #

#: A plausible opening plan. Long and varied enough to pass
#: `detect_degeneration` -- which is the point: a stub that hands the
#: orchestrator two sentences is testing a run that would now, correctly, be
#: refused for having no strategy.
_STUB_PLAN = """\
Round 1 plan for this seed.

WHAT THE WIKI SAYS ABOUT THE EARLY GAME. Descend steadily; do not clear levels
for their own sake. Flee at half HP rather than trading blows. Never melee a
floating eye. Keep the inventory unburdened.

THE HAZARD LADDER I EXPECT. Shallow levels are cheap to re-reach, so the
archive should be dense early and sparse later. The first real wall is the
mid-game gap where a single bad fight ends a lineage that took many attempts to
build.

HOW I WILL SPEND ATTEMPTS. Prefer the deepest checkpoint whose HP is healthy
over the deepest checkpoint outright: resuming into a fight already lost wastes
the whole attempt. Re-select a state that has been tried twice without an
advance only if nothing shallower is promising.

WHAT I WILL WATCH. Whether directives that prohibit an outcome are followed
more often than directives that prohibit a tool, and whether depth gained per
attempt falls off as the archive deepens.
"""


def _stub_stdout(argv, cwd, body, *, session_id="sess-1", usage=None):
    """prime-agent stdout for `body`, IN THE MODE `argv` ASKED FOR.

    Plain `--print` writes assistant text and nothing else; `--mode json`
    writes the session header and records. Every stub in this file goes through
    here, because a stub that answers json on a text round is what let the
    pilot's directive-extraction bug reach production green.
    """
    if "--mode" not in argv or argv[argv.index("--mode") + 1] != "json":
        return body + "\n"
    rec = {"type": "message", "message": {"role": "assistant", "content": body}}
    if usage:
        rec["usage"] = usage
    return "\n".join([
        json.dumps({"type": "session", "version": 3, "id": session_id,
                    "cwd": cwd}),
        json.dumps(rec)]) + "\n"


def _fake_prime_agent(session_id="019f-aaaa", reply="hello"):
    """Stands in for `prime-agent --print`, IN WHICHEVER MODE IT WAS ASKED FOR.

    THE UNFAITHFULNESS THIS FIXES, because it is the reason nothing in this
    suite caught the E16 pilot's central failure. This stub used to emit
    ``--mode json`` records on EVERY round, including the resumed rounds the
    session deliberately runs as plain text. So every test drove the json
    record parser, and the text path -- the one every round after the first
    actually takes -- was never exercised by anything but production.

    Plain ``--print`` writes assistant text and nothing else
    (``dist/modes/print-mode.js:80-95``): no session header, no usage, no
    records. That is what this returns when ``--mode json`` is absent.
    """
    seen = []

    def runner(argv, env, cwd, timeout_s):
        seen.append({"argv": list(argv), "env": dict(env), "cwd": cwd})
        if "--mode" not in argv or argv[argv.index("--mode") + 1] != "json":
            return reply + "\n", "", 0
        out = [json.dumps({"type": "session", "version": 3, "id": session_id,
                           "cwd": cwd}),
               json.dumps({"type": "message", "id": "aa", "parentId": None,
                           "message": {"role": "assistant", "content": reply},
                           "usage": {"prompt_tokens": 1000,
                                     "completion_tokens": 100,
                                     "cached_input_tokens": 500}})]
        return "\n".join(out) + "\n", "", 0
    return runner, seen


def test_session_argv_omits_no_session_and_resumes_by_id_from_round_two(tmp_path):
    runner, seen = _fake_prime_agent()
    sess = S.PrimeAgentSession(work_dir=tmp_path / "orch",
                               agent_dir=tmp_path / "agent", runner=runner)
    r1 = sess.ask("round one")
    r2 = sess.ask("round two")

    a1, a2 = seen[0]["argv"], seen[1]["argv"]
    # THE flag that separates the orchestrator from a player rollout. It is
    # checked first in createSessionManager and returns an in-memory session
    # unconditionally, so its presence would silently discard every --resume.
    assert "--no-session" not in a1 and "--no-session" not in a2
    # json mode on the DISCOVERY round only. It is the only way to learn a
    # session id, and it was measured HANGING when combined with --resume, so
    # it runs exactly once -- on the round that has nothing to resume.
    assert a1[a1.index("--mode") + 1] == "json"
    assert "--mode" not in a2, (
        "json mode + --resume was measured hanging where the identical text "
        "--print call succeeded 30s later; resumed rounds must be text")
    assert "--resume" not in a1, "round 1 creates the session"
    assert a2[a2.index("--resume") + 1] == "019f-aaaa"
    # Never a BARE --resume: with no argument it opens the interactive picker.
    assert a2.index("--resume") + 1 < len(a2)
    assert a2[a2.index("--resume") + 1].startswith("019f")
    # The provider is pinned, not inferred from the model pattern.
    assert "--provider" in a1
    # `--` before the prompt, so a prompt starting with a dash is not a flag.
    assert a1[-2] == "--" and a1[-1] == "round one"

    # A DEDICATED agent-state dir, under the correct (package-derived) name.
    assert seen[0]["env"][S.ENV_AGENT_DIR] == str(tmp_path / "agent")
    assert "PRIME_AGENT_DIR" not in [k for k in seen[0]["env"] if k == "PRIME_AGENT_DIR"]
    # A FIXED cwd, so --resume resolves as "local" instead of asking to fork.
    assert seen[0]["cwd"] == seen[1]["cwd"] == str(tmp_path / "orch")

    assert r1.session_id == r2.session_id == "019f-aaaa"
    assert not r2.continuity_broken
    # ONLY THE DISCOVERY ROUND CARRIES USAGE IN STDOUT. Text `--print` writes
    # assistant text and nothing else, so a resumed round's usage comes from
    # the SESSION FILE (`_scan_session_files`), never from stdout. A stub that
    # returned json records on a text round would report 2000 here and would be
    # lying about where the number came from.
    assert sess.usage_available and sess.usage_total["prompt_tokens"] == 1000
    # The reply still arrives on the text round -- verbatim, unparsed.
    assert r1.text == r2.text == "hello"
    assert sess.spend_usd > 0


def test_session_refuses_to_resume_from_a_foreign_cwd(tmp_path):
    """HANG 1. The guard must raise BEFORE any subprocess exists.

    prime-agent resolves a `--resume` whose cwd does not match the session's
    recorded one as a foreign session and asks an interactive "Fork this
    session into current directory?" confirm. A non-interactive driver never
    answers it, so the run hangs forever with no output and no error -- which
    on this budget is the whole run. So this is asserted at the point where it
    costs nothing: no launch at all.
    """
    runner, seen = _fake_prime_agent()
    sess = S.PrimeAgentSession(work_dir=tmp_path / "orch",
                               agent_dir=tmp_path / "agent", runner=runner)
    sess.ask("round one")
    assert sess.session_cwd, "round 1 must record the session's cwd"

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    sess.work_dir = elsewhere
    with pytest.raises(S.SessionCwdMismatch) as exc:
        sess.ask("round two")
    assert len(seen) == 1, "the guard must fire before the subprocess is created"
    assert "hangs a non-interactive driver" in str(exc.value)
    assert str(sess.session_cwd) in str(exc.value), (
        "the error must name the directory to re-run from")


def test_a_hanging_orchestrator_call_becomes_an_error_at_its_deadline(tmp_path):
    """HANGS 1-3, the backstop. A blocked call must die, loudly and countably.

    A timeout is a round lost to the scripted fallback. A hang is the run. The
    whole point of the hardening is to convert every instance of the second
    into the first, so `timed_out` is its own field rather than a substring of
    an error message.
    """
    def runner(argv, env, cwd, timeout_s):
        raise S.RoundTimeout(
            f"orchestrator call exceeded its {timeout_s:g}s deadline and was "
            f"killed (process group 1234)")

    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               runner=runner, timeout_s=30.0)
    res = sess.ask("hello")
    assert res.timed_out is True
    assert res.exit_code == -1
    assert "RoundTimeout" in res.error and "30s deadline" in res.error
    assert sess.timeouts == 1
    assert sess.budget_lines()["timeouts"] == 1
    # And it is a recorded round, not a swallowed one.
    rec = json.loads(sess.log_path.read_text().splitlines()[-1])
    assert rec["timed_out"] is True


def test_the_orchestrator_gets_a_private_tmpdir_and_never_exports_it(tmp_path):
    """HANG 2. 900s on the shared daemon socket against 3.5s with a private one.

    The second assertion is the one that protects the PLAYERS: an exported
    TMPDIR leaks into sandboxed rollouts where the path is not bound, and
    pointing it at a bind-mounted directory collapses concurrent rollouts back
    onto one daemon socket.
    """
    before = os.environ.get("TMPDIR")
    runner, seen = _fake_prime_agent()
    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               runner=runner)
    sess.ask("hi")
    got = seen[0]["env"]["TMPDIR"]
    assert Path(got).is_dir()
    assert got not in ("/tmp", "/var/tmp")
    assert not got.startswith("/tmp/prime-agent"), (
        "this is the shared socket that was measured hanging for 900s")
    assert os.environ.get("TMPDIR") == before, (
        "TMPDIR must be set on the orchestrator's own env dict only")


def test_text_mode_rounds_recover_id_usage_and_cost_from_the_session_file(tmp_path):
    """HANG 3's replacement mechanism, proven rather than assumed.

    Dropping json mode on resumed rounds only works if everything it printed is
    recoverable elsewhere. The session file carries strictly more: the header
    id AND the cwd AND the provider's own per-message cost.
    """
    agent = tmp_path / "agent"
    sessions = agent / S.SESSIONS_SUBDIR
    sessions.mkdir(parents=True)
    work = tmp_path / "orch"
    work.mkdir()
    path = sessions / "0000-file-stem-differs.jsonl"
    header_id = "01a047da-c7a0-757e-851c-de1bf8c7f9d5"
    path.write_text(json.dumps({"type": "session", "version": 3,
                                "id": header_id,
                                "cwd": str(work.resolve())}) + "\n")

    def runner(argv, env, cwd, timeout_s):
        # Text mode: stdout carries the reply and NOTHING else. Everything the
        # driver needs is appended to the session file, as the real one does.
        with open(path, "a") as fh:
            fh.write(json.dumps({
                "type": "message",
                "message": {"role": "assistant", "content": "ok",
                            "usage": {"input": 15000, "output": 200,
                                      "cacheRead": 256, "cacheWrite": 0,
                                      "cost": {"total": 0.025242}}}}) + "\n")
        return "ok\n", "", 0

    sess = S.PrimeAgentSession(work_dir=work, agent_dir=agent, runner=runner,
                               json_mode_policy=S.JSON_MODE_NEVER)
    res = sess.ask("round")
    assert res.session_id == header_id
    # THE ID IS NOT THE FILENAME STEM. Discovering it by "newest file in
    # sessions/" is the trap this module documents and must not fall into.
    assert Path(res.session_file).stem != header_id
    assert res.session_cwd == str(work.resolve())
    assert res.usage["prompt_tokens"] == 15000
    assert res.usage["cached_input_tokens"] == 256
    assert res.cost_usd_reported == pytest.approx(0.025242)
    assert sess.usage_available is True


def test_session_detects_a_silent_fork(tmp_path):
    """If the header id changes, the conversation forked and continuity is a
    false claim about the run. It is recorded, not smoothed over."""
    ids = iter(["id-A", "id-B"])

    def runner(argv, env, cwd, timeout_s):
        return _stub_stdout(argv, cwd, "ok", session_id=next(ids)), "", 0

    # json mode on BOTH rounds, so the stdout header is the id source under
    # test. Under the default `discovery_only` policy a resumed round is plain
    # text and carries no header at all -- the fork would then be caught by the
    # SESSION FILE scan instead, which is a different mechanism with its own
    # test and needs real files on disk to exercise.
    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               runner=runner,
                               json_mode_policy=S.JSON_MODE_ALWAYS)
    sess.ask("one")
    r2 = sess.ask("two")
    assert r2.continuity_broken
    assert sess.continuity_breaks == 1
    assert sess.session_ids == ["id-A", "id-B"]


def test_session_reports_usage_as_unavailable_rather_than_zero(tmp_path):
    def runner(argv, env, cwd, timeout_s):
        return (json.dumps({"type": "session", "id": "x"}) + "\n"
                + json.dumps({"type": "message",
                              "message": {"role": "assistant", "content": "hi"}})
                + "\n"), "", 0

    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               runner=runner)
    r = sess.ask("hi")
    assert r.usage == {} and r.spend_usd == 0.0
    assert sess.usage_available is False, (
        "an unknown orchestrator spend must be reported as unknown, never as $0")


def test_player_summaries_entering_the_orchestrator_are_capped(tmp_path):
    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               runner=lambda *a: ("", "", 0), summary_cap=100)
    long = "x" * 5000
    capped = sess.cap_summary(long)
    assert len(capped) <= 100
    assert capped.endswith("[...truncated]")
    assert sess.cap_summary("short") == "short"


def test_compaction_triggers_on_the_context_bound(tmp_path):
    runner, _ = _fake_prime_agent(reply="handoff text")
    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               runner=runner, context_chars=50)
    assert not sess.needs_compaction()
    sess.ask("x" * 60)
    assert sess.needs_compaction()

    cfg = cfg_for(tmp_path, selector="llm")
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(), session=sess)
    summary = orch.maybe_compact()
    assert summary == "handoff text"
    assert sess.compactions == 1
    assert (cfg.orchestrator_dir / "handoff_1.txt").read_text() == "handoff text"
    # A fresh session is started, and the chain is kept.
    assert len(sess.session_ids) >= 1


FIXTURES = HERE.parent / "fixtures"


def test_THE_PILOT_DIRECTIVE_survives_a_text_mode_round(tmp_path):
    """THE REGRESSION, on the pilot's own bytes.

    `fixtures/pilot_round1_print_stdout.txt` is the assistant text of the E16
    GE-wiki pilot's round-1 turn, copied verbatim out of the orchestrator's
    prime-agent session file
    (`gewiki_pilot/orchestrator/agent/sessions/01a04847-35f7-70b8-b336-5496c599c569.jsonl`,
    record 49). Plain `--print` writes exactly that text to stdout and nothing
    else, so this IS what the driver read.

    What the driver then did with it: handed it to the json-mode RECORD parser,
    which walks stdout line by line and treats any line that parses as a JSON
    object as a protocol record. The reply's last line is the decision object
    the round prompt demanded ("EXACTLY this JSON object on its own line"). It
    parsed, matched no record shape, and was dropped. The recorded reply was
    678 of 1,104 chars -- the rationale, minus the decision -- and attempt 1
    launched with "(orchestrator produced no directive for this attempt)".
    """
    raw = (FIXTURES / "pilot_round1_print_stdout.txt").read_text()

    # 1. The bytes really do carry a well-formed decision.
    assert '"checkpoint": "1"' in raw and '"directive"' in raw

    # 2. THE OLD PATH, reproduced exactly: the json record parser eats it.
    _h, eaten, _u = S.parse_json_mode_stdout(raw)
    assert len(eaten) == 678, "the pilot recorded 678 chars; fixture drifted"
    assert not S.parse_decision(eaten, ["1"]).valid

    # 3. THE FIX: a text-mode round is parsed as text.
    _h, text, _u = S.parse_round_stdout(raw, json_mode=False)
    assert text == raw.strip()
    dec = S.parse_decision(text, ["1"])
    assert dec.valid and dec.checkpoint_id == "1"
    assert dec.directive.startswith("Take the down-stairs on each level")
    assert "HP is below half of its maximum" in dec.directive

    # 4. END TO END through a real session and a real `decide`: the directive
    #    the pilot's orchestrator actually wrote reaches the player context.
    cfg = cfg_for(tmp_path / "run", selector="llm")
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)

    def runner(argv, env, cwd, timeout_s):
        # Round 1 is the discovery round (json); the decision round is text --
        # which is the mode the bug lived in.
        body = _STUB_PLAN if "--mode" in argv else raw.strip()
        return _stub_stdout(argv, cwd, body), "", 0

    sess = S.PrimeAgentSession(work_dir=cfg.orchestrator_dir,
                               agent_dir=cfg.orchestrator_dir / "agent",
                               log_path=cfg.orchestrator_log, runner=runner)
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(), session=sess)
    orch.open_discussion()
    choice = orch.decide([mkrow(1, dlvl=1, xl=1)])
    assert choice["source"] == "llm"
    assert choice["checkpoint_id"] == "1"
    assert choice["directive"].startswith("Take the down-stairs on each level")
    assert orch.llm_fallbacks == 0
    # And the classifier can see what KIND of directive it is, which the
    # pilot's "(orchestrator produced no directive…)" placeholder could not be.
    assert E.classify_directive_kind(choice["directive"]) != "none"


def test_json_mode_stream_is_not_multiplied_by_its_own_deltas():
    """The pilot's 11,478-char opening plan came back as 2,656,190 chars.

    `--mode json` re-emits the WHOLE message on every text delta
    (`message_update`, docs/json.md). The old parser appended the text of every
    record, so a message streamed in N chunks was concatenated N times as
    cumulative prefixes. The run read the result -- 589 unique lines in 18,088
    -- as the model stuck in a repetition loop, threw the opening plan away,
    and had no strategy on the record.

    The TEXT here is the pilot's real assistant output, read out of its session
    file; only the event framing is reconstructed, from docs/json.md.
    """
    parts = json.loads((FIXTURES / "pilot_opening_assistant_text.json").read_text())
    plan = parts[-1]
    assert len(plan) == 11478

    lines = [json.dumps({"type": "session", "version": 3, "id": "s", "cwd": "/w"}),
             json.dumps({"type": "message_start",
                         "message": {"role": "assistant", "content": []}})]
    # 60 cumulative prefixes, the same shape the provider streamed.
    for i in range(1, 61):
        cut = max(1, len(plan) * i // 60)
        lines.append(json.dumps({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta"},
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": plan[:cut]}]}}))
    final = {"role": "assistant",
             "content": [{"type": "text", "text": plan}],
             "usage": {"input_tokens": 24662, "output_tokens": 2576}}
    lines.append(json.dumps({"type": "message_end", "message": final}))
    lines.append(json.dumps({"type": "turn_end", "message": final,
                             "toolResults": []}))
    lines.append(json.dumps({"type": "agent_end", "messages": [final]}))
    stdout = "\n".join(lines) + "\n"
    assert len(stdout) > 20 * len(plan), "fixture must reproduce the blow-up"

    _h, text, usage = S.parse_json_mode_stdout(stdout)
    assert text == plan, "the reply must be the plan, once"
    # Usage is counted once too: `message_update` repeats it as well.
    assert usage["prompt_tokens"] == 24662 and usage["completion_tokens"] == 2576

    # The plan that comes out is usable; the 2.6 MB that used to come out is not.
    assert E.detect_degeneration(text)["degenerate"] is False


def test_degeneration_detector_flags_the_pilots_recorded_opening_plan():
    """`gewiki_pilot/orchestrator/opening_plan.txt` as the run wrote it.

    Rebuilt here from the same plan rather than vendored, because the file is
    2.6 MB. Both sides of the threshold are the pilot's own numbers: the
    recorded plan scored 0.033 unique lines, the real one 0.959.
    """
    parts = json.loads((FIXTURES / "pilot_opening_assistant_text.json").read_text())
    plan = parts[-1]
    recorded = "\n".join(plan[:max(1, len(plan) * i // 533)]
                         for i in range(1, 534))

    bad = E.detect_degeneration(recorded)
    assert bad["degenerate"] and "repetition loop" in bad["reason"]
    assert bad["unique_line_ratio"] < 0.1

    good = E.detect_degeneration(plan)
    assert not good["degenerate"]
    assert good["unique_line_ratio"] > 0.9

    # A short reply is not a plan, whatever it says.
    assert E.detect_degeneration("Plan: descend.")["degenerate"]
    # ...but the same reply IS a fine decision round, which has no length floor.
    assert not E.detect_degeneration("Plan: descend.", min_chars=0)["degenerate"]


def test_an_unusable_opening_plan_is_retried_and_then_RAISES(tmp_path):
    """No strategy on the record is a hard error, not a quiet start."""
    cfg = cfg_for(tmp_path / "run", selector="llm", opening_retries=1)
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    looped = "\n".join(["Now I have a complete picture. Let me synthesize "
                        "everything."] * 400)
    asked = []

    def runner(argv, env, cwd, timeout_s):
        asked.append(argv[-1])
        return _stub_stdout(argv, cwd, looped), "", 0

    sess = S.PrimeAgentSession(work_dir=cfg.orchestrator_dir,
                               agent_dir=cfg.orchestrator_dir / "agent",
                               log_path=cfg.orchestrator_log, runner=runner)
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(), session=sess)
    with pytest.raises(E.OpeningPlanUnusable):
        orch.open_discussion()

    # BOUNDED: the opening plus exactly `opening_retries` retries, and the
    # retry told the model what was wrong with the last one.
    assert len(asked) == 2 and "not usable" in asked[1]
    # THE RAW REPLY IS KEPT EITHER WAY -- both attempts, before the verdict.
    raw = cfg.orchestrator_dir / "raw"
    assert (raw / "opening_attempt1.stdout.txt").is_file()
    assert (raw / "opening_attempt2.stdout.txt").is_file()
    verdicts = json.loads(
        (cfg.orchestrator_dir / "opening_degeneration.json").read_text())
    assert len(verdicts) == 2
    assert all(v["degeneration"]["degenerate"] for v in verdicts)


def test_a_failed_opening_stops_the_run_and_says_so_in_the_summary(tmp_path):
    """The exception escapes, but not before the run directory records why."""
    cfg = cfg_for(tmp_path / "run", selector="llm", opening_retries=0,
                  budget_ceiling_usd=1000.0, milestone_dlvl=99,
                  milestone_dungeon=-1)
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)

    def runner(argv, env, cwd, timeout_s):
        return _stub_stdout(argv, cwd, "ok"), "", 0

    sess = S.PrimeAgentSession(work_dir=cfg.orchestrator_dir,
                               agent_dir=cfg.orchestrator_dir / "agent",
                               log_path=cfg.orchestrator_log, runner=runner)
    launched = []
    orch = E.Orchestrator(cfg, lambda ctx: launched.append(ctx), session=sess)
    orch.prepare()
    E.seed_archive(cfg)
    with pytest.raises(E.OpeningPlanUnusable):
        orch.run(max_attempts=2)
    assert launched == [], "no player may be launched without an opening plan"
    summary = json.loads(cfg.summary_path.read_text())
    assert summary["stop_reason"] == E.STOP_ORCHESTRATOR_FAILED
    assert "OpeningPlanUnusable" in summary["orchestrator_error"]


def test_round_log_carries_both_lengths(tmp_path):
    """`reply` is a parser's output; the raw byte count is the check on it."""
    raw = (FIXTURES / "pilot_round1_print_stdout.txt").read_text()

    def runner(argv, env, cwd, timeout_s):
        return _stub_stdout(argv, cwd, raw.strip()), "", 0

    log = tmp_path / "rounds.jsonl"
    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               log_path=log, runner=runner)
    sess.ask("discovery")   # json mode
    sess.ask("decision")    # text mode
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(recs) == 2
    text_round = recs[1]
    assert text_round["json_mode"] is False
    assert text_round["reply_chars"] == len(raw.strip())
    assert text_round["raw_stdout_chars"] == len(raw.strip()) + 1  # trailing \n


def test_json_mode_parser_reads_header_text_and_usage():
    stdout = "\n".join([
        json.dumps({"type": "session", "version": 3, "id": "abc", "cwd": "/w"}),
        json.dumps({"type": "model_change", "id": "1", "parentId": None}),
        json.dumps({"type": "message", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "part one "},
                        {"type": "text", "text": "part two"}]},
            "usage": {"input_tokens": 10, "output_tokens": 3}}),
    ])
    header, text, usage = S.parse_json_mode_stdout(stdout)
    assert header["id"] == "abc"
    assert text == "part one part two"
    assert usage["prompt_tokens"] == 10 and usage["completion_tokens"] == 3


# --------------------------------------------------------------------------- #
# 9. tier
# --------------------------------------------------------------------------- #

def test_e16_tier_differs_from_base_only_in_the_skill_set():
    import tool_tiers as T

    base = T.flags("base", "prime_agent")
    e16 = T.flags("e16_gewiki", "prime_agent")

    def as_pairs(flat):
        return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}

    b, e = as_pairs(base), as_pairs(e16)
    differing = {k for k in set(b) | set(e) if b.get(k) != e.get(k)}
    # tool_tier / tool_tier_hash are the tier's own provenance and MUST differ.
    assert differing == {"--taskset.env_args.skill_set",
                         "--taskset.env_args.tool_tier"}, differing
    assert b["--taskset.env_args.skill_set"] == "np_core,request_map,search"
    assert e["--taskset.env_args.skill_set"] == \
        "np_core,request_map,search,rollback,save,wiki"


def test_no_other_tier_publishes_save_or_wiki():
    import tool_tiers as T

    cfg = T.load()
    for tier in T.tiers(cfg):
        skills = T.contract_for(tier, cfg)["skill_set"]
        tokens = {t.strip() for t in skills.split(",")}
        if tier == "e16_gewiki":
            assert {"save", "wiki"} <= tokens
        else:
            assert not ({"save", "wiki"} & tokens), (
                f"tier {tier} would publish an E16 tool and change its served bytes")


# --------------------------------------------------------------------------- #
# 10. THE DRY RUN -- the whole loop, real engine, real archive, stub player
# --------------------------------------------------------------------------- #

class StubPlayer:
    """A player that plays real engine steps and reports a canned summary.

    It does everything a real player does to the ARCHIVE -- restores the
    checkpoint it was given (through the audited restore path), plays engine
    steps, and writes a new checkpoint -- and nothing a real player does to a
    model. That is the point: every disk-level assertion below is about code
    that also runs in a real attempt.
    """

    def __init__(self, outcomes, steps: int = 6):
        self.outcomes = list(outcomes)
        self.steps = steps
        self.seen = []

    def __call__(self, ctx: E.PlayerContext) -> E.PlayerResult:
        self.seen.append(ctx)
        env, meta = checkpoint_restore(ctx.checkpoint_dir,
                                       fidelity_log=ctx.fidelity_log)
        for _ in range(self.steps):
            env.step(ord("s"))
        # A harness-computed advance: gold feeds `score`, which is one of the
        # three Pareto objectives, so the frontier moves for a reason the
        # engine produced rather than one the stub asserted.
        env.modify(gold=1000 * ctx.attempt)
        target = ctx.archive_dir / f"c{100 + ctx.attempt}"
        saved = checkpoint_save(env, target,
                                name=f"attempt {ctx.attempt} state",
                                note="stub player checkpoint",
                                created_by="save")
        spec = self.outcomes[(ctx.attempt - 1) % len(self.outcomes)]
        return E.PlayerResult(
            stop_condition=spec.get("stop_condition", "died"),
            died=spec.get("died", True),
            calls=spec.get("calls", 25),
            spend_usd=spec.get("spend", 1.25),
            max_dlvl=saved["dlvl"], max_xl=saved["xl"],
            summary=spec.get("summary", "played six turns and stopped"),
            lesson=spec.get("lesson", "search less, descend more"),
            raw={"calls": spec.get("player_calls", [])},
        )


def test_dry_run_select_restore_play_save_ingest_over_four_attempts(tmp_path):
    cfg = cfg_for(tmp_path / "run", selector="scripted",
                  budget_ceiling_usd=1000.0, max_attempts=4,
                  stall_attempts=8, milestone_dlvl=99, milestone_dungeon=-1)
    player = StubPlayer([
        {"died": True, "stop_condition": "died", "spend": 2.0,
         "player_calls": [{"name": "np_move_to", "args": {"x": 5, "y": 5}}]},
        {"died": False, "stop_condition": "harness_timeout", "spend": 3.0,
         "player_calls": [{"name": "np_explore_level", "args": {}}]},
        {"died": True, "stop_condition": "died", "spend": 1.0,
         "player_calls": [{"name": "np_press_key", "args": {"key": "j"}}]},
        {"died": False, "stop_condition": "agent_completed", "spend": 0.5,
         "player_calls": []},
    ])
    orch = E.Orchestrator(cfg, player)
    orch.prepare()
    E.seed_archive(cfg)

    summary = orch.run(max_attempts=4)

    # -- the loop actually looped -------------------------------------------
    assert len(orch.attempts) == 4
    assert len(player.seen) == 4
    assert [c.attempt for c in player.seen] == [1, 2, 3, 4]
    # Every attempt resumed a checkpoint that existed at selection time.
    for ctx in player.seen:
        assert ctx.checkpoint_dir is not None and ctx.checkpoint_dir.is_dir()

    # -- the archive grew, and the new checkpoints are stamped ---------------
    names = sorted(p.name for p in checkpoint_list(cfg.archive_dir))
    assert names == ["c1", "c101", "c102", "c103", "c104"]
    for i in range(1, 5):
        meta = checkpoint_meta(cfg.archive_dir / f"c{100 + i}")
        assert meta["created_in_attempt"] == i
        assert meta["parent"] is not None
        # Numbers on a player-written checkpoint are still harness-computed.
        assert meta["score"] == 1000 * i
        assert meta["dlvl"] >= 1 and meta["max_hp"] > 0

    # -- attempts_from was incremented on the sources -----------------------
    total_attempts_from = sum(
        checkpoint_meta(p).get("attempts_from", 0)
        for p in checkpoint_list(cfg.archive_dir))
    assert total_attempts_from == 4

    # -- lessons were appended where the attempt started --------------------
    for ctx in player.seen:
        text = E.read_lessons(ctx.checkpoint_dir)
        assert "played six turns" in text
        assert "search less, descend more" in text
        assert "DIRECTIVE GIVEN:" in text

    # -- censoring is distinct from death -----------------------------------
    outcomes = [a["outcome"] for a in orch.attempts]
    assert outcomes == [E.OUTCOME_DIED, E.OUTCOME_CENSORED,
                        E.OUTCOME_DIED, E.OUTCOME_CENSORED]
    reasons = [a["censor_reason"] for a in orch.attempts]
    assert reasons == ["", E.CENSOR_WALL_CLOCK, "", E.CENSOR_EMPTY_COMPLETION]
    assert summary["attempts_died"] == 2 and summary["attempts_censored"] == 2

    # -- restore fidelity was audited on every restore ----------------------
    fid = [json.loads(l) for l in cfg.fidelity_path.read_text().splitlines() if l.strip()]
    assert len(fid) == 4
    assert all(r["ok"] for r in fid)
    assert set(fid[0]["fields"]) == set(E.AUDIT_FIELDS_FOR_TEST)

    # -- comparability accounting -------------------------------------------
    assert summary["frontier_advances"], "no frontier advance was recorded"
    adv = summary["frontier_advances"][0]
    for key in ("attempts_consumed", "cumulative_calls",
                "cumulative_spend_usd", "wall_clock_s"):
        assert key in adv
    best = summary["best_state"]
    assert best["attempts_to_reach"] >= 1
    assert summary["comparable_to_single_life_balrog"] is False
    assert "NOT comparable" in summary["comparability_note"]

    # -- luck accounting ----------------------------------------------------
    luck = json.loads(cfg.luck_path.read_text())
    assert sum(v["attempts"] for v in luck.values()) == 4
    for slot in luck.values():
        assert slot["directives"]

    # -- budget, two lines --------------------------------------------------
    assert summary["budget"]["player_usd"] == pytest.approx(6.5)
    assert summary["budget"]["orchestrator_usd"] == 0.0  # scripted arm
    assert summary["cumulative_spend_usd"] == pytest.approx(6.5)

    # -- every selection was recorded with its probabilities ----------------
    # The scripted selector's record lives under `scripted` now, where its own
    # `chosen_id` can no longer overwrite the id that was launched.
    sels = [json.loads(l) for l in cfg.selection_path.read_text().splitlines() if l.strip()]
    assert len(sels) == 4
    for s in sels:
        cands = s["scripted"]["candidates"]
        assert abs(sum(c["prob"] for c in cands) - 1.0) < 1e-9
        assert s["chosen_id"] in {c["id"] for c in cands}
        # This IS the scripted arm, so the two picks must agree -- and the
        # record must say so rather than leaving it to be inferred.
        assert s["scripted_would_pick"] == s["chosen_id"]
        assert s["scripted_agreed"] is True

    # -- provenance ---------------------------------------------------------
    prov = json.loads(cfg.provenance_path.read_text())
    assert prov["tier"] == "e16_gewiki"
    assert prov["commits"]["zombie_fix"] != "unknown"
    assert prov["wiki"]["sha256"]["why_do_i_keep_dying.md"]
    assert prov["selector"]["weights"] == {"balrog_min": 1.0, "novelty": 0.5,
                                           "attempts_from": 0.3}
    assert prov["comparable_to_single_life_balrog"] is False
    # The run's own copy of the wiki, not a reference to the source worktree.
    assert (cfg.wiki_dir / "why_do_i_keep_dying.md").is_file()
    assert (cfg.wiki_dir / "MANIFEST.json").is_file()
    assert (cfg.orchestrator_dir / "wiki_tool.py").is_file()


def test_dry_run_with_the_llm_orchestrator_records_directives_and_two_budget_lines(tmp_path):
    """The same loop with the decision seat occupied, still no inference.

    The session is real (`PrimeAgentSession`) down to argv assembly and session
    id handling; only the subprocess is scripted. So this exercises the code
    path a live run takes, including the directive reaching PlayerContext and
    the orchestrator's spend landing on its own budget line.
    """
    cfg = cfg_for(tmp_path / "run", selector="llm", budget_ceiling_usd=1000.0,
                  max_attempts=3, milestone_dlvl=99, milestone_dungeon=-1)
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)

    directives = iter([
        "from this state, search the north wall for a secret door",
        "descend immediately and do not fight anything on the way",
        "engrave Elbereth before exploring the east rooms",
    ])
    calls = {"n": 0}

    def runner(argv, env, cwd, timeout_s):
        calls["n"] += 1
        if calls["n"] == 1:                     # the opening discussion
            body = _STUB_PLAN
        else:
            body = ('Choosing the deepest healthy state.\n'
                    + json.dumps({"checkpoint": "1",
                                  "directive": next(directives),
                                  "rationale": "deepest healthy"}))
        return _stub_stdout(argv, cwd, body,
                            usage={"prompt_tokens": 4000,
                                   "completion_tokens": 300,
                                   "cached_input_tokens": 2000}), "", 0

    session = S.PrimeAgentSession(
        work_dir=cfg.orchestrator_dir, agent_dir=cfg.orchestrator_dir / "agent",
        log_path=cfg.orchestrator_log, runner=runner)
    player = StubPlayer([{"died": True, "spend": 1.0}])
    orch = E.Orchestrator(cfg, player, session=session)
    orch.prepare()
    E.seed_archive(cfg)
    summary = orch.run(max_attempts=3)

    # The opening discussion happened, BEFORE any launch, and is on disk.
    assert (cfg.orchestrator_dir / "opening_plan.txt").read_text() \
        .startswith("Round 1 plan for this seed.")
    # The opening was CHECKED for degeneration, and the verdict is on the
    # record next to the plan -- so "the plan was fine" is a measurement rather
    # than the absence of a complaint.
    degen = json.loads(
        (cfg.orchestrator_dir / "opening_degeneration.json").read_text())
    assert len(degen) == 1 and degen[0]["degeneration"]["degenerate"] is False
    rounds = [json.loads(l) for l in cfg.orchestrator_log.read_text().splitlines() if l.strip()]
    assert rounds[0]["kind"] == "opening"

    # Every launch carried a directive, and it is in the attempt record.
    assert all(c.directive for c in player.seen)
    assert "secret door" in player.seen[0].directive
    assert orch.attempts[0]["directive"] == player.seen[0].directive
    assert orch.attempts[0]["selection_source"] == "llm"
    assert all(a["directive_compliance"]["class"] for a in orch.attempts)

    # Directive text was also SERVED to disk for the served-bytes check.
    served = (Path(orch.attempts[0]["out_dir"]) / "directive_served.txt").read_text()
    assert served == player.seen[0].directive

    # TWO BUDGET LINES: the orchestrator is no longer free, and its cost is
    # not hidden inside the players'.
    assert summary["budget"]["player_usd"] == pytest.approx(3.0)
    assert summary["budget"]["orchestrator_usd"] > 0
    assert summary["orchestrator"]["rounds"] == 4       # 1 opening + 3 rounds
    assert summary["orchestrator"]["session_ids"] == ["sess-1"]
    assert summary["selection_sources"] == {"llm": 3}
    assert summary["llm_fallbacks"] == 0

    prov = json.loads(cfg.provenance_path.read_text())
    assert prov["orchestrator_session"]["ids"] == ["sess-1"]
    assert prov["orchestrator_session"]["session_resume_verified"] is None, (
        "resume continuity must be reported as UNVERIFIED until the probe runs")


def test_no_directive_control_mode_launches_with_no_instruction(tmp_path):
    cfg = cfg_for(tmp_path / "run", selector="scripted", no_directive=True,
                  budget_ceiling_usd=1000.0, milestone_dlvl=99,
                  milestone_dungeon=-1)
    player = StubPlayer([{"died": True, "spend": 0.5}])
    orch = E.Orchestrator(cfg, player)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=2)
    assert all(c.directive == "" for c in player.seen)
    assert all(a["directive_compliance"]["class"] == E.COMPLY_NONE
               for a in orch.attempts)


def test_matched_restart_is_n_independent_tries_from_one_fixed_state(tmp_path):
    """THE NULL ARM. Every planned control is an ablation WITHIN the method.

    Not one of them is a null: each is beaten by "we got N tries instead of
    one". The headline metric is a running maximum over attempts -- monotone
    non-decreasing by construction, unable to fall -- so comparing it to a
    single-life number compares max-of-N draws against one draw. This arm is
    the comparison every reader makes in their head, and without it there is no
    reading under which the method could have failed.

    What it must be, and what each assertion below pins: ONE fixed start state,
    no selection, no directive, no lessons, no archive the selector can see --
    but the SAME budget accounting and the SAME attempt records, or the two
    arms are not comparable at all.
    """
    cfg = cfg_for(tmp_path / "run", selector="scripted",
                  arm=E.ARM_MATCHED_RESTART, budget_ceiling_usd=1000.0,
                  max_attempts=4, stall_attempts=2, milestone_dlvl=99,
                  milestone_dungeon=-1)
    player = StubPlayer([{"died": True, "spend": 1.5}])
    orch = E.Orchestrator(cfg, player)
    orch.prepare()
    E.seed_archive(cfg)
    summary = orch.run(max_attempts=4)

    # N INDEPENDENT TRIES, all four of them. The stall stop must NOT truncate
    # this arm: it grows no frontier by construction, so an 8-attempt stall
    # rule would end the control at N=2 while the method it is compared against
    # ran to N=4 -- max-of-2 against max-of-4, called a matched comparison.
    assert len(orch.attempts) == 4
    assert orch.stop_reason != E.STOP_STALL

    # ONE FIXED START STATE, never re-selected.
    starts = {ctx.checkpoint_id for ctx in player.seen}
    assert starts == {"1"}, f"expected every attempt from c1, got {starts}"

    # NO DIRECTIVE, NO SELECTION, NO LEDGER, NO LESSONS.
    assert all(ctx.directive == "" for ctx in player.seen)
    assert all(ctx.ledger_text == "" for ctx in player.seen)
    assert all(a["selection_source"] == "matched_restart_fixed_start"
               for a in orch.attempts)
    assert E.read_lessons(cfg.archive_dir / "c1").strip() == "", (
        "the null arm must not accumulate lessons; N attempts that learn from "
        "each other are a weaker version of the method, not a null")

    # NO ARCHIVE THE SELECTOR CAN SEE: the player's own saves go to a scratch
    # directory per attempt and never enter the run's archive.
    assert [p.name for p in checkpoint_list(cfg.archive_dir)] == ["c1"]
    for ctx in player.seen:
        assert ctx.archive_dir != cfg.archive_dir
        assert ctx.archive_dir.name == "scratch_archive"

    # THE SAME ACCOUNTING AND THE SAME RECORDS, which is the only reason the
    # two arms can be compared.
    assert summary["budget"]["player_usd"] == pytest.approx(6.0)
    assert summary["experiment_arm"] == E.ARM_MATCHED_RESTART
    for a in orch.attempts:
        for key in ("attempt", "outcome", "censored", "max_dlvl", "max_xl",
                    "calls", "spend_usd", "cumulative_spend_usd",
                    "experiment_arm", "directive_compliance"):
            assert key in a, f"attempt record is missing {key}"
    prov = json.loads(cfg.provenance_path.read_text())
    assert prov["experiment_arm"] == E.ARM_MATCHED_RESTART
    assert "THE NULL" in prov["experiment_arm_meaning"]


def test_paired_control_runs_each_directive_against_its_own_control(tmp_path):
    """PER DIRECTIVE KIND, because a run-wide control proved not to be enough.

    Measured: a `descend_fast` directive inverted the player's first decision
    causally, while a `no_descend` directive was indistinguishable from the
    no-directive control -- because the control did not descend either. A
    prohibition against something the player was never going to do measures
    nothing, and a run-wide average pools the two into "directives work".
    """
    cfg = cfg_for(tmp_path / "run", selector="llm", paired_control=True,
                  budget_ceiling_usd=1000.0, max_attempts=4,
                  milestone_dlvl=99, milestone_dungeon=-1)
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    directives = iter(["descend to D10 immediately and take the down staircase",
                       "do not descend; clear this level and reach XL 3"])
    nth = {"n": 0}

    def runner(argv, env, cwd, timeout_s):
        nth["n"] += 1
        if nth["n"] == 1:                       # the opening discussion
            body = _STUB_PLAN
        else:
            body = ("plan\n" + json.dumps({"checkpoint": "1",
                                           "directive": next(directives),
                                           "rationale": "x"}))
        return _stub_stdout(argv, cwd, body, session_id="s1",
                            usage={"prompt_tokens": 100,
                                   "completion_tokens": 10,
                                   "cached_input_tokens": 0}), "", 0

    session = S.PrimeAgentSession(work_dir=cfg.orchestrator_dir,
                                  agent_dir=cfg.orchestrator_dir / "agent",
                                  log_path=cfg.orchestrator_log, runner=runner)
    player = StubPlayer([{"died": True, "spend": 0.5}])
    orch = E.Orchestrator(cfg, player, session=session)
    orch.prepare()
    E.seed_archive(cfg)
    summary = orch.run(max_attempts=4)

    # TWO ATTEMPTS PER DIRECTIVE, from the SAME checkpoint, one instruction
    # apart. A control launched from the state the treatment just produced
    # would be measuring the treatment.
    assert len(orch.attempts) == 4
    roles = [a["pair_role"] for a in orch.attempts]
    assert roles == [E.ROLE_TREATMENT, E.ROLE_CONTROL,
                     E.ROLE_TREATMENT, E.ROLE_CONTROL]
    assert [a["pair_id"] for a in orch.attempts] == [1, 1, 2, 2]
    # WITHIN a pair the checkpoint must be identical -- that is what makes the
    # pair a comparison. Across pairs the orchestrator is free to move.
    for pid in (1, 2):
        half = [a["from_checkpoint"] for a in orch.attempts
                if a["pair_id"] == pid]
        assert len(set(half)) == 1, (
            f"pair {pid} ran its control from a different checkpoint than its "
            f"treatment: {half}")

    # The control carries no directive; the treatment does.
    treatments = [a for a in orch.attempts if a["pair_role"] == E.ROLE_TREATMENT]
    controls = [a for a in orch.attempts if a["pair_role"] == E.ROLE_CONTROL]
    assert all(t["directive"] for t in treatments)
    assert all(c["directive"] == "" for c in controls)
    assert all(c["directive_compliance"]["class"] == E.COMPLY_NONE
               for c in controls)

    # PER KIND: the control inherits the kind it is controlling FOR, so the two
    # directive kinds are analysable separately rather than pooled.
    kinds = [a["directive_kind"] for a in orch.attempts]
    assert kinds[0] == kinds[1] and kinds[2] == kinds[3]
    assert kinds[0] != kinds[2], (
        "a 'descend' directive and a 'do not descend' directive must not be "
        "pooled into one control comparison")
    assert "no_descend" in kinds[2] and kinds[0] == "descend"

    pairs = summary["matched_pairs"]
    assert pairs["enabled"] is True
    assert pairs["complete_pairs"] == 2 and pairs["incomplete_pairs"] == []
    assert set(pairs["by_directive_kind"]) == {kinds[0], kinds[2]}
    for slot in pairs["pairs"]:
        assert slot["complete"] and "delta" in slot


def test_reseed_is_off_by_default_and_recorded_either_way(tmp_path):
    """The default must stay deterministic, and the choice must be legible.

    With the dice fixed, the difference between two attempts from one
    checkpoint comes from the model's choices and from the directive -- which
    is what makes a directive's effect attributable to the directive. Turning
    reseeding on adds a second, uncontrolled source of variance to every A/B.
    So it is a flag, not a default, and provenance says which was used.
    """
    cfg = cfg_for(tmp_path / "off", selector="scripted",
                  budget_ceiling_usd=1000.0, milestone_dlvl=99,
                  milestone_dungeon=-1)
    orch = E.Orchestrator(cfg, StubPlayer([{"died": True, "spend": 0.1}]))
    prov = orch.prepare()
    assert cfg.reseed_on_restore is False
    assert prov["reseed_on_restore"] is False
    assert "DETERMINISTIC (default)" in prov["reseed_semantics"]
    assert orch.reseed_for("1", 1) is None, (
        "with --reseed off the launcher must be handed None -- 'do not reseed "
        "at all' -- never a zero pair, which would be a third semantics")

    cfg2 = cfg_for(tmp_path / "on", selector="scripted", reseed_on_restore=True,
                   budget_ceiling_usd=1000.0, milestone_dlvl=99,
                   milestone_dungeon=-1)
    orch2 = E.Orchestrator(cfg2, StubPlayer([{"died": True, "spend": 0.1}]))
    prov2 = orch2.prepare()
    assert prov2["reseed_on_restore"] is True
    assert "diverge" in prov2["reseed_semantics"]
    # Reproducible from provenance, and DIFFERENT per attempt -- which is the
    # entire point of asking for it.
    a1, a2 = orch2.reseed_for("1", 1), orch2.reseed_for("1", 2)
    assert a1 and a2 and a1 != a2
    assert orch2.reseed_for("1", 1) == a1
    assert E.Orchestrator(cfg2, lambda c: None).reseed_for("1", 1) == a1


def test_reseed_changes_the_continuation_and_the_default_does_not(tmp_path):
    """The engine-level claim, checked against the engine.

    Two restores of one checkpoint continue BYTE-IDENTICALLY by default --
    restoring does not reroll the dice, whatever a Monte-Carlo reading of the
    archive would assume. `--reseed` is what makes them diverge, reproducibly.
    """
    import hashlib
    from nethack_core.env import NetHackCoreEnv
    from nethack_harness.checkpoints import _engine_of, _raw_of

    env = NetHackCoreEnv(task_name="NetHackChallenge-v0",
                         max_episode_steps=100_000)
    env.seed(core=1, disp=1)
    env.reset(character="Val-hum-neu-fem")
    env.step(13)
    for _ in range(20):
        env.step(ord("s"))
    env.step(27)
    ck = tmp_path / "c1"
    checkpoint_save(env, ck, name="probe", note="reseed probe",
                    created_by="test")

    def tail(reseed):
        e, meta = checkpoint_restore(ck, fidelity_log=None, reseed=reseed)
        raw = _raw_of(_engine_of(e))
        for _ in range(60):
            raw.step(ord("s"))
        o = raw.to_core_observation()
        digest = hashlib.sha256(
            bytes(o.chars) + o.blstats.astype("int64").tobytes()).hexdigest()
        return digest, meta.get("restored_with_reseed")

    d_a, seed_a = tail(None)
    d_b, seed_b = tail(None)
    assert d_a == d_b and seed_a is None and seed_b is None, (
        "the DEFAULT must be deterministic: two restores of one checkpoint "
        "replay byte-identically, so restore-and-retry is replay of a fixed "
        "stream rather than resampling")

    d_c, seed_c = tail((424242, 99))
    d_d, _ = tail((424242, 99))
    d_e, _ = tail((7, 7))
    assert seed_c == [424242, 99], "the resume must record what it reseeded with"
    assert d_c == d_d, "the same reseed must reproduce"
    assert d_c != d_a and d_c != d_e, "different seeds must diverge"


def test_stall_stop_fires_after_n_attempts_with_no_frontier_advance(tmp_path):
    cfg = cfg_for(tmp_path / "run", selector="scripted", stall_attempts=2,
                  budget_ceiling_usd=1000.0, milestone_dlvl=99,
                  milestone_dungeon=-1)

    def no_op_player(ctx):
        # Restores and plays, saves nothing -- so the frontier never moves.
        env, _ = checkpoint_restore(ctx.checkpoint_dir, fidelity_log=ctx.fidelity_log)
        env.step(ord("s"))
        return E.PlayerResult(died=True, spend_usd=0.1, calls=3,
                              summary="died immediately")

    orch = E.Orchestrator(cfg, no_op_player)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=10)
    assert orch.stop_reason == E.STOP_STALL
    assert len(orch.attempts) == 2


# --------------------------------------------------------------------------- #
# 20. the whole archive is the menu
#
# The defect: the round prompt showed the Pareto frontier over (Dlvl, XL,
# score) and told the orchestrator "Only ids that appear in the table are
# accepted". On one trajectory that frontier is a single row -- so an LLM
# hired to judge states was handed a candidate set of size one and could only
# justify a forced move. `parse_decision` had ALWAYS accepted any archive id;
# the table and the sentence were the whole constraint.
# --------------------------------------------------------------------------- #

@needs_real_archive
def test_the_real_archive_collapses_to_one_frontier_row(tmp_path):
    """The measurement the fix is answering. 24 states, 1 survivor."""
    rows = E.ledger_rows(copy_real_archive(tmp_path / "archive"))
    assert len(rows) == 24
    assert [r.id for r in E.pareto_frontier(rows)] == ["24"], (
        "if this ever fails the premise changed: the objective is no longer "
        "monotone along a single trajectory")


@needs_real_archive
def test_every_one_of_the_24_real_checkpoints_is_choosable(tmp_path):
    rows = E.ledger_rows(copy_real_archive(tmp_path / "archive"))
    attempts = [{"attempt": 1, "from_checkpoint": "1", "outcome": "censored",
                 "censor_reason": "interrupted", "max_dlvl": 8, "calls": 249,
                 "directive": "take the down-stairs on each level",
                 "directive_compliance": {"class": "unclassified"}}]
    text, cands = E.build_ledger(rows, attempts=attempts)

    # 1. EVERY id is in the ledger and in the recorded candidate set.
    assert {c["id"] for c in cands} == {str(i) for i in range(1, 25)}
    for i in range(1, 25):
        assert f"c{i} " in text or f"c{i}\n" in text, f"c{i} is unreachable"
    # And the sentence that used to hide 23 of them is gone.
    assert "dominated auto-checkpoint" not in text
    assert "not listed" not in text

    # 2. THE AXES THE OBJECTIVE CANNOT SEE are on every row.
    assert "Dungeons of Doom" in text and "branch" in text
    assert "hp%" in text
    assert "  96%" in text, "c17's 24/25 HP must be legible as a fraction"
    assert "level-entry" in text and "cadence" in text and "level-up" in text
    assert "seed" in text
    assert "entered Dlvl 6" in text, "the purpose note must survive"
    assert "censored:interrupt" in text, "outcomes of past resumes from c1"

    # 3. DUPLICATES COLLAPSED, not dropped. c13/c14 and c19/c20 are identical
    #    on every measured axis; each pair is one row that names both ids.
    assert "22 distinct measured position" in text
    assert "same measured state, each id choosable: c13 c14" in text
    assert "same measured state, each id choosable: c19 c20" in text
    dup = {c["id"]: c for c in cands}
    assert dup["14"]["same_state_as"] == "13"
    assert dup["20"]["same_state_as"] == "19"
    assert dup["13"]["duplicate_group_size"] == 2
    assert dup["17"]["same_state_as"] is None

    # 4. The branch and HP the selector needs are in the RECORD too, so a
    #    later analysis can ask what kind of state got picked.
    c17 = dup["17"]
    assert c17["branch"] == "Dungeons of Doom" and c17["dungeon_number"] == 0
    assert c17["kind"] == "level-entry"
    assert c17["hp_fraction"] == pytest.approx(24 / 25)
    assert c17["on_frontier"] is False and dup["24"]["on_frontier"] is True
    assert dup["1"]["outcomes"] == ["censored:interrupted@D8"]


def test_the_round_prompt_no_longer_claims_the_table_is_the_whole_menu():
    """A regression on the sentence that made 23 of 24 states unchoosable.

    It was also FALSE: `parse_decision` is handed every archive id, so the
    prompt was describing a constraint the code did not have.
    """
    p = E.Orchestrator.ROUND_PROMPT
    assert "Only ids that appear in the table are accepted" not in p
    assert "id from the table above" not in p
    assert "Any checkpoint id in the archive above is a legal choice." in p
    # The retry prompt lists ids for the same reason and must not narrow them.
    assert "appears in the table above" not in E.Orchestrator.DIRECTIVE_RETRY_PROMPT


def test_the_launched_id_is_never_shadowed_by_the_scripted_pick():
    """The selection-plumbing root cause, as a unit.

    The record used to end with ``**choice["selection"]``, and
    `Selection.to_json` carries a ``chosen_id`` of its own -- the SCRIPTED
    pick. Python applies ``**`` last, so the scripted id overwrote the launched
    one in every round. It looked harmless only because a one-row frontier made
    the two agree.
    """
    rows = [mkrow(17, dlvl=6, xl=2, score=391), mkrow(24, dlvl=8, xl=4, score=731)]
    cfg = E.OrchestratorConfig(run_dir=Path("/tmp/x"))
    scripted = E.select(rows, cfg, random.Random(0))
    assert scripted.chosen_id == "24"          # the frontier's only row

    choice = {"checkpoint_id": "17", "directive": "go back to D6 and take the "
              "other staircase", "source": "llm",
              "selection": scripted.to_json(),
              "candidates_shown": [{"id": "17"}, {"id": "24"}],
              "decision": {"valid": True}}
    rec = E.selection_record(3, choice, rows)

    assert rec["chosen_id"] == "17", "the launched id was overwritten again"
    assert rec["chosen_on_frontier"] is False
    assert rec["scripted_would_pick"] == "24"
    assert rec["scripted_agreed"] is False
    assert rec["n_candidates_shown"] == 2
    assert rec["frontier_ids"] == ["24"]
    # The scripted record is kept whole, where it cannot collide.
    assert rec["scripted"]["candidates"][0]["id"] == "24"
    assert "chosen_id" not in rec["scripted"]


@needs_real_archive
def test_a_dominated_checkpoint_is_accepted_launched_and_recorded(tmp_path):
    """End to end, on the real 24-checkpoint archive, with no inference.

    The orchestrator names c17 -- Dlvl 6, XL 2, score 391, dominated by c24 on
    all three objectives and therefore invisible to the old ledger. It must be
    accepted, the player must actually resume THAT state, and
    `selection.jsonl` must say c17 rather than the scripted selector's c24.
    """
    run = tmp_path / "run"
    cfg = cfg_for(run, selector="llm", budget_ceiling_usd=1000.0,
                  max_attempts=1, milestone_dlvl=99, milestone_dungeon=-1)
    cfg.archive_dir.parent.mkdir(parents=True, exist_ok=True)
    copy_real_archive(cfg.archive_dir)
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)

    shown = {}

    def runner(argv, env, cwd, timeout_s):
        text = argv[-1] if argv else ""
        if "ROUND" in text:
            shown["ledger"] = text
            body = ("c17 is at D6 with 24/25 HP and an unexplored side "
                    "passage; branching there.\n" + json.dumps(
                        {"checkpoint": "c17",
                         "directive": "from c17 take the north staircase and "
                                      "do not descend past D7 this attempt",
                         "rationale": "branch off the deepest line"}))
        else:
            body = _STUB_PLAN
        return _stub_stdout(argv, cwd, body), "", 0

    session = S.PrimeAgentSession(
        work_dir=cfg.orchestrator_dir, agent_dir=cfg.orchestrator_dir / "agent",
        log_path=cfg.orchestrator_log, runner=runner)
    player = StubPlayer([{"died": True, "spend": 1.0}])
    orch = E.Orchestrator(cfg, player, session=session)
    orch.prepare()
    summary = orch.run(max_attempts=1)

    # -- the model was shown all 24, and c17's row said what it needed to ----
    ledger = shown["ledger"]
    assert "c17" in ledger and "c24" in ledger
    assert "Any checkpoint id in the archive above is a legal choice." in ledger

    # -- the dominated choice was ACCEPTED and actually resumed -------------
    assert orch.llm_fallbacks == 0
    assert len(player.seen) == 1
    ctx = player.seen[0]
    assert ctx.checkpoint_id == "17"
    assert Path(ctx.checkpoint_dir).name == "c17"
    # ...from the real bundle, through the audited restore path.
    fid = [json.loads(l) for l in
           cfg.fidelity_path.read_text().splitlines() if l.strip()]
    assert fid and fid[-1]["id"] == "17" and fid[-1]["ok"] is True
    assert fid[-1]["fields"]["dlvl"]["engine"] == 6
    assert fid[-1]["fields"]["xl"]["engine"] == 2

    # -- and the record says c17, not the scripted selector's c24 -----------
    sels = [json.loads(l) for l in
            cfg.selection_path.read_text().splitlines() if l.strip()]
    assert len(sels) == 1
    s = sels[0]
    assert s["chosen_id"] == "17"
    assert s["source"] == "llm"
    assert s["chosen_on_frontier"] is False
    assert s["scripted_would_pick"] == "24"
    assert s["scripted_agreed"] is False
    assert s["frontier_ids"] == ["24"]
    assert s["n_candidates_shown"] == 24
    ids = {c["id"] for c in s["candidates_shown"]}
    assert ids == {str(i) for i in range(1, 25)}
    c17 = next(c for c in s["candidates_shown"] if c["id"] == "17")
    assert c17["dlvl"] == 6 and c17["kind"] == "level-entry"
    assert c17["hp_fraction"] == pytest.approx(24 / 25)

    # -- every downstream record agrees ------------------------------------
    assert orch.attempts[0]["from_checkpoint"] == "17"
    journal = [json.loads(l) for l in
               cfg.journal_path.read_text().splitlines() if l.strip()]
    opened = next(r for r in journal if r["event"] == "open")
    assert opened["from_checkpoint"] == "17"
    assert summary["selection"]["dominated_choices"] == 1
    assert summary["selection"]["dominated_ids"] == ["17"]
    assert summary["selection"]["scripted_differed"] == 1
    assert summary["selection"]["chosen_ids"] == ["17"]
    assert summary["selection"]["scripted_would_pick"] == ["24"]


@needs_real_archive
def test_recovery_reads_the_launched_id_out_of_the_selection_log(tmp_path):
    """The second half of the plumbing bug: it did not stop at the log.

    `reconcile_run` fills a recovered attempt's `from_checkpoint` from
    `selection.jsonl`, so a shadowed `chosen_id` became a WRONG resume in
    `attempts.jsonl` -- and every checkpoint attributed to that attempt was
    attributed to a resume that never happened.
    """
    run = tmp_path / "run"
    cfg = cfg_for(run, selector="llm")
    cfg.archive_dir.parent.mkdir(parents=True, exist_ok=True)
    copy_real_archive(cfg.archive_dir)
    rows = E.ledger_rows(cfg.archive_dir)
    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult())
    orch.prepare()
    E.clear_run_lock(cfg.run_dir)

    choice = {"checkpoint_id": "17", "directive": "branch at D6",
              "source": "llm",
              "selection": E.select(rows, cfg, random.Random(0)).to_json(),
              "candidates_shown": [], "decision": None}
    E._append_jsonl(cfg.selection_path, E.selection_record(1, choice, rows))

    out = cfg.run_dir / "attempts" / "a001"
    (out / "turns").mkdir(parents=True)
    (out / "turns" / "1_1_1787926656.ndjson").write_text(json.dumps({
        "turn": 1, "dlvl": 6, "max_dlvl_reached": 6, "hp": 24, "max_hp": 25,
        "applied": True,
        "status": {"depth": 6, "experience_level": 2, "time": 1716,
                   "hitpoints": 24, "max_hitpoints": 25, "score": 391},
        "tool_calls": [{"name": "request_map", "arguments": {}}]}) + "\n")

    E.reconcile_run(cfg, apply=True)
    rec = [json.loads(l) for l in
           cfg.attempts_path.read_text().splitlines() if l.strip()]
    assert rec[0]["from_checkpoint"] == "17", (
        "the recovered row named the scripted selector's pick again")


# Exposed for the dry run's field-set assertion without importing the engine
# module into the test's top level twice.
E.AUDIT_FIELDS_FOR_TEST = tuple(__import__(
    "nethack_harness.checkpoints", fromlist=["AUDIT_FIELDS"]).AUDIT_FIELDS)


# --------------------------------------------------------------------------- #
# 2b. IN-FLIGHT SPEND: the ceiling enforced DURING an attempt
# --------------------------------------------------------------------------- #
#
# `Budget.check()` is a PRE-LAUNCH gate, and its own docstring says cost "is
# therefore controlled by refusing the NEXT launch". That is only sound if an
# attempt's cost is bounded, and it is not -- players are uncapped by design.
# One runaway attempt can cross the ceiling with nothing able to notice until
# it returns: measured on a $9 smoke that reached an estimated $22.
#
# There is NO in-flight usage source (traces.jsonl is written at rollout
# completion; the interception proxy holds usage in memory; the turn files
# carry no token fields; both player arms run with session persistence off).
# So the guard runs on an ESTIMATE, and these tests pin that the estimate is
# labelled as one everywhere it appears.


def test_the_spend_rate_is_calibrated_from_this_runs_own_attempts():
    """$/hour across ALL experiments spreads 8.4x (p10 $9.75, p90 $82.15 over
    349 real rollouts) and is useless as a global constant. WITHIN one cell it
    spreads 2.4x median, which is usable — so the rate must come from this
    run's own completed attempts as soon as it has any, and from the prior only
    until then."""
    cold = E.calibrate_spend_rate([])
    assert cold["rate_source"] == "prior"
    assert cold["rate_usd_per_hour"] == E.DEFAULT_SPEND_RATE_USD_PER_HOUR
    assert cold["rate_basis_attempts"] == 0

    warm = E.calibrate_spend_rate([{"wall_s": 3600.0, "spend_usd": 20.0},
                                   {"wall_s": 3600.0, "spend_usd": 40.0}])
    assert warm["rate_source"] == "run_calibrated"
    assert warm["rate_basis_attempts"] == 2
    assert warm["rate_usd_per_hour"] == pytest.approx(30.0)


def test_calibration_uses_what_was_BILLED_not_what_was_costed():
    """A retried attempt's `spend_usd` costs one rollout out of the two or
    three it paid for. Calibrating on that number would teach the guard a rate
    lower than the run's real burn, which is the wrong direction for a
    ceiling."""
    rate = E.calibrate_spend_rate([
        {"wall_s": 3600.0, "spend_usd": 10.0, "spend_usd_billed_upper_est": 30.0}])
    assert rate["rate_usd_per_hour"] == pytest.approx(30.0)


def test_short_attempts_do_not_poison_the_rate():
    """Under two minutes is mostly launch overhead; its ratio says nothing
    about burn rate and would swing the pooled number either way."""
    rate = E.calibrate_spend_rate([{"wall_s": 30.0, "spend_usd": 9.0}])
    assert rate["rate_source"] == "prior", "a 30s attempt calibrated the rate"


def test_in_flight_spend_never_claims_to_be_a_measurement(tmp_path):
    """No in-flight usage source exists. Every field says so, so that nothing
    downstream can mistake this for the costed number from `traces.jsonl`."""
    rate = {"rate_usd_per_hour": 36.0, "rate_source": "prior",
            "rate_basis_attempts": 0}
    est = E.estimate_inflight_spend(tmp_path, 3600.0, rate, safety_factor=2.0)
    assert est["spend_so_far_usd"] == pytest.approx(36.0)
    assert est["spend_so_far_upper_usd"] == pytest.approx(72.0)
    assert est["spend_so_far_is_estimate"] is True
    assert est["spend_so_far_source"] == "wall_clock_rate"
    assert est["spend_so_far_rate_source"] == "prior"


def test_rollouts_started_is_the_live_retry_signal(tmp_path):
    """`eval.log` is written by the eval CLI as it goes, unlike `launch.log`
    which only lands after the process exits. It is therefore the one place a
    retry storm is visible WHILE it is costing money."""
    assert E.rollouts_started(tmp_path) == 0, "no log means unobservable, not zero"
    (tmp_path / "eval.log").write_text(
        "INFO rollout start: id=aaa task=0 harness=nethack-prime-agent\n"
        "WARNING retrying rollout aaa (retry 1/2) after error: ProviderError\n"
        "INFO rollout start: id=bbb task=0 harness=nethack-prime-agent\n"
        "WARNING retrying rollout bbb (retry 2/2) after error: ProviderError\n"
        "INFO rollout start: id=ccc task=0 harness=nethack-prime-agent\n")
    assert E.rollouts_started(tmp_path) == 3


def test_progress_stream_carries_spend_so_far(tmp_path):
    """(a) of the fix: `progress.jsonl` already streamed depth, HP and turns
    while an attempt played, but the only money on it was
    `cumulative_spend_usd_before` — frozen at attempt start. A run could not be
    watched for spend at all."""
    cfg = cfg_for(tmp_path, budget_ceiling_usd=1000.0, progress_interval_s=0.1)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)
    seen = []

    def player(ctx):
        # Let the monitor take at least one sample while "playing".
        time.sleep(0.5)
        seen.append(ctx)
        return E.PlayerResult(died=True, spend_usd=1.0)

    orch = E.Orchestrator(cfg, player)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=1)

    rows = [json.loads(ln) for ln in
            (cfg.run_dir / "progress.jsonl").read_text().splitlines() if ln.strip()]
    assert rows, "no progress samples"
    assert all("spend_so_far_usd" in r for r in rows), \
        "spend_so_far missing from the progress stream"
    assert all(r["spend_so_far_is_estimate"] is True for r in rows)
    # And it MOVES — a frozen number would be the bug this replaces.
    vals = [r["spend_so_far_usd"] for r in rows]
    assert vals[-1] > vals[0], vals


def test_the_ceiling_is_enforced_during_an_attempt_not_only_before_it(tmp_path):
    """(b) of the fix, and the headline case. The pre-launch gate passes — the
    run has plenty of headroom when the attempt starts — and then the attempt
    itself runs past the ceiling. Nothing in the old design could notice."""
    cfg = cfg_for(tmp_path, budget_ceiling_usd=10.0, min_headroom_usd=1.0,
                  progress_interval_s=0.1,
                  # $3600/hr == $1/s: this attempt blows a $10 ceiling in
                  # seconds instead of hours, with no inference and no waiting.
                  inflight_spend_prior_usd_per_hour=3600.0,
                  inflight_spend_safety_factor=1.0)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    class Runaway:
        """A player that would never stop on its own — the uncapped case."""

        def __init__(self):
            self.stopped = False
            self.ran_s = None

        def __call__(self, ctx):
            t0 = time.time()
            for _ in range(300):          # ~30s worst case, stopped far sooner
                if self.stopped:
                    break
                time.sleep(0.1)
            self.ran_s = time.time() - t0
            return E.PlayerResult(stop_condition="error",
                                  error="SIGTERM: killed mid-rollout",
                                  spend_usd=9.0)

        def terminate_active(self, grace_s=60.0):
            self.stopped = True
            return True

    player = Runaway()
    orch = E.Orchestrator(cfg, player)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=1)

    assert player.stopped, "the guard never stopped the runaway attempt"
    assert player.ran_s < 25, "the attempt ran to completion instead of being stopped"

    rows = [json.loads(ln) for ln in
            (cfg.attempts_path).read_text().splitlines() if ln.strip()]
    assert len(rows) == 1
    row = rows[0]
    # The row says WHY it ended. Left alone it would have read
    # `censored:harness_error` — a SIGTERM looks exactly like a crash — and the
    # run's censoring table would have been wrong about its own decision.
    assert row["outcome"] == E.OUTCOME_CENSORED
    assert row["censor_reason"] == E.CENSOR_BUDGET_STOP
    assert row["stop_condition"] == "budget_stop"
    # And it carries the evidence for the stop, including the admission that
    # the decision was made on an estimate.
    stop = row["budget_stop"]
    assert stop["decided_on"] == "estimate"
    assert stop["ceiling_usd"] == 10.0
    assert stop["projected_total_usd"] >= stop["ceiling_usd"] - stop["min_headroom_usd"]
    assert stop["rate_source"] == "prior"


def test_a_launcher_that_cannot_be_stopped_still_records_the_breach(tmp_path):
    """Degrade, never go silent. Injected launchers (every test in this file,
    and anything embedding the orchestrator) expose no `terminate_active`. The
    attempt then runs to completion — but the breach is on the progress stream,
    and the run still halts at the next pre-launch gate."""
    cfg = cfg_for(tmp_path, budget_ceiling_usd=5.0, min_headroom_usd=1.0,
                  progress_interval_s=0.1,
                  inflight_spend_prior_usd_per_hour=3600.0,
                  inflight_spend_safety_factor=1.0)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    def unstoppable(ctx):
        time.sleep(6.0)                   # past a $5 ceiling at $1/s
        return E.PlayerResult(died=True, spend_usd=6.0)

    orch = E.Orchestrator(cfg, unstoppable)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=2)

    rows = [json.loads(ln) for ln in
            (cfg.run_dir / "progress.jsonl").read_text().splitlines() if ln.strip()]
    assert any(r["event"] == "budget_stop" for r in rows), \
        "the breach was not recorded"
    assert orch.stop_reason == E.STOP_BUDGET


def test_the_in_flight_guard_can_be_turned_off(tmp_path):
    """It runs on an estimate, so it has to be possible to say no to it."""
    cfg = cfg_for(tmp_path, budget_ceiling_usd=5.0, min_headroom_usd=1.0,
                  progress_interval_s=0.1, enforce_inflight_budget=False,
                  inflight_spend_prior_usd_per_hour=3600.0)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    def player(ctx):
        time.sleep(3.0)
        return E.PlayerResult(died=True, spend_usd=1.0)

    orch = E.Orchestrator(cfg, player)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=1)
    rows = [json.loads(ln) for ln in
            (cfg.attempts_path).read_text().splitlines() if ln.strip()]
    assert rows[0]["outcome"] == E.OUTCOME_DIED, "guard fired while disabled"
    # The estimate is still STREAMED — observing is free, acting is what was
    # turned off.
    prog = [json.loads(ln) for ln in
            (cfg.run_dir / "progress.jsonl").read_text().splitlines() if ln.strip()]
    assert all("spend_so_far_usd" in r for r in prog)


# --------------------------------------------------------------------------- #
# 2c. a retried rollout's cost is VISIBLE
# --------------------------------------------------------------------------- #

def test_a_retried_attempt_says_how_many_rollouts_it_paid_for(tmp_path):
    """`run_with_retry` replays the whole trajectory and returns only the LAST
    attempt's trace, so `traces.jsonl` costs one rollout for an attempt that
    billed three. Measured: $31.43 for one `censored:harness_error` attempt
    that ran `retry 1/2` then `retry 2/2`. The count comes from the eval CLI's
    own warning lines, which `SubprocessPlayer` captures into `launch.log`."""
    out = tmp_path / "a001"
    out.mkdir()
    (out / "launch.log").write_text(
        "INFO rollout start: id=aaa task=0\n"
        "WARNING retrying rollout aaa (retry 1/2) after error: ProviderError\n"
        "WARNING retrying rollout bbb (retry 2/2) after error: ProviderError\n"
        "INFO rollout done: id=ccc task=0 reward=0.000 turns=193 stop=ProviderError\n")
    billed = E.rollouts_billed(out)
    assert billed["rollouts_paid"] == 3
    assert billed["retries_observed"] == 2
    assert billed["retry_errors"] == ["ProviderError", "ProviderError"]


def test_an_attempt_with_no_retries_is_not_marked_as_paying_extra(tmp_path):
    out = tmp_path / "a001"
    out.mkdir()
    (out / "launch.log").write_text("INFO rollout done: id=aaa task=0 reward=1.0\n")
    assert E.rollouts_billed(out)["rollouts_paid"] == 1
    # A missing log is "unobservable", and must not be reported as a retry.
    assert E.rollouts_billed(tmp_path / "nope")["source"] == "unavailable"


def test_the_attempt_record_flags_spend_as_a_lower_bound_when_retried(tmp_path):
    """The point of the count: an attempt that cost 3x must not be averaged
    into the per-attempt cost curve as though it were a normal one."""
    cfg = cfg_for(tmp_path, budget_ceiling_usd=1000.0)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    def player(ctx):
        return E.PlayerResult(died=True, spend_usd=10.0, rollouts_paid=3,
                              retry_errors=["ProviderError", "ProviderError"])

    orch = E.Orchestrator(cfg, player)
    orch.prepare()
    E.seed_archive(cfg)
    orch.run(max_attempts=1)
    row = [json.loads(ln) for ln in
           (cfg.attempts_path).read_text().splitlines() if ln.strip()][0]
    assert row["rollouts_paid"] == 3
    assert row["spend_is_lower_bound"] is True
    assert row["spend_usd"] == pytest.approx(10.0), "the costed number is untouched"
    assert row["spend_usd_billed_upper_est"] == pytest.approx(30.0)
    assert row["retry_errors"] == ["ProviderError", "ProviderError"]
