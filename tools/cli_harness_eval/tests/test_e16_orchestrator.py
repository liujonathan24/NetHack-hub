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

    many = E.render_ledger([mkrow(i, dlvl=i, xl=i, score=i * 10,
                                  name=f"n{i}", created_by="save")
                            for i in range(1, 31)])
    body = [ln for ln in many.splitlines() if ln.strip().startswith(("->", "1", "2", "3"))]
    assert len(body) <= 40  # capped, not the whole archive
    assert "measured by the harness" in many


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


def test_an_invented_checkpoint_id_falls_back_and_is_recorded_as_a_fallback(tmp_path):
    cfg = cfg_for(tmp_path, selector="llm")
    rows = [mkrow(1, dlvl=2, xl=1, score=10)]

    class FakeSession(S.SessionBase):
        kind = "fake"

        def ask(self, prompt, *, kind="round"):
            return S.RoundResult(text='{"checkpoint": "c99", "directive": "go"}')

    orch = E.Orchestrator(cfg, lambda ctx: E.PlayerResult(),
                          session=FakeSession(work_dir=tmp_path / "orch"))
    choice = orch.decide(rows)
    assert choice["source"] == "scripted_fallback"
    assert choice["checkpoint_id"] == "1"          # the scripted selector's pick
    assert "not in the archive" in choice["decision"]["fallback_reason"]
    assert orch.llm_fallbacks == 1


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

def _fake_prime_agent(session_id="019f-aaaa", reply="hello"):
    """Stands in for `prime-agent --print --mode json`."""
    seen = []

    def runner(argv, env, cwd, timeout_s):
        seen.append({"argv": list(argv), "env": dict(env), "cwd": cwd})
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
    assert sess.usage_available and sess.usage_total["prompt_tokens"] == 2000
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
        out = json.dumps({"type": "session", "version": 3, "id": next(ids)})
        msg = json.dumps({"type": "message",
                          "message": {"role": "assistant", "content": "ok"}})
        return out + "\n" + msg + "\n", "", 0

    sess = S.PrimeAgentSession(work_dir=tmp_path / "o", agent_dir=tmp_path / "a",
                               runner=runner)
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
    sels = [json.loads(l) for l in cfg.selection_path.read_text().splitlines() if l.strip()]
    assert len(sels) == 4
    for s in sels:
        assert abs(sum(c["prob"] for c in s["candidates"]) - 1.0) < 1e-9
        assert s["chosen_id"] in {c["id"] for c in s["candidates"]}

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
            body = "My plan: prefer healthy deep states; avoid the wraith."
        else:
            body = ('Choosing the deepest healthy state.\n'
                    + json.dumps({"checkpoint": "1",
                                  "directive": next(directives),
                                  "rationale": "deepest healthy"}))
        out = [json.dumps({"type": "session", "version": 3, "id": "sess-1"}),
               json.dumps({"type": "message",
                           "message": {"role": "assistant", "content": body},
                           "usage": {"prompt_tokens": 4000,
                                     "completion_tokens": 300,
                                     "cached_input_tokens": 2000}})]
        return "\n".join(out) + "\n", "", 0

    session = S.PrimeAgentSession(
        work_dir=cfg.orchestrator_dir, agent_dir=cfg.orchestrator_dir / "agent",
        log_path=cfg.orchestrator_log, runner=runner)
    player = StubPlayer([{"died": True, "spend": 1.0}])
    orch = E.Orchestrator(cfg, player, session=session)
    orch.prepare()
    E.seed_archive(cfg)
    summary = orch.run(max_attempts=3)

    # The opening discussion happened, BEFORE any launch, and is on disk.
    assert (cfg.orchestrator_dir / "opening_plan.txt").read_text().startswith("My plan")
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
            body = "my plan"
        else:
            body = ("plan\n" + json.dumps({"checkpoint": "1",
                                           "directive": next(directives),
                                           "rationale": "x"}))
        return ("\n".join([
            json.dumps({"type": "session", "version": 3, "id": "s1",
                        "cwd": cwd}),
            json.dumps({"type": "message",
                        "message": {"role": "assistant", "content": body},
                        "usage": {"prompt_tokens": 100,
                                  "completion_tokens": 10,
                                  "cached_input_tokens": 0}})]) + "\n", "", 0)

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


# Exposed for the dry run's field-set assertion without importing the engine
# module into the test's top level twice.
E.AUDIT_FIELDS_FOR_TEST = tuple(__import__(
    "nethack_harness.checkpoints", fromlist=["AUDIT_FIELDS"]).AUDIT_FIELDS)
