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
    assert got["rubric"] == "keyword-over-first-N-calls"


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
    # json mode on every round: it is how the id is discovered AND how
    # continuity is checked.
    assert a1[a1.index("--mode") + 1] == "json"
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
