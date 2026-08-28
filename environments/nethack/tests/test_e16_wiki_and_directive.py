"""E16 step 4 gate: the `wiki` tool, the publication guard, and the directive.

Three things are defended here, and the third is the one this project has been
burned by twice:

1. THE WIKI TOOL reads the curated two-page subset and nothing else, caps every
   response, and answers the questions the design names (wraith, prayer
   cooldown, Elbereth) with the right sections.
2. THE PUBLICATION GUARD. `save` and `wiki` must reach a served prompt ONLY
   through a tier that names them explicitly. Every preset -- `full` included,
   which publishes every registered skill -- must filter them, or merely
   registering a skill changes the served bytes of every existing arm and
   silently voids the comparison against the frozen E14/E15 control.
3. THE DIRECTIVE ACTUALLY REACHES THE MODEL. Not "the code passes it": the test
   boots a real env with a directive, drives one tool call, and reads the block
   back out of the TURN TRACE's `rendered_user_message` -- the harness's own
   record of the bytes that were served. A directive that lives only in our
   JSON is exactly the class of failure that shipped a wrong SKILL.md to 150
   E15 rollouts.
"""
import asyncio
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve()
ENVDIR = HERE.parents[1]
sys.path.insert(0, str(ENVDIR))

import nethack as m  # noqa: E402
from nethack_harness.checkpoints import (  # noqa: E402
    AUDIT_FIELDS, META_JSON, atomic_write, checkpoint_meta, checkpoint_restore,
    checkpoint_save, restore_fidelity,
)
from nethack_harness.integrity import CheckpointIntegrityError  # noqa: E402
from nethack_harness.wiki_kb import (  # noqa: E402
    RESPONSE_CHAR_CAP, TRUNCATION_MARKER, WikiKB, cap, install,
)

WIKI_SRC = pathlib.Path("/root/nld/e15-wiki/configs/continual/wiki")

pytestmark = pytest.mark.skipif(
    not WIKI_SRC.is_dir(),
    reason=f"curated wiki subset not present at {WIKI_SRC}")


@pytest.fixture(scope="module")
def kb():
    return WikiKB(WIKI_SRC)


# --------------------------------------------------------------------------- #
# 1. the knowledge base
# --------------------------------------------------------------------------- #

def test_kb_loads_exactly_the_curated_subset(kb):
    ids = sorted(p.id for p in kb.pages)
    assert ids == ["standard_strategy", "why_do_i_keep_dying"]
    # Provenance travels with the pages rather than being asserted in prose.
    assert kb.manifest["retrieved"] == "2026-08-25"
    assert "nethackwiki.com" in kb.manifest["source"]


def test_no_args_lists_pages_and_their_sections(kb):
    toc = kb.toc()
    assert "why_do_i_keep_dying" in toc and "standard_strategy" in toc
    assert "Praying" in toc and "The early game" in toc
    # Duplicate section titles are disambiguated by PATH -- `standard_strategy`
    # has four sections called "Goals", and a tool that offered them as four
    # identical names would be unusable.
    assert "The early game / Goals" in toc and "The mid game / Goals" in toc
    assert len(toc) <= RESPONSE_CHAR_CAP


def test_page_and_section_reads(kb):
    section = kb.read("why_do_i_keep_dying", "Praying")
    assert "Praying" in section
    assert "700" in section, "the prayer-cooldown numbers must survive"
    assert len(section) <= RESPONSE_CHAR_CAP

    # A whole page is capped and says so.
    page = kb.read("standard_strategy")
    assert len(page) <= RESPONSE_CHAR_CAP
    assert page.endswith(TRUNCATION_MARKER)

    # An ambiguous section returns all matches AND names them.
    goals = kb.read("standard_strategy", "Goals")
    assert "4 sections match" in goals
    assert "The early game / Goals" in goals


def test_unknown_page_and_section_are_answered_not_swallowed(kb):
    assert "No wiki page" in kb.read("dragons_and_taxes")
    # ...and the answer says what IS available, so the model can retry.
    assert "why_do_i_keep_dying" in kb.read("dragons_and_taxes")
    missing = kb.read("why_do_i_keep_dying", "Tax Policy")
    assert "No section" in missing and "Praying" in missing


@pytest.mark.parametrize("query,must_contain", [
    ("wraith", "wraith"),
    ("prayer", "pray"),
    ("Elbereth", "elbereth"),
    ("floating eye", "floating eye"),
])
def test_query_spot_checks_from_the_design(kb, query, must_contain):
    """The design's own acceptance for step 4: these queries return the right
    sections. Each of these is a death class the E15 autopsy named."""
    got = kb.search(query)
    assert "section(s) match" in got
    assert must_contain.lower() in got.lower()
    assert len(got) <= RESPONSE_CHAR_CAP


def test_a_search_that_is_truncated_still_names_every_hit(kb):
    """A capped response must not hide what it could not show.

    Otherwise a model asking about `pray` sees two of four sections and has no
    way to know the other two exist -- which turns the cap into silent data
    loss rather than a budget.
    """
    got = kb.search("pray")
    index_line = got.splitlines()[0]
    n = int(index_line.split()[0])
    for _ in range(n):
        pass
    assert index_line.count(";") == n - 1, "every hit is named on the index line"
    if got.endswith(TRUNCATION_MARKER):
        assert index_line in got


def test_no_match_is_a_helpful_answer_not_an_empty_one(kb):
    got = kb.search("qwertyuiop")
    assert "No wiki section matches" in got and "wiki()" in got


def test_cap_counts_the_marker_inside_the_budget():
    text = "y" * 5000
    out = cap(text, limit=100)
    assert len(out) <= 100 and out.endswith(TRUNCATION_MARKER)


def test_install_copies_the_pages_and_returns_their_hashes(tmp_path):
    hashes = install(WIKI_SRC, tmp_path / "wiki")
    assert set(hashes) == {"MANIFEST.json", "standard_strategy.md",
                           "why_do_i_keep_dying.md"}
    for name, digest in hashes.items():
        assert len(digest) == 64
        assert (tmp_path / "wiki" / name).read_bytes() == (WIKI_SRC / name).read_bytes()
    # The copy is a working KB on its own -- that is the point of copying it.
    assert WikiKB(tmp_path / "wiki").toc()


# --------------------------------------------------------------------------- #
# 2. the publication guard
# --------------------------------------------------------------------------- #

PRESETS = ["full", "dir8", "move", "netplay", "netplay_true", "np_core",
           "balrog80"]


@pytest.mark.parametrize("preset", PRESETS)
def test_no_preset_publishes_save_or_wiki(preset):
    from nethack_harness.helpers import _build_skill_adapter_callables as build

    names = {getattr(a, "__name__", "") for a in
             build(skill_set=preset, describe_args=True)}
    assert "save" not in names, f"{preset} would publish `save`"
    assert "wiki" not in names, f"{preset} would publish `wiki`"


def test_the_frozen_base_surface_is_byte_identical_in_size():
    """The E10/E14 baseline surface must not have moved.

    `np_core,request_map,search` is the frozen [base] tier's skill_set and the
    control arm for every E15 probe. Registering `save` and `wiki` must leave
    it at exactly the tools it had.
    """
    from nethack_harness.helpers import _build_skill_adapter_callables as build

    names = sorted(getattr(a, "__name__", "") for a in
                   build(skill_set="np_core,request_map,search",
                         describe_args=True))
    assert len(names) == 10
    assert "save" not in names and "wiki" not in names


def test_the_e16_tier_surface_publishes_exactly_the_two_new_tools():
    from nethack_harness.helpers import _build_skill_adapter_callables as build

    base = {getattr(a, "__name__", "") for a in
            build(skill_set="np_core,request_map,search", describe_args=True)}
    e16 = {getattr(a, "__name__", "") for a in
           build(skill_set="np_core,request_map,search,rollback,save,wiki",
                 describe_args=True)}
    assert e16 - base == {"rollback", "save", "wiki"}
    assert base - e16 == set()


def test_the_absolute_guard_still_holds_for_menu_tools():
    """`_HARNESS_OWNED` is absolute; `_E16_UNPUBLISHED_BY_DEFAULT` is not.

    Naming `menu_option` explicitly must still publish nothing -- otherwise the
    weaker guard for the E16 tools would have weakened the strong one too.
    """
    from nethack_harness.helpers import _build_skill_adapter_callables as build

    names = {getattr(a, "__name__", "") for a in
             build(skill_set="np_core,search,menu_option,inventory_item",
                   describe_args=True)}
    assert "menu_option" not in names and "inventory_item" not in names


# --------------------------------------------------------------------------- #
# 3. the wiki SKILL, dispatched
# --------------------------------------------------------------------------- #

class _FakeEnv:
    pass


def _call_wiki(env, **kwargs):
    from nethack_harness.tools.skills import registry
    return registry.call("wiki", env, None, **kwargs)


def test_wiki_skill_reads_the_kb_configured_on_the_env(tmp_path):
    install(WIKI_SRC, tmp_path / "wiki")
    env = _FakeEnv()
    from nethack_harness.tools.skills import WIKI_KB_ATTR
    setattr(env, WIKI_KB_ATTR, str(tmp_path / "wiki"))

    toc = _call_wiki(env)
    assert "why_do_i_keep_dying" in toc.feedback
    assert toc.actions == [], "the wiki must cost no game time"
    assert toc.interrupted

    hit = _call_wiki(env, query="wraith")
    assert "wraith" in hit.feedback.lower()

    page = _call_wiki(env, page="why_do_i_keep_dying", section="Praying")
    assert "Praying" in page.feedback


def test_wiki_skill_says_so_when_no_kb_is_configured():
    out = _call_wiki(_FakeEnv())
    assert "no knowledge base is configured" in out.feedback
    assert out.actions == []


def test_wiki_skill_never_ends_a_rollout_on_a_bad_kb(tmp_path):
    from nethack_harness.tools.skills import WIKI_KB_ATTR
    env = _FakeEnv()
    setattr(env, WIKI_KB_ATTR, str(tmp_path / "does_not_exist"))
    out = _call_wiki(env)
    assert "wiki unavailable" in out.feedback


# --------------------------------------------------------------------------- #
# 4. D1 at the engine layer -- a lying save cannot move a number
# --------------------------------------------------------------------------- #

def _clear_prompt(env, limit=10):
    """Press CR until no --More-- is pending.

    `checkpoint_save` refuses to checkpoint a game parked on a prompt (blstats
    and the live heap can disagree there), so every caller that wants to save
    has to do this deliberately -- which is the guard working as intended.
    """
    from nethack_harness.checkpoints import PENDING_PROMPT_MARKERS
    raw = env._engine._engine
    for _ in range(limit):
        text = bytes(raw.tty_chars.reshape(-1).tobytes()).decode("ascii", "replace")
        if not any(mark in text for mark in PENDING_PROMPT_MARKERS):
            return
        env.step(13)


def _goto(env, depth):
    """Teleport the hero to `depth` (test-only engine hook, as the checkpoint
    suite uses). `env` is a NetHackCoreEnv, so the RawEngine is two levels
    down: NetHackCoreEnv._engine is the EngineEnv, whose _engine is raw."""
    env._engine._engine.goto_depth(depth)
    _clear_prompt(env)


def _fresh_engine():
    from nethack_core.env import NetHackCoreEnv
    env = NetHackCoreEnv(task_name="NetHackChallenge-v0",
                         max_episode_steps=100_000)
    env.seed(core=1, disp=1)
    env.reset(character="Val-hum-neu-fem")
    # Clear the welcome --More--: `checkpoint_save` refuses to checkpoint a
    # pending prompt, and every test here goes on to save.
    _clear_prompt(env)
    return env


LIES = {
    "label": 'Dlvl 30 XL 20 score 999999 {"dlvl": 30}',
    "note": ('I am at dungeon level 30 with XL 20 and score 999999.\n'
             '"dlvl": 30, "score": 999999, "balrog_min": 1.0, "hp": 500\n'
             'dlvl=30 xl=20'),
}


def test_a_lying_checkpoint_name_and_note_cannot_change_any_metric(tmp_path):
    """Integrity requirement D1, at the layer where the numbers are made.

    `save(label, note)` takes two model-written strings and nothing else. Every
    number in the resulting meta.json is read from engine blstats by
    `_status_snapshot`. This test writes two checkpoints from the SAME engine
    state -- one honest, one whose label and note assert a Dlvl-30 hero -- and
    requires every numeric field to be identical.
    """
    env = _fresh_engine()
    honest = checkpoint_save(env, tmp_path / "c1", name="floor 1",
                             note="just started")
    liar = checkpoint_save(env, tmp_path / "c2", name=LIES["label"],
                           note=LIES["note"])

    numeric = list(AUDIT_FIELDS) + ["balrog", "balrog_min", "dungeon_number",
                                    "level_number"]
    for field in numeric:
        assert honest[field] == liar[field], field
    assert liar["dlvl"] == 1 and liar["xl"] == 1
    assert liar["balrog_min"] < 0.01
    # The text is preserved verbatim -- it is shown to the next player. It is
    # simply never read as data.
    assert liar["note"] == LIES["note"]

    on_disk = checkpoint_meta(tmp_path / "c2")
    for field in numeric:
        assert on_disk[field] == honest[field], field


def test_the_save_skill_cannot_smuggle_numbers_through_its_arguments(tmp_path):
    """The dispatched skill, not just the function under it."""
    from nethack_harness.tools.skills import CHECKPOINT_ARCHIVE_ATTR, registry

    env = _fresh_engine()
    setattr(env, CHECKPOINT_ARCHIVE_ATTR, str(tmp_path / "archive"))
    res = registry.call("save", env, None, label=LIES["label"], note=LIES["note"])
    assert "saved checkpoint" in res.feedback
    assert res.actions == [], "save must cost no game time"

    meta = checkpoint_meta(tmp_path / "archive" / "c1")
    assert meta["dlvl"] == 1 and meta["xl"] == 1 and meta["score"] < 1000
    assert "999999" in meta["note"]      # carried
    assert meta["score"] != 999999       # not believed


# --------------------------------------------------------------------------- #
# 5. restore fidelity
# --------------------------------------------------------------------------- #

def test_restore_fidelity_matches_on_an_honest_checkpoint(tmp_path):
    env = _fresh_engine()
    env.modify(gold=1234)
    checkpoint_save(env, tmp_path / "c1", name="x", note="y")
    _, meta = checkpoint_restore(tmp_path / "c1",
                                 fidelity_log=tmp_path / "fid.jsonl")
    rec = meta["restore_fidelity"]
    assert rec["ok"]
    assert set(rec["fields"]) == set(AUDIT_FIELDS)
    assert all(f["match"] for f in rec["fields"].values())
    logged = [json.loads(l) for l in
              (tmp_path / "fid.jsonl").read_text().splitlines() if l.strip()]
    assert len(logged) == 1 and logged[0]["ok"]


def test_a_meta_that_disagrees_with_the_engine_is_an_integrity_error(tmp_path):
    """A restore that does not match its own meta must RAISE, not continue.

    This is the property that keeps the archive honest: the experiment's depth
    claims are claims about states the archive describes, so a resume that
    quietly hands back a different state would let the run record states that
    never existed, and let the selector optimise over them.

    The mismatch is injected the only way that is possible without corrupting
    the bundle -- by editing meta.json, which is exactly the shape a stale or
    hand-edited archive entry would have.
    """
    env = _fresh_engine()
    checkpoint_save(env, tmp_path / "c1", name="x", note="y")
    meta = checkpoint_meta(tmp_path / "c1")
    meta["dlvl"] = 17                       # this game is on Dlvl 1
    atomic_write(tmp_path / "c1" / META_JSON,
                 json.dumps(meta, indent=2, sort_keys=True) + "\n")

    with pytest.raises(CheckpointIntegrityError) as exc:
        checkpoint_restore(tmp_path / "c1", fidelity_log=tmp_path / "fid.jsonl")
    assert "restore fidelity failure" in str(exc.value)
    assert "dlvl: meta=17" in str(exc.value)
    # The failing audit is on the record too, not only in the exception.
    logged = [json.loads(l) for l in
              (tmp_path / "fid.jsonl").read_text().splitlines() if l.strip()]
    assert logged[-1]["ok"] is False
    assert logged[-1]["fields"]["dlvl"]["match"] is False

    # And the audit can be turned off only EXPLICITLY -- so nothing silently
    # skips it.
    env2, meta2 = checkpoint_restore(tmp_path / "c1", audit=False)
    assert "restore_fidelity" not in meta2


def test_restore_fidelity_reads_the_live_engine_not_the_file(tmp_path):
    env = _fresh_engine()
    checkpoint_save(env, tmp_path / "c1", name="x", note="y")
    env2, _ = checkpoint_restore(tmp_path / "c1")
    honest = checkpoint_meta(tmp_path / "c1")
    assert restore_fidelity(env2, honest)["ok"]
    lying = dict(honest, score=999999)
    rec = restore_fidelity(env2, lying)
    assert not rec["ok"] and rec["fields"]["score"]["engine"] != 999999


# --------------------------------------------------------------------------- #
# 6. THE DIRECTIVE REACHES THE MODEL -- served-bytes check
# --------------------------------------------------------------------------- #

DIRECTIVE = ("from c12, avoid the east corridor and try the south door before "
             "descending")


def _env_with(tmp_path, **kwargs):
    trace_dir = tmp_path / "turns"
    trace_dir.mkdir(parents=True, exist_ok=True)
    env = m.load_environment(
        task_spec="full_nle", skill_set="np_core,request_map,search,rollback,save,wiki",
        n_examples=1, explicit_seeds=[1], character="Val-hum-neu-fem",
        variant="BBOX_MIN", trace_dir=str(trace_dir), **kwargs)
    import verifiers as vf
    state = vf.State({"task": {"tier": "full_nle", "seed": 1}, "info": {}})
    asyncio.run(env.setup_state(state))
    return env, state, trace_dir


def _served_text(content):
    return m.content_to_text(content) if hasattr(m, "content_to_text") else str(content)


def test_the_directive_appears_verbatim_in_the_served_observation(tmp_path):
    """Not "the code passes it": the bytes the model was sent carry it.

    Read back out of the TURN TRACE (`rendered_user_message`), which is the
    harness's own record of what was served -- the same channel every
    served-bytes verification in this program uses.
    """
    env, state, trace_dir = _env_with(tmp_path, directive=DIRECTIVE)
    content = asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    served = _served_text(content)

    expected = m.DIRECTIVE_BLOCK_FORMAT.format(directive=DIRECTIVE)
    assert expected in served, (
        "the directive block is not in the served observation:\n"
        + served[:1500])
    # It is a DELIMITED block of its own, at the very front of the turn.
    assert served.lstrip().startswith("[ORCHESTRATOR DIRECTIVE for this attempt:")

    # And the same bytes are in the trace file on disk.
    files = list(trace_dir.glob("*.ndjson"))
    assert files, "no turn trace was written"
    records = [json.loads(l) for f in files
               for l in f.read_text().splitlines() if l.strip()]
    assert records, "the turn trace is empty"
    assert any(expected in (r.get("rendered_user_message") or "")
               for r in records), (
        "the directive is not in the harness's own record of the served bytes")


def test_the_directive_is_served_once_not_every_turn(tmp_path):
    env, state, _ = _env_with(tmp_path, directive=DIRECTIVE)
    first = _served_text(asyncio.run(env._apply_tool_call(state, "search", {"times": 1})))
    second = _served_text(asyncio.run(env._apply_tool_call(state, "search", {"times": 1})))
    assert DIRECTIVE in first
    assert DIRECTIVE not in second, (
        "a per-turn directive would be a different treatment from a per-attempt "
        "one, and would inflate every rollout's prompt")


def test_no_directive_serves_no_directive_block(tmp_path):
    """The control mode has to be a real control: no block, no leftovers."""
    env, state, _ = _env_with(tmp_path)
    served = _served_text(asyncio.run(env._apply_tool_call(state, "search", {"times": 1})))
    assert "ORCHESTRATOR DIRECTIVE" not in served


def test_the_ledger_is_served_and_a_resume_banner_carries_engine_numbers(tmp_path):
    """A resumed player is told where it is -- from the checkpoint's meta."""
    env0 = _fresh_engine()
    env0.modify(gold=555)
    meta = checkpoint_save(env0, tmp_path / "c7", name="before the mines",
                           note="descend from here")

    ledger = "CHECKPOINT ARCHIVE (1 saved state)."
    env, state, _ = _env_with(
        tmp_path, directive=DIRECTIVE, ledger_text=ledger,
        resume_checkpoint=str(tmp_path / "c7"),
        fidelity_log=str(tmp_path / "fid.jsonl"))

    assert state["resumed_checkpoint"] == str(tmp_path / "c7")
    assert state["restore_fidelity"]["ok"]

    served = _served_text(asyncio.run(env._apply_tool_call(state, "search", {"times": 1})))
    assert DIRECTIVE in served
    assert ledger in served
    assert "RESUMED FROM CHECKPOINT 7" in served
    assert "before the mines" in served
    assert "descend from here" in served
    # The banner's NUMBERS are the harness's, and the note is labelled as the
    # author's own words so a model cannot read it as a measurement.
    assert f"Dlvl {meta['dlvl']}" in served
    assert "author's own words" in served
    # The hero really is the checkpoint's hero, not a fresh game.
    assert (state["structured_obs"].status or {}).get("gold") == 555


# --------------------------------------------------------------------------- #
# 7. the archive's BALROG pair is the SAME quantity every report uses
# --------------------------------------------------------------------------- #

def test_the_archive_balrog_pair_equals_what_the_trace_metrics_path_publishes(tmp_path):
    """One state, two code paths, one number.

    `nethack_v1.NetHackTask.finalize` publishes `balrog_pct` /
    `balrog_min_pct` from `balrog_both(max_dlvl_reached, max_xp_level)`. A
    checkpoint's meta must carry the SAME pair for the same state, or the
    archive and the traces describe the same run in two different currencies
    and every cross-reference between them is wrong.

    This is a regression test with a body count: the first implementation used
    `progression_score`, the deprecated analytic proxy. It is not a rescaling
    -- at the states seed 1 reaches (D12-D15, XL 3-5) it reads ~0.2x low on the
    max and ~2x HIGH on the min -- and because both numbers looked plausible,
    nothing would have caught it downstream.
    """
    from nethack_harness.prompt.balrog import balrog_both, progression_score

    env = _fresh_engine()
    _goto(env, 4)
    _goto(env, 2)
    meta = checkpoint_save(env, tmp_path / "c1", name="deep", note="",
                           max_dlvl=4, max_xl=1)

    # What nethack_v1.finalize would publish for this rollout:
    hi, lo = balrog_both(4, 1, reached_planes=False, ascended=False)
    assert meta["balrog"] == hi
    assert meta["balrog_min"] == lo
    assert 100.0 * meta["balrog"] == 100.0 * hi   # the published % is a view

    # And it is NOT the deprecated proxy, which is the bug this test exists for.
    assert meta["balrog"] != progression_score(4, 1)
    assert meta["balrog_metric"].startswith("balrog_both")


def test_the_save_skill_records_the_runs_high_water_marks_not_the_current_floor(tmp_path):
    """BALROG scores the deepest level a rollout TOUCHED.

    A checkpoint written after climbing back up must still carry the run's
    high-water pair -- otherwise every "went deep, retreated, saved" state
    reports as the shallow run it is standing in, and the archive
    systematically under-reports exactly the states Go-Explore is built to
    collect.

    The `save` skill takes only `(label, note)`, so the marks reach it through
    the env (`HIGH_WATER_*_ATTR`), which is what the harness mirrors each turn.
    """
    from nethack_harness.checkpoints import (
        HIGH_WATER_DLVL_ATTR, HIGH_WATER_XL_ATTR,
    )
    from nethack_harness.prompt.balrog import balrog_both
    from nethack_harness.tools.skills import CHECKPOINT_ARCHIVE_ATTR, registry

    env = _fresh_engine()
    _goto(env, 9)
    _goto(env, 2)                      # climbed back up
    setattr(env, HIGH_WATER_DLVL_ATTR, 9)
    setattr(env, HIGH_WATER_XL_ATTR, 4)
    setattr(env, CHECKPOINT_ARCHIVE_ATTR, str(tmp_path / "archive"))

    registry.call("save", env, None, label="back at 2", note="regrouping")
    meta = checkpoint_meta(tmp_path / "archive" / "c1")

    assert meta["dlvl"] == 2, "the hero really is on Dlvl 2"
    assert meta["max_dlvl_reached"] == 9 and meta["max_xp_level"] == 4
    assert (meta["balrog"], meta["balrog_min"]) == balrog_both(9, 4)
    # ...and the pair is strictly better than the standing-floor pair, which is
    # the whole point of tracking the high-water mark.
    assert meta["balrog"] > balrog_both(2, 4)[0]


def test_without_high_water_marks_the_pair_falls_back_to_the_checkpoints_own_state(tmp_path):
    """A missing mark must be a floor, never a guess."""
    from nethack_harness.prompt.balrog import balrog_both

    env = _fresh_engine()
    _goto(env, 3)
    meta = checkpoint_save(env, tmp_path / "c1", name="x", note="")
    assert meta["max_dlvl_reached"] == meta["dlvl"]
    assert (meta["balrog"], meta["balrog_min"]) == balrog_both(meta["dlvl"],
                                                               meta["xl"])


# --------------------------------------------------------------------------- #
# 8. prefix.jsonl -- written, round-tripped, and honest about its limits
# --------------------------------------------------------------------------- #

def test_prefix_jsonl_is_written_from_the_running_transcript_and_round_trips(tmp_path):
    from nethack_harness.checkpoints import (
        CONVERSATION_PREFIX_ATTR, checkpoint_prefix,
    )

    env = _fresh_engine()
    transcript = [
        {"role": "assistant", "content": "I will search the north wall.",
         "tool_calls": [{"name": "search", "arguments": {"times": 5}}]},
        {"role": "user", "content": "=== STATUS === HP 16/16 Dlvl 1"},
    ]
    setattr(env, CONVERSATION_PREFIX_ATTR, transcript)
    checkpoint_save(env, tmp_path / "c1", name="x", note="")

    got = checkpoint_prefix(tmp_path / "c1")
    assert got == transcript
    assert (tmp_path / "c1" / "prefix.jsonl").stat().st_size > 0


def test_an_absent_transcript_leaves_an_empty_prefix_not_a_broken_one(tmp_path):
    from nethack_harness.checkpoints import checkpoint_prefix

    env = _fresh_engine()
    checkpoint_save(env, tmp_path / "c1", name="x", note="")
    assert checkpoint_prefix(tmp_path / "c1") == []
    assert (tmp_path / "c1" / "prefix.jsonl").stat().st_size == 0


def test_the_harness_accumulates_a_transcript_across_turns(tmp_path):
    """The write half, end to end through the real env.

    Two tool calls, and the checkpoint the player would write carries both --
    which is what makes `prefix.jsonl` non-empty in a real run rather than the
    reserved-and-always-empty file it was.
    """
    from nethack_harness.checkpoints import (
        CONVERSATION_PREFIX_ATTR, checkpoint_prefix,
    )
    from nethack_harness.tools.skills import CHECKPOINT_ARCHIVE_ATTR

    env, state, _ = _env_with(tmp_path, checkpoint_archive=str(tmp_path / "archive"))
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))

    game_env = state["env"]
    buf = getattr(game_env, CONVERSATION_PREFIX_ATTR, [])
    assert buf, "no transcript was accumulated"
    assert any(r["role"] == "user" and "STATUS" in r["content"] for r in buf)
    assert any(r["role"] == "assistant" and r.get("tool_calls") for r in buf)
    assert all(len(r["content"]) <= m.CONVERSATION_PREFIX_CHARS for r in buf)

    # And `save` puts it in the checkpoint.
    from nethack_harness.tools.skills import registry
    setattr(game_env, CHECKPOINT_ARCHIVE_ATTR, str(tmp_path / "archive"))
    registry.call("save", game_env, state["structured_obs"],
                  label="mid-run", note="testing the prefix")
    got = checkpoint_prefix(tmp_path / "archive" / "c1")
    assert got and got == buf


def test_a_resumed_player_is_shown_the_prefix_as_quoted_text(tmp_path):
    """The READ half is a REPLAY, and the served bytes say so.

    True prefix continuity -- the resumed player's conversation IS the
    checkpoint's prefix -- is blocked by the player scaffolds, which launch
    with sessions off. What a resume can do is quote it, labelled, exactly as
    `resume_from` already does. This test pins the labelling, so nobody reads
    the run as having had continuity it did not have.
    """
    import sys as _sys
    _sys.path.insert(0, str(HERE.parents[3] / "tools" / "cli_harness_eval"))
    from e16_orchestrator import render_prefix
    from nethack_harness.checkpoints import CONVERSATION_PREFIX_ATTR

    env = _fresh_engine()
    setattr(env, CONVERSATION_PREFIX_ATTR, [
        {"role": "assistant", "content": "Heading for the down stair on the east side.",
         "tool_calls": [{"name": "np_move_to", "arguments": {"x": 60, "y": 9}}]},
        {"role": "user", "content": "=== STATUS === HP 12/16 Dlvl 5"},
    ])
    checkpoint_save(env, tmp_path / "c1", name="x", note="")

    text = render_prefix(tmp_path / "c1")
    assert "THE PLAN THAT WAS LIVE WHEN THIS STATE WAS SAVED" in text
    assert "quoted from that session, not your own memory" in text
    assert "east side" in text
    # An empty prefix renders nothing at all, rather than an empty banner.
    env2 = _fresh_engine()
    checkpoint_save(env2, tmp_path / "c2", name="x", note="")
    assert render_prefix(tmp_path / "c2") == ""


# --------------------------------------------------------------------------- #
# 9. a CORRUPT checkpoint is an error, not a full-HP death
# --------------------------------------------------------------------------- #

def test_a_damaged_bundle_body_is_caught_by_the_checksum(tmp_path):
    """The gap the file-existence guard cannot see.

    `missing_level_files()` only `stat()`s: a level file whose NAME is present
    but whose BODY is damaged restores "cleanly", the audit passes, and NetHack
    then `panic()`s several turns later -- arriving as `terminated` with the
    hero at full HP. That is the E15 misattribution class one `how_done` code
    over, and no amount of post-hoc checking can undo it once the run has
    scored it. The checksum catches it at read time, before an engine ever
    loads the bytes.
    """
    from nethack_harness.checkpoints import STATE_BUNDLE, unpack_bundle

    env = _fresh_engine()
    checkpoint_save(env, tmp_path / "c1", name="x", note="")
    path = tmp_path / "c1" / STATE_BUNDLE
    blob = bytearray(path.read_bytes())
    # One bit, deep in a section body -- the smallest damage there is.
    blob[-64] ^= 0x01
    path.write_bytes(bytes(blob))

    with pytest.raises(CheckpointIntegrityError) as exc:
        unpack_bundle(bytes(blob))
    assert "checksum mismatch" in str(exc.value)

    # And the restore path refuses too, rather than handing back a game.
    with pytest.raises(CheckpointIntegrityError):
        checkpoint_restore(tmp_path / "c1")


def test_trailing_garbage_and_a_rewritten_header_are_both_refused(tmp_path):
    from nethack_harness.checkpoints import (
        STATE_BUNDLE, pack_bundle, unpack_bundle,
    )

    blob = pack_bundle({"level": b"L" * 8, "player": b"P" * 4,
                        "levelfiles": b"F" * 2}, header_extra={"seed": [1, 1]})
    assert unpack_bundle(blob)[1]["level"] == b"L" * 8

    with pytest.raises(CheckpointIntegrityError) as exc:
        unpack_bundle(blob + b"junk")
    assert "trailing" in str(exc.value)

    # A header that claims version 2 but carries no checksum has been edited.
    # Located by the length prefix, not by hunting for a brace -- the header
    # contains a nested array, so the first `}` is inside it.
    import json as _json
    import struct as _struct
    from nethack_harness.checkpoints import BUNDLE_MAGIC

    off = len(BUNDLE_MAGIC)
    (hlen,) = _struct.unpack("<I", blob[off:off + 4])
    off += 4
    header = _json.loads(blob[off:off + hlen])
    body = blob[off + hlen:]
    header.pop("sha256")
    stripped = _json.dumps(header, sort_keys=True).encode()
    rebuilt = BUNDLE_MAGIC + _struct.pack("<I", len(stripped)) + stripped + body
    with pytest.raises(CheckpointIntegrityError) as exc:
        unpack_bundle(rebuilt)
    assert "no sha256" in str(exc.value)


def test_panic_is_an_engine_fault_not_a_death():
    """`how_done` is an ALLOWLIST now, so PANICKED cannot be scored.

    The first version whitelisted TRICKED (12) alone. A damaged level-file body
    reaches `panic()` -> `done(PANICKED)` (11) instead, which sailed through as
    `died=True` at full HP with `engine_error=None`.
    """
    from nethack_harness.integrity import (
        GAME_OUTCOME_HOW_DONE, HOW_PANICKED, HOW_TRICKED, assert_not_tricked,
        engine_fault_code, tricked_marker_in,
    )

    class _Fake:
        def __init__(self, code):
            self.done, self.how_done = True, code

    assert engine_fault_code(_Fake(HOW_PANICKED)) == HOW_PANICKED
    assert engine_fault_code(_Fake(HOW_TRICKED)) == HOW_TRICKED
    # Real outcomes are NOT faults: dying (0-10), quitting (13), escaping (14),
    # ascending (15).
    for code in sorted(GAME_OUTCOME_HOW_DONE):
        assert engine_fault_code(_Fake(code)) is None, code
    # An unknown future abort code is a fault by default, not a death.
    assert engine_fault_code(_Fake(99)) == 99

    with pytest.raises(CheckpointIntegrityError) as exc:
        assert_not_tricked(_Fake(HOW_PANICKED), where="after restore")
    assert "PANICKED" in str(exc.value)

    # The panic banner is a marker too -- it prints before `done` flips.
    assert tricked_marker_in({"message": "Suddenly, the dungeon collapses."})


def test_the_dungeon_check_fails_closed_when_the_engine_cannot_answer():
    """An unverifiable dungeon is not a verified one.

    This used to return quietly, so the one guard between a corrupt resume and
    a scoreable death was skipped exactly when the engine was odd enough not to
    answer.
    """
    from nethack_harness.integrity import assert_dungeon_on_disk

    class _EngineThatWontAnswer:
        # Looks like a RawEngine to `_raw_engine` (both attributes present),
        # then refuses the question -- the shape that used to pass silently.
        def snapshot(self):
            raise NotImplementedError

        def missing_level_files(self):
            raise RuntimeError("no live game")

    with pytest.raises(CheckpointIntegrityError) as exc:
        assert_dungeon_on_disk(_EngineThatWontAnswer(), where="probe")
    assert "could not run" in str(exc.value)

    # A caller with NO engine at all is a different case and must NOT raise:
    # there is no dungeon to verify, and turning that into an error would break
    # every non-engine-backed caller (test doubles, curriculum envs) while
    # protecting nothing.
    assert_dungeon_on_disk(object(), where="probe")


# --------------------------------------------------------------------------- #
# 10. meta.json cannot record a state that never existed
# --------------------------------------------------------------------------- #

def test_saving_on_a_pending_prompt_is_refused_not_silently_poisoned(tmp_path):
    """meta comes from blstats; the bundle comes from the heap; a prompt
    separates them.

    Measured: with a --More-- pending over a deferred level transition, meta
    recorded Dlvl 4 for a state that restores to Dlvl 2. The save SUCCEEDED,
    so the poisoned row reached the archive and every consumer that reads meta
    WITHOUT restoring -- the frontier selector, the stop conditions, the run
    summary, the BALROG fields -- believed it.
    """
    from nethack_harness.checkpoints import (
        CheckpointSavepointError, PENDING_PROMPT_MARKERS,
    )

    env = _fresh_engine()
    env._engine._engine.goto_depth(4)      # deliberately NOT cleared
    raw = env._engine._engine
    tty = bytes(raw.tty_chars.reshape(-1).tobytes()).decode("ascii", "replace")
    assert any(mark in tty for mark in PENDING_PROMPT_MARKERS), (
        "this test needs a pending prompt to be meaningful")

    with pytest.raises(CheckpointSavepointError) as exc:
        checkpoint_save(env, tmp_path / "c1", name="x", note="")
    assert "parked on a" in str(exc.value)
    assert not (tmp_path / "c1").exists(), "a refused save must leave nothing"

    # Clearing it makes the save legal again -- the guard is about the MOMENT,
    # not about the game being broken.
    _clear_prompt(env)
    meta = checkpoint_save(env, tmp_path / "c1", name="x", note="")
    assert meta["dlvl"] == 4


def test_the_save_skill_reports_a_bad_moment_instead_of_ending_the_rollout(tmp_path):
    """A pending prompt is a bad moment, not a broken game.

    `save` re-raises CheckpointIntegrityError on purpose -- a corrupt game must
    not be silently checkpointed. A savepoint refusal must NOT take that path,
    or a model calling save at a --More-- would kill its own uncapped rollout.
    """
    from nethack_harness.tools.skills import CHECKPOINT_ARCHIVE_ATTR, registry

    env = _fresh_engine()
    env._engine._engine.goto_depth(4)
    setattr(env, CHECKPOINT_ARCHIVE_ATTR, str(tmp_path / "archive"))
    res = registry.call("save", env, None, label="mid-prompt", note="")
    assert "save failed" in res.feedback
    assert "parked on a" in res.feedback
    assert res.actions == []


def test_branch_identity_is_audited_on_restore(tmp_path):
    """`dungeon_number` drives the Sokoban stop condition and was unauditable."""
    from nethack_harness.checkpoints import AUDIT_FIELDS as F

    assert "dungeon_number" in F and "level_number" in F
    env = _fresh_engine()
    checkpoint_save(env, tmp_path / "c1", name="x", note="")
    meta = checkpoint_meta(tmp_path / "c1")
    meta["dungeon_number"] = 4            # claim Sokoban from the Dungeons of Doom
    atomic_write(tmp_path / "c1" / META_JSON,
                 json.dumps(meta, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CheckpointIntegrityError) as exc:
        checkpoint_restore(tmp_path / "c1")
    assert "dungeon_number" in str(exc.value)


def test_one_unreadable_meta_does_not_make_the_whole_archive_unlistable(tmp_path):
    """A single damaged entry must not be a total loss of the run's state."""
    from nethack_harness.checkpoints import checkpoint_list

    env = _fresh_engine()
    checkpoint_save(env, tmp_path / "c1", name="a", note="")
    checkpoint_save(env, tmp_path / "c2", name="b", note="")
    (tmp_path / "c2" / META_JSON).write_text("{not json")

    listed = checkpoint_list(tmp_path)
    assert (tmp_path / "c1") in listed
    # ...and the broken one raises a CheckpointIntegrityError when READ, rather
    # than a bare JSONDecodeError that callers would have to know about.
    with pytest.raises(CheckpointIntegrityError):
        checkpoint_meta(tmp_path / "c2")
    with pytest.raises(CheckpointIntegrityError):
        checkpoint_meta(tmp_path / "c404")


# --------------------------------------------------------------------------- #
# AUTOMATIC CHECKPOINTS -- the archive grows without the model's cooperation
#
# THE MEASUREMENT THAT FORCED THIS SECTION. The E16 GE-wiki pilot ran one real
# attempt: 187 skill calls, Dlvl 1 -> 3, $13.36. It called `save` ZERO times.
# `attempts.jsonl` recorded `new_checkpoints: []`, and round 2 of the
# orchestrator was handed the same one-row archive round 1 had seen. Go-Explore
# without archive growth is not Go-Explore; it is repeated restarts from the
# seed state at $13 each. Nothing in the harness had ever written a checkpoint
# on its own -- the design named level entry, level-up and every 150 game turns,
# and none of the three existed in code.
# --------------------------------------------------------------------------- #

def _auto(state):
    return state.get("_auto_ck") or {}


def _archive_ids(root):
    from nethack_harness.checkpoints import checkpoint_list
    return sorted(p.name for p in checkpoint_list(root))


def test_auto_checkpoint_is_off_unless_an_archive_is_configured(tmp_path):
    """No arm outside E16 acquires this by having it default to on."""
    env, state, _ = _env_with(tmp_path)
    assert _auto(state)["enabled"] is False


def test_the_first_call_seeds_the_watermarks_and_saves_nothing(tmp_path):
    """A rollout RESUMED at Dlvl 4 must not re-save Dlvl 4 on call one.

    The state it resumed from is already in the archive; firing "entered
    Dlvl 4" would duplicate it and put a second row on the frontier that is the
    same state.
    """
    archive = tmp_path / "archive"
    env, state, _ = _env_with(tmp_path, checkpoint_archive=str(archive))
    assert _auto(state)["enabled"] is True
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
    assert _archive_ids(archive) == []
    assert _auto(state)["seen_dlvl"] == {1}
    assert _auto(state)["saved"] == []


def test_a_periodic_auto_checkpoint_fires_on_game_turns(tmp_path):
    """The 150-game-turn trigger, driven by the real engine clock.

    This is the trigger for a rollout that is making progress the depth and XL
    triggers cannot see -- wandering a large level for hundreds of turns. The
    interval is shortened here so the check costs seconds instead of minutes;
    the production default is 150.
    """
    archive = tmp_path / "archive"
    env, state, _ = _env_with(tmp_path, checkpoint_archive=str(archive),
                              auto_checkpoint_turn_interval=20)
    for _ in range(10):
        asyncio.run(env._apply_tool_call(state, "np_explore_level", {}))
        if _auto(state)["saved"]:
            break
    saved = _auto(state)["saved"]
    assert saved, (
        f"no periodic auto-checkpoint after "
        f"{state['structured_obs'].status.get('time')} game turns")
    assert saved[0]["trigger"].startswith("turn_")
    ids = _archive_ids(archive)
    assert ids, "the trigger fired but nothing reached the archive"

    # IT IS A REAL, RESTORABLE CHECKPOINT, audited the same way every other
    # restore in this tree is -- an archive row that cannot be resumed is worse
    # than no row, because the selector will choose it.
    meta = checkpoint_meta(archive / ids[0])
    assert meta["created_by"] == "auto"
    restored, rmeta = checkpoint_restore(archive / ids[0])
    assert rmeta["restore_fidelity"]["ok"]
    restored.close()


def test_auto_checkpoints_grow_the_archive_WHEN_THE_ROLLOUT_DESCENDS(tmp_path):
    """The pilot's exact scenario, with the fix: a real descent, real growth.

    THE DESCENT IS REAL AND THE PATH IS REAL. `goto_depth` is the engine's own
    level transition -- the same one a staircase runs, used here because
    walking a seed until it happens to find stairs makes a launch gate that
    costs minutes and can time out. Everything AFTER it is production code:
    the hook runs inside `_apply_tool_call`, off the same shaped observation
    the model is served, and the checkpoint is written by the same
    `checkpoint_save` the `save` skill calls.

    `save` is never called. The pilot ended this scenario -- 187 calls,
    Dlvl 1 -> 3 -- with `new_checkpoints: []` and a one-row archive.
    """
    archive = tmp_path / "archive"
    env, state, _ = _env_with(tmp_path, checkpoint_archive=str(archive))
    inner = state["env"]
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))  # seed
    start = int(state["structured_obs"].status.get("depth") or 1)
    assert _archive_ids(archive) == [], "nothing may be saved before the descent"

    # ESC first: a pending combat `--More--` eats the wait keystroke
    # `goto_depth` uses to run its deferred goto, and the transition would
    # silently not happen (test_persistent_checkpoint.py:170-175).
    from nethack_harness.checkpoints import _engine_of, _raw_of
    _raw_of(_engine_of(inner)).step(27)
    _raw_of(_engine_of(inner)).goto_depth(start + 1)
    # The model's next call is what the harness sees the new depth on -- and
    # right after a level transition the game is commonly parked on a
    # `--More--`, which is the case the deferral exists for. Give it a few
    # quiescent calls, exactly as a real rollout would have.
    for _ in range(6):
        asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))
        if _auto(state)["saved"]:
            break

    auto = _auto(state)
    assert int(state["structured_obs"].status.get("depth")) == start + 1
    assert auto["saved"], (
        f"the rollout descended to Dlvl "
        f"{state['structured_obs'].status.get('depth')} and the archive is "
        f"still empty -- this is the pilot's failure, unfixed. "
        f"pending={auto['pending']} errors={auto['errors']}")
    entry = [x for x in auto["saved"] if x["trigger"].startswith("level_entry_")]
    assert entry, f"no level-entry checkpoint among {auto['saved']}"
    assert entry[0]["dlvl"] == start + 1

    ids = _archive_ids(archive)
    assert ids, "auto-checkpoint reported a save that is not in the archive"
    metas = [checkpoint_meta(archive / i) for i in ids]
    # The player never called `save`. That is the whole point.
    assert all(mt["created_by"] == "auto" for mt in metas), \
        [mt["created_by"] for mt in metas]
    assert any(mt["dlvl"] == start + 1 for mt in metas)

    # AND IT IS A CHECKPOINT A LATER ATTEMPT CAN ACTUALLY RESUME, audited the
    # way every restore in this tree is. An archive row that will not restore
    # is worse than no row, because the selector will choose it.
    deep = next(i for i, mt in zip(ids, metas) if mt["dlvl"] == start + 1)
    restored, rmeta = checkpoint_restore(archive / deep)
    assert rmeta["restore_fidelity"]["ok"]
    assert int(rmeta["dlvl"]) == start + 1
    restored.close()


def test_a_trigger_blocked_by_a_pending_prompt_is_DEFERRED_not_dropped(tmp_path):
    """The guard interaction the design note called out, tested directly.

    `checkpoint_save` refuses while the game is parked on a `--More--`, and it
    is right to: meta.json is built from blstats while the bundle serializes
    the live heap, and over a deferred level transition the two disagree
    (measured: meta said Dlvl 4 for a state that restores to Dlvl 2). But a
    `--More--` is MOST likely exactly when the most important trigger fires,
    because descending prints one. Dropping the save there would systematically
    lose the checkpoints the frontier is made of.
    """
    import nethack as mod
    from nethack_harness.checkpoints import CheckpointSavepointError

    archive = tmp_path / "archive"
    archive.mkdir()
    env, state, _ = _env_with(tmp_path, checkpoint_archive=str(archive))
    asyncio.run(env._apply_tool_call(state, "search", {"times": 1}))  # seed

    calls = {"n": 0}
    real = mod._maybe_auto_checkpoint.__globals__  # noqa: SLF001

    import nethack_harness.checkpoints as ckmod
    original = ckmod.checkpoint_save

    def refusing(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CheckpointSavepointError("refusing: --More-- is up")
        return original(*a, **kw)

    ckmod.checkpoint_save = refusing
    try:
        auto = _auto(state)
        auto["pending"].append(("level_entry_d2", "entered Dlvl 2"))
        mod._maybe_auto_checkpoint(state)
        assert auto["deferrals"] == 1
        assert [p[0] for p in auto["pending"]] == ["level_entry_d2"], (
            "a save refused because of a pending prompt was DROPPED; the "
            "checkpoints that matter most are the ones taken at level entry, "
            "which is when a --More-- is most likely to be up")
        assert auto["saved"] == []

        # ...and it lands on the next quiescent call, still labelled with the
        # trigger that earned it.
        mod._maybe_auto_checkpoint(state)
        assert auto["pending"] == []
        assert len(auto["saved"]) == 1
        assert auto["saved"][0]["trigger"] == "level_entry_d2"
        assert _archive_ids(archive)
    finally:
        ckmod.checkpoint_save = original


def test_auto_checkpoints_appear_in_the_turn_trace(tmp_path):
    """`Trace.metrics` is written in the driver process from `NetHackState`,
    which never sees the env's state dict. The turn NDJSON is the only place
    the harness can report what its own auto-saves did.

    TWO PROPERTIES, and the second is a constraint this tree imposes on every
    optional turn-record field. (1) On a rollout WITH an archive the block is
    always present and carries `enabled`, so "off" is distinguishable from "on
    and saved nothing". (2) On a rollout WITHOUT one -- every arm outside E16,
    including the frozen E14/E15 controls this tree diffs against -- the key is
    ABSENT, so their traces stay byte-identical."""
    archive = tmp_path / "archive"
    env, state, trace_dir = _env_with(tmp_path,
                                      checkpoint_archive=str(archive),
                                      auto_checkpoint_turn_interval=20)
    for _ in range(6):
        asyncio.run(env._apply_tool_call(state, "np_explore_level", {}))
    records = [json.loads(l) for f in trace_dir.glob("*.ndjson")
               for l in f.read_text().splitlines() if l.strip()]
    assert records
    blocks = [r.get("auto_checkpoint") for r in records]
    assert all(b is not None for b in blocks), (
        "auto-checkpoint bookkeeping is missing from the turn trace")
    assert blocks[-1]["enabled"] is True

    env2, state2, dir2 = _env_with(tmp_path / "off")
    asyncio.run(env2._apply_tool_call(state2, "search", {"times": 1}))
    recs2 = [json.loads(l) for f in dir2.glob("*.ndjson")
             for l in f.read_text().splitlines() if l.strip()]
    # NO ARCHIVE -> NO KEY. This is what keeps every non-E16 arm's turn trace
    # byte-identical to what it wrote before this feature existed.
    assert recs2 and "auto_checkpoint" not in recs2[-1]

    # ...but an E16 rollout that DISABLED it still says so, rather than looking
    # like an arm that never had the feature.
    env3, state3, dir3 = _env_with(tmp_path / "disabled",
                                   checkpoint_archive=str(tmp_path / "arch3"),
                                   auto_checkpoint=False)
    asyncio.run(env3._apply_tool_call(state3, "search", {"times": 1}))
    recs3 = [json.loads(l) for f in dir3.glob("*.ndjson")
             for l in f.read_text().splitlines() if l.strip()]
    assert recs3 and recs3[-1]["auto_checkpoint"]["enabled"] is False
    assert recs3[-1]["auto_checkpoint"]["saved"] == []


def test_the_trigger_policy_without_an_engine():
    """The three conditions, and the watermarks that stop them re-firing."""
    auto = {"seen_dlvl": {1}, "last_xl": 1, "last_periodic_turn": 0,
            "interval": 150}
    T = m._auto_checkpoint_triggers

    assert T(auto, 1, 1, 10) == []                     # nothing happened
    assert [t[0] for t in T(auto, 2, 1, 20)] == ["level_entry_d2"]
    assert T(auto, 2, 1, 30) == []                     # not again for Dlvl 2
    assert [t[0] for t in T(auto, 2, 2, 40)] == ["level_up_xl2"]
    assert T(auto, 2, 2, 60) == []
    assert [t[0] for t in T(auto, 2, 2, 160)] == ["turn_160"]
    # The clock advances by WHOLE intervals, so one 400-turn call does not
    # reset the phase to an arbitrary point.
    assert auto["last_periodic_turn"] == 150
    assert [t[0] for t in T(auto, 2, 2, 460)] == ["turn_460"]
    assert auto["last_periodic_turn"] == 450
    # Going back UP a level is not a new level, and losing XL is not a level-up.
    assert T(auto, 1, 2, 460) == []
    assert T(auto, 1, 1, 460) == []
