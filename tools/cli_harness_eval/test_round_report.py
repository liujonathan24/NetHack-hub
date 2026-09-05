"""The round report is the reflection pass's only view of its own history.

Each test corresponds to a way this got it wrong on real data (E13-v2,
run v2-corpus6to10-r1, 2026-08-25).
"""
import json, pathlib, importlib.util

_MOD = pathlib.Path(__file__).resolve().parent / "round_report.py"
_spec = importlib.util.spec_from_file_location("round_report", _MOD)
rr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rr)


def _round(tmp, name, rows, store=None):
    cell = tmp / name / "corpus__prime_agent"
    (cell / "turns").mkdir(parents=True)
    with (cell / "traces.jsonl").open("w") as fh:
        for seed, dlvl, xl, calls, stop in rows:
            fh.write(json.dumps({
                "task": {"data": {"seed": seed, "idx": 99}},
                "metrics": {"max_dlvl_reached": dlvl, "max_xp_level": xl,
                            "skill_calls": calls, "died": stop == "game_over"},
                "stop_condition": stop}) + "\n")
    if store is not None:
        (cell / "harness_state.before.json").write_text(
            json.dumps({"entries": {"memory": store}}))
    return cell


def _entry(eid, title, content, version=1):
    return {eid: {"id": eid, "kind": "memory", "title": title,
                  "content": content, "version": version}}


def _seeds(rep):
    return {r["seed"] for rows in rep["per_seed"].values() for r in rows}


def test_delta_is_seed_matched_not_mean_of_means(tmp_path):
    """A partial round must be compared seed-for-seed.

    Averaging round 3's single finished seed against round 1's full five read
    as -4.20 dlvl and halted a live run on arrival order alone.
    """
    _round(tmp_path, "round1", [(6, 4, 2, 100, "game_over"), (7, 12, 5, 150, "game_over")])
    _round(tmp_path, "round2", [(7, 12, 5, 150, "game_over")])
    d = rr.build(tmp_path)["rounds"][1]["vs_round1"]
    assert d["shared_seeds"] == 1
    assert d["d_dlvl"] == 0.0


def test_idx_is_not_used_as_the_seed(tmp_path):
    """`idx` is the row index; using it labels a corpus cell as the held-out set."""
    _round(tmp_path, "round1", [(6, 4, 2, 100, "game_over"), (7, 12, 5, 150, "game_over")])
    assert _seeds(rr.build(tmp_path)) == {6, 7}


def test_attribution_only_covers_rounds_an_entry_changed_in(tmp_path):
    """An entry carried unchanged through a bad round is not a candidate for it."""
    old = _entry("keep", "carried", "unchanged text")
    _round(tmp_path, "round1", [(6, 8, 4, 100, "game_over")], store={})
    _round(tmp_path, "round2", [(6, 8, 4, 100, "game_over")], store=old)
    new = {**old, **_entry("fresh", "new rule", "brand new text")}
    _round(tmp_path, "round3", [(6, 2, 1, 100, "game_over")], store=new)
    rep = rr.build(tmp_path)
    byid = {e["id"]: e for e in rep["entries"]}
    assert byid["keep"]["changed_before"] == ["round2"]
    assert byid["fresh"]["changed_before"] == ["round3"]
    r3 = [r for r in rep["regressions"] if r["round"] == "round3"]
    assert r3, "round3 lost 6 dlvl and must be reported"
    assert {c["id"] for c in r3[0]["candidates"]} == {"fresh"}
    assert r3[0]["discriminating"] is True


def test_simultaneous_changes_are_not_reported_as_separate_verdicts(tmp_path):
    """Entries added at one boundary cannot be told apart by that round."""
    _round(tmp_path, "round1", [(6, 8, 4, 100, "game_over")], store={})
    both = {**_entry("a", "A", "text a"), **_entry("b", "B", "text b")}
    _round(tmp_path, "round2", [(6, 2, 1, 100, "game_over")], store=both)
    rep = rr.build(tmp_path)
    r2 = [r for r in rep["regressions"] if r["round"] == "round2"][0]
    assert r2["n_changed"] == 2 and r2["discriminating"] is False
    assert "CANNOT tell them apart" in rr.render(rep)


def test_rewritten_content_is_detected_even_when_the_title_is_stale(tmp_path):
    """update_memory keeps the original title, so presence-keying misses edits."""
    _round(tmp_path, "round1", [(6, 8, 4, 100, "game_over")], store={})
    _round(tmp_path, "round2", [(6, 8, 4, 100, "game_over")],
           store=_entry("m", "Rest before descending", "rest to full HP first"))
    _round(tmp_path, "round3", [(6, 2, 1, 100, "game_over")],
           store=_entry("m", "Rest before descending", "do NOT descend past XL x 2", 2))
    m = {e["id"]: e for e in rr.build(tmp_path)["entries"]}["m"]
    assert m["changed_before"] == ["round2", "round3"]
    assert "XL x 2" in m["content"]


def test_xl_falls_back_to_turn_files_for_pre_change_runs(tmp_path):
    """Runs recorded before max_xp_level was published must still score."""
    cell = tmp_path / "round1" / "corpus__prime_agent"
    (cell / "turns").mkdir(parents=True)
    (cell / "traces.jsonl").write_text(json.dumps({
        "task": {"data": {"seed": 6}},
        "metrics": {"max_dlvl_reached": 8, "skill_calls": 100},
        "stop_condition": "game_over"}) + "\n")
    (cell / "turns" / "6_1_1.ndjson").write_text(
        json.dumps({"status": {"experience_level": 4}}) + "\n")
    rows = rr.build(tmp_path)["per_seed"]["round1"]
    assert rows[0]["xl"] == 4
    assert rows[0]["balrog_pct"] is not None
