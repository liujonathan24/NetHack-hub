"""What `merge_harness_stores.py` does past the happy path.

The 30/30 recovery run proved the tool reassembles five *disjoint* private
stores. Disjoint is the easy case. This file pins the four cases where a merge
can quietly lose or invent a rollout's work, because none of them shows up as a
failed cell:

  * two rollouts editing the SAME entry id (who wins, and is the loser recorded
    even when the winner was already in place?),
  * an entry a player DELETED coming back from canonical,
  * a private store that is corrupt or truncated -- the runtime loader treats
    that as an empty store, and so would a merge that reused it,
  * an empty canonical, i.e. round 1.

Stores are authored through `rlm.harness` itself, in Prime Agent's kernel venv,
so the on-disk shape under test is the one the player actually writes -- not a
hand-rolled dict that could drift from it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
KERNEL_PY = pathlib.Path("/root/.prime/agent/kernel-venv/bin/python")

_spec = importlib.util.spec_from_file_location(
    "merge_harness_stores", REPO / "tools/cli_harness_eval/merge_harness_stores.py"
)
mhs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mhs)

pytestmark = pytest.mark.skipif(
    not KERNEL_PY.exists(), reason=f"Prime Agent kernel venv not present at {KERNEL_PY}"
)


def _author(body: str) -> None:
    """Run `body` in the kernel venv with `get_harness_state` in scope.

    A subprocess, not an import: `rlm` lives in Prime Agent's 3.11 kernel venv,
    which is not the venv this suite runs in -- and that separation is the point,
    since the player writes from the kernel and the merge runs from the runner.
    """
    src = textwrap.dedent(
        """
        import json, pathlib, shutil, time
        from rlm.harness import get_harness_state
        def store(d):
            return get_harness_state(state_dir=pathlib.Path(d), global_=True)
        """
    ) + textwrap.dedent(body)
    proc = subprocess.run(
        [str(KERNEL_PY), "-c", src], capture_output=True, text=True, cwd=REPO
    )
    assert proc.returncode == 0, f"authoring failed:\n{proc.stdout}\n{proc.stderr}"


def _entries(path: pathlib.Path, kind: str = "memory") -> dict:
    return json.loads((path / "harness_state.json").read_text())["entries"][kind]


def _rollout(root: pathlib.Path, trace_id: str) -> pathlib.Path:
    """The path layout the harness's copy-merge mode creates and the merge tool
    reads: `<install-dir>/agent-<trace id>/harness/`."""
    d = root / f"agent-{trace_id}" / "harness"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- (a) same id, two rollouts, different content ----------------------------


def test_the_newer_updated_at_wins_a_same_id_conflict(tmp_path):
    canonical = tmp_path / "canonical"
    a, b = _rollout(tmp_path, "aaa"), _rollout(tmp_path, "bbb")
    _author(f"""
        s = store({str(canonical)!r})
        s.create_memory("Elbereth", "Engrave Elbereth.", id="elbereth")
        s.save()
        for d in ({str(a)!r}, {str(b)!r}):
            shutil.copy({str(canonical)!r} + "/harness_state.json", d + "/harness_state.json")
        # `updated_at` has microsecond resolution, but sleep so the ordering is
        # from the clock and not from a tie the comparison would resolve by
        # falling back to canonical.
        sa = store({str(a)!r}); sa.update_memory("elbereth", "Elbereth", "A: dust does not work."); sa.save()
        time.sleep(0.01)
        sb = store({str(b)!r}); sb.update_memory("elbereth", "Elbereth", "B: wand of digging is better."); sb.save()
    """)

    a_at = _entries(a)["elbereth"]["updated_at"]
    b_at = _entries(b)["elbereth"]["updated_at"]
    assert b_at > a_at, (a_at, b_at)

    report = mhs.merge(canonical, [a / "harness_state.json", b / "harness_state.json"])

    kept = _entries(canonical)["elbereth"]
    assert kept["content"] == "B: wand of digging is better.", kept
    assert kept["updated_at"] == b_at
    # One conflict per losing store: A vs the seeded canonical, then B vs A.
    assert len(report["conflicts"]) == 2, report["conflicts"]
    assert [c["from"] for c in report["conflicts"]] == ["agent-aaa", "agent-bbb"]
    assert [c["kept"] for c in report["conflicts"]] == ["incoming", "incoming"]
    # The losing content is preserved in the report, not just the winner's.
    assert "A: dust does not work." in report["conflicts"][1]["canonical_content"]


def test_a_conflict_is_recorded_even_when_the_outcome_does_not_change(tmp_path):
    """The case a silent merge hides: the OLDER edit arrives second, canonical
    already holds the winner, nothing about the store changes -- and five seeds'
    disagreement would vanish. It must still be reported."""
    canonical = tmp_path / "canonical"
    new, old = _rollout(tmp_path, "new"), _rollout(tmp_path, "old")
    _author(f"""
        s = store({str(canonical)!r})
        s.create_memory("Elbereth", "Engrave Elbereth.", id="elbereth")
        s.save()
        for d in ({str(new)!r}, {str(old)!r}):
            shutil.copy({str(canonical)!r} + "/harness_state.json", d + "/harness_state.json")
        so = store({str(old)!r}); so.update_memory("elbereth", "Elbereth", "OLD opinion."); so.save()
        time.sleep(0.01)
        sn = store({str(new)!r}); sn.update_memory("elbereth", "Elbereth", "NEW opinion."); sn.save()
    """)

    # Newest store first, so the older one loses and changes nothing.
    report = mhs.merge(
        canonical, [new / "harness_state.json", old / "harness_state.json"]
    )

    assert _entries(canonical)["elbereth"]["content"] == "NEW opinion."
    assert report["updated"] == [{"kind": "memory", "id": "elbereth", "from": "agent-new"}]
    losing = [c for c in report["conflicts"] if c["kept"] == "canonical"]
    assert len(losing) == 1, report["conflicts"]
    assert losing[0]["from"] == "agent-old"
    assert losing[0]["incoming_content"] == "OLD opinion."
    assert losing[0]["canonical_content"] == "NEW opinion."
    assert losing[0]["incoming_updated_at"] < losing[0]["canonical_updated_at"]


def test_an_identical_entry_in_two_stores_is_unchanged_not_a_conflict(tmp_path):
    """Every rollout starts from a COPY of canonical, so most entries arrive
    N times untouched. Those must not be reported as disagreements or the report
    is noise."""
    canonical = tmp_path / "canonical"
    a, b = _rollout(tmp_path, "aaa"), _rollout(tmp_path, "bbb")
    _author(f"""
        s = store({str(canonical)!r})
        s.create_memory("Elbereth", "Engrave Elbereth.", id="elbereth")
        s.create_memory("Food", "Eat before Weak.", id="food")
        s.save()
        for d in ({str(a)!r}, {str(b)!r}):
            shutil.copy({str(canonical)!r} + "/harness_state.json", d + "/harness_state.json")
    """)

    report = mhs.merge(canonical, [a / "harness_state.json", b / "harness_state.json"])
    assert report["conflicts"] == []
    assert report["added"] == []
    assert report["unchanged"] == 4  # 2 entries x 2 stores
    assert report["totals"]["memory"] == 2


# -- (b) a player DELETES an entry -------------------------------------------


def test_an_entry_a_player_deleted_stays_deleted(tmp_path):
    """The arm has to be able to UNLEARN.

    Merge was a pure union keyed by (kind, id), so `delete_memory` in a private
    copy was silently reverted and the report showed nothing -- indistinguishable
    from a clean no-op. That matters twice over: "remove the wrong previous ones"
    is half the experiment, and the scaffold renders only 6 entries per kind, so
    an un-evictable store quietly truncates instead of improving.

    Deletion is detected by diffing a copy against the BASELINE it was seeded
    from: in the baseline, absent from the copy => deleted on purpose.
    """
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _author(f"""
        st = store({str(canonical)!r})
        st.create_memory(title='keep me', content='a good lesson', global_=True)
        st.create_memory(title='wrong', content='later disproved', global_=True)
        """)
    baseline = tmp_path / "baseline.json"
    baseline.write_text((canonical / "harness_state.json").read_text())

    private = _rollout(tmp_path, "t0")
    shutil.copy(canonical / "harness_state.json", private / "harness_state.json")
    _author(f"""
        st = store({str(private)!r})
        assert st.delete_memory('wrong', global_=True)
        """)

    report = mhs.merge(
        canonical, [private / "harness_state.json"], baseline)

    assert "wrong" not in _entries(canonical), "a deliberate deletion was reverted"
    assert "keep_me" in _entries(canonical)
    assert [d["id"] for d in report["deleted"]] == ["wrong"]
    assert report["deleted"][0]["from"] == "agent-t0"


def test_a_deletion_racing_an_edit_keeps_the_edit_and_reports_it(tmp_path):
    """One player deletes an entry while another rewrites it. Deleting what
    someone just rewrote would discard the newer evidence, so the edit wins --
    and the disagreement is recorded rather than resolved silently."""
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _author(f"""
        st = store({str(canonical)!r})
        st.create_memory(title='contested', content='v1', global_=True)
        """)
    baseline = tmp_path / "baseline.json"
    baseline.write_text((canonical / "harness_state.json").read_text())

    deleter, editor = _rollout(tmp_path, "del"), _rollout(tmp_path, "edit")
    for d in (deleter, editor):
        shutil.copy(canonical / "harness_state.json", d / "harness_state.json")
    _author(f"""
        st = store({str(deleter)!r})
        assert st.delete_memory('contested', global_=True)
        """)
    _author(f"""
        import time; time.sleep(1.1)
        st = store({str(editor)!r})
        st.update_memory('contested', title='contested',
                         content='v2 with new evidence', global_=True)
        """)

    report = mhs.merge(
        canonical,
        [deleter / "harness_state.json", editor / "harness_state.json"], baseline)

    assert "contested" in _entries(canonical)
    assert "v2 with new evidence" in _entries(canonical)["contested"]["content"]
    assert report["deleted"] == []
    assert len(report["delete_conflicts"]) == 1
    assert report["delete_conflicts"][0]["resolution"] == "kept the edit"

@pytest.mark.parametrize("damage", ["truncated", "empty", "trailing-garbage"])
def test_a_damaged_private_store_is_refused_loudly_not_treated_as_empty(tmp_path, damage):
    """`rlm.harness.load()` swallows a broken state file and returns an EMPTY
    state (`harness.py`: "Treat it as empty"). A merge that did the same would
    drop that rollout's entire contribution and still print a success line.
    """
    canonical = tmp_path / "canonical"
    good, bad = _rollout(tmp_path, "good"), _rollout(tmp_path, "bad")
    _author(f"""
        s = store({str(canonical)!r})
        s.create_memory("Elbereth", "Engrave Elbereth.", id="elbereth")
        s.save()
        for d in ({str(good)!r}, {str(bad)!r}):
            shutil.copy({str(canonical)!r} + "/harness_state.json", d + "/harness_state.json")
        sg = store({str(good)!r}); sg.create_memory("G", "good rollout's lesson", id="from_good"); sg.save()
        sb = store({str(bad)!r}); sb.create_memory("B", "bad rollout's lesson", id="from_bad"); sb.save()
    """)
    bad_file = bad / "harness_state.json"
    # `damage` is the parametrize LABEL; map it to the actual corruption.
    _DAMAGE = {
        "truncated": lambda t: t[: len(t) // 2],
        "empty": lambda t: "",
        "trailing-garbage": lambda t: t + "\n}}}not json",
    }
    bad_file.write_text(_DAMAGE[damage](bad_file.read_text()))

    before = (canonical / "harness_state.json").read_text()
    with pytest.raises(SystemExit) as exc:
        mhs.merge(
            canonical, [good / "harness_state.json", bad_file]
        )
    msg = str(exc.value)
    assert "unreadable" in msg and "Refusing to merge" in msg, msg
    assert str(bad_file) in msg, "the operator has to be told WHICH store"

    # Canonical must be untouched: a partial merge that kept only the good
    # store would look like a completed round.
    assert (canonical / "harness_state.json").read_text() == before
    assert "from_good" not in _entries(canonical)


def test_a_missing_private_store_is_the_one_case_that_is_not_an_error(tmp_path):
    """Absent != corrupt. A rollout that wrote nothing has an empty (or no)
    file, and `--run-dir` only lists stores that exist; a merge must not abort
    the whole round over one."""
    canonical = tmp_path / "canonical"
    _author(f"""
        s = store({str(canonical)!r})
        s.create_memory("Elbereth", "Engrave Elbereth.", id="elbereth")
        s.save()
    """)
    report = mhs.merge(canonical, [tmp_path / "agent-gone" / "harness" / "harness_state.json"])
    assert report["totals"]["memory"] == 1
    assert report["conflicts"] == [] and report["added"] == []


# -- (d) round 1: no canonical yet -------------------------------------------


def test_an_empty_canonical_absorbs_every_store_as_new(tmp_path):
    """Round 1: canonical does not exist. It must be created, and every private
    entry must land as `added` -- not be skipped because there was nothing to
    merge into."""
    canonical = tmp_path / "canonical"
    a, b = _rollout(tmp_path, "aaa"), _rollout(tmp_path, "bbb")
    _author(f"""
        sa = store({str(a)!r})
        sa.create_memory("A1", "a one", id="a1"); sa.create_prompt_note("A2", "a two", id="a2")
        sa.save()
        sb = store({str(b)!r})
        sb.create_memory("B1", "b one", id="b1")
        sb.save()
    """)
    assert not canonical.exists()

    report = mhs.merge(canonical, [a / "harness_state.json", b / "harness_state.json"])

    assert (canonical / "harness_state.json").exists()
    assert sorted(_entries(canonical)) == ["a1", "b1"]
    assert sorted(_entries(canonical, "prompt")) == ["a2"]
    assert len(report["added"]) == 3
    assert report["conflicts"] == [] and report["updated"] == []
    assert report["totals"] == {"prompt": 1, "memory": 2, "skill": 0, "subagent": 0}
    # The written file must be shaped so `rlm.harness` can load it back: an
    # empty-canonical round that produced a file the kernel rejects would read
    # as "the agent learned nothing" in round 2.
    data = json.loads((canonical / "harness_state.json").read_text())
    assert data["schema"] == 1
    assert set(data["entries"]) == set(mhs.KINDS)


def test_the_merged_file_round_trips_through_the_kernels_own_loader(tmp_path):
    """The consumer of the merged file is `rlm.harness.load()` in the next
    round's kernel, so assert against IT rather than against our own reader."""
    canonical = tmp_path / "canonical"
    a = _rollout(tmp_path, "aaa")
    _author(f"""
        sa = store({str(a)!r})
        sa.create_memory("Kick", "Kick locked doors, do not #force.", id="kick")
        sa.save()
    """)
    mhs.merge(canonical, [a / "harness_state.json"])

    out = tmp_path / "roundtrip.json"
    _author(f"""
        s = store({str(canonical)!r})
        got = s.get("memory", "kick")
        pathlib.Path({str(out)!r}).write_text(json.dumps(
            {{"title": got.title, "content": got.content}} if got else None))
    """)
    assert json.loads(out.read_text()) == {
        "title": "Kick",
        "content": "Kick locked doors, do not #force.",
    }


# -- the writer the merge tool replaces --------------------------------------


def test_the_merge_writes_atomically_and_leaves_no_tmp_behind(tmp_path):
    """The reason the tool exists: `rlm.harness.save()` is `open("w")` +
    `json.dump` with no tmp+rename, so a reader can see a half file. The merge
    writes `.tmp` then `replace`s."""
    canonical = tmp_path / "canonical"
    a = _rollout(tmp_path, "aaa")
    _author(f"""
        sa = store({str(a)!r}); sa.create_memory("A", "a", id="a1"); sa.save()
    """)
    mhs.merge(canonical, [a / "harness_state.json"])
    assert not (canonical / "harness_state.json.tmp").exists()
    assert sorted(p.name for p in canonical.iterdir()) == ["harness_state.json"]
