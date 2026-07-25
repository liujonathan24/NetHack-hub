import pathlib, stat, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.cli_harness_eval.workspace import build_workspace


def test_builds_all_three_affordances(tmp_path):
    ws = build_workspace(tmp_path / "ws", objective="Descend as deep as you can.")
    assert (ws / "AGENTS.md").exists()
    assert (ws / "CLAUDE.md").read_text() == (ws / "AGENTS.md").read_text()
    assert len(list((ws / "wiki").glob("*.md"))) >= 100
    assert "Descend" in (ws / "memory" / "objective.md").read_text()


def test_wiki_pages_are_read_only(tmp_path):
    # Load-bearing global constraint: an arm must not be able to corrupt its
    # own reference corpus mid-run and diverge from the other arms. A
    # refactor that silently drops the chmod(0o444) must fail this test.
    ws = build_workspace(tmp_path / "ws", objective="x")
    pages = list((ws / "wiki").glob("*.md"))
    assert pages
    for page in pages:
        assert stat.S_IMODE(page.stat().st_mode) == 0o444


def test_agents_md_carries_the_system_prompt(tmp_path):
    ws = build_workspace(tmp_path / "ws", objective="x")
    assert "STRATEGY: DESCEND ASAP" in (ws / "AGENTS.md").read_text()


def test_no_map_is_seeded(tmp_path):
    ws = build_workspace(tmp_path / "ws", objective="x")
    assert not (ws / "map").exists()
    assert not list(ws.glob("*map*"))


def test_rebuild_wipes_memory_despite_read_only_wiki_files(tmp_path):
    # memory/ must be wiped per rollout so a CLI arm can't carry cross-seed
    # notes across rollouts (a protocol leak the harness's fresh Journal
    # doesn't have). The wiki pages from the first build are chmod 0o444, so
    # this also exercises that rmtree-over-read-only-files doesn't blow up.
    dest = tmp_path / "ws"
    build_workspace(dest, objective="first rollout's notes")
    (dest / "memory" / "scratch.md").write_text("leftover notes from rollout 1")

    ws = build_workspace(dest, objective="second rollout")

    assert not (ws / "memory" / "scratch.md").exists()
    assert (ws / "memory" / "objective.md").read_text() == "# Objective\n\nsecond rollout\n"
    assert len(list((ws / "wiki").glob("*.md"))) >= 100
