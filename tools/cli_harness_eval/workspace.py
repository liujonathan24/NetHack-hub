"""Build the sandbox workspace handed to CLI-agent arms.

Each item is the filesystem counterpart of an affordance the native harness
supplies through prompt and tools, so the arms are capability-matched:
  AGENTS.md   <- SYSTEM_PROMPT
  wiki/       <- wiki_lookup / wiki_search
  memory/     <- Journal (objective + notes)
No map is seeded: it would duplicate the observation channel and destroy the
"does the scaffold build its own map notes?" measurement.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

_ENV = Path(__file__).resolve().parents[2] / "environments" / "nethack"
_SNAPSHOT = _ENV / "wiki" / "snapshot.json"
# Filenames each CLI looks for; all get the same bytes.
_PROMPT_FILES = ("AGENTS.md", "CLAUDE.md")


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "page"


def _force_rmtree(path: Path) -> None:
    """rmtree that tolerates read-only wiki pages from a prior build.

    shutil.rmtree only needs the *parent* directory to be writable to unlink
    a read-only file on Linux, so this is normally unnecessary — but we chmod
    everything writable first anyway so a rebuild never depends on that
    platform detail (and doesn't go flaky under a stricter umask/ACL setup).
    """
    for p in path.rglob("*"):
        p.chmod(0o755 if p.is_dir() else 0o644)
    shutil.rmtree(path)


def build_workspace(dest: Path, *, objective: str) -> Path:
    import sys
    sys.path.insert(0, str(_ENV))
    from nethack_harness.prompt.rendering import SYSTEM_PROMPT

    dest = Path(dest)
    if dest.exists():
        _force_rmtree(dest)           # memory/ is wiped per rollout
    (dest / "wiki").mkdir(parents=True)
    (dest / "memory").mkdir()

    primer = SYSTEM_PROMPT + (
        "\n\n=== WORKSPACE ===\n"
        "`wiki/` holds NetHack reference pages (read-only) — grep it.\n"
        "`memory/` is yours: keep notes there across turns. `memory/objective.md`"
        " is your goal.\n"
    )
    for name in _PROMPT_FILES:
        (dest / name).write_text(primer)

    pages = json.loads(_SNAPSHOT.read_text())
    for page in pages:
        path = dest / "wiki" / f"{_slug(page['title'])}.md"
        path.write_text(f"# {page['title']}\n\n{page['body']}\n")
        path.chmod(0o444)

    (dest / "memory" / "objective.md").write_text(f"# Objective\n\n{objective}\n")
    return dest
