"""E16 `wiki` tool — a curated two-page knowledge base, read from disk.

WHAT THIS IS NOT
----------------
It is not a dump and not a retrieval system. ``nethack_harness/tools/wiki.py``
already ships a 2k-page snapshot behind ``wiki_lookup``/``wiki_search``; that
surface is part of other arms' frozen tool sets and is untouched here.

This module serves the SAME two curated pages E15's reflection agent read —
``why_do_i_keep_dying.md`` (~15.6 KB) and ``standard_strategy.md`` (~12.9 KB),
with their ``MANIFEST.json`` — because at ~28 KB total the whole corpus is
smaller than one BM25 index and the right retrieval algorithm is "read the
page". Provenance (nethackwiki.com extract API, retrieved 2026-08-25, per-file
sha) lives in the manifest and is copied into the run directory alongside the
pages, so a run is reproducible from its own output tree even if the source
worktree moves or changes.

THE THREE CALL SHAPES
---------------------
``wiki()``                 the table of contents: pages + their section headings
``wiki(page=...)``         one page, or ``wiki(page=..., section=...)``
``wiki(query=...)``        grep over section bodies; matching sections, headed

Every response is capped at :data:`RESPONSE_CHAR_CAP` characters and says so
when it truncates, because an uncapped page read is ~15 KB of prompt — several
times the whole observation — and would silently dominate the context budget
the rest of the surface was measured under.

PARSING
-------
The pages are MediaWiki extracts: ``# Title`` on line 1, then ``== H2 ==`` /
``=== H3 ===`` headings. Section titles are NOT unique (``standard_strategy``
has four ``Goals``), so every section also carries a PATH
(``The early game / Goals``) and ambiguous lookups are reported as ambiguous
rather than silently resolved to the first hit.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

#: Hard cap on the characters any one ``wiki`` response may return.
RESPONSE_CHAR_CAP = 2000

#: Appended (inside the cap) when a response was cut short.
TRUNCATION_MARKER = "\n[...truncated; narrow with section= or query=]"

MANIFEST_JSON = "MANIFEST.json"

#: ``== Heading ==`` / ``=== Heading ===`` — MediaWiki extract style.
_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def _norm(s: str) -> str:
    """Loose key for matching a page or section name a model typed."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


class Section:
    """One heading and the text under it, with its path from the page title."""

    __slots__ = ("title", "level", "path", "body")

    def __init__(self, title: str, level: int, path: str, body: str):
        self.title = title
        self.level = level
        self.path = path
        self.body = body

    def render(self) -> str:
        return f"== {self.path} ==\n{self.body}".rstrip()


class Page:
    """One curated wiki page: its title, its raw text, and its sections."""

    def __init__(self, page_id: str, title: str, text: str):
        self.id = page_id
        self.title = title
        self.text = text
        self.preamble, self.sections = _split_sections(title, text)

    def find_sections(self, name: str) -> list:
        """Sections whose title or path matches ``name`` (loose, case-free)."""
        key = _norm(name)
        if not key:
            return []
        exact = [s for s in self.sections
                 if _norm(s.title) == key or _norm(s.path) == key]
        if exact:
            return exact
        return [s for s in self.sections
                if key in _norm(s.path) or _norm(s.path).endswith(key)]

    def render(self) -> str:
        parts = [f"# {self.title}"]
        if self.preamble:
            parts.append(self.preamble)
        parts.extend(s.render() for s in self.sections)
        return "\n\n".join(p for p in parts if p).rstrip()


def _split_sections(title: str, text: str) -> tuple:
    """``(preamble, [Section, ...])`` for one MediaWiki-extract page."""
    lines = text.splitlines()
    preamble: list[str] = []
    sections: list[Section] = []
    # Stack of (level, title) for the path of the section being filled.
    stack: list[tuple] = []
    cur: Optional[list] = None  # [level, title, path, [body lines]]

    def close():
        if cur is not None:
            sections.append(Section(cur[1], cur[0], cur[2],
                                    "\n".join(cur[3]).strip()))

    for line in lines:
        if line.startswith("# ") and not sections and cur is None and not preamble:
            continue  # the page title line
        m = _HEADING_RE.match(line)
        if m:
            close()
            level = len(m.group(1))
            name = m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            path = " / ".join([t for _, t in stack] + [name])
            stack.append((level, name))
            cur = [level, name, path, []]
            continue
        (cur[3] if cur is not None else preamble).append(line)
    close()
    return "\n".join(preamble).strip(), sections


def cap(text: str, limit: int = RESPONSE_CHAR_CAP) -> str:
    """Trim ``text`` to ``limit`` characters INCLUDING the truncation marker.

    The marker is part of the budget on purpose: a cap that a truncation notice
    can push past is not a cap, and the whole point of this one is that a
    ``wiki`` response can never quietly become the largest thing in the prompt.
    """
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(TRUNCATION_MARKER))
    return text[:keep].rstrip() + TRUNCATION_MARKER


class WikiKB:
    """The curated pages under one directory, parsed once.

    ``root`` holds the ``.md`` pages plus ``MANIFEST.json``. Missing manifest is
    tolerated (the pages are still readable); a missing directory is not — the
    caller gets an explicit error rather than an empty, silently useless tool.
    """

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"wiki knowledge base not found: {self.root}")
        self.manifest = {}
        man = self.root / MANIFEST_JSON
        if man.is_file():
            try:
                self.manifest = json.loads(man.read_text())
            except Exception:
                self.manifest = {}
        titles = {}
        for entry in self.manifest.get("pages", []) or []:
            if entry.get("file"):
                titles[entry["file"]] = entry.get("title", "")
        self.pages: list[Page] = []
        for path in sorted(self.root.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            title = titles.get(path.name) or _first_title(text) or path.stem
            self.pages.append(Page(path.stem, title, text))
        if not self.pages:
            raise FileNotFoundError(f"wiki knowledge base has no pages: {self.root}")

    # -- lookup ------------------------------------------------------------ #

    def find_page(self, name: str) -> Optional[Page]:
        key = _norm(name)
        if not key:
            return None
        for p in self.pages:  # exact on id or title
            if _norm(p.id) == key or _norm(p.title) == key:
                return p
        for p in self.pages:  # then substring, either direction
            if key in _norm(p.id) or key in _norm(p.title) or _norm(p.title) in key:
                return p
        return None

    # -- the three response shapes ----------------------------------------- #

    def toc(self) -> str:
        """Pages and their section headings — what ``wiki()`` with no args says."""
        out = ["Available wiki pages (call wiki(page=...) or wiki(query=...)):"]
        for p in self.pages:
            out.append(f"\n[{p.id}] {p.title}")
            for s in p.sections:
                out.append(f"  - {s.path}")
        out.append("\nwiki(query='wraith') greps every section body.")
        return cap("\n".join(out))

    def read(self, page: str, section: Optional[str] = None) -> str:
        p = self.find_page(page)
        if p is None:
            have = ", ".join(x.id for x in self.pages)
            return f"No wiki page {page!r}. Available pages: {have}."
        if not section:
            return cap(p.render())
        hits = p.find_sections(section)
        if not hits:
            have = "; ".join(s.path for s in p.sections)
            return cap(f"No section {section!r} in {p.title!r}. Sections: {have}")
        if len(hits) > 1:
            body = "\n\n".join(h.render() for h in hits)
            note = (f"[{len(hits)} sections match {section!r} in {p.title!r}: "
                    + "; ".join(h.path for h in hits) + "]\n")
            return cap(note + body)
        return cap(f"[{p.title}]\n" + hits[0].render())

    def search(self, query: str) -> str:
        """Sections whose heading or body contains ``query`` (case-free)."""
        q = (query or "").strip().lower()
        if not q:
            return "wiki(query=...) needs a non-empty query."
        hits = []
        for p in self.pages:
            for s in p.sections:
                hay = f"{s.path}\n{s.body}".lower()
                if q in hay:
                    hits.append((p, s, hay.count(q)))
        if not hits:
            return (f"No wiki section matches {query!r}. "
                    f"Call wiki() for the list of pages and sections.")
        # Most on-topic first: heading matches beat body matches, then count.
        hits.sort(key=lambda t: (q not in t[1].path.lower(), -t[2]))
        # The index line lists EVERY hit before any body, so a response the cap
        # truncates still tells the model what else it can ask for by name.
        parts = [f"{len(hits)} section(s) match {query!r}: "
                 + "; ".join(f"{p.id}:{s.path}" for p, s, _ in hits)]
        for p, s, _n in hits:
            parts.append(f"\n[{p.id}] {s.render()}")
        return cap("\n".join(parts))

    # -- provenance -------------------------------------------------------- #

    def file_hashes(self) -> dict:
        """``{filename: sha256}`` for every file in the KB, for provenance.json."""
        import hashlib
        out = {}
        for path in sorted(self.root.iterdir()):
            if path.is_file():
                out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out


def _first_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def install(src, dest) -> dict:
    """Copy the knowledge base into a run directory. Returns ``{name: sha256}``.

    The design doc asks for this explicitly: the pages are copied into the E16
    run directory at launch so the run stays reproducible if the source
    worktree moves or is edited. Copy is byte-for-byte and the hashes go into
    ``provenance.json``, so a later analysis can prove which bytes were served.
    """
    import hashlib
    import shutil

    src, dest = Path(src), Path(dest)
    if not src.is_dir():
        raise FileNotFoundError(f"wiki source not found: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in sorted(src.iterdir()):
        if not path.is_file():
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        hashes[path.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


__all__ = [
    "MANIFEST_JSON",
    "RESPONSE_CHAR_CAP",
    "TRUNCATION_MARKER",
    "Page",
    "Section",
    "WikiKB",
    "cap",
    "install",
]
