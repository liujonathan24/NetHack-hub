"""The primitive floor. FROZEN -- this file is re-materialised every rollout.

`netplay/*` is your own editable code; `netplay/_base.py` is not. It is the
only sanctioned door to the game, and every edit you make to it is reverted at
the start of the next episode. Import it, build on it, do not rewrite it.

The six primitives below are the irreducible surface: one keystroke, one map
read, and the four survival calls that cannot be expressed as keystrokes. Every
composite policy in this package is built from these and nothing else.

    from netplay import _base

    obs = await _base.screen()          # full map, costs no game time
    obs = await _base.press(">")        # one keystroke

Parsing helpers (`status`, `features`, `monsters`, `grid`) are provided frozen
because every composite needs them and duplicating them would be worse than
sharing them. They are a convenience, not a constraint -- a composite is free
to parse the raw observation itself.
"""

# Maintainer notes (not model-facing).
#
# Beacons. Every primitive here writes a correlation record to the call log
# before and after it dispatches. The log is CLIENT-side bookkeeping only: the
# published MCP tool schemas are frozen and must never be changed for logging,
# so the correlation id rides our own record, never the call. At rollout
# teardown the harness reconciles this log's call count against the count the
# tool server recorded server-side; a composite that produces observations
# without a matching server-side call is fabricating (see SPEC §6.2 rail 2).
#
# The log path arrives in `NETPLAY_CALL_LOG`. When it is unset -- unit tests,
# or an interactive kernel -- logging degrades to a no-op rather than raising:
# a beacon failure must never break the game.

from __future__ import annotations

import itertools
import json
import os
import re
import time
from typing import Any

# Bound PRIVATELY. `import nethack` would make `_base.nethack` a public handle
# on the MCP shim, and any composite could then reach the game through it
# without passing a beacon -- defeating the reconciliation this module exists
# to make possible. The gate also rejects attribute access to it by name.
import nethack as _nethack

__all__ = [
    "press", "screen", "rest", "pray", "apply", "search", "kick",
    "status", "features", "monsters", "grid", "position", "tile",
    "messages", "prompt_open",
    "check", "call_count",
]

_counter = itertools.count(1)
_calls = 0


def _log(record: dict[str, Any]) -> None:
    """Append one beacon record. Never raises -- see the maintainer note."""
    path = os.environ.get("NETPLAY_CALL_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        pass


async def _call(tool: str, **kwargs: Any) -> str:
    """Dispatch one MCP tool call, bracketed by beacons."""
    global _calls
    cid = next(_counter)
    _calls += 1
    _log({"cid": cid, "tool": tool, "args": kwargs, "phase": "call",
          "t": time.time()})
    result = await getattr(_nethack, tool)(**kwargs)
    text = result if isinstance(result, str) else str(result)
    _log({"cid": cid, "tool": tool, "phase": "result", "t": time.time(),
          "len": len(text)})
    return text


def call_count() -> int:
    """MCP calls this kernel has made through the floor, this episode."""
    return _calls


# ---------- the six primitives ----------

async def press(key: str) -> str:
    """Press ONE key: a letter, digit, punctuation, or ESC/SPACE/ENTER."""
    return await _call("np_press_key", key=key)


async def screen() -> str:
    """The full map and surroundings. Costs no game time."""
    return await _call("request_map")


async def rest(count: int = 5) -> str:
    """Rest in place `count` moves, or until something happens."""
    return await _call("np_rest", count=count)


async def pray() -> str:
    """Pray. Saves you at critical HP or starving -- rarely. Do not spam."""
    return await _call("np_pray")


async def apply(item_letter: str | None = None) -> str:
    """Apply (use) a tool. Omit `item_letter` to be prompted."""
    if item_letter is None:
        return await _call("np_apply")
    return await _call("np_apply", item_letter=item_letter)


async def search(times: int = 1) -> str:
    """Search adjacent tiles for hidden doors/passages, `times` tries (1-20)."""
    return await _call("search", times=times)


async def kick(x: int, y: int) -> str:
    """Kick the tile at (x, y) -- a locked door, a chest.

    The one primitive that is not a single action. Kicking is Ctrl-D, which the
    keystroke floor cannot send (`np_press_key` rejects control characters), so
    the only route to it is `np_kick`, which walks adjacent to the target
    first. That bundled walk is why kicking stays a primitive: unlike move_to
    or melee, its core cannot be rebuilt from press + screen.
    """
    return await _call("np_kick", x=x, y=y)


# ---------- frozen parsing helpers ----------

_SECTION = re.compile(r"^=== ([A-Z][A-Z ]*?)(?: \(.*?\))? ===\s*(.*)$")
_XY = re.compile(r"\((\d+),\s*(\d+)\)")


def _sections(obs: str) -> dict[str, str]:
    """Split an observation into its `=== NAME ===` sections.

    A section's text is whatever follows the header on the same line plus every
    line up to the next header, so both the inline form (`=== ADJACENT === N=.`)
    and the block form (`=== MAP ===` then rows) parse the same way.
    """
    out: dict[str, list[str]] = {}
    current = None
    for line in obs.splitlines():
        m = _SECTION.match(line.strip()) if line.strip().startswith("===") else None
        if m:
            current = m.group(1).strip()
            out.setdefault(current, [])
            if m.group(2).strip():
                out[current].append(m.group(2).strip())
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v).strip("\n") for k, v in out.items()}


def status(obs: str) -> dict[str, Any]:
    """The `=== STATUS ===` line as a dict.

    Keys are lowercased NetHack status names (`hp`, `maxhp`, `ac`, `dlvl`,
    `turn`, `xp`, `pos`). Missing keys are simply absent -- a caller that needs
    a default should say so at the call site.
    """
    text = _sections(obs).get("STATUS", "")
    out: dict[str, Any] = {}
    hp = re.search(r"HP:\s*(\d+)/(\d+)", text)
    if hp:
        out["hp"], out["maxhp"] = int(hp.group(1)), int(hp.group(2))
    for key, pat in (("ac", r"AC:\s*(-?\d+)"), ("dlvl", r"Dlvl:\s*(\d+)"),
                     ("turn", r"Turn:\s*(\d+)"), ("xp", r"XP:\s*(\d+)"),
                     ("gold", r"\$:\s*(\d+)")):
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    pos = re.search(r"Pos:\s*\((\d+),\s*(\d+)\)", text)
    if pos:
        out["pos"] = (int(pos.group(1)), int(pos.group(2)))
    return out


def position(obs: str) -> tuple[int, int] | None:
    """Your (x, y) in the MAP frame, or None if the status line is absent."""
    return status(obs).get("pos")


def messages(obs: str) -> str:
    """The `=== MESSAGES ===` block, stripped. Empty when nothing happened."""
    return _sections(obs).get("MESSAGES", "").strip()


def prompt_open(obs: str) -> bool:
    """True when the game is waiting on you -- a [ynq], a menu, a --More--.

    Auto-dismiss is OFF, so the game clock is FROZEN until you answer. If an
    observation looks unchanged, check this before pressing anything else.
    """
    text = messages(obs)
    return bool(re.search(r"\[[^\]]*\]|--More--|\?\s*\[", text))


def features(obs: str) -> dict[str, list[tuple[int, int]]]:
    """`=== VISIBLE FEATURES ===` as {label: [(x, y), ...]}.

    Labels are the harness's own wording with the coordinates stripped, e.g.
    `"stairs DOWN"`, `"door (closed)"`. Matching on a substring (`"stairs
    DOWN" in label`) is more robust than equality.
    """
    text = _sections(obs).get("VISIBLE FEATURES", "")
    out: dict[str, list[tuple[int, int]]] = {}
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        coords = [(int(a), int(b)) for a, b in _XY.findall(chunk)]
        if not coords:
            continue
        label = _XY.sub("", chunk).replace(" at ", "").strip(" ,")
        out.setdefault(label, []).extend(coords)
    return out


def monsters(obs: str) -> list[dict[str, Any]]:
    """`=== VISIBLE MONSTERS ===` as a list of dicts.

    Each has `name`, `pos`, and `pet` (True when the harness tagged it as a
    pet). Pets are reported like every other monster -- attacking one is a
    real mistake the harness will not stop, so check the flag.
    """
    text = _sections(obs).get("VISIBLE MONSTERS", "")
    out: list[dict[str, Any]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _XY.search(chunk)
        if not m:
            continue
        name = chunk[: m.start()].replace(" at ", "").strip(" ,")
        out.append({
            "name": name,
            "pos": (int(m.group(1)), int(m.group(2))),
            "pet": "PET" in chunk.upper(),
        })
    return out


def grid(obs: str) -> list[str]:
    """The `=== MAP ===` block as a list of rows, row 0 first.

    Empty when the map is withheld (call `screen()` first). Rows are NOT
    padded -- index with care, or use `tile()`.
    """
    for name, text in _sections(obs).items():
        if name.startswith("MAP"):
            rows = text.splitlines()
            return rows if any(r.strip() for r in rows) else []
    return []


def tile(obs: str, x: int, y: int) -> str | None:
    """The glyph at (x, y) in the MAP frame, or None if off the rendered map."""
    rows = grid(obs)
    if not (0 <= y < len(rows)):
        return None
    row = rows[y]
    return row[x] if 0 <= x < len(row) else None


# ---------- the in-episode compile gate ----------

def check() -> dict[str, Any]:
    """Compile and re-import every module in this package. Call after editing.

    Returns {"ok": bool, "errors": [...]}. This is the same check the harness
    runs unconditionally at rollout teardown, so a tree that fails here will
    not be offered to the round merge -- fix it before you finish.
    """
    import ast
    import importlib
    import pathlib

    root = pathlib.Path(__file__).resolve().parent
    errors: list[dict[str, str]] = []
    for path in sorted(root.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    if not errors:
        try:
            importlib.reload(importlib.import_module("netplay"))
        except Exception as exc:
            errors.append({"file": "netplay/__init__.py",
                           "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": not errors, "errors": errors}
