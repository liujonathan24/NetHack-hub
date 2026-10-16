"""Read NetHack's menus, text windows, prompts and messages off the tty screen.

Only what is on the screen is read, so the agent sees no more than a human would.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# `(end)` / `(1 of 2)` / `--More--` as the LAST thing on a row.
_TERMINATOR_RE = re.compile(r"(\(end\)|\(\d+ of \d+\)|--More--)\s*$")
_PAGE_RE = re.compile(r"\((\d+) of (\d+)\)")
# "a - item"; the mark is "+" when selected, "#" when selected with a count.
_ENTRY_RE = re.compile(r"^([a-zA-Z0-9$#*?:<>@_-]) ([-+#]) (.*)$")


@dataclass
class TtyWindow:
    """A NetHack menu / text window as drawn on the terminal."""
    lines: list[str]                 # body rows, left edge removed, rstripped
    terminator: str                  # "(end)", "(1 of 2)" or "--More--"
    col: int                         # the window's left column (offx)
    end_row: int
    entries: list[tuple[str, str, str]] = field(default_factory=list)  # (letter, mark, text)
    rows: list[int] = field(default_factory=list)  # tty row of each line

    @property
    def is_menu(self) -> bool:
        return bool(self.entries)

    @property
    def page(self) -> Optional[tuple[int, int]]:
        m = _PAGE_RE.search(self.terminator)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def signature(self) -> tuple:
        return (tuple(self.lines), self.terminator)


def tty_rows(obs) -> list[str]:
    tty = getattr(obs, "tty_chars", None)
    if tty is None and isinstance(obs, dict):
        tty = obs.get("tty_chars")
    if tty is None:
        return []
    try:
        arr = np.asarray(tty).astype(np.uint8, copy=False)
        arr = np.where((arr >= 32) & (arr < 127), arr, 32).astype(np.uint8)
        return [bytes(row).decode("ascii") for row in arr]
    except (TypeError, ValueError):
        return []


def tty_cursor(obs) -> Optional[tuple[int, int]]:
    """(row, col) of the terminal cursor, or None."""
    cur = getattr(obs, "tty_cursor", None)
    if cur is None and isinstance(obs, dict):
        cur = obs.get("tty_cursor")
    if cur is None:
        return None
    try:
        c = np.asarray(cur).reshape(-1)
        return int(c[0]), int(c[1])
    except (TypeError, ValueError, IndexError):
        return None


def misc_flags(obs) -> Optional[tuple[bool, bool, bool]]:
    """(in_yn_function, in_getlin, xwaitforspace), or None when unavailable."""
    misc = getattr(obs, "misc", None)
    if misc is None and isinstance(obs, dict):
        misc = obs.get("misc")
    if misc is None:
        return None
    try:
        m = np.asarray(misc).reshape(-1)
        return bool(m[0]), bool(m[1]), bool(m[2])
    except (TypeError, ValueError, IndexError):
        return None


def obs_message(obs) -> str:
    msg = getattr(obs, "message", None)
    if msg is None and isinstance(obs, dict):
        msg = obs.get("message")
    if msg is None:
        return ""
    try:
        raw = bytes(np.asarray(msg, dtype=np.uint8).tobytes()) if not isinstance(msg, (bytes, bytearray)) else bytes(msg)
        return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    except (TypeError, ValueError):
        return ""


def row0(obs) -> str:
    rows = tty_rows(obs)
    return rows[0].strip() if rows else ""


_GETPOS_MARKERS = ("where do you want to travel to?", "(for instructions type a",
                   "pick an object", "pick a monster", "where do you want to")


def is_getpos_text(text: str) -> bool:
    """A position prompt (travel `_`, farlook `;`). The cursor is on the map and
    `misc` stays all zero, so only the prompt text shows it."""
    low = (text or "").lower()
    return any(m in low for m in _GETPOS_MARKERS)


def find_window(obs) -> Optional[TtyWindow]:
    """The menu / text window currently drawn on the terminal, or None.

    tty NetHack draws a window from row 0 down to a terminator line printed at
    the window's left column, and keeps the cursor inside it while it waits.
    """
    rows = tty_rows(obs)
    if not rows:
        return None
    cur = tty_cursor(obs)
    # Row 0 is skipped: a --More-- there is the message line's prompt.
    for r in range(len(rows) - 1, 0, -1):
        m = _TERMINATOR_RE.search(rows[r])
        if not m:
            continue
        col = m.start(1)
        if rows[r][col:].strip() != m.group(1):
            continue
        # A full-screen text window starts at column 0, one left of its --More--.
        if col == 1 and any(row[:1].strip() for row in rows[:r]):
            col = 0
        # A window left over on screen does not hold the cursor.
        if cur is not None and not (0 <= cur[0] <= r and cur[1] >= col):
            continue
        body = [row[col:].rstrip() for row in rows[:r]]
        body_rows = list(range(r))
        while body and not body[0].strip():
            body.pop(0)
            body_rows.pop(0)
        while body and not body[-1].strip():
            body.pop()
            body_rows.pop()
        ind = [len(b) - len(b.lstrip()) for b in body if b.strip()]
        cut = min(ind) if ind else 0
        body = [b[cut:] for b in body]
        win = TtyWindow(lines=body, terminator=m.group(1), col=col, end_row=r,
                        entries=_entries(body), rows=body_rows)
        # An open text window keeps xwaitforspace set. With all flags clear its
        # text is a leftover, even when the hero stands inside it.
        flags = misc_flags(obs)
        if not win.is_menu and flags is not None and not any(flags):
            continue
        return win
    return None


def _entries(lines) -> list[tuple[str, str, str]]:
    out = []
    for b in lines:
        em = _ENTRY_RE.match(b.strip())
        if em:
            out.append((em.group(1), em.group(2), em.group(3).strip()))
    return out


def _still_open(obs, prev: TtyWindow) -> bool:
    """Is `prev` still the window NetHack is waiting in?"""
    flags = misc_flags(obs)
    if flags is not None and not any(flags):
        return False
    cur = tty_cursor(obs)
    if cur is None:
        return bool(flags and any(flags))
    return 0 <= cur[0] <= prev.end_row and cur[1] >= prev.col


def resolve_window(obs, prev: Optional[TtyWindow] = None) -> Optional[TtyWindow]:
    """The open window, carrying `prev` forward across partial redraws.

    Toggling a pick-several entry repaints only that row, so while the game
    still waits in `prev`, keep it and take the repainted selection marks.
    """
    fresh = find_window(obs)
    if fresh is not None:
        return fresh
    if prev is None or not _still_open(obs, prev):
        return None
    rows = tty_rows(obs)
    lines = list(prev.lines)
    for i, r in enumerate(prev.rows or ()):
        if i >= len(lines) or r >= len(rows):
            break
        now = rows[r][prev.col:].strip()
        pm = _ENTRY_RE.match(lines[i].strip())
        em = _ENTRY_RE.match(now)
        if pm and em and pm.group(1) == em.group(1) and pm.group(3).strip() == em.group(3).strip():
            lead = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            lines[i] = lead + now
    return TtyWindow(lines=lines, terminator=prev.terminator, col=prev.col,
                     end_row=prev.end_row, entries=_entries(lines), rows=list(prev.rows))


def format_window(win: TtyWindow, indent: str = "  ") -> list[str]:
    out = [f"{indent}{ln}" if ln.strip() else "" for ln in win.lines]
    out.append(f"{indent}{win.terminator}")
    return out


#: Cap on message lines shown per call; `bound_lines` keeps the head and tail.
MAX_MESSAGE_LINES = 40
_HEAD_LINES = 12


def awaiting_answer(obs) -> bool:
    """Is the game reading an answer (a prompt, a --More--, a window) rather
    than a command?"""
    flags = misc_flags(obs)
    cur = tty_cursor(obs)
    return bool(flags and any(flags)) or (cur is not None and cur[0] == 0) or find_window(obs) is not None


#: Keys that leave a command unfinished in command context: a count digit, or
#: a prefix (`F`, `m`, `g`, `G`, `M`) that waits for its direction.
_UNFINISHED_KEYS = frozenset(ord(c) for c in "0123456789FmgGM")


def _key(action) -> Optional[int]:
    try:
        return int(action) if action is not None else None
    except (TypeError, ValueError):
        return None


class _HarvestState:
    """Running state for collecting new messages over consecutive engine steps.

    A completed command clears the message line before it runs, so after one
    every message is new; while the game reads an answer, or the rest of a
    command, the line persists, so only a change is new. `prev` starts as the
    message already on screen.
    """

    __slots__ = ("prev", "waiting", "seen_windows", "held", "last_prompt")

    def __init__(self, baseline_obs=None) -> None:
        self.prev = obs_message(baseline_obs) if baseline_obs is not None else ""
        self.waiting = baseline_obs is not None and awaiting_answer(baseline_obs)
        self.seen_windows: set = set()
        #: (signature, lines) of the text window on screen; its lines join the
        #: messages once it closes.
        self.held = None
        #: The prompt on the message line after the previous step, if any.
        self.last_prompt = ""
        if baseline_obs is not None:
            cur = tty_cursor(baseline_obs)
            if cur is not None and cur[0] == 0 and "--More--" not in row0(baseline_obs):
                self.last_prompt = self.prev.rstrip()


def _step_text(obs, st: "_HarvestState", action=None) -> list[str]:
    """What one engine step newly printed, as display lines."""
    out: list[str] = []
    if not st.waiting and _key(action) in _UNFINISHED_KEYS:
        st.waiting = True
    win = find_window(obs)
    sig = win.signature() if (win is not None and not win.is_menu) else None
    if st.held is not None and st.held[0] != sig:
        out.extend(st.held[1])
        st.held = None
    cur = tty_cursor(obs)
    msg = obs_message(obs)
    more_on_top = "--More--" in row0(obs)
    # Cursor on row 0 without --More--: NetHack is reading input on the message
    # line (a question, an item prompt, an extended command), not printing.
    prompting = cur is not None and cur[0] == 0 and not more_on_top
    # "<prompt> <answer>" left after answering echoes the agent's own key.
    new = msg
    if st.last_prompt and msg.startswith(st.last_prompt):
        rest = msg[len(st.last_prompt):].lstrip()
        rest = rest[1:].strip() if rest else ""   # drop the one-key answer
        new = rest if len(rest) > 1 else ""
    if new and not prompting and (msg != st.prev or not st.waiting):
        out.append(new)
    st.prev = msg
    st.waiting = awaiting_answer(obs)
    st.last_prompt = msg.rstrip() if (prompting and msg) else ""
    if sig is not None and sig not in st.seen_windows:
        st.seen_windows.add(sig)
        st.held = (sig, [ln.strip() for ln in win.lines if ln.strip()])
    return out


def new_messages_per_step(observations, baseline_obs=None, actions=None):
    """Yield, for each observation, the list of lines it newly printed.

    `actions`, when given, holds the key that produced each observation."""
    st = _HarvestState(baseline_obs)
    acts = list(actions or ())
    for i, o in enumerate(observations or ()):
        yield o, _step_text(o, st, acts[i] if i < len(acts) else None)


def bound_lines(lines: list[str], cap: int = MAX_MESSAGE_LINES) -> list[str]:
    if len(lines) <= cap:
        return list(lines)
    tail = cap - _HEAD_LINES
    n = len(lines) - _HEAD_LINES - tail
    return lines[:_HEAD_LINES] + [f"... ({n} more messages elided) ..."] + lines[-tail:]


_INVENTORY_KEYS = (ord("i"), ord("I"))


def apply_inventory_listing(state: dict, inventory, call_log=None) -> None:
    """Show each item as the game's own inventory listing last described it.

    The listing text is used only while the item's `inv_strs` text is unchanged
    since the listing was captured. Mutates the items' `description`.
    """
    if call_log is not None and call_log.inventory_listing and not getattr(call_log, "_inv_used", False):
        call_log._inv_used = True
        state["_inv_tty"] = {
            "listing": dict(call_log.inventory_listing),
            "seen": {str(it.letter): str(it.description) for it in inventory or ()},
        }
    tt = state.get("_inv_tty")
    if not tt:
        return
    listing, seen = tt["listing"], tt["seen"]
    for it in inventory or ():
        l = str(it.letter)
        if l in listing and seen.get(l) == it.description and listing[l] != it.description:
            it.description = listing[l]


class CallScreenLog:
    """Collect the text of every engine step taken during one tool call.

    Patches `env.step` on the instance inside the `with` block. Read-only: it
    never steps the engine or changes an action.
    """

    def __init__(self, env, baseline_obs=None, prev_window=None) -> None:
        self._env = env
        #: The open menu / text window after the latest step (see `resolve_window`).
        self.window: Optional[TtyWindow] = prev_window
        self._orig = None
        self._had_own = False
        self._installed = False
        self.lines: list[str] = []
        self.steps = 0
        self._st = _HarvestState(baseline_obs)
        #: Step (1-based) at which the travel/getpos prompt last appeared.
        self.getpos_step = 0
        #: letter -> description from an inventory listing (`i`) seen this call.
        self.inventory_listing: Optional[dict] = None
        self._inv_open = False
        #: Game turn when the call started (None if unknown).
        self.start_turn = None
        #: blstats of the latest step that still had them (NLE zeroes them at game over).
        self.live_blstats = None
        self._keep_live(baseline_obs)
        try:
            if baseline_obs is not None:
                self.start_turn = int(np.asarray(getattr(baseline_obs, "blstats"))[20])
        except Exception:
            self.start_turn = None

    def _keep_live(self, obs) -> None:
        bl = getattr(obs, "blstats", None)
        if bl is not None and np.asarray(bl).any():
            self.live_blstats = np.array(bl)

    def _note(self, obs, action=None) -> None:
        self.steps += 1
        self._keep_live(obs)
        # An `i` typed as an answer (to a prompt, or in a menu) is not the
        # inventory command.
        at_command = self.window is None and not self._st.waiting
        self.window = resolve_window(obs, self.window)
        # Keep the game's own `i` listing: `inv_strs` is refreshed only by
        # update_inventory(), so it can lag behind (a dagger that got dull).
        a = _key(action)
        if a in _INVENTORY_KEYS and at_command:
            self._inv_open = True
        if self._inv_open:
            w = self.window
            if w is not None and w.is_menu:
                listing = dict(self.inventory_listing or {})
                listing.update({l: t for (l, _m, t) in w.entries})
                self.inventory_listing = listing
            else:
                self._inv_open = False
        before = self._st.prev
        self.lines.extend(_step_text(obs, self._st, a))
        if self._st.prev != before and is_getpos_text(self._st.prev):
            self.getpos_step = self.steps

    def __enter__(self) -> "CallScreenLog":
        env = self._env
        if env is None:
            return self
        try:
            orig = env.step
            self._had_own = "step" in getattr(env, "__dict__", {})

            def _logging_step(action, *a, **kw):
                out = orig(action, *a, **kw)
                try:
                    self._note(out[0] if isinstance(out, tuple) else out, action)
                except Exception:
                    pass  # observation-only; must never break a step
                return out

            env.step = _logging_step
            self._orig = orig
            self._installed = True
        except Exception:
            self._installed = False
        return self

    def __exit__(self, *exc) -> None:
        if not self._installed:
            return
        try:
            if self._had_own:
                self._env.step = self._orig
            else:
                del self._env.step
        except Exception:
            pass
        self._installed = False
        # This object stays in the rollout state after the call: drop engine refs.
        self._env = None
        self._orig = None
