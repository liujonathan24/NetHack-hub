"""Read NetHack's UI off the terminal the way a human player sees it.

Why this module exists
----------------------
`nethack_core.observations.shape()` parses menus with a bottom-up regex that
stops at the first non-item row. A NetHack menu is item rows *interleaved with
class headers* ("Weapons", "Comestibles"), so the parse kept only the entries
below the last header. Measured on the np_core surface: the inventory (`i`)
rendered as a single garbled `[0] an uncur ration`, the pick-up menu (`,`)
showed "Pick up what?" with no entries at all, and text windows ("Things that
are here:", the enlightenment screen) were not rendered at all. Every one of
these is fully visible on `tty_chars`.

What is read, and from where
----------------------------
* **Windows.** tty NetHack draws a menu or text window as a rectangle whose
  left edge is its `offx` column, starting at row 0, with a terminator line at
  the bottom: `(end)`, `(N of M)` for multi-page menus, or `--More--`. The
  terminator is printed at `offx`, and the cursor is parked right after it
  while the window waits. So: find the terminator row, take every row above it
  from that column on. The part of each row LEFT of `offx` is the dungeon (or
  the copyright banner remnant) and is ignored.
* **Pending input.** `misc` (in_yn_function, in_getlin, xwaitforspace) plus the
  cursor. At the command prompt the cursor sits on the hero; while NetHack is
  asking something on the message line the cursor is on row 0. A remembered
  message is never used to decide that the game is waiting.
* **Messages.** The per-step `message` field is the text NetHack wrote to the
  message line during that step (empty when it wrote nothing), so collecting
  it on every engine step of a call gives every message of the call, in order,
  including each link of a `--More--` chain.

Only what is on the screen is read, so nothing here can show the agent more
than a human at the terminal would see.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# `(end)` / `(1 of 2)` / `--More--` as the LAST thing on a row.
_TERMINATOR_RE = re.compile(r"(\(end\)|\(\d+ of \d+\)|--More--)\s*$")
_PAGE_RE = re.compile(r"\((\d+) of (\d+)\)")
# "a - item", "a + item" (selected), "a # item" (count selected). NetHack menu
# accelerators are letters, plus a few symbols in special menus.
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
        """True when it offers selectable entries (letters)."""
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
    """(row, col) of the terminal cursor, or None when the obs has none."""
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
    """The message-line text NetHack wrote during the step that produced `obs`."""
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
    """A position prompt (travel `_`, farlook `;`, ...). It is drawn on the
    message line but the cursor sits on the map, so `misc` stays all zero and
    the cursor-on-row-0 test cannot see it."""
    low = (text or "").lower()
    return any(m in low for m in _GETPOS_MARKERS)


def find_window(obs) -> Optional[TtyWindow]:
    """The menu / text window currently drawn on the terminal, or None.

    A `--More--` on row 0 is the message line's own prompt, not a window, and is
    not returned here. When the obs carries a cursor, the window only counts if
    the cursor is inside it (NetHack parks it after the terminator while the
    window waits, or on the entry just toggled), so a remnant can never be
    mistaken for a live window.
    """
    rows = tty_rows(obs)
    if not rows:
        return None
    cur = tty_cursor(obs)
    # The bottom-most terminator is the window's last line. A full-screen
    # page uses all 24 rows ("(1 of 2)" on row 23); the status lines never
    # end in a terminator, so scanning them is safe.
    for r in range(len(rows) - 1, 0, -1):
        m = _TERMINATOR_RE.search(rows[r])
        if not m:
            continue
        col = m.start(1)
        # The terminator must be alone on its row from `col` on, and the row
        # left of it must not be window text (it is dungeon or blank).
        if rows[r][col:].strip() != m.group(1):
            continue
        # Live windows hold the cursor: after the terminator while waiting,
        # or on the entry just toggled in a pick-several menu.
        if cur is not None and not (0 <= cur[0] <= r and cur[1] >= col):
            continue
        body = [row[col:].rstrip() for row in rows[:r]]
        body_rows = list(range(r))
        # Trim blank rows at both ends, keep interior blanks (they separate
        # sections, exactly as on screen).
        while body and not body[0].strip():
            body.pop(0)
            body_rows.pop(0)
        while body and not body[-1].strip():
            body.pop()
            body_rows.pop()
        # Remove the common indent so the text starts at column 0.
        ind = [len(b) - len(b.lstrip()) for b in body if b.strip()]
        cut = min(ind) if ind else 0
        body = [b[cut:] for b in body]
        win = TtyWindow(lines=body, terminator=m.group(1), col=col, end_row=r,
                        entries=_entries(body), rows=body_rows)
        # A text window waits for its dismissal key, so the engine's
        # xwaitforspace is set for as long as it is open. Its text can stay
        # on screen after it closes (the Quest entry text did, for the rest of
        # the game), and the cursor-inside test passes whenever the hero
        # stands inside that remnant; with the flags all clear it is gone.
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

    Toggling an entry in a pick-several menu makes the engine repaint only
    that entry's row; the rest of the window's rectangle comes back as the
    dungeon underneath (measured: after `,` then `a`, only "a + a +0 dagger"
    and the title were on screen). So when no complete window is visible but
    the game is still waiting inside the one seen before, keep that window and
    update the rows that were repainted (their selection marks).
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
    """The window exactly as on screen, one line per row, terminator last."""
    out = [f"{indent}{ln}" if ln.strip() else "" for ln in win.lines]
    out.append(f"{indent}{win.terminator}")
    return out


# ---------------------------------------------------------------------------
# Per-call message harvest
# ---------------------------------------------------------------------------

#: Cap on rendered message lines per call. A 100-step explore can print a lot;
#: keep the head and the tail (the outcome) and say how many were elided.
MAX_MESSAGE_LINES = 40
_HEAD_LINES = 12


class _HarvestState:
    """Running state for a message harvest over consecutive engine steps.

    This engine does NOT clear the message line between steps: `message` (and
    tty row 0) keep showing the last thing printed until something new is
    printed. Measured: "You drop a +0 dagger." was still the `message` of two
    later `s` (search) steps. So a message is new only when it CHANGES; the
    running `prev` is seeded with the message that was already on screen when
    the call started, which is what stops "You kill the newt!" being reported
    again by four later calls.
    """

    __slots__ = ("prev", "seen_windows", "held", "last_prompt")

    def __init__(self, baseline_obs=None) -> None:
        self.prev = obs_message(baseline_obs) if baseline_obs is not None else ""
        self.seen_windows: set = set()
        #: The text window on screen right now, as (signature, lines). Its
        #: lines join the message list once it is gone from the screen; while
        #: it is still up, the waiting-for-input banner shows it instead.
        self.held = None
        #: The prompt text on the message line after the previous step, if
        #: the game was asking something there.
        self.last_prompt = ""
        if baseline_obs is not None:
            cur = tty_cursor(baseline_obs)
            if cur is not None and cur[0] == 0 and "--More--" not in row0(baseline_obs):
                self.last_prompt = self.prev.rstrip()


def _step_text(obs, st: "_HarvestState") -> list[str]:
    """What one engine step newly printed, as display lines (possibly empty)."""
    out: list[str] = []
    win = find_window(obs)
    sig = win.signature() if (win is not None and not win.is_menu) else None
    if st.held is not None and st.held[0] != sig:
        out.extend(st.held[1])
        st.held = None
    cur = tty_cursor(obs)
    msg = obs_message(obs)
    more_on_top = "--More--" in row0(obs)
    # Cursor on row 0 without a --More-- means NetHack is reading input on the
    # message line: a question ("[yn]"), an item prompt, or the echo of an
    # extended command being typed. That is a prompt, not a message; if it is
    # still pending, the waiting-for-input banner shows it.
    prompting = cur is not None and cur[0] == 0 and not more_on_top
    # Answering a prompt leaves "<prompt> <answer>" on the message line
    # ("In what direction? h"). That is an echo of the agent's own key, not
    # something the game said.
    new = msg
    if st.last_prompt and msg.startswith(st.last_prompt):
        rest = msg[len(st.last_prompt):].lstrip()
        rest = rest[1:].strip() if rest else ""   # drop the one-key answer
        new = rest if len(rest) > 1 else ""
    if new and msg != st.prev and not prompting:
        out.append(new)
    st.prev = msg
    st.last_prompt = msg.rstrip() if (prompting and msg) else ""
    if sig is not None and sig not in st.seen_windows:
        st.seen_windows.add(sig)
        st.held = (sig, [ln.strip() for ln in win.lines if ln.strip()])
    return out


def new_messages_per_step(observations, baseline_obs=None):
    """Yield, for each observation, the list of lines it newly printed."""
    st = _HarvestState(baseline_obs)
    for o in observations or ():
        yield o, _step_text(o, st)


def harvest_messages(observations, baseline_obs=None) -> list[str]:
    """Every message the game newly printed across `observations`, in order.

    `baseline_obs` is the observation before the first step (its message is
    old news). Text windows (not menus) are included once each.
    """
    lines: list[str] = []
    st = _HarvestState(baseline_obs)
    for o in observations or ():
        for ln in _step_text(o, st):
            if not lines or lines[-1] != ln:
                lines.append(ln)
    if st.held is not None:
        lines.extend(st.held[1])
    return lines


def bound_lines(lines: list[str], cap: int = MAX_MESSAGE_LINES) -> list[str]:
    if len(lines) <= cap:
        return list(lines)
    tail = cap - _HEAD_LINES
    n = len(lines) - _HEAD_LINES - tail
    return lines[:_HEAD_LINES] + [f"... ({n} more messages elided) ..."] + lines[-tail:]


_INVENTORY_KEYS = (ord("i"), ord("I"))


def apply_inventory_listing(state: dict, inventory, call_log=None) -> None:
    """Show each item as the game's own inventory listing last described it.

    `state["_inv_tty"]` keeps the last listing together with the `inv_strs`
    text each item had when it was captured. An item's listing text is used
    only while its `inv_strs` text is still that captured text; as soon as
    the engine reports anything new for the letter, `inv_strs` wins again.
    Mutates the items' `description` in place.
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
    """Collect the text of every engine step taken during ONE tool call.

    Patches `env.step` on the instance for the duration of the `with` block
    (the same seam `helpers.TurnRecorder` uses; this nests inside it and is
    removed first). Read-only: it never steps the engine and never changes an
    action, so replay determinism is unaffected.
    """

    def __init__(self, env, baseline_obs=None, prev_window=None) -> None:
        self._env = env
        #: The open menu / text window after the latest step (None when none),
        #: carried across partial redraws -- see `resolve_window`.
        self.window: Optional[TtyWindow] = prev_window
        self._orig = None
        self._had_own = False
        self._installed = False
        self.lines: list[str] = []
        self.steps = 0
        self._st = _HarvestState(baseline_obs)
        #: Step index (1-based) at which the travel/getpos prompt last appeared.
        self.getpos_step = 0
        #: letter -> description from an inventory listing (`i`) seen this call.
        self.inventory_listing: Optional[dict] = None
        self._inv_open = False
        self._waiting = False
        #: Game turn when the call started (None if unknown).
        self.start_turn = None
        try:
            if baseline_obs is not None:
                self.start_turn = int(np.asarray(getattr(baseline_obs, "blstats"))[20])
        except Exception:
            self.start_turn = None
        if baseline_obs is not None:
            _f, _c = misc_flags(baseline_obs), tty_cursor(baseline_obs)
            self._waiting = bool(_f and any(_f)) or (_c is not None and _c[0] == 0)

    def _note(self, obs, action=None) -> None:
        self.steps += 1
        # Was the game at the command prompt before this key? (An `i` typed
        # as an ANSWER -- to an item prompt, or in a pick-up menu -- is not
        # the inventory command.)
        at_command = self.window is None and not self._waiting
        self.window = resolve_window(obs, self.window)
        flags = misc_flags(obs)
        cur = tty_cursor(obs)
        self._waiting = bool(flags and any(flags)) or (cur is not None and cur[0] == 0)
        # The inventory as the game itself lists it (`i` / `I`). The engine's
        # `inv_strs` is only refreshed when NetHack calls update_inventory(),
        # which some changes never trigger: after "Your dagger gets dull." the
        # `i` menu said "c - a -2 dagger" while `inv_strs` still said "+0".
        # The listing is what a human sees, so the renderer prefers it.
        try:
            a = int(action) if action is not None else None
        except (TypeError, ValueError):
            a = None
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
        for ln in _step_text(obs, self._st):
            if not self.lines or self.lines[-1] != ln:
                self.lines.append(ln)
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
        # Keep the harvested text, drop the engine references: this object
        # stays in the rollout state after the call.
        self._env = None
        self._orig = None
