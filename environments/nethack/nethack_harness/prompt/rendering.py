"""Prompt rendering: SYSTEM_PROMPT, per-variant observation formatters, the
descent/E1/E2 observation blocks, and the canonical per-turn chat formatter.

Extracted verbatim from nethack.py (no logic change). The single most-edited
surface of the harness lives here.
"""
from __future__ import annotations

import re
from typing import Optional

from nethack_harness.memory.journal import Journal

# --------------------------------------------------------------------------- #
# System prompt                                                                #
# --------------------------------------------------------------------------- #
#
# The prompt is ASSEMBLED from the resolved tool list, not hand-written as one
# string. NetPlay builds its skill list from the live repository
# (`skill_selection.py:133`) and BALROG builds its action list from the env
# (`nle/__init__.py`), which is why neither can advertise a tool that does not
# exist. Ours could, and did: under `skill_set="netplay"` — what every arm of
# the CLI-harness comparison runs — no `move(direction=...)` adapter is built,
# yet seven places told the agent to call it (exp1 measured 232 rejected `move`
# attempts under the ASCII encoding). Each block below declares the tools it
# names; blocks whose tools are not published are dropped.

_PROMPT_HEAD = """You are playing NetHack, a procedurally-generated dungeon-crawling roguelike.

Each turn shows: map (ASCII), stats, inventory, messages, any menu. Act by
calling one tool.

=== COORDINATES ===
One frame, everywhere. `=== MAP ===` row 0 is the TOP row of the dungeon;
`Pos: (x,y)` indexes it directly (your `@` is on map row y, column x); every
coordinate in VISIBLE FEATURES and every (x, y) argument a tool takes is in
that same frame. Copy coordinates straight across — no offsets.

=== STRATEGY PRIMER ===
GLYPH KEY:
Terrain: `>` stairs DOWN, `<` stairs UP (NOT down), `_` altar, `{` fountain,
`}` pool, `#` corridor, `.` floor, `|`/`-` walls, `+` closed door (or a
spellbook lying on the floor), `\\` throne, `$` gold, `%` food/corpse,
`[`/`)`/`(`/`*`/`?` items. Creatures are LETTERS (a-z, A-Z): `d` canine,
`f` feline, `F` lichen/fungus, `r` rat, `x` grid bug, `B` bat, `k` kobold,
`o` orc, `@` humans (and YOU). No "fireplace" glyph — adjacent `f` is a
creature. `@` hides the tile under you — read UNDER PLAYER.

VISIBLE FEATURES lists every stairs/altar/fountain/door on the visible map
with (x,y). VISIBLE MONSTERS names each one with its distance and bearing.
If `stairs DOWN` isn't listed, no `>` is visible — don't pattern-match the
grid; explore or `search`. To descend: (1) find `>`, (2) walk ON it,
(3) call `descend`. If descend fails, recheck UNDER PLAYER."""

_PROMPT_DOORWAYS_HEAD = """DOORWAYS & WALL GAPS: a single non-wall tile inside a wall row is a
doorway you can walk through. Examples:
  `--.---` (horizontal wall with `.`) → walk through the `.`.
  `|.....|` with `-` in the middle → that `-` is broken wall; walk it.
  `-----|-----` (`|` inside a horizontal wall row) → OPEN DOOR; just walk."""

_PROMPT_DOORWAYS_TAIL = """When stuck in a room, scan every wall row for a tile that doesn't match
`-` or `|`. That's your exit. VISIBLE FEATURES already lists them for you as
`door (open/gap)` and `door (closed)`."""

_PROMPT_PITFALLS = """Pitfalls: `eat`/`quaff`/`read` need an `item` arg. Menus auto-dismiss; never
call menu/inventory tools."""

_PROMPT_STAY_ALIVE_HEAD = """=== STAY ALIVE (death is what stops you, not the clock) ===
Most runs end in DEATH, not time — usually one of:"""

_PROMPT_STARVATION = """- **Starvation.** Don't let Hunger reach Weak/Fainting. `pickup` every food item
  and corpse you pass; `eat(item=...)` BEFORE you get Weak (fresh corpses of
  non-poisonous monsters are food). Resting costs exactly as much clock as
  walking, so never rest while Hungry or worse — eat instead."""

_PROMPT_MELEE_SWARM = """- **Melee swarm at low HP.** A fox/jackal/newt chips you to death. Don't melee at
  low HP — `engrave_elbereth` (scares most monsters), or `pray` (once, when HP
  is critical), or flee toward stairs."""

_PROMPT_RANGED = """- **Ranged / approaching threats.** Kill dangerous monsters from a distance with
  `throw(item=..., direction=...)` (daggers, darts, rocks, spears) instead of
  letting them reach you. Hit it before it hits you."""

_PROMPT_TAIL = """Your top-level goal is pre-pinned as `Objective:` in JOURNAL. Keep acting
every turn — the episode ends only when your character dies or you run out of
calls, never because you stopped, summarized, or declared the run finished."""

# (required tools, text). A block is emitted only when every tool it names is
# in the published set, so the advertised surface can never exceed the real one.
#
# This is the VERBOSE variant (pre-Task-18 behavior, byte-for-byte): the full
# strategy primer, the DESCEND ASAP workhorse pitch, the STAY ALIVE sermon and
# the pitfalls list. Kept as an explicit A/B opt-in (`SYSTEM_PROMPT_VERBOSE`,
# `load_environment(verbose_prompt=True)`) — see `_PROMPT_BLOCKS_MINIMAL`
# below for the default.
_PROMPT_BLOCKS_VERBOSE: list[tuple[tuple[str, ...], str]] = [
    ((), _PROMPT_HEAD),
    ((), _PROMPT_DOORWAYS_HEAD),
    (("kick",),
     "  `-----+-----` → `+` is a closed door; walk into it to open it, or\n"
     "  `kick` it if it says \"locked\"."),
    ((), _PROMPT_DOORWAYS_TAIL),
    (("move_to",), "Use `move_to(x,y)` to walk to any of them."),
    (("explore_and_descend",),
     "=== STRATEGY: DESCEND ASAP ===\n"
     "**Your workhorse is `explore_and_descend`.** It auto-explores the level\n"
     "(opening doors, searching for hidden passages), walks to the down `>` and\n"
     "descends ONE floor, then hands control back to you. It returns early if\n"
     "your HP drops or you get hungry — then heal/eat and call it again.\n"
     "If it returns \"descended 0 floor(s)\" TWICE IN A ROW, stop calling it:\n"
     "pick an unexplored frontier or a door from VISIBLE FEATURES and go there\n"
     "yourself. Repeating a call that just failed is how rollouts starve."),
    (("engrave_elbereth", "pray", "attack", "eat"),
     "If HP critical: `engrave_elbereth` or `pray`. Hostile adjacent + healthy\n"
     "HP: `attack(direction=...)`. Hungry: `eat(item=...)`."),
    (("eat", "quaff", "read"), _PROMPT_PITFALLS),
    ((), _PROMPT_STAY_ALIVE_HEAD),
    (("pickup", "eat"), _PROMPT_STARVATION),
    (("engrave_elbereth", "pray"), _PROMPT_MELEE_SWARM),
    (("throw",), _PROMPT_RANGED),
    ((), _PROMPT_TAIL),
]

# --------------------------------------------------------------------------- #
# Task 18: BALROG-minimal system prompt (the new default)                     #
# --------------------------------------------------------------------------- #
#
# research-sota-methods.md: BALROG's entire objective scaffolding is two
# sentences and NetPlay's default is "Win the game."; we gave MORE goal
# structure than either and NetPlay's own ablation LOST depth (2.60 -> 2.00)
# when tactical goal text was added. The trace analyses agree from the other
# side: `recall`/`pin_objective` were never called in 1,173 Claude Code calls.
#
# Keep only what the agent cannot derive on its own: the COORDINATES frame
# paragraph (Task 17), the glyph key, and the action list (SKILLS CHEAT
# SHEET, assembled below same as the verbose variant). Delete the STRATEGY
# PRIMER prose, the DESCEND ASAP section, the STAY ALIVE sermon and the
# pitfalls list. Append BALROG's two objective sentences verbatim
# (`balrog/environments/nle/__init__.py:107-131`) in place of our pinned-
# objective machinery.
_PROMPT_HEAD_MINIMAL = """You are playing NetHack, a procedurally-generated dungeon-crawling roguelike.

Each turn shows: map (ASCII), stats, inventory, messages, any menu. Act by
calling one tool.

=== COORDINATES ===
One frame, everywhere. `=== MAP ===` row 0 is the TOP row of the dungeon;
`Pos: (x,y)` indexes it directly (your `@` is on map row y, column x); every
coordinate in VISIBLE FEATURES and every (x, y) argument a tool takes is in
that same frame. Copy coordinates straight across — no offsets."""

_PROMPT_GLYPH_KEY_MINIMAL = """=== GLYPH KEY ===
Terrain: `>` stairs DOWN, `<` stairs UP (NOT down), `_` altar, `{` fountain,
`}` pool, `#` corridor, `.` floor, `|`/`-` walls, `+` closed door (or a
spellbook lying on the floor), `\\` throne, `$` gold, `%` food/corpse,
`[`/`)`/`(`/`*`/`?` items. Creatures are LETTERS (a-z, A-Z): `d` canine,
`f` feline, `F` lichen/fungus, `r` rat, `x` grid bug, `B` bat, `k` kobold,
`o` orc, `@` humans (and YOU). No "fireplace" glyph — adjacent `f` is a
creature. `@` hides the tile under you — read UNDER PLAYER."""

# BALROG's objective text, verbatim (`nle/__init__.py:107-131`): "Explore the
# environment to find the stairs down to the next level." is one of its Tips
# bullets and "Your goal is to get as far as possible in the game." is its
# closing line. This replaces our `pin_objective`/JOURNAL machinery entirely
# for the minimal prompt — no per-tier description, no milestone list.
#
# CLI-harness eval: `stop_condition = "agent_completed"` only means the CLI
# process exited 0 with no `@stop` firing mid-rollout — under `--print` mode
# that happens the moment the model stops emitting tool calls, whether or not
# the character is still alive, costing 15-25% of measured rollouts. One
# sentence, not a section (see the module-level note above on why the prompt
# stays BALROG-minimal): tell the agent explicitly that ending its turn is not
# how the episode ends.
_PROMPT_TAIL_MINIMAL = """Explore the environment to find the stairs down to the next level.
Your goal is to get as far as possible in the game. Keep acting every turn —
the episode ends only when your character dies or you run out of calls, never
because you stopped, summarized, or declared the run finished."""

_PROMPT_BLOCKS_MINIMAL: list[tuple[tuple[str, ...], str]] = [
    ((), _PROMPT_HEAD_MINIMAL),
    ((), _PROMPT_GLYPH_KEY_MINIMAL),
    ((), _PROMPT_DOORWAYS_HEAD),
    (("kick",),
     "  `-----+-----` → `+` is a closed door; walk into it to open it, or\n"
     "  `kick` it if it says \"locked\"."),
    ((), _PROMPT_DOORWAYS_TAIL),
    (("move_to",), "Use `move_to(x,y)` to walk to any of them."),
    ((), _PROMPT_TAIL_MINIMAL),
]

# One line per skill for the cheat sheet, emitted in this order and filtered to
# the published set. Adding a skill without a blurb is fine — it simply does not
# appear; adding a blurb for a skill nobody publishes is a no-op.
_SKILL_BLURBS: tuple[tuple[str, str], ...] = (
    # Listed FIRST because it is the one capability this engine has that stock
    # NetHack does not, and an agent that does not know undo exists will never
    # ask for it. The first `rollback` ablation published the tool correctly but
    # had no blurb here, so the agent called it ZERO times across 5 rollouts --
    # that measured tool adoption, not whether undo helps. (Same failure mode as
    # the journal tools, unused across 15 rollouts, and as the balrog80 arm.)
    ("rollback",
     "**UNDO — this game supports it**: `rollback(n)` rewinds the last n turns, "
     "putting the game back exactly as it was. NetHack is normally "
     "irreversible; here it is not. Use it as soon as a move turns out badly: "
     "you walked into a fight you are losing, stepped on a trap, ate something "
     "that made you ill, or wasted turns in a dead end. If you are about to "
     "die, roll back and choose differently. Costs one turn and no game time."),
    ("explore_and_descend",
     "**PRIMARY — dive**: `explore_and_descend` — explore the level + descend a "
     "floor, then returns to you."),
    ("move_to", "Reach a specific visible tile: `move_to(x, y)`"),
    ("move", "Step one tile: `move(direction=N|NE|E|...)`"),
    ("north", "Step one tile: `north` / `northeast` / `east` / `southeast` / "
              "`south` / `southwest` / `west` / `northwest`"),
    ("autoexplore", "Explore one hop: `autoexplore`"),
    ("find_and_descend", "Path to a visible `>` and descend: `find_and_descend`"),
    ("descend", "Descend: `descend` (must be standing on `>`)"),
    ("pickup", "Pickup: `pickup`"),
    ("search", "Search for hidden doors: `search(times=10)`"),
    ("kick", "Force a locked door: `kick(direction=...)`"),
    ("attack", "Melee: `attack(direction=N|...)` — never on a `[PET`-tagged monster"),
    ("throw", "Ranged: `throw(item=..., direction=...)`"),
    ("eat", "Eat before Weak: `eat(item=...)`"),
    ("quaff", "Drink: `quaff(item=...)`"),
    ("read", "Read: `read(item=...)`"),
    ("engrave_elbereth", "Scare monsters off: `engrave_elbereth`"),
    ("pray", "Last resort (once per ~1000 turns): `pray`"),
    ("add_note", "Notes: `add_note`"),
    ("recall", "Recall a note: `recall(query=...)`"),
    ("pin_objective", "Re-pin your goal: `pin_objective`"),
    ("wiki_lookup", "Wiki: `wiki_lookup(page=\"kobold\")`"),
    ("wiki_search", "Wiki search: `wiki_search(query=\"cockatrice\")`"),
)


#: The state key carrying the tool names actually published to the agent for
#: THIS rollout. Written once by `NetHackVerifiersEnv.setup_state` from the
#: resolved adapter list and read per-render by `_fix_hint_vocabulary`.
#:
#: HISTORY (docs/HARNESS_DEFECTS.md 3.7). This used to be a module-level
#: `_PUBLISHED_TOOLS` set that `render_system_prompt` wrote as a side effect.
#: Nothing restored it, so *constructing an environment changed how every later
#: render in the same process behaved*: after any `load_environment`, hint
#: sentences naming `search` / `engrave_elbereth` / `attack` / `kick` were
#: silently deleted from every subsequent render, including renders belonging to
#: a completely different (or entirely synthetic) game. It made render-level
#: tests order-dependent -- `test_hint_actionability` passed alone and failed
#: after `test_golden_obs` -- and `tests/golden/obs_configs.py` carried a
#: save/restore workaround for it.
#:
#: Per-render state is the fix: the set now travels with the rollout it belongs
#: to, so two environments with different skill sets can render in the same
#: process, in any order, without touching each other. A render with no state
#: (or a state that predates this key) rewrites nothing -- the same
#: "unknown publisher => leave the text alone" default the global had at import.
PUBLISHED_TOOLS_STATE_KEY = "_published_tools"


def published_tools_for(state) -> set:
    """The published-tool set for this render, or an empty set when unknown."""
    if not state:
        return set()
    try:
        return set(state.get(PUBLISHED_TOOLS_STATE_KEY) or ())
    except (AttributeError, TypeError):
        return set()


#: Canonical hint vocabulary -> the tool that is actually bound. The HINT text
#: was written against an older skill set and never reconciled with what
#: `netplay_true` publishes. Measured across 8 cells: 1,722 hints recommended
#: `search` and 260 `engrave_elbereth` -- NEITHER IS BOUND -- plus 592 `descend`
#: (bound as np_down), 484 `attack(direction=..)` and 371 `kick(direction=..)`
#: (both bound only as coordinate calls). ~3,058 dead recommendations, and the
#: agent dutifully tried to follow them.
#: SAFE renames: same call signature, different bound name. Renaming these is
#: purely cosmetic and always correct.
_HINT_TOOL_ALIASES: tuple[tuple[str, str], ...] = (
    ("find_and_descend", "np_down"),
    ("descend", "np_down"),
    ("rest", "np_rest"),
    ("pray", "np_pray"),
    ("eat", "np_eat"),
    ("pickup", "np_pickup"),
    ("move_to", "np_move_to"),
    ("look", "np_look"),
)

#: Capabilities the HINT text references that CANNOT be safely renamed, because
#: either nothing is bound for them or the bound tool takes different arguments:
#:   search / engrave_elbereth -> nothing bound at all under netplay_true
#:   attack(direction=..)      -> only np_melee_attack(x, y) exists
#:   kick(direction=..)        -> only np_kick(x, y) exists
#: Renaming `attack` to `np_melee_attack` would keep the `direction=` argument
#: and produce a call that fails on every invocation, which is worse than
#: silence. So when the canonical tool is unbound we DELETE the advice.
#: Measured dead recommendations across 8 cells: search 1,722, descend 592,
#: attack 484, kick 371, engrave_elbereth 260.
_HINT_UNSAFE: tuple[str, ...] = ("search", "engrave_elbereth", "attack", "kick")


def _fix_hint_vocabulary(hint: str, published_tools) -> str:
    """Rewrite a HINT so it only ever names a tool the agent can actually call.

    `published_tools` is REQUIRED and explicit: this function used to read a
    module-level set that `render_system_prompt` wrote as a side effect, which
    made the output depend on which environments had been constructed earlier in
    the process (see `PUBLISHED_TOOLS_STATE_KEY`). An empty/None set means "we do
    not know what is bound", and the hint is returned untouched.

    Safe-renames where the signature matches; deletes the sentence entirely
    where the capability is unbound or takes different arguments. Deleting a
    whole sentence (rather than the token) keeps the hint grammatical -- an
    earlier version left fragments like "Try a different frontier, or for a
    hidden passage."
    """
    published = set(published_tools or ())
    if not hint or not published:
        return hint
    import re as _re
    for canonical, bound in _HINT_TOOL_ALIASES:
        if canonical in published:
            continue
        if bound in published:
            hint = _re.sub(rf"`{canonical}\b", f"`{bound}", hint)
            hint = _re.sub(rf"\b{canonical}\(", f"{bound}(", hint)
    for canonical in _HINT_UNSAFE:
        if canonical in published:
            continue
        # Drop any SENTENCE that mentions the unbound capability.
        kept = [seg for seg in _re.split(r"(?<=[.!])\s+", hint)
                if not _re.search(rf"`?\b{canonical}\b`?", seg)]
        hint = " ".join(kept)
    return _re.sub(r"\s{2,}", " ", hint).strip()


def render_system_prompt(published_tools=None, verbose: bool = False) -> str:
    """Assemble the system prompt for a specific published tool set.

    `published_tools=None` means "everything the registry knows", which is the
    module-level :data:`SYSTEM_PROMPT` default. Callers that know their
    `skill_set` (``load_environment``, the v1 taskset, the CLI workspace
    builder) pass the resolved adapter names so the prompt cannot advertise a
    tool the agent is unable to call.

    `verbose=False` (default, Task 18) renders the BALROG-minimal prompt:
    the COORDINATES paragraph, the glyph key, the action list, and BALROG's
    two objective sentences — no strategy prose. `verbose=True` renders the
    pre-Task-18 prompt (`SYSTEM_PROMPT_VERBOSE`), so the A/B is this one flag,
    not a `git revert`.

    PURE. This function has no side effects: it used to also stash
    `published_tools` in a module global that `_fix_hint_vocabulary` read, which
    meant building an environment silently re-programmed every later render in
    the process (docs/HARNESS_DEFECTS.md 3.7). The observation renderer now takes
    the set from per-rollout state instead -- see `PUBLISHED_TOOLS_STATE_KEY`.
    """
    if published_tools is None:
        from nethack_harness.tools.skills import registry as _registry

        available = set(_registry.all_schemas())
    else:
        available = set(published_tools)

    blocks = _PROMPT_BLOCKS_VERBOSE if verbose else _PROMPT_BLOCKS_MINIMAL
    parts = [text for required, text in blocks
             if all(t in available for t in required)]
    sheet = [f"- {blurb}" for name, blurb in _SKILL_BLURBS if name in available]
    if sheet:
        parts.insert(len(parts) - 1, "=== SKILLS CHEAT SHEET ===\n" + "\n".join(sheet))
    return "\n\n".join(parts)


#: BALROG-minimal (Task 18 default): coordinates + glyph key + action list +
#: BALROG's two objective sentences. No strategy prose.
SYSTEM_PROMPT = render_system_prompt()

#: Pre-Task-18 prompt: the strategy primer, DESCEND ASAP, STAY ALIVE, and the
#: pitfalls list. Opt in via `load_environment(verbose_prompt=True)`.
SYSTEM_PROMPT_VERBOSE = render_system_prompt(verbose=True)


# ---------- observation formatting for chat ----------
#
# Per-turn observation cost is the dominant line item in our token bill
# (~27k tok/turn at v0.0.15, ~4M tok per 150-turn rollout). Three cheap
# token-savers per docs/PROMPTING_SURVEY.md, with state-tracking so we don't
# re-send static content:
#
#   1. Strip blank tty rows from the map view.
#   2. Glyph-run encode long runs of `.` and `#` in the map view.
#   3. Inventory diff-only: emit "(unchanged)" when the inventory letter set
#      hasn't changed since last render (only meaningful if `state` is
#      threaded through; the helper still works with state=None).
#
# Combined target: ~30-40% token reduction on map-heavy turns.
# Toggle off by passing compact=False (e.g. for debugging / replay viewer).


#: tty row 0 is NetHack's message line, not part of the dungeon. Dropping it
#: makes displayed row index == map row index == the `y` every tool takes, which
#: is what lets an agent locate itself by counting rows. BALROG strips the same
#: row for the same reason (`balrog/environments/nle/base.py:184`:
#: `ascii_map = "\n".join(ascii_map.split("\n")[1:])`).
_TTY_MESSAGE_ROWS = 1


def _map_rows_only(map_view: str) -> str:
    """Drop the tty message row so row N of the rendered map is map row N."""
    rows = map_view.split("\n")
    # Only a full tty render (24 rows: message + 21 map + 2 status) carries the
    # message row. Anything shorter is already map-only; leave it alone.
    if len(rows) < 22:
        return map_view
    return "\n".join(rows[_TTY_MESSAGE_ROWS:])


def _render_ascii_map(structured, state) -> str:
    """The `=== MAP ===` body: chars-plane render when available, tty legacy
    render otherwise.

    `nethack_harness.prompt.ascii_map.render_map_from_chars` reads `chars` —
    the game's actual dungeon state, which cannot contain menu/prompt prose
    — instead of `structured.map_view` (built from `tty_chars`, see that
    module's docstring for the bleed this avoids). `state["raw_obs"]` is
    always set by the time a real turn renders (nethack.py sets it before
    calling the turn template); the tty fallback exists only so a
    hand-built StructuredObservation with no `raw_obs` in state (a handful
    of synthetic-state unit tests that pass `include_map=False` anyway)
    still renders something rather than raising.
    """
    if state is not None:
        raw = state.get("raw_obs")
        chars = getattr(raw, "chars", None) if raw is not None else None
        if chars is not None:
            from nethack_harness.prompt.ascii_map import render_map
            # `render_map` (not `render_map_from_chars`) so the MAP is the
            # RECONCILED engine grid: `chars` alone drops terrain the engine's
            # own reveal overlay emitted once and then stopped repeating. See
            # prompt/engine_grid.py.
            return render_map(raw)
    return _map_rows_only(structured.map_view)


def _strip_blank_rows(map_view: str) -> str:
    """Drop fully-blank rows; trim trailing whitespace per row."""
    out = []
    for row in map_view.splitlines():
        r = row.rstrip()
        if r:
            out.append(r)
    return "\n".join(out)


def _glyph_run_encode(map_view: str, min_run: int = 5) -> str:
    """Replace runs of `.` (floor) or `#` (corridor) of length >= min_run
    with `<ch>{N}`. Lossless; reversible. Saves ~15-25% on dungeon-row
    length once corridors are visible.
    """
    import re
    def _sub(m):
        ch = m.group(0)[0]
        return f"{ch}{{{len(m.group(0))}}}"
    pattern = re.compile(r"\.{" + str(min_run) + r",}|#{" + str(min_run) + r",}")
    return "\n".join(pattern.sub(_sub, row) for row in map_view.splitlines())


def _inventory_fingerprint(inventory) -> tuple:
    """Cheap hashable signature so we can diff inventories across turns."""
    return tuple((it.letter, it.description) for it in inventory)


def _run_length_encode_messages(messages) -> list:
    """Collapse consecutive identical messages into `text (xN)`. Saves tokens
    on combat spam ("You hit the kobold. You hit the kobold. ..."). Order-
    preserving; only consecutive duplicates collapse.
    """
    if not messages:
        return []
    out = []
    last = None
    run = 0
    for m in messages:
        if m == last:
            run += 1
        else:
            if last is not None:
                out.append(f"{last} (x{run})" if run > 1 else last)
            last = m
            run = 1
    if last is not None:
        out.append(f"{last} (x{run})" if run > 1 else last)
    return out


# ---------- variant-specific renderers (wave-1 baselines) ----------


def _glyph_to_words(ch: str) -> str:
    """Tiny vocab mapper from ASCII glyph -> natural-language token. Used by
    variant B (BALROG NLE text wrapper)."""
    _TBL = {
        ".": "floor", "#": "corridor", "|": "wall", "-": "wall", "+": "door",
        ">": "stairs-down", "<": "stairs-up", "_": "altar", "{": "fountain",
        "}": "pool", "\\": "throne", "$": "gold", "%": "food",
        "[": "armor", ")": "weapon", "(": "tool", "*": "rock", "?": "scroll",
        "!": "potion", "=": "ring", "/": "wand", "@": "you", " ": "void",
    }
    if ch in _TBL:
        return _TBL[ch]
    if ch.isalpha():
        return f"creature({ch})"
    return f"glyph({ch})"


def _format_obs_balrog(structured, journal, state, journal_max_chars: int) -> str:
    """Variant B: BALROG / NLE language wrapper. No ASCII grid; render a
    natural-language description of the scene, status, inventory, messages.
    Tests the hypothesis that the ASCII map adds little for small LLMs."""
    lines: list[str] = []
    if journal is not None and not journal.is_empty():
        lines.append("=== JOURNAL ===")
        lines.append(journal.render(max_chars=journal_max_chars))
        lines.append("")
    s = structured.status or {}
    lines.append("=== STATUS ===")
    lines.append(
        f"HP {s.get('hitpoints','?')}/{s.get('max_hitpoints','?')}  "
        f"AC {s.get('armor_class','?')}  "
        f"Dlvl {s.get('depth','?')}  "
        f"Turn {s.get('time','?')}  "
        f"XP {s.get('experience_level','?')}"
    )
    if "x" in s and "y" in s:
        lines.append(f"Position: ({s['x']},{s['y']})")
    c = structured.character or {}
    if c:
        lines.append(f"Character: {c.get('role','?')} ({c.get('race','?')}, {c.get('alignment','?')})")
    lines.append("")
    if structured.inventory:
        lines.append("=== INVENTORY ===")
        # Corpses carry no visible age in NetHack, and rot is what kills the
        # agent in ~18% of deaths. Reconstruct it from the kill log.
        _gt = None
        try:
            _gt = (getattr(structured, "status", None) or {}).get("time")
            from nethack_harness.prompt.corpse_age import note_kills, annotate_corpse
            note_kills(state, getattr(structured, "messages", None) or [], _gt)
        except Exception:
            annotate_corpse = None
        for item in structured.inventory:
            desc = item.description
            if annotate_corpse is not None:
                try:
                    desc = annotate_corpse(desc, state, _gt)
                except Exception:
                    pass
            lines.append(f"  {item.letter}: {desc}")
        lines.append("")
    under = getattr(structured, "under_player", None)
    if under:
        lines.append(f"=== UNDER PLAYER === {under}")
        lines.append("")
    adj = getattr(structured, "adjacent", None) or {}
    if adj:
        order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        bits = []
        for d in order:
            tile = adj.get(d, " ")
            bits.append(f"{d}={_glyph_to_words(tile[0] if tile else ' ')}")
        lines.append("=== ADJACENT ===")
        lines.append("  " + ", ".join(bits))
        lines.append("")
    # Visible features (parsed from the grid) — the "what's around me" block
    # without forcing the model to read ASCII.
    if state is not None and "raw_obs" in state:
        try:
            from nethack_harness.prompt.features import (
                monsters_in_sight, visible_feature_strings,
            )
            features = visible_feature_strings(state["raw_obs"])
            if features:
                lines.append("=== VISIBLE FEATURES ===")
                for f in features:
                    lines.append(f"  - {f}")
                lines.append("")
            hostiles = monsters_in_sight(state["raw_obs"])
            if hostiles:
                lines.append("=== VISIBLE MONSTERS ===")
                for h in hostiles:
                    lines.append(f"  - {h}")
                lines.append("")
        except Exception:
            pass
    if structured.messages:
        lines.append("=== MESSAGES ===")
        for m in _run_length_encode_messages(structured.messages):
            lines.append(f"  {m}")
        lines.append("")
    return "\n".join(lines)


def _format_obs_glyphbox(structured, journal, state, journal_max_chars: int) -> str:
    """Variant G: Glyphbox-style obs. ASCII map + explicit player coords +
    adjacent-tile descriptions + visible-hostile list + inventory + messages.
    Closest analog to Ken Wang's Glyphbox harness."""
    # Reuse the canonical formatter (it already emits ADJACENT and VISIBLE
    # GLYPHS/FEATURES blocks). The Glyphbox delta is the *intent* to pair
    # this obs with the code-mode tool — toggled via interface="code"
    # in load_environment, not here.
    return format_observation_as_chat(
        structured, journal, state=state, compact=True,
        journal_max_chars=journal_max_chars,
    )


# ---------- Experiment 1 baseline encodings (faithful ports — STUBS) ----------
#
# The `G` (Glyphbox) and `N` (NetPlay) variants above are *approximations* built
# on OUR observation extraction: `G` reuses `format_observation_as_chat`, and `N`
# reuses the canonical ASCII grid and only swaps the skill set. They are NOT the
# baselines' native observation encodings, so a token-efficiency / long-horizon
# comparison against them is apples-to-oranges (we are comparing our render under
# a different tool set, not the prior framework's own state serialization).
#
# For Experiment 1 (Encoding Ablations) we want an apples-to-apples comparison of
# the *encodings themselves*: our structured maps (B1/JSON/TOON) vs. the exact
# textual state each prior framework feeds its LLM. These two functions are the
# seam for that. They deliberately raise NotImplementedError until a faithful
# port lands, so the `NETPLAY` / `GLYPHBOX` variants are registered and routable
# (see prompt_spec.VARIANT_REGISTRY and tools/encoding_eval/prime_runner.py) but
# fail loudly rather than silently emitting our own render mislabeled as a
# baseline.


def _format_obs_netplay(structured, journal, state, journal_max_chars: int) -> str:
    """Variant NETPLAY: faithful port of NetPlay's native observation text.

    Source: Jeurissen et al., "Playing NetHack with LLMs" (CoG 2024),
    github.com/CommanderCero/NetPlay — `nethack_agent/agent.py ::
    describe_current_state()` rendered over the persistent
    `nethack_agent/tracking.py :: Level` model. See docs/netplay-vs-our-harness.md
    for the full pseudocode of that model.

    What a faithful port must emit (NO ASCII grid — NetPlay never shows one):
      * A natural-language enumeration of the ROOM / CORRIDOR graph the agent has
        discovered (nodes + exits), derived from the persistent `Level.features`
        + `Level.graph`, not from a per-step glyph grid.
      * Per-tile memory surfaced as prose: which regions are fully explored vs.
        still border unseen rock (`Level.has_seen`), open/closed doors and
        `door_open_attempts`, `search_count` hotspots.
      * Visible + remembered items and monsters as a described list (kind, rough
        location relative to the player / room), the message log, inventory, and
        blstats status line.

    Why it is a stub: NetPlay's description is a projection of a *stateful* Level
    model that accumulates across turns. Our harness rebuilds a stateless grid
    each step (docs/netplay-vs-our-harness.md §2), so a faithful port must either
    (a) port `tracking.Level` (has_seen / room graph / search_count / door
    attempts) and render its `describe_current_state`, or (b) vendor NetPlay's
    renderer against our `raw_obs`. Until then this raises so the variant is
    registered but never silently substitutes our own text.
    """
    raise NotImplementedError(
        "NETPLAY baseline encoding is a stub. Port NetPlay's describe_current_state "
        "over a persistent Level model (see docstring + docs/netplay-vs-our-harness.md) "
        "before running the NETPLAY variant. The `N` variant is NOT this — it is our "
        "canonical ASCII obs with NetPlay's skill set only."
    )


def _format_obs_glyphbox_native(structured, journal, state, journal_max_chars: int) -> str:
    """Variant GLYPHBOX: faithful port of Glyphbox's native observation text.

    Source: Ken (Ken Wang) Glyphbox harness, github.com/kenforthewin/glyphbox.
    See nethack_harness/tools/code_mode.py (which already implements Glyphbox's
    *code-execution action surface*) and nethack_harness/navigation/pathfinding.py
    (which notes where we deliberately diverge from glyphbox's glyph routing).

    What a faithful port must emit (Glyphbox's own state serialization):
      * Glyphbox's glyph->text map serialization of the observed grid using
        `nle.nethack.glyph_id_to_...` class routing (its native scheme), rather
        than our cleaned `glyph_clean_chars` LUT — the token footprint differs and
        that difference is exactly what Experiment 1 measures.
      * Its status / inventory / message blocks in Glyphbox's own layout and
        wording, so token counts are comparable to the published harness.
      * Paired with the code-execution tool (interface="code", code_mode.py), which
        IS already the Glyphbox action surface — the observation is the missing half.

    Why it is a stub: the `G` variant reuses OUR `format_observation_as_chat`
    render; a token-efficiency comparison needs Glyphbox's *own* serialization.
    Until that renderer is vendored/ported this raises so GLYPHBOX is registered
    and routable but never silently emits our render mislabeled as the baseline.
    """
    raise NotImplementedError(
        "GLYPHBOX baseline encoding is a stub. Port Glyphbox's native glyph->text "
        "serialization (github.com/kenforthewin/glyphbox; see code_mode.py for its "
        "already-ported action surface) before running the GLYPHBOX variant. The `G` "
        "variant is NOT this — it reuses our canonical formatter."
    )


def _format_obs_summarize_reset(structured, journal, state, journal_max_chars: int) -> str:
    """Variant R: CPP/GPP summarize-and-reset. Same per-turn formatter as
    B1; the difference lives in get_prompt_messages — see _compact_chat_history
    behavior under summarize_and_reset=True, which hard-drops everything
    before the most recent belief-state checkpoint."""
    return format_observation_as_chat(
        structured, journal, state=state, compact=True,
        journal_max_chars=journal_max_chars,
    )


# --------------------------------------------------------------------------- #
# Death announcement (docs/HARNESS_DEFECTS.md 3.6)                             #
# --------------------------------------------------------------------------- #
#
# THE DEFECT. The turn on which the character dies rendered `HP: 0/N`, a normal
# MAP, and whatever HINT the ladder happened to produce -- "Hostile adjacent
# (SE). Call `attack(...)` -- your HP is healthy." was observed on a corpse.
# Nothing in the observation said "you are dead". `state["died"]` is set at the
# BOTTOM of `env_response`, after the render, and the "[Your character is dead
# ...]" note is appended by the *next* `_apply_tool_call` -- by which time every
# tool is gated and the only thing the agent can still do (rollback) is the one
# thing nothing has told it about. So the agent spent its death turn issuing a
# combat order and learned it was dead one call later, from a refusal.
#
# THE FIX. Announce it from the observation renderer, which runs on the death
# turn itself, and detect it from the observation rather than from harness
# bookkeeping so the ordering inside `env_response` cannot matter:
#   * `hitpoints <= 0` in the shaped status -- the same signal nethack.py's own
#     zero-HP fallback calls authoritative,
#   * NetHack's own death-screen phrases in the messages or on the tty,
#   * `state["died"]` / `state["ascended"]`, when the harness got there first.
# Any one is enough; the block is emitted FIRST, above JOURNAL, and the HINT
# ladder is suppressed so the observation cannot simultaneously say "you are
# dead" and "attack the kobold".

#: NetHack's death sequence, in the order the player sees it: the death message,
#: then the DYWYPI prompt, then Final Attributes, then the tombstone. Mirrors
#: `helpers._DEATH_MARKERS` (kept local so `prompt/` does not import the env
#: package) and adds the phrases that appear on the *first* death frame, which
#: is exactly the frame `_DEATH_MARKERS` was too late for.
_DEATH_SCREEN_MARKERS = (
    "You die",
    "You died",
    "killed by",
    "starved to death",
    "petrified by",
    "drowned",
    "Do you want your possessions identified",
    "Do you want to see your attributes",
    "Goodbye ",
    "You made the top ",
)

#: Ascension is the other terminal outcome and must not be reported as death.
_ASCENSION_SCREEN_MARKERS = (
    "ascended to demigod",
    "ascended to demigoddess",
    "You offer the Amulet",
)


def _tty_plane(state):
    raw = (state or {}).get("raw_obs") if state else None
    return getattr(raw, "tty_chars", None) if raw is not None else None


def _tty_flat(state) -> str:
    """The whole tty as ONE string, rows concatenated, for marker containment.

    The death message is frequently NOT in `structured.messages`: NetHack paints
    "You die..." plus the DYWYPI prompt over the map, and the harness's own
    menu/--More-- auto-dismiss loop can consume it before `shape()` runs. The
    tty still carries it, so it is the more reliable place to look.

    `tobytes()` + one decode, not 1,920 `chr()` calls: this runs on every render,
    alive or dead, for a check that is negative on ~99.9% of turns. Rows are
    padded to full width, so a marker never straddles the join.
    """
    tty = _tty_plane(state)
    if tty is None:
        return ""
    try:
        if getattr(tty, "itemsize", 0) == 1:
            return tty.tobytes().decode("latin-1")
    except Exception:
        pass
    try:
        return "".join("".join(chr(int(c)) for c in row) for row in tty)
    except Exception:
        return ""


def _tty_lines(state) -> list[str]:
    """The tty row by row. Only called once the character is already dead."""
    tty = _tty_plane(state)
    if tty is None:
        return []
    try:
        return ["".join(chr(int(c)) for c in row) for row in tty]
    except Exception:
        return []


def _death_cause(structured, state) -> Optional[str]:
    """The game's own words for how the run ended, or None."""
    msgs = list(getattr(structured, "messages", None) or [])
    for msg in reversed(msgs):
        if any(m in msg for m in _DEATH_SCREEN_MARKERS):
            return msg.strip()
    for line in _tty_lines(state):
        line = line.strip()
        if "killed by" in line or "starved to death" in line:
            return line
    return None


def _is_dead(structured, state) -> bool:
    """True when this observation is the one belonging to a dead character."""
    if state is not None and state.get("ascended"):
        return False
    if state is not None and state.get("died"):
        return True
    s = getattr(structured, "status", None) or {}
    hp = s.get("hitpoints")
    if hp is not None:
        try:
            if int(hp) <= 0:
                return True
        except (TypeError, ValueError):
            pass
    haystack = "\n".join(list(getattr(structured, "messages", None) or [])
                         + [_tty_flat(state)])
    if any(m in haystack for m in _ASCENSION_SCREEN_MARKERS):
        return False
    return any(m in haystack for m in _DEATH_SCREEN_MARKERS)


def _game_over_block(structured, state) -> list[str]:
    """`=== GAME OVER ===`, emitted on the death turn itself. [] when alive."""
    if not _is_dead(structured, state):
        return []
    s = getattr(structured, "status", None) or {}
    hp = s.get("hitpoints", 0)
    hp_max = s.get("max_hitpoints", "?")
    where = f"Dlvl {s.get('depth', '?')}"
    when = f"turn {s.get('time', '?')}"
    cause = _death_cause(structured, state)
    out = [
        "=== GAME OVER ===",
        f"YOUR CHARACTER IS DEAD. HP {hp}/{hp_max} on {where} at {when}."
        + (f" {cause}" if cause else ""),
        "The game is over. Every further tool call is REFUSED without touching "
        "the engine — you cannot move, fight, eat, pray or descend, and nothing "
        "you do now changes the outcome.",
    ]
    if "rollback" in published_tools_for(state):
        out.append(
            "ONE action still works: `rollback(n)` rewinds the last n turns and "
            "puts the game back as it was BEFORE you died. If you want to keep "
            "playing, call it now and then choose differently."
        )
    out.append("")
    return out


def _hero_depth(structured) -> int:
    """Current dungeon level, or -1 when the status block has no depth."""
    try:
        return int((structured.status or {}).get("depth", -1))
    except Exception:
        return -1


# Mirrors nethack_harness.tools.skills.eat()'s food-keyword filter exactly, so
# the HINT ladder can never recommend an item the eat() skill would then
# reject, and never claims "no food" when eat() would actually find some.
_FOOD_KEYWORDS = ("food", "corpse", "ration", "fruit", "apple", "pancake", "cookie", "tripe")


def _edible_inventory_items(structured) -> list:
    """Inventory items eat() would accept, in inventory order."""
    inv = getattr(structured, "inventory", None) or []
    return [it for it in inv if any(k in it.description.lower() for k in _FOOD_KEYWORDS)]


def _remember_stairs_down(state, structured, feats) -> None:
    """Memoise every `>` we can see, KEYED BY DEPTH.

    Once the player steps onto `>` the `@` overlay hides it, so without memory
    the agent oscillates on and off the stairs. But (x,y) means a different tile
    on every floor, and the memo is never cleared on descent — so an unkeyed set
    asserts "you are standing on stairs DOWN" on Dlvl 2 at a coordinate that
    held the stairs on Dlvl 1, and `descend` then fails.
    """
    from nethack_harness.prompt.features import stairs_down

    depth = _hero_depth(structured)
    for f in stairs_down(feats):
        state["_seen_stairs_down"].add((depth, f.x, f.y))


def _remembered_stairs_down(state, structured) -> set:
    """Remembered `>` coordinates on the CURRENT floor, as {(x, y)}."""
    depth = _hero_depth(structured)
    out = set()
    for entry in state.get("_seen_stairs_down") or ():
        if len(entry) == 3:
            d, x, y = entry
            if d == depth:
                out.add((x, y))
        elif len(entry) == 2:   # legacy unkeyed entry; treat as current floor
            out.add(tuple(entry))
    return out


def _tried_exits_on_level(state, depth) -> set:
    """Exits already recommended-and-abandoned on the CURRENT floor, as {(x,y)}.

    KEYED BY DEPTH for the same reason `_remembered_stairs_down` is: (x,y) means
    a different tile on every level, and an unkeyed memo would carry Dlvl 1's
    "already tried" doors onto Dlvl 2 and permanently exclude legitimate exits
    there (the class of bug Task 17 fixed for the `>` memo).
    """
    out = set()
    for entry in (state.get("_tried_exits") if state is not None else None) or ():
        if len(entry) == 3:
            d, x, y = entry
            if d == depth:
                out.add((x, y))
    return out


def _mark_exit_tried(state, depth: int, x: int, y: int) -> None:
    """Record that (x,y) on floor `depth` was recommended repeatedly without
    the situation changing, so later renders route around it instead of
    cycling back."""
    if state is None:
        return
    state.setdefault("_tried_exits", set()).add((depth, x, y))


def _descent_status_block(structured, state) -> list[str]:
    """Wave-2 descent-salience block. Diagnosis (see experiment_log.md Wave-2):
    in EVERY failing rollout the down-stairs `>` never appeared in VISIBLE
    FEATURES — the agent wandered/oscillated around the starting room for
    300-1900 game-turns and DIED (usually starvation while Fainting). Runs
    that descended did so in <65 game-turns. So the dominant failure is
    "never reveal/reach `>`, then starve."

    This block makes the descent objective and the time-pressure impossible
    to miss, every turn. It is gated on state["_descent_salient"] (set for
    variants ND/FD) so the baselines are unaffected.

    Emits one of:
      DOWNSTAIRS: VISIBLE at (x,y) — call find_and_descend NOW to path there
                  and descend in one action.
      DOWNSTAIRS: not found yet — call find_and_descend to explore toward
                  unrevealed territory (it auto-paths to `>` the moment it is
                  seen). Avoid repeated search/pickup; they burn game-turns.
    Plus a level-clock warning once the in-game turn count on the current
    level grows large (starvation territory).
    """
    if not state or not state.get("_descent_salient"):
        return []
    out: list[str] = []
    stairs_xy = None
    try:
        from nethack_harness.prompt.features import stairs_down, visible_features
        found = stairs_down(visible_features(state["raw_obs"]))
        if found:
            stairs_xy = (found[0].x, found[0].y)
    except Exception:
        pass
    # Memoized stairs (player may be standing on them, hiding the glyph).
    remembered = _remembered_stairs_down(state, structured)
    if stairs_xy is None and remembered:
        try:
            px = int(structured.status.get("x", -1))
            py = int(structured.status.get("y", -1))
            if (px, py) in remembered:
                stairs_xy = (px, py)
        except Exception:
            pass
        if stairs_xy is None:
            stairs_xy = next(iter(remembered))
    out.append("=== DESCENT STATUS ===")
    if stairs_xy is not None:
        out.append(
            f"DOWNSTAIRS: VISIBLE at {stairs_xy}. Your goal is to descend. "
            f"Call `find_and_descend` NOW — it paths to `>` and descends in "
            f"one action. (Or `descend` if you are already standing on it.)"
        )
    else:
        out.append(
            "DOWNSTAIRS: not found yet on this level. Call `find_and_descend` "
            "to push exploration into unrevealed territory; it auto-walks to "
            "`>` and descends the instant the stairs are seen. Do NOT "
            "loop on `search`/`pickup` — every wasted turn risks starvation."
        )
    # Level clock: NLE in-game turn counter. Starvation deaths in the failing
    # runs clustered at T:600-1900. Warn early so the agent prioritizes descent.
    try:
        t = int(structured.status.get("time", 0))
        if t >= 250:
            out.append(
                f"CLOCK: {t} in-game turns elapsed and still on Dlvl "
                f"{structured.status.get('depth','?')}. You are taking too "
                f"long — descend before hunger kills you."
            )
    except Exception:
        pass
    out.append("")
    return out


# ---------- Wave-3 / E1: frontier-surface obs blocks ----------

# 8-compass bearings shared across the E1 blocks.
_E1_BEARINGS = [
    ("N",  0, -1),
    ("NE", 1, -1),
    ("E",  1,  0),
    ("SE", 1,  1),
    ("S",  0,  1),
    ("SW", -1, 1),
    ("W", -1,  0),
    ("NW", -1, -1),
]


def _e1_bearing(dx: int, dy: int) -> str:
    """Quantize (dx, dy) to one of 8 compass bearings. Uses atan2-style
    bucketing — cheap and deterministic. Returns "@" when dx == dy == 0."""
    if dx == 0 and dy == 0:
        return "@"
    # Choose the bearing whose unit vector has the highest dot product with
    # the (dx, dy) heading. Equivalent to nearest-octant in 22.5° buckets.
    import math
    best = None
    best_dot = -1e9
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    for name, bx, by in _E1_BEARINGS:
        bnorm = math.hypot(bx, by) or 1.0
        dot = (ux * (bx / bnorm)) + (uy * (by / bnorm))
        if dot > best_dot:
            best_dot = dot
            best = name
    return best or "?"


def _e1_classify_frontier(chars, x: int, y: int) -> str:
    """One-word frontier-tile classifier. Reads the rendered glyph; falls
    back to 'tile' on anything exotic."""
    try:
        ch = chr(int(chars[y, x]))
    except Exception:
        return "tile"
    if ch == "#":
        return "corridor"
    if ch == ".":
        return "room edge"
    if ch == "<":
        return "stairs up"
    if ch == ">":
        return "stairs down"
    if ch in "+'":
        return "doorway"
    return "tile"


def _e1_frontiers_block(state) -> list[str]:
    """Render the 3-5 nearest unexplored frontiers (Wave-3 Track C).

    Surfaces the output of nethack_harness.navigation.pathfinding.find_frontiers — which
    the harness already computes for autoexplore — to the model. The
    "(no frontiers ...)" fallback directly cues the search skill, which
    is the documented out for sealed-room rollouts.
    """
    if not state or "raw_obs" not in state:
        return []
    try:
        from nethack_harness.navigation.pathfinding import find_frontiers
        raw = state["raw_obs"]
        chars = getattr(raw, "chars", None)
        if chars is None:
            return []
        blstats = getattr(raw, "blstats", None)
        if blstats is None:
            return []
        px, py = int(blstats[0]), int(blstats[1])
        frontiers = find_frontiers(chars)
    except Exception:
        return []

    out: list[str] = ["=== FRONTIERS ==="]
    if not frontiers:
        out.append("(no frontiers — try `search` for hidden passages)")
        out.append("")
        return out

    # Sort by Chebyshev distance from the agent (matches the A* heuristic
    # used by autoexplore). Cap at 5; cap line length at ~60 chars.
    scored = []
    for (fx, fy) in frontiers:
        d = max(abs(fx - px), abs(fy - py))
        scored.append((d, fx, fy))
    scored.sort(key=lambda t: t[0])
    for (d, fx, fy) in scored[:5]:
        bearing = _e1_bearing(fx - px, fy - py)
        kind = _e1_classify_frontier(chars, fx, fy)
        line = f"({fx}, {fy})  ~{d} steps {bearing}  — {kind}"
        if len(line) > 60:
            line = line[:57] + "..."
        out.append(line)
    out.append("")
    return out


def _e1_exploration_block(state, structured) -> list[str]:
    """Coverage + progress-delta indicator (Wave-3 Track C).

    Surfaces scout_tiles_seen (cumulative tiles revealed on the current
    dlvl) and scout_delta (newly revealed this turn). Persistent 0-delta
    is the oscillation signature we want the model to recognize and
    correct (by switching skills / picking a different frontier).
    """
    if not state:
        return []
    out: list[str] = []
    try:
        dlvl = state.get("max_dlvl_reached", 1)
        # scout_tiles_seen is keyed by (dlvl, x, y); filter to current dlvl.
        tiles = state.get("scout_tiles_seen") or set()
        cur_tiles = sum(1 for k in tiles if isinstance(k, tuple) and len(k) == 3 and k[0] == dlvl)
    except Exception:
        cur_tiles = 0
    # Count frontiers cheaply (re-uses the find_frontiers call cost; if
    # this becomes a hotspot we can memoize). On most maps len(frontiers)
    # is tiny (<20) so the extra walk is fine.
    n_frontiers = 0
    try:
        if "raw_obs" in state:
            from nethack_harness.navigation.pathfinding import find_frontiers
            chars = getattr(state["raw_obs"], "chars", None)
            if chars is not None:
                n_frontiers = len(find_frontiers(chars))
    except Exception:
        pass
    line = f"Explored: {cur_tiles} tiles, {n_frontiers} frontiers open"
    delta = state.get("scout_delta")
    if delta is None:
        line += " — turn 0 (no delta yet)"
    elif delta > 0:
        line += f" — revealed {int(delta)} new tiles"
    else:
        line += " — revealed 0 — retreading"
    out.append("=== EXPLORATION ===")
    out.append(line)
    out.append("")
    return out


def _e1_spatial_belief_block(state, structured) -> list[str]:
    """Replacement for the legacy descent-salience block (Wave-3 Track C).

    Instead of exhorting "descend now!", emit a compact spatial belief:
    bearings to the nearest 3 unexplored frontiers + any known
    stairs-down coordinates. Pure information — no nagging.
    """
    if not state:
        return []
    out: list[str] = ["=== SPATIAL BELIEF ==="]
    try:
        from nethack_harness.navigation.pathfinding import find_frontiers
        raw = state.get("raw_obs")
        chars = getattr(raw, "chars", None) if raw is not None else None
        blstats = getattr(raw, "blstats", None) if raw is not None else None
        if chars is not None and blstats is not None:
            px, py = int(blstats[0]), int(blstats[1])
            frontiers = find_frontiers(chars)
            if frontiers:
                scored = sorted(
                    ((max(abs(fx - px), abs(fy - py)), fx, fy) for (fx, fy) in frontiers),
                    key=lambda t: t[0],
                )[:3]
                bearings = ", ".join(f"{_e1_bearing(fx - px, fy - py)}~{d}" for d, fx, fy in scored)
                out.append(f"Unexplored bearings: {bearings}")
            else:
                out.append("Unexplored bearings: none (level fully revealed)")
    except Exception:
        pass
    # Only THIS floor's remembered stairs: (x,y) is a different tile per level.
    seen = _remembered_stairs_down(state, structured)
    if seen:
        # Cap at 3 coords to keep the line short.
        coords = ", ".join(f"({x},{y})" for (x, y) in sorted(seen)[:3])
        out.append(f"Known stairs DOWN: {coords}")
    else:
        out.append("Known stairs DOWN: none on this floor yet")
    out.append("")
    return out


_VARIANT_FORMATTERS = {
    # variant code -> formatter callable, or None to use the canonical formatter
    "B1": None,
    "B0": None,        # same formatter as B1; B0 just turns compact_obs off via kwargs
    "G": _format_obs_glyphbox,
    "B": _format_obs_balrog,
    "N": None,         # NetPlay differs only in skill_set, not in obs formatter
    # Experiment 1 baseline encodings (faithful ports — STUBS, raise until
    # implemented). Distinct from G/N above, which reuse OUR extraction.
    "NETPLAY": _format_obs_netplay,
    "GLYPHBOX": _format_obs_glyphbox_native,
    "R": _format_obs_summarize_reset,
    "P": None,         # Continual Harness uses canonical formatter + refinement directive
    "CH": None,        # Full Continual Harness — same formatter; addendum/macros injected in get_prompt_messages
    # Wave-2 descent variants share the canonical formatter; the descent-salience
    # block is injected via state["_descent_salient"] (set in setup_state).
    "ND": None,        # NetPlay skill set + descent-salience block + clock warning
    "FD": None,        # find_and_descend autopilot: minimal skill set + salience block
    # Wave-3 Track C: frontier-surface obs. Canonical formatter; the four
    # new blocks (FRONTIERS, EXPLORATION, SPATIAL BELIEF, status delta) are
    # gated on state["_e1_obs"] (set in setup_state when variant == "E1").
    "E1": None,
    # Wave-3 Track C v2: paint frontier-adjacent unexplored tiles with '?'
    # directly on the map. No text blocks — the spatial cue lives in the
    # glyph grid the model already parses. Gated on state["_e2_obs"].
    "E2": None,
}


def _paint_frontiers_on_map(map_view: str, chars, frontiers, cap: int = 40) -> str:
    """Overlay '?' on truly-unseen tiles adjacent to each frontier.

    Frontiers are walkable tiles bordering unexplored space; the adjacent
    unseen tiles are where unrevealed content sits. Painting them as '?'
    gives the model a visual cue in the spatial layout it already reads,
    so the FRONTIERS/EXPLORATION/SPATIAL BELIEF text blocks become
    redundant.

    Only paints where the existing glyph is ' ' (space) — never overwrites
    walls, floor, items, or the agent. Caps the total paint count so a
    pathological map can't blow up the obs.
    """
    if not frontiers or chars is None:
        return map_view
    try:
        from nethack_harness.navigation.pathfinding import is_truly_unseen
    except Exception:
        return map_view
    rows = map_view.split("\n")
    grid = [list(r) for r in rows]
    h = len(grid)
    painted = 0
    seen: set[tuple[int, int]] = set()
    for (fx, fy) in frontiers:
        if painted >= cap:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = fx + dx, fy + dy
                if (nx, ny) in seen:
                    continue
                if not (0 <= ny < h):
                    continue
                row = grid[ny]
                if not (0 <= nx < len(row)):
                    continue
                if row[nx] != " ":
                    continue
                try:
                    if not is_truly_unseen(chars, nx, ny):
                        continue
                except Exception:
                    continue
                row[nx] = "?"
                seen.add((nx, ny))
                painted += 1
                if painted >= cap:
                    break
            if painted >= cap:
                break
    return "\n".join("".join(r) for r in grid)


def format_observation_as_chat(
    structured,
    journal: Optional[Journal] = None,
    state: Optional[dict] = None,
    compact: bool = True,
    journal_max_chars: int = 2000,
    include_map: bool = True,
    include_local: bool = True,
    sparse_entities: bool = False,
) -> str:
    """Render a StructuredObservation as a text block for the user message.

    When `state` is threaded through, we deduplicate static content
    across turns. `compact=False` disables all token-savers (used by tests
    and the replay viewer to inspect raw content).

    Task 18 Step 2: `state["_self_dispatch"]` (set in `setup_state` from the
    env's `self_dispatch` constructor flag) gates the JOURNAL block and the
    HINT ladder off. Claude Code and Prime Agent manage their own reasoning
    and memory internally, so these are redundant scaffolding for them — the
    trace analyses found `recall`/`pin_objective` never called across 1,173
    Claude Code calls. The control arm never sets this flag (default False)
    and keeps both blocks unchanged: it is the v0 baseline and must not drift.
    """
    self_dispatch = bool(state and state.get("_self_dispatch"))
    lines: list[str] = []
    # Death is announced on the turn it happens, ABOVE everything else, and it
    # suppresses the HINT ladder so the observation cannot say "you are dead"
    # and "attack the hostile to your SE" in the same breath. See
    # `_game_over_block` for why this is detected from the observation rather
    # than from `state["died"]` (which is set after this render).
    game_over = _game_over_block(structured, state)
    lines.extend(game_over)
    if journal is not None and not journal.is_empty() and not self_dispatch:
        # Diff-only journal: when state is threaded through and the journal
        # hasn't changed since last render, emit "(unchanged)" instead of the
        # full block. Saves ~journal_max_chars/turn on stretches with no
        # journal writes. Belief-state ticks every 25 turns will refresh.
        cur_keys = tuple(sorted(journal.notes.keys()))
        cur_fp = (journal.objective, cur_keys)
        prev_fp = state.get("_journal_fingerprint") if state is not None else None
        lines.append("=== JOURNAL ===")
        if compact and prev_fp == cur_fp:
            # Diff-only: omit notes, but ALWAYS surface the pinned objective.
            # After history compaction strips turn 1, the agent has no way to
            # recall its goal from context — and the 9071d001 trace showed
            # zero `recall` calls, so the model wasn't retrieving it either.
            if journal.objective:
                lines.append(f"Objective: {journal.objective}")
                lines.append("(notes unchanged since last turn)")
            else:
                lines.append("(unchanged since last turn)")
        else:
            lines.append(journal.render(max_chars=journal_max_chars))
        if state is not None:
            state["_journal_fingerprint"] = cur_fp
        lines.append("")
    # Wave-2: descent-salience block (variants ND/FD). Placed immediately after
    # the journal/objective so it's the first concrete thing the model reads.
    lines.extend(_descent_status_block(structured, state))
    # Wave-3 Track C (variant E1): frontier + coverage + spatial-belief
    # blocks. Gated on state["_e1_obs"] so the legacy variants stay
    # bit-identical. The SPATIAL BELIEF block REPLACES (does not augment)
    # the legacy descent-salience block — but E1 sets _descent_salient=False
    # in setup_state, so the call above is a no-op for E1.
    if state is not None and state.get("_e1_obs"):
        lines.extend(_e1_frontiers_block(state))
        lines.extend(_e1_exploration_block(state, structured))
        lines.extend(_e1_spatial_belief_block(state, structured))
    if include_map and sparse_entities:
        # Entity-only map: the ASCII grid is NOT rendered at all. The whole
        # point is to drop terrain, so falling through to the grid below would
        # make this the most expensive encoding rather than the cheapest.
        from nethack_harness.prompt.sparse_map import sparse_entity_map
        lines.append("=== MAP (entities only; terrain omitted) ===")
        lines.append(sparse_entity_map(structured, state))
        lines.append("")
    elif include_map:
        lines.append("=== MAP ===")
        map_view = _render_ascii_map(structured, state)
        # Wave-3 Track C v2 (variant E2): paint '?' over truly-unseen tiles
        # adjacent to each frontier, directly on the map. Done BEFORE compaction
        # so glyph-RLE still applies to floor/corridor runs — and AFTER the
        # message row is dropped, because frontier coordinates are map-frame.
        if state is not None and state.get("_e2_obs"):
            try:
                from nethack_harness.navigation.pathfinding import find_frontiers
                raw = state.get("raw_obs")
                chars = getattr(raw, "chars", None) if raw is not None else None
                if chars is not None:
                    frontiers = find_frontiers(chars)
                    map_view = _paint_frontiers_on_map(map_view, chars, frontiers)
            except Exception:
                pass
        if compact:
            map_view = _strip_blank_rows(map_view)
            map_view = _glyph_run_encode(map_view)
        lines.append(map_view)
        lines.append("")
    lines.append("=== STATUS ===")
    s = structured.status
    # Include max-dlvl-reached when state is threaded so the model can see
    # progression at a glance (e.g. on a return-to-prev-level via stairs up).
    max_dlvl = state.get("max_dlvl_reached") if state else None
    dlvl_part = f"Dlvl: {s.get('depth', '?')}"
    if max_dlvl is not None and max_dlvl > s.get("depth", 0):
        dlvl_part = f"Dlvl: {s.get('depth', '?')} (max reached: {max_dlvl})"
    pos_part = ""
    if "x" in s and "y" in s:
        pos_part = f"  Pos: ({s['x']},{s['y']})"
    # NLE hunger_state: 0=Satiated, 1=Normal, 2=Hungry, 3=Weak, 4=Fainting, 5=Starving.
    # Only surface when non-normal so the status line stays compact.
    _HUNGER_LABEL = {0: "Satiated", 2: "Hungry", 3: "Weak", 4: "Fainting", 5: "Starving"}
    hunger_part = ""
    h = s.get("hunger_state")
    if h is not None and h in _HUNGER_LABEL:
        hunger_part = f"  Hunger: {_HUNGER_LABEL[h]}"
    lines.append(f"HP: {s.get('hitpoints', '?')}/{s.get('max_hitpoints', '?')}  "
                 f"AC: {s.get('armor_class', '?')}  "
                 f"{dlvl_part}  "
                 f"Turn: {s.get('time', '?')}  "
                 f"XP: {s.get('experience_level', '?')}  "
                 f"$: {s.get('gold', 0)}{pos_part}{hunger_part}")
    c = structured.character
    if c:
        lines.append(f"Character: {c.get('role', '?')} ({c.get('race', '?')}, {c.get('alignment', '?')})")
    lines.append("")
    if structured.inventory:
        prev_fp = state.get("_inv_fingerprint") if state is not None else None
        cur_fp = _inventory_fingerprint(structured.inventory)
        if compact and prev_fp == cur_fp:
            lines.append("=== INVENTORY (unchanged) ===")
        else:
            lines.append("=== INVENTORY ===")
            # Corpse rot is invisible in NetHack and kills the agent in ~18% of
            # deaths, so annotate age here. NOTE: this is the block
            # `format_observation_as_chat` actually uses -- the near-identical
            # one earlier in this file belongs to a different render path, and
            # patching only that one left the feature silently dead.
            _gt2 = (getattr(structured, "status", None) or {}).get("time")
            _ann = None
            try:
                from nethack_harness.prompt.corpse_age import note_kills, annotate_corpse as _ann
                note_kills(state, getattr(structured, "messages", None) or [], _gt2)
            except Exception:
                _ann = None
            for item in structured.inventory:
                _d = item.description
                if _ann is not None:
                    try:
                        _d = _ann(_d, state, _gt2)
                    except Exception:
                        pass
                lines.append(f"  {item.letter}: {_d}")
        if state is not None:
            state["_inv_fingerprint"] = cur_fp
        lines.append("")
    if include_local:
        # UNDER PLAYER: critically tells the agent what tile @ is hiding.
        # Especially important for stairs (`>` down vs `<` up).
        under = getattr(structured, "under_player", None)
        if under:
            lines.append(f"=== UNDER PLAYER === {under}")
            lines.append("")
        adj = getattr(structured, "adjacent", None) or {}
        if adj:
            # Cheap "what's around me" block; saves the model from parsing the
            # map for adjacent tiles. Always emitted (even compact=False) — it's
            # a strictly additive signal worth ~30 tokens.
            order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            adj_line = " ".join(f"{d}={adj.get(d, '?')}" for d in order)
            if not sparse_entities:
                lines.append(f"=== ADJACENT === {adj_line}")
            lines.append("")
        # NEXT-ACTION HINT: the model kept missing the moment to descend or
        # attack adjacent hostiles. If on stairs down, say so. Else if stairs
        # down adjacent, say which direction. Else if a letter glyph (monster)
        # is adjacent and HP is healthy, suggest attack. Strictly directive;
        # the model still has to call the tool.
        hint = None
        # Stairs-memory override: if the player is currently standing on a tile
        # we've previously observed as `>`, suggest descend. Fires above all
        # other hints (including HP-critical retreat) because (a) descending is
        # cheap, (b) the alternative is the oscillation loop.
        if state is not None and "_seen_stairs_down" in state and structured.status:
            try:
                px = int(structured.status.get("x", -1))
                py = int(structured.status.get("y", -1))
                if (px, py) in _remembered_stairs_down(state, structured):
                    hint = (
                        f"You are standing on stairs DOWN at ({px},{py}) — call "
                        f"`descend` now. The `>` glyph is hidden under your `@`."
                    )
            except Exception:
                pass
        # HP-critical override — fires regardless of stairs / monsters.
        if structured.status:
            hp = structured.status.get("hitpoints", 0)
            hp_max = structured.status.get("max_hitpoints", 1) or 1
            if hp / hp_max < 0.3 and hp > 0:
                # NB: no "`search(times=20)` to rest" here. Resting costs the
                # same in-game clock as walking, and the artifact ends with two
                # back-to-back `Searched x20.` at HP 13/32 while Hungry — the
                # prompt's own starvation warning and its rest advice were in
                # direct conflict.
                hint = (
                    f"HP critical ({hp}/{hp_max}). Options in order: `engrave_elbereth` "
                    f"(scares most monsters) → retreat toward known-safe ground → "
                    f"`pray` if not on cooldown. Avoid melee until HP is back "
                    f"above 70%. Do NOT rest with `search` while Hungry or worse."
                )
            else:
                h = structured.status.get("hunger_state")
                if h is not None and h >= 3:
                    # Weak (3) or worse — agent will start losing HP unless they eat.
                    label = ("Weak", "Fainting", "Starving")[min(h - 3, 2)]
                    edible = _edible_inventory_items(structured)
                    if edible:
                        # Name the actual letter — 40/40 observed turns told the
                        # agent to eat a literal `<food letter>` placeholder,
                        # which no tool call accepts.
                        it = edible[0]
                        also = (f" (also: {', '.join(e.letter for e in edible[1:3])})"
                                if len(edible) > 1 else "")
                        hint = (
                            f"Hunger is at {label}. You have food: {it.letter} "
                            f"({it.description}){also}. Call "
                            f"`eat(item=\"{it.letter}\")` now."
                        )
                    else:
                        # No edible item exists — telling the agent to `eat`
                        # anyway is unactionable (this is the bug: 40/40 traced
                        # turns did exactly that, and 2/5 seeds starved). Say
                        # plainly there is no food and give a reachable action.
                        hint = (
                            f"Hunger is at {label}. No food in inventory — "
                            "nothing to eat. `pray` once for divine aid (skip "
                            "if you've prayed recently), or push toward "
                            "unexplored ground / the next floor to find a "
                            "corpse or ration."
                        )
        if hint is None and under and "stairs DOWN" in under:
            hint = "You are on stairs down. Call `descend` now."
        elif hint is None and under and under.startswith("on tile:"):
            # Item under the player — suggest pickup.
            hint = f"Item here ({under}). Call `pickup` to grab it before moving on."
        else:
            for d, tile in adj.items():
                if "stairs DOWN" in tile:
                    hint = (
                        f"Stairs down are one step {d}. Step onto them, then "
                        f"call `descend`."
                    )
                    break
            if hint is None:
                # Adjacent letter glyph == hostile (the obs renderer labels
                # monsters as their letter). HP-aware: only suggest engagement
                # when above 50% HP; otherwise suggest retreat.
                mon_dir = None
                for d, tile in adj.items():
                    if len(tile) >= 1 and tile[0].isalpha() and tile not in ("@",):
                        # Skip pets — extract_adjacent now flags them when glyphs
                        # are available. Recommending `attack` on a pet would
                        # trigger the "really attack" peaceful prompt and damage
                        # alignment if confirmed.
                        if "PET" in tile:
                            continue
                        mon_dir = d
                        break
                if mon_dir is not None and structured.status:
                    hp = structured.status.get("hitpoints", 0)
                    hp_max = structured.status.get("max_hitpoints", 1) or 1
                    if hp / hp_max >= 0.5:
                        hint = f"Hostile adjacent ({mon_dir}). Call `attack(direction=\"{mon_dir}\")` — your HP is healthy."
                    else:
                        hint = f"Hostile adjacent ({mon_dir}) and HP is low ({hp}/{hp_max}). Consider `engrave_elbereth` or retreat to safer ground."
                # Stairs visible (but not adjacent): proactively suggest move_to.
                # Trace 9071d001 had stairs visible for many turns without the
                # agent navigating to them — it kept autoexploring.
                if hint is None and state is not None and "raw_obs" in state:
                    try:
                        from nethack_harness.prompt.features import (
                            stairs_down, visible_features,
                        )
                        found = stairs_down(visible_features(state["raw_obs"]))
                        if found:
                            f0 = found[0]
                            # Only recommend walking there if a route actually
                            # exists. Under `tune.reveal_map` the whole level is
                            # visible from turn 1, so the stairs are frequently
                            # SEEN long before they are REACHABLE -- measured: a
                            # revealed rollout was handed this identical hint for
                            # 28 consecutive turns while every move_to bounced off
                            # `It's solid stone.`, ending at Dlvl 1 with
                            # descent_reward 0. Under fog the hint simply never
                            # fired that early, so this only bites the revealed
                            # arms. Unknown reachability keeps the old text.
                            reachable = True
                            try:
                                from nethack_harness.tools.netplay_true import (
                                    get_agent,
                                )
                                _ag = get_agent(state["env"])
                                reachable = (
                                    _ag.get_path_to(f0.x, f0.y) is not None
                                )
                            except Exception:
                                reachable = True
                            if reachable:
                                hint = (
                                    f"Stairs DOWN visible at ({f0.x},{f0.y}). "
                                    f"Call `move_to(x={f0.x}, y={f0.y})` to walk "
                                    "to them, then `descend`."
                                )
                            else:
                                hint = (
                                    f"Stairs DOWN are visible at ({f0.x},{f0.y}) "
                                    "but NO ROUTE to them is known yet — walking "
                                    "straight there will fail. Find a way through "
                                    "first: open a closed door on this room's "
                                    "wall, or explore toward them."
                                )
                    except Exception:
                        pass
        # Don't let secondary overrides clobber the standing-on-stairs hint.
        _on_stairs_override = bool(hint and "standing on stairs DOWN" in hint)
        # Pet-blocking detection: when the message buffer says "X is in the way!"
        # the move failed because the pet/peaceful occupies the tile. Trace
        # 9071d001 had the model stuck in long pet-blocking loops. Override any
        # weaker hint with a clear "go around" directive.
        if structured.messages and not _on_stairs_override:
            for msg in structured.messages[-4:]:
                if "is in the way" in msg:
                    hint = (
                        "A pet/peaceful is blocking your move. Walk a perpendicular "
                        "direction first to let it pass, or `search(times=1)` to "
                        "wait one turn while it moves."
                    )
                    break
        # Locked-door detection: NLE prints "This door is locked." when the
        # agent tries to move INTO a closed-locked `+`. Find the adjacent `+`
        # direction and tell the model to kick. Without this hint, traces show
        # the model giving up on the door and wandering / eating randomly.
        if structured.messages and not _on_stairs_override:
            for msg in structured.messages[-4:]:
                if "door is locked" in msg or "The door is locked." in msg:
                    door_dir = None
                    for d, tile in adj.items():
                        if tile.startswith("+") or tile == "+":
                            door_dir = d
                            break
                    if door_dir is not None:
                        hint = (
                            f"Door to {door_dir} is LOCKED. Call `kick(direction=\"{door_dir}\")` "
                            f"(may take 2-5 tries) to break it open, then walk through."
                        )
                    else:
                        hint = (
                            "A door is locked. Step adjacent to it, then "
                            "`kick(direction=...)` (2-5 tries) to break it open."
                        )
                    break
        # No stairs DOWN visible: route toward the nearest EXIT of the room.
        #
        # This block used to open with "only exit is a door at (x,y)" — a claim
        # that was routinely false (artifact turn 13 asserted a single exit on a
        # turn whose own feature list carried five) because the ranking used
        # `re.search` over each feature *string*, and a string packs up to three
        # coordinate pairs, so every door but the first was invisible to it. It
        # also called open doorways "locked" candidates and invited the agent to
        # kick tiles it could have walked through. Now it ranks over the
        # structured feature list, states how many exits there are, and only
        # mentions kicking for a genuinely closed door.
        # Set only when this render's hint is the exit-navigation one below,
        # so the repeat-escalation block can mark that SPECIFIC exit "tried"
        # once it has been repeated without progress — instead of the model
        # cycling back to it later (13,10 -> 26,10 -> 34,4 -> ... -> 13,10 in
        # the traced run, 26% of turns carrying a stale "not working" warning
        # that never actually changed the recommendation).
        exit_hint_target: Optional[tuple[int, int, int]] = None
        if hint is None and not _on_stairs_override and state is not None and "raw_obs" in state:
            try:
                from nethack_harness.prompt.features import (
                    DOOR_CLOSED, exits, nearest_exit, stairs_down, visible_features,
                )
                feats = visible_features(state["raw_obs"])
                if not stairs_down(feats) and structured.status:
                    all_exits = exits(feats)
                    px = int(structured.status.get("x", 0))
                    py = int(structured.status.get("y", 0))
                    depth = _hero_depth(structured)
                    tried = _tried_exits_on_level(state, depth)
                    untried = [f for f in all_exits if (f.x, f.y) not in tried]
                    best = nearest_exit(untried, px, py) if untried else None
                    if best is not None:
                        n = len(all_exits)
                        kind = ("a closed door" if best.label == DOOR_CLOSED
                                else "an open doorway")
                        extra = (" If it says \"locked\", `kick` toward it."
                                 if best.label == DOOR_CLOSED else "")
                        others = f" ({n} exits visible.)" if n > 1 else ""
                        hint = (
                            f"No `>` visible. Nearest way out is {kind} at "
                            f"({best.x},{best.y}) — `move_to(x={best.x}, "
                            f"y={best.y})`.{extra}{others}"
                        )
                        exit_hint_target = (depth, best.x, best.y)
                    elif all_exits:
                        # Every exit on this level was already recommended and
                        # abandoned (see the escalation block below) — cycling
                        # through them again is the thrash this fixes.
                        hint = (
                            "Every visible exit from this area has already "
                            "been tried without progress. Call "
                            "`search(times=10)` for a hidden passage instead "
                            "of retrying a known exit."
                        )
            except Exception:
                pass
        # A hint the agent has already followed twice without the game state
        # changing is not advice, it is noise: artifact turns 13-20 carried one
        # byte-identical hint eight turns running while the agent stayed stuck.
        # Escalate instead of repeating.
        if hint and state is not None:
            prev = state.get("_last_hint")
            repeats = int(state.get("_hint_repeats", 0)) + 1 if hint == prev else 1
            state["_last_hint"] = hint
            state["_hint_repeats"] = repeats
            if repeats >= 3:
                hint = (
                    f"{hint} [This same suggestion has now been made {repeats} "
                    "turns running and the situation has not changed — it is not "
                    "working. Try a different frontier, a different exit, or "
                    "`search` for a hidden passage.]"
                )
                if exit_hint_target is not None:
                    _mark_exit_tried(state, *exit_hint_target)
        if hint and not self_dispatch:
            hint = _fix_hint_vocabulary(hint, published_tools_for(state))
        # A dead character has no next action; the HINT ladder is HP/hunger/
        # adjacency advice that is meaningless (and actively misleading) once the
        # run is over. GAME OVER above already says what to do.
        if game_over:
            hint = None
        if hint and not self_dispatch:
            lines.append(f"=== HINT === {hint}")
            lines.append("")
    # Hostiles-in-sight + VISIBLE FEATURES: render in BOTH compact and
    # non-compact modes. Trace 5/16 (no-compact, 3 seeds, Qwen3.5-9B) showed
    # the agent NEVER calling `descend` because the pre-parsed feature block
    # was gated to compact mode only. Non-compact agents have to scan the
    # ASCII grid themselves and routinely confuse `<` for `>` on dense maps.
    from nethack_harness.prompt.features import (
        format_features, monsters_in_sight, stairs_down, visible_features,
    )
    if state is not None and "raw_obs" in state:
        try:
            feats = visible_features(state["raw_obs"])
            # Memoize stairs DOWN coords across turns so a subsequent step
            # ONTO the stairs (which hides `>` under `@`) still recognizes
            # the descend opportunity. These are map-frame coords and are
            # compared against map-frame `blstats` above — before the frame
            # fix the memo was tty-frame, so the "you are standing on stairs"
            # hint fired one tile NORTH of the stairs and `descend` then failed.
            if "_seen_stairs_down" in state:
                _remember_stairs_down(state, structured, feats)
            features = format_features(feats)
            # Statues belong HERE, in the features area -- not on the map.
            # A statue draws the monster's letter, and that glyph is the game's
            # own output: rewriting it would make our ASCII/JSON map an
            # unfaithful render of the engine, which is the one thing the map
            # must never be. But nothing else in the observation mentions
            # statues either (they are excluded from `glyph_is_monster`, from
            # the NetPlay tracker, and from the feature scan), so the agent
            # attacks them and gets "There is no monster at (x,y)" at zero
            # game-turn cost. Naming them in the features list is derived
            # knowledge in the place derived knowledge goes.
            try:
                from nethack_harness.prompt.features import statues as _statues
                features = list(features) + _statues(state["raw_obs"])
            except Exception:
                pass
            if features:
                if not sparse_entities:
                    lines.append(f"=== VISIBLE FEATURES === {'; '.join(features)}")
                    lines.append("")
            hostiles = monsters_in_sight(state["raw_obs"])
            if hostiles and not sparse_entities:
                lines.append(f"=== VISIBLE MONSTERS === {'; '.join(hostiles)}")
                lines.append("")
            # The "unseen" arm of the seen-vs-remembered axis. OPT-IN per
            # variant so the pair differs in exactly this block and nothing
            # else. See features.remembered_monsters for why the earlier
            # always-on version was withdrawn.
            if state.get("_remember_monsters") and not sparse_entities:
                try:
                    from nethack_harness.prompt.features import remembered_monsters
                    _st = (structured.status or {})
                    _rm = remembered_monsters(state, state["raw_obs"],
                                              _st.get("time"), _st.get("depth"))
                    if _rm:
                        lines.append(f"=== REMEMBERED MONSTERS (seen earlier, not visible now) === {'; '.join(_rm)}")
                        lines.append("")
                except Exception:
                    pass
            # (A `REMEMBERED MONSTERS` block lived here. Removed: it was built on
            # the theory that the char plane retains monsters the glyph plane has
            # dropped. A 7,500-step audit found ZERO such cells -- chars and glyphs
            # are two views of one NLE buffer and cannot drift. The block had no
            # death invalidation (it advertised a kobold 22 turns after the agent
            # killed it) and counted LLM turns while STATUS reports game turns,
            # ~15x apart. It manufactured phantom targets rather than preventing
            # them. The real cause is statues / the `I` marker; fix those.

        except Exception:
            pass
    if structured.messages:
        lines.append("=== MESSAGES ===")
        msgs = _run_length_encode_messages(structured.messages) if compact else list(structured.messages)
        for m in msgs:
            lines.append(f"  {m}")
        lines.append("")
    if structured.menu:
        # Menus are auto-dismissed by the harness via ESC after each step;
        # if you see this block, dismissal didn't fully clear (rare).
        lines.append("=== MENU (harness will auto-dismiss; ignore) ===")
        for i, opt in enumerate(structured.menu):
            lines.append(f"  [{i}] {opt.description}")
        lines.append("")
    if structured.inventory_prompt:
        p = structured.inventory_prompt
        # Inventory prompts are auto-dismissed; eat/quaff/read take an `item`
        # arg and bundle the selection in-skill, so this block should not
        # normally appear.
        lines.append(f"=== PROMPT: {p['action']} (harness will auto-dismiss; pass `item` to eat/quaff/read) ===")
        for i, item in enumerate(p["items"]):
            lines.append(f"  [{i}] {item.description}")
        lines.append("")
    return "\n".join(lines)


# ---------- the env class ----------

