---
name: nethack
description: Play NetHack. Every move, attack and descent goes through this skill's async tools — and through `netplay`, a Python package of your own that you can read, rewrite and extend.
---

# NetHack — tool API and your own policy code

Two layers, and the difference between them matters.

**`nethack`** is the game boundary — seven primitives, listed below. That is the
complete set of tools: there is nothing else to call, and nothing else to find.
You cannot edit them.

**`netplay`** is your own code. It is ordinary Python on this kernel's
`sys.path`. It ships with exactly THREE policies — `move_to(x, y)`,
`explore()` and `attack(x, y)`, the code twins of the three retired composite
tools — and **everything else is yours to write**. Any policy you build there
persists into the next episode.

So: the primitives are fixed and the skills built on them are yours. If a
policy plays badly, the fix is to rewrite the policy, not to look for a better
tool.

Every tool is **async** — always `await`. The return value is the rendered
observation (`=== MAP ===`, `=== STATUS ===`, `=== INVENTORY ===`). Print it and
read it before deciding the next call. It is your only view of the game.

## The primitive floor (the complete tool set)

Seven calls. Six are single actions; `np_kick` is the exception, noted below.

Coordinates: `x` is the column (0–78), `y` is the row (0–20) in the MAP frame:
row 0 is the FIRST row of the `=== MAP ===` block, the same frame `Pos:` and all
`VISIBLE FEATURES` coordinates use. Do NOT count rows from the raw terminal
screen — that yields an off-by-one that silently misses every target.

```python
await nethack.request_map()                 # Show the FULL map + surroundings in this
                                            # turn's observation. The map is otherwise
                                            # withheld — call this whenever you need to
                                            # see the level. Costs no game time.
await nethack.np_press_key(key=">")         # Press ONE key, exactly as typed at the
                                            # NetHack prompt. Letters, digits, punctuation
                                            # like > < , . # $, plus "ESC", "SPACE", "ENTER".
await nethack.np_pray()                     # Pray. Saves you at critical HP or starving —
                                            # rarely. Do not spam.
await nethack.np_apply(item_letter="a")     # Apply (use) a tool. item_letter optional.
await nethack.np_rest(count=5)              # Rest in place count moves, or until something
                                            # happens.
await nethack.search(times=10)              # Search adjacent tiles for hidden doors /
                                            # passages, times consecutive tries (1–20).

await nethack.np_kick(x=31, y=7)            # Kick the tile at (x, y) — a locked door,
                                            # a chest. Walks adjacent first if needed.
                                            # This is the one call that is not a single
                                            # keystroke: kicking is Ctrl-D, which the
                                            # keystroke tools cannot send, so it stays a
                                            # primitive rather than something you rebuild.
```

Movement and attacking are `np_press_key`: the vi keys `hjkl` (W/S/N/E) and
`yubn` (NW/NE/SW/SE) move one step, and moving into a monster attacks it.
Descending is `>` while standing on a `>` staircase; ascending is `<` on a `<`.

## Your code

```python
import netplay

print(await netplay.move_to(30, 7))   # walk to a coordinate: plan, step, re-plan
print(await netplay.explore())        # auto-explore: frontier to frontier until
                                      # the level is revealed or something happens
print(await netplay.attack(30, 7))    # approach, pursue, and melee that monster
```

That is the whole seed: `move.py` (`move_to`, the BFS route finder, the
walk-interruption logic), `explore.py` (`explore`, `frontiers`) and
`attack.py` (`attack`). Anything else you want does not exist yet and is
yours to build: create new `.py` files beside them
(`os.path.dirname(netplay.__file__)`) and they load automatically on the next
`import netplay`. Export what you want callable via an `__all__` list in the
new file.

### The edit protocol

1. Read the file first. `netplay/_base.py` documents the primitive floor and the
   parsing helpers (`status`, `features`, `monsters`, `grid`, `messages`), which
   are already written and tested — use them rather than re-parsing observations.
2. Edit with the built-in `edit` skill (a targeted, single-occurrence replace):
   `await edit(path="<netplay dir>/move.py", old_str=..., new_str=...)`.
   It is Prime Agent's native file editor -- prefer it over rewriting the whole
   file. (Whole-file writes work too, but `edit` keeps your change surgical.)
3. **Call `netplay.check()` after every edit.** It compiles the whole tree and
   re-imports it, and returns `{"ok": bool, "errors": [...]}`. A file that does
   not compile takes the policy down for every later episode, not just yours.
4. `import importlib; importlib.reload(netplay)` to pick your change up in this
   kernel.
5. Note what you changed and why in `netplay/NOTES.md`. That file is yours, it
   is never overwritten, and it is how the next episode learns what you tried.

### What is frozen

`src/nethack/__init__.py`, `SKILL.md`, `pyproject.toml` and `netplay/_base.py`
are restored from the repository at the start of **every** episode. Editing them
is not forbidden so much as futile — the change is gone before anyone sees it.
Do not spend a turn on it.

`_base` is also the only sanctioned route to the game. A policy that returns an
observation it did not get from `_base` is fabricating: the score comes from the
engine, which only advances on a real call, so an invented map scores **zero**
while looking exactly like progress in your own transcript.

Do not add third-party imports. `netplay` may import the standard library,
`netplay.*`, and nothing else. Editing `pyproject.toml` triggers a ten-minute
rebuild and will be reverted anyway.

## Interactive prompts (auto-dismiss is OFF)

When the game asks something — `[ynq]`, "What do you want to eat?", a `--More--`
page, a menu — the harness will NOT answer it for you. The observation shows the
prompt; answer it yourself with `np_press_key` (the choice letter, `y`, `n`,
`ESC` to cancel, `SPACE` to page). The game clock is FROZEN while a prompt is
open: if the observation looks unchanged, check for an open prompt first.

## Discipline

- One call, read the observation, then decide. Never batch blind sequences.
- This file plus `netplay/_base.py` are the whole API. There is nothing to
  discover by probing.
- A skill that fails says why in its returned message — read it and change plan;
  don't repeat the call unchanged.
- `memory/objective.md` in your workspace restates the goal; `memory/` is yours
  for notes.
