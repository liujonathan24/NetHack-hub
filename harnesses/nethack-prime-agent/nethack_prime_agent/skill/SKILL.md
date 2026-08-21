---
name: nethack
description: Play NetHack. The only way to act in the game — every move, attack, descent and item use goes through this skill's async tools, called from the IPython kernel.
---

# NetHack — complete tool API

This file is the authoritative API reference. Everything you need is here —
do NOT spend calls on `help()`, `list_tools()`, or schema dumps; the JSON
schemas are empty and `help()` adds nothing.

The game runs in a separate process, reached over MCP. One call = one skill
executed against the live game. Every tool is **async** — always `await`.
The return value is the rendered observation (a text block with `=== MAP ===`,
`=== STATUS ===`, `=== INVENTORY ===` sections). **Print it and read it before
deciding the next call.** It is your only view of the game.

## The tools (exact signatures — this is the full set)

Coordinates: `x` is the column (0–78, left to right), `y` is the row (0–20,
top to bottom) in the MAP frame: row 0 is the FIRST row of the `=== MAP ===`
block, the same frame `Pos:` and all `VISIBLE FEATURES` coordinates use. Do
NOT count rows from the raw terminal screen (it has extra message/status
lines) — that yields an off-by-one that silently misses every target.

```python
await nethack.request_map()                 # Show the FULL map + surroundings in this
                                            # turn's observation. The map is otherwise
                                            # withheld — call this whenever you need to
                                            # see the level. Costs no game time.

await nethack.np_explore_level()            # Auto-explore: walks the level revealing
                                            # rooms, corridors and doors. Interrupts
                                            # itself when something notable happens.

await nethack.np_move_to(x=54, y=5)         # Pathfind to tile (x, y) in one call.
                                            # If no route is known it says so —
                                            # explore more first.

await nethack.np_melee_attack(x=30, y=7)    # Pursue the monster at (x, y) and attack
                                            # in melee until it dies (or you must stop).

await nethack.np_kick(x=31, y=7)            # Kick the tile at (x, y) — locked doors etc.

await nethack.np_press_key(key=">")         # Press ONE key, exactly as if typed at the
                                            # NetHack prompt. Letters (both cases),
                                            # digits, and punctuation like > < , . # $
                                            # all work, plus "ESC", "SPACE", "ENTER".
                                            # Use it to answer the game's own prompts
                                            # and menus (pick the item's letter, y/n,
                                            # ESC to cancel).

await nethack.np_pray()                     # Pray to your god. Saves you when HP is
                                            # critical (below ~1/7 of max) or you are
                                            # starving — but only rarely; do not spam.

await nethack.np_apply(item_letter="a")     # Apply (use) a tool from your inventory.
                                            # item_letter optional; omit to be prompted.

await nethack.np_rest(count=5)              # Rest in place count moves (default 5) or
                                            # until something happens.

await nethack.search(times=10)              # Search adjacent tiles for hidden doors /
                                            # passages, times consecutive tries (1–20).
```

## How to descend (the objective is to go DOWN)

1. `await nethack.request_map()` — find the down staircase, shown as `>`.
2. `await nethack.np_move_to(x=..., y=...)` — walk onto that tile.
3. `await nethack.np_press_key(key=">")` — descend. That's it.

There is no `descend` tool — descending IS pressing `>` while standing on the
staircase. Ascending is `<` on a `<` staircase.

## Interactive prompts (auto-dismiss is OFF)

When the game asks something — `[ynq]`, "What do you want to eat?", a `--More--`
page, a menu — the harness will NOT answer it for you. The observation will show
the prompt; answer it yourself with `np_press_key` (the choice letter, `y`, `n`,
`ESC` to cancel, `SPACE` to page). The game clock is FROZEN while a prompt is
open: if the observation looks unchanged, check for an open prompt first.

## Discipline

- One call, read the observation, then decide. Never batch blind sequences.
- There is a hard budget of skill calls for the episode; when it is spent the
  episode ends. Spend calls on progress, not probing — this file already
  contains the whole API.
- A skill that fails says why in its returned message (e.g. "Tile (14, 12) is
  blocked... It's solid stone.") — read it and change plan; don't repeat the
  call unchanged.
- `memory/objective.md` in your workspace restates the goal; `memory/` is yours
  for notes.
