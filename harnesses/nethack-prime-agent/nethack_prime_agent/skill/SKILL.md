---
name: nethack
description: Play NetHack. The only way to act in the game — every move, attack, descent and item use goes through this skill's async tools, called from the IPython kernel. Tools are discovered from the running game server at session start.
---

# NetHack

The game runs in a separate process and is reached over MCP from the kernel.
There is no other channel: the shell cannot see the game, and there is no
keyboard interface. One call = one skill executed against the live game.

## Usage

```python
import nethack

# 1. Discover what this game exposes. The tool set is defined by the server and
#    is deliberately restricted -- do not assume a tool exists.
for tool in await nethack.list_tools():
    print(tool["name"], "-", tool["description"])

# 2. Inspect a tool's arguments (rendered from its JSON Schema).
help(nethack.move_to)

# 3. Call it. Keyword args must match the tool's input schema.
print(await nethack.explore_and_descend())
```

Notes:

- Every tool is `async` — always `await`.
- The return value is the rendered observation: a text block with `=== MAP ===`,
  `=== STATUS ===`, `=== INVENTORY ===`, `=== ADJACENT ===` sections. **Print it
  and read it before deciding the next call.** It is the only view of the game.
- Do not batch blind sequences of calls. NetHack is turn-based and adversarial;
  the observation after each call is what tells you whether the previous one
  worked.
- There is a hard budget of skill calls for the episode. When it is spent the
  server refuses further calls and the episode ends — spend calls on progress,
  not on probing.
- **Never use `press_key` or `type_text` to move.** `move_to(x, y)` pathfinds
  for you in one call; a movement key covers one tile and usually walks into a
  wall. If `move_to` fails, its message includes a `[why: ...]` clause naming
  the blocker and the remedy (open the named door, `reveal` the unexplored gap,
  or `search` near dead ends) — follow it instead of retrying by hand.
- If the observation contains an `=== OBJECTIVE ===` block, treat its FOCUS
  line as your current priority: it names which scoring axis is worth pursuing
  right now and what the next checkpoint is worth.
- A call that returns an error raises `McpToolError`; a missing tool name raises
  `AttributeError` naming the tools that do exist. Neither ends the episode.
- `NotEnabled` means the harness did not wire the server up. It is a setup bug,
  not something to work around, and not something the user can fix mid-episode.
