---
name: explore
description: Checkpointed exploration (Prime-Agent-Explore). The orchestrator reads the checkpoint leaderboard, selects one checkpoint, and writes a directive; the next player then starts from that checkpoint's game and conversation. Use explore.leaderboard(), explore.select(), explore.stop(), and explore.answer() from the Python REPL.
---

# Explore

One run is N attempts at one game. Each attempt is one player. The
environment saves checkpoints while the player plays. Between attempts, the
orchestrator picks the checkpoint the next player starts from.

The factory machine `explore` runs the loop. This skill is the API to the
run's ledger. Call it from the Python REPL.

## If you are the orchestrator

1. Read the leaderboard:

    print(explore.leaderboard())

   Each row is one checkpoint. The numbers come from the environment. The
   `tried` and `ended` columns say how many attempts started there and how
   they ended. The label is text a player wrote; it is not a measurement.

2. Choose one checkpoint. Write a directive for the next player. A
   directive is strategy, not keystrokes: at most two sentences, at most
   300 characters, no key names.

3. Select it, then end your final answer with the fenced block:

    result = explore.select("c7", "Find the downstairs on this level before you fight.")
    print(explore.answer(result))

   Copy the printed block as the last thing in your final answer. The
   machine reads `decision` and `directive` from it.

4. To end the run, select nothing:

    print(explore.answer(explore.stop("the frontier has not moved in 5 attempts")))

Rules. A player never sees another attempt. Only the checkpoint id and the
directive cross from your text into the run. Do not write numbers into
your directive as facts; the ledger is the record.

## If you are the environment or the harness

    explore.checkpoint(attempt=3, score={"dlvl": 5, "xl": 4, "hp": 31}, env_handle=handle, call=142, parent="c6", label="before the mines")
    explore.outcome(attempt=3, result="died", checkpoint_from="c6", reason="killed by a soldier ant")

`result` is one of `died`, `censored`, `budget`, `alive`.

## If you are the root session

    result = await explore.run("explore")

This starts the machine and a watcher. The watcher records each player's
session file into the ledger, so `select` can cut the conversation at the
checkpoint.

## Settings

* `EXPLORE_RUN_DIR`: the run directory (the ledger lives there).
* `EXPLORE_ADAPTER`: `module:Class` of the environment adapter, or
  `scripted` for tests. The adapter restores a game and returns the MCP
  servers the player must use.
