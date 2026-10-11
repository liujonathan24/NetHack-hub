---
name: explore
description: Checkpointed exploration. An orchestrator picks a checkpoint and writes a directive; a player resumes that checkpoint's game and conversation; repeat.
version: 1
author: NetHack-hub
---

# explore

## What this machine does

Three states. `opener` plays the first game. `orchestrator` reads the
checkpoint leaderboard, selects one checkpoint, and writes a directive.
`player` starts from that checkpoint: its conversation is a copy of the
earlier player's conversation at the save (`fork_from`), and its game is
its own (`mcp_servers`). After each player, the orchestrator runs again.
The orchestrator keeps its memory across rounds (`fork_from: "self"`).

The run ends when the orchestrator answers `next: null`, or when the
player state reaches its run limit.

## How to run it

    result = await explore.run("explore")

Set `EXPLORE_RUN_DIR` and `EXPLORE_ADAPTER` first. Watch progress with
`await rlm.factory.status(result["run_id"])`.

```machine-spec
{
  "run": {
    "max_parallel": 1,
    "max_transitions": 64,
    "failure_policy": "continue"
  },
  "states": [
    {
      "id": "opener",
      "entry": true,
      "max_entries": 1,
      "failure_policy": "continue",
      "subagent": {
        "name": "explore-opener",
        "prompt": "Play the game from the start. Use the game skill. Play until you die or the harness stops you. You do not need to report anything at the end."
      }
    },
    {
      "id": "orchestrator",
      "max_entries": 29,
      "failure_policy": "continue",
      "inputs": [
        {"name": "summary", "type": "json", "from": "player.summary", "optional": true}
      ],
      "outputs": [
        {"name": "decision", "type": "json"},
        {"name": "directive", "type": "json"}
      ],
      "subagent": {
        "name": "explore-orchestrator",
        "fork_from": "self",
        "prompt": "You are the orchestrator of a checkpointed exploration run. An attempt just ended. Run `print(explore.leaderboard())` and read it. Pick the one checkpoint the next player must start from, and write a directive for that player: strategy only, at most two sentences, at most 300 characters, no key names. Then run `result = explore.select(\"<checkpoint id>\", \"<directive>\")` and `print(explore.answer(result))`. If the run must end, run `print(explore.answer(explore.stop(\"<reason>\")))` instead. Your final answer must end with the printed fenced json block and nothing after it. The previous player's summary, when it wrote one: {summary}"
      }
    },
    {
      "id": "player",
      "max_entries": 29,
      "failure_policy": "continue",
      "inputs": [
        {"name": "decision", "type": "json", "from": "orchestrator.decision"},
        {"name": "directive", "type": "json", "from": "orchestrator.directive"}
      ],
      "outputs": [
        {"name": "summary", "type": "json"}
      ],
      "subagent": {
        "name": "explore-player",
        "fork_from": {"input": "decision", "path": "next.session_ref"},
        "mcp_servers": {"input": "decision", "path": "next.mcp_servers"},
        "prompt": "{directive}\nYou are resumed from a checkpoint. Your game is restored to the state you saved. Request the current observation first; the last moves in your memory may not have happened in the restored game. Play until you die or the harness stops you. When you stop, end your final answer with a fenced json block of the form {\"summary\": {\"outcome\": \"<alive|died|stopped>\", \"note\": \"<one line>\"}}."
      }
    }
  ],
  "transitions": [
    {"from": "opener", "to": "orchestrator"},
    {"from": "player", "to": "orchestrator"},
    {"from": "orchestrator", "to": "player", "when": {"output": "decision", "path": "next", "op": "exists"}}
  ]
}
```
