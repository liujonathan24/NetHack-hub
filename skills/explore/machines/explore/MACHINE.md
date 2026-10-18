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

The run ends when the orchestrator answers a decision without `next`
(`explore.stop()`), or when the player state reaches its run limit.

What each role sees. The orchestrator sees the leaderboard (through
`explore.leaderboard(token=...)`) and the previous player's one-line
summary. A player sees only the directive and the resume note: the
`decision` input is hidden, so no leaderboard, no other attempt, no path,
and no server token reaches a player. The ledger reads are gated by the
run's orchestrator token (`{{EXPLORE_TOKEN}}` in this file, filled in per
run; `explore.json` holds it), so a player that calls them by mistake is
refused.

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
        "prompt": "Play the game from the start. Use the game skill: await every game call in your own cell and read its observation before the next call. Do not run game calls in background tasks, and do not message other agents. Play until you die or the harness stops you. You do not need to report anything at the end."
      }
    },
    {
      "id": "orchestrator",
      "max_entries": 29,
      "retries": 2,
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
        "prompt": "You are the orchestrator of a checkpointed exploration run. Your token is {{EXPLORE_TOKEN}}. An attempt just ended. Run `print(explore.leaderboard(token=\"{{EXPLORE_TOKEN}}\"))` and read it. Pick the one checkpoint the next player must start from, and write a directive for that player: strategy only, at most two sentences, at most 300 characters, no key names. Then run `result = explore.select(\"<checkpoint id>\", \"<directive>\", token=\"{{EXPLORE_TOKEN}}\")` and `print(explore.answer(result))`. If the run must end, run `print(explore.answer(explore.stop(\"<reason>\")))` instead. Every `run` above means: execute the code with your ipython tool. Code written in your message text runs nothing and ends the round with no decision. Your final answer is the printed fenced json block, copied exactly as printed, character for character, and nothing else. The previous player's summary, when it wrote one: {summary}"
      }
    },
    {
      "id": "player",
      "max_entries": 29,
      "failure_policy": "continue",
      "inputs": [
        {"name": "decision", "type": "json", "from": "orchestrator.decision", "hidden": true},
        {"name": "directive", "type": "json", "from": "orchestrator.directive", "verbatim": true}
      ],
      "outputs": [
        {"name": "summary", "type": "json"}
      ],
      "subagent": {
        "name": "explore-player",
        "fork_from": {"input": "decision", "path": "next.session_ref"},
        "mcp_servers": {"input": "decision", "path": "next.mcp_servers"},
        "prompt": "{directive}\nYou continue your own game from a checkpoint you saved earlier. The game is restored to that saved state. Your memory of the moves after the save does not apply: request the current observation first and play from what you see. Await every game call in your own cell and read its observation before the next call; do not run game calls in background tasks, and do not message other agents. Play until you die or the harness stops you. When you stop, end your final answer with a fenced json block of the form {\"summary\": {\"outcome\": \"<alive|died|stopped>\", \"note\": \"<one line>\"}}."
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
