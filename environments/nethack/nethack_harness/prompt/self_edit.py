"""E13 self-edit block: ask the player to persist what it learned.

`players_may_edit` gives each rollout a writable copy of the continual-harness
store, but provisioning is not instruction: nothing in the baseline prompt asks
the model to record anything. Measured across E9's 15 control rollouts, exactly
one player called `rlm.harness.create_memory(global_=True)` unprompted -- so
relying on spontaneity would make the arm's independent variable fire ~7% of the
time.

Prompt-only, appended to the system prompt. Default off, so every other arm --
including [base] and its frozen document -- stays byte-identical.

The budget in the text is not our policy, it is the scaffold's:
`formatHarnessStateForPrompt` renders at most 6 entries per kind at 180
characters each, so anything past that is invisible to the next player. That is
what makes deletion part of the job rather than an afterthought.
"""
from __future__ import annotations

SELF_EDIT_BLOCK = (
    "\n=== PERSISTING WHAT YOU LEARN ===\n"
    "You have a continual-harness store that carries into FUTURE games. What you\n"
    "write here outlives this episode and is rendered into the next player's\n"
    "prompt. (In a code arm, your netplay edits persist separately, on their own\n"
    "channel.) From the kernel (these are synchronous -- do NOT await them):\n"
    "  rlm.get_harness_state(global_=True)                     # read what is known\n"
    "  rlm.harness.create_memory(title=..., content=..., global_=True)\n"
    "  rlm.harness.update_memory(id=..., title=..., content=..., global_=True)\n"
    "  rlm.harness.delete_memory(id=..., global_=True)\n"
    "  rlm.harness.create_skill(title=..., content=...,\n"
    "      reference={'type':'python','import':'nethack',\n"
    "                 'call_pattern':'await nethack.np_press_key(key=\">\")'},\n"
    "      arguments={}, global_=True)\n"
    "\n"
    "Write only what would have helped you at the START of this game: a mechanic\n"
    "you got wrong, a tool that failed in a way its description did not predict, a\n"
    "sequence that worked. Not narration of this run, and not anything true only\n"
    "of this dungeon -- the next game is a different map.\n"
    "\n"
    "HARD BUDGET, because of how the store is rendered into the next player's\n"
    "prompt: at most 6 entries PER KIND are shown, each truncated to 180\n"
    "characters. Past that is invisible and wasted. So make every entry useful at\n"
    "180 characters -- concrete and imperative -- and once you are at the cap,\n"
    "adding means DELETING. Delete the entry this game contradicted, including one\n"
    "you or an earlier game wrote: an entry that made things worse is the most\n"
    "valuable thing you can remove.\n"
)
