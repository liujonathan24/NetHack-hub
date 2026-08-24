# Removing the call budget from what the model sees

**Status:** baseline v2. Not measured yet.

The agent is supposed to be playing NetHack — trying to win — not playing a
budgeted evaluation. Four model-facing strings told it otherwise. This change
removes them, which makes a **new baseline**, not a correction.

## What the model was told, and what it now says

No prompt ever stated the number 200. What leaked was the *existence* of a
budget, and in one place a strategy derived from it.

| # | Where | Before | After |
|---|---|---|---|
| 1 | `SKILL.md` + `SKILL.baseline.md` | "There is a hard budget of skill calls for the episode; when it is spent the episode ends. **Spend calls on progress, not probing** — this file already contains the whole API." | "This file already contains the whole API, so there is nothing to discover by probing it." |
| 2 | `SKILL.md` + `SKILL.baseline.md` | "do NOT **spend calls** on `help()`, `list_tools()`, or schema dumps; the JSON schemas are empty and `help()` adds nothing." | "`help()`, `list_tools()` and schema dumps add nothing — the JSON schemas are empty and everything is documented here." |
| 3 | `rendering.py`, **both** `_PROMPT_TAIL` and `_PROMPT_TAIL_MINIMAL` | "the episode ends only when your character dies **or you run out of calls**, never because you stopped…" | "the episode ends only when your character dies, never because you stopped…" |
| 4 | `nethack_v1.py` end-of-episode message | `[Call budget exhausted: 200 skill calls used. The episode is over.]` | `[The episode is over.]` |

(1) is the one that shaped play: it is not disclosure, it is instruction to
economise. (3) reached **every** skill-interface arm since `69d7bce`
(2026-07-28); (1) and (2) reached every `prime_agent` cell since `be58af1`
(2026-07-26), including E10's baseline.

## Why it mattered in practice

A reflection pass over a single game wrote this into the continual-harness store
as a *learned lesson*:

> `prioritize_descent_over_exploration`: "Find stairs FAST… **Budget is limited
> — spend on descent, not looting.**"

That entry would then have been served to every later player as accumulated
wisdom. The other four entries from that run (vault-guard mechanics, prompt
loops) are grounded in what actually happened in the game; this one is grounded
in an artifact of how we measure.

## The risk in change 3, stated plainly

The clause "or you run out of calls" was not decoration. It was added in
`69d7bce` because, without an explicit statement that ending a turn is not how
the episode ends, **15–25% of rollouts finished early** with the model
declaring itself done under `--print`. The anti-give-up half of the sentence is
kept; only the budget half is removed. If early exits return, that is the cause,
and it will look like a performance drop rather than a prompt regression.

If we want to separate the two effects, changes 1/2/4 can land first and 3
second, each with its own cell.

## This is a new baseline, not a fix

`SKILL.baseline.md` is the frozen document `[base]` serves, pinned by hash so it
cannot drift silently (PR #33). Editing it necessarily breaks byte-identity with
what E10 measured.

- **v1**: sha256 `8585082860c7…`, `aee5c43^` verbatim — the document every
  rollout in `outputs/e10_baseline/` was served. E10's reference numbers
  (median 3.54 / mean 5.34 / 14 of 15 died / ceiling dl11) describe **v1** and
  remain valid for it.
- **v2**: this change. Its numbers do not exist yet.

Nothing may compare a v2 cell against E10's table. A v2 baseline has to be
measured on its own before any cross-experiment claim rests on it.

After editing the document, regenerate the pin:

```
sha256sum harnesses/nethack-prime-agent/nethack_prime_agent/skill/SKILL.baseline.md
# paste into _BASELINE_SKILL_SHA256 in nethack_prime_agent/__init__.py
# and into test_the_baseline_skill_doc_is_byte_identical_to_what_e10_served
```

Two tests guard the property going forward: one asserts no served prompt or
skill doc matches `budget|run out of calls|call limit|spend calls`, the other
that the end-of-episode message does not state the budget's size. Both assert on
served text, not source comments — explaining why the budget is hidden is fine.
