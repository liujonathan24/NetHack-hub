# Baseline v2: play to win, not to a budget

**Status:** baseline v2. Not measured yet.

The agent is supposed to be playing NetHack — trying to win — not playing a
budgeted evaluation. Four model-facing strings told it otherwise. This change
removes them, which makes a **new baseline**, not a correction.

## Change A: what the model was told about the budget, and what it now says

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

## Change B: the objective was stated as depth

Removing the budget language is only half of it. The prompt also framed the goal
as **depth**, which is a proxy we invented — the game's objective is to win, and
winning requires staying alive.

| Where | Before | After |
|---|---|---|
| `_PROMPT_TAIL_MINIMAL` | "Explore the environment to find the stairs down to the next level. **Your goal is to get as far as possible in the game.**" | "Your goal is to **win: survive, and descend toward the Amulet of Yendor. Dying ends the run.**" |
| task description | "**Descend as deep into the dungeon as you can** and survive; ultimately win the game…" | "**Win the game**: retrieve the Amulet of Yendor and escape. Descending is how you progress; surviving is what lets you keep progressing." |
| `SKILL.md` heading | "## How to descend **(the objective is to go DOWN)**" | "## How to descend" |

### What the old framing produced

| metric | E10 baseline (v1 prompt) |
|---|---|
| died | **14 of 15** |
| mean XL | 2.80 |
| XL/Dlvl | **0.47** |
| Dlvl − XL gap | +3.53 |
| XL vs human norm at final depth | −1.73 |

The agent descended about twice as fast as it levelled, arriving underpowered
and dying. Across all 25 E10+E11 games, the only two that were deep *and* alive
at the end were the only two XL-6 games.

**E11's descent gates may have been treating a symptom this prompt caused.**
E11b forced levelling and moved XL/Dlvl 0.47 → 0.74 by *blocking* descent. If v2
produces a similar shift with no gate, part of E11 was compensating for our own
wording. If it does not, the gate is doing independent work. Either answer is
worth having.

### Deliberately not added

No tactical advice. Not "level up before descending", not "fight what you can
handle" — that would be E11's gate rewritten as prose, and any leveling change
would then be ours rather than the model's. The prompt states the objective and
stops.

## How to judge v2 — not on depth

`max_dlvl` and BALROG% both reward depth, so an agent that survives longer at
shallower depth scores **worse** on the number we have been reporting while doing
the thing we want. Report these first:

- **death rate** and **game turns survived** — "more alive", made explicit
- **XL/Dlvl** and **Dlvl − XL gap** — is it arriving prepared
- **stop_condition split**: `game_over` vs `call_budget_exhausted` — the shift
  toward the latter *is* the behaviour change
- **max_dlvl / BALROG%** — kept, secondary, expected to soften

`aggregate.py` already emits `max_xp`, `died` and `game_turns` per rollout, so
the ratios need no new instrumentation.

**Prediction, recorded before the run so it can be wrong:** lower mean depth,
higher survival, higher XL/Dlvl, more cells ending at the cap. If depth falls
*and* survival does not rise, the change did not do what we think, and the first
thing to check is whether early self-termination returned.

## Two changes in one branch

This branch contains two separable interventions: **A** removes the budget
language, **B** reframes the objective from depth to winning. A single v2 cell
cannot attribute its result to one or the other. That is a deliberate trade for
time: if the combined effect is what we want, the attribution question may never
need answering; if it is ambiguous, A and B can be split and run separately.

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
