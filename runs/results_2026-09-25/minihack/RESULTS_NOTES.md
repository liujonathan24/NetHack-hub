# MiniHack panel — results notes

## 1. Why the paired design is necessary: the same seed solves or fails on the draw

**Two unmodified base episodes, same task, same seed 0, same model, same
protocol, different outcome:**

| source | run | steps | progression |
|---|---|--:|--:|
| seed-0 calibration (`/root/nld/gen_runs/minihack/MiniHack-CorridorBattle-Dark-v0_s0_base`) | base-only | 47 | **1.0 (solved)** |
| panel PAE run, attempt 1 (`MiniHack-CorridorBattle-Dark-v0_s0_pae`) | attempt 1 = unmodified base episode | 24 | **0.0 (failed)** |

Nothing differs but the sampling draw: BALROG runs GLM-5.2 at
`temperature: 1.0`, and MiniHack's `progression` is binary, so a single episode
is one Bernoulli trial whose outcome flips with the draw.

**Consequence for the method.** Any design that compares a PAE run against a
*stored* base run measures the difference between two draws, not the effect of
the method. Had we scored this cell against the stored calibration base, the
same run that demonstrably rescued a failure would have recorded delta
`1.0 - 1.0 = 0.0`. The within-run paired design -- attempt 1 of each run IS the
base sample, drawn under the same conditions in the same process -- is what
makes the delta attributable to the resume rather than to the draw.

This is the single clearest argument in the panel for the paired protocol, and
it is measured, not hypothetical.

## 2. The cleanest PAE rescue on MiniHack (CorridorBattle-Dark, seed 0)

| attempt | from checkpoint | steps | progression |
|--:|---|--:|--:|
| 1 | -- (fresh episode, the paired base sample) | 24 | 0.0 |
| 2 | c4 | 27 | **1.0 (solved)** |

- paired delta **+1.0**, `stop_reason: solved` after 2 of 10 attempts
- restore byte-identical: replayed digest `9466f4a29c91c068` == saved digest,
  15 steps replayed, `text_identical: true`
- prompt parity exact: 32 -> 33 messages, the one extra being the directive
- `client_retry_exhausted` 0; 1 invalid action; $0.1811 billed; 2 min

Classified **non-zero**, not zero-by-ceiling: the base sample genuinely failed
and the resume genuinely rescued it.

## 3. Delta classification used throughout

- **non-zero** -- `best > attempt1`; the resume changed the outcome.
- **zero-by-ceiling** -- `attempt1 == 1.0`; attempt 1 already solved, so the
  loop broke immediately and PAE had no room to improve. NOT a null result.
- **zero-by-floor** -- `attempt1 == 0.0` and no attempt of N ever scored.

Pooling these three would misreport both the ceiling and the floor cases, so
every per-task table reports the counts separately.

## 4. Two failures that left no trace in the artifact you would normally check

Both cost real time on this panel, and both are the same class: the job stops
or misbehaves, and the file you would naturally inspect looks fine.

### 4a. A missing `cd` silently truncated the batch after the first task

`games/minihack/launch.sh` ended by invoking

```
"$PY" -m tools.balrog_pae.games.minihack.aggregate_minihack "$OUT_ROOT" ...
```

without `cd "$REPO"`, so Python could not resolve the `tools` package and
exited `ModuleNotFoundError`. Under the script's own `set -euo pipefail` that
status propagated and killed the batch -- **after the first task finished**.

Observed: CorridorBattle-Dark completed all five seeds, then the batch died at
the aggregate step. Boxoban-Medium, Boxoban-Hard, Quest-Easy s2-s4 and
Quest-Medium never launched. **Every per-run `run.log` was clean and every
`summary.json` was valid**; the only trace was a `ModuleNotFoundError` in the
batch-level log, which is not the file you check when asking "how are the runs
going?". The failure would have been found only by noticing absent directories.

This was independent of the serial-vs-parallel decision: the serial queue would
have stopped after CorridorBattle-Dark either way. Fixed in commit 5d80c38 by
wrapping the aggregate in `( cd "$REPO" && ... )`, matching `run_one`.

### 4b. Killing a launcher does not kill the run it spawned

Killing the two launcher shells during the queue reorder did **not** kill their
python child. A Quest-Easy seed 2 run survived as an orphan and kept playing --
competing for the endpoint and spending money on a cell that was supposed to be
deferred, while the process tree suggested the batch was stopped. Parent death
does not propagate here.

Two consequences, both now standard practice in this directory:

* After stopping a batch, **re-check `pgrep -af 'balrog_pae.run'`** for orphans
  and kill them by explicit PID.
* Remove any partial run dir (no `summary.json`) before relaunching, or the
  launcher re-enters it with stale `archive/` and `attempts/` state. Its skip
  rule keys on `summary.json`, which a killed run never writes.

### 4c. Related operational note: never `pkill -f` a pattern from a shell whose own command line contains it

`pkill -f gate.sh` matched the very shell running it and killed the caller
(exit 144) before the rest of the command ran -- three times on this panel,
once taking a monitor with it and once silently skipping a file write, so a
launch script that appeared to have been created did not exist. Use explicit
PIDs obtained from a prior `ps`/`pgrep`, and verify with `kill -0` afterwards.

### 4d. A monitor reported "PANEL COMPLETE" while five runs were still playing

The completion watcher was armed when the panel was a 15-run, three-task
design, and its exit test was a hard-coded `>= 15` summary count. The panel
later grew to eight tasks. At 15 summaries the watcher printed
**"PANEL COMPLETE: 15/15"** and stopped, while Boxoban-Medium s0/s1,
Corridor-R3 s1, MazeWalk-15x15 s0 and Quest-Medium s0 were all still running.

Same class as 4a and 4b: the artifact you check reported success, and the
underlying state was different. Here it was worse than silence, because a
positive completion signal invites you to stop looking.

Rule: a watcher's exit condition must be derived from the work actually queued
(no launchers and no runs alive), never from a count fixed when it was armed.
The replacement keys on `pgrep` for live launchers/runs plus the gate condition.

## 5. Boxoban retry-exhaustion rate: measured twice, different answers

`client_retry_exhausted` marks a step where five CONSECUTIVE completions came
back `finish_reason="length"` with no content, so BALROG's client gave up and
returned an empty completion, which the evaluator scores as an invalid action
and replaces with `default_action` (`north`). Such a step is the harness moving
the agent, not the agent playing.

**We measured this twice on Boxoban-Medium and got materially different rates.
Both are reported; the larger sample is the one to quote.**

| measurement | exhausted/steps | rate | 95% CI (Wilson) |
|---|---|--:|---|
| diagnostic, partial episode, seed 0 paused at 23 steps | 1/23 | 4.3% | [0.8%, 21.0%] |
| **gate seeds s0+s1, attempt 1** | **12/40** | **30.0%** | **[18.1%, 45.4%]** |

Forty steps is still a small sample and the claim carries that uncertainty: the
two intervals **overlap** in the 18-21% region, so the difference between the
runs is not established, only the point estimates differ. What is established
is that the rate is materially above zero and plausibly near a third.

### The earlier retraction was over-confident

On the 23-step sample this agent reported 82% legal / 14% exact-match-defaulted
/ 5% exhaustion-defaulted, and on that basis **retracted** the concern that
Boxoban post-fix measures degraded play, concluding it was "an agent playing,
not a random walker" and that exclusion rested on cost alone. That retraction
was drawn from a single partial episode and was too confident for its evidence.

At 30% exhaustion the picture is different in kind, not degree: roughly a third
of moves are `north` chosen by the harness. That is a **validity** problem
independent of cost, and it is the concern the retraction dismissed.

Note also the composition changed. The 23-step sample had 14%
exact-match-defaulted (the model's own output failing BALROG's exact-match
rule) and 5% exhaustion. The 40-step sample has **0%** exact-match and **30%**
exhaustion. When the model does return content it is now well-formed; the
losses have moved entirely into completions that never arrive.

### What would fix it, and why it was not applied

`reasoning_effort="none"` returned well-formed content in 6/6 probe calls
across both Boxoban tasks (see the probe record), at 2-192 output tokens rather
than 8192. It was NOT adopted: it changes sampling, so Boxoban would no longer
match the other seven tasks or the 25 completed Crafter/TextWorld runs. Reading
the discarded `reasoning` field is not an alternative -- BALROG's validity rule
is exact string membership, and the real parser extracted a legal action from
truncated reasoning in **0/5** samples.

## 6. Boxoban is excluded from the MiniHack panel — on COMPARABILITY, not cost

**Decision: MiniHack-Boxoban-Medium-v0 and MiniHack-Boxoban-Hard-v0 are excluded
from the eight-task panel. Medium was stopped partway through attempt 1 on seeds
0 and 1; Hard was never launched.**

### The decisive measurement

`client_retry_exhausted` marks a step where five CONSECUTIVE completions came
back `finish_reason="length"` with no content. BALROG's client then returns an
empty completion, the evaluator scores it an invalid action and substitutes
`default_action` (`north`). **That step is the harness moving the agent.**

Boxoban-Medium, seeds 0 and 1, attempt 1, stopped in progress:

| seed | steps | legal | exact-match-defaulted | **exhaustion-defaulted** |
|---|--:|--:|--:|--:|
| 0 | 42 | 26 (62%) | 0 (0%) | **16 (38%)** |
| 1 | 35 | 24 (69%) | 0 (0%) | **11 (31%)** |
| **pooled** | **77** | **50 (65%)** | **0 (0%)** | **27 (35.1%)** |

Pooled 95% Wilson interval on the exhaustion rate: **[25%, 46%]**.

**Against every other task in the panel: 23 completed runs across 7 tasks,
`client_retry_exhausted` total = 0.** Not low — zero.

### The rate rose as the sample grew; it did not regress

Measured three times, on growing samples:

| sample | exhausted/steps | rate | 95% CI |
|---|--:|--:|---|
| diagnostic, seed 0, partial episode | 1/23 | 4.3% | [0.8%, 21.0%] |
| gate seeds, early | 12/40 | 30.0% | [18.1%, 45.4%] |
| **gate seeds, at stop** | **27/77** | **35.1%** | **[25%, 46%]** |

The first interval and the last **do not overlap**. The 4.3% figure was not an
unlucky draw from the same distribution; it was measuring a different regime.

### The composition shifted, which is why the samples disagree

The 23-step sample was 82% legal / 14% exact-match-defaulted / 5% exhaustion.
The 77-step sample is 65% legal / **0%** exact-match / **35%** exhaustion. When
the model returns content at all it is now well-formed; the losses moved
entirely into completions that never arrive. Early-episode behaviour does not
predict full-episode behaviour here — plausibly because the observation window
fills as the episode proceeds and the model spends more of the budget reasoning.

### Two errors recorded, because the reader should see the sequence

1. **This agent** reported the 4.3% figure from one partial episode and
   **retracted** the degradation concern on it, concluding Boxoban was "an agent
   playing, not a random walker" and that exclusion rested on cost alone. That
   retraction was far too confident for a 23-step sample.
2. **The coordinator** relayed that retraction upstream and repeated the
   "cost alone" conclusion.

Both were wrong the same way. The rate was measured three times on growing
samples and the final figure should not be presented as though it were obvious.

### Pace and what was bought

At the stop: **335 s/step (s0) and 378 s/step (s1)** — against 2.9-3.5 s/step on
the navigation tasks — because an exhausted step burns five full 8192-token
completions before defaulting. Neither seed closed a single attempt of ten in
~3.8 h. Completing attempt 1 alone needed a further 5.5-6.8 h per seed; all five
Medium seeds plus Hard would have been on the order of a week of wall-clock.

Provider-reported cost of the two partial attempts: **not directly measurable**.
`billed_by_provider_usd` is written into `summary.json`, and a run stopped mid
-attempt never writes one. Reconstructing from `trace.jsonl` gives only a LOWER
BOUND, because the ledger records the tokens of *successful* calls only:

  * kept calls over 77 steps: 242,299 in / 162,600 out -> ~$0.43 billed
  * >= 135 wasted 8192-token completions (27 exhausted steps x 5 retries)
    -> >= $2.31 billed
  * **lower bound ~$2.73 billed**, excluding truncated calls on steps that
    eventually succeeded, which are invisible to the ledger but were billed.

An earlier draft of this note stated "$9.09 billed" for these two attempts. That
figure was unsupported -- it was derived by differencing panel totals that did
not contain the partial runs at all -- and is withdrawn. The retry waste being
invisible to the token ledger, and visible only in `billed_by_provider_usd`, is
precisely why that field is the figure of record; here the stop deprived us of
it.

### Why this is comparability, not cost

A Boxoban progression number would describe an agent roughly **one third of
whose moves were chosen by the harness**, set beside seven tasks whose
exhaustion count is exactly zero. That is not the same measurement, and no
budget fixes it. Had the author funded the remaining ~$180, the resulting cell
still could not be reported alongside the others.

### Unrecoverable without breaking comparability another way

`reasoning_effort="none"` returned well-formed content in **6/6** probe calls
across both Boxoban tasks, at 2-192 output tokens instead of 8192. It was not
adopted: it changes sampling, so Boxoban would differ from the other seven tasks
AND from the 25 completed Crafter/TextWorld runs. Reading the discarded
`reasoning` field is not an alternative — BALROG's validity rule is exact string
membership and its real parser extracted a legal action from truncated reasoning
in **0/5** samples. `thinking_budget` is inert on this path (read only by the
Google and Bedrock wrappers, never by `OpenAIWrapper`).
