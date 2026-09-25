# Per-run cost and wall-clock on MiniHack are not stable quantities

**For the appendix cost columns. Do not quote a MiniHack per-run price from a
single seed.**

## The measurement

Same task, same protocol (Quest-Easy, PAE arm, N=10, K=3, 100-step cap),
adjacent seeds:

| seed | total player steps | attempts | steps/attempt | wall     | billed      |
|-----:|-------------------:|---------:|--------------:|----------|-------------|
| 0    |  **50**            | 10       |  5.0          |  7.1 min | **$0.1211** |
| 1    | **595**            | 10       | 59.5          | 73 min   | **$1.8496** |

A **12x spread in work, a 15x spread in cost and a 10x spread in wall-clock on
the same task under identical protocol** (both cells: N=10, K=3, plateau 99,
100-step cap, PAE arm). Both scored delta +0.0, so the spread is pure variance
in how long the agent survived, not a difference in outcome. MiniHack episode
length is dominated by when the agent dies, which varies seed to seed. Seed 0's
episodes died at 1-12 steps; seed 1's ran near the 100-step cap.

## Why it matters twice over

1. **Cost.** Per-run billed cost scales with steps, so it inherits the same 8x
   spread. The $0.1211 measured on seed 0 is not "the price of a Quest-Easy
   run"; it is the price of an unusually short one -- seed 1 cost 15x more for
   the same protocol and the same (zero) result. Quote the 5-seed total, or
   quote a range with the spread shown. Never extrapolate 5 seeds from 1.

2. **It produces a rate illusion.** Minutes-per-attempt differed ~10.6x between
   these seeds and was briefly misread (by this agent) as endpoint contention
   from concurrent jobs. Normalising per STEP instead:

   - seed 0: 50 steps / 7.1 min  = **8.5 s/step** (endpoint to itself)
   - seed 1: 409 steps / 60 min  = **8.8 s/step** (two other consumers active)

   = **1.03x**. Three concurrent consumers on this endpoint cost ~3%, not 14x.
   The lesson: on MiniHack, normalise per step. Any rate whose denominator is
   "attempts" or "runs" is not comparable across seeds.

## Planning figures actually supported by measurement

- three-task panel (Quest-Easy, Quest-Medium, CorridorBattle-Dark; 5 seeds,
  N=10): **~$20.7 billed**, from the owner's measured per-step token model at
  billed/list 0.369 (independently validated at 0.393 on the completed panel).
- throughput: **~8.6 s/step**; runs are 400-900 player steps, so
  **20-30 hours** for 15 runs, with wide variance for the reason above.

## The costing rule: failure is expensive, success is nearly free

**The least informative cells are the most expensive ones.** This is a rule, not
an anecdote, and it inverts the usual intuition that spend tracks effort or
engagement.

Mechanism: `pae.Run.go` breaks on `progression >= 1.0`. A task the agent solves
therefore bills 1-3 attempts; a task it never solves bills all N attempts, each
playing to the 100-step cap (on tasks with no death condition).

Measured across the MiniHack panel:

| task | outcome | seeds | steps/run | billed/run |
|---|---|--:|--:|--:|
| MazeWalk-9x9 | solves on attempt 1, every seed | 5 | 2-44 | **$0.034 avg** |
| CorridorBattle-Dark | solves on attempt 2-3, 4 of 5 seeds | 5 | 68-366 | $0.404 avg |
| Corridor-R3 | never solves | 1 so far | **1010** | **$4.207** |

That is a **124x** spread between the fully-solved task and the never-solved
one, in the direction opposite to informativeness: the cell that tells us least
about PAE costs the most.

### The projection error this caused, recorded so it is not repeated

The three added tasks were estimated at ~$15 combined by extrapolating from
CorridorBattle-Dark's measured per-run cost. That was wrong by 2-3x: every task
measured at the time happened to be one that SOLVES, so the sample only ever
contained cheap runs. Corridor-R3 alone is tracking to ~$21 (5 x $4.21).

**Rule for estimating a MiniHack task's cost: first ask whether the agent
solves it.** If it does, cost is a small multiple of one episode. If it does
not, cost is N attempts x the step cap, which on these tasks is 10-100x more.
Never extrapolate from a solving task to a non-solving one.
