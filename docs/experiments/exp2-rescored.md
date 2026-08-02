# Exp 2, re-scored — what the repaired accounting says

Same 80 rollout records, same engine, same traces. Only the *reading* changed.
Everything below comes out of `tools/cli_harness_eval/aggregate.py` now, rather
than out of a hand reconstruction, so the next sweep gets the same treatment for
free.

Reproduce with:

```bash
PI_API_KEY=… PYTHONPATH=.:environments/nethack \
  .venv-cli-eval/bin/python tools/stall_watchdog.py --backfill outputs/e2_encoding_sweep
PI_API_KEY=… PYTHONPATH=.:environments/nethack \
  .venv-cli-eval/bin/python -m tools.cli_harness_eval.aggregate outputs/e2_encoding_sweep
```

## 1. The bill

| | before | after |
|---|--:|--:|
| traced spend | $408.54 | **$1,083.22** |
| wallet | $1,230.50 | $1,230.50 |
| unexplained | $821.96 (67%) | **$147.28 (12%)** |

Three separate defects, in descending order of damage:

1. **`cached_input_tokens` was subtracted instead of added.**
   `verifiers.v1.types.Usage` states that `prompt_tokens` *excludes* cache reads
   and `input_tokens` adds them back; the traces show it plainly — the first
   call of every claude_code rollout carries `prompt_tokens=28,
   cached_input_tokens=384`, a "subset" 13× larger than its set. The old
   `min(cached, prompt)` clamp priced those 384 tokens at zero. Across the
   sweep: **334,155,338 cache-read tokens, 52.4% of all input.**
2. **The price table was a third-party guess.** $1.40/$4.40 from aggregator
   pages citing "official Z.ai rates"; Prime's own `/models` endpoint says
   **$1.68 / $5.28**, and publishes *no* cached-read discount, so a cache hit
   bills at the full input rate. `refresh_price_tables()` now reads the
   endpoint on every aggregation and caches to `results/model_prices.json`.
3. **Killed attempts left no record at all.** See §3.

The remaining $147.28 is the floor on what the 40 watchdog-killed attempts and
the untraced retries burned; their per-call usage died with their processes and
is not recoverable.

Note: the earlier hand reconstruction billed `completion + reasoning` as
3,665,888 tokens. `reasoning_tokens` is a *subset* of `completion_tokens`, so
the real figure is 2,196,666 and that line was ~$7.76 over.

## 2. The scores

`n_err` rollouts — the 402 stubs and anything else the infrastructure ended —
are now excluded from every score and reported on the row. **31 of 80 records.**
Where a 402 landed mid-game, the turn file exists and holds a real but
*censored* depth; those seeds are dropped too, because a rollout the provider
cut off is not a rollout the agent could not push past. 49 rollouts survive as
measurements, which is what the hand count also found.

Depth is now read from `trace.metrics.max_dlvl_reached` when a rollout's turn
file was quarantined or never written. Both scorers previously reached it
through `max_dlvl_reached or 1` and scored a flat dlvl 1 for the 27% of
rollouts with no turn file — against true depths up to 13.

Vision-on cells, corrected:

| cell | n | n_err | depth ± SE | BALROG % | died | $/rollout |
|---|:-:|:-:|:-:|:-:|:-:|--:|
| claude_code B0 | 2 | 3 | 11.50 ± 2.50 | 19.51 | 100% | $23.32 |
| claude_code BBOX_JSON | 4 | 1 | 8.50 ± 1.66 | 10.56 | 75% | $32.71 |
| claude_code BBOX | 5 | 0 | 7.80 ± 2.35 | 12.29 | 60% | $15.05 |
| claude_code SPARSE_ONDEMAND | 2 | 3 | 4.00 ± 1.00 | 2.20 | 50% | $40.66 |
| prime_agent B0 | 4 | 1 | 4.00 ± 1.29 | 2.34 | 25% | $31.93 |
| prime_agent BBOX_JSON | 3 | 2 | 4.67 ± 0.88 | 2.65 | 67% | $13.54 |
| prime_agent BBOX | 4 | 1 | 6.00 ± 2.68 | 7.98 | 50% | $23.25 |
| prime_agent SPARSE_ONDEMAND | 4 | 1 | 4.00 ± 1.29 | 2.31 | 50% | $10.14 |

The conclusions do not move: vision-on is still worth ~2 levels, SPARSE_ONDEMAND
is still the worst encoding, and claude_code is still ahead. What moves is the
*confidence* — three cells are n=2..3 with 1–3 records thrown away, which is
what makes this sweep unable to answer the scaffold question.

## 3. What the watchdog was throwing away

The eval CLI writes `traces.jsonl` when a rollout *finishes*. A SIGKILLed one
never does, so 40 killed attempts — 6,195 turn records, 32,145 in-game turns —
existed only as quarantined NDJSON. The watchdog now reconstructs a trace-shaped
record per killed attempt into `traces.partial.jsonl` (`stop_condition:
watchdog_stall`), and `--backfill` does the same retroactively for sweeps killed
before it learned to.

In exp 2 all 40 were superseded by successful retries, so they add nothing to
`n` — the aggregator de-dups them against the real record. Their value here is
the count itself: **prime_agent was killed 30 times to claude_code's 10**, and
that ratio is the single largest sample-destroying asymmetry in the sweep.

## 4. Two items from the fix list that were already done

* **#3 (drop the per-turn JOURNAL)** and **#4 (remove the HINT block)** need no
  code change for the CLI arms. Both are gated on `not self_dispatch`
  (`rendering.py:1326`, `:1730`, `:1737`) and all four CLI arm configs set
  `self_dispatch = true`. Measured across every rendered observation in the
  sweep — 7,540 claude_code and 6,212 prime_agent — **zero** contain either
  block. The 386 B/turn and 239 B/turn figures describe the *control* arm.

## 5. One item from the fix list that does not work as specified

* **#2 (charge budget per decision via `max_parallel_skill_calls = 1`).** The
  referee has no view of assistant-turn boundaries; it detects a batch as calls
  arriving within `parallel_batch_window_s` (0.5s) of each other. Measured on
  this sweep's own turn timestamps:

  | | median gap | inside 0.5s |
  |---|--:|--:|
  | consecutive `np_press_key('s')` inside a prime_agent kernel loop | 2.89s | 1.0% |
  | all prime_agent calls | 19.3s | 1.9% |
  | all claude_code calls | 22.2s | 0.0% |

  Each loop iteration is a full MCP round-trip plus an engine step, so the
  "kernel loop" it was meant to close is nowhere near the window. Setting it to
  1 would refuse ~2% of one arm's calls and none of the other's. Widening the
  window to ~6s to catch the loop would refuse **9–13% of both arms' genuine
  decisions**.

  The loop's real fix is #1 — publish `search(times≤20)`, verified to take the
  surface from 33 to 34 tools — which collapses a 20–50-call loop into one call.
