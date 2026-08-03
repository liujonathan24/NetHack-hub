# Prime Agent vs Claude Code on NetHack — release plan

The deliverable: a report showing how two CLI agent scaffolds compare at playing
NetHack **when the harness is fair and the accounting is honest** — and that
getting to fairness was most of the work.

## The claims, and the cell that pays for each

| # | Claim | Evidence | New spend |
|---|---|---|---|
| C1 | On identical terms, Prime Agent matches Claude Code on depth and **wins on budget** — fewer LLM calls to every BALROG milestone past 1%, more game-turns per call, lower $ per BALROG point | Tier 1 | ~$180 |
| C2 | Naive harness comparisons are dominated by infrastructure, not agents — and the errors ran against Prime Agent | exp2 forensics + exp3a diagnostic (already run) | $0 |
| C3 | Full observability is worth ~2 dungeon levels; the ranking of scaffolds is not an artifact of the vision setting | Tier 2 + exp2's 16-cell A/B | ~$180 |
| C4 | The scaffold result transfers to a stronger model family | Tier 3 (GPT-5.6-sol) | ~$550–700 |
| C5 | Both agents progress 1–2 orders of magnitude slower than human ascension pace; rate curves + turns-to-milestone against human extrapolations | analysis of Tiers 1–3 | $0 |
| C6 | Lean, on-demand observations (BBOX_MIN) halve cost without hurting depth, and expose each scaffold's *information-purchasing policy* (reveal ~2% of calls when the feed is free → 15–21% when it costs a call) | exp2 vs exp3 traces | $0 |
| C7 | *(appendix)* Post-compaction efficiency: does either scaffold degrade after its context is compacted? | retrospective on exp2 long rollouts + exp3b events | $0 |

## The cells

Everything: BBOX_MIN encoding · surface `netplay_true,reveal,search` (33 tools,
no rollback) · 5 seeds (0–4) · 200-call cap · 2h timeout · prescriptive
dismissal notices · pre-first-turn watchdog coverage · team billing.

| tier | cell | arm | model | vision |
|---|---|---|---|---|
| 1 | `V1__claude_code` | claude_code | glm-5.2 | **on** |
| 1 | `V1__prime_agent` | prime_agent | glm-5.2 | **on** |
| 2 | `FOG__claude_code` | claude_code | glm-5.2 | **off** (fog of war) |
| 2 | `FOG__prime_agent` | prime_agent | glm-5.2 | **off** |
| 3 | `GPT__claude_code` | claude_code | gpt-5.6-sol | on |
| 3 | `GPT__prime_agent` | prime_agent | gpt-5.6-sol | on |

Six cells, ~$910–1,060 total. Tiers are separable: 1 alone supports the
headline; 2 and 3 can follow after reading tier 1.

**Why the off-diagonal cells are paired, not single.** The unit of comparison is
always cc-vs-pa *within a condition*. A lone fog cell or lone GPT cell could
only be read against a different condition's other arm — which is exp2's
confounding mistake all over again. Every condition ships both arms or not at
all.

**Cells deliberately NOT run, and why:**
- *Encoding factorial* — settled in exp2 (BBOX ≈ BBOX_JSON ≈ B0, SPARSE_ONDEMAND
  worse); BBOX_MIN's effect is measured against exp3a. Re-running it buys noise.
- *GPT × fog* — the fog axis is established at GLM; the model tier tests
  transfer of the scaffold ranking, not a full factorial.
- *balrog80 surface* — different action-surface question, different release.
- *MAX_PARALLEL=1 cell* — measured no-op (cc emits exactly 1 call/turn; 0.0% of
  its calls fall inside the batch window).
- *800-call horizon* — median death is at ~110–150 calls; 2/10 exp3a rollouts
  reached 200. The cap-check guards the assumption; if GPT-5.6-sol survives to
  the cap >20% of the time, the capped seeds get extended, not the whole grid.

**Known risks, stated up front:**
- GPT-5.6-sol may hit the 200-cap often (stronger model, longer survival). Plan:
  extend capped seeds to 400 calls only.
- n=5 vs the documented 2.2-point variance floor: depth differences inside
  ±2 points are reported as parity. The budget-efficiency claims rest on
  paired per-seed comparisons, which are tighter.
- BBOX_MIN was the encoding on which prime_agent looked relatively best in
  exp2's (noisy) cells; stated in limitations.

## Report skeleton

1. **Setup** — engine (pinned ac80760), BALROG progression (max *and* min,
   always), arms, action surface, what a "call" is on each scaffold.
2. **Why naive comparisons mislead** — exp2 forensics: watchdog kills 30v10,
   402 stubs scored as shallow rollouts, rollback deadlock, wear-loops, a cost
   meter 6.6× under. The harness is part of the experiment.
3. **Making it fair** — the fix list, each with its measured motivation: search
   published to both arms; the pa-only batching rule stripped; sandbox on;
   rollback removed (both exp3a kills sat directly downstream of it; undo-death
   is a crutch no baseline has); prescriptive prompt-dismissal notices;
   watchdog partial traces + pre-first-turn coverage; provider-priced cost
   meter.
4. **Results on equal terms** (tier 1) — depth, BALROG max/min, milestone speed
   in game-turns AND LLM calls (calls default: it is the budget and the $ axis),
   died %, $ per BALROG point.
5. **Observability** (tier 2 + exp2) — fog costs ~2 levels; scaffold ranking
   under both settings.
6. **Model transfer** (tier 3) — same grid on GPT-5.6-sol.
7. **Human calibration** — carry-forward rate curves vs constant-rate
   extrapolations of published ascension lengths (10.5k/24.5k/75k turns), with
   the concavity caveat: compare turns-to-milestone, not raw %/turn.
8. **Appendices** — compaction: detect context-reset events in traces (measured:
   input drops 96–98k → ~0 at call ~89–103 in cc long rollouts), compare BALROG/
   call and dead-call rate before vs after per arm. Encoding economics: exp3a
   cost per rollout halved vs BBOX prediction; cache share 52→86%.
   reveal-policy comparison. Limitations.

## Execution order

1. Tier 1 now (~90 min). Read it.
2. Tier 2 + Tier 3 together (~2h, mostly GPT wall clock), or tier 2 first if
   the GPT budget needs a separate sign-off.
3. Analyses + report assembly ($0, no wall clock dependency).
