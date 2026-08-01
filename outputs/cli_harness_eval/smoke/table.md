| Arm | n | Depth (mean ± SE) | BALROG % (mean ± SE) | Died % | Actions used (measured, mean ± SE) | Cost/rollout ($) | sec/call 1st half → 2nd half | Post-death drain |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| control | 1 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0% | total_tool_calls=20.00 ± 0.00 | $0.1590 ± 0.0000 | 45.5s → 29.8s | 0.0 |
| claude_code | 1 | 2.00 ± 0.00 | 1.54 ± 0.00 | 100% | skill_calls=8.00 ± 0.00 | $0.0381 ± 0.0000 | 20.2s → 13.2s | 0.0 |
| prime_agent | 1 | 2.00 ± 0.00 | 1.85 ± 0.00 | 0% | skill_calls=20.00 ± 0.00 | $0.1534 ± 0.0000 | 89.4s → 24.5s | 0.0 |

Notes: 'Actions used' is the MEASURED count, never the nominal 150-call budget -- the control arm's `max_turns` caps LM turns (a no-tool-call turn burns the turn; extra parallel tool calls past the first are dropped), so it gets *at most* the nominal budget in executed skills, while the CLI arms' toolset-side referee grants exactly that many. The arms are therefore NOT budget-matched; do not read a lower control action count as the model choosing to stop early.

wrote outputs/cli_harness_eval/smoke/table.md and outputs/cli_harness_eval/smoke/table.json
