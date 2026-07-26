| Arm | n | Depth (mean ± SE) | BALROG % (mean ± SE) | Died % | Actions used (measured, mean ± SE) | Cost/rollout ($) | sec/call 1st half → 2nd half | Post-death drain |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| control | 1 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0% | total_tool_calls=20.00 ± 0.00 | $0.1725 ± 0.0000 | 12.9s → 1.5s | 0.0 |
| claude_code | 1 | 2.00 ± 0.00 | 1.54 ± 0.00 | 0% | skill_calls=20.00 ± 0.00 | $0.1427 ± 0.0000 | 29.9s → 13.4s | 0.0 |
| prime_agent | 1 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0% | skill_calls=20.00 ± 0.00 | $0.1941 ± 0.0000 | 28.8s → 40.3s | 0.0 |

Notes: 'Actions used' is the MEASURED count, never the nominal 150-call budget -- the control arm's `max_turns` caps LM turns (a no-tool-call turn burns the turn; extra parallel tool calls past the first are dropped), so it gets *at most* the nominal budget in executed skills, while the CLI arms' toolset-side referee grants exactly that many. The arms are therefore NOT budget-matched; do not read a lower control action count as the model choosing to stop early.

wrote outputs/cli_harness_eval/smoke4/table.md and outputs/cli_harness_eval/smoke4/table.json
