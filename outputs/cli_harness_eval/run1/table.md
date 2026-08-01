| Arm | n | Depth (mean ± SE) | BALROG % (mean ± SE) | Died % | Actions used (measured, mean ± SE) | Cost/rollout ($) | sec/call 1st half → 2nd half | Post-death drain |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| control | 5 | 2.20 ± 0.49 | 1.09 ± 0.45 | 80% | total_tool_calls=230.20 ± 62.09 | $6.5440 ± 2.7238 | 4.5s → 5.3s | 0.0 |
| claude_code | 5 | 1.60 ± 0.60 | 1.28 ± 0.53 | 100% | skill_calls=234.60 ± 40.46 | $8.2535 ± 2.5857 | 13.9s → 14.7s | 0.2 |
| prime_agent | 5 | 2.20 ± 0.58 | 1.22 ± 0.52 | 20% | skill_calls=94.60 ± 21.81 | $1.3646 ± 0.3520 | 18.0s → 30.5s | 0.0 |

Notes: 'Actions used' is the MEASURED count, never the nominal 150-call budget -- the control arm's `max_turns` caps LM turns (a no-tool-call turn burns the turn; extra parallel tool calls past the first are dropped), so it gets *at most* the nominal budget in executed skills, while the CLI arms' toolset-side referee grants exactly that many. The arms are therefore NOT budget-matched; do not read a lower control action count as the model choosing to stop early.
