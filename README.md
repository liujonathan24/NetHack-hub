# E15 probe-grid data branch

Seed registry, raw transcripts, and rendered game viewers for the E15
failure-mode probe experiments (and their E14 uncapped base control).
Code lives on the code branches; this branch is data only.

- `registry.json` — every rollout: arm, rep, seed, metrics, attempt counts,
  turn-file paths, tier provenance (tool_tier / registry hash / commit).
  Regenerated as arms complete; the live view is the "E15 Probe Grid" artifact.
- `raw/<cell>/` — per-cell: `config.toml` (resolved cell config),
  `engine_provenance.json`, `traces.jsonl.gz` (per-rollout metrics),
  `<seed>_<pid>_<ts>.ndjson.gz` (full per-turn transcripts: rendered
  observation, model reasoning, tool call/result, engine state). Multiple
  files per seed = infra re-runs; the largest is the completed attempt.
- `rendered/<arm>.html` — self-contained replay viewer (tools/game_viewer):
  all 15 games of an arm with per-move frames, aligned reasoning, call log.
  Open locally in any browser; no server needed.

Run 2 (clean): 1 rep x 5 seeds per arm; reps 2-3 on request. Control:
`raw/e14_base_r*`, `rendered/base_e14.html` (E14, 3x5, valid).

- `raw/<tier>_r1/`, `rendered/<tier>_r1.html` — clean run-2 probe arms.
- `raw/<tier>_fix1/` — fixed-code reruns of the P arms (P1 gate-before-dispatch
  + deficit-gated firing; P2 retreat-only crisis text + arrival-norm pacing;
  P3 rollback death-window attempt — the window did NOT operate in fix1, a
  fix2 is pending; see the tracker artifact for audit flags per arm).
- `invalidated/wrongdoc/` — the first probe run (served the E10-era skill doc
  instead of Baseline-v2; every arm ~half of control). Kept for provenance,
  not comparable to anything.
- P1 r1 is a placebo (the advisory fired after the descent had executed);
  v2_bjson r1 is invalid (harness loop broke in 3/5 rollouts).
