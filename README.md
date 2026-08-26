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

Arms land here as their 3x5 grids complete. Control: `raw/e14_base_r*`,
`rendered/base_e14.html`.
