# Experiments tab

One home where every experiment is **defined**, running on shared **post-monolith
plumbing**. Each experiment is a name in [`run.py`](run.py); the machinery it
needs lives in the engine (`nethack_core`: snapshot/branch, 17 difficulty knobs,
the encodings' one canonical map model) and in shared infra, not copied per
experiment.

## Post-monolith run model

Every experiment runs the same way after the engine/Hub split:

- **The engine is an external package.** There is no `third_party/NetHack` in
  this repo; `nethack_core` is a dependency and `libnethack.so` is located via
  `NLE_LIB_PATH`. `experiments/common.py` wires this up.
- **This machine is the orchestrator; the model is remote.** The env runs on
  local CPU; tokens stream from **Prime Inference**. Team billing is automatic
  via `PRIME_TEAM_ID` (or `~/.prime/config.json`). See `experiments/common.py`.

## Run any experiment

```bash
python -m experiments.run <name> --smoke   # free/keyless/dry — proves wiring (~$0)
python -m experiments.run <name> --real    # calls the model (cheap defaults)
```

| name | Experiment | Runner (plumbing) | `--smoke` |
|---|---|---|---|
| `encoding`  | **Exp 1** encoding ablations (ASCII/JSON/TOON/IMG vs NetPlay/GlyphBox baselines) | `tools/encoding_eval/` | dry aggregate |
| `harness`   | **Exp 2** harness fixes across frontier models (Gemini/GLM/GPT-5.5) | `configs/eval/*` + cross-model runner | — |
| `continual` | **Exp 3** continual-harness optimization loop | `approaches/continuous_harness/` | `--dry-run` (no API) |
| `explore`   | **Exp 3** go-explore / `branch()` exploration | `approaches/go_explore/` | real (keyless) |
| `variance`  | **Exp 3** cross-seed variance on the 6-floor curriculum | `approaches/analysis/seed_variance.py` | mock |
| `ablations` | Level-modification ablations (vision / health / doors / luck) | `tools/ablation_sweep.py` | dry / verify |

Full write-ups (goal, infra, how-to-run, what-remains) live in
[`docs/experiments/`](../docs/experiments/).

## Status

- `continual` and `explore` run post-monolith today (verified). `explore` is
  keyless, so its `--smoke` is a real short run.
- `encoding` runs via `prime eval run nethack` (proven end-to-end on Qwen3.5-9B).
- `variance` and `ablations` land with their PRs (#6, #5); the tab delegates to
  them and reports clearly if a runner isn't on the current branch yet.
- `harness` (Exp 2) runner is the remaining stub.
