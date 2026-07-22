# NetHack-Hub

The **verifiers / Prime Intellect environment + LLM harness + research/eval
tooling** for training and evaluating language-model agents on NetHack.

This repo was extracted from the former `NetHackHarness` monorepo. It is
**layer 2**: the RL/eval environment and everything built on top of it. It
**depends on** the NetHack engine (**layer 1**, the repo `NetHack-engine`, the
renamed engine half of `NetHackHarness`) and imports it under the
`nethack_core` namespace. The engine never imports the Hub.

```
NetHack-engine  (nethack_core)  <-- external dependency (the fork engine + ctypes binding)
        ▲
        │ imports
NetHack-Hub     (this repo: environments/nethack, approaches/, tools/, docs/)
```

## What's here

- `environments/nethack/` — the verifiers environment (`nethack.py`), the LLM
  harness (`nethack_harness/`: prompt, tools/skills, memory, navigation,
  refiner), configs, experiments, tests, and the Prime Hub wheel-deploy
  machinery (`hatch_build.py`, `pyproject.toml`).
- `approaches/` — research agents built on the env: Go-Explore, Voyager, RLM,
  Continuous-Harness.
- `tools/` — research / evaluation tooling: `encoding_eval/`, `eval_instrument.py`,
  `run_eval.sh`, `trace_analyze.py`, `export_trials.py`, `dashboard.py`,
  `profile_env.py`, `build_valkyrie_model.py`, `build_wiki_index.py`,
  `curriculum_demo.py`, `curriculum_gifs.py`, `knob_gifs.py`, `record_demo.py`,
  `render_floor.py`.
- `configs/endpoints.toml`, `wiki/snapshot.json`, `outputs/` (research outputs),
  and `docs/` (reference docs: eval recipes, parity report, capabilities,
  repo map, etc.).

The web console (`play_server.py`, `tools/webconsole/`, `tools/rollout_view/`,
`Dockerfile.console`, `deploy/wasm-web`, videos) lives in a separate
`NetHack-console` repo. The engine (`nethack_core/`, `nethack_interface/`,
`third_party/NetHack`) lives in `NetHack-engine`.

## The engine dependency

The env sources `nethack_core` from the engine package rather than a vendored
copy. `environments/nethack/pyproject.toml` declares:

```toml
nethack-core @ git+https://github.com/liujonathan24/NetHack-engine.git
```

(If that URL 404s because the engine repo is not yet populated, fall back to
`git+https://github.com/liujonathan24/NetHackHarness.git`.)

`nethack_core` wraps a compiled NetHack fork engine (`libnethack.so`). Installing
the engine package builds/carries that binary; see the engine repo's README.

## Install & eval

```bash
# From this repo, install the env (pulls in nethack-core from the engine repo):
pip install -e environments/nethack

# Eval with verifiers:
cd environments/nethack
vf-eval nethack -m <model> -n <num_episodes>
# or the research eval harness:
bash ../../tools/run_eval.sh
```

`approaches/` and `tools/` import both `nethack_core` (engine) and
`nethack_harness` (this repo's harness package under
`environments/nethack/nethack_harness`), so keep the env installed / on
`PYTHONPATH` when running them.

## The default task

The env runs the **standard full NetHack ascension game** (`nethack.GameSpec` /
`FULL_GAME_SPEC`). The former 13-tier named curriculum ladder has been removed
(see the extraction notes). `nethack_harness/curriculum/` is kept as an empty
seam for a future six-floor "primitives" curriculum that depends on unmerged
fork hooks.

## Prime Hub deploy

The env deploys to the Prime Hub as a wheel that carries the engine `.so` + data
files (`hatch_build.py` build hook + the `artifacts` / `force-include` config in
`environments/nethack/pyproject.toml`). That mechanism is preserved; note the
build-time reconciliation with the external engine dependency described in the
extraction notes.
