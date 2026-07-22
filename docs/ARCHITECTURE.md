# Architecture — repositories & their functions

The stack is split into three repositories by responsibility, plus the shared
residency registry. Dependencies point **downward** only: the Hub depends on the
engine, the console reads both, and nothing points back up into the Hub (the
engine imports nothing from the Hub — enforced by the coupling fix). Every
experiment runs **post-monolith**: this machine orchestrates the environment
(CPU) and the model is called remotely from Prime Inference.

```mermaid
graph TB
  subgraph ENGINE["🛠️ NetHack-engine · layer 1 · the substrate"]
    direction TB
    FORK["third_party/NetHack (submodule)<br/>custom NetHack fork → libnethack.so<br/>invocation hooks · luck knob"]
    NC["nethack_core<br/>ctypes binding over the fork<br/>• NetHackCoreEnv (nle-gym path)<br/>• EngineEnv: snapshot / restore / branch,<br/>&nbsp;&nbsp;tune (17 knobs), modify, state hooks<br/>• map_model · glyphs · rewards"]
    NI["nethack_interface<br/>typed Observation + RawAction<br/>(pure substrate — no Hub dep)"]
    FORK --> NC
  end

  subgraph HUB["📦 NetHack-hub · layer 2 · the environment"]
    direction TB
    ENV["environments/nethack — pkg 'nethack'<br/>Verifiers env · GAME_SPECS:<br/>full_nle · six_floor_primitives"]
    HARN["nethack_harness<br/>skills + code-mode · prompt + BALROG ·<br/>navigation · memory · interface (typed Action) ·<br/>curriculum (six-floor)"]
    APPR["approaches<br/>continuous_harness · go_explore ·<br/>voyager · analysis (seed variance)"]
    TOOLS["tools<br/>encoding_eval · harness_sweep · ablation_sweep"]
    TAB["experiments/ — THE TAB<br/>run.py + common.py (post-monolith plumbing)<br/>encoding · harness · continual · explore · variance · ablations"]
    ENV --> HARN
    TAB --> APPR
    TAB --> TOOLS
    TAB --> ENV
  end

  subgraph CONSOLE["🖥️ NetHack-console · layer 3 · the viewer"]
    RV["tools/rollout_view · deploy/<br/>trace viewer · web play · replay export"]
  end

  subgraph RES["🌐 RL-Residency/residency-environments"]
    REP["environments/nethack (branch feat/nethack)<br/>= the Hub env, contributed as a subdir"]
  end

  PRIME["☁️ Prime Inference<br/>Qwen / GLM / … — model runs remote;<br/>THIS box orchestrates the env (CPU)"]

  HUB -->|"depends on (external pkg): imports<br/>nethack_core + nethack_interface"| ENGINE
  CONSOLE -->|reads nethack_interface + nethack_core| ENGINE
  CONSOLE -->|reads nethack_harness + nethack| HUB
  HUB -.->|env subdir published| RES
  TAB -.->|"prime eval / vf-eval (team-billed)"| PRIME
```

## Repositories

| Repo | Layer | Function |
|---|---|---|
| **NetHack-engine** (`NetHackHarness`) | 1 · substrate | The controllable NetHack: `nethack_core` (ctypes over the fork → `libnethack.so`; `EngineEnv` with snapshot/restore/**branch**, 17 tune knobs, `modify`, state hooks) + `nethack_interface` (typed `Observation` + `RawAction`). Imports nothing from the Hub. |
| **NetHack-hub** (`NetHack-hub`) | 2 · environment | The training/eval env (`nethack`): skills + code-mode, prompt + BALROG progression, navigation, memory, typed `Action`, the six-floor curriculum; research `approaches/`; eval `tools/`; the **experiments tab**. Depends on the engine as an external package. |
| **NetHack-console** (`NetHack-console`) | 3 · viewer | Rollout viewers, web play, replay export. Reads the engine (`nethack_interface`/`nethack_core`) and the Hub (`nethack_harness`/`nethack`). |
| **residency-environments** (`RL-Residency`) | registry | Shared org repo; the Hub's `environments/nethack/` is contributed as a subdirectory on branch `feat/nethack`. |

## The experiments tab

`experiments/run.py <name> [--smoke | --real]` is the one entry point. Each
experiment is **defined** there and **delegates** to its runner; the shared
post-monolith plumbing (`experiments/common.py`: locate `libnethack.so`, team
billing, `uv`) is not duplicated per experiment. Full write-ups live in
[`docs/experiments/`](experiments/).

| name | Experiment | Runner |
|---|---|---|
| `encoding`  | Exp 1 encoding ablations | `tools/encoding_eval/` |
| `harness`   | Exp 2 harness modifications | `tools/harness_sweep.py` |
| `continual` | Exp 3 continual-harness loop | `approaches/continuous_harness/` |
| `explore`   | Exp 3 go-explore / `branch()` | `approaches/go_explore/` |
| `variance`  | Exp 3 cross-seed variance (6-floor) | `approaches/analysis/seed_variance.py` |
| `ablations` | level-modification ablations | `tools/ablation_sweep.py` |
