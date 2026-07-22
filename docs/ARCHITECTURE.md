# Architecture — repositories & their functions

`nethack-engine` turns upstream NetHack into a controllable substrate — the NLE
replacement, with customizability and curriculum-learning built in. It powers two
downstream repos: `nethack-hub` (the Prime Intellect environment) and
`nethack-console` (the viewer).

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Inter, ui-sans-serif, system-ui, sans-serif','fontSize':'14px','lineColor':'#94a3b8'},'flowchart':{'curve':'basis','htmlLabels':true,'nodeSpacing':38,'rankSpacing':64}}}%%
flowchart TB
  subgraph ENGINE["nethack-engine — the NLE replacement"]
    direction LR
    SRC["Multi-threaded<br/>NetHack source"]:::src
    subgraph CUST["Customizability"]
      direction TB
      S1["Snapshotting"]:::cust
      S2["Difficulty knobs"]:::cust
    end
    subgraph CURR["Curriculum learning"]
      direction TB
      C1["Custom dungeon floors"]:::curr
      C2["Guided teleport"]:::curr
      C3["Custom navigation"]:::curr
    end
  end

  subgraph HUB["nethack-hub — Prime Intellect"]
    direction TB
    H1["Continual harness"]:::hub
    H2["Go-Explore"]:::hub
    H3["Voyager"]:::hub
  end

  subgraph CONSOLE["nethack-console — viewer"]
    direction TB
    V1["Rollout viewer"]:::con
    V2["Web play"]:::con
    V3["Replay export"]:::con
  end

  ENGINE ==> HUB
  ENGINE ==> CONSOLE

  classDef src  fill:#f8fafc,stroke:#64748b,color:#0f172a;
  classDef cust fill:#ecfeff,stroke:#0e7490,color:#083344;
  classDef curr fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b;
  classDef hub  fill:#ecfdf5,stroke:#047857,color:#064e3b;
  classDef con  fill:#fff7ed,stroke:#b45309,color:#7c2d12;
  style ENGINE  fill:#ffffff,stroke:#cbd5e1,stroke-width:2px;
  style CUST    fill:#f6feff,stroke:#a5f3fc;
  style CURR    fill:#f7f8ff,stroke:#c7d2fe;
  style HUB     fill:#ffffff,stroke:#6ee7b7,stroke-width:2px;
  style CONSOLE fill:#ffffff,stroke:#fdba74,stroke-width:2px;
```

## The three parts

- **nethack-engine** — the NLE replacement. Wraps a custom NetHack fork through a
  ctypes binding and adds what the `nle` wrapper can't: **customizability**
  (in-memory snapshotting, live difficulty knobs) and **curriculum-learning**
  hooks (custom dungeon floors, guided teleport, custom navigation).
- **nethack-hub** — the Prime Intellect environment: the `nethack` Verifiers env
  and the exploration experiments (continual harness, Go-Explore, Voyager).
- **nethack-console** — the viewer: rollout viewer, web play, replay export.

Dependencies flow one way — the hub and console build on the engine; the engine
never depends on them.
