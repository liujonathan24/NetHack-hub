# Experiment 1 — Encoding Ablations

> Status: **plan + scaffolding landed** (this PR). The text/pixel encoding matrix
> is runnable today; the two prior-framework baselines (NetPlay, Glyphbox) are
> registered as first-class variants but their native renderers are stubs that
> raise until ported (§e).

---

## (a) Goal & why we care

**Goal.** Evaluate our canonical map encodings against established LLM-based
baselines, specifically **NetPlay** and **Glyphbox**. Compare the *token
efficiency* and *long-horizon planning capabilities* of our structured
representations against these existing state-of-the-art approaches.

**Why we care.** We need to demonstrate that our observation extraction and
action APIs (such as our code-execution environment) provide **superior
grounding** for the model than prior frameworks. Concretely, Experiment 1 is the
apples-to-apples measurement that turns "our encodings are better" from an
assertion into a table: for a fixed model and fixed task, does a given
observation encoding let the model descend further (long-horizon) while spending
fewer tokens per turn (efficiency)?

The unit under test is the **encoding** — how a NetHack game state is serialized
into the bytes the LLM reads — holding the model, the task/tier, and (where
possible) the action surface fixed.

---

## (b) Infrastructure that exists — technical

### The encodings (the "variant" seam)

A rollout's observation form is selected by the `variant` env-arg, resolved once
per rollout into a `PromptSpec`.

- **Encoder core.** `environments/nethack/nethack_harness/prompt/map_encoders.py`
  serializes the canonical `MapModel` (built by `nethack_core.map_model.build_map_model`
  from `raw_obs`) as either `json_encode(model, detail=...)` or
  `toon_encode(model, detail=...)`. Both project the **same** model at a
  selectable `detail` (`full` = rich entity attrs + RLE grid; `minimal` =
  kind/coord/desc only, no grid) — so JSON and TOON can never diverge in content,
  only in token layout.
- **Variant → PromptSpec registry.**
  `environments/nethack/nethack_harness/prompt/prompt_spec.py` :: `VARIANT_REGISTRY`
  / `resolve_spec(variant, system_prompt)`. A `PromptSpec` bundles the four things
  that define a rollout's prompt: the observation form (`ObsSpec`: `ascii` vs
  `img`), the system prompt, the **per-turn template** (`turn_template(...)` that
  renders the user message), and the tools (`ToolSpec`).
- **Per-turn templates** (all in `prompt_spec.py`):
  - `_canonical_template` — B1 and most variants (ASCII grid + status/inventory).
  - `_structured_map_template("json"|"toon")` — emits `=== MAP (JSON|TOON) ===`
    built from the canonical map model; `map_detail` rides on `state["map_detail"]`.
  - `_image_template("glyph"|"tty")` — returns a multimodal `[image_url, text]`
    content list; `glyph` → `GlyphMapper` tiles, `tty` → tty-text raster
    (`nethack_harness/prompt/image_render.py`). The image is the sole spatial
    channel; the text block is journal + status + inventory only.
  - `_formatter_template(fn)` — adapts a legacy `_format_obs_*` renderer
    (`rendering.py`) to the template signature. Used by B (BALROG NL scene),
    G (Glyphbox-ish), R (summarize-and-reset), and now the NETPLAY/GLYPHBOX stubs.

**The canonical map encodings under test (ours):**

| variant | form | notes |
|---|---|---|
| `B1` | ASCII grid | default shipping render; `_canonical_template` |
| `JSON` | structured JSON map | `map_detail=full`/`minimal`; `map_encoders.json_encode` |
| `TOON` | in-repo TOON (token-frugal) | same model as JSON; `map_encoders.toon_encode` |
| `IMG` | rendered glyph tiles (pixels) | VLM only |
| `IMG_TTY` | tty raster (pixels) | VLM only |

`map_detail` (`full` / `minimal`) is threaded end-to-end:
`load_environment(map_detail=...)` → `NetHackVerifiersEnv.__init__` (`self.map_detail`)
→ `setup_state` writes `state["map_detail"]` (nethack.py ~L434) →
`_structured_map_template` reads it. So the JSON/TOON cells can be run at two
detail levels without code changes.

### How prime_runner drives the encoding × model matrix

`tools/encoding_eval/prime_runner.py` is the operational wiring that turns a
matrix cell `{variant, map_detail, model}` into a real eval:

1. **Cell → command.** `build_command(cell, ...)` maps the cell to a
   `prime eval run nethack` invocation. It sets `--env-args` with `variant`,
   `trace_dir`, `max_turns`, `tier`, `n_examples` (+ `map_detail` when present),
   forces `--max-concurrent 1`, `--save-results`, `--disable-tui`, and points
   `--output-dir` at `<run_dir>/<cell>/eval_out`.
2. **Model routing.** `_model_for_cell(cell)` picks the model: pixel variants
   (`IMG`, `IMG_TTY`) → the VLM (`qwen/qwen3-vl-8b-instruct`); all text encodings
   → the instruct model (`Qwen/Qwen3.5-4B`). A cell's explicit `model` overrides
   both. The baseline encodings (NETPLAY/GLYPHBOX) are text → instruct model.
3. **`import nethack` collision fix.** It prepends `environments/nethack` onto
   `PYTHONPATH` so `import nethack` resolves to the worktree env, not a stale
   `site-packages/nethack.py` shim (the Prime hub publishes this env as the bare
   module `nethack`).
4. **Trace capture → samples.** Each cell's `trace_dir` collects per-turn NDJSON.
   `make_runner(...)` returns a `runner(cell) -> [sample]` closure that prefers
   prime's local `results.jsonl` rows (`_samples_from_results`: proper rubric
   rewards + `token_usage`, enriched with `tokens_per_turn` and `dollars` via
   `_PRICES`), and falls back to deriving samples from the NDJSON trace
   (`_sample_from_trace`: descent from max dungeon level).
5. **Matrix orchestration.** `tools/encoding_eval/run.py :: run_matrix(matrix,
   runner=...)` iterates `encodings × models`, calls the runner per cell, and
   passes the per-cell sample lists to `aggregate_cells`. The runner is an
   **injectable seam** — tests inject a stub, so the whole pipeline is exercisable
   with zero model calls.

`prime_runner.py`'s `__main__` writes `<run_dir>/table.json` and
`<run_dir>/table.md` and prints the markdown table.

### What aggregate.py computes

`tools/encoding_eval/aggregate.py :: aggregate_cells({cell: [sample,...]})` →
`{"rows": {cell: metrics}}`, reusing `tools.eval_instrument.summarize_eval` and
`nethack_harness.prompt.balrog`. Per encoding it reports:

- **Long-horizon / progression:**
  - `descent_rate` + Wilson 95% CI (`ci_lo`/`ci_hi`) — fraction of rollouts that
    reached dlvl ≥ 2 (`summarize_eval`).
  - `max_dlvl` — deepest dungeon level touched across the cell's rollouts.
  - `progression_score` + `progression_tier` — BALROG-style P(ascend) proxy from
    `(max_dlvl, xp_level)` via `balrog.progression_score` (analytic
    `(DL/50)^1.3·(XL/30)^0.6`), bucketed `spawn/early/past_mines/midgame/endgame`.
  - `avg_score`, `failure_taxonomy` — from `summarize_eval`.
- **Token efficiency:**
  - `tokens_per_turn` — mean over rollouts of `(input+output tokens)/num_turns`
    (from prime's `token_usage`). This is the headline efficiency number.
  - `dollars_per_run` — mean $ per rollout from `_PRICES` × token usage.

`table_to_markdown(table)` renders the comparison table (`n`, `descent_rate`,
`progression_tier`, `max_dlvl`, `tokens_per_turn`, `dollars_per_run`).

### How traces / replay capture the exact LLM input per encoding

The **replay layer is what makes the comparison auditable**: it lets us show the
*exact bytes/pixels* each encoding fed the model, side by side.

- **Trace writer.** `nethack_harness/helpers.py :: _write_trace_entry` (fires
  when `trace_dir` is set) writes one NDJSON line per turn with, among others:
  `turn`, `variant`, `raw_grid` (human tty frame), `status`/`dlvl`/`hp`,
  `rendered_user_message` (obs text), and crucially **`rendered_user_content`** —
  the full multimodal content the model saw. For IMG/IMG_TTY the image is written
  to `<run_dir>/images/` and referenced by path (not inline base64), so the exact
  pixels are replayable.
- **Replay renderer.** `tools/encoding_eval/replay.py :: render_replay(run_dir,
  form="human"|"llm")` renders a recorded rollout as either the human game-state
  form (`raw_grid` frames) or the **exact LLM-input form** (`rendered_user_content`,
  text + image path refs). `REPLAY_LOG_KEYS` is the stable on-disk seam the Group B
  `tools/launchpad` viewer reads.

So for any encoding you can dump the literal input the model received per turn —
the ground truth for a token-efficiency and grounding comparison.

### The code-execution / action API (the "grounding" claim)

Two orthogonal seams: `variant` selects the **observation encoding**; `interface`
selects the **action surface**:

- `interface="skill"` (default) — one OpenAI-function-calling tool per skill
  (`nethack_harness/tools/skills.py`), grouped into named `skill_set`s
  (`helpers.py`): `full`, `netplay`, `dir8`, `move`, comma-lists.
- `interface="code"` — a single `code(source=...)` tool executing Python against a
  curated `nh` namespace (`nethack_harness/tools/code_mode.py`) — this is the
  code-execution environment the "why we care" refers to, and is itself modeled on
  Glyphbox's action surface.

Experiment 1's grounding argument is exactly the interaction of these two seams:
our structured encodings **plus** our code/skill action API vs. the prior
frameworks' native encodings + action surfaces.

---

## (c) What it does — the narrative

Imagine you're handing the same NetHack game to the same language model over and
over, but each time you describe the game differently. Once you hand it an ASCII
map. Once you hand it a compact JSON object listing the player, the monsters, and
a run-length-encoded grid. Once a screenshot. Once the exact prose description
that a competing research system (NetPlay) would have shown its model. Experiment
1 asks a simple question: **which description lets the model play best, and which
description is cheapest to send?**

"Play best" is measured by how deep into the dungeon the model gets — NetHack is
a long-horizon game, so descending multiple floors is real evidence of planning,
not luck. "Cheapest" is measured in tokens per turn: a description that wins on
depth but costs 3× the tokens is a different trade-off than one that wins on both.

The experiment runs a **matrix**: every encoding × the models we can afford to
call, on a fixed exploration task, for a fixed number of turns. Every turn of
every game is recorded — not just the score, but the literal text or image the
model was shown. That recording is the honesty mechanism: when we claim "our JSON
encoding beat NetPlay's prose at half the tokens," anyone can replay both games
turn-by-turn and see the exact inputs.

The punchline we're trying to earn: our structured observation extraction, paired
with our code-execution action API, gives a small model **better grounding** —
more descent per token — than the observation formats prior frameworks used. If
true, that's the case for our whole harness design.

---

## (d) How to run it end-to-end

### Setup (once)

```bash
# Clone the hub and make the env importable.
git clone https://github.com/liujonathan24/NetHack-hub && cd NetHack-hub

# Engine (read-only) provides NLE + nethack_core.
export NLE_LIB_PATH=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness/third_party/NetHack/src/build/libnethack.so
export PYTHONPATH="$PWD/environments/nethack:/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness:$PYTHONPATH"
# uv at /home/jl0796/.local/bin/uv; hub pyproject needs
# tool.hatch.metadata.allow-direct-references=true (already merged, PR #4).
```

### A single real eval (proven-working shape)

```bash
prime eval run nethack \
  -m Qwen/Qwen3.5-4B --provider prime \
  --num-examples 1 --max-tokens 2048 \
  --env-args '{"variant":"JSON","map_detail":"full","trace_dir":"outputs/evals/exp1_smoke/JSON__full","max_turns":40,"tier":"corridor_explore","n_examples":8}'
```

The env runs **locally** (CPU on this box); the model is called **remotely** from
Prime Inference (`api.pinference.ai`). Swap `variant`/`map_detail` to move across
the matrix. **Do not run the full matrix casually — it costs money.** Use
`--dry-run` on the runner (below) to re-aggregate existing traces for free.

### The full matrix via prime_runner

```bash
python -m tools.encoding_eval.prime_runner \
  --run-dir outputs/evals/exp1 \
  --num-examples 8 --rollouts-per-example 1 \
  --max-tokens 2048 --max-turns 40 \
  --tier corridor_explore --n-examples 8
# add --include-baselines once NETPLAY/GLYPHBOX renderers are ported (§e)
# add --dry-run to re-aggregate already-captured traces without model calls
```

### The encoding × model (× map_detail) matrix

**Encodings (rows).** Ours: `B1` (ASCII), `JSON:full`, `JSON:minimal`,
`TOON:full`, `TOON:minimal`, `IMG`, `IMG_TTY`. Baselines (once ported):
`NETPLAY`, `GLYPHBOX`.

**Models (columns).** Text encodings → instruct models available on Prime
Inference: `Qwen/Qwen3.5-{0.8B,2B,4B,9B}`. Pixel encodings (`IMG`, `IMG_TTY`) →
the VLM `qwen/qwen3-vl-8b-instruct`. `prime_runner._model_for_cell` routes text
vs. pixel automatically; pass an explicit per-cell `model` to sweep model size.

| encoding | detail | model class | code file |
|---|---|---|---|
| B1 | — | instruct | `_canonical_template` |
| JSON | full / minimal | instruct | `map_encoders.json_encode` |
| TOON | full / minimal | instruct | `map_encoders.toon_encode` |
| IMG | — | VLM | `_image_template("glyph")` |
| IMG_TTY | — | VLM | `_image_template("tty")` |
| NETPLAY | — | instruct | `_format_obs_netplay` (stub) |
| GLYPHBOX | — | instruct | `_format_obs_glyphbox_native` (stub) |

To sweep `map_detail`, add both `{"variant":"JSON","map_detail":"full"}` and
`{"variant":"JSON","map_detail":"minimal"}` cells (same for TOON). To sweep model
size, set `matrix["models"]` to a list of Qwen ids instead of `[None]`.

### Where results / tables land

- Per-cell traces: `outputs/evals/<run>/<VARIANT[__detail]>/*.ndjson`
  (+ `images/` for pixel cells, + `eval_out/**/results.jsonl`).
- Aggregated table: `outputs/evals/<run>/table.json` and `.../table.md`
  (`aggregate.table_to_markdown`).
- Per-rollout audit: `render_replay(run_dir, form="human")` and
  `render_replay(run_dir, form="llm")` — the latter dumps the exact per-turn LLM
  input for each encoding.

---

## (e) Baselines gap — NetPlay & Glyphbox

### What is present today

The repo already engages both frameworks, but **not** as comparable observation
encodings:

- **NetPlay is studied as an *action-layer* comparison, not an encoding.**
  `docs/netplay-vs-our-harness.md` (+ `netplay-parity-report.md`,
  `netplay-parity-writeup.md`) give a full side-by-side of NetPlay's skills/level
  model vs. ours. In code, the `N` variant reuses **our** canonical ASCII obs and
  only swaps in NetPlay's *skill set* (`helpers.py`, `skill_set == "netplay"`:
  `move_to`, `explore_and_descend`, `attack`, … — no low-level `move`). So `N`
  varies the **actions**, holding our encoding fixed — the opposite of Experiment 1.
- **Glyphbox is present as (i) an action surface and (ii) an *approximate* obs.**
  `nethack_harness/tools/code_mode.py` is an explicit port of Glyphbox's
  code-execution action surface (`interface="code"`). The `G` variant
  (`_format_obs_glyphbox`) is documented as the "closest analog" but **reuses our
  `format_observation_as_chat` render** — its docstring says the real Glyphbox
  delta is the *intent* to pair with code-mode, not a native serialization.

**Conclusion:** neither NetPlay's nor Glyphbox's *native observation encoding* is
present. `N` and `G` compare our render under a different tool set — an
apples-to-oranges token-efficiency comparison. A faithful Experiment 1 needs the
prior frameworks' own state serializations behind the same `variant` seam.

### What this PR added (the seam, as stubs)

Two first-class variants are now **registered and routable** but **raise
`NotImplementedError` until ported** — so the matrix is executable end-to-end and
fails loudly rather than silently mislabeling our render as a baseline:

- `NETPLAY` → `rendering._format_obs_netplay` (registered in
  `_VARIANT_FORMATTERS` and `VARIANT_REGISTRY`).
- `GLYPHBOX` → `rendering._format_obs_glyphbox_native`.
- `prime_runner.BASELINE_ENCODINGS` + a `--include-baselines` flag append these
  two cells (text → instruct model). Off by default, so the shipped matrix is
  unchanged.

### How to make each a comparable encoding (the port)

**NetPlay** — *observation format:* NetPlay never shows an ASCII grid. Its LLM
sees `describe_current_state()` (`netplay/nethack_agent/agent.py`) rendered over a
**persistent `Level` model** (`netplay/nethack_agent/tracking.py`): a
natural-language enumeration of the discovered **room/corridor graph** (nodes +
exits), per-tile memory (`has_seen`, `search_count`, `door_open_attempts`), a
described list of visible + remembered items/monsters, the message log, inventory,
and the blstats status line. See `docs/netplay-vs-our-harness.md §2` for the full
`Level` pseudocode.

*How to slot it in:* implement `_format_obs_netplay(structured, journal, state,
journal_max_chars)`. The blocker is state: NetPlay's description projects a
*stateful* `Level` that accumulates across turns, whereas our harness rebuilds a
stateless grid each step. Two options:
  1. Port a minimal `tracking.Level` (has_seen / room graph / search_count / door
     attempts) into `nethack_harness`, update it each turn from `state["raw_obs"]`,
     and render its `describe_current_state`.
  2. Vendor NetPlay's renderer directly against our `raw_obs` (less faithful on the
     room-graph memory, but a first cut).
Keep the **action surface fixed** via the existing `netplay` `skill_set` so the
comparison isolates the encoding.

**Glyphbox** — *observation format:* Glyphbox serializes the grid with its own
glyph→text scheme using `nle.nethack` glyph-class routing (see
`nethack_harness/navigation/pathfinding.py`, which notes where we deliberately
diverge from that routing), plus its own status/inventory/message layout. The
token footprint of that native serialization differs from our cleaned
`glyph_clean_chars` LUT — and that difference is exactly what Experiment 1
measures.

*How to slot it in:* implement `_format_obs_glyphbox_native(...)` to emit
Glyphbox's own serialization (vendor/port from `github.com/kenforthewin/glyphbox`),
and pair it with `interface="code"` (already Glyphbox's action surface in
`code_mode.py`). Run `GLYPHBOX` against the same tier/model as our encodings.

Because both slot behind the same `variant` seam, they flow through `prime_runner`
(same `trace_dir`, same `results.jsonl`, same token accounting) and
`aggregate.py` unchanged — so token efficiency + descent land in the same table,
apples-to-apples, and the replay viewer dumps their exact per-turn LLM input.

---

## (f) What remains / open questions / risks

**Remains (the real work):**

1. **Port `_format_obs_netplay`** — the `tracking.Level` state model + its
   `describe_current_state` renderer (§e option 1 or 2). Highest-value baseline;
   also the most work because of the persistent state.
2. **Port `_format_obs_glyphbox_native`** — Glyphbox's native glyph→text
   serialization; pair with `interface="code"`.
3. **Decide the action-surface control.** For a clean *encoding* ablation, hold
   the action surface fixed across all encoding rows (e.g. the `netplay` skill_set
   or `interface="code"`). Note that NetPlay's and Glyphbox's own encodings were
   co-designed with their own action surfaces — so we should report **both** a
   fixed-action comparison (isolates encoding) and a native-pairing comparison
   (each framework as its authors intended).
4. **Token counting for the stubs.** `tokens_per_turn` comes from prime's
   `token_usage`; confirm the baseline renders produce comparable tokenization
   (same tokenizer) so the efficiency numbers are fair.

**Open questions:**

- **What tier / horizon best exposes long-horizon planning?** `corridor_explore`
  is the proven default, but multi-floor descent (higher `max_turns` + a tier with
  a large enough `max_episode_steps`) is where encoding differences on *planning*
  (not just single-screen reading) should show. See the `prime_runner` NOTE on
  `max_turns` vs. tier `max_episode_steps`.
- **map_detail interaction.** Does `minimal` (no grid) close the token gap without
  hurting descent? That's a headline sub-result of the ours-vs-ours rows.
- **Pixel vs. text fairness.** IMG/IMG_TTY require the VLM; their `tokens_per_turn`
  (image tokens) isn't directly comparable to text encodings — report them in a
  separate sub-table.

**Risks:**

- **Cost.** The full matrix is many paid remote calls. Mitigation: `--dry-run`
  re-aggregation, tiny `--num-examples`/`--max-turns` for smoke, and staging
  (ours-vs-ours first, then add baselines).
- **Baseline fidelity.** A half-ported NetPlay/Glyphbox render would make the
  comparison misleading. The stubs **raise on purpose** so a partial port can't
  silently ship as a baseline number.
- **Model availability drift.** The TOML default `Qwen/Qwen3.5-VL-7B` does **not**
  exist on Prime Inference; `prime_runner` pins the real VLM
  `qwen/qwen3-vl-8b-instruct`. Re-verify model ids before a run.
- **`import nethack` name collision** with the published hub module — already
  handled by `prime_runner`'s `PYTHONPATH` prepend, but any hand-run must replicate it.

---

## Verification (this PR)

- `py_compile` clean on `rendering.py`, `prompt_spec.py`, `prime_runner.py`.
- `VARIANT_REGISTRY` builds and now contains `NETPLAY` + `GLYPHBOX` alongside all
  16 prior variants; existing variants unchanged.
- `resolve_spec("NETPLAY").turn_template(...)` raises `NotImplementedError` (the
  seam is routable but fails loudly); existing variants (e.g. `B1`) resolve fine.
- `nethack` env module imports cleanly with the engine on `PYTHONPATH`.
- `run_matrix` + `aggregate.table_to_markdown` produce a table with a `NETPLAY`
  row from synthetic samples (no model calls) — pipeline is intact.

Import-check command used:

```bash
export NLE_LIB_PATH=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness/third_party/NetHack/src/build/libnethack.so
export PYTHONPATH="$PWD/environments/nethack:/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness"
/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness/.venv/bin/python -c \
  "from nethack_harness.prompt.prompt_spec import VARIANT_REGISTRY; print(sorted(VARIANT_REGISTRY))"
```
