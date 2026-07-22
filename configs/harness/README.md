# Harness overlay versions (`NETHACK_HARNESS`)

Experiment 2 needs to run the **same model** under **different harness versions**
to attribute progression to the harness, not the model. The seam already exists:

`environments/nethack/harness_overlay.py::apply_overlay` reads the env var
`NETHACK_HARNESS=<name>` inside `load_environment` and overlays four well-defined
harness surfaces (no code edit required per version):

1. `system_prompt` — replace / append / patch the module `SYSTEM_PROMPT`.
2. `per_step_prompt.template` — select an existing `_VARIANT_FORMATTERS` entry.
3. `tools.enabled` / `tools.disabled` — mask the skill registry (the action surface).
4. `rewards` — rebind reward weights (`scout`, `descent`, `success`, `ascension`).

`baseline` (or an unset `NETHACK_HARNESS`) is the shipped default: no overlay.
`fixes` is the candidate architecture we are validating.

## The one wiring gap

`apply_overlay` loads a version by calling
`tools.launchpad.core.harness.load_harness(name)` — and the `tools/launchpad`
package is **not vendored into this repo yet**. Until it lands (or a small TOML
loader replaces it), an unknown `NETHACK_HARNESS` name is a *logged no-op*: the
default in-source harness is used. That means today a `baseline` vs `fixes`
sweep differs only in label, not behavior.

These TOMLs therefore document the **intended overlay schema** (the fields
`harness_overlay.py` consumes) and are ready to drive real ablations the moment
the loader is present. See `docs/experiments/exp2_harness_modifications.md`
→ "What remains".
