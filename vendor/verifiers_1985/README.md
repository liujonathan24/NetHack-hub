# Vendored patch: PrimeIntellect-ai/verifiers PR #1985 (external harness loader)

PR: https://github.com/PrimeIntellect-ai/verifiers/pull/1985 ("Support external harness"),
branch `feat/generic-external-v1-plugins` off `main`. Status at vendor time: **OPEN**,
mergeable. Fetched with:

```bash
gh pr diff 1985 --repo PrimeIntellect-ai/verifiers > vendor/verifiers_1985/loaders.patch
```

`loaders.patch` touches 12 files total (docs, skills reference, 3 new/changed test fixtures,
`verifiers/v1/loaders.py`, `verifiers/v1/runtimes/{base,modal,prime}.py`,
`verifiers/v1/utils/compile.py`). Verified with `grep -c "^diff --git" loaders.patch` → 12.

## What was actually applied, and why not all of it

The patch was created against a commit of `main` that has drifted from the released
**verifiers 0.2.1** wheel we installed (`.venv-cli-eval`, see below). Concretely:

- `verifiers/v1/loaders.py` in 0.2.1 does **not** have the `import contextvars` line, the
  `if kind == "harness" and plugin_id == "default": ... "renamed to bash"` special case, or a
  `builtin_harness_ids()` helper that the patch's hunks assume exist. All three post-date the
  0.2.1 release. As a result **both hunks in `loaders.py` fail to apply mechanically** (`git
  apply` and GNU `patch` both reject them: `Hunk #1 FAILED at 2`, `Hunk #2 FAILED at 78`).
- `verifiers/v1/utils/compile.py` **does not exist at all** in 0.2.1 (`verifiers/v1/utils/`
  has no `compile.py`) — that file/feature was added after 0.2.1. This hunk cannot apply by
  any means against this release; skipped as not applicable.
- `verifiers/v1/runtimes/{base,modal,prime}.py` apply cleanly under GNU `patch -p1` (one with
  fuzz 1, one with line-offset only, one clean) — but they implement an **unrelated** feature
  (`stop_confirmed()` fail-closed sandbox teardown / remote timeout capping), not harness
  resolution. Out of scope for this task's interface contract
  (`verifiers.v1.loaders.harness_class`), so **deliberately not applied**, to keep the vendored
  diff minimal and reviewable. Left unpatched in `.venv-cli-eval`.
- docs/skills/tests hunks: not applied (brief's Step 5 excludes `tests/*` and `docs/*`; the
  skills reference hunk is likewise doc-only).

**Only `verifiers/v1/loaders.py` was reconciled and applied**, by hand, against the actual
0.2.1 source (reproducing the same net semantic change the patch makes to `_import_plugin`:
try both the namespaced candidate `f"{group}.{module}"` and the bare `module` name in order,
and if neither imports, raise `ModuleNotFoundError` naming *both* candidates instead of
whichever one `importlib.util.find_spec` happened to pick). The `builtin_harness_ids()` hint
line from the PR's pre-image was **not** carried over — that function doesn't exist in 0.2.1
and including it would raise `NameError` on every failed lookup.

## The command that worked

`git apply` (even with `--directory=`) **silently no-ops** here: when run from inside a path
that is itself part of a git working tree (`.venv-cli-eval/lib/python3.12/site-packages/...`
lives inside the `cli-harness-eval` worktree), `git apply` resolves each patch's target path
**relative to the enclosing repo's top-level, not the current working directory** — even with
`--directory=<prefix>` set. Since `site-packages/verifiers/v1/loaders.py` doesn't exist at the
worktree root, git silently prints `Skipped patch '...'` for every file in the patch and exits
0, which looks like success (`git apply ... && echo "APPLIED OK"` prints "APPLIED OK") while
having changed nothing. Confirmed by reproducing the exact same silent-skip behavior in a
disposable throwaway repo at matching nesting depth, and by grepping the "patched" file
afterwards and finding it byte-identical to the original.

GNU `patch` (not `git apply`) resolves purely relative to `cwd`, unaffected by any enclosing
repo, and is what was actually used to determine hunk applicability:

```bash
SITE=/scratch/gpfs/ZHUANGL/jl0796/NetHack-hub/.worktrees/cli-harness-eval/.venv-cli-eval/lib/python3.12/site-packages
cd "$SITE" && patch -p1 --dry-run < /scratch/gpfs/ZHUANGL/jl0796/NetHack-hub/.worktrees/cli-harness-eval/vendor/verifiers_1985/loaders.patch
```

Output showed both `loaders.py` hunks FAILED (context drift, see above), `runtimes/base.py`
succeeded with fuzz 1, `runtimes/modal.py` succeeded cleanly, `runtimes/prime.py` succeeded
with offsets, and `docs/*`, `skills/*`, `tests/*`, `utils/compile.py` had no file to patch
against (either out of scope or nonexistent in this release).

Because the `loaders.py` hunks did not apply mechanically to 0.2.1, the fix was applied as a
**manual, hand-reconciled edit** to
`.venv-cli-eval/lib/python3.12/site-packages/verifiers/v1/loaders.py`'s `_import_plugin`
function (not a scripted patch command) — see the diff embodied in `loaders.patch` for the
target's *intent*; the actual bytes written to the installed package differ from the patch's
post-image because the pre-image differs. This is exactly the "patch did not apply cleanly —
reconcile against the installed version, then re-run" branch the task brief anticipated.

**Verifiers version this was applied to:** `verifiers==0.2.1`, installed via
`uv pip install --python .venv-cli-eval/bin/python verifiers==0.2.1 ...` into the dedicated
`.venv-cli-eval` venv (see repo root — this venv is disposable and gitignored; `rm -rf` and
reinstalling `verifiers==0.2.1` plus reapplying this same hand-edit reproduces it).

## Reproducing the fix in a fresh venv

1. `uv venv .venv-cli-eval --python 3.12 && uv pip install --python .venv-cli-eval/bin/python verifiers==0.2.1 numpy gymnasium datasets pillow pytest pytest-asyncio`
2. Edit `.venv-cli-eval/lib/python3.12/site-packages/verifiers/v1/loaders.py`'s
   `_import_plugin` function to try both `f"{group}.{module}"` and `module` before raising,
   reporting both in the error message (see the reconciled function committed in this repo's
   history / this README's description above — no automated patch application was possible).
3. Confirm: `pytest tests/test_vendored_loader.py -v` → 1 passed.

## Verification

- RED: `pytest tests/test_vendored_loader.py -v` against unpatched 0.2.1 →
  `AssertionError: assert 'verifiers.v1.harnesses.' in "harness ... (tried to import
  'definitely_not_installed_harness'). ..."` (message names only one candidate).
- GREEN: same command after the hand-reconciled edit → `1 passed`.
- Regression check: `pytest environments/nethack/tests/ tests/ -q` before and after the
  `loaders.py` edit both show the same `10 failed, 15{5,6} passed` split (the loaders.py change
  does not touch anything exercised by the pre-existing 10 v1-rollout failures — see
  `task-8-report.md` for what those are and why they're unrelated to this vendoring).
