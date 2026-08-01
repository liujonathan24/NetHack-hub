# RUNBOOK — running experiments on this box

Reconstructed 2026-08-01. The handoff referred to a `RUNBOOK.md`, but no file by
that name exists in any commit of any of the three repos (verified across every
ref). This file replaces it, and documents the setup as actually built here.

Companion docs (both real, both on `exp/cli-harness-eval`):
`docs/HARNESS_DEFECTS.md` — what's broken/fixed/open.
`docs/experiments/exp2-cli-harness-results.md` — results and pinned state.

---

## 0. Read this first: the disk is ephemeral

This is a Prime Intellect **CPU sandbox**. One ext4 root disk (`/dev/vda3`), no
persistent volume mounted. `/workspace`, `/data` and `/ephemeral` are plain
directories on that same disk — none of them survive sandbox deletion. Default
sandbox lifetime is 240 minutes plus an idle timeout.

Everything below — repos, venv, the built engine — dies with the box. Push
branches or `prime sandbox download` anything you want to keep.

---

## 1. Layout

| path | what | pinned at |
|---|---|---|
| `/root/NetHack-engine` | layer 1: `nethack_core`, `nethack_interface`, fork submodule | `main` @ `9f09fd0` |
| `/root/NetHack-engine/third_party/NetHack` | the fork engine source (**this is what builds**) | `fefd557` ✅ |
| `/root/NetHack-hub` | layer 2: env, harness, eval tooling | `exp/cli-harness-eval` @ `63de299` |
| `/root/NetHack` | standalone clone of the fork — **decoy, see §5** | `main` @ `07b9446` ❌ |

Hub is on `exp/cli-harness-eval`, not `main`, deliberately: `main` is missing
4,431 lines of the §1–2 harness repairs (`corpse_age.py`, `sparse_map.py`,
`objective_hint.py`, `engine_provenance.py`, `progress.py`,
`tools/pycompat/sitecustomize.py`) plus both handoff docs. Running on `main`
reproduces the *pre-repair* numbers.

## 2. Environment

```bash
export PATH="$HOME/.local/bin:$PATH"
source /root/NetHack-hub/.venv-cli-eval/bin/activate     # python 3.12.13
cd /root/NetHack-hub
export ENG=/root/NetHack-engine                          # NOT the /scratch default
export PYTHONPATH="$PWD/tools/pycompat:$ENG:$PWD:$PWD/environments/nethack"
```

`.venv-cli-eval` matches the pinned state: Python 3.12, `verifiers==0.2.1` stock.

Installed editable: `nethack-core`, `nethack-interface` (from
`/root/NetHack-engine`), `nethack==0.0.69`, `nethack-prime-agent==0.1.0`.
The env's git-URL engine deps were bypassed with `--no-deps` so it uses the
locally built `.so`, not a re-fetched copy.

## 3. Building the engine

Already built. To rebuild after touching fork source:

```bash
JOBS=16 bash /root/NetHack-engine/nethack_core/build_engine.sh
```

Toolchain installed here: `cmake bison flex build-essential libncurses-dev
libbz2-dev zlib1g-dev`. The stock script fails twice on a clean box — `cmake`
missing, then `bzlib.h` missing. It also needs Python dev headers; it was
configured against the 3.12 venv interpreter, not system 3.10:

```bash
cmake -S "$ENG/third_party/NetHack/src" -B "$ENG/third_party/NetHack/src/build" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE="$(which python)"
```

`nethack_core/_engine.py` resolves the `.so` to
`third_party/NetHack/src/build/libnethack.so`. There is also a **stale committed
3.5 MB `libnethack.so`** at `third_party/NetHack/lib/` — it is never loaded, but
don't be fooled by it. `NLE_LIB_PATH` overrides everything if set.

## 4. Preflight — run this before every cell

```bash
python tools/cli_harness_eval/engine_provenance.py --check
```

Current output (passing):

```
[engine] head=9f09fd0 submodule=fefd557 pinned=fefd557 \
  so=.../src/build/libnethack.so built=2026-08-01T02:13:53 \
  newest_src=tests/verify_determinism.c@2026-08-01T02:12:25 flags=ok
```

Exit 4 means the `.so` predates its source, the submodule is dirty, or it sits
off the pinned commit. `ALLOW_STALE_ENGINE=1` bypasses loudly — a run launched
that way is not reproducible and must say so in its notes.

## 5. `/root/NetHack` is a decoy

It is a standalone clone of the fork at `main` (`07b9446`), **not** the pinned
`fefd557`. The build reads the submodule under `NetHack-engine`, so edits made
in `/root/NetHack` affect nothing. This is the same class of trap that cost
~30 mysterious test failures per exp2 §3.

It also carries a **committed dangling symlink**:
`nethackdir -> /scratch/gpfs/ZHUANGL/jl0796/.uv_cache/.../nle/nethackdir`.
Tracked in git (mode 120000), points at a Princeton cluster path, resolves
nowhere here.

## 6. Running a cell

Launchers default `ENG=/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness`, which does
not exist here — always set `ENG`. Arm configs also hard-code `trace_dir` and
`path_prepend` to cluster paths; `launch_cell.sh` overrides `trace_dir` for you,
but `path_prepend` you must override yourself.

```bash
# control arm (v0 in-process) — the pinned best config, HARNESS_DEFECTS §5
MODEL="google/gemini-3-flash-preview" \
EXTRA_ARGS='{"variant":"BBOX","skill_set":"netplay_true,reveal,rollback",
             "belief_state_interval":25,"max_turns":400,
             "explicit_seeds":[0],"n_examples":1}' \
  tools/encoding_eval/launch_encoding_cell.sh BBOX "$PWD/outputs/run1" 400 1

# claude_code arm
MODEL="z-ai/glm-5.2" \
  tools/cli_harness_eval/launch_cell.sh claude_code "$PWD/outputs/run1_cc" 400 5

# prime_agent arm — BLOCKED, see §8
MODEL="z-ai/glm-5.2" \
  tools/cli_harness_eval/launch_cell.sh prime_agent "$PWD/outputs/run1_pa" 400 5 \
  --harness.path-prepend "/usr/bin:/root/.local/bin"
```

Use **absolute** paths in launcher env vars. A relative `trace_dir` resolves
inside a rollout's ephemeral workdir and is discarded at teardown — this class of
bug appeared four separate times (exp2 §4 trap 4).

### Mandatory stall watchdog

Two unbounded hangs are still open and **neither reaches `env.step`**, so the
engine's own `no_progress_timeout` can never fire (`HARNESS_DEFECTS` §3.1). Kill
any rollout whose newest `turns/<seed>_*.ndjson` has not been written in ~300 s.
Re-arm it per batch — it exits when the queue drains.

### Before re-running a seed

Move `turns/<seed>_*.ndjson` aside. `turns/` is shared per cell and grading
groups by seed prefix, so a killed run's partial NDJSON merges with the relaunch.
Never glob `turns/*.ndjson` to aggregate — join on the PID in the filename, or
read `traces.jsonl`.

## 7. Test status on this box

| suite | result |
|---|---|
| `NetHack-engine/tests` + `environments/nethack/tests` | **363 passed**, 4 failed |
| `NetHack-hub/tests` | **98 passed**, 4 failed, 2 skipped |

**461 passed, 8 failed.** All 8 diagnosed, none caused by this setup:

1. **4 × `tests/test_cli_aggregate.py`** — needs
   `acceptance/task{10,13}_*.traces.jsonl`. Those files were **never committed to
   any ref**; only the `.turns.ndjson` and `.config.toml` siblings exist. Likely
   caught by the `traces.jsonl` gitignore rule from `dc78f47`. Cannot pass from a
   clean clone — the fixtures need re-cutting or the tests need repointing.
2. **3 × `test_nle_language_variant.py`** — `nle_language.py:34` hard-codes
   `_DEFAULT_PYTHON = "/home/jl0796/.claude/jobs/6389f3db/tmp/nlw-venv/bin/python"`.
   `nle-language-wrapper` fails to build here. Affects only the `NLE_LANG`
   variant, not the pinned configs.
3. **1 × `test_prompt_minimal.py`** — genuine drift, not environmental:
   `SYSTEM_PROMPT` is **3301 chars against a 3000 budget**. Failing at the pinned
   commit. Either the prompt grew past its cap or the cap needs raising —
   a decision, not a fix.

## 8. What can and cannot run

All three arms are ready. Each needs only `prime login` (§ below).

| arm | binary | status |
|---|---|---|
| `control` (v0) | — (in-process) | ready |
| `claude_code` | `claude` 2.1.220 | ready |
| `prime_agent` | `prime-agent` 0.3.3 @ `/usr/bin` | ready — matches the `version = "0.3.3"` pin |

`prime-agent` (the agent scaffold, arm 3) is a **different thing** from `prime`
(the Prime Intellect CLI). Both are installed. The scaffold is **not** on npm —
`npm install -g prime-agent` 404s. It comes from Prime's own installer:

```bash
curl -fsSL https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh | sh
```

That pulls `releases/v0.3.3/prime-agent-0.3.3.tgz`, verifies its checksum, and
installs system-wide to `/usr/bin/prime-agent`. Its harness deliberately does not
self-install it (unlike `claude_code`, which curls a pinned release), so this
step is manual on every fresh box.

`PrimeAgentHarness.setup()` passes here: binary resolves, version pin verified,
`sh -c 'command -v prime-agent && prime-agent --version'` returns `0.3.3`.

**`path_prepend` was retargeted for this box** in both prime_agent configs —
`/usr/bin:/root/.local/bin` (node 22 and prime-agent are system-wide via the
installer, not nvm; `uv` is in `~/.local/bin`, and prime-agent needs it to build
its IPython kernel venv). The cluster value is preserved in a comment beside it.
This is machine-specific: retarget per box, don't carry it across.

`prime_agent`'s harness deliberately does not self-install its CLI (unlike
`claude_code`, which curls a pinned release). `npm install -g prime-agent`
returns **404 — not in the registry**; `@primeintellect/prime-agent` and
neighbours 404 too. The arm needs the real source: private registry, GitHub
release, or a tarball. Config pins `version = "0.3.3"`.

### Inference: Prime Inference only

All inference goes through Prime, not per-vendor keys.
`configs/endpoints.toml` already routes it — nothing to change:

| registry id | models | notes |
|---|---|---|
| `pinference` | `Qwen/*` | |
| `pinference-glm` | `z-ai/glm-*` (incl. `glm-5.2`) | |
| `prime-team` | `google/gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.5-flash` | carries `X-Prime-Team-ID` |

All three point at `https://api.pinference.ai/api/v1` and read **`PI_API_KEY`**.
The CLI-arm configs read **`PRIME_API_KEY`** for the same key — export both.

Prime CLI 0.6.21 is installed (`uv tool install prime`). Log in, then export:

```bash
prime login --headless --plain          # interactive — paste the key it asks for
prime whoami                            # confirm
key=$(python -c "import json;print(json.load(open('$HOME/.prime/config.json'))['api_key'])")
export PI_API_KEY="$key" PRIME_API_KEY="$key"
```

**Billing:** Prime bills the personal balance ($0) unless `X-Prime-Team-ID`
names a funded team, otherwise every call returns `insufficient_funds`. The
`prime-team` block hard-codes team `cmotasmp5005ppyp07dcoh50u`
(`team_name = "jonathanliu"`). Check with `prime teams` / `prime wallet`; switch
with `prime switch`. Do **not** pass `-p prime` on the CLI — it overrides the
registry entry and drops the header.

### bwrap sandbox: off by choice

Both `configs/prime_agent.toml` and `configs/prime_agent_b80.toml` now read
`sandbox = false`.

This is **not** the cluster blocker from exp2 §2. That blocker was
`max_user_namespaces = 0`; this box reads `256381`, `bubblewrap 0.6.1` is
installed, and `bwrap --unshare-user --dev-bind / / true` succeeds. We are
choosing not to wrap: there is no `/scratch` here to hide, and the box is itself
a disposable single-tenant sandbox.

What that re-exposes, per the config's own note: Prime Agent's only builtin is
`ipython`, a full interpreter `disabled_tools` cannot constrain. A prior
unconfined run globbed `/scratch` and printed a live MCP bearer token into its
own trace. Keep credentials out of this box's environment beyond the one
inference key, and treat traces as sensitive.

Keep the two configs' values identical or the b80 comparison drifts.

## 9. Verified working

Live game, seed 0, `Val-hum-neu-fem`, BBOX, at the pinned config:

```
=== MAP (hidden; call reveal(x1,y1,x2,y2) to view a region) ===
=== STATUS ===
HP: 16/16  AC: 6  Dlvl: 1  Turn: 1  XP: 1  $: 0  Pos: (3,6)
Character: valkyrie (human, neutral)
=== ADJACENT === N=$(gold) ... SE=d(dog/canine)[PET — don't attack] ...
=== VISIBLE FEATURES === weapon at (15,9); gold at (3,5)
=== VISIBLE MONSTERS === little dog at (4,7), 1 step SE [PET - don't attack]
```

BBOX correctly withholds the grid, the pet guard is live, and `VISIBLE FEATURES`
is uncapped — i.e. the §1–2 repairs are in the build.
