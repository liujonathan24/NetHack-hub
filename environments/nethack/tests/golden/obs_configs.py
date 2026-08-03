"""The golden-observation matrix: one place that defines WHICH observations are
pinned and HOW they are produced.

Imported by both the re-bless script (`record_obs_snapshots.py`) and the check
(`../test_golden_obs.py`) so the two can never disagree about the config.

WHY THIS EXISTS
---------------
`docs/HARNESS_DEFECTS.md` §4.1: "Nothing pins the observation format. Add a
golden-file test so the next fix fails loudly instead of silently rebasing every
result."  The same `BBOX` config scored 3.58 -> 2.45 -> 3.71 across three
harness-wide fixes, two of which (statue naming, feature un-truncation) were
literally observation *content*.  Nobody noticed the numbers had stopped being
comparable, because nothing recorded what the agent had been shown.

These snapshots are that record.  They are deliberately NOT a gate -- see
`../test_golden_obs.py` for the warn-vs-fail line.

WHAT IS PINNED
--------------
The turn-1 observation (`NetHackEnv._render_obs_text` on a freshly set-up state)
for a small matrix chosen to span the two axes that actually restructure the
text:

  * map delivery      -- fog of war vs `tune={"reveal_map": 1.0}`.  The revealed
                         map is what makes `HINT` and the long-form
                         `VISIBLE FEATURES` / `VISIBLE MONSTERS` lines appear at
                         all, so the fog cells alone would pin none of them.
  * variant           -- `B0` renders the uncompressed ASCII grid; `BBOX` hides
                         it behind `reveal(...)`, which is the case where the
                         non-map sections are the agent's *only* channel.
  * compaction        -- `compact_obs=True` is a separate turn_template branch.

Seed, character and skill set are fixed so the text is deterministic; see the
determinism note in `record_obs_snapshots.py`.

Only turn 1 is captured.  A mid-rollout observation would cover journal growth
and skill feedback too, but it needs a scripted action sequence to stay
deterministic and would churn on every skill-message tweak -- which is exactly
the kind of noise that would train people to ignore this check.
"""
from __future__ import annotations

import os

# Fixed factors, shared by every cell in the matrix.
SEED = 0
CHARACTER = "Val-hum-neu-fem"
SKILL_SET = "netplay_true,reveal,rollback"
MAX_TURNS = 50

REVEAL_TUNE = {"reveal_map": 1.0}

# name -> kwargs overlaid on the fixed factors above.
# Keep this list SHORT.  Every entry costs an engine boot on every test run, and
# a matrix nobody reads is a matrix nobody re-blesses.
CONFIGS = {
    "b0_fog":         dict(variant="B0",   tune=None,        compact_obs=False),
    "b0_reveal":      dict(variant="B0",   tune=REVEAL_TUNE, compact_obs=False),
    "bbox_fog":       dict(variant="BBOX", tune=None,        compact_obs=False),
    "bbox_reveal":    dict(variant="BBOX", tune=REVEAL_TUNE, compact_obs=False),
    "b0_fog_compact": dict(variant="B0",   tune=None,        compact_obs=True),
}

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs")

# One command to re-bless.  Quoted verbatim in test failures and in the header
# of every snapshot file, so whoever hits a diff does not have to go looking.
REBLESS_CMD = (
    "PYTHONPATH=$PWD/tools/pycompat:$ENG:$PWD:$PWD/environments/nethack "
    ".venv-cli-eval/bin/python environments/nethack/tests/golden/record_obs_snapshots.py"
)


def snapshot_path(name: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{name}.txt")


async def render_observation(name: str) -> str:
    """Build the turn-1 observation for one matrix cell.

    Deliberately goes through the public `load_environment` entry point and the
    real `_render_obs_text`, not a hand-rolled template call -- the thing under
    observation is what a rollout actually shows the model.
    """
    import nethack  # imported lazily: needs PYTHONPATH set up by the caller

    # NOTE (was a workaround, now isn't). This function used to save and restore
    # the process-global `rendering._PUBLISHED_TOOLS`, because
    # `load_environment` -> `render_system_prompt(published_tools=...)` wrote it
    # and nothing put it back — so recording a snapshot here truncated the exit
    # hint that `test_hint_actionability` asserts on, in a DIFFERENT test file.
    # The global is gone: the published-tool set now travels in per-rollout state
    # (`rendering.PUBLISHED_TOOLS_STATE_KEY`, written by `setup_state`), so this
    # file has no process-wide side effects to contain.
    cfg = CONFIGS[name]
    env = nethack.load_environment(
        variant=cfg["variant"],
        skill_set=SKILL_SET,
        max_turns=MAX_TURNS,
        explicit_seeds=[SEED],
        n_examples=1,
        character=CHARACTER,
        compact_obs=cfg["compact_obs"],
        tune=cfg["tune"],
    )
    ex = env.dataset[0]
    state = {
        "task": {"seed": SEED},
        "info": ex["info"],
        "prompt": ex["prompt"],
        "responses": [],
        "turn": 0,
        "id": f"golden_obs_{name}",
        "model": "golden_obs",
    }
    state = await env.setup_state(state)
    return env._render_obs_text(state)
