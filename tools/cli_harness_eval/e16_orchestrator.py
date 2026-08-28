#!/usr/bin/env python3
"""E16 Go-Explore orchestrator: the loop that turns an archive into attempts.

    read the archive -> select a checkpoint -> launch ONE player from it ->
    ingest what it did -> repeat, until budget / stall / milestone.

THE ORCHESTRATOR IS AN LM, AND THAT IS A CHANGE
----------------------------------------------
The v3 design's orchestrator was scripted and genuinely free. It is now a
PERSISTENT Prime Agent conversation (`e16_session.PrimeAgentSession`) that
opens with a planning discussion over the wiki and the seed's known hazards,
then each round reads the updated ledger and the finishing player's summary,
CHOOSES the next checkpoint, and writes that player a DIRECTIVE. Two
consequences are load-bearing:

* The orchestrator is no longer free. Its spend is a SECOND BUDGET LINE
  (:class:`Budget`), never folded into the players' total.
* The scripted softmax selector below is kept as `selector="scripted"` -- the
  ABLATION. Without something to be measured against, "the LLM orchestrator
  steered the search" is unfalsifiable, and the null hypothesis ("it narrated
  while random restarts did the work") produces the same depth curve.
  :func:`classify_directive_compliance` exists for the same reason.

WHY THE RECORD-KEEPING IS THE POINT
-----------------------------------
A Go-Explore run is trivially easy to over-claim. "Reached Dlvl 15" means
something entirely different when it took one life than when it took forty
resumes of a curated archive, and the difference is invisible in the number.
E15 also shipped an arm whose numbers were corrupted because rollouts that
ended by wall-clock, harness error or empty completion were counted alongside
deaths. So this module treats six things as OUTPUT, not documentation:

1. NO MODEL-AUTHORED METRICS. Every numeric field in the ledger comes from
   ``nethack_harness.checkpoints``, computed from engine blstats at save time.
   Model-written text (a checkpoint's name, its note, a lesson) is carried and
   displayed but never read by the selector, the frontier, or any summary
   number. :func:`ledger_rows` builds rows only from the harness-computed keys.
2. RESTORE FIDELITY. Every restore re-reads blstats and compares them with the
   checkpoint's meta (``checkpoints.restore_fidelity``); a mismatch raises. The
   records land in ``restore_fidelity.jsonl``.
3. COMPARABILITY. Every frontier advance records the attempts, calls, spend and
   wall-clock it consumed, and ``summary.json`` carries ``attempts_to_reach``
   next to every best-state claim, plus an explicit
   ``comparable_to_single_life_balrog: false``.
4. LUCK. ``luck.json`` is the outcome distribution per checkpoint: how many
   attempts started there and how they ended. Monte-Carlo re-rolling is then
   visible rather than hidden inside a best-of.
5. CENSORING. ``censored`` is a distinct outcome from ``died``, with a reason
   (wall clock, harness error, empty completion, budget stop, integrity error).
   Nothing collapses them.
6. PROVENANCE. ``provenance.json``: tier name + registry hash, the three git
   commits, the wiki file hashes, the RNG seed, the selector weights, the
   orchestrator's session id(s), and start time.

An LLM orchestrator makes point 1 MORE important, not less. It may narrate
progress however it likes; the only things that cross from its text into the
run are a checkpoint ID (validated against the archive) and a DIRECTIVE
(carried as text to the player, and measured for compliance). No number it
writes is ever recorded as a number.

THE PLAYER IS INJECTED
----------------------
:class:`Orchestrator` never spawns a process itself; it calls a
``launcher(ctx) -> PlayerResult``. :class:`SubprocessPlayer` is the real one
(``launch_cell.sh`` with the ``e16_gewiki`` tier). Tests pass a stub that plays
scripted engine steps, which is what makes an end-to-end loop test possible
with no inference at all.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "environments" / "nethack"))

from nethack_harness.checkpoints import (  # noqa: E402
    LESSONS_MD,
    META_JSON,
    PREFIX_JSONL,
    atomic_write,
    checkpoint_list,
    checkpoint_meta,
)

# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #

#: The character's game ended.
OUTCOME_DIED = "died"
OUTCOME_ASCENDED = "ascended"
#: The character was still alive; something outside the game stopped the
#: attempt. NEVER merged with `died` -- see the module docstring, point 5.
OUTCOME_CENSORED = "censored"

#: Why a censored attempt was censored. Every one of these is "the
#: infrastructure stopped a live game", and each is a different bias: a
#: wall-clock stop truncates long games, an empty completion truncates
#: whichever game the provider hiccupped on, a budget stop truncates the last
#: attempt of a run. Recording the reason is what lets an analysis decide
#: whether a censored attempt can be pooled with anything.
CENSOR_WALL_CLOCK = "wall_clock"
CENSOR_HARNESS_ERROR = "harness_error"
CENSOR_EMPTY_COMPLETION = "empty_completion"
CENSOR_BUDGET_STOP = "budget_stop"
CENSOR_INTEGRITY = "integrity_error"
CENSOR_UNKNOWN = "unknown"
#: The ORCHESTRATOR process died while this attempt was in flight -- signal,
#: crash, or the box going away -- and the attempt was finalized afterwards
#: from what it had already written to disk. Distinct from every reason above
#: because it is the only one whose numbers are a LOWER BOUND recovered from
#: partial evidence rather than a rollout's own report: `spend_usd` on such a
#: row is what the trace file proved, and `spend_known: false` says when even
#: that was unavailable. Pooling these with completed attempts understates
#: spend and depth simultaneously.
CENSOR_INTERRUPTED = "interrupted"
#: The ENV tool server (`python -m nethack_v1`) died of a fatal native signal
#: mid-attempt -- SIGSEGV/SIGABRT out of the NetHack C extension, not an
#: external kill. Split out of `harness_error` because the two demand different
#: responses: a harness error is a thing the harness said, while this is the
#: game process ceasing to exist under a live rollout, which truncates the
#: attempt at a turn count that has nothing to do with the game. Evidence is
#: `<attempt>/env_crash.log`, written by `nethack_v1.arm_crash_capture()`; the
#: row carries `env_crash` with the signal faulthandler named.
CENSOR_ENV_CRASH = "env_crash"

CENSOR_REASONS = frozenset({
    CENSOR_WALL_CLOCK, CENSOR_HARNESS_ERROR, CENSOR_EMPTY_COMPLETION,
    CENSOR_BUDGET_STOP, CENSOR_INTEGRITY, CENSOR_UNKNOWN, CENSOR_INTERRUPTED,
    CENSOR_ENV_CRASH,
})

#: `stop_condition` strings the eval CLI writes for an infrastructure stop.
_STOP_TO_CENSOR = {
    "harness_timeout": CENSOR_WALL_CLOCK,
    "rollout_timeout": CENSOR_WALL_CLOCK,
    "timeout": CENSOR_WALL_CLOCK,
    "error": CENSOR_HARNESS_ERROR,
    "agent_completed": CENSOR_EMPTY_COMPLETION,
    "budget_stop": CENSOR_BUDGET_STOP,
    "max_turns_reached": CENSOR_BUDGET_STOP,
    "call_budget_exhausted": CENSOR_BUDGET_STOP,
}

# Stop conditions.
STOP_BUDGET = "budget_ceiling"
STOP_STALL = "no_new_frontier"
STOP_MILESTONE = "milestone"
STOP_MAX_ATTEMPTS = "max_attempts"
STOP_EMPTY_ARCHIVE = "empty_archive"
#: The orchestrator itself failed -- no usable opening plan, or no usable
#: directive after its bounded retries. NOT a normal stop: the run also raises.
STOP_ORCHESTRATOR_FAILED = "orchestrator_failed"

#: NetHack numbers its branches in dungeon.def order. Sokoban is 4.
SOKOBAN_DUNGEON_NUMBER = 4

# --------------------------------------------------------------------------- #
# experiment arms
# --------------------------------------------------------------------------- #

#: The method under test.
ARM_GO_EXPLORE = "go_explore"
#: THE NULL. See :class:`OrchestratorConfig.arm`.
ARM_MATCHED_RESTART = "matched_restart"
ARMS = (ARM_GO_EXPLORE, ARM_MATCHED_RESTART)

#: An attempt's role inside a matched pair (``--paired-control``).
ROLE_SOLO = "solo"
ROLE_TREATMENT = "treatment"
ROLE_CONTROL = "control"

#: Cross-experiment fallback $/hour spend rate, used by the in-flight spend
#: estimator only until this run has completed an attempt to calibrate against.
#: Set just above the p90 ($82.15/hr) measured over 349 real rollouts in this
#: tree's `outputs/`. Defined here because `OrchestratorConfig` defaults to it;
#: the reasoning, the measurements and the estimator live together further down
#: under "in-flight spend".
DEFAULT_SPEND_RATE_USD_PER_HOUR = 90.0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass
class OrchestratorConfig:
    """Everything the loop is allowed to depend on, in one auditable object."""

    run_dir: Path
    #: Source of the curated wiki pages; copied into ``run_dir/wiki`` at start.
    wiki_src: Path = Path("/root/nld/e15-wiki/configs/continual/wiki")
    tier: str = "e16_gewiki"
    #: THE EXPERIMENT ARM. Not the launcher's arm (that is `player_arm`).
    #:
    #: ``go_explore``      the method: archive, selection, directives, lessons.
    #: ``matched_restart`` THE NULL. N independent player attempts from ONE
    #:                     fixed start state, with no archive, no selection, no
    #:                     directive and no lessons -- the multi-restart
    #:                     baseline the method has to beat.
    #:
    #: WHY THIS ARM EXISTS AT ALL, since it is the one that can end the
    #: experiment: every other control in this design (`--selector scripted`,
    #: `--no-directive`, no-wiki) is an ablation WITHIN the method, and not one
    #: of them is a null. Each is beaten by "we got N tries instead of one".
    #: The headline number is a running maximum over attempts -- monotone
    #: non-decreasing by construction, unable to fall -- so comparing it to a
    #: single-life number compares max-of-N draws against one draw. Without a
    #: matched-N baseline there is no reading of the result under which the
    #: method could have failed, and a claim that cannot fail is not a finding.
    arm: str = ARM_GO_EXPLORE
    #: The LAUNCHER's arm name -- which scaffold `launch_cell.sh` runs. Distinct
    #: from `arm` above, and deliberately so: the experiment arm is a property
    #: of the design, the player arm is a property of the harness, and E15 lost
    #: an arm to exactly the kind of confusion that follows from one name.
    player_arm: str = "prime_agent"
    #: matched_restart only: the checkpoint every attempt restarts from. Empty
    #: means "the archive's seed checkpoint" (c1). The red team's C1 wants a
    #: FIXED, early state -- so it is named once and never re-selected.
    start_checkpoint_id: str = ""
    #: "llm" -- a persistent Prime Agent session chooses the next checkpoint
    #: and may pass a short directive to the player (the primary design).
    #: "scripted" -- the softmax selector below chooses (the ABLATION, kept so
    #: the LLM orchestrator has something to be measured against; deleting it
    #: would leave "did the LLM orchestrator help?" unanswerable).
    selector: str = "llm"
    #: The GAME seed. The design runs seed 1 and only seed 1, deliberately.
    game_seed: int = 1
    #: The SELECTOR's RNG seed -- separate from the game seed so the selection
    #: sequence is reproducible independently of the dungeon.
    rng_seed: int = 20260828

    # Selector weights (design Q7: deferred, starting values 1.0 / 0.5 / 0.3).
    w_balrog: float = 1.0
    w_novelty: float = 0.5
    w_attempts: float = 0.3
    temperature: float = 1.0

    # Budget. HARD: checked before every launch, never after.
    budget_ceiling_usd: float = 385.0
    #: Refuse a launch that cannot be afforded even in the best case. An
    #: uncapped player can cost anything, so this is a floor on remaining
    #: headroom, not a prediction.
    min_headroom_usd: float = 5.0
    #: Enforce the ceiling DURING an attempt, not only before launching one.
    #: Off, the ceiling is only a pre-launch gate and a single runaway attempt
    #: can blow through it unnoticed -- measured on a $9 smoke that reached an
    #: estimated $22. On, the progress monitor terminates the attempt cleanly
    #: as `censored:budget_stop` when its ESTIMATED spend would breach.
    enforce_inflight_budget: bool = True
    #: The $/hour rate used before this run has a completed attempt to
    #: calibrate against. See `DEFAULT_SPEND_RATE_USD_PER_HOUR`.
    inflight_spend_prior_usd_per_hour: float = DEFAULT_SPEND_RATE_USD_PER_HOUR
    #: Multiplier on the estimated rate for the ENFORCEMENT number only. See
    #: `estimate_inflight_spend`: the two errors are not symmetric.
    inflight_spend_safety_factor: float = 2.0
    #: Seconds between SIGTERM and SIGKILL when the guard stops an attempt.
    #: The eval CLI is given a chance to finalize `traces.jsonl` first -- a
    #: clean stop is what makes the spend of the stopped attempt knowable.
    inflight_stop_grace_s: float = 60.0

    # Stop conditions.
    max_attempts: int = 200
    stall_attempts: int = 8
    milestone_dlvl: int = 20
    milestone_dungeon: int = SOKOBAN_DUNGEON_NUMBER

    #: Distinct states that get a FULL ledger row (name, note, lesson). Not a
    #: cap on what the selector may choose: past this, the remaining states are
    #: rendered one compact line each and stay legal choices. See
    #: :func:`build_ledger` -- an archive of hundreds must stay readable
    #: without any state becoming unchoosable, which is the defect this whole
    #: rendering exists to undo.
    ledger_max_rows: int = 40
    #: Lessons carried into a resumed player's first observation.
    ledger_max_lessons: int = 8

    # ---- LLM orchestrator session ---------------------------------------- #
    #: Hard cap on the player prose that enters the ORCHESTRATOR's conversation
    #: each round. The orchestrator's context is the one thing in this design
    #: that grows monotonically with the run's length, and an uncapped player
    #: summary is the fastest way to blow it: an uncapped player session can
    #: emit thousands of words about one game. 1500 chars is roughly a
    #: paragraph and a half -- enough for "died to a wraith on D15 after
    #: melee'ing it", which is the whole informational content.
    orchestrator_summary_cap: int = 1500
    #: THE HARD CONTEXT BOUND for the orchestrator's conversation, in TOKENS.
    #: Enforced by prime-agent's OWN threshold compaction, configured into the
    #: run's private agent dir at session build time
    #: (`e16_session.configure_native_compaction`). There is no hand-off prompt
    #: of ours any more and no fresh-session reseed: the conversation stays one
    #: session and the CLI compacts it in place. What was actually configured
    #: -- including the failure cases, where the bound could NOT be enforced --
    #: is written into provenance.
    orchestrator_context_limit_tokens: int = 128 * 1024
    #: Filled in by `build_session`: the record of what compaction the CLI was
    #: actually configured with. Provenance reads it; nothing decides on it.
    orchestrator_compaction: Optional[dict] = None
    #: Set by `build_session` when the preferred TMPDIR was unusable (its
    #: daemon socket would have exceeded the kernel's sun_path limit). Empty
    #: when it was usable. Provenance carries it: a run that moved its socket
    #: somewhere else should say where.
    orchestrator_tmpdir_note: str = ""
    #: `--decide-only`: USD that must remain before another decision round is
    #: bought. Sized for a decision round (cents), not for a player launch --
    #: see `min_headroom_usd`, which is the launch gate and would refuse every
    #: round of a decide-only run under a small ceiling.
    decide_only_headroom_usd: float = 0.25
    #: Agent-state dir for the orchestrator session. MUST NOT be shared with a
    #: player sandbox: /root/.prime/agent holds daemon-workers/ and
    #: session-leases/, and a booting experiment scanning it has been measured
    #: reaping another experiment's live sessions.
    orchestrator_agent_dir: Optional[Path] = None
    orchestrator_model: str = ""
    #: Per-round wall-clock cap on one orchestrator turn.
    orchestrator_timeout_s: float = 900.0
    #: BOUNDED retries when a selection round produces no usable decision, and
    #: when the opening round produces no usable plan. After these, the run
    #: RAISES (`DirectiveExtractionFailed` / `OpeningPlanUnusable`) rather than
    #: continuing without its treatment -- see `OrchestratorRoundFailed`.
    #: Bounded rather than unbounded because a model that has answered wrongly
    #: twice in one session usually keeps doing it, and each retry is a paid
    #: round against the same ceiling.
    directive_retries: int = 2
    opening_retries: int = 2
    #: CONTROL MODE, RUN-WIDE. Launch every player with no directive at all --
    #: same archive, same selection, no instructions.
    #:
    #: ITS MEASURED LIMIT, which is why `paired_control` exists below: run-wide
    #: it answers "does the directive channel matter on average", and the sims
    #: showed that is not enough. A `descend_fast` directive inverted the
    #: player's first decision causally (walk to the down staircase vs melee the
    #: jackal; 0 attacks vs 1, and 15 explores vs 1 at 25 calls). A `no_descend`
    #: directive was INDISTINGUISHABLE from this control -- because the control
    #: did not descend either. A prohibition against something the player was
    #: never going to do measures nothing, and pooling both kinds into one
    #: run-wide control hides that: the average says "directives do something"
    #: while half the directive kinds are untested.
    no_directive: bool = False
    #: CONTROL MODE, PER DIRECTIVE KIND. Every directive the orchestrator issues
    #: is run TWICE from the SAME checkpoint: once with it (`treatment`) and
    #: once without (`control`). Each directive is then compared against its own
    #: control rather than against a run-level average, which is the only way
    #: "this KIND of directive changes behaviour" is answerable.
    #:
    #: It doubles the attempt count for a given number of directives, and that
    #: is the price of the comparison rather than an accident: both attempts
    #: draw on the same budget line and emit the same attempt records.
    paired_control: bool = False
    #: RESEED ON RESTORE. Default OFF, and that is a deliberate choice rather
    #: than an oversight.
    #:
    #: WHAT IS TRUE TODAY (measured by the red team's probe, not assumed):
    #: `checkpoint_restore` resets the engine to the game's ORIGINAL seed and
    #: replays no history, so two restores of one checkpoint continue
    #: byte-identically. Restoring does NOT reroll the dice.
    #:
    #: WHY THAT IS THE DEFAULT: with the dice fixed, the difference between two
    #: attempts from one checkpoint comes from the model's choices and from the
    #: directive it was given -- which is exactly what makes a directive's
    #: effect attributable. Turn reseeding on and every A/B between a directive
    #: and its control acquires a second, uncontrolled source of variance, and
    #: the causal reading of the sims' `descend_fast` inversion would not have
    #: been available.
    #:
    #: WHY THE FLAG EXISTS ANYWAY: determinism has a real cost the design must
    #: state rather than enjoy. The dungeon BELOW a checkpoint is regenerated
    #: from a stream restarted at position 0, so the deep levels an E16 run
    #: reaches are NOT the seed-1 deep levels E15 mapped, and per-checkpoint
    #: outcome spread reflects policy variation only -- it understates the luck
    #: a real run is exposed to. A reader who wants the Monte-Carlo reading is
    #: one CLI argument away from it, and provenance records which was used.
    reseed_on_restore: bool = False
    #: Whether `--resume` was PROVEN to carry history (the two-call probe). Set
    #: by the run, written to provenance; never assumed.
    session_resume_verified: Optional[bool] = None
    #: THE ORCHESTRATOR'S RECORDED CWD. Written into provenance at round 0 and
    #: asserted before every later round; a `--resume` from anywhere else takes
    #: prime-agent's "fork this session?" branch and hangs forever. Empty until
    #: `prepare()` fills it in.
    orchestrator_cwd: str = ""
    #: The orchestrator's PRIVATE TMPDIR. The unsandboxed orchestrator otherwise
    #: shares /tmp/prime-agent-0/daemon.sock with every prime-agent on the box:
    #: measured at 900s of hang against 3.5s with a private one. Never
    #: exported -- see e16_session for why an exported TMPDIR breaks players.
    orchestrator_tmpdir: Optional[Path] = None
    #: `discovery_only` (default) | `always` | `never`. See e16_session: json
    #: mode + `--resume` was measured hanging where text `--print` succeeded.
    orchestrator_json_mode: str = "discovery_only"

    @property
    def archive_dir(self) -> Path:
        return self.run_dir / "archive"

    @property
    def wiki_dir(self) -> Path:
        return self.run_dir / "wiki"

    #: Seconds between :class:`AttemptProgressMonitor` samples. The monitor is
    #: the ONLY live signal a running attempt emits -- the orchestrator is
    #: blocked in `subprocess.run` for the whole rollout, so without it the run
    #: is unobservable except by reading the archive.
    progress_interval_s: float = 15.0

    @property
    def attempts_path(self) -> Path:
        return self.run_dir / "attempts.jsonl"

    @property
    def journal_path(self) -> Path:
        """Append-only attempt LIFECYCLE log: one `open` and one `close` each.

        `attempts.jsonl` carries FINAL rows only, and every reader in this tree
        (`pairs`, `luck`, the aggregators) assumes that. The journal is where
        an attempt exists between its launch and its result, so a run that dies
        mid-attempt leaves a record saying which attempt was in flight, from
        which checkpoint, under which directive, at which pid -- which is
        exactly what `reconcile_run` needs to finalize it afterwards.
        """
        return self.run_dir / "attempts_journal.jsonl"

    @property
    def progress_path(self) -> Path:
        """`tail -f` this. One line per sample while an attempt plays."""
        return self.run_dir / "progress.jsonl"

    @property
    def recovery_path(self) -> Path:
        return self.run_dir / "recovery.json"

    @property
    def selection_path(self) -> Path:
        return self.run_dir / "selection.jsonl"

    @property
    def fidelity_path(self) -> Path:
        return self.run_dir / "restore_fidelity.jsonl"

    @property
    def provenance_path(self) -> Path:
        return self.run_dir / "provenance.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def luck_path(self) -> Path:
        return self.run_dir / "luck.json"

    @property
    def orchestrator_dir(self) -> Path:
        """The orchestrator session's workspace (its cwd, its transcript)."""
        return self.run_dir / "orchestrator"

    @property
    def orchestrator_log(self) -> Path:
        """Every orchestrator round: prompt, reply, spend, session id."""
        return self.run_dir / "orchestrator_rounds.jsonl"


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #

#: Ledger fields, and the ONLY fields any number in this module may come from.
#: Each is computed by nethack_harness.checkpoints from engine blstats at save
#: time. `name`/`note` are carried alongside as TEXT and are never read here.
NUMERIC_FIELDS = ("dlvl", "xl", "hp", "max_hp", "gameturn", "score",
                  "balrog", "balrog_min", "max_dlvl_reached", "max_xp_level",
                  "dungeon_number", "level_number", "visits", "attempts_from")


@dataclass
class Row:
    """One checkpoint as the selector sees it."""

    id: str
    path: Path
    dlvl: int = 0
    xl: int = 1
    hp: int = 0
    max_hp: int = 0
    gameturn: int = 0
    score: int = 0
    balrog: float = 0.0
    balrog_min: float = 0.0
    dungeon_number: int = 0
    level_number: int = 0
    max_dlvl_reached: int = 0
    max_xp_level: int = 1
    visits: int = 0
    attempts_from: int = 0
    created_at: float = 0.0
    created_by: str = ""
    parent: Optional[str] = None
    # TEXT. Displayed, never measured.
    name: str = ""
    note: str = ""

    @property
    def key(self) -> tuple:
        """The Pareto objective: (Dlvl, XL, score), per the design."""
        return (self.dlvl, self.xl, self.score)


def row_from_meta(path: Path, meta: dict) -> Row:
    """Build a :class:`Row` from a checkpoint's meta.

    STRICT ABOUT PROVENANCE, and this is integrity requirement 1 in code: every
    numeric attribute is read from a harness-computed key and coerced to a
    number. ``name``/``note`` are copied verbatim into text-only fields. There
    is no path by which a string a model wrote can become a number here --
    a note of ``"dlvl": 99`` is a note, and a name of ``"Dlvl 30"`` is a name.
    """
    def num(key, default=0):
        try:
            v = meta.get(key, default)
            return default if v is None else (float(v) if isinstance(default, float) else int(v))
        except (TypeError, ValueError):
            return default

    return Row(
        id=str(meta.get("id") or path.name),
        path=path,
        dlvl=num("dlvl"), xl=num("xl", 1), hp=num("hp"), max_hp=num("max_hp"),
        gameturn=num("gameturn"), score=num("score"),
        balrog=num("balrog", 0.0), balrog_min=num("balrog_min", 0.0),
        dungeon_number=num("dungeon_number"), level_number=num("level_number"),
        max_dlvl_reached=num("max_dlvl_reached"), max_xp_level=num("max_xp_level", 1),
        visits=num("visits"), attempts_from=num("attempts_from"),
        created_at=num("created_at", 0.0),
        created_by=str(meta.get("created_by") or ""),
        parent=(str(meta["parent"]) if meta.get("parent") is not None else None),
        name=str(meta.get("name") or ""),
        note=str(meta.get("note") or ""),
    )


def ledger_rows(archive) -> list:
    """Every complete checkpoint under ``archive``, as rows, oldest first."""
    return [row_from_meta(p, checkpoint_meta(p)) for p in checkpoint_list(archive)]


def creation_order(paths: list) -> list:
    """Checkpoint dirs in the order the engine wrote them, oldest first.

    `created_at` is harness-written at save time. The id is only a tiebreak,
    and a numeric one -- ``c10`` was written after ``c9``, and any ordering
    that sorts them as strings would build the lineage backwards.
    """
    def key(path):
        try:
            ts = float(checkpoint_meta(path).get("created_at") or 0.0)
        except Exception:
            ts = 0.0
        return (ts, _id_key_name(Path(path).name))
    return sorted(paths, key=key)


def parent_chain(paths: list, resume_id: Optional[str]) -> list:
    """``[(path, parent id), ...]`` -- one attempt's checkpoints as a PATH.

    THE DEFECT THIS FIXES. Every checkpoint an attempt wrote used to be
    stamped with the checkpoint that ATTEMPT resumed from, so an attempt that
    saved 23 states produced 23 siblings of one node: a star with 23 spokes,
    in which nothing records that c6 was written after c5 in the same life.
    The mechcheck archive is exactly that -- c2..c24 all pointing at c1 -- and
    it is unusable as a tree.

    A life is a CHAIN. The first checkpoint an attempt writes descends from the
    state it resumed; the second descends from the first; and so on. The chain
    is joined to the rest of the archive at the resume point, so a later
    attempt resuming the SAME state starts a second child there and the fork is
    visible as a fork -- which is the only way branching, the behaviour this
    experiment is trying to observe, shows up in the structure at all.
    """
    out = []
    prev = str(resume_id) if resume_id else None
    for path in creation_order(paths):
        out.append((Path(path), prev))
        try:
            prev = str(checkpoint_meta(path).get("id") or Path(path).name.lstrip("c"))
        except Exception:
            prev = Path(path).name.lstrip("c")
    return out


def pareto_frontier(rows: list) -> list:
    """Rows not dominated on (Dlvl, XL, score).

    ``a`` dominates ``b`` when it is >= on all three and > on at least one.
    Ties (identical keys) are all kept: two different states with the same
    coordinates are two different places to resume from, and collapsing them
    would quietly hide half the archive from the selector.
    """
    out = []
    for r in rows:
        dominated = any(
            all(x >= y for x, y in zip(o.key, r.key)) and o.key != r.key
            for o in rows
        )
        if not dominated:
            out.append(r)
    return out


#: NetHack numbers its branches in dungeon.def order, and blstats carries the
#: number (`checkpoints._status_snapshot`). Depth alone cannot express branch:
#: a Mines D7 and a Dungeons-of-Doom D7 are different places with different
#: monsters, different loot and different reasons to go there -- and on this
#: seed that distinction is the single most decision-relevant thing about a
#: state that (Dlvl, XL, score) cannot see.
DUNGEON_NAMES = {
    0: "Dungeons of Doom", 1: "Gehennom", 2: "Gnomish Mines", 3: "The Quest",
    4: "Sokoban", 5: "Fort Ludios", 6: "Vlad's Tower", 7: "Elemental Planes",
    8: "Astral Plane",
}
DUNGEON_SHORT = {0: "main", 1: "gehennom", 2: "MINES", 3: "quest", 4: "SOKO",
                 5: "ludios", 6: "vlad", 7: "planes", 8: "astral"}

#: Why a checkpoint exists. Derived from HARNESS-set fields only -- see
#: :func:`checkpoint_kind` for why a model's `save` can never wear one of the
#: automatic labels.
KIND_SEED = "seed"
KIND_MODEL_SAVE = "model-save"
KIND_LEVEL_ENTRY = "level-entry"
KIND_LEVEL_UP = "level-up"
KIND_CADENCE = "cadence"
KIND_AUTO = "auto"
KIND_UNKNOWN = "?"


def branch_name(dungeon_number: int) -> str:
    return DUNGEON_NAMES.get(int(dungeon_number or 0),
                             f"branch {int(dungeon_number or 0)}")


def branch_short(dungeon_number: int) -> str:
    return DUNGEON_SHORT.get(int(dungeon_number or 0),
                             f"d{int(dungeon_number or 0)}")


def checkpoint_kind(row: Row) -> str:
    """What KIND of checkpoint this is, from harness-set provenance only.

    ``created_by`` is set by the writer, not by the model's text: the seed
    state is written by the orchestrator, automatic checkpoints by the harness
    (`nethack.py:_maybe_auto_checkpoint`), and a model's ``save`` skill is
    stamped ``save``. Only for ``created_by == "auto"`` -- states no model
    named -- is the trigger read off the harness-written name/note, so a model
    that saves a state called ``"auto level_entry_d9"`` still shows up as
    ``model-save``. The label is a LABEL: nothing downstream measures it.
    """
    by = (row.created_by or "").strip().lower()
    if by == "orchestrator":
        return KIND_SEED
    if by == "save":
        return KIND_MODEL_SAVE
    if by == "auto":
        text = f"{row.name} {row.note}"
        if "level_entry" in text or "entered Dlvl" in text:
            return KIND_LEVEL_ENTRY
        if "level_up" in text or "reached XL" in text:
            return KIND_LEVEL_UP
        # BOTH CADENCE VOCABULARIES. The trigger counts LM calls now
        # (`call_37`, "20 LM calls elapsed"); archives written before that
        # change say `turn_222` / "150 game turns elapsed" and must keep
        # classifying as cadence rather than falling through to `auto`.
        if ("call_" in text or "LM calls elapsed" in text
                or "turn_" in text or "game turns elapsed" in text):
            return KIND_CADENCE
        return KIND_AUTO
    return by or KIND_UNKNOWN


#: Kinds, most informative first. Used only to pick which member of a group of
#: measurably identical states gets to be its display row.
_KIND_RANK = {KIND_MODEL_SAVE: 0, KIND_SEED: 1, KIND_LEVEL_ENTRY: 2,
              KIND_LEVEL_UP: 3, KIND_CADENCE: 4, KIND_AUTO: 5, KIND_UNKNOWN: 6}


def hp_fraction(row: Row) -> Optional[float]:
    """HP as a fraction of max, or ``None`` when max HP is unknown.

    NOT part of the Pareto objective and deliberately so: a state at 8/43 and a
    state at 43/43 are the same point in (Dlvl, XL, score) and are completely
    different places to resume from. This is one of the axes the objective
    cannot see, which is the whole reason the selector is a language model.
    """
    return (row.hp / row.max_hp) if row.max_hp else None


def novelty(row: Row, rows: list) -> float:
    """How rare this checkpoint's DEPTH is in the archive, in ``[0, 1)``.

    Not "how few times has it been visited" -- ``attempts_from`` already covers
    that, and using visits for both terms would double-count one signal under
    two names. This is a state-space measure: a checkpoint on a floor the
    archive barely covers is novel, twenty checkpoints on Dlvl 3 are not.
    """
    if not rows:
        return 0.0
    same = sum(1 for r in rows if r.dlvl == row.dlvl)
    return 1.0 - (same / len(rows))


def _minmax(values: list) -> list:
    """Scale to ``[0, 1]``; an all-equal list becomes all zeros.

    WHY NORMALIZE AT ALL (an implementation decision the design leaves open):
    ``balrog_min`` is BALROG's own scale, where the numbers are small and very
    unevenly spaced -- 0.0 at Dlvl 1/XL 1, 0.021 at D12/XL3, 0.029 at D15/XL5.
    Fed raw into ``w1*balrog_min + w2*novelty - w3*attempts_from``, the depth
    term is one to two orders of magnitude smaller than the other two, so
    w1=1.0 would mean "ignore depth" -- the opposite of what the weight says.
    Min-max over the CANDIDATE SET puts all three terms on the same scale,
    which is the only reading under which the documented weights mean what they
    say, and it makes them re-tunable without re-deriving scale factors --
    which matters because Q7 defers them.

    A CONSEQUENCE WORTH KNOWING BEFORE TUNING: BALROG's min is the min over the
    (Dlvl, XL) axes, and the XL axis scores 0 below XL 2. Early in a run every
    candidate therefore has ``balrog_min == 0``, this term normalizes to all
    zeros, and selection is driven entirely by novelty and the attempts-from
    discount. That is correct behaviour for a metric that says "this state has
    not demonstrably progressed on both axes", not a bug -- but a run whose
    first dozen attempts look novelty-driven is showing this, and the candidate
    records carry ``balrog`` (the max) alongside so the alternative weighting
    can be evaluated from the logs without re-running anything.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


@dataclass
class Selection:
    """A selection, with everything needed to reproduce or audit it."""

    chosen_id: Optional[str]
    candidates: list           # [{"id", "score", "prob", terms...}]
    weights: dict
    temperature: float
    rng_seed: int
    draw: float
    n_rows: int
    reason: str = "softmax_over_frontier"

    def to_json(self) -> dict:
        return asdict(self)


def select(rows: list, cfg: OrchestratorConfig, rng: random.Random) -> Selection:
    """Softmax over the Pareto frontier. Deterministic given ``rng`` and rows.

    ``score(c) = w1*norm(balrog_min) + w2*novelty - w3*norm(attempts_from)``,
    then ``P(c) = softmax(score / temperature)``. The returned record carries
    every candidate's per-term values and its final probability, because "the
    selector chose c7" is not a reproducible statement and "the selector chose
    c7 with p=0.41 out of these six, from this RNG state" is.
    """
    weights = {"balrog_min": cfg.w_balrog, "novelty": cfg.w_novelty,
               "attempts_from": cfg.w_attempts}
    if not rows:
        return Selection(None, [], weights, cfg.temperature, cfg.rng_seed,
                         draw=-1.0, n_rows=0, reason="empty_archive")

    front = pareto_frontier(rows)
    nb = _minmax([r.balrog_min for r in front])
    na = _minmax([float(r.attempts_from) for r in front])
    nov = [novelty(r, rows) for r in front]
    scores = [cfg.w_balrog * b + cfg.w_novelty * v - cfg.w_attempts * a
              for b, v, a in zip(nb, nov, na)]

    t = cfg.temperature if cfg.temperature > 0 else 1e-9
    top = max(scores)
    exps = [math.exp((s - top) / t) for s in scores]  # shift: overflow-safe
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]

    # Inverse-CDF over a SORTED-BY-ID candidate order, so the draw does not
    # depend on filesystem iteration order.
    order = sorted(range(len(front)), key=lambda i: _id_key(front[i].id))
    draw = rng.random()
    acc, pick = 0.0, order[-1]
    for i in order:
        acc += probs[i]
        if draw <= acc:
            pick = i
            break

    cands = [{
        "id": front[i].id,
        "dlvl": front[i].dlvl, "xl": front[i].xl, "score_stat": front[i].score,
        "balrog_min": front[i].balrog_min, "balrog": front[i].balrog,
        "norm_balrog_min": nb[i], "novelty": nov[i],
        "attempts_from": front[i].attempts_from, "norm_attempts_from": na[i],
        "weighted_score": scores[i], "prob": probs[i],
    } for i in order]
    return Selection(front[pick].id, cands, weights, cfg.temperature,
                     cfg.rng_seed, draw=draw, n_rows=len(rows))


def _id_key(ident: str) -> tuple:
    """Sort ids numerically when they are numeric, lexically otherwise."""
    return (0, int(ident), "") if str(ident).isdigit() else (1, 0, str(ident))


def selection_record(attempt: int, choice: dict, rows: list,
                     frontier: Optional[set] = None) -> dict:
    """One round of `selection.jsonl`: what was chosen, out of what, vs what.

    THE BUG THIS REPLACES, because it was silent and it defeated the whole
    measurement. The record used to be built as::

        {"chosen_id": chosen_id, ..., **choice["selection"]}

    and ``choice["selection"]`` is :meth:`Selection.to_json`, which has a
    ``chosen_id`` of its OWN -- the SCRIPTED selector's pick. Python applies
    the ``**`` last, so the scripted pick overwrote the id that was actually
    launched, in every round, of every LLM-selector run. While the frontier
    collapsed to one row the two agreed by construction and nothing looked
    wrong; the moment the model can choose a dominated state -- the point of
    this whole change -- the log would have recorded the frontier state the
    scripted selector wanted while a different checkpoint was resumed. It also
    wrote ``chosen_id: null`` whenever the scripted selector declined to pick
    (an empty candidate set), for an attempt that demonstrably launched.

    It did not stop at the log. `reconcile_run` treats `selection.jsonl` as a
    pre-launch record and fills a recovered attempt's ``from_checkpoint`` from
    ``sel["chosen_id"]`` -- so the shadowed value propagated into
    ``attempts.jsonl``, and every checkpoint attributed to that attempt was
    attributed to a resume that never happened.

    Now the launched id is written last and alone under ``chosen_id``, the
    scripted selector's whole record lives under ``scripted`` where it cannot
    collide, and the two are compared explicitly:

    * ``candidates_shown`` -- one record per checkpoint the model was actually
      shown, from the same call that rendered the table.
    * ``scripted_would_pick`` / ``scripted_agreed`` -- the ablation comparison
      the run is for: did the LM's judgement differ from the softmax's, and
      how often. `round2_offline.py` had to reconstruct this by hand.
    * ``chosen_on_frontier`` -- whether the choice was a dominated state, which
      is the behaviour this change exists to make possible and therefore the
      one that has to be counted.
    """
    scripted = dict(choice.get("selection") or {})
    scripted_pick = scripted.pop("chosen_id", None)
    directive = choice.get("directive") or ""
    chosen_id = choice.get("checkpoint_id")
    front = set(frontier if frontier is not None
                else {r.id for r in pareto_frontier(rows)})
    shown = choice.get("candidates_shown")
    return {
        "attempt": attempt,
        "source": choice.get("source"),
        # THE ID THAT WAS LAUNCHED. Nothing below may shadow it.
        "chosen_id": chosen_id,
        "chosen_on_frontier": (chosen_id in front) if chosen_id else None,
        "directive": directive,
        "directive_kind": classify_directive_kind(directive),
        "directive_lint": lint_directive(directive),
        "n_rows": len(rows),
        "frontier_ids": sorted(front, key=_id_key),
        "candidates_shown": shown,
        "n_candidates_shown": (len(shown) if shown is not None else None),
        "scripted_would_pick": scripted_pick,
        "scripted_agreed": (None if (scripted_pick is None or chosen_id is None)
                            else scripted_pick == chosen_id),
        "scripted": scripted,
        "llm": choice.get("decision"),
    }


# --------------------------------------------------------------------------- #
# rendering the ledger for the player
# --------------------------------------------------------------------------- #

def render_attempt_history(attempts: list, limit: int = 12) -> str:
    """What was tried, from where, with what instruction, and how it ended.

    The DIRECTIVE belongs here, next to the outcome, because that pairing is
    the orchestrator's only feedback channel about its own instructions: "told
    it to take the south door -> died anyway" is a lesson about the level, and
    "told it to take the south door -> it went north" is a lesson about the
    player. Neither is legible from the outcome alone.
    """
    if not attempts:
        return "ATTEMPT HISTORY: none yet."
    lines = ["ATTEMPT HISTORY (most recent last):"]
    for a in attempts[-limit:]:
        outcome = a.get("outcome", "?")
        if a.get("censor_reason"):
            outcome += f":{a['censor_reason']}"
        comply = (a.get("directive_compliance") or {}).get("class", "-")
        lines.append(
            f"  #{a.get('attempt')} from c{a.get('from_checkpoint')} "
            f"-> {outcome} (max Dlvl {a.get('max_dlvl')}, {a.get('calls')} calls)"
            f"\n      directive: {(a.get('directive') or '(none)')[:160]}"
            f"\n      followed?: {comply}"
        )
    return "\n".join(lines)


@dataclass
class StateGroup:
    """One measured POSITION in the archive, and every checkpoint sitting on it.

    Two checkpoints written a few calls apart at the same depth, XL, score,
    game turn and HP are, on every axis anything here can measure, the same
    place. The last mechcheck attempt produced 5 checkpoints of which 4 were
    pairwise identical that way, and the run before it produced c13/c14 and
    c19/c20. A ledger that lists them separately spends its width on
    repetition; one that DROPS them hides states from the selector. So they are
    collapsed into one row that names every id in the group -- summarizing
    duplicates, never states.
    """

    key: tuple
    members: list                      # oldest first
    rep: Row = None                    # the member the row is rendered from

    @property
    def ids(self) -> list:
        return [m.id for m in self.members]

    @property
    def count(self) -> int:
        return len(self.members)


def state_key(row: Row) -> tuple:
    """Everything measured about WHERE a checkpoint is. The dedupe key.

    Includes branch and HP, which the Pareto objective excludes: two states
    that differ in either are different places, however equal their
    (Dlvl, XL, score) is.
    """
    return (row.dungeon_number, row.level_number, row.dlvl, row.xl, row.score,
            row.gameturn, row.hp, row.max_hp)


def group_states(rows: list) -> list:
    """Collapse measurably-identical checkpoints. Order-stable, oldest first."""
    groups: list = []
    index: dict = {}
    for r in rows:
        k = state_key(r)
        g = index.get(k)
        if g is None:
            g = StateGroup(key=k, members=[r], rep=r)
            index[k] = g
            groups.append(g)
        else:
            g.members.append(r)
    for g in groups:
        # The most informative member is the row's face -- a model's `save`
        # over an automatic one, a level-entry over a cadence tick.
        g.rep = min(g.members,
                    key=lambda m: (_KIND_RANK.get(checkpoint_kind(m), 9),
                                   m.created_at, _id_key(m.id)))
    return groups


def attempts_by_checkpoint(attempts: Optional[list]) -> dict:
    """``{checkpoint id: [attempt record, ...]}`` in run order."""
    out: dict = {}
    for a in attempts or []:
        cid = a.get("from_checkpoint")
        if cid is None:
            continue
        out.setdefault(str(cid), []).append(a)
    return out


def _or_unknown(value) -> str:
    """Render a possibly-null measurement for a human. Null reads as unknown.

    Every number an attempt row carries can be null, because an attempt whose
    rollout never reported and which left no turn files measured nothing. The
    orchestrator is TOLD these numbers in its round text, so rendering null as
    `0` would teach it that a crashed attempt reached Dlvl 0 and made 0 calls.
    """
    return "unknown" if value is None else str(value)


def outcome_label(attempt: dict) -> str:
    """``died@D9`` / ``censored:wall_clock@D8`` -- what came of resuming here."""
    o = str(attempt.get("outcome") or "?")
    if attempt.get("censor_reason"):
        o += f":{attempt['censor_reason']}"
    d = attempt.get("max_dlvl")
    return f"{o}@D{d}" if d else o


#: Boilerplate the harness puts in front of every automatic note. Stripped for
#: display only -- the note on disk is untouched.
_NOTE_PREFIX = "automatic checkpoint: "


def _trim(text: str, limit: int) -> str:
    text = (text or "").strip()
    if text.startswith(_NOTE_PREFIX):
        text = text[len(_NOTE_PREFIX):]
    return text[:limit - 1] + "…" if len(text) > limit else text


def lesson_excerpt(checkpoint_dir, max_chars: int = 90) -> str:
    """The last line of this checkpoint's ``lessons.md``, flattened. TEXT."""
    try:
        text = read_lessons(checkpoint_dir)
    except Exception:
        return ""
    body = [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if not body:
        return ""
    last = body[-1]
    return last[:max_chars - 1] + "…" if len(last) > max_chars else last


#: Header widths, in one place so the column line and the rows cannot drift.
_TRIED_W = 22
_LEDGER_HEADER = ("   id    from  branch  Dlvl  XL      HP   hp%    turn  score  "
                  "dup  kind         " + "tried".ljust(_TRIED_W)
                  + " name / why saved")


def _parent_cell(r: Row) -> str:
    """The `from` column: which checkpoint this one descends from."""
    return f"c{r.parent}" if r.parent else "root"


def _state_line(g: StateGroup, *, current_id: Optional[str],
                frontier: set, tried: dict, mark_lesson: bool = True) -> str:
    r = g.rep
    frac = hp_fraction(r)
    pct = f"{round(100 * frac):>3d}%" if frac is not None else "   ?"
    mark = "->" if current_id in g.ids else (" *" if g.ids and
                                             (set(g.ids) & frontier) else "  ")
    # `attempts_from` is the harness's own counter on the checkpoint and can
    # exceed what this run's attempt history holds (it survives across runs);
    # the outcomes come from the history. Both are shown, and neither is
    # invented from the other.
    runs = [a for i in g.ids for a in tried.get(i, [])]
    n_from = max(sum(m.attempts_from for m in g.members), len(runs))
    if runs:
        tried_col = f"{n_from}: " + ",".join(outcome_label(a)
                                             for a in runs[-2:])
        if len(tried_col) > _TRIED_W:
            tried_col = tried_col[:_TRIED_W - 1] + "…"
    elif n_from:
        tried_col = f"{n_from}: (no outcome on record)"[:_TRIED_W]
    else:
        tried_col = "-"
    text = (r.name or "-")[:26]
    note = _trim(r.note, 46)
    if note:
        text += f" | {note}"
    line = (f"{mark} c{r.id:<4} {_parent_cell(r):>5} "
            f"{branch_short(r.dungeon_number):>7} "
            f"{r.dlvl:>4}  {r.xl:>2}  {r.hp:>3}/{r.max_hp:<3} {pct}  "
            f"{r.gameturn:>6} {r.score:>6}  "
            f"{('x' + str(g.count)) if g.count > 1 else '  ':>3}  "
            f"{checkpoint_kind(r):<12} {tried_col:<{_TRIED_W}} {text}")
    extra = []
    if g.count > 1:
        extra.append(f"       same measured state, each id choosable: "
                     + " ".join("c" + i for i in g.ids))
    if mark_lesson:
        lesson = lesson_excerpt(r.path)
        if lesson:
            extra.append(f"       lesson: {lesson}")
    return "\n".join([line] + extra)


def _compact_line(g: StateGroup) -> str:
    r = g.rep
    frac = hp_fraction(r)
    pct = f"{round(100 * frac)}%" if frac is not None else "?"
    ids = " ".join("c" + i for i in g.ids)
    return (f"    {ids:<14} <-{_parent_cell(r)} "
            f"{branch_short(r.dungeon_number)} D{r.dlvl} XL{r.xl} "
            f"{r.hp}/{r.max_hp} {pct} t{r.gameturn} s{r.score} "
            f"{checkpoint_kind(r)}")


def _detail_priority(g: StateGroup, frontier: set, tried: dict) -> tuple:
    """Which groups get a full row when the archive outgrows the detail budget.

    Lower sorts first. Nothing here can REMOVE a state from the ledger -- a
    group that loses gets a compact line carrying every measured axis but the
    free text, and its ids stay legal choices. This only decides who gets the
    name, note and lesson.
    """
    r = g.rep
    return (
        0 if (set(g.ids) & frontier) else 1,
        0 if checkpoint_kind(r) in (KIND_MODEL_SAVE, KIND_SEED) else 1,
        0 if any(tried.get(i) for i in g.ids) else 1,
        0 if checkpoint_kind(r) in (KIND_LEVEL_ENTRY, KIND_LEVEL_UP) else 1,
        -r.dlvl, -r.gameturn,
    )


def build_ledger(rows: list, cfg: Optional[OrchestratorConfig] = None,
                 current_id: Optional[str] = None,
                 attempts: Optional[list] = None) -> tuple:
    """``(ledger text, candidate records)`` -- the archive as the selector sees it.

    THE WHOLE ARCHIVE, NOT THE FRONTIER, and that is the point of this
    function. It used to print the Pareto frontier over (Dlvl, XL, score) plus
    named saves and hide the rest behind "(N dominated auto-checkpoint(s) not
    listed)". On a single NetHack trajectory that objective is monotone in all
    three coordinates at once -- later states dominate earlier ones by
    construction -- so the frontier collapsed to ONE row out of 24, and an LLM
    selector paid for its judgement was handed a candidate set of size one and
    reduced to justifying a forced move. `parse_decision` had always accepted
    any id in the archive; only the prompt and this table said otherwise.

    So: every checkpoint is reachable as a choice here. What the rendering
    spends its budget on is the axes the objective cannot see -- branch, HP
    fraction, game turn, why the state was saved, and what happened to the
    attempts that already resumed from it -- because those are exactly the
    grounds on which a dominated state can be the better place to branch from.

    LEGIBILITY AT SCALE, without hiding anything. Measurably identical
    checkpoints collapse into one row that names all their ids
    (:func:`group_states`); the archive is sectioned by dungeon branch and
    ordered shallow-to-deep within it; and past ``cfg.ledger_max_rows`` groups
    the remainder are rendered one compact line each, carrying every measured
    axis but the free text. An archive of hundreds stays readable and every id
    in it is still a legal choice.

    The second return value is the candidate set AS SHOWN -- one record per
    checkpoint, not per group -- so `selection.jsonl` can record what the model
    was actually offered rather than what the scripted selector would have
    offered, which are no longer the same set.
    """
    cfg = cfg or OrchestratorConfig(run_dir=Path("."))
    if not rows:
        base = ("CHECKPOINT ARCHIVE: empty. Nothing has been saved yet -- "
                "this is the first attempt. Use save(label, note) at states "
                "worth returning to.")
        if attempts:
            base += "\n\n" + render_attempt_history(attempts)
        return base, []

    frontier = {r.id for r in pareto_frontier(rows)}
    tried = attempts_by_checkpoint(attempts)
    groups = group_states(rows)
    ranked = sorted(groups, key=lambda g: _detail_priority(g, frontier, tried))
    detail = set(id(g) for g in ranked[:max(1, int(cfg.ledger_max_rows))])

    lines = [
        f"CHECKPOINT ARCHIVE: {len(rows)} saved state(s) in {len(groups)} "
        f"distinct measured position(s). EVERY id below is a legal choice -- "
        f"this is the WHOLE archive, not a shortlist and not a frontier.",
        f"All numbers are measured by the harness from the game engine, not "
        f"written by anyone; name, note and lesson are text and are never "
        f"measured.",
        f"  from   = the checkpoint this one descends from. The archive is a "
        f"TREE: within one attempt each state's parent is the previous state "
        f"that attempt saved, and a state with two children is a point two "
        f"attempts branched from.",
        f"  branch = which dungeon branch (main / MINES / SOKO ...). Depth "
        f"alone cannot tell a Mines level from a Dungeons-of-Doom one.",
        f"  hp%    = HP as a fraction of max. Two states with the same Dlvl, "
        f"XL and score can be a full-strength state and a nearly-dead one.",
        f"  dup    = how many checkpoints sit at this identical measured "
        f"state; every one of their ids is listed and each is choosable.",
        f"  kind   = why it was written: level-entry, level-up, cadence "
        f"(periodic), model-save (a player chose to save it), seed.",
        f"  tried  = how many attempts resumed from here (`attempts_from`) and "
        f"how the ones this run recorded ended.",
        f"  *      = on the (Dlvl, XL, score) Pareto frontier.",
    ]

    by_branch: dict = {}
    for g in groups:
        by_branch.setdefault(int(g.rep.dungeon_number or 0), []).append(g)
    for dnum in sorted(by_branch):
        bgroups = sorted(by_branch[dnum],
                         key=lambda g: (g.rep.dlvl, g.rep.gameturn,
                                        _id_key(g.rep.id)))
        n_ck = sum(g.count for g in bgroups)
        lines.append("")
        lines.append(f"BRANCH {dnum} -- {branch_name(dnum)} "
                     f"({branch_short(dnum)}): {n_ck} checkpoint(s), "
                     f"{len(bgroups)} distinct, Dlvl "
                     f"{min(g.rep.dlvl for g in bgroups)}-"
                     f"{max(g.rep.dlvl for g in bgroups)}")
        lines.append(_LEDGER_HEADER)
        overflow = []
        for g in bgroups:
            if id(g) in detail:
                lines.append(_state_line(g, current_id=current_id,
                                         frontier=frontier, tried=tried))
            else:
                overflow.append(g)
        if overflow:
            lines.append(f"  ALSO IN THIS BRANCH AND EQUALLY CHOOSABLE "
                         f"({sum(g.count for g in overflow)} checkpoint(s), "
                         f"same measurements, name/note omitted for width):")
            lines.extend(_compact_line(g) for g in overflow)

    if attempts:
        lines.append("")
        lines.append(render_attempt_history(attempts))

    cands = []
    for g in groups:
        shown = "detail" if id(g) in detail else "compact"
        for m in g.members:
            frac = hp_fraction(m)
            cands.append({
                "id": m.id,
                "parent": m.parent,
                "dlvl": m.dlvl, "xl": m.xl, "score": m.score,
                "dungeon_number": m.dungeon_number,
                "branch": branch_name(m.dungeon_number),
                "level_number": m.level_number,
                "hp": m.hp, "max_hp": m.max_hp,
                "hp_fraction": (round(frac, 4) if frac is not None else None),
                "gameturn": m.gameturn,
                "kind": checkpoint_kind(m),
                "attempts_from": m.attempts_from,
                "outcomes": [outcome_label(a) for a in tried.get(m.id, [])],
                "on_frontier": m.id in frontier,
                "same_state_as": (g.rep.id if m.id != g.rep.id else None),
                "duplicate_group_size": g.count,
                "shown_as": shown,
            })
    cands.sort(key=lambda c: _id_key(c["id"]))
    return "\n".join(lines), cands


def render_ledger(rows: list, cfg: Optional[OrchestratorConfig] = None,
                  current_id: Optional[str] = None,
                  attempts: Optional[list] = None) -> str:
    """The archive as text. See :func:`build_ledger` for what it contains."""
    return build_ledger(rows, cfg, current_id, attempts)[0]


def archive_tree(rows: list) -> dict:
    """``{"roots": [...], "children": {id: [id, ...]}, "problems": [...]}``.

    The archive's lineage as a graph, with every way it can fail to be a tree
    named rather than smoothed over: a parent that does not exist, a cycle, a
    second root. Callers that want to LOOK at the tree use
    :func:`render_archive_tree`; callers that want to ASSERT on it read
    ``problems``.
    """
    by_id = {r.id: r for r in rows}
    children: dict = {r.id: [] for r in rows}
    roots: list = []
    problems: list = []
    for r in rows:
        if r.parent is None:
            roots.append(r.id)
        elif r.parent not in by_id:
            problems.append(f"c{r.id}: parent c{r.parent} is not in the archive")
            roots.append(r.id)
        elif r.parent == r.id:
            problems.append(f"c{r.id}: is its own parent")
            roots.append(r.id)
        else:
            children[r.parent].append(r.id)
    for k in children:
        children[k].sort(key=_id_key)
    # CYCLES. Walk up from every node; a node that does not reach a root in
    # `len(rows)` steps is in one.
    for r in rows:
        seen = set()
        cur = r.id
        while cur is not None and cur in by_id:
            if cur in seen:
                problems.append(f"c{r.id}: its ancestry contains a cycle "
                                f"through c{cur}")
                break
            seen.add(cur)
            cur = by_id[cur].parent
    if len(roots) > 1:
        problems.append(f"{len(roots)} roots ({', '.join('c' + i for i in roots)}); "
                        f"a tree has one")
    if rows and not roots:
        problems.append("no root: every checkpoint claims a parent")
    return {"roots": sorted(set(roots), key=_id_key), "children": children,
            "problems": problems, "by_id": by_id}


def render_archive_tree(rows: list) -> str:
    """The archive as an indented tree, for a human to eyeball.

    Deliberately plain text and deliberately not pruned: this is what the
    branching structure of a run LOOKS like, and a picture that hides the
    boring parts of it would hide exactly the runs where nothing branched.
    """
    if not rows:
        return "ARCHIVE TREE: empty."
    t = archive_tree(rows)
    out = [f"ARCHIVE TREE: {len(rows)} checkpoint(s), "
           f"{len(t['roots'])} root(s)."]
    if t["problems"]:
        out.append("NOT A VALID TREE:")
        out.extend("  ! " + p for p in t["problems"])
    seen: set = set()

    def walk(ident: str, prefix: str, last: bool, top: bool) -> None:
        if ident in seen:                       # a cycle; already reported
            out.append(f"{prefix}+- c{ident} (already shown -- cycle)")
            return
        seen.add(ident)
        r = t["by_id"][ident]
        stem = "" if top else ("`- " if last else "+- ")
        out.append(f"{prefix}{stem}c{ident}  {branch_short(r.dungeon_number)} "
                   f"D{r.dlvl} XL{r.xl} s{r.score} t{r.gameturn} "
                   f"{r.hp}/{r.max_hp}  {checkpoint_kind(r)}"
                   + (f"  attempts_from={r.attempts_from}"
                      if r.attempts_from else "")
                   + (f"  \"{_trim(r.name, 32)}\"" if r.name else ""))
        kids = t["children"].get(ident, [])
        pad = prefix if top else prefix + ("   " if last else "|  ")
        for i, k in enumerate(kids):
            walk(k, pad, i == len(kids) - 1, False)

    for root in t["roots"]:
        walk(root, "", True, True)
    stranded = [r.id for r in rows if r.id not in seen]
    if stranded:
        out.append("UNREACHED FROM ANY ROOT (this is the defect, not a view "
                   "option): " + " ".join("c" + i for i in stranded))
    return "\n".join(out)


def read_lessons(checkpoint_dir) -> str:
    p = Path(checkpoint_dir) / LESSONS_MD
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def append_lesson(checkpoint_dir, text: str, *, heading: str = "") -> str:
    """Append to a checkpoint's ``lessons.md``. Atomic; returns the new content.

    Read-modify-atomic-write rather than an append: ``atomic_write`` is the
    module's one durable-write primitive, and a half-written lesson file is a
    corrupted archive entry that a later attempt would read as truth.
    """
    d = Path(checkpoint_dir)
    old = read_lessons(d)
    block = (f"\n## {heading}\n" if heading else "\n") + text.rstrip() + "\n"
    new = (old.rstrip() + "\n" if old.strip() else "") + block
    atomic_write(d / LESSONS_MD, new)
    return new


def render_lessons(checkpoint_dir, limit: int = 8) -> str:
    """The most recent lesson blocks from a checkpoint, for the resume banner."""
    text = read_lessons(checkpoint_dir).strip()
    if not text:
        return ""
    blocks = [b.strip() for b in text.split("\n## ") if b.strip()]
    tail = blocks[-limit:]
    return "LESSONS FROM EARLIER ATTEMPTS AT THIS STATE:\n" + "\n".join(
        ("## " + b if not b.startswith("#") else b) for b in tail)


def render_prefix(checkpoint_dir, max_chars: int = 1500) -> str:
    """The checkpoint's own conversation prefix, as text, if it has one.

    LIMITATION, STATED RATHER THAN HIDDEN. The design asks for true prefix
    continuity -- the resumed player's conversation IS the checkpoint's
    ``prefix.jsonl``. Neither CLI scaffold supports that: claude_code runs with
    ``--no-session-persistence`` and prime_agent's ``--print`` mode has no
    conversation to seed (see nethack.py's `resume_from` note, which hit the
    same wall). What is possible today is what `resume_from` already does:
    replay the tail of that conversation as TEXT in the first observation. So
    this returns text, the orchestrator labels it as a quotation, and the
    difference from real prefix continuity is recorded in provenance.json as
    ``prefix_continuity: "text_only"``.

    WHAT "TEXT ONLY" DOES AND DOES NOT MEAN -- measured, because the two
    readings have opposite consequences for H4. It is a statement about the
    MECHANISM (a resumed player reads a transcript rather than having had the
    conversation, so there is no inherited session and no server-side prompt
    cache). It is NOT a statement about DELIVERY. The replayed text does reach
    the model: one real seed-1 rollout resumed from a checkpoint with a
    non-empty ``prefix.jsonl`` carries this block byte-identically in exactly
    one served node -- the first observation -- along with a canary string
    that exists nowhere else in the harness, the wiki, the ledger table or the
    directive. See ``outputs/e16_launchfix/``. So H4 is testable as "lessons
    plus prefix CONTENT transfer", and only the stronger "prefix continuity of
    SESSION" is out of reach.

    ``prefix.jsonl`` is written by ``nethack.py:_append_conversation_prefix``
    after every turn and picked up by ``checkpoint_save`` off the env; it is
    empty only for checkpoints written outside a live rollout (``seed_archive``
    makes one such: c1 has no conversation behind it).
    """
    p = Path(checkpoint_dir) / PREFIX_JSONL
    if not p.is_file() or p.stat().st_size == 0:
        return ""
    recs = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not recs:
        return ""
    parts = []
    for r in recs[-6:]:
        role = r.get("role", "?")
        content = r.get("content")
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        parts.append(f"({role}) {' '.join(str(content or '').split())[:300]}")
    body = "\n".join(parts)[:max_chars]
    return ("THE PLAN THAT WAS LIVE WHEN THIS STATE WAS SAVED (quoted from that "
            "session, not your own memory):\n" + body)


# --------------------------------------------------------------------------- #
# the orchestrator's own failures, which are LOUD by construction
# --------------------------------------------------------------------------- #

class OrchestratorRoundFailed(RuntimeError):
    """Base: a round that was supposed to steer the run did not.

    WHY THESE RAISE INSTEAD OF FALLING BACK. E16's treatment is the directive
    channel and the opening plan. When either one silently degrades to "no
    instruction", the run does not become a slightly worse go_explore run -- it
    becomes its own control, launching real money at attempts that measure the
    thing the experiment was going to compare against. That is the E15 silent-
    substitution class exactly, and it happened here: the GE-wiki pilot's
    round 1 produced a well-formed directive, the driver's parser deleted the
    line carrying it, and attempt 1 was launched at $13.36 with
    "(orchestrator produced no directive for this attempt)" served in its place.
    Nothing in the run said anything was wrong.

    A run that cannot get a directive out of its orchestrator must stop and say
    so. The evidence -- the raw bytes the extraction failed on -- is written to
    the run directory before the exception leaves.
    """


class DirectiveExtractionFailed(OrchestratorRoundFailed):
    """A selection round yielded no usable checkpoint+directive decision."""


class OpeningPlanUnusable(OrchestratorRoundFailed):
    """The opening round produced no usable plan (empty, or degenerate)."""


# --------------------------------------------------------------------------- #
# degeneration detection
# --------------------------------------------------------------------------- #

#: A reply shorter than this cannot be a strategy, whatever it says.
DEGEN_MIN_CHARS = 400

#: Below this many non-blank lines the ratio test is not meaningful -- a short
#: reply legitimately repeats itself (a table, a list of ids).
DEGEN_MIN_LINES = 24

#: unique non-blank lines / non-blank lines. Below this the reply is looping.
#:
#: CALIBRATED ON THE PILOT, both sides. The GE-wiki pilot's opening round as
#: the driver recorded it: 18,088 non-blank lines, 589 unique -> 0.033. The
#: SAME round's actual model output, read off the session file: 73 non-blank
#: lines, 70 unique -> 0.959. Two orders of magnitude apart, so the threshold
#: is not a fine judgement.
DEGEN_UNIQUE_LINE_RATIO = 0.35

#: One line repeated at least this often is a loop even if the ratio survives
#: (a long reply can bury a tight loop in otherwise varied text).
DEGEN_MAX_LINE_REPEATS = 25


def detect_degeneration(text: str, *, min_chars: int = DEGEN_MIN_CHARS,
                        min_lines: int = DEGEN_MIN_LINES,
                        unique_ratio: float = DEGEN_UNIQUE_LINE_RATIO,
                        max_line_repeats: int = DEGEN_MAX_LINE_REPEATS) -> dict:
    """Is this reply a usable answer, or is the model (or the pipe) looping?

    Returns the VERDICT AND ITS INPUTS -- ``{"degenerate", "reason", "chars",
    "lines", "unique_lines", "unique_line_ratio", "max_line_repeats",
    "most_repeated_line"}`` -- because a bounded retry that fires on a
    judgement nobody can re-derive is not auditable. The numbers go on the
    round record whether the verdict was "fine" or not.

    Cheap and text-only on purpose: it runs on every orchestrator reply, it
    must never itself cost inference, and its two failure modes are asymmetric.
    A false NEGATIVE costs one wasted round; a false POSITIVE costs a retry.
    Neither can silently change what a player is told, which is the property
    that matters.
    """
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counts: dict = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    n_lines = len(lines)
    n_unique = len(counts)
    ratio = (n_unique / n_lines) if n_lines else 1.0
    top_line, top_n = "", 0
    if counts:
        top_line, top_n = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))
    out = {
        "degenerate": False, "reason": "",
        "chars": len(text), "lines": n_lines, "unique_lines": n_unique,
        "unique_line_ratio": round(ratio, 4),
        "max_line_repeats": top_n,
        "most_repeated_line": top_line[:200],
    }
    if len(text.strip()) < min_chars:
        out["degenerate"] = True
        out["reason"] = (f"reply is {len(text.strip())} chars, under the "
                         f"{min_chars}-char floor for a usable answer")
        return out
    if n_lines >= min_lines and ratio < unique_ratio:
        out["degenerate"] = True
        out["reason"] = (f"repetition loop: {n_unique} unique lines out of "
                         f"{n_lines} ({ratio:.3f} < {unique_ratio}); most "
                         f"repeated {top_n}x: {top_line[:120]!r}")
        return out
    if n_lines >= min_lines and top_n >= max_line_repeats:
        out["degenerate"] = True
        out["reason"] = (f"repetition loop: one line repeated {top_n}x "
                         f"(>= {max_line_repeats}): {top_line[:120]!r}")
    return out


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #

class BudgetExceeded(RuntimeError):
    """Raised when a launch is refused because the ceiling is reached."""


@dataclass
class Budget:
    """Cumulative spend against a hard ceiling, on TWO LINES.

    The design's rationale for the ceiling living HERE rather than in a
    per-attempt cap: players are uncapped, because the deepest run in the
    program took 353 calls and a call cap would truncate exactly the deep
    pushes this experiment exists to produce. Cost is therefore controlled by
    refusing the NEXT launch, which is only safe if the check happens before
    the launch and never after it.

    TWO LINES, NOT ONE. The design's v2 note that "the orchestrator is nearly
    free" was true of the scripted selector and is FALSE of the LLM
    orchestrator, which is a long-lived session whose context grows with every
    round. Its spend is tracked separately from the players' so a run can say
    which half of the ceiling went where -- and so a growing orchestrator
    context shows up as a rising line rather than as noise inside the player
    total. The ceiling binds their SUM.
    """

    ceiling_usd: float
    min_headroom_usd: float = 5.0
    player_usd: float = 0.0
    orchestrator_usd: float = 0.0
    #: How many attempts contributed an ESTIMATE rather than a measurement to
    #: `player_usd`, and how much of it is estimated. Non-zero makes the whole
    #: player line a LOWER BOUND, and a ceiling read off a lower bound is not a
    #: ceiling -- so it is carried on the budget itself rather than being
    #: recomputed by whoever remembers to.
    player_unknown_attempts: int = 0
    player_estimated_usd: float = 0.0

    @property
    def spent_usd(self) -> float:
        return self.player_usd + self.orchestrator_usd

    @property
    def remaining(self) -> float:
        return self.ceiling_usd - self.spent_usd

    def can_launch(self) -> bool:
        return self.remaining >= self.min_headroom_usd

    def check(self) -> None:
        if not self.can_launch():
            raise BudgetExceeded(
                f"budget ceiling reached: ${self.spent_usd:.2f} of "
                f"${self.ceiling_usd:.2f} spent (players ${self.player_usd:.2f} "
                f"+ orchestrator ${self.orchestrator_usd:.2f}), "
                f"${self.remaining:.2f} left (< ${self.min_headroom_usd:.2f} "
                f"headroom). Refusing to launch."
            )

    def add_player(self, usd: float, *, known: bool = True) -> None:
        """Book one attempt's player spend.

        `known=False` means `usd` is an ESTIMATE (the attempt's own in-flight
        `spend_so_far_usd`), booked because the alternative -- booking zero for
        an attempt that demonstrably played -- is the failure this counter
        exists to make impossible to miss. It is still added: money that was
        spent and cannot be measured is money spent, and a ceiling that ignores
        it is not a ceiling.
        """
        amount = float(usd or 0.0)
        self.player_usd += amount
        if not known:
            self.player_unknown_attempts += 1
            self.player_estimated_usd += amount

    def add_orchestrator(self, usd: float) -> None:
        self.orchestrator_usd += float(usd or 0.0)

    def lines(self) -> dict:
        return {
            "player_usd": round(self.player_usd, 4),
            "orchestrator_usd": round(self.orchestrator_usd, 4),
            "total_usd": round(self.spent_usd, 4),
            "ceiling_usd": self.ceiling_usd,
            "remaining_usd": round(self.remaining, 4),
            # Says out loud when the two lines above are not measurements.
            "player_usd_is_lower_bound": self.player_unknown_attempts > 0,
            "player_attempts_with_unknown_spend": self.player_unknown_attempts,
            "player_estimated_usd": round(self.player_estimated_usd, 4),
        }


# --------------------------------------------------------------------------- #
# the player interface
# --------------------------------------------------------------------------- #

@dataclass
class PlayerContext:
    """Everything one player attempt is launched with.

    ``directive`` is part of the launch CONTRACT, not decoration. It is what
    makes this a directed search rather than Monte-Carlo resampling of a
    curated archive, and the difference between those two is the experiment's
    main claim -- so it is a required field, it is served verbatim in the
    player's first observation, and what the player did with it is measured
    (:func:`classify_directive_compliance`).

    The only legal empty directive is the ``--no-directive`` control mode,
    which is itself a useful ablation: same archive, same selection, no
    instructions. If that arm matches the directive arm, the orchestrator's
    strategy adds nothing, and that is a finding worth being able to reach.
    """

    attempt: int
    checkpoint_dir: Optional[Path]
    checkpoint_id: Optional[str]
    archive_dir: Path
    wiki_dir: Path
    out_dir: Path
    ledger_text: str
    fidelity_log: Path
    game_seed: int
    tier: str
    #: The LAUNCHER's arm (which scaffold to run), not the experiment arm.
    arm: str
    directive: str = ""
    #: The experiment arm this attempt belongs to (`go_explore` /
    #: `matched_restart`), carried on the context so an attempt record is
    #: self-describing even read on its own.
    experiment_arm: str = ARM_GO_EXPLORE
    #: Matched-pair bookkeeping. `pair_id` is shared by a directive and its own
    #: control; `pair_role` says which is which. `solo` for an unpaired attempt.
    pair_id: Optional[int] = None
    pair_role: str = ROLE_SOLO
    #: What KIND of directive this is, so a control can be compared against the
    #: directive kind it controls for rather than against a run-wide average.
    directive_kind: str = "none"
    #: ``(core, disp)`` to reseed the engine's RNG with after restore, or None
    #: for the deterministic default. See `OrchestratorConfig.reseed_on_restore`.
    reseed: Optional[tuple] = None


@dataclass
class PlayerResult:
    """What a player attempt reports back. Numbers only; text stays text."""

    #: Raw stop reason as the harness recorded it (`traces.jsonl`'s
    #: `stop_condition`, or the launcher's own label).
    stop_condition: str = ""
    died: bool = False
    ascended: bool = False
    error: str = ""
    calls: int = 0
    spend_usd: float = 0.0
    wall_s: float = 0.0
    max_dlvl: int = 0
    max_xl: int = 1
    #: WAS THERE A MEASUREMENT AT ALL. False means every number above that the
    #: trace was supposed to carry is absent, not zero. It defaults True so a
    #: result built from a real trace needs no ceremony, and is set False on
    #: exactly the paths where `traces.jsonl` is missing or empty -- the paths
    #: that used to hand `spend_usd=0.0` / `calls=0` / `max_dlvl=0` to the
    #: accounting as though the attempt had played nothing. Measured on
    #: treesmoke2: three attempts, 205 LM turns, Dlvl 4, all three booked at
    #: $0.00 / 0 calls / max_dlvl 0 while the archive held Dlvl 4 XL 5.
    spend_known: bool = True
    #: Where `calls` / `max_dlvl` / `max_xl` came from: "trace" (the rollout
    #: reported) or "turns" (recovered from the env's own per-turn NDJSON after
    #: the rollout failed to report). "none" when there was no evidence either
    #: way, in which case those fields are UNKNOWN and the row carries null.
    evidence_source: str = "trace"
    #: Non-empty when the env tool server died of a fatal native signal; the
    #: text is what `<attempt>/env_crash.log` says (e.g. "Fatal Python error:
    #: Segmentation fault").
    env_crash: str = ""
    #: Model-written text. Recorded, appended to lessons.md, never measured.
    summary: str = ""
    lesson: str = ""
    exit_code: int = 0
    #: How many FULL rollouts this attempt was billed for. >1 means the eval CLI
    #: replayed the trajectory (see `rollouts_billed`) and `spend_usd` -- costed
    #: from the single surviving trace -- is a LOWER BOUND on what was paid.
    rollouts_paid: int = 1
    #: The error type behind each discarded attempt, in order.
    retry_errors: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


Launcher = Callable[[PlayerContext], PlayerResult]


def classify_outcome(result: PlayerResult) -> tuple:
    """``(outcome, censor_reason_or_empty)`` for one attempt.

    THE RULE: `died` requires the GAME to have ended the character. Everything
    else that stops a live attempt is `censored`, with a reason. E15 conflated
    these -- an attempt killed by the wall clock at Dlvl 11 was counted as a
    death at Dlvl 11 -- and it corrupted an arm's death-rate and depth numbers
    simultaneously (a death is a completed observation, a censoring is a lower
    bound). They must never be pooled without saying so.
    """
    if result.ascended:
        return OUTCOME_ASCENDED, ""
    # Checked BEFORE `error`, because the error the launcher reports for a
    # crashed env ("empty traces.jsonl in ...") describes the symptom the
    # orchestrator saw, not the thing that happened. `env_crash` is the
    # env process's own dying words.
    if result.env_crash:
        return OUTCOME_CENSORED, CENSOR_ENV_CRASH
    if result.error:
        low = result.error.lower()
        if "integrity" in low or "tricked" in low:
            return OUTCOME_CENSORED, CENSOR_INTEGRITY
        return OUTCOME_CENSORED, CENSOR_HARNESS_ERROR
    stop = (result.stop_condition or "").strip()
    if stop in _STOP_TO_CENSOR:
        return OUTCOME_CENSORED, _STOP_TO_CENSOR[stop]
    if stop.endswith("Error"):
        return OUTCOME_CENSORED, CENSOR_HARNESS_ERROR
    if result.died:
        return OUTCOME_DIED, ""
    if stop in ("", "died", "game_over"):
        # No error, no recognised infra stop, and the game did not report a
        # death: something ended a live attempt and did not say what.
        return OUTCOME_CENSORED, CENSOR_UNKNOWN
    return OUTCOME_CENSORED, CENSOR_UNKNOWN


# --------------------------------------------------------------------------- #
# directive compliance
# --------------------------------------------------------------------------- #

COMPLY_FOLLOWED = "followed"
COMPLY_PARTIAL = "partial"
COMPLY_IGNORED = "ignored"
COMPLY_VIOLATED = "violated"        # did the thing it was told to avoid
COMPLY_PREVENTED = "prevented"      # ended before it could act on it
COMPLY_NONE = "no_directive"
COMPLY_UNKNOWN = "unclassified"     # no call data to judge from

#: One clause's verdict. The three-valued result is the point: `no-evidence`
#: is NOT `satisfied`, and collapsing them is how "do not X" scores `followed`
#: on an attempt that never had the chance to do X.
CLAUSE_SATISFIED = "satisfied"
CLAUSE_VIOLATED = "violated"
CLAUSE_NO_EVIDENCE = "no-evidence"

#: The two kinds of clause a directive can contain, scored SEPARATELY.
#:
#: THE MEASUREMENT THAT FORCED THIS. The E16 sims first scored both directive
#: arms `prevented` on one enum, because both hit their call budget with a goal
#: clause unmet -- and that erased the fact that the `no_descend` arm's
#: PROHIBITION ("do NOT descend") had been fully and checkably obeyed for 26
#: calls. A prohibition is scoreable at any budget: not descending in ten calls
#: is real evidence. Reaching XL 3 in ten calls was never possible, so the goal
#: clause had no evidence either way. One enum cannot carry both facts, and the
#: one it carried was the less informative one.
KIND_PROHIBITION = "prohibition"
KIND_GOAL = "goal"

#: Words that carry no instruction, stripped before matching.
_STOPWORDS = frozenset("""
a an the and or but if then than to from into onto at by for with of on in out
up down is are was were be been being do does did doing this that these those
it its your you yours we our us i me my try trying go going get getting take
taking use using keep keeping make making please should must can could would
will shall may might now first next again more most very just only also
attempt attempts state states turn turns time times back
""".split())

#: Words that flip the sense of what follows: the directive is telling the
#: player NOT to do something. Doing it anyway is `violated`, which is a
#: DIFFERENT finding from `ignored` -- it means the instruction was read and
#: overridden, not missed.
_AVOID_WORDS = frozenset({"avoid", "not", "never", "don't", "dont", "without",
                          "skip", "stay", "away", "no"})

#: How many of the attempt's first calls the rubric looks at. A directive is an
#: instruction about what to do NEXT; whether the player was still honouring it
#: 200 calls later is a different question this rubric does not claim to answer.
DIRECTIVE_WINDOW = 12


#: The TOOL vocabulary -- the names of things the player CALLS, as opposed to
#: outcomes the game REACHES. A prohibition phrased over any of these is
#: unscoreable, and :func:`lint_directive` says so.
_TOOL_WORDS = frozenset({
    "explore", "exploring", "explores", "search", "searching", "searches",
    "move", "moving", "moves", "press", "pressing", "key", "keys",
    "wiki", "rollback", "save", "look", "looking", "map", "request_map",
    "np_explore_level", "np_move_to", "np_press_key", "np_search",
    "np_melee_attack", "np_kick", "np_pickup", "np_travel",
    "call", "calls", "calling", "tool", "tools", "skill", "skills",
})

#: OUTCOME words -- things the engine measures, which a prohibition CAN be
#: scored against. Kept beside the tool list so the lint can say what a better
#: phrasing would look like rather than only what is wrong.
_OUTCOME_WORDS = frozenset({
    "descend", "descending", "descent", "downstairs", "deeper", "dlvl",
    "die", "dying", "death", "level", "floor", "clear", "cleared",
    "fight", "fighting", "melee", "attack", "attacking", "kill", "killing",
    "engrave", "pray", "praying", "ascend", "hp", "starve",
})

#: Tokens that name DESCENT, which the harness measures directly
#: (``metrics.descent_count`` / ``max_dlvl_reached``). A clause containing one
#: of these is scored against the METRIC, not against a keyword match on a call
#: -- which is the difference between "it typed `>`" and "it went down".
_DESCEND_TOKENS = frozenset({"descend", "descending", "descent", "descended",
                             "downstairs", "deeper", "downward", "down"})

#: Skill names / keys that constitute an ATTEMPT to descend.
_DESCEND_CALL_TOKENS = ("np_down", "np_descend", "np_stairs_down", '">"', "'>'")


def _tokens(text: str) -> list:
    return [w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS]


#: Where one clause ends and the next begins. Deliberately coarse: a directive
#: is one or two sentences of instruction, not prose, and over-splitting costs
#: only precision on a clause that then scores `no-evidence`.
_CLAUSE_SPLIT = re.compile(
    r"(?:[.;]+|\s+(?:and|but|then|while|also|however)\s+|,\s+)", re.I)


def split_directive_clauses(directive: str) -> list:
    """A directive as a list of ``{"kind", "text", "tokens"}`` clauses.

    Prohibitions and goals are separated because they are scoreable under
    different conditions -- see :data:`KIND_PROHIBITION`. A segment that names
    something to do BEFORE its avoidance word yields both: "descend fast, do
    not fight" is one goal and one prohibition, not one muddled clause.
    """
    directive = (directive or "").strip()
    if not directive:
        return []
    out = []
    for seg in _CLAUSE_SPLIT.split(directive):
        seg = seg.strip()
        if not seg:
            continue
        lowered = re.findall(r"[a-z0-9_']+", seg.lower())
        avoid_at = next((i for i, w in enumerate(lowered)
                         if w in _AVOID_WORDS), None)
        if avoid_at is None:
            toks = _tokens(seg)
            if toks:
                out.append({"kind": KIND_GOAL, "text": seg, "tokens": toks})
            continue
        head = " ".join(lowered[:avoid_at])
        tail = " ".join(lowered[avoid_at:])
        head_toks = _tokens(head)
        if head_toks:
            out.append({"kind": KIND_GOAL, "text": head.strip(),
                        "tokens": head_toks})
        tail_toks = _tokens(tail)
        if tail_toks:
            out.append({"kind": KIND_PROHIBITION, "text": tail.strip(),
                        "tokens": tail_toks})
    return out


def lint_directive(directive: str) -> list:
    """Warnings about a directive's PHRASING, before its compliance is scored.

    THE ONE THAT MATTERS: a prohibition over a TOOL is not scoreable, and the
    sims produced the counter-example rather than the argument. The
    `descend_fast` arm was told "do not explore" and called ``np_explore_level``
    seven times, which any syntactic check scores `violated`. Its own reasoning
    says why: "No valid path found. I need to explore more first". It explored
    IN ORDER TO descend -- in service of the directive's own goal. A checker
    that never reads the model's prose (which is the whole point of a checker,
    since the prose is the thing under test) cannot separate "explored instead
    of obeying" from "explored in order to obey", so scoring the second as a
    violation understates compliance and scoring it as compliance would make
    the rubric unfalsifiable.

    The cheap fix is upstream: prohibit OUTCOMES, not tools. "Do not clear the
    level" is checkable against what the game shows; "do not explore" is not.
    So the warning is raised here, recorded on the attempt, and fed back into
    the orchestrator's next round -- where the phrasing can actually change.

    Returns a list of ``{"code", "clause", "message"}``. Empty is good news.
    """
    warnings = []
    clauses = split_directive_clauses(directive)
    if not (directive or "").strip():
        return warnings
    for c in clauses:
        if c["kind"] != KIND_PROHIBITION:
            continue
        tool_hits = sorted(set(c["tokens"]) & _TOOL_WORDS)
        outcome_hits = sorted(set(c["tokens"]) & _OUTCOME_WORDS)
        if tool_hits and not outcome_hits:
            warnings.append({
                "code": "tool_prohibition",
                "clause": c["text"],
                "tokens": tool_hits,
                "message": (
                    f"prohibition names a TOOL ({', '.join(tool_hits)}), not an "
                    f"OUTCOME, and is therefore UNSCOREABLE: the player may call "
                    f"that tool in service of the directive's own goal, and no "
                    f"check that stays off the model's prose can tell that from "
                    f"disobedience. Prefer an outcome: 'do not clear the level' "
                    f"rather than 'do not explore'."),
            })
    if clauses and all(c["kind"] == KIND_PROHIBITION for c in clauses):
        warnings.append({
            "code": "no_goal_clause",
            "clause": directive.strip(),
            "tokens": [],
            "message": ("directive is prohibition-only, so it is satisfied by "
                        "dying on turn 2; it cannot distinguish steering from "
                        "an attempt that never got started. Pair every "
                        "prohibition with something to DO."),
        })
    if clauses and all(c["kind"] == KIND_GOAL for c in clauses) \
            and not any(set(c["tokens"]) & _OUTCOME_WORDS for c in clauses):
        warnings.append({
            "code": "no_measured_outcome",
            "clause": directive.strip(),
            "tokens": [],
            "message": ("no clause names anything the harness measures "
                        "(descent, depth, XL, fighting, engraving), so every "
                        "clause will be scored by surface keyword match on the "
                        "tool calls and a compliant player that took an "
                        "unnamed route will score low."),
        })
    return warnings


RUBRIC_NAME = "per-clause (prohibition/goal) over the first N calls + metrics"


def _call_haystack(calls) -> str:
    return " ".join(
        f"{c.get('name', '')} "
        f"{c.get('args') if isinstance(c.get('args'), str) else json.dumps(c.get('args') or {})}"
        for c in calls if isinstance(c, dict)
    ).lower()


def _descent_evidence(metrics: Optional[dict], haystack: str) -> Optional[bool]:
    """Did this attempt DESCEND? ``None`` when nothing can answer.

    Metrics first, calls second, and the order is the whole point: `>` was
    typed is not the same fact as the hero went down a level, and only the
    first of those is what a "descend" clause is about.
    """
    metrics = metrics or {}
    for key in ("descent_count",):
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            return bool(v > 0)
    start, best = metrics.get("start_dlvl"), metrics.get("max_dlvl")
    if isinstance(start, (int, float)) and isinstance(best, (int, float)):
        return bool(best > start)
    if any(t in haystack for t in _DESCEND_CALL_TOKENS):
        # An ATTEMPT to descend. Enough to violate a prohibition, never enough
        # to satisfy a goal -- so this is only consulted where that asymmetry
        # is handled by the caller.
        return True
    return None


def score_clause(clause: dict, calls, haystack: str,
                 metrics: Optional[dict]) -> dict:
    """One clause's verdict, with the evidence that produced it.

    Two evidence bases, used in this order:

    1. HARNESS METRICS, where the clause names something the engine measures.
       Today that is descent, which is the clause kind the sims actually
       exercised. This is an OUTCOME check and it is the strong one.
    2. SURFACE KEYWORD MATCH over the tool calls, for everything else. Its
       limits are real and are reported with every verdict: it matches tokens,
       so "take the south door" scores against a call whose args mention
       ``south``, and a player that complied by a route the directive did not
       name scores low.

    Deliberately not an LM judge. An LM grading whether a second LM followed a
    third LM's instruction is three models deep and is not a measurement.
    """
    toks = set(clause["tokens"])
    is_prohibition = clause["kind"] == KIND_PROHIBITION
    if toks & _DESCEND_TOKENS:
        went_down = _descent_evidence(metrics, haystack)
        if went_down is None:
            return {"name": clause["text"][:60], "kind": clause["kind"],
                    "result": CLAUSE_NO_EVIDENCE, "basis": "metric",
                    "why": "no descent metric and no descend call in the window"}
        if is_prohibition:
            return {"name": clause["text"][:60], "kind": clause["kind"],
                    "result": CLAUSE_VIOLATED if went_down else CLAUSE_SATISFIED,
                    "basis": "metric",
                    "why": ("descended (or called a descend skill) despite the "
                            "prohibition" if went_down else
                            "no descent registered and no descend skill or '>' "
                            "key in the examined calls")}
        return {"name": clause["text"][:60], "kind": clause["kind"],
                "result": CLAUSE_SATISFIED if went_down else CLAUSE_VIOLATED,
                "basis": "metric",
                "why": "descent_count/max_dlvl says the hero "
                       + ("did" if went_down else "did not") + " go down"}

    if not calls:
        return {"name": clause["text"][:60], "kind": clause["kind"],
                "result": CLAUSE_NO_EVIDENCE, "basis": "calls",
                "why": "no tool calls to read"}
    hits = sorted(t for t in toks if t in haystack)
    if is_prohibition:
        return {"name": clause["text"][:60], "kind": clause["kind"],
                "result": CLAUSE_VIOLATED if hits else CLAUSE_SATISFIED,
                "basis": "calls", "matched": hits,
                "why": (f"prohibited token(s) {hits} appear in the calls"
                        if hits else
                        f"none of {sorted(toks)} appear in the examined calls")}
    need = max(1, (len(toks) + 1) // 2)
    if len(hits) >= need:
        result = CLAUSE_SATISFIED
    elif hits:
        result = CLAUSE_VIOLATED   # partial match: some of the goal, not enough
    else:
        result = CLAUSE_VIOLATED
    return {"name": clause["text"][:60], "kind": clause["kind"],
            "result": result, "basis": "calls", "matched": hits,
            "why": f"{len(hits)}/{len(toks)} goal token(s) matched "
                   f"({hits}); {need} needed"}


def _kind_label(scored: list, kind: str, *, pursued: bool,
                ended_early: bool) -> dict:
    """One clause kind's label, from its clauses' verdicts.

    The asymmetry between the kinds is deliberate and is the finding this
    structure exists to preserve: a GOAL can be `prevented` (the attempt ran
    out before the goal could be reached), a PROHIBITION cannot. Not descending
    in ten calls is real evidence of obedience; reaching XL 3 in ten calls was
    never possible.
    """
    sel = [r for r in scored if r["kind"] == kind]
    sat = [r for r in sel if r["result"] == CLAUSE_SATISFIED]
    vio = [r for r in sel if r["result"] == CLAUSE_VIOLATED]
    noe = [r for r in sel if r["result"] == CLAUSE_NO_EVIDENCE]
    if not sel:
        label = "n/a"
    elif vio and not sat:
        label = (COMPLY_PREVENTED
                 if kind == KIND_GOAL and pursued and ended_early
                 else COMPLY_IGNORED)
    elif vio:
        label = COMPLY_PARTIAL
    elif sat:
        label = COMPLY_FOLLOWED
    else:
        label = COMPLY_PREVENTED if kind == KIND_GOAL else COMPLY_UNKNOWN
    return {"label": label, "clauses": [r["name"] for r in sel],
            "n_satisfied": len(sat), "n_violated": len(vio),
            "n_no_evidence": len(noe)}


def _headline(by_kind: dict) -> str:
    """The single field the design asks for, DERIVED from the per-kind labels.

    Kept only because one field is asked for, and derived rather than measured
    so that the two labels it summarises are always the more informative
    record.

    ``violated`` outranks ``ignored`` when the directive was PURELY a
    prohibition and that prohibition was broken. The distinction is the one the
    original enum was right about and is worth carrying forward: an instruction
    that was reachable and was overridden is a different fact about the player
    from an instruction it never engaged with, and only the first is evidence
    that the orchestrator's channel works at all.
    """
    labels = [v["label"] for v in by_kind.values() if v["label"] != "n/a"]
    if not labels:
        return COMPLY_UNKNOWN
    if all(l == COMPLY_FOLLOWED for l in labels):
        return COMPLY_FOLLOWED
    proh = by_kind.get(KIND_PROHIBITION, {}).get("label", "n/a")
    goal = by_kind.get(KIND_GOAL, {}).get("label", "n/a")
    if proh == COMPLY_IGNORED and goal == "n/a":
        return COMPLY_VIOLATED
    if COMPLY_IGNORED in labels and COMPLY_FOLLOWED not in labels:
        return COMPLY_IGNORED
    if COMPLY_PREVENTED in labels and not (
            {COMPLY_IGNORED, COMPLY_PARTIAL} & set(labels)):
        return COMPLY_PREVENTED
    return COMPLY_PARTIAL


def classify_directive_compliance(directive: str, calls, outcome: str,
                                  window: int = DIRECTIVE_WINDOW,
                                  metrics: Optional[dict] = None) -> dict:
    """Did the player do what the orchestrator told it to? Scored PER CLAUSE.

    WHY THIS EXISTS, and it is the research point rather than a nicety: without
    compliance data there is no way to tell "the orchestrator steered the
    search" from "the orchestrator narrated while random restarts did the
    work". Those two produce identical depth curves and opposite conclusions,
    and the second is the null hypothesis this experiment has to be able to
    fail against. So compliance is measured per attempt, not assumed.

    WHY PER CLAUSE. See :data:`KIND_PROHIBITION`: one enum cannot carry a
    directive that mixes a prohibition with an aspiration, and when the sims
    made it try, the label it produced erased the only half that had evidence.
    Prohibition and goal are scored separately, both are recorded, and the
    single ``class`` field is DERIVED from them rather than measured.

    ``calls`` is a sequence of ``{"name": str, "args": dict|str}`` -- the
    player's tool calls in order. ``metrics`` is the harness's own numbers for
    the attempt (``descent_count``, ``max_dlvl``, ``start_dlvl``, ``died``,
    ``budget_exhausted``), used wherever a clause names something measured.

    Returns a dict whose ``class`` key is the derived headline (so every
    existing reader keeps working) alongside ``by_clause_kind``, ``clauses``,
    ``warnings`` (the phrasing lint) and ``rationale``.
    """
    directive = (directive or "").strip()
    base = {"class": COMPLY_NONE, "headline": COMPLY_NONE,
            "by_clause_kind": {}, "clauses": [], "matched": [], "missed": [],
            "violated": [], "window": window, "n_calls_examined": 0,
            "pursued": False, "instrumental_ambiguity": False,
            "warnings": [], "rationale": "", "rubric": RUBRIC_NAME}
    if not directive:
        base["rationale"] = ("no directive was served (control mode or a round "
                             "in which the orchestrator produced none)")
        return base

    clauses = split_directive_clauses(directive)
    warnings = lint_directive(directive)
    base["warnings"] = warnings
    calls = list(calls or [])[:window]
    base["n_calls_examined"] = len(calls)
    metrics = dict(metrics or {})
    haystack = _call_haystack(calls)

    if not clauses:
        base["class"] = base["headline"] = COMPLY_UNKNOWN
        base["rationale"] = ("the directive reduced to no scoreable clause "
                             "after stopword removal")
        return base

    scored = [score_clause(c, calls, haystack, metrics) for c in clauses]
    base["clauses"] = scored

    # `pursued` -- the BEHAVIOURAL signal, kept separate from the outcome so
    # "tried hard and did not arrive" is never silently merged with "did not
    # try". This is ambiguity 1 from the sims, made a field instead of a note.
    goal_toks = {t for c in clauses if c["kind"] == KIND_GOAL for t in c["tokens"]}
    pursued = bool(goal_toks & set(re.findall(r"[a-z0-9_]+", haystack)))
    if goal_toks & _DESCEND_TOKENS:
        pursued = pursued or any(t in haystack for t in _DESCEND_CALL_TOKENS)
    base["pursued"] = pursued
    ended_early = bool(metrics.get("budget_exhausted")
                       or metrics.get("died")
                       or outcome in (OUTCOME_DIED, OUTCOME_CENSORED))

    if not calls:
        # Nothing to read. `prevented` when the attempt ended, `unclassified`
        # when it did not -- never `followed`, which is the vacuous label that
        # would make a fast death look like obedience. The per-clause record is
        # still written: "which clauses could not be scored, and why" is the
        # part a later analysis needs, and dropping it here would make a
        # zero-call attempt the one case with no auditable rubric output.
        base["by_clause_kind"] = {
            KIND_PROHIBITION: _kind_label(scored, KIND_PROHIBITION,
                                          pursued=pursued,
                                          ended_early=ended_early),
            KIND_GOAL: _kind_label(scored, KIND_GOAL, pursued=pursued,
                                   ended_early=ended_early),
        }
        base["class"] = base["headline"] = (
            COMPLY_PREVENTED if ended_early else COMPLY_UNKNOWN)
        base["rationale"] = ("no tool calls were recorded for this attempt, so "
                             "no clause backed by call evidence could be "
                             "scored either way; per-clause verdicts are "
                             "recorded as no-evidence rather than as "
                             "compliance")
        return base

    by_kind = {
        KIND_PROHIBITION: _kind_label(scored, KIND_PROHIBITION,
                                      pursued=pursued, ended_early=ended_early),
        KIND_GOAL: _kind_label(scored, KIND_GOAL,
                               pursued=pursued, ended_early=ended_early),
    }
    base["by_clause_kind"] = by_kind
    headline = _headline(by_kind)
    base["class"] = base["headline"] = headline

    # Back-compat surface for readers that only ever wanted token lists.
    base["matched"] = sorted({m for r in scored for m in r.get("matched", [])})
    base["violated"] = sorted({m for r in scored
                               if r["kind"] == KIND_PROHIBITION
                               and r["result"] == CLAUSE_VIOLATED
                               for m in r.get("matched", [])})
    base["missed"] = sorted({t for c in clauses for t in c["tokens"]}
                            - set(base["matched"]))

    # THE INSTRUMENTAL-ACTION AMBIGUITY, recorded rather than resolved. See
    # `lint_directive`: a tool-prohibition violated while the directive's own
    # goal was unmet and being pursued is exactly the case the sims measured --
    # "do not explore" broken seven times by a player whose stated reason was
    # "I need to explore more to find the path" to the staircase it had been
    # told to reach. The label stands (the rubric does not read prose, so it
    # cannot know), but the rationale says the label may understate compliance,
    # so nobody downstream reads it as a clean violation.
    tool_prohibition = any(w["code"] == "tool_prohibition" for w in warnings)
    prohibition_broken = by_kind[KIND_PROHIBITION]["label"] in (
        COMPLY_IGNORED, COMPLY_PARTIAL)
    goal_unmet = by_kind[KIND_GOAL]["label"] in (
        COMPLY_IGNORED, COMPLY_PARTIAL, COMPLY_PREVENTED)
    instrumental = bool(tool_prohibition and prohibition_broken
                        and goal_unmet and pursued)
    base["instrumental_ambiguity"] = instrumental

    parts = [
        f"prohibition clauses -> {by_kind[KIND_PROHIBITION]['label']} "
        f"({by_kind[KIND_PROHIBITION]['n_satisfied']} satisfied / "
        f"{by_kind[KIND_PROHIBITION]['n_violated']} violated)",
        f"goal clauses -> {by_kind[KIND_GOAL]['label']} "
        f"({by_kind[KIND_GOAL]['n_satisfied']} satisfied / "
        f"{by_kind[KIND_GOAL]['n_violated']} violated)",
        f"headline {headline} is DERIVED from those two, not measured",
        f"pursued={pursued} (behaviour, reported separately from outcome so "
        f"'tried and did not arrive' is not merged with 'did not try')",
    ]
    if instrumental:
        parts.append(
            "INSTRUMENTAL-ACTION AMBIGUITY: this directive prohibits a TOOL "
            "while setting a goal that the tool can serve, and the player broke "
            "the prohibition with the goal still unmet and visibly being "
            "pursued. The measured case: 'do not explore' broken seven times by "
            "a player whose own reasoning was 'I need to explore more to find "
            "the path' to the staircase the same directive told it to reach. A "
            "check that stays off the model's prose cannot separate 'explored "
            "instead of obeying' from 'explored in order to obey', so this "
            "label MAY UNDERSTATE compliance and must not be read as a clean "
            "violation. The fix is upstream: prohibit outcomes, not tools.")
    if headline == COMPLY_FOLLOWED and not any(
            r["result"] == CLAUSE_SATISFIED and r["basis"] == "metric"
            for r in scored):
        parts.append(
            "NO METRIC-BACKED POSITIVE EVIDENCE: this `followed` rests on "
            "keyword matches and absences only. 'Do not descend' is also "
            "satisfied by dying on turn 2.")
    base["rationale"] = " | ".join(parts)
    return base


#: A coarse KIND for a directive, so a control can be paired with the directive
#: it controls FOR rather than with the run as a whole. The sims' finding: a
#: `no_descend` directive was indistinguishable from the no-directive control
#: because the control did not descend either -- a prohibition against
#: something the model was not going to do measures nothing. Pairing per kind is
#: what makes that comparison mean anything.
DIRECTIVE_KIND_RULES = (
    ("no_descend", KIND_PROHIBITION, _DESCEND_TOKENS),
    ("descend", KIND_GOAL, _DESCEND_TOKENS),
    ("no_fight", KIND_PROHIBITION, frozenset({"fight", "fighting", "melee",
                                              "attack", "attacking", "kill"})),
    ("fight", KIND_GOAL, frozenset({"fight", "fighting", "melee", "attack",
                                    "kill", "experience"})),
    ("no_explore", KIND_PROHIBITION, frozenset({"explore", "exploring",
                                                "search", "searching"})),
    ("explore", KIND_GOAL, frozenset({"explore", "exploring", "clear",
                                      "search", "searching", "rooms"})),
    ("engrave", KIND_GOAL, frozenset({"engrave", "elbereth"})),
    ("pray", KIND_GOAL, frozenset({"pray", "praying", "prayer"})),
    ("ascend_stairs_up", KIND_GOAL, frozenset({"upstairs", "back", "return"})),
)


def classify_directive_kind(directive: str) -> str:
    """A short tag naming what KIND of instruction this directive is.

    Prohibitions are checked before goals on the same vocabulary, because
    "do not descend" and "descend" share every content token and only the
    clause kind separates them.
    """
    clauses = split_directive_clauses(directive)
    if not clauses:
        return "none" if not (directive or "").strip() else "other"
    tags = []
    for tag, kind, vocab in DIRECTIVE_KIND_RULES:
        for c in clauses:
            if c["kind"] == kind and (set(c["tokens"]) & vocab):
                tags.append(tag)
                break
    if not tags:
        return "other"
    return "+".join(sorted(set(tags)))


def calls_from_turns(out_dir) -> list:
    """The player's tool calls, in order, from its per-turn NDJSON.

    `turns/<seed>_<pid>_<ts>.ndjson` is written by the env itself
    (`helpers._write_trace_entry`), one line per LM turn with the `tool_calls`
    that turn made -- so this reads what the player DID, from the harness's own
    record, not from anything the player said about itself.
    """
    out = []
    for path in all_turn_files(out_dir):
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for call in rec.get("tool_calls") or []:
                if isinstance(call, dict):
                    out.append({"name": call.get("name") or call.get("tool") or "",
                                "args": call.get("args") or call.get("arguments") or {}})
                elif isinstance(call, str):
                    out.append({"name": call, "args": {}})
    return out


# --------------------------------------------------------------------------- #
# durability: write as we play
# --------------------------------------------------------------------------- #
#
# THE FAILURE THIS SECTION EXISTS FOR, in one paragraph.
#
# The `mechcheck` pilot ran one attempt for 26 minutes. It restored c1, played
# 249 LM turns to Dlvl 8 / XL 4, and wrote 23 auto-checkpoints into the archive
# -- every one of them a valid, restorable bundle. Then the orchestrator's view
# of it stopped: `attempts.jsonl` was never created, `summary.json` was never
# written, and no spend was ever attributed. The checkpoints were durable; the
# BOOKKEEPING was not, because every record of an attempt was written at the
# END of `ingest`, which only runs after `launch_cell.sh` returns. An attempt
# that never returns therefore leaves side effects in the archive with no row
# to attribute them to -- 24 checkpoints and no attempt -- which is precisely
# the "censored is indistinguishable from died, spend is unaccounted" corruption
# class this experiment was built to make impossible.
#
# Three mechanisms, in the order they run:
#
#   1. THE JOURNAL (`attempts_journal.jsonl`). One `open` line BEFORE the
#      player launches, one `close` line after it is finalized. An attempt that
#      exists only as an `open` is an attempt that was interrupted, and the
#      `open` carries everything needed to finalize it later.
#   2. THE PROGRESS STREAM (`progress.jsonl`). A background sampler appends the
#      attempt's live state every `progress_interval_s` while it plays. The
#      orchestrator is blocked in `subprocess.run` for the whole rollout, so
#      this is the only thing that makes a running attempt observable -- and
#      its `idle_s` column is the signal that would have shown the mechcheck
#      run was not dead but sitting in a 12-minute harness relaunch gap.
#   3. RECONCILIATION (`reconcile_run`). On startup, orphaned checkpoints are
#      attributed to the attempts whose windows contain them, interrupted
#      attempts are finalized as `censored:interrupted`, and anything that
#      cannot be attributed is REPORTED rather than quietly absorbed.


class RunInterrupted(BaseException):
    """A signal ended the run. Deliberately NOT an ``Exception``.

    It inherits ``BaseException`` so that no ``except Exception`` anywhere in
    the loop can swallow an operator's SIGTERM and convert a killed run into a
    censored attempt that then continues -- while still unwinding the stack, so
    the ``finally`` that finalizes the in-flight attempt actually runs. A bare
    default SIGTERM would skip all of that: Python's default disposition
    terminates the process without running a single cleanup handler, which is
    exactly how mechcheck's bookkeeping would have been lost even if the
    process HAD been signalled.
    """

    def __init__(self, signame: str, signum: int):
        super().__init__(f"run interrupted by {signame}")
        self.signame = signame
        self.signum = signum


def _now_iso(ts: Optional[float] = None) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts if ts is not None else time.time(),
                                  timezone.utc).isoformat(timespec="seconds")


def tail_turn_record(path, max_bytes: int = 1 << 20) -> Optional[dict]:
    """The last COMPLETE turn record in an NDJSON file, read from the tail.

    Reading from the tail rather than parsing the file is what makes sampling
    cheap enough to do every 15 seconds: mechcheck's turn file reached 6 MB in
    26 minutes, and a monitor that re-read it each sample would cost more than
    the thing it is monitoring. A partial final line (the writer is appending
    underneath us) is skipped, not repaired.
    """
    try:
        path = Path(path)
        size = path.stat().st_size
        if not size:
            return None
        with open(path, "rb") as fh:
            start = max(0, size - max_bytes)
            fh.seek(start)
            blob = fh.read()
        if start:
            blob = blob.split(b"\n", 1)[-1]  # drop the partial first line
        for line in reversed(blob.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # the writer is mid-append; the line before is whole
            if isinstance(rec, dict):
                return rec
        return None
    except OSError:
        return None


#: Where a turn record keeps the engine's blstats. `helpers._write_trace_entry`
#: writes them under `status`; `applied` in that schema is a BOOLEAN. Reading
#: the wrong one is silent -- it yields an empty dict and every recovered
#: number defaults, which is how a mechcheck row first came back claiming XL 1
#: on a game whose own record says XL 4. Both names are tried, in this order,
#: and only a dict is accepted.
_BLSTATS_KEYS = ("status", "blstats", "applied")


def _turn_blstats(rec: dict) -> dict:
    for key in _BLSTATS_KEYS:
        val = rec.get(key)
        if isinstance(val, dict):
            return val
    return {}


def _turn_files(turns_dir) -> list:
    try:
        return sorted(Path(turns_dir).glob("*.ndjson"),
                      key=lambda p: p.stat().st_mtime)
    except OSError:
        return []


def all_turn_files(out_dir) -> list:
    """Every per-turn NDJSON an attempt left behind, QUARANTINED ONES INCLUDED.

    `tools/stall_watchdog.py` MOVES the turn file of a killed rollout out of
    `turns/` into `turns.stalled/<stamp>_pid<pid>/` before the orchestrator
    ingests the attempt. Nothing here knew that, so every post-hoc reader
    (`calls_from_turns`, `_max_dlvl_from_turns`, `_attempt_dir_evidence`) saw
    an empty `turns/` and concluded the attempt had played nothing.

    That is how treesmoke2 reported `max_dlvl_per_attempt: [0]` and
    `calls_to_reach: 0` for three attempts that played 205 LM turns to Dlvl 4
    -- while its own archive held the Dlvl 4 / XL 5 checkpoints those turns
    produced. An attempt row and the archive contradicting each other about
    the same run is worse than either being wrong alone: a reader of
    summary.json would conclude the method did nothing.

    Quarantined files are real engine records; they are quarantined because the
    ROLLOUT was killed, not because the turns are suspect. Sorted by mtime so
    the last element is the attempt's last turn wherever it now lives.
    """
    out = list(_turn_files(Path(out_dir) / "turns"))
    stalled = Path(out_dir) / "turns.stalled"
    if stalled.is_dir():
        try:
            out.extend(p for p in stalled.glob("*/*.ndjson") if p.is_file())
        except OSError:
            pass
    try:
        out.sort(key=lambda p: p.stat().st_mtime)
    except OSError:
        pass
    return out


def sample_attempt(turns_dir, archive_dir, *, baseline: frozenset = frozenset()
                   ) -> dict:
    """One observation of an attempt in flight, from ITS OWN files.

    Every number here is read off the harness's per-turn record or off the
    archive directory -- there is no channel through which a running player
    could report a number about itself into this stream.
    """
    files = _turn_files(turns_dir)
    now = time.time()
    out: dict = {
        "turn_files": len(files),
        "turn_file": files[-1].name if files else "",
        "idle_s": None,
        "lm_turn": None, "game_turn": None, "dlvl": None,
        "max_dlvl": None, "xl": None, "hp": None, "max_hp": None,
        "score": None, "tool_calls_last_turn": None,
        "checkpoints_total": 0, "checkpoints_new": [],
    }
    if files:
        newest = files[-1]
        try:
            out["idle_s"] = round(now - newest.stat().st_mtime, 1)
        except OSError:
            pass
        rec = tail_turn_record(newest)
        if rec:
            blstats = _turn_blstats(rec)
            out["lm_turn"] = rec.get("turn")
            out["game_turn"] = blstats.get("time") or rec.get("lm_turn")
            out["dlvl"] = rec.get("dlvl") or blstats.get("depth")
            out["max_dlvl"] = rec.get("max_dlvl_reached")
            out["xl"] = blstats.get("experience_level")
            out["hp"] = rec.get("hp", blstats.get("hitpoints"))
            out["max_hp"] = rec.get("max_hp", blstats.get("max_hitpoints"))
            out["score"] = blstats.get("score")
            out["tool_calls_last_turn"] = [
                (c.get("name") if isinstance(c, dict) else str(c))
                for c in (rec.get("tool_calls") or [])]
    try:
        names = {p.name for p in checkpoint_list(archive_dir)}
    except Exception:
        names = set()
    out["checkpoints_total"] = len(names)
    out["checkpoints_new"] = sorted(names - set(baseline), key=_id_key_name)
    return out


def _id_key_name(name: str) -> tuple:
    return _id_key(str(name).lstrip("c"))


# --------------------------------------------------------------------------- #
# in-flight spend: what an attempt has cost SO FAR
# --------------------------------------------------------------------------- #
#
# THERE IS NO IN-FLIGHT USAGE SOURCE. This was investigated exhaustively before
# anything below was written, and the answer is negative on every channel:
#
#   * `traces.jsonl` -- the only file carrying per-call `usage` -- is written by
#     `verifiers/v1/cli/output.py:write_trace` when a ROLLOUT COMPLETES, and
#     `save_config` truncates it to zero bytes at launch. An interrupted attempt
#     has an empty one (confirmed: e16_runs/mechcheck/attempts/a001).
#   * The interception server holds usage in memory only
#     (`verifiers/v1/interception/server.py` appends `ModelCall(usage=...)` to
#     `session.trace.calls`, an in-memory list) and exposes no metrics route.
#     `stall_watchdog.py` already documents this: usage "lives only in the
#     interception proxy's memory, which died with the process".
#   * The per-turn NDJSON carries NO token fields at all -- `helpers.py`'s
#     `_write_trace_entry` writes game state and messages; grep for
#     `prompt_tokens|usage|input_tokens` over the harness finds nothing.
#   * `eval.log` (which IS written live) logs only FAILED model calls and
#     rollout start/done -- no per-call usage.
#   * The one format in this tree that does carry live per-message
#     `usage.cost.total` is the agent session file, and BOTH player arms
#     disable it (`--no-session`, `--no-session-persistence`). Turning that on
#     would change what the agent itself does, i.e. change the experiment, and
#     is not something to do to a $385 run for a bookkeeping convenience.
#
# So what follows is AN ESTIMATE, and every field it produces says so. It is a
# spend RATE times elapsed generation time, because wall clock is the one
# quantity that is perfectly observable in flight.
#
# WHY A RATE, AND WHY PER HOUR. Measured over 349 real rollouts in this tree's
# own `outputs/`:
#     $/hour   median 25.7   p10  9.8   p90  82.2   (p90/p10 = 8.4x)
#     $/call   median 0.064  p10  0.026 p90  0.161  (p90/p10 = 6.2x)
#     $/LM-turn: USELESS -- 0.071 in gewiki_pilot vs 1.428 in methodtest, a 20x
#         spread, because calls-per-turn ranges from 1.0 to 9.3.
# Model calls are not observable in flight, so $/call is unusable however tight
# it is. That leaves $/hour. Across ALL experiments its spread is 8.4x, which
# would be useless -- but WITHIN one cell (same arm, same model, same config)
# the max/min ratio is median 2.4x and p90 4.1x over 67 cells with >=4 rollouts.
# That is why the rate is calibrated from THIS RUN's own completed attempts
# whenever there are any, and the cross-experiment prior is used only until the
# first one lands.

# `DEFAULT_SPEND_RATE_USD_PER_HOUR` (90.0) is defined at the top of this module
# because `OrchestratorConfig` defaults to it. It sits just above the measured
# p90 of $82.15/hr: the prior's job is to keep an early runaway from hiding
# behind an optimistic rate, and it is replaced by this run's own data as soon
# as one attempt finishes.

_ROLLOUT_START_RE = re.compile(r"rollout start: id=(\S+)")


def rollouts_started(out_dir) -> int:
    """How many rollouts the eval CLI has STARTED for this attempt, live.

    `eval.log` is written by the eval CLI as it goes (unlike `launch.log`,
    which `SubprocessPlayer` writes in one shot after the process exits), so
    this is the one retry signal available WHILE an attempt is in flight. On
    the methodtest attempt it reads 3, matching the three turn files and the
    two `retrying rollout` warnings. 0 means "not observable", not "none".
    """
    try:
        text = (Path(out_dir) / "eval.log").read_text(errors="replace")
    except OSError:
        return 0
    return len(_ROLLOUT_START_RE.findall(text))


def calibrate_spend_rate(attempts: list, *,
                         prior_usd_per_hour: float = DEFAULT_SPEND_RATE_USD_PER_HOUR
                         ) -> dict:
    """A $/hour spend rate for this run, from its own completed attempts.

    Uses `spend_usd_billed_upper_est` (which accounts for whole-rollout
    retries) over `wall_s`, pooled rather than averaged per attempt -- a pooled
    ratio is not dominated by a 90-second attempt that happened to cost a
    dollar. Attempts shorter than two minutes are dropped entirely: their
    ratio is mostly launch overhead and would bias the rate either way.

    ONLY MEASURED ATTEMPTS COUNT. A row with `spend_known: false` carries an
    estimate produced BY THIS RATE; feeding it back in would calibrate the
    rate against its own output and freeze whatever the prior happened to be.
    """
    spend = 0.0
    wall = 0.0
    n = 0
    for row in attempts or []:
        if not row.get("spend_known", True):
            continue
        try:
            w = float(row.get("wall_s") or 0.0)
            s = float(row.get("spend_usd_billed_upper_est")
                      or row.get("spend_usd") or 0.0)
        except (TypeError, ValueError):
            continue
        if w < 120.0 or s <= 0.0:
            continue
        spend += s
        wall += w
        n += 1
    if n and wall > 0:
        return {"rate_usd_per_hour": round(spend / wall * 3600.0, 4),
                "rate_source": "run_calibrated", "rate_basis_attempts": n}
    return {"rate_usd_per_hour": round(float(prior_usd_per_hour), 4),
            "rate_source": "prior", "rate_basis_attempts": 0}


def estimate_inflight_spend(out_dir, elapsed_s: float, rate: dict, *,
                            safety_factor: float = 2.0) -> dict:
    """`spend_so_far` for the attempt in flight. ALWAYS an estimate.

    Two numbers, deliberately:
      * `spend_so_far_usd` -- the best guess, rate x elapsed. This is what a
        human reading `progress.jsonl` wants.
      * `spend_so_far_upper_usd` -- the same times `safety_factor`. This is
        what the budget guard enforces against, because the two errors are not
        symmetric: over-estimating stops an attempt early and keeps its
        checkpoints, while under-estimating overshoots a hard ceiling, which is
        the failure this exists to prevent (a $9 smoke reached an estimated $22
        with nothing able to notice).

    `safety_factor` defaults to 2.0 against a measured within-cell rate spread
    of 2.4x median / 4.1x p90 -- i.e. it covers the typical attempt and not the
    worst one, which is the honest reading of what a 2x guard buys.
    """
    elapsed = max(0.0, float(elapsed_s or 0.0))
    per_hour = float(rate.get("rate_usd_per_hour") or 0.0)
    best = per_hour * elapsed / 3600.0
    started = rollouts_started(out_dir)
    return {
        "spend_so_far_usd": round(best, 4),
        "spend_so_far_upper_usd": round(best * float(safety_factor), 4),
        # Never let this be mistaken for a measurement. There is no in-flight
        # usage source; see the block comment above.
        "spend_so_far_is_estimate": True,
        "spend_so_far_source": "wall_clock_rate",
        "spend_so_far_rate_usd_per_hour": round(per_hour, 4),
        "spend_so_far_rate_source": rate.get("rate_source"),
        "spend_so_far_rate_basis_attempts": rate.get("rate_basis_attempts", 0),
        "spend_so_far_safety_factor": float(safety_factor),
        # Live retry signal: >1 means the eval CLI is replaying the trajectory
        # and this attempt is billing for more than one rollout.
        "rollouts_started": started,
    }


class AttemptProgressMonitor:
    """A daemon thread that makes a running attempt tail-able.

    NOT a watchdog: it kills nothing and decides nothing. It only writes, which
    is why it is safe to run beside the real stall watchdog. Every sample is
    wrapped -- a monitor that raised would take down the run it was there to
    observe, which is strictly worse than a missing line.
    """

    def __init__(self, *, paths: list, attempt: int, turns_dir, archive_dir,
                 interval_s: float = 15.0, baseline: frozenset = frozenset(),
                 started_at: Optional[float] = None,
                 extra: Optional[dict] = None,
                 spend_fn: Optional[Callable] = None,
                 guard: Optional[Callable] = None):
        import threading
        self.paths = [Path(p) for p in paths]
        self.attempt = attempt
        self.turns_dir = Path(turns_dir)
        self.archive_dir = Path(archive_dir)
        #: `() -> dict` of `spend_so_far_*` fields merged into every sample.
        #: Still not a watchdog: this monitor only WRITES the estimate.
        self.spend_fn = spend_fn
        #: `(record) -> None`, run after the record is written. This is the one
        #: thing in this class that can act, and it is injected rather than
        #: implemented here so the monitor stays a pure observer and the
        #: decision to stop an attempt stays with the orchestrator that owns
        #: the budget. Wrapped like everything else: a raising guard must not
        #: take down the run it is guarding.
        self.guard = guard
        self.interval_s = max(1.0, float(interval_s))
        self.baseline = frozenset(baseline)
        self.started_at = started_at or time.time()
        self.extra = dict(extra or {})
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="e16-progress",
                                        daemon=True)

    # -- writing ----------------------------------------------------------- #

    def emit(self, record: dict) -> None:
        for path in self.paths:
            try:
                _append_jsonl(path, record)
            except Exception:
                pass

    def sample(self, event: str = "progress", **extra) -> dict:
        rec = {"event": event, "attempt": self.attempt, "ts": time.time(),
               "iso": _now_iso(), "wall_s": round(time.time() - self.started_at, 1),
               **self.extra}
        try:
            rec.update(sample_attempt(self.turns_dir, self.archive_dir,
                                      baseline=self.baseline))
        except Exception as exc:
            rec["sample_error"] = f"{type(exc).__name__}: {exc}"
        if self.spend_fn is not None:
            try:
                rec.update(self.spend_fn())
            except Exception as exc:
                rec["spend_sample_error"] = f"{type(exc).__name__}: {exc}"
        rec.update(extra)
        self.samples += 1
        self.emit(rec)
        # The record is on disk BEFORE the guard runs, so the sample that
        # justified a budget stop is durable even if the stop itself explodes.
        if self.guard is not None:
            try:
                self.guard(rec)
            except Exception:
                pass
        return rec

    # -- lifecycle --------------------------------------------------------- #

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def start(self) -> "AttemptProgressMonitor":
        self._thread.start()
        return self

    def stop(self, event: str = "attempt_end", **extra) -> dict:
        self._stop.set()
        try:
            self._thread.join(timeout=5)
        except Exception:
            pass
        return self.sample(event, **extra)


# --------------------------------------------------------------------------- #
# reconciliation: attributing what an interrupted attempt left behind
# --------------------------------------------------------------------------- #

#: How long after an attempt's last measured byte a checkpoint may still be
#: attributed to it. A rollout's final `save` lands after its final turn
#: record, and an interrupted attempt's "last byte" is whatever the kill left
#: behind -- so the window is held open past it. It is never held open into the
#: NEXT attempt's window: two attempts must never be able to claim one
#: checkpoint, which is the double-counting this whole path exists to prevent.
ATTRIBUTION_SLACK_S = 300.0


def _pid_alive(pid: int) -> bool:
    """True only for a RUNNING process; a zombie counts as dead."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            raw = fh.read()
        return raw[raw.rindex(")") + 2:].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def _pid_start_time(pid: int) -> Optional[float]:
    """Wall-clock start time of `pid`, or None. Guards against PID REUSE.

    A stale lock file naming a pid the kernel has since handed to something
    else must not read as "the run is live" -- that would make a crashed run
    permanently unrecoverable. The pid AND its start time have to match.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            raw = fh.read()
        tail = raw[raw.rindex(")") + 2:].split()
        with open("/proc/stat") as fh:
            btime = next(float(l.split()[1]) for l in fh if l.startswith("btime "))
        return btime + float(tail[19]) / (os.sysconf("SC_CLK_TCK") or 100)
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def run_lock_path(run_dir) -> Path:
    return Path(run_dir) / "orchestrator.pid"


def write_run_lock(run_dir) -> dict:
    """Stamp the run directory with the pid that owns it, and when it started."""
    pid = os.getpid()
    rec = {"pid": pid, "start_time": _pid_start_time(pid), "iso": _now_iso(),
           "argv": list(sys.argv)}
    try:
        atomic_write(run_lock_path(run_dir),
                     json.dumps(rec, indent=2, sort_keys=True) + "\n")
    except Exception:
        pass
    return rec


def clear_run_lock(run_dir) -> None:
    try:
        run_lock_path(run_dir).unlink()
    except OSError:
        pass


def read_run_lock(run_dir) -> Optional[int]:
    """The pid the lock names, if that exact process is still running."""
    try:
        rec = json.loads(run_lock_path(run_dir).read_text())
    except Exception:
        return None
    pid = rec.get("pid")
    if not isinstance(pid, int) or pid == os.getpid() or not _pid_alive(pid):
        return None
    want = rec.get("start_time")
    got = _pid_start_time(pid)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)) \
            and abs(want - got) > 2.0:
        return None      # the pid was recycled; the run is dead after all
    return pid


def live_orchestrator_pids(run_dir) -> list:
    """PIDs of orchestrator processes that currently hold ``run_dir``.

    THE GUARD THAT KEEPS RECOVERY FROM BECOMING THE CORRUPTION. Reconciliation
    finalizes the in-flight attempt and stamps ownership onto archive
    checkpoints; doing that to a run that is still playing would write a
    `censored` row for a live attempt and hand its future checkpoints to it.
    On a shared box this is not hypothetical -- the mechcheck run that
    motivated all of this was still alive, still writing, and looked dead from
    its file timestamps alone.

    TWO INDEPENDENT SIGNALS, unioned, because either alone has a blind spot:
    the lock file this code writes (which a run predating it does not have),
    and a scan of every process's argv for this run directory (which misses a
    holder launched through a wrapper that does not name it). A false positive
    costs a wait; a false negative costs the run.
    """
    want = str(Path(run_dir).resolve())
    out = []
    locked = read_run_lock(run_dir)
    if locked is not None:
        out.append(locked)
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    # OUR OWN ANCESTORS ARE NOT HOLDERS. `timeout`, `nohup` and every shell
    # wrapper re-expose the wrapped command line as their own argv, so a plain
    # `timeout 300 e16_orchestrator.py RUN --reconcile` would otherwise report
    # its own launcher as a live orchestrator and refuse to do anything.
    skip = set()
    anc = os.getpid()
    for _ in range(64):
        skip.add(anc)
        try:
            with open(f"/proc/{anc}/stat") as fh:
                raw = fh.read()
            anc = int(raw[raw.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if anc <= 1:
            break
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in skip:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        parts = [a.decode(errors="replace") for a in argv if a]
        if not any("e16_orchestrator" in p for p in parts):
            continue
        if any(os.path.realpath(p) == want for p in parts if p.startswith("/")):
            out.append(pid)
    return sorted(set(out))


def read_jsonl(path) -> list:
    out = []
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line survives as a skipped line, never a crash
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _attempt_dir_evidence(out_dir: Path) -> dict:
    """What an attempt directory proves about itself with no journal at all.

    The recovery path for a run that predates the journal -- mechcheck is
    exactly this case: 24 checkpoints, an `attempts/a001/` full of served bytes
    and turn records, and nothing that says an attempt ever existed.
    """
    ev: dict = {"out_dir": str(out_dir), "started_at": None, "ended_at": None,
                "directive": "", "turns": 0, "calls": 0,
                "max_dlvl": 0, "max_xl": 1, "game_turn": 0}
    try:
        served = out_dir / "directive_served.txt"
        if served.is_file():
            ev["directive"] = served.read_text(errors="replace").strip()
            ev["started_at"] = served.stat().st_mtime
    except OSError:
        pass
    files = all_turn_files(out_dir)
    if files:
        try:
            ev["started_at"] = min([ev["started_at"] or files[0].stat().st_mtime,
                                    files[0].stat().st_mtime])
            ev["ended_at"] = max(p.stat().st_mtime for p in files)
        except OSError:
            pass
        rec = tail_turn_record(files[-1])
        if rec:
            ev["turns"] = int(rec.get("turn") or 0)
            blstats = _turn_blstats(rec)
            ev["game_turn"] = int(blstats.get("time") or 0)
            ev["max_xl"] = int(blstats.get("experience_level")
                               or rec.get("max_xp_level") or 1)
    ev["calls"] = len(calls_from_turns(out_dir))
    ev["max_dlvl"] = int(_max_dlvl_from_turns(out_dir) or 0)
    if ev["ended_at"] is None:
        try:
            ev["ended_at"] = out_dir.stat().st_mtime
        except OSError:
            ev["ended_at"] = ev["started_at"]
    return ev


#: What `nethack_v1.arm_crash_capture()` names the file it arms faulthandler
#: on, and the string CPython's faulthandler prints on a fatal native signal.
#: Duplicated rather than imported: the orchestrator runs in a process that
#: does not (and must not) import the env package.
ENV_CRASH_LOG_NAME = "env_crash.log"
ENV_CRASH_MARKER = "Fatal Python error"

#: How quiet a progress sample has to be before it stops counting as "the
#: attempt was still playing here". Well under the stall watchdog's 300 s so a
#: normal slow turn still counts, well over a sampling interval so one late
#: sample does not.
PROGRESS_IDLE_FLOOR_S = 120.0


def env_crash_evidence(out_dir) -> str:
    """The env tool server's dying words, or "" if it did not die that way.

    The env drives NetHack through a C extension and a SIGSEGV/SIGABRT in
    there kills the tool server with no Python traceback, no log line, and no
    "tool server down" counterpart to the launcher's startup line -- while its
    merged stdout/stderr is deleted with the runtime workdir by the same
    teardown that failed to notice (`v1/runtimes/subprocess.py:cleanup`).
    `nethack_v1.arm_crash_capture()` redirects both dumpers into the ATTEMPT
    directory, which outlives the runtime; this reads whichever fired.

    TWO CHANNELS, and the ENGINE one is the load-bearing half. `libnethack.so`
    installs its own SIGSEGV/SIGABRT/SIGFPE/SIGBUS handlers when it loads, so
    for a fault inside the engine the sentinel wins and CPython's faulthandler
    never runs: the native backtrace in `nle_crash_<pid>.txt` is the only
    record. faulthandler's `env_crash.log` covers the other case, a fatal
    signal on the Python side of the server.
    """
    out_dir = Path(out_dir)
    try:
        dumps = sorted(out_dir.glob("nle_crash_*.txt"),
                       key=lambda p: p.stat().st_mtime)
    except OSError:
        dumps = []
    if dumps:
        text = dumps[-1].read_text(errors="replace")
        head = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
        # e.g. "=== NLE SENTINEL: SIGSEGV === | pid=... | FAULTING ENV: ..."
        return (" | ".join(head) or dumps[-1].name)[:500]
    path = out_dir / ENV_CRASH_LOG_NAME
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        if ENV_CRASH_MARKER in line:
            frame = next((ln.strip() for ln in lines[i + 1:] if ln.strip()), "")
            return (line.strip() + (f" | {frame}" if frame else ""))[:500]
    return ""


def progress_spend_estimate(out_dir) -> dict:
    """The last in-flight spend ESTIMATE this attempt published, if any.

    `AttemptProgressMonitor` writes `spend_so_far_usd` (and its upper twin, and
    the rate it used) into `<attempt>/progress.jsonl` every sampling interval,
    precisely because mid-attempt spend is otherwise unknowable. Until now
    nothing ever read it back: when the rollout failed to report, the attempt
    was booked at $0.00 and the estimate died in the log.

    THE SAMPLE IS TAKEN WHERE PLAY STOPPED, NOT WHERE THE ATTEMPT DID. The
    estimate is wall-clock * a $/hour rate, and a crashed attempt's wall clock
    runs on long after the game process is gone: the stall watchdog needs its
    full 300 s of silence before it will act, and no tokens are bought in that
    time. Charging it anyway inflates the lower bound by minutes of nothing --
    and because the bound is booked against a HARD ceiling, an inflated one
    shortens every later attempt through the in-flight guard. So take the last
    sample the attempt was still moving in (`idle_s` under the stall floor),
    and fall back to the final sample only if no sample was ever live.

    Returns `{}` when there is nothing to read. Every key keeps its
    `spend_so_far_*` name so it can never be mistaken for a measurement.
    """
    keys = ("spend_so_far_usd", "spend_so_far_upper_usd",
            "spend_so_far_source", "spend_so_far_is_estimate",
            "spend_so_far_rate_usd_per_hour", "spend_so_far_rate_source",
            "spend_so_far_safety_factor")
    rows = [r for r in read_jsonl(Path(out_dir) / "progress.jsonl")
            if isinstance(r.get("spend_so_far_usd"), (int, float))]
    if not rows:
        return {}
    live = [r for r in rows
            if isinstance(r.get("idle_s"), (int, float))
            and r["idle_s"] < PROGRESS_IDLE_FLOOR_S]
    rec = live[-1] if live else rows[-1]
    out = {k: rec[k] for k in keys if k in rec}
    out["spend_so_far_sampled_at_wall_s"] = rec.get("wall_s")
    out["spend_so_far_sampled_while_playing"] = bool(live)
    return out


def _partial_spend(out_dir: Path) -> tuple:
    """``(usd, known)`` for an interrupted attempt.

    `traces.jsonl` is written when the ROLLOUT finishes, so an interrupted
    attempt usually has a zero-byte one and there is no honest number to put
    on the row. Reporting 0.0 as if measured would understate the run's spend
    against a hard ceiling, so the row carries `spend_known: false` and the
    summary carries the count of such rows. A budget that silently forgets
    money it spent is the same class of bug as an attempt that vanishes.
    """
    try:
        res = read_trace_result(out_dir)
        if res.spend_usd:
            return float(res.spend_usd), True
    except Exception:
        pass
    return 0.0, False


def interrupted_record(*, attempt: int, opened: dict, evidence: dict,
                       reason: str, detail: str = "") -> dict:
    """A FINAL attempt row built from an interrupted attempt's own leftovers.

    Same shape as `Orchestrator.ingest`'s row -- same keys, same types -- so
    every existing reader works on it unchanged, and three extra keys say it
    was recovered rather than reported: `recovered`, `spend_known`,
    `recovery_detail`.
    """
    spend, spend_known = _partial_spend(Path(evidence["out_dir"]))
    started = opened.get("ts") or evidence.get("started_at") or 0.0
    ended = evidence.get("ended_at") or started
    # The journal's copy first, then the bytes actually SERVED to the player.
    # `directive_served.txt` is the last-resort source and the most literal one:
    # it is what the player read, whatever anyone else recorded.
    directive = opened.get("directive") or evidence.get("directive") or ""
    return {
        "attempt": attempt,
        "from_checkpoint": opened.get("from_checkpoint"),
        "directive": directive,
        "directive_compliance": {"class": COMPLY_UNKNOWN,
                                 "reason": "attempt interrupted before it "
                                           "reported; not scored"},
        "experiment_arm": opened.get("experiment_arm") or ARM_GO_EXPLORE,
        "pair_id": opened.get("pair_id"),
        "pair_role": opened.get("pair_role") or ROLE_SOLO,
        "directive_kind": opened.get("directive_kind")
                          or classify_directive_kind(directive),
        "directive_lint": lint_directive(directive),
        "reseed": opened.get("reseed"),
        "selection_source": opened.get("selection_source"),
        "orchestrator_decision": None,
        "outcome": OUTCOME_CENSORED,
        "censored": True,
        "censor_reason": CENSOR_INTERRUPTED,
        "stop_condition": "orchestrator_interrupted",
        "error": detail or reason,
        "calls": int(evidence.get("calls") or 0),
        "spend_usd": round(spend, 6),
        "spend_known": spend_known,
        "wall_s": round(max(0.0, float(ended) - float(started)), 2),
        "max_dlvl": int(evidence.get("max_dlvl") or 0),
        "depth_disagreement": None,
        "max_xl": int(evidence.get("max_xl") or 1),
        "new_checkpoints": list(evidence.get("new_checkpoints") or []),
        "frontier_advanced": None,
        "attempts_since_frontier_advance": None,
        "cumulative_calls": None,
        "cumulative_spend_usd": None,
        "cumulative_wall_s": None,
        "model_text": {"summary": "", "lesson": ""},
        "out_dir": evidence["out_dir"],
        "recovered": True,
        "recovery_detail": detail or reason,
        "lm_turns_written": int(evidence.get("turns") or 0),
        "game_turn_reached": int(evidence.get("game_turn") or 0),
    }


class RunStillLive(RuntimeError):
    """Refusing to reconcile a run directory another orchestrator still holds."""


def reconcile_run(cfg: OrchestratorConfig, *, apply: bool = True,
                  force: bool = False) -> dict:
    """Reconcile the archive and the attempt records. Idempotent.

    Three questions, answered in order, from files only -- no session, no
    model, no spend:

      1. Which attempts were left in flight? (a journal `open` with no `close`,
         or an `attempts/aNNN/` directory with no row at all.) Each is finalized
         as `censored:interrupted` and appended to `attempts.jsonl`.
      2. Which archive checkpoints belong to no attempt? Each is attributed to
         the attempt whose [start, end] window contains its harness-written
         `created_at`, and its `meta.json` is stamped `created_in_attempt` /
         `attributed_by: recovery`.
      3. What is left over? REPORTED, in `unattributed_checkpoints`, and never
         silently absorbed -- a checkpoint nobody can attribute is a fact about
         the run, and the one place it can be noticed is here.
    """
    live = live_orchestrator_pids(cfg.run_dir)
    if live and not force:
        raise RunStillLive(
            f"REFUSING to reconcile {cfg.run_dir}: orchestrator pid(s) "
            f"{live} are still holding it. Reconciliation finalizes the "
            f"in-flight attempt and stamps ownership onto archive "
            f"checkpoints; doing that under a live run would write a "
            f"`censored` row for an attempt that is still playing and hand "
            f"its future checkpoints to it. Wait for the run to end (or stop "
            f"it) and reconcile then -- it is idempotent, so nothing is lost "
            f"by waiting.")

    rows = read_jsonl(cfg.attempts_path)
    finalized = {int(r["attempt"]) for r in rows if isinstance(r.get("attempt"), int)}
    events = read_jsonl(cfg.journal_path)
    opened: dict = {}
    closed: set = set()
    for ev in events:
        n = ev.get("attempt")
        if not isinstance(n, int):
            continue
        if ev.get("event") == "open":
            opened[n] = ev
        elif ev.get("event") == "close":
            closed.add(n)

    # (1) in-flight attempts, from the journal and then from the directories.
    interrupted = sorted(n for n in opened if n not in closed and n not in finalized)
    dirs = {}
    attempts_root = cfg.run_dir / "attempts"
    if attempts_root.is_dir():
        for p in sorted(attempts_root.glob("a[0-9]*")):
            try:
                dirs[int(p.name.lstrip("a"))] = p
            except ValueError:
                continue
    # SELECTION.JSONL IS ALSO A PRE-LAUNCH RECORD. It is appended in
    # `run_attempt` before the launch and names the checkpoint chosen and the
    # directive written for each attempt -- so a run that predates the journal
    # still has an authoritative source for both. Without it a recovered
    # mechcheck row came back with `from_checkpoint: null`, which would have
    # left 23 checkpoints attributed to an attempt that started from nowhere.
    selections = {}
    for s in read_jsonl(cfg.selection_path):
        if isinstance(s.get("attempt"), int):
            selections[s["attempt"]] = s
    for n in sorted(dirs):
        if n not in finalized and n not in opened:
            # No journal at all -- a run from before the journal existed, or one
            # killed between mkdir and the first append. The directory is still
            # evidence, and refusing to read it is how mechcheck lost an attempt.
            opened[n] = {"attempt": n, "event": "open", "recovered_from": "dir",
                         "out_dir": str(dirs[n]), "ts": None}
            interrupted.append(n)
    # Fill every gap in an `open` record from the selection log. Journal first,
    # selection second: the journal is what the launch actually used.
    for n, op in opened.items():
        sel = selections.get(n)
        if not sel:
            continue
        if not op.get("from_checkpoint"):
            op["from_checkpoint"] = sel.get("chosen_id")
        if not op.get("directive"):
            op["directive"] = sel.get("directive") or ""
        if not op.get("directive_kind"):
            op["directive_kind"] = sel.get("directive_kind") or ""
        if not op.get("selection_source"):
            op["selection_source"] = sel.get("source")
    interrupted = sorted(set(interrupted))
    # An attempt whose recorded pid is STILL RUNNING is not interrupted, it is
    # in flight. Its window is still used for attribution (so its checkpoints
    # are not handed to the attempt before it) but no row is written for it:
    # a `censored` row for a live attempt is a lie the dataset would keep.
    still_live: list = []
    for n in list(interrupted):
        p = opened[n].get("pid")
        if isinstance(p, int) and _pid_alive(p) and not force:
            still_live.append({"attempt": n, "pid": p})
            interrupted.remove(n)
    open_attempts = sorted(set(interrupted) | {s["attempt"] for s in still_live})

    new_rows: list = []
    windows: list = []  # (attempt, start_ts, end_ts, from_checkpoint)
    for r in rows:
        n = r.get("attempt")
        od = Path(r.get("out_dir") or (attempts_root / f"a{n:03d}"))
        ev = _attempt_dir_evidence(od) if od.is_dir() else {}
        windows.append((n, ev.get("started_at"), ev.get("ended_at"),
                        r.get("from_checkpoint"), r))
    for n in open_attempts:
        op = opened[n]
        od = Path(op.get("out_dir") or (attempts_root / f"a{n:03d}"))
        ev = _attempt_dir_evidence(od)
        start = op.get("ts") or ev.get("started_at")
        windows.append((n, start, ev.get("ended_at"), op.get("from_checkpoint"),
                        None))

    # (2) orphaned checkpoints -> the attempt whose window contains them.
    seed_ids = set()
    orphans: list = []
    for path in checkpoint_list(cfg.archive_dir):
        try:
            meta = checkpoint_meta(path)
        except Exception:
            continue
        if meta.get("created_by") == "orchestrator":
            # The seed state, written by `seed_archive` before any attempt
            # existed. It is not an orphan and must never be attributed to one.
            seed_ids.add(path.name)
            continue
        if meta.get("created_in_attempt"):
            continue
        orphans.append((path, meta))

    # NON-OVERLAPPING WINDOWS, in start order. Attempts run strictly one after
    # another, so an attempt's window closes where the next one opens -- and
    # without that clamp the slack below lets a finished attempt swallow the
    # next attempt's checkpoints, which is a worse failure than not attributing
    # them at all (it would put one attempt's evidence on another's row).
    windows = [w for w in windows if w[1] is not None]
    windows.sort(key=lambda w: float(w[1]))
    clamped: list = []
    for i, (n, start, end, parent, _row) in enumerate(windows):
        end = float(end or start)
        # A checkpoint can land after the attempt's last TURN record -- the
        # save is the last thing a dying rollout does -- so the window is held
        # open past it, but never into the next attempt.
        end = max(end, float(start)) + ATTRIBUTION_SLACK_S
        if i + 1 < len(windows):
            end = min(end, float(windows[i + 1][1]))
        clamped.append((n, float(start), end, parent))

    attributed: list = []
    unattributed: list = []
    held_by_live: list = []
    per_attempt_new: dict = {}
    #: attempt -> [(path, meta, the id that attempt resumed)]. Filled in the
    #: loop, stamped after it, because `parent_chain` needs the whole set.
    to_stamp: dict = {}
    for path, meta in orphans:
        created = meta.get("created_at")
        if not isinstance(created, (int, float)):
            try:
                created = path.stat().st_mtime
            except OSError:
                created = None
        owner = None
        if created is not None:
            # The LAST attempt that had started when this checkpoint was
            # written. Reading it that way rather than as "the first window
            # that matches" is what keeps the answer stable when two windows'
            # measured ends disagree by a second.
            for n, start, end, parent in clamped:
                if start - 1.0 <= float(created) <= end:
                    owner = (n, parent)
        if owner is None:
            unattributed.append({"checkpoint": path.name, "created_at": created,
                                 "created_by": meta.get("created_by"),
                                 "why": "no attempt window contains its created_at"})
            continue
        n, parent = owner
        if any(s["attempt"] == n for s in still_live):
            # It belongs to an attempt that is STILL PLAYING. `ingest` will
            # stamp it when that attempt finishes; stamping it here would race
            # a live writer for no benefit.
            held_by_live.append({"checkpoint": path.name, "attempt": n})
            continue
        per_attempt_new.setdefault(n, []).append(path.name)
        attributed.append({"checkpoint": path.name, "attempt": n,
                           "created_at": created})
        # STAMPED PER ATTEMPT, BELOW, not here. The lineage of a checkpoint
        # depends on the OTHER checkpoints the same attempt wrote, so it cannot
        # be decided one orphan at a time -- doing it that way is what made
        # every recovered archive a star.
        to_stamp.setdefault(n, []).append((path, meta, parent))

    if apply:
        for n, items in to_stamp.items():
            resume_id = next((par for _p, _m, par in items if par), None)
            parents = dict(parent_chain([p for p, _m, _par in items],
                                        resume_id))
            for path, meta, _par in items:
                meta["created_in_attempt"] = n
                if meta.get("parent") is None and parents.get(path):
                    meta["parent"] = str(parents[path])
                meta["attributed_by"] = "recovery"
                atomic_write(path / META_JSON,
                             json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # (1b) now that attribution is known, write the finalized rows.
    for n in interrupted:
        op = opened[n]
        od = Path(op.get("out_dir") or (attempts_root / f"a{n:03d}"))
        ev = _attempt_dir_evidence(od)
        ev["new_checkpoints"] = sorted(per_attempt_new.get(n, []), key=_id_key_name)
        detail = op.get("interrupt_reason") or (
            "the orchestrator process ended while this attempt was in flight; "
            "the row was reconstructed from the attempt's own turn records and "
            "the archive checkpoints inside its window")
        rec = interrupted_record(attempt=n, opened=op, evidence=ev,
                                 reason=CENSOR_INTERRUPTED, detail=detail)
        new_rows.append(rec)
        if apply:
            _append_jsonl(cfg.attempts_path, rec)
            _append_jsonl(cfg.journal_path,
                          {"event": "close", "attempt": n, "ts": time.time(),
                           "iso": _now_iso(), "outcome": OUTCOME_CENSORED,
                           "censor_reason": CENSOR_INTERRUPTED,
                           "recovered": True})

    # An already-final row that gained checkpoints (the crash landed between
    # the archive write and the row) has them added to its recovery note rather
    # than to the row: attempts.jsonl stays append-only.
    late = {n: names for n, names in per_attempt_new.items()
            if n in finalized}

    report = {
        "ts": time.time(), "iso": _now_iso(), "run_dir": str(cfg.run_dir),
        "applied": bool(apply),
        "forced": bool(force),
        "live_orchestrator_pids": live,
        "attempts_still_live": still_live,
        "checkpoints_held_by_live_attempts": held_by_live,
        "attempts_already_final": sorted(finalized),
        "attempts_finalized_now": [r["attempt"] for r in new_rows],
        "checkpoints_total": len(checkpoint_list(cfg.archive_dir)),
        "seed_checkpoints": sorted(seed_ids, key=_id_key_name),
        "checkpoints_attributed": attributed,
        "checkpoints_attributed_to_already_final_attempts": late,
        "unattributed_checkpoints": unattributed,
        "spend_unknown_attempts": [r["attempt"] for r in new_rows
                                   if not r.get("spend_known")],
    }
    if apply:
        atomic_write(cfg.recovery_path,
                     json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def _git_commit(repo: Path) -> str:
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        if rev.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=15).stdout.strip()
        sha = rev.stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _submodule_commit(repo: Path, path: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "submodule", "status", path],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        return out.split()[0].lstrip("+-U") if out else "unknown"
    except Exception:
        return "unknown"


def build_provenance(cfg: OrchestratorConfig, wiki_hashes: dict,
                     engine_repo: Optional[Path] = None) -> dict:
    """Everything needed to say what this run WAS, written before it starts."""
    from tool_tiers import contract_for, registry_hash  # local: same dir

    engine_repo = Path(engine_repo or os.environ.get("ENG", "/root/NetHack-engine"))
    return {
        "experiment": "E16_go_explore",
        "run_dir": str(cfg.run_dir),
        "started_at": time.time(),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier": cfg.tier,
        "tier_registry_sha256_16": registry_hash(),
        "tier_contract": contract_for(cfg.tier),
        # TWO ARMS UNDER TWO NAMES, because they are two different things and
        # one name for both is how an arm gets misreported.
        "arm": cfg.arm,
        "experiment_arm": cfg.arm,
        "player_arm": cfg.player_arm,
        "experiment_arm_meaning": (
            "matched_restart = THE NULL: N independent attempts from one fixed "
            "start state, no archive, no selection, no directive, no lessons. "
            "go_explore = the method." if cfg.arm == ARM_MATCHED_RESTART else
            "go_explore = the method (archive, selection, directives, "
            "lessons). Its null is --arm matched_restart at the same N."),
        "start_checkpoint_id": cfg.start_checkpoint_id or "(archive seed)",
        "game_seed": cfg.game_seed,
        # RESEED: which of the two stochastic semantics this run used. Recorded
        # because they are not the same experiment. Default OFF: a restore
        # resets the engine to the game's original seed and replays no history,
        # so two restores of one checkpoint continue byte-identically and the
        # difference between two attempts comes from the model and the
        # directive rather than from dice -- which is what makes a directive's
        # effect attributable. ON makes restores genuinely Monte-Carlo, at the
        # cost of that attributability.
        "reseed_on_restore": cfg.reseed_on_restore,
        "reseed_semantics": (
            "restore re-seeds the RNG from (sha256(rng_seed:checkpoint:attempt)) "
            "AFTER load, so two attempts from one checkpoint diverge"
            if cfg.reseed_on_restore else
            "DETERMINISTIC (default): restore resets to the game's original "
            "seed and replays no history. Two restores of one checkpoint "
            "continue byte-identically; the dungeon below a checkpoint is "
            "regenerated from a stream restarted at position 0, so depths here "
            "are NOT on the same world an uninterrupted seed-1 run traverses. "
            "Per-checkpoint outcome spread reflects policy variation only and "
            "understates real NetHack luck."),
        "commits": {
            "zombie_fix": _git_commit(REPO),
            "engine": _git_commit(engine_repo),
            "engine_submodule_third_party_NetHack":
                _submodule_commit(engine_repo, "third_party/NetHack"),
        },
        "wiki": {"source": str(cfg.wiki_src), "dest": str(cfg.wiki_dir),
                 "sha256": wiki_hashes},
        "selector": {
            "mode": cfg.selector,
            "kind": ("llm_persistent_session (scripted softmax is the fallback "
                     "and the ablation)" if cfg.selector == "llm"
                     else "scripted_softmax_over_pareto_frontier"),
            "rng_seed": cfg.rng_seed,
            "weights": {"balrog_min": cfg.w_balrog, "novelty": cfg.w_novelty,
                        "attempts_from": cfg.w_attempts},
            "temperature": cfg.temperature,
            "normalization": "min-max over the candidate set; see _minmax",
        },
        "budget": {"ceiling_usd": cfg.budget_ceiling_usd,
                   "min_headroom_usd": cfg.min_headroom_usd},
        "stop_conditions": {"max_attempts": cfg.max_attempts,
                            "stall_attempts": cfg.stall_attempts,
                            "milestone_dlvl": cfg.milestone_dlvl,
                            "milestone_dungeon": cfg.milestone_dungeon},
        "directive": {
            "mode": ("none (run-wide control)" if cfg.no_directive
                     else "per-attempt"),
            "paired_control": cfg.paired_control,
            "paired_control_meaning": (
                "every directive is ALSO run from the same checkpoint without "
                "it, so each directive KIND has its own control. A run-wide "
                "--no-directive average cannot separate a directive that "
                "changed behaviour from one that prohibited something the "
                "player was not going to do anyway."),
            "served_as": "first observation, "
                         "[ORCHESTRATOR DIRECTIVE for this attempt: ...]",
            "compliance_rubric": RUBRIC_NAME,
            "compliance_rubric_detail": (
                f"clauses split into prohibition/goal; each scored "
                f"satisfied/violated/no-evidence over the first "
                f"{DIRECTIVE_WINDOW} tool calls, and against harness metrics "
                f"where the clause names a measured outcome (descent). The "
                f"single `class` field is DERIVED from the two per-kind labels, "
                f"never measured directly. A prohibition over a TOOL is "
                f"lint-warned as unscoreable and the instrumental-action "
                f"ambiguity is recorded in each attempt's rationale."),
        },
        "orchestrator_session": {
            "kind": "prime_agent --print [--mode json on the discovery round "
                    "only] --resume <id>, dedicated "
                    "PRIME_AGENT_CODING_AGENT_DIR, private TMPDIR, asserted "
                    "fixed cwd, hard per-round timeout",
            "agent_dir": str(cfg.orchestrator_agent_dir or
                             (cfg.orchestrator_dir / "agent")),
            # THE THREE HANGS, each named with what was done about it, because
            # a mitigation nobody can find in the record is a mitigation nobody
            # can check.
            "cwd": cfg.orchestrator_cwd,
            "cwd_policy": ("recorded at round 0 and asserted before every "
                           "resumed launch; a mismatch raises before any "
                           "subprocess exists, because --resume from a foreign "
                           "cwd hangs on an interactive fork confirm"),
            "tmpdir": str(cfg.orchestrator_tmpdir or ""),
            "tmpdir_policy": ("private, set on the orchestrator's own env dict "
                              "and never exported; the shared "
                              "/tmp/prime-agent-0/daemon.sock was measured at "
                              "900s of hang against 3.5s with a private one"),
            "tmpdir_note": cfg.orchestrator_tmpdir_note,
            "json_mode_policy": cfg.orchestrator_json_mode,
            "json_mode_policy_why": ("--mode json + --resume was measured "
                                     "hanging where the identical text --print "
                                     "call succeeded 30s later, so json mode "
                                     "runs only on the round that has no "
                                     "session id to resume; the id, cwd and "
                                     "per-round usage come from the session "
                                     "file on every later round"),
            "round_timeout_s": cfg.orchestrator_timeout_s,
            "ids": [],  # filled in as rounds happen
            # NEVER assumed: set by the two-call probe, or left None with the
            # run saying it did not check.
            "session_resume_verified": cfg.session_resume_verified,
            "context_bound": {
                "player_summary_cap_chars": cfg.orchestrator_summary_cap,
                "context_limit_tokens": cfg.orchestrator_context_limit_tokens,
                # WHOSE COMPACTION. Ours is gone; this says what the CLI's was
                # set to, and says so even when it could not be set.
                "compaction": cfg.orchestrator_compaction,
            },
        },
        # Honest statements about what this run is NOT.
        "comparable_to_single_life_balrog": False,
        "prefix_continuity": "text_only",
        # The distinction that decides whether H4 is testable, kept in the
        # record so nobody has to re-derive it. "text_only" is about the
        # MECHANISM, not about delivery: the replayed prefix was measured
        # arriving byte-identically in the first served observation of a real
        # resumed rollout (outputs/e16_launchfix/). What is out of reach is
        # continuity of SESSION -- an inherited conversation and its prompt
        # cache -- not continuity of CONTENT.
        "prefix_continuity_detail": (
            "the checkpoint's conversation is REPLAYED as quoted text in the "
            "first observation, and that text was verified present in the "
            "served bytes. H4 is therefore testable as 'lessons + prefix "
            "CONTENT transfer'; it is NOT a test of session continuity, and "
            "any claim about inherited conversation state would be false."),
        "prefix_continuity_verified_in_served_bytes": True,
        "metrics_authored_by": "harness (engine blstats at save time)",
        "orchestrator_text_authored_by": "the orchestrator LM -- displayed and "
                                         "recorded, never measured",
    }


# --------------------------------------------------------------------------- #
# the orchestrator
# --------------------------------------------------------------------------- #

class Orchestrator:
    """The select -> launch -> ingest loop, with all its bookkeeping."""

    def __init__(self, cfg: OrchestratorConfig, launcher: Launcher,
                 *, budget: Optional[Budget] = None, session=None):
        self.cfg = cfg
        self.launcher = launcher
        self.budget = budget or Budget(cfg.budget_ceiling_usd,
                                       cfg.min_headroom_usd)
        self.rng = random.Random(cfg.rng_seed)
        #: The orchestrator's own conversation. ``None`` in scripted mode --
        #: which is exactly what makes the ablation an ablation: same loop,
        #: same archive, same records, no LM in the decision.
        self.session = session
        self.attempts: list = []
        self.frontier_advances: list = []
        self.cum_calls = 0
        self.cum_wall = 0.0
        self.stop_reason: Optional[str] = None
        self.started_at = time.time()
        self._since_advance = 0
        self.opened = False
        self.llm_fallbacks = 0
        self._pair_counter = 0
        #: Per-attempt verdicts from the opening round's degeneration check.
        self.opening_attempts: list = []
        #: Set when an :class:`OrchestratorRoundFailed` ended the run. It goes
        #: into the summary before the exception is re-raised, so the run
        #: directory names its own cause of death.
        self.orchestrator_error: str = ""
        #: Phrasing warnings raised on the last directive, fed back into the
        #: next orchestrator round. A lint nobody reads changes nothing.
        self._last_lint: list = []
        #: THE IN-FLIGHT ATTEMPT, or None. Set from the journal `open` line
        #: written BEFORE the player launches and cleared only when the row is
        #: final. Anything that unwinds the stack -- an exception, a signal,
        #: `main`'s last-resort handler -- finalizes whatever is in here, so an
        #: attempt can no longer leave checkpoints in the archive with no row.
        self._open: Optional[dict] = None
        #: The live progress sampler for the in-flight attempt.
        self._monitor: Optional[AttemptProgressMonitor] = None
        #: Set by `_inflight_budget_guard` when it stops an attempt mid-flight;
        #: read (and cleared) by `_launch_one` so the row is labelled
        #: `censored:budget_stop` rather than looking like a harness error.
        self._budget_stop: Optional[dict] = None
        self._spend_rate: dict = {}
        #: What `reconcile_run` found at startup. Carried into the summary so a
        #: recovered run says out loud that it was recovered.
        self.recovery: dict = {}
        self.resumed = False
        #: Set by the signal handlers; named on the finalized attempt row.
        self.interrupted_by: str = ""
        #: True while `run_decide_only` is driving. Everything it writes says
        #: so, and this is what puts it in the summary.
        self.decide_only = False

    # -- setup ------------------------------------------------------------- #

    def prepare(self) -> dict:
        """Create the run tree, copy the wiki in, write provenance.json.

        The wiki is copied rather than referenced, per the design: the pages
        the players and the orchestrator read are the run's OWN copy, so the
        served knowledge is recoverable byte-for-byte from the output tree even
        if the source worktree moves or is edited mid-run. Their hashes go into
        provenance.
        """
        from nethack_harness.wiki_kb import install as install_wiki

        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.archive_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
        # THE CWD, RECORDED AT ROUND 0. Everything after this asserts against
        # it. A --resume from anywhere else takes prime-agent's "fork this
        # session into current directory?" branch and hangs a non-interactive
        # driver forever -- which is a silent $700 failure, not an error.
        self.cfg.orchestrator_cwd = str(self.cfg.orchestrator_dir.resolve())
        if self.cfg.orchestrator_tmpdir is None:
            self.cfg.orchestrator_tmpdir = (
                Path(self.cfg.orchestrator_agent_dir
                     or (self.cfg.orchestrator_dir / "agent")) / "tmp")
        Path(self.cfg.orchestrator_tmpdir).mkdir(parents=True, exist_ok=True)
        # WHO OWNS THIS DIRECTORY, written down. Recovery and resume both refuse
        # to touch a run another orchestrator is still holding, and the only way
        # to know that reliably is for the holder to say so.
        write_run_lock(self.cfg.run_dir)
        hashes = install_wiki(self.cfg.wiki_src, self.cfg.wiki_dir)
        # The orchestrator reads the SAME bytes the players are served, through
        # a tiny CLI over the same WikiKB. Two knowledge bases with the same
        # name would be a confound wearing a disguise.
        write_orchestrator_wiki_tool(self.cfg)
        prov = build_provenance(self.cfg, hashes)
        atomic_write(self.cfg.provenance_path,
                     json.dumps(prov, indent=2, sort_keys=True) + "\n")
        return prov

    def finish(self) -> dict:
        """Fold the orchestrator's session ids into provenance and re-write it."""
        try:
            prov = json.loads(self.cfg.provenance_path.read_text())
        except Exception:
            return {}
        sess = prov.setdefault("orchestrator_session", {})
        if self.session is not None:
            sess["ids"] = list(getattr(self.session, "session_ids", []))
            sess["kind_used"] = self.session.kind
            sess["compactions"] = self.session.compactions
            sess["continuity_breaks"] = self.session.continuity_breaks
            sess["usage_available"] = self.session.usage_available
            # The hardening's own record: how often each guard fired.
            sess["timeouts"] = getattr(self.session, "timeouts", 0)
            sess["cwd_asserts"] = getattr(self.session, "cwd_asserts", 0)
            sess["session_cwd_from_header"] = getattr(self.session,
                                                      "session_cwd", "")
            sess["cost_usd_reported"] = round(
                getattr(self.session, "cost_usd_reported", 0.0), 6)
        sess["session_resume_verified"] = self.cfg.session_resume_verified
        atomic_write(self.cfg.provenance_path,
                     json.dumps(prov, indent=2, sort_keys=True) + "\n")
        return prov

    # -- reading ----------------------------------------------------------- #

    def rows(self) -> list:
        return ledger_rows(self.cfg.archive_dir)

    def frontier_ids(self, rows=None) -> set:
        return {r.id for r in pareto_frontier(rows if rows is not None else self.rows())}

    # -- the LLM orchestrator ---------------------------------------------- #

    OPENING_PROMPT = """\
You are the ORCHESTRATOR of a Go-Explore experiment on NetHack. You do not play
the game. You direct a series of player sessions, each of which resumes a saved
checkpoint and plays until it dies.

This is ONE long conversation across the whole run. Everything you work out now
is available to you at every later round, so think properly before the first
launch.

THE SETUP
  Seed {seed}, character {character}, every attempt on the same dungeon. This
  is a single-seed study by design: you are meant to learn this one dungeon
  deeply, and what you learn about it is expected to be specific to it.
  Players have: the standard action surface, rollback(n), save(label, note),
  and wiki(page|query|section) over the same two curated pages you have.

YOUR KNOWLEDGE BASE (read it now, before deciding anything)
  {wiki_dir}
    why_do_i_keep_dying.md   standard_strategy.md   MANIFEST.json
  Use your IPython kernel to read them. Also available:
    python {wiki_tool} --list
    python {wiki_tool} --query <term>
    python {wiki_tool} --page <page> --section <section>

ROUND 1 IS DISCUSSION, NOT A LAUNCH. Reply with:
  1. What you learned from the wiki that you expect to matter for this run,
     and why.
  2. Your strategy for the run: what kinds of checkpoint you will resume from,
     what you will tell players to do, and what would make you change course.
  3. What evidence would tell you your strategy is wrong.
Do not choose a checkpoint yet. There is only the starting state so far.
"""

    ROUND_PROMPT = """\
ROUND {round}. Budget so far: players ${player_usd:.2f}, you ${orch_usd:.2f},
ceiling ${ceiling:.2f}.

{last_block}

{ledger}

Choose the checkpoint the next player resumes from, and write it a DIRECTIVE:
one or two sentences of concrete instruction for THIS attempt. The directive is
served to the player verbatim at the top of its first observation.

There is NO required form. Say the strategy you actually want followed, in
whatever terms express it -- go and kill the thing you know is on that level,
avoid it entirely, farm here until XL 5 and then take it on, take this route
down. You are not writing for a grader; you are steering a player.

Any checkpoint id in the archive above is a legal choice.

Reply with a short rationale and then EXACTLY this JSON object on its own line:
{{"checkpoint": "<id from the archive above>", "directive": "<your instruction>", "rationale": "<one line>"}}

An id that is not in the archive falls back to the scripted selector and is
recorded as your round having failed to decide.
"""

    OPENING_RETRY_PROMPT = """\
Your last reply was not usable: {reason}

Write the opening discussion again, once, as a single coherent plan. Do not
repeat yourself and do not restate sections you have already written.
"""

    DIRECTIVE_RETRY_PROMPT = """\
Your last reply could not be used: {reason}

Reply again. Prose first if you want it, then EXACTLY this JSON object alone on
the final line, with an id from the archive above ({ids}):
{{"checkpoint": "<id>", "directive": "<your instruction>", "rationale": "<one line>"}}

The directive must be a non-empty instruction. This attempt cannot be launched
without one: an attempt served no directive is indistinguishable from the
control arm, and would be recorded as a failure of this round rather than as a
player result.
"""

    # NO HAND-WRITTEN COMPACTION PROMPT. There used to be one here: past
    # 400k sent characters the run asked the orchestrator to summarize itself
    # and continued in a FRESH session seeded with that summary. It is gone.
    # Compaction is now the CLI's own -- prime-agent has native threshold
    # compaction (`settings.compaction`), and we configure it to fire at a hard
    # 128K-token context bound rather than writing a summarization prompt of
    # our own. See `e16_session.configure_native_compaction`. The conversation
    # stays ONE session across the whole run, which is also what makes
    # `--resume` continuity meaningful.

    def open_discussion(self, character: str = "Val-hum-neu-fem") -> Optional[dict]:
        """Round 1: the planning phase, before any player is launched.

        Deliberately separated from the first selection. An orchestrator that
        reads the wiki in the same breath as it picks a checkpoint has no
        recorded strategy to be held to, and "did it follow its own plan?"
        stops being answerable. This round produces a plan on the record.
        """
        if self.session is None or self.opened:
            return None
        prompt = self.OPENING_PROMPT.format(
            seed=self.cfg.game_seed, character=character,
            wiki_dir=self.cfg.wiki_dir,
            wiki_tool=self.cfg.orchestrator_dir / "wiki_tool.py",
        )
        tries: list = []
        res = None
        degen: dict = {}
        for i in range(max(0, int(self.cfg.opening_retries)) + 1):
            kind = "opening" if i == 0 else f"opening_retry{i}"
            res = self.session.ask(
                prompt if i == 0 else self.OPENING_RETRY_PROMPT.format(
                    reason=degen.get("reason", "unusable reply")),
                kind=kind)
            self.budget.add_orchestrator(res.spend_usd)
            # THE RAW REPLY IS RECORDED BEFORE IT IS JUDGED, every attempt, so
            # a rejected round leaves the evidence it was rejected on. The
            # pilot's opening was thrown away as a repetition loop when the
            # bytes would have shown the loop was in the reader.
            self._record_raw(f"opening_attempt{i + 1}", res)
            degen = detect_degeneration(res.text or "")
            tries.append({"kind": kind, "error": res.error,
                          "session_id": res.session_id, "degeneration": degen})
            if not degen["degenerate"] and not res.error:
                break
        self.opened = True
        self.opening_attempts = tries
        atomic_write(self.cfg.orchestrator_dir / "opening_plan.txt", res.text or "")
        atomic_write(self.cfg.orchestrator_dir / "opening_degeneration.json",
                     json.dumps(tries, indent=2, sort_keys=True) + "\n")
        if degen["degenerate"]:
            # HARD ERROR. Every later round is anchored on this plan and every
            # attempt is steered by directives written against it; proceeding
            # without one is proceeding without the strategy the arm is named
            # for, and there would be nothing in the output tree to say so.
            raise OpeningPlanUnusable(
                f"the orchestrator's opening round produced no usable plan "
                f"after {len(tries)} attempt(s): {degen['reason']}. Raw replies "
                f"are in {self.cfg.orchestrator_dir}/raw/ and the verdicts in "
                f"opening_degeneration.json. Refusing to launch players with no "
                f"opening strategy -- an arm whose plan is missing is not the "
                f"arm it reports being."
            )
        return {"text": res.text, "error": res.error, "session_id": res.session_id,
                "degeneration": degen, "attempts": tries}

    def _record_raw(self, label: str, res) -> Path:
        """Write one round's RAW stdout (and its parsed reply) to the run dir.

        Separate from `orchestrator_rounds.jsonl` on purpose: the log carries
        the parsed reply, and the whole class of failure this exists for is the
        parsed reply disagreeing with the bytes it came from.
        """
        raw_dir = self.cfg.orchestrator_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{label}.stdout.txt"
        atomic_write(path, getattr(res, "raw_stdout", "") or "")
        atomic_write(raw_dir / f"{label}.reply.txt", getattr(res, "text", "") or "")
        return path

    def decide(self, rows: list) -> dict:
        """Who goes next and what they are told. Validated, then recorded.

        Returns ``{"checkpoint_id", "directive", "source", "selection",
        "candidates_shown", "decision"}``. ``source`` is one of ``llm``, ``scripted``, or
        ``scripted_fallback`` -- the last one meaning the LM was asked and did
        not produce a usable answer. The distinction matters: an arm whose
        "LLM selections" were half fallbacks is not the arm it claims to be,
        and pooling them would hide the LM's failure rate inside the scripted
        selector's behaviour.
        """
        n = len(self.attempts) + 1
        sel = select(rows, self.cfg, self.rng)  # always computed: it is the
        # fallback AND the record of what the ablation would have done.
        out = {"checkpoint_id": sel.chosen_id, "directive": "",
               "source": "scripted", "selection": sel.to_json(),
               "candidates_shown": None, "decision": None}
        if self.cfg.selector != "llm" or self.session is None:
            return out

        # The orchestrator is shown the LAST TREATMENT, not the last attempt.
        # Under --paired-control the last attempt is a control that was
        # deliberately given no instruction; feeding "your directive was:
        # (none)" back as the orchestrator's own last round would teach it that
        # its directives are not being delivered.
        last = self._last_treatment()
        last_block = "This is the first launch." if last is None else (
            f"LAST ATTEMPT (#{last['attempt']} from c{last['from_checkpoint']}): "
            f"{last['outcome']}"
            + (f":{last['censor_reason']}" if last["censor_reason"] else "")
            + f", max Dlvl {_or_unknown(last['max_dlvl'])}, "
              f"{_or_unknown(last['calls'])} calls.\n"
            f"Your directive was: {last.get('directive') or '(none)'}\n"
            f"Compliance rubric said: "
            f"{(last.get('directive_compliance') or {}).get('class')}\n"
            f"The player's own account (truncated to the orchestrator's "
            f"per-round allowance):\n"
            + self.session.cap_summary(last.get("model_text", {}).get("summary", ""))
        )
        # THE LINT, FED BACK. A warning that only lands in a log changes
        # nothing; a warning the orchestrator reads before writing its next
        # directive changes the directive. This is the cheap half of the
        # tool-vs-outcome fix -- the expensive half would be a smarter rubric,
        # and the sims' recommendation was explicitly the cheap half.
        if self._last_lint:
            last_block += (
                "\n\nWARNING about how you phrased that directive -- it "
                "affects whether compliance can be measured at all:\n"
                + "\n".join(f"  - {w['message']}" for w in self._last_lint))
        if self.cfg.paired_control:
            last_block += (
                "\n\nNOTE: every directive you write is also run a second time "
                "from the same checkpoint WITHOUT it, as its own control. Two "
                "attempts per round is expected; the ledger shows both.")
        # THE MENU AND THE RECORD OF IT ARE THE SAME OBJECT. `build_ledger`
        # returns the text the model is shown and one record per checkpoint IN
        # that text, so `selection.jsonl` cannot drift from what was offered --
        # which is exactly how the old record came to log the scripted
        # selector's one-row frontier as "the candidates" for rounds the model
        # was shown the whole archive.
        ledger_text, candidates_shown = build_ledger(rows, self.cfg,
                                                     attempts=self.attempts)
        out["candidates_shown"] = candidates_shown
        prompt = self.ROUND_PROMPT.format(
            round=n, player_usd=self.budget.player_usd,
            orch_usd=self.budget.orchestrator_usd,
            ceiling=self.cfg.budget_ceiling_usd,
            last_block=last_block,
            ledger=ledger_text,
        )
        from e16_session import parse_decision
        # EVERY id in the archive, and the prompt now says so. `parse_decision`
        # has always validated against this set rather than against the
        # frontier; for 24 checkpoints and a one-row table, the prompt was the
        # only thing making 23 of them unchoosable.
        ids = [r.id for r in rows]

        # BOUNDED RETRY, THEN RAISE. A round that produces no decision used to
        # fall through to the scripted selector with an empty directive, and
        # the run continued looking like a go_explore run. It was not one: the
        # attempt it launched was a control. See `OrchestratorRoundFailed`.
        tries: list = []
        res = None
        dec = None
        for i in range(max(0, int(self.cfg.directive_retries)) + 1):
            kind = f"round{n}" if i == 0 else f"round{n}_retry{i}"
            if i == 0:
                ask_text = prompt
            else:
                reason = (dec.fallback_reason or "no decision found") \
                    if dec is not None else "no reply"
                if dec is not None and dec.valid and not dec.directive.strip():
                    reason = ("the JSON object parsed but its `directive` was "
                              "empty")
                ask_text = self.DIRECTIVE_RETRY_PROMPT.format(
                    reason=reason, ids=", ".join(ids))
            res = self.session.ask(ask_text, kind=kind)
            self.budget.add_orchestrator(res.spend_usd)
            self._record_raw(f"decide_a{n}_attempt{i + 1}", res)
            dec = parse_decision(res.text, ids)
            # NO LENGTH FLOOR ON A DECISION ROUND. The opening round is a plan
            # and a 200-char one is not a plan; a decision round is one line of
            # rationale plus a JSON object, and a terse one is a good one. What
            # still disqualifies it is the LOOP -- the reply repeating itself
            # around a decision the model never finished.
            degen = detect_degeneration(res.text or "", min_chars=0)
            # Under `--no-directive` the round only has to name a CHECKPOINT:
            # the empty directive is the treatment there, not a failure, and
            # discarding the LM's choice would silently turn the control arm
            # into a scripted-selector arm as well.
            usable = (dec.valid and not degen["degenerate"]
                      and (bool(dec.directive.strip()) or self.cfg.no_directive))
            tries.append({
                "kind": kind, "parsed": dec.parsed, "valid": dec.valid,
                "fallback_reason": dec.fallback_reason,
                "directive_empty": dec.valid and not dec.directive.strip(),
                "degeneration": degen,
                "reply_chars": len(res.text or ""),
                "raw_stdout_chars": len(getattr(res, "raw_stdout", "") or ""),
                "error": res.error,
            })
            if usable:
                break

        out["decision"] = {
            "parsed": dec.parsed, "valid": dec.valid,
            "fallback_reason": dec.fallback_reason,
            "rationale": dec.rationale, "raw": (dec.raw or "")[:4000],
            "session_id": res.session_id,
            "continuity_broken": res.continuity_broken,
            "error": res.error,
            "attempts": tries,
        }
        if dec.valid and (dec.directive.strip() or self.cfg.no_directive):
            out["checkpoint_id"] = dec.checkpoint_id
            out["directive"] = dec.directive
            out["source"] = "llm"
        elif self.cfg.no_directive:
            # The one legal empty directive is the explicit run-wide control,
            # and it is handled above. Reaching here in that mode means the
            # round named no valid CHECKPOINT either -- a selection failure,
            # recorded as a fallback. It does not raise: no directive was ever
            # going to be served, so nothing about the treatment is misreported.
            self.llm_fallbacks += 1
            out["source"] = "scripted_fallback"
        else:
            self.llm_fallbacks += 1
            out["source"] = "scripted_fallback"
            atomic_write(self.cfg.orchestrator_dir /
                         f"failed_decision_attempt{n}.json",
                         json.dumps({"attempts": tries, "valid_ids": ids},
                                    indent=2, sort_keys=True) + "\n")
            raise DirectiveExtractionFailed(
                f"attempt {n}: the orchestrator produced no usable directive "
                f"after {len(tries)} round(s). Last reason: "
                f"{tries[-1].get('fallback_reason') or tries[-1]}. Raw stdout "
                f"and parsed replies for every attempt are in "
                f"{self.cfg.orchestrator_dir}/raw/. Refusing to launch a "
                f"directive attempt with no directive: it would be recorded as "
                f"a go_explore treatment and would in fact be a control, which "
                f"is the substitution that invalidated E15."
            )
        if self.cfg.no_directive:
            out["directive"] = ""  # the explicit control mode
        return out

    def _last_treatment(self) -> Optional[dict]:
        """The last attempt that actually carried a directive, if any."""
        for a in reversed(self.attempts):
            if a.get("pair_role") != ROLE_CONTROL:
                return a
        return None

    # `maybe_compact` IS GONE. It used to ask the model for a hand-off
    # summary at 400k sent chars and continue in a fresh session seeded with
    # it -- our own summarization prompt, our own idea of what was worth
    # keeping, and a broken conversation every time it fired. prime-agent
    # compacts natively at a token threshold it computes from the model's real
    # context window, so the bound is configured once (128K tokens, in
    # `e16_session.configure_native_compaction`) and the CLI does the rest
    # inside the one session.

    # -- stopping ---------------------------------------------------------- #

    def milestone_row(self, rows: list) -> Optional[Row]:
        for r in rows:
            if r.dlvl >= self.cfg.milestone_dlvl:
                return r
            if r.dungeon_number == self.cfg.milestone_dungeon:
                return r
        return None

    def should_stop(self, rows: list) -> Optional[str]:
        if len(self.attempts) >= self.cfg.max_attempts:
            return STOP_MAX_ATTEMPTS
        if not self.budget.can_launch():
            return STOP_BUDGET
        if not rows:
            return STOP_EMPTY_ARCHIVE
        if self.cfg.arm == ARM_MATCHED_RESTART:
            # NO FRONTIER STOPS IN THE NULL ARM, and this is a correctness
            # requirement rather than a convenience. matched_restart grows no
            # archive by construction, so `_since_advance` would reach
            # `stall_attempts` on schedule and truncate the control at 8
            # attempts while the method it is being compared against ran 12 --
            # producing max-of-8 against max-of-12 and calling it a matched
            # comparison. N is fixed by --max-attempts and by the budget, and
            # by nothing else.
            # `max_dlvl` is null on an attempt that left no evidence at all;
            # an unknown depth cannot satisfy a milestone, and must not raise.
            best = max((a["max_dlvl"] for a in self.attempts
                        if a.get("max_dlvl") is not None), default=0)
            if best >= self.cfg.milestone_dlvl:
                return STOP_MILESTONE
            return None
        if self.milestone_row(rows) is not None:
            return STOP_MILESTONE
        if self._since_advance >= self.cfg.stall_attempts:
            return STOP_STALL
        return None

    # -- one attempt ------------------------------------------------------- #

    def run_attempt(self, rows: list) -> dict:
        """One round of the loop. Returns the LAST attempt record it produced.

        Three shapes, because the arms are genuinely different loops rather
        than one loop with flags:

        * ``matched_restart``   -- no orchestrator round, no selection, no
          directive: one attempt from the fixed start state. See
          :meth:`run_matched_restart_attempt`.
        * ``go_explore`` + ``--paired-control`` -- one orchestrator round
          produces one directive, and that directive is run TWICE from the same
          checkpoint, with and without it.
        * ``go_explore`` -- the plain case: one round, one attempt.
        """
        cfg = self.cfg
        if cfg.arm == ARM_MATCHED_RESTART:
            return self.run_matched_restart_attempt(rows)

        n = len(self.attempts) + 1
        choice = self.decide(rows)
        chosen_id = choice["checkpoint_id"]
        directive = choice["directive"]
        _append_jsonl(cfg.selection_path,
                      selection_record(n, choice, rows,
                                       frontier=self.frontier_ids(rows)))

        ck_dir = next((r.path for r in rows if r.id == chosen_id), None)

        if not cfg.paired_control:
            return self._launch_one(rows, chosen_id, ck_dir, directive, choice,
                                    pair_id=None, pair_role=ROLE_SOLO)

        # PAIRED: the same checkpoint, twice, differing only in the directive.
        # The treatment goes first so that an interrupted pair leaves the
        # directive arm complete rather than a control with nothing to control
        # for -- and `pair_complete` on each record says whether its partner
        # ever ran, so a truncated run cannot be analysed as if it had.
        self._pair_counter += 1
        pair_id = self._pair_counter
        rec_t = self._launch_one(rows, chosen_id, ck_dir, directive, choice,
                                 pair_id=pair_id, pair_role=ROLE_TREATMENT)
        rows_now = self.rows()
        # The control resumes THE SAME checkpoint. Not the frontier as it now
        # stands: a control launched from a state the treatment just produced
        # would be measuring the treatment.
        ck_dir_c = next((r.path for r in rows_now if r.id == chosen_id), ck_dir)
        try:
            self.budget.check()
        except BudgetExceeded:
            # An unpartnered treatment is worse than no pair: it enters the
            # dataset looking like a solo directive attempt. `pairs()` derives
            # `complete: false` for it from the attempts that exist, and the
            # summary carries `incomplete_pairs` so a truncated run says so.
            self.write_summary()
            raise
        rec_c = self._launch_one(rows_now, chosen_id, ck_dir_c, "", choice,
                                 pair_id=pair_id, pair_role=ROLE_CONTROL,
                                 directive_kind_override=rec_t["directive_kind"])
        return rec_c

    def _launch_one(self, rows: list, chosen_id, ck_dir, directive: str,
                    choice: dict, *, pair_id, pair_role: str,
                    directive_kind_override: Optional[str] = None) -> dict:
        """Build the context for one attempt, launch it, ingest the result."""
        cfg = self.cfg
        n = len(self.attempts) + 1

        # HARD budget gate: before the launch, never after. Placed after the
        # orchestrator round on purpose -- that round's tokens are already
        # spent and already on the budget's second line, so the check sees the
        # true total rather than a stale one.
        self.budget.check()

        ledger_text = render_ledger(rows, cfg, current_id=chosen_id,
                                    attempts=self.attempts)
        if ck_dir is not None:
            for extra in (render_lessons(ck_dir, cfg.ledger_max_lessons),
                          render_prefix(ck_dir)):
                if extra:
                    ledger_text += "\n\n" + extra

        out_dir = cfg.run_dir / "attempts" / f"a{n:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = PlayerContext(
            attempt=n, checkpoint_dir=ck_dir, checkpoint_id=chosen_id,
            archive_dir=cfg.archive_dir, wiki_dir=cfg.wiki_dir,
            out_dir=out_dir, ledger_text=ledger_text,
            fidelity_log=cfg.fidelity_path, game_seed=cfg.game_seed,
            tier=cfg.tier, arm=cfg.player_arm, directive=directive,
            experiment_arm=cfg.arm, pair_id=pair_id, pair_role=pair_role,
            directive_kind=(directive_kind_override
                            if directive_kind_override is not None
                            else classify_directive_kind(directive)),
            reseed=self.reseed_for(chosen_id, n),
        )
        if not directive and not cfg.no_directive and pair_role != ROLE_CONTROL \
                and cfg.selector == "llm":
            # THE LAST GATE, and it refuses rather than substitutes.
            #
            # This used to serve "(orchestrator produced no directive for this
            # attempt; play as you judge best)" -- visible in the record, yes,
            # but only to someone reading it, and the attempt still ran. In the
            # GE-wiki pilot it ran for $13.36 and 187 calls, and what it
            # measured was the no-directive control while every artifact
            # labelled it a go_explore treatment. `decide` now raises before
            # reaching here, so this is defence in depth against a future
            # caller assembling a context by hand.
            raise DirectiveExtractionFailed(
                f"attempt {n}: refusing to launch a treatment attempt from "
                f"c{chosen_id} with an empty directive. The directive IS the "
                f"treatment; an attempt served none is a control, and running "
                f"it under a treatment label is the substitution that "
                f"invalidated E15. Use --no-directive if a control is what was "
                f"wanted."
            )
        atomic_write(out_dir / "ledger_served.txt", ledger_text)
        atomic_write(out_dir / "directive_served.txt", ctx.directive)
        self._decision_of_attempt = choice

        before = {p.name for p in checkpoint_list(cfg.archive_dir)}
        before_front = self.frontier_ids(rows)
        t0 = time.time()
        # THE PROVISIONAL ROW, BEFORE THE LAUNCH. From here until `ingest`
        # returns, this attempt exists on disk: which checkpoint it resumed,
        # what it was told, when it started and under which pid. Everything the
        # player writes into the archive from now on is attributable even if
        # this process never runs another line of Python.
        self._budget_stop = None
        self.open_attempt(ctx, before, t0)
        try:
            result = self.launcher(ctx)
        except Exception as exc:  # a launcher blow-up is a CENSORED attempt
            # The launcher never returned, so nothing costed the rollout --
            # `spend_known=False` keeps that from being booked as $0.00, and
            # `ingest` recovers what the env's turn files prove.
            result = _unreported_result(
                ctx.out_dir, f"{type(exc).__name__}: {exc}",
                rollouts_billed(ctx.out_dir))
            result.wall_s = time.time() - t0
        except BaseException as exc:
            # A SIGNAL, or anything else that is not an ordinary error. The row
            # is finalized HERE, before the exception continues on its way --
            # mechcheck's 24 orphaned checkpoints are what happens when this
            # path does not exist. The attempt is `censored:interrupted`, never
            # `died`: the character was alive when the infrastructure stopped.
            detail = (f"{type(exc).__name__}: {exc}" if str(exc)
                      else type(exc).__name__)
            self.finalize_interrupted(ctx, detail=detail, t0=t0)
            raise
        if not result.wall_s:
            result.wall_s = time.time() - t0
        result = self._apply_budget_stop(result)
        return self.ingest(ctx, result, before, before_front)

    def _apply_budget_stop(self, result: PlayerResult) -> PlayerResult:
        """Relabel an attempt the in-flight guard stopped.

        The player was SIGTERMed, so whatever it reports back -- a nonzero exit,
        a missing `traces.jsonl`, a `ProviderError` from a call cut off
        mid-flight -- would otherwise classify as `censored:harness_error` and
        be indistinguishable from a real infrastructure failure. It is neither:
        it is this run deciding to stop spending, and the record has to say so
        or the run's own censoring table is wrong about why its attempts ended.

        `budget_stop` maps to `CENSOR_BUDGET_STOP` through `_STOP_TO_CENSOR`,
        so the row reads `censored:budget_stop`. The measured spend, the depth
        and the checkpoints are all left exactly as reported -- only the label
        changes, and the evidence for the relabel travels with it.
        """
        stop = self._budget_stop
        if stop is None:
            return result
        self._budget_stop = None
        result.stop_condition = "budget_stop"
        # `classify_outcome` censors on `error` FIRST and with the wrong
        # reason, so the harness-side error text is moved out of the field that
        # drives classification and kept as evidence instead.
        if result.error:
            stop["player_error"] = result.error
            result.error = ""
        raw = dict(result.raw or {})
        raw["budget_stop"] = stop
        result.raw = raw
        self._say(
            f"[budget] attempt {stop.get('attempt')} STOPPED in flight: "
            f"estimated ${stop.get('spend_so_far_upper_usd')} on top of "
            f"${stop.get('spend_before_usd')} would reach "
            f"${stop.get('projected_total_usd')} against a "
            f"${stop.get('ceiling_usd')} ceiling "
            f"(rate {stop.get('rate_usd_per_hour')}/hr, "
            f"{stop.get('rate_source')}; ESTIMATE, not a measurement)")
        return result

    # -- durability: the in-flight attempt ---------------------------------- #

    def open_attempt(self, ctx: PlayerContext, before: set, t0: float) -> dict:
        """Journal one attempt as OPEN and start its progress stream."""
        cfg = self.cfg
        rec = {
            "event": "open",
            "attempt": ctx.attempt,
            "attempt_id": f"a{ctx.attempt:03d}",
            "ts": t0,
            "iso": _now_iso(t0),
            "pid": os.getpid(),
            "from_checkpoint": ctx.checkpoint_id,
            "checkpoint_dir": (str(ctx.checkpoint_dir)
                               if ctx.checkpoint_dir else None),
            "directive": ctx.directive,
            "directive_kind": ctx.directive_kind,
            "experiment_arm": ctx.experiment_arm,
            "pair_id": ctx.pair_id,
            "pair_role": ctx.pair_role,
            "reseed": (list(ctx.reseed) if ctx.reseed else None),
            "selection_source": (getattr(self, "_decision_of_attempt", None)
                                 or {}).get("source"),
            "out_dir": str(ctx.out_dir),
            "archive_before": sorted(before, key=_id_key_name),
            "cumulative_spend_usd_before": round(self.budget.spent_usd, 6),
            "cumulative_calls_before": self.cum_calls,
        }
        self._open = rec
        _append_jsonl(cfg.journal_path, rec)
        # A SECOND COPY, inside the attempt's own directory. The journal can be
        # lost with the run directory's top level (or predate this code); an
        # attempt that carries its own open record is recoverable from the
        # directory alone, which is the shape mechcheck was left in.
        try:
            atomic_write(ctx.out_dir / "attempt_open.json",
                         json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n")
        except Exception:
            pass
        # THE IN-FLIGHT BUDGET GUARD. `cumulative_spend_usd_before` below is
        # frozen at attempt start, so without this the stream says nothing
        # about the money the attempt is spending RIGHT NOW -- which is how a
        # $9 smoke reached an estimated $22 with nothing able to notice. The
        # estimator is honest about being an estimate (see
        # `estimate_inflight_spend`); the guard enforces against its UPPER
        # number, never its best guess.
        rate = calibrate_spend_rate(
            self.attempts, prior_usd_per_hour=cfg.inflight_spend_prior_usd_per_hour)
        self._spend_rate = rate

        def _spend_fn(_t0=t0, _out=ctx.out_dir, _rate=rate):
            return estimate_inflight_spend(
                _out, time.time() - _t0, _rate,
                safety_factor=cfg.inflight_spend_safety_factor)

        self._monitor = AttemptProgressMonitor(
            paths=[cfg.progress_path, ctx.out_dir / "progress.jsonl"],
            attempt=ctx.attempt, turns_dir=ctx.out_dir / "turns",
            archive_dir=cfg.archive_dir, interval_s=cfg.progress_interval_s,
            baseline=frozenset(before), started_at=t0,
            spend_fn=_spend_fn,
            guard=(self._inflight_budget_guard
                   if cfg.enforce_inflight_budget else None),
            extra={"from_checkpoint": ctx.checkpoint_id,
                   "directive_kind": ctx.directive_kind,
                   "pair_role": ctx.pair_role,
                   "spend_rate_usd_per_hour": rate["rate_usd_per_hour"],
                   "spend_rate_source": rate["rate_source"],
                   "cumulative_spend_usd_before": round(self.budget.spent_usd, 6)})
        self._monitor.sample("attempt_start", directive=ctx.directive)
        self._monitor.start()
        # summary.json now exists from the FIRST launch rather than from the
        # first completed attempt, and says which attempt is in flight.
        self.write_summary()
        return rec

    # -- the in-flight budget guard ---------------------------------------- #
    def _inflight_budget_guard(self, rec: dict) -> None:
        """Stop the attempt in flight if its estimated spend breaches the
        ceiling. Called from the progress monitor's thread, after the sample
        that justified it is already on disk.

        WHY THIS EXISTS. `Budget.check()` is a PRE-LAUNCH gate -- its own
        docstring says cost "is therefore controlled by refusing the NEXT
        launch". That is sound only if an attempt's cost is bounded, and it is
        not: players are uncapped by design. One runaway attempt can therefore
        cross the ceiling with nothing in the system able to notice until it
        returns, which is exactly what happened on the $9 smoke that reached an
        estimated $22. On $385 that is not tolerable.

        Enforced against `spend_so_far_upper_usd`, never the best guess: see
        `estimate_inflight_spend` for why the two errors are not symmetric.
        Fires at most once per attempt -- a second SIGTERM to a process already
        shutting down would just make a clean stop into a dirty one.
        """
        if self._budget_stop is not None:
            return
        try:
            est = float(rec.get("spend_so_far_upper_usd") or 0.0)
        except (TypeError, ValueError):
            return
        projected = self.budget.spent_usd + est
        if projected < self.budget.ceiling_usd - self.budget.min_headroom_usd:
            return
        reason = {
            "attempt": rec.get("attempt"),
            "wall_s": rec.get("wall_s"),
            "spend_so_far_usd": rec.get("spend_so_far_usd"),
            "spend_so_far_upper_usd": round(est, 4),
            "spend_before_usd": round(self.budget.spent_usd, 4),
            "projected_total_usd": round(projected, 4),
            "ceiling_usd": self.budget.ceiling_usd,
            "min_headroom_usd": self.budget.min_headroom_usd,
            "rate_usd_per_hour": rec.get("spend_so_far_rate_usd_per_hour"),
            "rate_source": rec.get("spend_so_far_rate_source"),
            "rollouts_started": rec.get("rollouts_started"),
            # Said out loud on the record: this stop was decided on an
            # estimate, because no in-flight usage source exists.
            "decided_on": "estimate",
        }
        self._budget_stop = reason
        if self._monitor is not None:
            try:
                self._monitor.sample("budget_stop", budget_stop=reason)
            except Exception:
                pass
        self._stop_active_player(reason)

    def _stop_active_player(self, reason: dict) -> None:
        """Ask the launcher to end the running attempt cleanly.

        Best-effort and duck-typed: the real player exposes `terminate_active`,
        and an injected test launcher generally does not. A launcher that
        cannot be stopped still gets the `budget_stop` record and the run still
        halts at the next pre-launch gate -- degraded, but never silent.
        """
        stop = getattr(self.launcher, "terminate_active", None)
        if not callable(stop):
            self._say(f"[budget] in-flight ceiling breach on attempt "
                      f"{reason.get('attempt')} but this launcher cannot be "
                      f"stopped; the run will halt at the next launch gate")
            return
        try:
            stop(grace_s=self.cfg.inflight_stop_grace_s)
        except Exception as exc:
            self._say(f"[budget] terminate_active failed: {exc!r}")

    def _say(self, msg: str) -> None:
        try:
            sys.stderr.write(msg.rstrip("\n") + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    def close_attempt(self, record: dict) -> None:
        """Journal an attempt as CLOSED and stop its progress stream."""
        if self._monitor is not None:
            try:
                self._monitor.stop("attempt_end", outcome=record.get("outcome"),
                                   censor_reason=record.get("censor_reason"),
                                   spend_usd=record.get("spend_usd"),
                                   cumulative_spend_usd=record.get("cumulative_spend_usd"),
                                   new_checkpoints=record.get("new_checkpoints"))
            except Exception:
                pass
            self._monitor = None
        _append_jsonl(self.cfg.journal_path, {
            "event": "close", "attempt": record.get("attempt"),
            "ts": time.time(), "iso": _now_iso(),
            "outcome": record.get("outcome"),
            "censor_reason": record.get("censor_reason"),
            "spend_usd": record.get("spend_usd"),
            "new_checkpoints": record.get("new_checkpoints"),
        })
        self._open = None

    def finalize_interrupted(self, ctx: Optional[PlayerContext] = None, *,
                             detail: str = "", t0: Optional[float] = None) -> Optional[dict]:
        """Finalize the in-flight attempt as ``censored:interrupted``. Never raises.

        Called from the launch path's ``except BaseException``, from ``run``'s
        ``finally``, and from ``main``'s last-resort handler -- three chances,
        because the one that matters is whichever one the failure allows to
        run. It is idempotent: `self._open` is cleared by the first.
        """
        opened = self._open
        if opened is None:
            return None
        try:
            n = int(opened.get("attempt") or (ctx.attempt if ctx else 0))
            out_dir = Path(opened.get("out_dir")
                           or (ctx.out_dir if ctx else self.cfg.run_dir))
            ev = _attempt_dir_evidence(out_dir)
            if t0:
                ev["started_at"] = t0
            after = {p.name for p in checkpoint_list(self.cfg.archive_dir)}
            ev["new_checkpoints"] = sorted(
                after - set(opened.get("archive_before") or []), key=_id_key_name)
            reason = detail or self.interrupted_by or "interrupted"
            rec = interrupted_record(attempt=n, opened=opened, evidence=ev,
                                     reason=CENSOR_INTERRUPTED, detail=reason)
            # Spend and calls STILL move the run's counters. An interrupted
            # attempt that cost money and is not charged for it is a budget
            # that lies, which is the same defect as an attempt that vanishes.
            self.budget.add_player(rec["spend_usd"],
                                   known=bool(rec.get("spend_known")))
            self.cum_calls += int(rec["calls"] or 0)
            self.cum_wall += rec["wall_s"]
            rec["cumulative_calls"] = self.cum_calls
            rec["cumulative_spend_usd"] = round(self.budget.spent_usd, 4)
            rec["cumulative_wall_s"] = round(self.cum_wall, 1)
            for p in checkpoint_list(self.cfg.archive_dir):
                if p.name in rec["new_checkpoints"] and ctx is not None:
                    try:
                        self._stamp_new_checkpoint(p, ctx, PlayerResult())
                    except Exception:
                        pass
            self.attempts.append(rec)
            _append_jsonl(self.cfg.attempts_path, rec)
            self.close_attempt(rec)
            self.write_summary()
            return rec
        except Exception:
            # Last resort: leave the journal entry OPEN and clear only the
            # in-memory handle. An `open` with no `close` is exactly what
            # `reconcile_run` finalizes on the next start, so failing here
            # degrades to "recovered one start later", never to "lost".
            self._open = None
            return None

    # -- the null arm ------------------------------------------------------- #

    def start_row(self, rows: list) -> Optional[Row]:
        """The ONE state ``matched_restart`` restarts from, every attempt.

        Named once and never re-selected: the whole content of this arm is that
        it does not select. Defaults to the archive's seed checkpoint, which is
        the oldest row -- the entrance state every go_explore run also begins
        from, so the two arms share a start.
        """
        if not rows:
            return None
        if self.cfg.start_checkpoint_id:
            want = str(self.cfg.start_checkpoint_id).lstrip("c")
            found = next((r for r in rows if r.id == want), None)
            if found is None:
                raise ValueError(
                    f"--start-checkpoint {self.cfg.start_checkpoint_id!r} is "
                    f"not in the archive (have: "
                    f"{sorted((r.id for r in rows), key=_id_key)})")
            return found
        return min(rows, key=lambda r: (r.created_at, _id_key(r.id)))

    def run_matched_restart_attempt(self, rows: list) -> dict:
        """One independent restart from the fixed start state.

        NO archive (the player writes its checkpoints into a per-attempt
        scratch directory that the selector never reads), NO selection, NO
        directive, NO lessons and NO ledger. What is left is the thing every
        reader compares the method against in their head: N tries from a good
        state. It draws on the SAME budget and emits the SAME attempt record
        shape, which is the only reason the two arms are comparable at all.
        """
        cfg = self.cfg
        n = len(self.attempts) + 1
        start = self.start_row(rows)
        if start is None:
            raise ValueError("matched_restart needs a seeded archive to name a "
                             "start state")
        self.budget.check()

        out_dir = cfg.run_dir / "attempts" / f"a{n:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # A THROWAWAY archive per attempt. The player's `save` skill still
        # works -- removing it would change the tool surface and make the arms
        # differ on a second axis -- but nothing it writes is ever selected
        # from, read back, or allowed to grow a frontier. That is what "no
        # archive" means operationally.
        scratch = out_dir / "scratch_archive"
        scratch.mkdir(parents=True, exist_ok=True)

        # THE SAME SHAPE the go_explore rounds write, so one reader can count
        # both arms -- with every "what was offered" field explicitly empty,
        # because this arm offers nothing. `chosen_id` is the launched id here
        # too, and there is no scripted pick to shadow it with.
        _append_jsonl(cfg.selection_path, {
            "attempt": n, "source": "matched_restart_fixed_start",
            "chosen_id": start.id,
            "chosen_on_frontier": start.id in self.frontier_ids(rows),
            "directive": "", "directive_kind": "none",
            "directive_lint": [], "llm": None,
            "reason": ("no selection: matched_restart restarts from one fixed "
                       "state by construction"),
            "candidates_shown": [], "n_candidates_shown": 0,
            "frontier_ids": sorted(self.frontier_ids(rows), key=_id_key),
            "scripted_would_pick": None, "scripted_agreed": None,
            "scripted": {}, "n_rows": len(rows),
        })

        ctx = PlayerContext(
            attempt=n, checkpoint_dir=start.path, checkpoint_id=start.id,
            archive_dir=scratch, wiki_dir=cfg.wiki_dir, out_dir=out_dir,
            ledger_text="", fidelity_log=cfg.fidelity_path,
            game_seed=cfg.game_seed, tier=cfg.tier, arm=cfg.player_arm,
            directive="", experiment_arm=ARM_MATCHED_RESTART,
            pair_id=None, pair_role=ROLE_SOLO, directive_kind="none",
            reseed=self.reseed_for(start.id, n),
        )
        atomic_write(out_dir / "ledger_served.txt", "")
        atomic_write(out_dir / "directive_served.txt", "")
        self._decision_of_attempt = {"source": "matched_restart_fixed_start",
                                     "decision": None}

        before = {p.name for p in checkpoint_list(cfg.archive_dir)}
        before_front = self.frontier_ids(rows)
        t0 = time.time()
        try:
            result = self.launcher(ctx)
        except Exception as exc:
            result = _unreported_result(
                ctx.out_dir, f"{type(exc).__name__}: {exc}",
                rollouts_billed(ctx.out_dir))
            result.wall_s = time.time() - t0
        if not result.wall_s:
            result.wall_s = time.time() - t0
        result = self._apply_budget_stop(result)
        return self.ingest(ctx, result, before, before_front)

    # -- reseeding ---------------------------------------------------------- #

    def reseed_for(self, checkpoint_id, attempt: int) -> Optional[tuple]:
        """``(core, disp)`` for this attempt's restore, or None (the default).

        Derived from the run's RNG seed, the checkpoint and the attempt index,
        so it is reproducible from provenance and so two attempts from one
        checkpoint get DIFFERENT dice -- which is the entire point of asking
        for it. Returns None unless ``--reseed`` was passed, and the None is
        what the launcher turns into "do not reseed at all" rather than into
        "reseed with zero".
        """
        if not self.cfg.reseed_on_restore:
            return None
        key = f"{self.cfg.rng_seed}:{checkpoint_id}:{attempt}".encode()
        digest = hashlib.sha256(key).digest()
        core = int.from_bytes(digest[:4], "big") or 1
        disp = int.from_bytes(digest[4:8], "big") or 1
        return (core, disp)

    # -- ingestion --------------------------------------------------------- #

    def ingest(self, ctx: PlayerContext, result: PlayerResult,
               before: set, before_front: set) -> dict:
        """Fold one finished attempt back into the archive and the record."""
        cfg = self.cfg
        outcome, censor_reason = classify_outcome(result)

        # WHAT THIS ATTEMPT IS ALLOWED TO CLAIM. `spend_known` is False on
        # exactly the paths where no rollout ever reported (`_unreported_result`
        # -- a dead env tool server, a killed launcher). On those paths there is
        # no token usage anywhere, so there is nothing to cost; the honest
        # numbers are the attempt's own last in-flight ESTIMATE for money and
        # the env's own turn files for everything else.
        #
        # Booking 0.0 instead -- which is what a bare `PlayerResult()` used to
        # hand over -- is the specific defect this run was blocked on: a run
        # whose archive holds Dlvl 4 checkpoints reporting `player_usd: 0.0`,
        # `cumulative_calls: 0` and `max_dlvl_per_attempt: [0]`, with
        # `attempts_with_unknown_spend: 0` asserting those zeros were solid.
        # Zero is a measurement. Unknown is the truth, and it must be said.
        spend_known = bool(getattr(result, "spend_known", True))
        estimate = {} if spend_known else progress_spend_estimate(ctx.out_dir)
        booked = float(result.spend_usd or 0.0)
        spend_source = "trace"
        if not spend_known:
            booked = float(estimate.get("spend_so_far_usd") or 0.0)
            spend_source = ("progress_estimate_lower_bound" if estimate
                            else "unavailable")
        self.budget.add_player(booked, known=spend_known)
        # UNKNOWN, NOT ZERO. A censored attempt that left no turn file at all
        # gets null on every count -- the summary must not be able to say a run
        # did nothing when its own archive says otherwise.
        unknown_counts = (not spend_known
                          and getattr(result, "evidence_source", "trace") == "none")
        n_calls = None if unknown_counts else int(result.calls or 0)
        row_dlvl = None if unknown_counts else int(result.max_dlvl or 0)
        row_xl = None if unknown_counts else int(result.max_xl or 1)
        self.cum_calls += int(n_calls or 0)
        self.cum_wall += float(result.wall_s or 0.0)

        # DIRECTIVE COMPLIANCE, from the harness's own per-turn record of what
        # the player CALLED -- not from anything the player said about itself.
        # Computed here, before the lesson append, because the lesson carries
        # it: the next orchestrator round reads "told it X -> it did Y" out of
        # the checkpoint's own lessons.md as well as out of the ledger.
        calls = calls_from_turns(ctx.out_dir) or (result.raw or {}).get("calls") or []
        # Harness metrics travel into the rubric so an OUTCOME clause is scored
        # against what the engine measured, not against whether the model typed
        # a matching word. `start_dlvl` comes from the checkpoint's own meta,
        # which is harness-computed, so "descend" means "deeper than where this
        # attempt started" rather than "deeper than level 1".
        metrics = dict((result.raw or {}).get("metrics") or {})
        metrics.setdefault("max_dlvl", int(result.max_dlvl or 0))
        metrics.setdefault("died", bool(result.died))
        start_dlvl = None
        if ctx.checkpoint_dir is not None:
            try:
                start_dlvl = int(checkpoint_meta(ctx.checkpoint_dir).get("dlvl") or 0)
            except Exception:
                start_dlvl = None
        if start_dlvl is not None:
            metrics.setdefault("start_dlvl", start_dlvl)
        compliance = classify_directive_compliance(ctx.directive, calls, outcome,
                                                   metrics=metrics)
        lint = lint_directive(ctx.directive)
        if ctx.pair_role != ROLE_CONTROL:
            self._last_lint = lint

        # New checkpoints the player wrote. Discovered by diffing the archive,
        # not by trusting a count the player reported.
        after_paths = checkpoint_list(cfg.archive_dir)
        new_dirs = [p for p in after_paths if p.name not in before]
        # THE LINEAGE IS A CHAIN, NOT A FAN. See `parent_chain`: within one
        # attempt each checkpoint's parent is the previous one this attempt
        # wrote, and only the first points back at the state that was resumed.
        chain = parent_chain(new_dirs, ctx.checkpoint_id)
        new_dirs = [p for p, _ in chain]  # creation order, so the record's
        # `new_checkpoints` list reads as the trajectory it was.
        for p, parent_id in chain:
            self._stamp_new_checkpoint(p, ctx, result, parent_id)

        # attempts_from++ on the source checkpoint, atomically.
        #
        # NO LESSONS IN THE NULL ARM, and this is the line that makes
        # "no lessons" true rather than intended: matched_restart bumps the
        # counter (an honest record of how many attempts started there) but
        # writes nothing back into lessons.md, so its N attempts never learn
        # from each other. A control that accumulated lessons would be a weaker
        # version of the method, not a null.
        if ctx.checkpoint_dir is not None:
            self._bump_attempts_from(ctx.checkpoint_dir)
        if ctx.checkpoint_dir is not None \
                and ctx.experiment_arm != ARM_MATCHED_RESTART:
            heading = (f"attempt {ctx.attempt} ({outcome}"
                       + (f":{censor_reason}" if censor_reason else "") + ")")
            body = (f"DIRECTIVE GIVEN: {ctx.directive or '(none)'}\n"
                    f"COMPLIANCE (keyword rubric): {compliance['class']}\n\n"
                    + (result.summary or "(no summary reported)").strip())
            if result.lesson:
                body += "\n\nLESSON: " + result.lesson.strip()
            append_lesson(ctx.checkpoint_dir, body, heading=heading)

        rows_after = self.rows()
        after_front = self.frontier_ids(rows_after)
        advanced = sorted(after_front - before_front, key=_id_key)
        if advanced:
            self._since_advance = 0
            best = max((r for r in rows_after if r.id in advanced),
                       key=lambda r: r.key)
            adv = {
                "attempt": ctx.attempt,
                "new_frontier_ids": advanced,
                "best_new": {"id": best.id, "dlvl": best.dlvl, "xl": best.xl,
                             "score": best.score, "balrog_min": best.balrog_min},
                # COMPARABILITY ACCOUNTING: what this advance cost.
                "attempts_consumed": ctx.attempt,
                "cumulative_calls": self.cum_calls,
                "cumulative_spend_usd": round(self.budget.spent_usd, 4),
                "wall_clock_s": round(time.time() - self.started_at, 1),
            }
            self.frontier_advances.append(adv)
        else:
            self._since_advance += 1

        choice = getattr(self, "_decision_of_attempt", None) or {}
        record = {
            "attempt": ctx.attempt,
            "from_checkpoint": ctx.checkpoint_id,
            "directive": ctx.directive,
            "directive_compliance": compliance,
            # PER-KIND PAIRING. `pair_id` joins a directive to its own control;
            # `directive_kind` is what makes the join analysable per kind rather
            # than only per run. Whether a pair is COMPLETE is derived in
            # `pairs()` from the attempts actually present, and never stamped on
            # the record here -- a record written when its partner has not run
            # yet cannot honestly claim either way, and an attempts.jsonl line
            # that had to be rewritten later would stop being append-only.
            "experiment_arm": ctx.experiment_arm,
            "pair_id": ctx.pair_id,
            "pair_role": ctx.pair_role,
            "directive_kind": ctx.directive_kind,
            "directive_lint": lint,
            "reseed": (list(ctx.reseed) if ctx.reseed else None),
            "selection_source": choice.get("source"),
            "orchestrator_decision": choice.get("decision"),
            "outcome": outcome,
            "censored": outcome == OUTCOME_CENSORED,
            "censor_reason": censor_reason,
            "stop_condition": result.stop_condition,
            "error": result.error,
            # The env tool server's own dying words when it died of a fatal
            # native signal, so `censor_reason: env_crash` can be audited
            # rather than believed.
            "env_crash": result.env_crash or None,
            "calls": n_calls,
            # WHERE THE THREE NUMBERS ABOVE AND BELOW CAME FROM. "trace" = the
            # rollout reported them; "turns" = the rollout never reported and
            # they were recovered from the env's own per-turn NDJSON (a LOWER
            # bound, since the file stops where the env died); "none" = there
            # was no evidence at all and `calls`/`max_dlvl`/`max_xl` are null.
            "evidence_source": getattr(result, "evidence_source", "trace"),
            "spend_usd": round(booked, 6),
            # `spend_known: false` says the number above is not a measurement.
            # It is the attempt's last published in-flight estimate
            # (`spend_so_far_usd`), booked as a LOWER BOUND rather than as a
            # zero -- and counted in the summary's
            # `attempts_with_unknown_spend`, which is what stops a run's cost
            # curve from silently averaging in free attempts that were not free.
            "spend_known": spend_known,
            "spend_source": spend_source,
            "spend_estimate": estimate or None,
            # WHAT THIS ATTEMPT ACTUALLY BILLED. A whole-rollout retry replays
            # the trajectory from turn 1 and the discarded attempts' usage never
            # reaches `traces.jsonl`, so `spend_usd` above costs ONE rollout for
            # an attempt that may have paid for two. Recorded, not corrected:
            # an attempt that cost 2x must be visible as such instead of being
            # averaged into the per-attempt cost curve as though it were normal.
            # `spend_usd_billed_upper_est` is exactly what its name says -- an
            # UPPER estimate that assumes each discarded attempt got as far as
            # the surviving one, which is right for a rate-limit storm late in a
            # rollout and too high for a first-call schema rejection. It exists
            # to be looked at, never to be summed as if it were measured.
            "rollouts_paid": int(getattr(result, "rollouts_paid", 1) or 1),
            "retry_errors": list(getattr(result, "retry_errors", []) or []),
            "spend_is_lower_bound": (
                int(getattr(result, "rollouts_paid", 1) or 1) > 1
                or not spend_known),
            "spend_usd_billed_upper_est": round(
                float(estimate.get("spend_so_far_upper_usd") or booked)
                if not spend_known else
                float(result.spend_usd or 0.0)
                * int(getattr(result, "rollouts_paid", 1) or 1), 6),
            "wall_s": round(float(result.wall_s or 0.0), 2),
            "max_dlvl": row_dlvl,
            # Recorded, never smoothed over: a metric and the turn files
            # disagreeing about depth is a fact about the harness, and the one
            # place it can be noticed is here.
            "depth_disagreement": (result.raw or {}).get("depth_disagreement"),
            # Present only when the in-flight guard stopped this attempt. Says
            # what was estimated, against which ceiling, on which rate -- so a
            # `censored:budget_stop` row can be audited rather than believed.
            "budget_stop": (result.raw or {}).get("budget_stop"),
            "max_xl": row_xl,
            "new_checkpoints": [p.name for p in new_dirs],
            "frontier_advanced": bool(advanced),
            "attempts_since_frontier_advance": self._since_advance,
            "cumulative_calls": self.cum_calls,
            "cumulative_spend_usd": round(self.budget.spent_usd, 4),
            "cumulative_wall_s": round(self.cum_wall, 1),
            # Model-written text, quarantined under one key so no aggregation
            # can reach it by accident.
            "model_text": {"summary": result.summary, "lesson": result.lesson},
            "out_dir": str(ctx.out_dir),
        }
        self.attempts.append(record)
        _append_jsonl(cfg.attempts_path, record)
        # The row is final: close the journal entry and stop the progress
        # stream BEFORE the summary, so a crash between them leaves a closed
        # attempt with a stale summary (recoverable) rather than an open one.
        self.close_attempt(record)
        self.write_summary()
        return record

    def _stamp_new_checkpoint(self, path: Path, ctx: PlayerContext,
                              result: PlayerResult,
                              parent_id: Optional[str] = None) -> None:
        """Record which attempt produced a checkpoint, and its lineage.

        The `save` skill knows nothing about attempts or parents -- it writes
        from inside the game. Stamping here is what makes the archive a tree
        rather than a pile. Numeric fields are NOT touched.

        ``parent_id`` comes from :func:`parent_chain`: the PREVIOUS checkpoint
        this same attempt wrote, or the state the attempt resumed for the first
        one. Passing ``ctx.checkpoint_id`` for all of them is what built the
        star this argument exists to replace.
        """
        try:
            meta = checkpoint_meta(path)
        except Exception:
            return
        meta.setdefault("parent", None)
        if parent_id is None:
            parent_id = ctx.checkpoint_id
        if meta.get("parent") is None and parent_id:
            meta["parent"] = str(parent_id)
        meta["created_in_attempt"] = ctx.attempt
        atomic_write(path / META_JSON,
                     json.dumps(meta, indent=2, sort_keys=True) + "\n")

    def _bump_attempts_from(self, path: Path) -> None:
        try:
            meta = checkpoint_meta(path)
        except Exception:
            return
        meta["attempts_from"] = int(meta.get("attempts_from", 0) or 0) + 1
        atomic_write(path / META_JSON,
                     json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # -- reporting --------------------------------------------------------- #

    def luck(self) -> dict:
        """Outcome distribution per source checkpoint (integrity req. 4)."""
        out: dict = {}
        for a in self.attempts:
            key = a["from_checkpoint"] or "(cold start)"
            slot = out.setdefault(key, {"attempts": 0, "outcomes": {},
                                        "censor_reasons": {},
                                        "max_dlvl_per_attempt": []})
            slot["attempts"] += 1
            slot["outcomes"][a["outcome"]] = slot["outcomes"].get(a["outcome"], 0) + 1
            if a["censor_reason"]:
                slot["censor_reasons"][a["censor_reason"]] = \
                    slot["censor_reasons"].get(a["censor_reason"], 0) + 1
            slot["max_dlvl_per_attempt"].append(a["max_dlvl"])
            # Directive + compliance per attempt, so "the same checkpoint, told
            # different things, went differently" is readable -- which is the
            # only way to tell steering apart from re-rolling on ONE state.
            slot.setdefault("directives", []).append({
                "attempt": a["attempt"],
                "directive": a.get("directive", ""),
                "compliance": (a.get("directive_compliance") or {}).get("class"),
                "outcome": a["outcome"],
                "max_dlvl": a["max_dlvl"],
            })
        return out

    def pairs(self) -> dict:
        """Directive/control pairs, grouped, with per-KIND aggregates.

        THIS IS THE ARTIFACT THE PER-KIND CONTROL EXISTS FOR. A run-level
        "directives vs no directives" average pooled a `descend_fast` directive
        that inverted the player's first decision with a `no_descend` directive
        that was indistinguishable from doing nothing -- because the control
        did not descend either. The pooled number said "directives work" while
        half the directive kinds were untested. Grouping by
        ``directive_kind`` and pairing each directive with ITS OWN control is
        what makes "this kind of instruction changes behaviour" answerable.

        ``complete`` is derived here rather than stamped on the attempt record:
        a treatment whose control never ran is not a solo directive attempt and
        must not be analysed as one.
        """
        groups: dict = {}
        for a in self.attempts:
            pid = a.get("pair_id")
            if pid is None:
                continue
            slot = groups.setdefault(pid, {"pair_id": pid, "treatment": None,
                                           "control": None})
            role = a.get("pair_role")
            if role in (ROLE_TREATMENT, ROLE_CONTROL):
                slot[role] = {
                    "attempt": a["attempt"],
                    "from_checkpoint": a["from_checkpoint"],
                    "directive": a.get("directive", ""),
                    "directive_kind": a.get("directive_kind"),
                    "compliance": (a.get("directive_compliance") or {}).get("class"),
                    "by_clause_kind": (a.get("directive_compliance") or {}).get(
                        "by_clause_kind"),
                    "outcome": a["outcome"], "max_dlvl": a["max_dlvl"],
                    "max_xl": a["max_xl"], "calls": a["calls"],
                    "spend_usd": a["spend_usd"],
                }
        by_kind: dict = {}
        out_pairs = []
        for pid in sorted(groups):
            slot = groups[pid]
            slot["complete"] = bool(slot["treatment"] and slot["control"])
            kind = (slot["treatment"] or slot["control"] or {}).get(
                "directive_kind") or "unknown"
            slot["directive_kind"] = kind
            # Same start state, one instruction apart: the delta IS the
            # comparison. Reported only for complete pairs, because a delta
            # against a control that never ran is not a delta.
            if slot["complete"]:
                # A delta against an UNKNOWN is not a delta either: an attempt
                # that left no evidence carries null, and null minus a number
                # is null, never zero.
                def _delta(key, _t=slot["treatment"], _c=slot["control"]):
                    a, b = _t.get(key), _c.get(key)
                    return None if a is None or b is None else a - b

                slot["delta"] = {"max_dlvl": _delta("max_dlvl"),
                                 "max_xl": _delta("max_xl"),
                                 "calls": _delta("calls")}
                k = by_kind.setdefault(kind, {"pairs": 0, "sum_dlvl_delta": 0,
                                              "sum_xl_delta": 0,
                                              "treatment_compliance": {}})
                k["pairs"] += 1
                k["sum_dlvl_delta"] += slot["delta"]["max_dlvl"] or 0
                k["sum_xl_delta"] += slot["delta"]["max_xl"] or 0
                cls = slot["treatment"]["compliance"]
                k["treatment_compliance"][cls] = \
                    k["treatment_compliance"].get(cls, 0) + 1
            out_pairs.append(slot)
        for k in by_kind.values():
            k["mean_dlvl_delta"] = round(k["sum_dlvl_delta"] / k["pairs"], 3)
            k["mean_xl_delta"] = round(k["sum_xl_delta"] / k["pairs"], 3)
        return {
            "enabled": bool(self.cfg.paired_control),
            "pairs": out_pairs,
            "complete_pairs": sum(1 for p in out_pairs if p["complete"]),
            "incomplete_pairs": [p["pair_id"] for p in out_pairs
                                 if not p["complete"]],
            "by_directive_kind": by_kind,
            "note": ("Each directive is paired with its OWN no-directive "
                     "control from the SAME checkpoint. A run-wide "
                     "--no-directive average cannot tell a directive that "
                     "changed behaviour from one that prohibited something the "
                     "player was never going to do."),
        }

    def selection_stats(self) -> dict:
        """The ablation comparison, rolled up from ``selection.jsonl``.

        "The LLM orchestrator steered the search" is only measurable against
        "the scripted softmax would have picked the same thing anyway", and
        until the record carried both picks per round that comparison had to be
        reconstructed by hand. ``dominated_choices`` counts the rounds the
        model chose a state OFF the (Dlvl, XL, score) frontier -- the move the
        old one-row table made impossible.
        """
        rounds = read_jsonl(self.cfg.selection_path)
        rounds = [r for r in rounds if r.get("source") != "matched_restart_fixed_start"]
        agreed = [r for r in rounds if r.get("scripted_agreed") is True]
        differed = [r for r in rounds if r.get("scripted_agreed") is False]
        dominated = [r for r in rounds if r.get("chosen_on_frontier") is False]
        return {
            "rounds": len(rounds),
            "scripted_agreed": len(agreed),
            "scripted_differed": len(differed),
            "dominated_choices": len(dominated),
            "dominated_ids": [r.get("chosen_id") for r in dominated],
            "candidates_shown_per_round": [r.get("n_candidates_shown")
                                           for r in rounds],
            "chosen_ids": [r.get("chosen_id") for r in rounds],
            "scripted_would_pick": [r.get("scripted_would_pick")
                                    for r in rounds],
            "note": ("`chosen_id` is the id that was LAUNCHED; "
                     "`scripted_would_pick` is what the softmax ablation would "
                     "have chosen from the same archive. They were the same "
                     "field once, and the scripted pick won."),
        }

    def summary(self) -> dict:
        rows = self.rows()
        best = max(rows, key=lambda r: r.key) if rows else None
        died = sum(1 for a in self.attempts if a["outcome"] == OUTCOME_DIED)
        censored = sum(1 for a in self.attempts if a["outcome"] == OUTCOME_CENSORED)
        out = {
            "run_dir": str(self.cfg.run_dir),
            # LOUD, AND FIRST. A decide-only run has no players in it; nothing
            # downstream may read its attempt rows as measurements.
            "mode": "decide_only" if self.decide_only else "full",
            "synthetic": bool(self.decide_only),
            "synthetic_note": SYNTHETIC_NOTE if self.decide_only else "",
            "experiment_arm": self.cfg.arm,
            "player_arm": self.cfg.player_arm,
            "reseed_on_restore": self.cfg.reseed_on_restore,
            "attempts": len(self.attempts),
            "attempts_died": died,
            "attempts_censored": censored,
            "attempts_ascended": sum(1 for a in self.attempts
                                     if a["outcome"] == OUTCOME_ASCENDED),
            "censor_reasons": _count(a["censor_reason"] for a in self.attempts
                                     if a["censor_reason"]),
            "checkpoints": len(rows),
            "frontier": sorted((r.id for r in pareto_frontier(rows)), key=_id_key),
            "cumulative_calls": self.cum_calls,
            # TWO LINES. An LLM orchestrator is not free, and a single total
            # would hide which half of the ceiling the run actually spent.
            "budget": self.budget.lines(),
            "cumulative_spend_usd": round(self.budget.spent_usd, 4),
            "budget_ceiling_usd": self.cfg.budget_ceiling_usd,
            "orchestrator": (self.session.budget_lines()
                             if self.session is not None else
                             {"kind": "scripted", "spend_usd": 0.0}),
            "selector": self.cfg.selector,
            "llm_fallbacks": self.llm_fallbacks,
            "directive_compliance": _count(
                (a.get("directive_compliance") or {}).get("class", COMPLY_UNKNOWN)
                for a in self.attempts),
            # PER CLAUSE KIND, beside the headline rather than instead of it.
            # The headline is derived; these are what it was derived from, and
            # a run whose prohibitions were all obeyed while every goal was
            # prevented is a different result from one where both failed.
            "compliance_by_clause_kind": {
                kind: _count(
                    ((a.get("directive_compliance") or {})
                     .get("by_clause_kind") or {}).get(kind, {}).get("label", "n/a")
                    for a in self.attempts)
                for kind in (KIND_PROHIBITION, KIND_GOAL)
            },
            "directive_kinds": _count(a.get("directive_kind") or "none"
                                      for a in self.attempts),
            # The phrasing lint, aggregated. A run whose directives were mostly
            # tool-prohibitions has a compliance column that cannot be read.
            "directive_lint": _count(
                w["code"] for a in self.attempts
                for w in (a.get("directive_lint") or [])),
            "instrumental_ambiguities": sum(
                1 for a in self.attempts
                if (a.get("directive_compliance") or {}).get(
                    "instrumental_ambiguity")),
            "matched_pairs": self.pairs(),
            "selection_sources": _count(a.get("selection_source") or "scripted"
                                        for a in self.attempts),
            "selection": self.selection_stats(),
            "wall_clock_s": round(time.time() - self.started_at, 1),
            "stop_reason": self.stop_reason,
            "frontier_advances": self.frontier_advances,
            "luck": self.luck(),
            # The orchestrator's OWN health, reported next to the run's, so a
            # run that stopped because its steering broke does not have to be
            # diagnosed from the round log.
            "orchestrator_error": self.orchestrator_error,
            "opening_plan": ({"attempts": len(self.opening_attempts),
                              "degeneration":
                                  self.opening_attempts[-1]["degeneration"],
                              "attempts_detail": self.opening_attempts}
                             if self.opening_attempts else None),
            # -- DURABILITY, reported rather than assumed ------------------- #
            # `in_flight` is what makes a summary written mid-attempt honest:
            # it names the attempt that has no final row yet, so a reader of a
            # summary from a run that later died can tell "attempt 1 was still
            # playing" apart from "there was never an attempt 1".
            "in_flight": ({"attempt": self._open.get("attempt"),
                           "from_checkpoint": self._open.get("from_checkpoint"),
                           "started_iso": self._open.get("iso"),
                           "pid": self._open.get("pid")}
                          if self._open else None),
            "resumed": self.resumed,
            "interrupted_by": self.interrupted_by,
            "attempts_recovered": sum(1 for a in self.attempts
                                      if a.get("recovered")),
            # An attempt whose spend could not be recovered makes the run's
            # total a LOWER BOUND, and a budget ceiling read off a lower bound
            # is not a ceiling. Counted here so nobody has to notice it.
            #
            # It used to be `recovered AND NOT spend_known`, which could only
            # ever count the crash-recovery path -- so treesmoke2's three
            # censored attempts, none of them `recovered`, reported
            # `player_usd: 0.0` alongside `attempts_with_unknown_spend: 0`:
            # a run that spent real money on 205 turns of play, asserting it
            # spent nothing and that the nothing was measured. The `recovered`
            # conjunct is gone; `spend_known` is now written by every row
            # writer and is the whole test.
            "attempts_with_unknown_spend": sum(
                1 for a in self.attempts if not a.get("spend_known", True)),
            # The same statement for the play numbers: how many attempts left
            # no evidence at all, so their `calls` / `max_dlvl` / `max_xl` are
            # null rather than measured. `cumulative_calls` and
            # `best_state.calls_to_reach` are lower bounds while this is
            # non-zero.
            "attempts_with_unknown_progress": sum(
                1 for a in self.attempts if a.get("calls") is None),
            "counts_are_lower_bounds": any(
                (not a.get("spend_known", True))
                or a.get("evidence_source") in ("turns", "none")
                for a in self.attempts),
            "recovery": self.recovery or None,
            "unattributed_checkpoints":
                list((self.recovery or {}).get("unattributed_checkpoints") or []),
        }
        if best is not None:
            # COMPARABILITY: the best-state claim NEVER travels alone.
            out["best_state"] = {
                "id": best.id, "dlvl": best.dlvl, "xl": best.xl,
                "score": best.score, "balrog": best.balrog,
                "balrog_min": best.balrog_min,
                "dungeon_number": best.dungeon_number,
                "attempts_to_reach": _attempts_to_reach(self.frontier_advances,
                                                        best.id, len(self.attempts)),
                "calls_to_reach": _field_to_reach(self.frontier_advances, best.id,
                                                  "cumulative_calls", self.cum_calls),
                "spend_to_reach_usd": _field_to_reach(self.frontier_advances, best.id,
                                                      "cumulative_spend_usd",
                                                      round(self.budget.spent_usd, 4)),
            }
        out["comparability_note"] = (
            "Go-Explore depth is NOT comparable to single-life BALROG depth. "
            "best_state was reached after attempts_to_reach separate resumed "
            "attempts from a curated archive; a BALROG number is one life. Any "
            "comparison must state both numbers or it is wrong."
        )
        out["comparable_to_single_life_balrog"] = False
        return out

    def write_summary(self) -> dict:
        s = self.summary()
        atomic_write(self.cfg.summary_path,
                     json.dumps(s, indent=2, sort_keys=True, default=str) + "\n")
        atomic_write(self.cfg.luck_path,
                     json.dumps(self.luck(), indent=2, sort_keys=True) + "\n")
        return s

    # -- the loop ---------------------------------------------------------- #

    # -- resuming ----------------------------------------------------------- #

    def resume(self) -> dict:
        """Pick a dead run back up without losing or re-spending anything.

        Order matters. Reconciliation runs FIRST, so the attempt that was in
        flight when the last process died is a finalized `censored:interrupted`
        row -- and its checkpoints are attributed -- before any counter is
        rebuilt from those rows. Then:

        * the attempt rows, the call/wall/spend counters and the frontier
          history are read back from `attempts.jsonl`, so cumulative spend is
          CONTINUOUS across the restart and nothing is charged twice;
        * the orchestrator's conversation is re-attached by its recorded
          session id, so the optimization continues instead of restarting --
          which is the whole reason the session is persistent;
        * `opened` is set, because the opening plan is already on the record
          and paying for a second one would both cost money and replace the
          plan every earlier attempt was steered by.
        """
        cfg = self.cfg
        live = live_orchestrator_pids(cfg.run_dir)
        if live:
            raise RunStillLive(
                f"REFUSING to resume {cfg.run_dir}: orchestrator pid(s) "
                f"{live} are still running against it. Two orchestrators on "
                f"one archive would select from each other's checkpoints, "
                f"charge one budget twice, and write attempt rows with "
                f"colliding ids. Stop the running one first.")
        self.resumed = True
        self.recovery = reconcile_run(cfg)

        rows = read_jsonl(cfg.attempts_path)
        rows.sort(key=lambda r: int(r.get("attempt") or 0))
        self.attempts = rows
        self.cum_calls = sum(int(r.get("calls") or 0) for r in rows)
        self.cum_wall = sum(float(r.get("wall_s") or 0.0) for r in rows)
        self.budget.player_usd = sum(float(r.get("spend_usd") or 0.0) for r in rows)
        # A resumed run inherits the LOWER-BOUND-ness of what it is resuming:
        # rebuilt from the same rows, so the second half of a run cannot quietly
        # report a measured total the first half never had.
        unknown = [r for r in rows if not r.get("spend_known", True)]
        self.budget.player_unknown_attempts = len(unknown)
        self.budget.player_estimated_usd = sum(
            float(r.get("spend_usd") or 0.0) for r in unknown)
        # THE ORCHESTRATOR'S OWN LINE, recovered from its round log rather than
        # from the summary: the round log is append-only and fsynced per round,
        # so it survives exactly the crashes the summary does not.
        self.budget.orchestrator_usd = sum(
            float(r.get("spend_usd") or 0.0) for r in read_jsonl(cfg.orchestrator_log))

        prior = {}
        try:
            prior = json.loads(cfg.summary_path.read_text())
        except Exception:
            prior = {}
        self.frontier_advances = list(prior.get("frontier_advances") or [])
        self.llm_fallbacks = int(prior.get("llm_fallbacks") or 0)
        self.opening_attempts = list((prior.get("opening_plan") or {}).get(
            "attempts_detail") or [])
        # Wall clock CONTINUES rather than restarting from zero: a resumed run
        # that reported its own uptime would understate what the result cost.
        self.started_at = time.time() - float(prior.get("wall_clock_s") or 0.0)
        # Attempts since the last frontier advance, recomputed from the rows.
        since = 0
        for r in reversed(rows):
            if r.get("frontier_advanced"):
                break
            since += 1
        self._since_advance = since
        self._pair_counter = max((int(r.get("pair_id") or 0) for r in rows),
                                 default=0)

        sess = self.restore_session()
        state = {
            "resumed": True,
            "attempts_recovered": len(rows),
            "recovery": self.recovery,
            "budget": self.budget.lines(),
            "session": sess,
        }
        self.write_summary()
        return state

    def restore_session(self) -> dict:
        """Re-attach the orchestrator conversation by its recorded session id.

        Read out of `orchestrator_rounds.jsonl`, which every round appends to
        and fsyncs, rather than out of provenance -- provenance is rewritten at
        the END of a run and a run that died never wrote it.
        """
        out = {"attached": False, "session_id": "", "rounds": 0}
        if self.session is None:
            out["reason"] = "scripted selector: no session to resume"
            return out
        rounds = read_jsonl(self.cfg.orchestrator_log)
        # THE PROBE IS NOT THE RUN. `--probe-session` opens its own throwaway
        # conversation in the same agent dir and logs it here; resuming THAT id
        # would continue a two-sentence continuity check instead of the run's
        # strategy. Its spend still counts (it was billed), its id does not.
        chain = [r for r in rounds
                 if not str(r.get("kind") or "").startswith("probe")]
        ids = [r.get("session_id") for r in chain if r.get("session_id")]
        if not ids:
            out["reason"] = "no session id on record; the next round opens one"
            return out
        self.session.session_id = ids[-1]
        self.session.session_ids = list(dict.fromkeys(ids))
        self.session.rounds = max((int(r.get("round") or 0) for r in chain),
                                  default=0)
        self.session.sent_chars = max(
            (int(r.get("sent_chars_cumulative") or 0) for r in chain), default=0)
        self.session.spend_usd = sum(float(r.get("spend_usd") or 0.0)
                                     for r in rounds)
        self.session.compactions = sum(1 for r in chain
                                       if r.get("kind") == "compaction")
        for r in reversed(chain):
            if r.get("session_cwd"):
                self.session.session_cwd = r["session_cwd"]
                break
            if r.get("session_file"):
                self.session.session_file = r["session_file"]
        # The opening plan is already bought and already on the record.
        self.opened = any(str(r.get("kind") or "").startswith("opening")
                          for r in rounds)
        out.update({"attached": True, "session_id": self.session.session_id,
                    "session_ids": list(self.session.session_ids),
                    "rounds": self.session.rounds,
                    "sent_chars": self.session.sent_chars,
                    "opening_already_bought": self.opened})
        return out

    # -- the loop ----------------------------------------------------------- #

    def install_signal_handlers(self) -> list:
        """Turn SIGTERM/SIGINT into an exception that unwinds the stack.

        Python's DEFAULT SIGTERM disposition terminates the process without
        running one `finally`, so an operator's `kill` and a box teardown both
        destroy the in-flight attempt's record. Raising instead means the
        launch path's `except BaseException` finalizes the row, the archive's
        new checkpoints are attributed to it, and the summary is flushed --
        before the process goes.
        """
        installed = []
        for signame in ("SIGTERM", "SIGINT", "SIGHUP"):
            signum = getattr(signal, signame, None)
            if signum is None:
                continue

            def handler(num, _frame, _name=signame):
                self.interrupted_by = _name
                raise RunInterrupted(_name, num)

            try:
                signal.signal(signum, handler)
                installed.append(signame)
            except (ValueError, OSError):
                # Not the main thread (a test, an embedded caller): the loop
                # still works, it just cannot intercept that signal.
                continue
        return installed

    def run(self, max_attempts: Optional[int] = None) -> dict:
        limit = max_attempts if max_attempts is not None else self.cfg.max_attempts
        self.install_signal_handlers()
        # summary.json exists from the START, not from the first completed
        # attempt. A run that dies in its opening round now leaves a summary
        # that says so instead of leaving an empty directory.
        self.write_summary()
        try:
            return self._run(limit)
        finally:
            # THE BACKSTOP. Whatever left the loop -- a signal, a raise, an
            # orchestrator failure -- an attempt that is still open is finalized
            # here rather than lost. Idempotent: the launch path usually got
            # there first and `_open` is already None.
            if self._open is not None:
                self.finalize_interrupted(
                    detail=(f"the orchestrator exited on {self.interrupted_by} "
                            f"while this attempt was in flight")
                    if self.interrupted_by else
                    "the orchestrator exited while this attempt was in flight")
            # The lock goes LAST, after the record is final: a reader that sees
            # no lock must be able to trust that nothing is still being written.
            clear_run_lock(self.cfg.run_dir)

    def _run(self, limit: int) -> dict:
        # ROUND 1 IS DISCUSSION. The plan goes on the record before any player
        # is launched, so "did it follow its own strategy?" stays answerable.
        #
        # AN ORCHESTRATOR FAILURE STOPS THE RUN AND IS WRITTEN DOWN. It is
        # re-raised, never converted into a stop_reason and swallowed -- the
        # caller has to see it -- but the summary is flushed first, so the run
        # directory says WHY it stopped instead of just ending mid-file.
        try:
            self.open_discussion()
        except OrchestratorRoundFailed as exc:
            self.stop_reason = STOP_ORCHESTRATOR_FAILED
            self.orchestrator_error = f"{type(exc).__name__}: {exc}"
            self.write_summary()
            raise
        while len(self.attempts) < limit:
            rows = self.rows()
            stop = self.should_stop(rows)
            if stop:
                self.stop_reason = stop
                break
            try:
                self.run_attempt(rows)
            except BudgetExceeded:
                self.stop_reason = STOP_BUDGET
                break
            except OrchestratorRoundFailed as exc:
                self.stop_reason = STOP_ORCHESTRATOR_FAILED
                self.orchestrator_error = f"{type(exc).__name__}: {exc}"
                self.write_summary()
                raise
        else:
            self.stop_reason = self.stop_reason or STOP_MAX_ATTEMPTS
        self.finish()
        return self.write_summary()

    # -- the decision loop, with no players -------------------------------- #

    def run_decide_only(self, outcomes, rounds: Optional[int] = None) -> dict:
        """The orchestrator's ROUNDS, against synthetic attempt outcomes.

        WHY THIS EXISTS. A player attempt costs ~$31 and half an hour; an
        orchestrator decision round costs ~$0.08. Until this mode there was no
        supported way to exercise the decision loop at all -- `should_stop`
        gates the round and the launch behind the same check -- so the question
        the whole selection redesign was for ("does it USE the freedom to pick
        a dominated checkpoint, branch, and adapt to outcomes?") could only be
        asked at 400x the price of the answer.

        Each round is the real thing: the real `decide`, the real ledger built
        from a real archive, the real parser, the real `selection.jsonl`
        record. What is fake is only what comes BACK -- the attempt outcome,
        which the caller supplies. `outcomes` is one spec per round::

            {"outcome": "died", "censor_reason": "", "max_dlvl": 8,
             "max_xl": 4, "calls": 120, "summary": "...", "lesson": "...",
             "compliance": "ignored", "spend_usd": 31.0}

        NOTHING HERE CAN BE MISTAKEN FOR A REAL RUN. Every record this writes
        -- the selection line, the attempt row, the lesson appended to the
        checkpoint, the summary -- carries ``synthetic: true``, and the run
        summary carries ``mode: decide_only``. The player budget line stays at
        zero because no player ran: a spec's ``spend_usd`` is recorded on the
        row as ``synthetic_spend_usd`` and is NOT added to the budget, because
        adding it would make the ceiling stop a run that had spent nothing.

        `outcomes` may instead be a CALLABLE ``(round, choice, attempts) ->
        spec``, with ``rounds`` saying how many to run. A static list cannot
        probe the behaviours this mode exists for: "the state it just chose has
        now died there twice" is a fact about what it chose, and writing the
        sequence in advance means either guessing its choices or feeding back
        outcomes that do not match them. The CLI passes a list, because a JSON
        file is what an operator has; a study driver passes a function.

        NO STOP CONDITIONS EXCEPT THE BUDGET. `should_stop` is deliberately not
        consulted: the milestone and stall rules exist to stop paying for
        players, and here there are none. The round count is the caller's.
        """
        self.decide_only = True
        if callable(outcomes):
            if not rounds:
                raise ValueError("a callable outcome source needs `rounds`")
            plan = [None] * int(rounds)
            outcome_for = outcomes
        else:
            plan = list(outcomes)
            outcome_for = lambda n, choice, attempts: plan[n - 1]  # noqa: E731
        if self.session is None:
            raise ValueError("--decide-only needs the LLM selector: with "
                             "--selector scripted there is no decision round "
                             "to exercise, only the softmax.")
        self.install_signal_handlers()
        self.write_summary()
        try:
            try:
                self.open_discussion()
            except OrchestratorRoundFailed as exc:
                self.stop_reason = STOP_ORCHESTRATOR_FAILED
                self.orchestrator_error = f"{type(exc).__name__}: {exc}"
                self.write_summary()
                raise
            for _slot in plan:
                # THE CEILING, ON A DECISION ROUND'S SCALE. `budget.check()`
                # refuses below `min_headroom_usd`, which is sized for LAUNCHING
                # AN UNCAPPED PLAYER ($5 by default) -- against a decide-only
                # ceiling of $8 or $15 that gate would refuse every round while
                # the actual cost of one is a few cents. There is no launch here
                # to reserve headroom for, so the ceiling is the ceiling.
                if self.budget.remaining <= self.cfg.decide_only_headroom_usd:
                    self.stop_reason = STOP_BUDGET
                    break
                rows = self.rows()
                if not rows:
                    self.stop_reason = STOP_EMPTY_ARCHIVE
                    break
                n = len(self.attempts) + 1
                try:
                    choice = self.decide(rows)
                except OrchestratorRoundFailed as exc:
                    self.stop_reason = STOP_ORCHESTRATOR_FAILED
                    self.orchestrator_error = f"{type(exc).__name__}: {exc}"
                    self.write_summary()
                    raise
                rec = selection_record(n, choice, rows,
                                       frontier=self.frontier_ids(rows))
                rec["synthetic"] = True
                rec["synthetic_note"] = SYNTHETIC_NOTE
                _append_jsonl(self.cfg.selection_path, rec)
                spec = outcome_for(n, choice, list(self.attempts))
                self.ingest_synthetic(choice, dict(spec or {}))
            else:
                self.stop_reason = self.stop_reason or STOP_MAX_ATTEMPTS
        finally:
            clear_run_lock(self.cfg.run_dir)
        self.finish()
        return self.write_summary()

    def ingest_synthetic(self, choice: dict, spec: dict) -> dict:
        """Fold one SYNTHETIC attempt outcome back into the record.

        The same shape `ingest` produces, minus everything that could only come
        from a player: no new checkpoints, no compliance rubric (the spec says
        what happened instead), no player spend. `attempts_from` is bumped and
        the lesson IS appended to the source checkpoint, because both of those
        are what the orchestrator reads next round -- a probe that skipped them
        would be probing a different ledger than the one a real run shows.
        """
        n = len(self.attempts) + 1
        chosen_id = choice.get("checkpoint_id")
        directive = choice.get("directive") or ""
        ck_dir = next((r.path for r in self.rows() if r.id == chosen_id), None)
        outcome = str(spec.get("outcome") or OUTCOME_DIED)
        censor = str(spec.get("censor_reason") or "")
        compliance = {"class": str(spec.get("compliance") or "not_scored"),
                      "source": "synthetic: supplied by the operator, not "
                                "measured from any player's calls"}
        if ck_dir is not None:
            self._bump_attempts_from(ck_dir)
            body = (f"{SYNTHETIC_NOTE}\n\n"
                    f"DIRECTIVE GIVEN: {directive or '(none)'}\n"
                    f"WHAT THE (SYNTHETIC) ATTEMPT DID: "
                    f"{compliance['class']}\n\n"
                    + str(spec.get("summary") or "(no summary)").strip())
            if spec.get("lesson"):
                body += "\n\nLESSON: " + str(spec["lesson"]).strip()
            append_lesson(ck_dir, body,
                          heading=f"attempt {n} (SYNTHETIC, {outcome}"
                                  + (f":{censor}" if censor else "") + ")")
        record = {
            "attempt": n,
            "synthetic": True,
            "synthetic_note": SYNTHETIC_NOTE,
            "from_checkpoint": chosen_id,
            "directive": directive,
            "directive_compliance": compliance,
            "directive_kind": classify_directive_kind(directive),
            "directive_lint": lint_directive(directive),
            "experiment_arm": self.cfg.arm,
            "pair_id": None, "pair_role": ROLE_SOLO,
            "reseed": None,
            "selection_source": choice.get("source"),
            "orchestrator_decision": choice.get("decision"),
            "outcome": outcome,
            "censored": outcome == OUTCOME_CENSORED,
            "censor_reason": censor,
            "stop_condition": str(spec.get("stop_condition") or outcome),
            "error": str(spec.get("error") or ""),
            "calls": int(spec.get("calls") or 0),
            # NOT ON THE BUDGET. Named so it can never be summed as if it were.
            "spend_usd": 0.0,
            "synthetic_spend_usd": float(spec.get("spend_usd") or 0.0),
            "rollouts_paid": 0,
            "wall_s": 0.0,
            "max_dlvl": int(spec.get("max_dlvl") or 0),
            "max_xl": int(spec.get("max_xl") or 1),
            "new_checkpoints": [],
            "frontier_advanced": False,
            "attempts_since_frontier_advance": self._since_advance,
            "cumulative_calls": self.cum_calls,
            "cumulative_spend_usd": round(self.budget.spent_usd, 4),
            "cumulative_wall_s": round(self.cum_wall, 1),
            "model_text": {"summary": spec.get("summary") or "",
                           "lesson": spec.get("lesson") or ""},
            "out_dir": "",
        }
        self._since_advance += 1
        self.attempts.append(record)
        _append_jsonl(self.cfg.attempts_path, record)
        self.write_summary()
        return record


SYNTHETIC_NOTE = ("SYNTHETIC. No player was launched and no player budget was "
                  "spent; the outcome fed back to the orchestrator was written "
                  "by the operator to probe the decision loop.")


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _attempts_to_reach(advances: list, ident: str, default: int) -> int:
    for a in advances:
        if ident in a.get("new_frontier_ids", []):
            return a["attempts_consumed"]
    return default


def _field_to_reach(advances: list, ident: str, key: str, default):
    for a in advances:
        if ident in a.get("new_frontier_ids", []):
            return a.get(key, default)
    return default


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line durably. Never leaves a partial line behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    with open(path, "a") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------------------------------- #
# the real player: one eval rollout, launched through launch_cell.sh
# --------------------------------------------------------------------------- #

class SubprocessPlayer:
    """Launch one uncapped eval rollout resumed from a checkpoint.

    Everything that makes this cell what it is goes through `launch_cell.sh`
    and the tier registry -- model, character, variant, tune, doc, the
    fix-flags -- exactly as every other cell in this tree does, so an E16
    attempt is a normal cell with three extra env_args. The E16-only arguments
    (`resume_checkpoint`, `checkpoint_archive`, `wiki_dir`, `ledger_text`,
    `fidelity_log`) travel through `E16_ARGS`, a whitelisted JSON knob the
    launcher accepts ALONGSIDE `TOOL_TIER` because none of those dotted paths
    is one any tier writes (unlike `ENV_ARGS`, which the launcher refuses next
    to a tier for exactly that reason).
    """

    def __init__(self, repo: Path = REPO, timeout_s: float = 30 * 3600,
                 env: Optional[dict] = None):
        self.repo = Path(repo)
        self.timeout_s = timeout_s
        self.env = env
        #: The running `launch_cell.sh` (which `exec`s the eval CLI, so this
        #: handle IS the eval process). Published so the orchestrator's
        #: in-flight budget guard can end the attempt from the monitor thread.
        self._active: Optional[subprocess.Popen] = None

    def terminate_active(self, grace_s: float = 60.0) -> bool:
        """End the running attempt: SIGTERM, then SIGKILL after `grace_s`.

        SIGTERM first and with a real grace period because the eval CLI writes
        `traces.jsonl` on the way out -- a clean stop is the difference between
        a stopped attempt whose spend is KNOWN and one more row carrying
        `spend_known: false`, which is the accounting hole this whole area
        exists to close. Deliberately NOT `start_new_session`: the player stays
        in the orchestrator's process group so an operator's Ctrl-C still
        reaches it, which `finalize_interrupted` depends on.

        Returns True if there was something to stop. Safe to call from another
        thread and safe to call when nothing is running.
        """
        proc = self._active
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.terminate()
        except OSError:
            return False
        deadline = time.monotonic() + max(0.0, float(grace_s))
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return True
            time.sleep(0.2)
        try:
            proc.kill()
        except OSError:
            pass
        return True

    def __call__(self, ctx: PlayerContext) -> PlayerResult:
        e16 = {
            "checkpoint_archive": str(ctx.archive_dir),
            "wiki_dir": str(ctx.wiki_dir),
            "ledger_text": ctx.ledger_text,
            "fidelity_log": str(ctx.fidelity_log),
            # Part of the launch contract, not an extra: nethack.py renders it
            # into the FIRST served observation using DIRECTIVE_BLOCK_FORMAT.
            "directive": ctx.directive,
        }
        if ctx.checkpoint_dir is not None:
            e16["resume_checkpoint"] = str(ctx.checkpoint_dir)
        if ctx.reseed is not None:
            # Only ever present when --reseed was asked for, so a run that did
            # not ask for it cannot acquire it through a stale default. Sent as
            # a JSON STRING because launch_cell's E16_ARGS whitelist requires
            # scalars (env_args reach the eval CLI as dotted scalars anyway).
            e16["reseed"] = json.dumps([int(ctx.reseed[0]), int(ctx.reseed[1])])
        env = dict(self.env or os.environ)
        env.update({
            "TOOL_TIER": ctx.tier,
            "SEEDS": json.dumps([ctx.game_seed]),
            "E16_ARGS": json.dumps(e16),
            "STALL_WATCHDOG": env.get("STALL_WATCHDOG", "1"),
        })
        cmd = [str(self.repo / "tools" / "cli_harness_eval" / "launch_cell.sh"),
               ctx.arm, str(ctx.out_dir), "0", "1"]
        t0 = time.time()
        # Popen, not `subprocess.run`: the handle has to be reachable from the
        # progress-monitor thread so the in-flight budget guard can stop the
        # attempt. Same semantics otherwise -- captured output, same timeout,
        # same process group (see `terminate_active`).
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        self._active = proc
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        finally:
            self._active = None
        (ctx.out_dir / "launch.log").write_text((stdout or "") + "\n" + (stderr or ""))
        res = read_trace_result(ctx.out_dir)
        res.exit_code = proc.returncode
        res.wall_s = time.time() - t0
        if proc.returncode != 0 and not res.error:
            res.error = f"launch_cell exited {proc.returncode}"
            res.stop_condition = res.stop_condition or "error"
        return res


# `verifiers/v1/retries.py:run_with_retry` logs exactly this when it throws a
# rollout away and replays the whole trajectory:
#     retrying rollout <trace-id> (retry 1/2) after error: ProviderError
_ROLLOUT_RETRY_RE = re.compile(
    r"retrying rollout (\S+) \(retry (\d+)/(\d+)\) after error: (\S+)")


def rollouts_billed(out_dir) -> dict:
    """How many FULL rollouts this attempt actually paid for.

    THE PROBLEM THIS EXISTS FOR. A whole-rollout retry replays the trajectory
    from turn 1, but `run_with_retry` returns only the LAST attempt's trace --
    the discarded attempts' per-call `usage` never reaches `traces.jsonl`, so
    `read_trace_result` costs one rollout for an attempt that billed two or
    three. Measured: $31.43 for one `censored:harness_error` attempt that ran
    `retry 1/2` and `retry 2/2`. Averaged into a per-attempt cost curve
    unlabelled, that silently understates the run's real burn rate.

    The only surviving record of those discarded attempts is the eval CLI's own
    warning line, which `SubprocessPlayer` captures into `launch.log`. That is
    the source used here -- no patching of vendored `verifiers`, and it is
    written whether the attempt succeeded on retry or failed on all of them
    (the trace's `errors` are NOT: `run_with_retry` prepends the retry history
    only `if trace.errors`, so a rollout that SUCCEEDS on retry 2 leaves no
    trace-side evidence at all).

    An E16 attempt is one rollout (`SubprocessPlayer` launches n=1), so
    `rollouts_paid` is exact here; with more rollouts per cell it is a lower
    bound on the billed count, never an over-claim.
    """
    out = {"rollouts_paid": 1, "retries_observed": 0, "retry_errors": [],
           "source": "launch.log"}
    try:
        text = (Path(out_dir) / "launch.log").read_text(errors="replace")
    except OSError:
        out["source"] = "unavailable"   # no log == no evidence, not "no retries"
        return out
    errs = [m.group(4) for m in _ROLLOUT_RETRY_RE.finditer(text)]
    out["retries_observed"] = len(errs)
    out["rollouts_paid"] = 1 + len(errs)
    out["retry_errors"] = errs
    return out


def _unreported_result(out_dir: Path, error: str, billed: dict) -> PlayerResult:
    """The result for an attempt whose ROLLOUT never reported.

    `traces.jsonl` is written when the rollout finishes. When the env tool
    server dies under it, or the launcher is killed, the file is missing or
    empty and there is no report -- but the attempt still PLAYED, and the env
    wrote every turn of it to `turns/` (or, after the stall watchdog, to
    `turns.stalled/`) and every checkpoint of it to the archive.

    This used to return a bare `PlayerResult()`, whose dataclass defaults are
    `calls=0`, `spend_usd=0.0`, `max_dlvl=0`. Those zeros then flowed into the
    budget, the cumulative call count, and `luck.max_dlvl_per_attempt` as if
    they had been measured. treesmoke2 is the measured case: 3 attempts, 205 LM
    turns, Dlvl 4, XL 5, 16 checkpoints -- reported as `player_usd: 0.0`,
    `cumulative_calls: 0`, `max_dlvl_per_attempt: [0]`, with
    `attempts_with_unknown_spend: 0` claiming the zeros were solid.

    So: recover what the env's own files prove (calls, depth, XL, turns) and
    say plainly that spend was never measured. `evidence_source` says which
    channel every number came from, and is "none" when the attempt really did
    leave nothing -- the only case in which those fields are honestly unknown.
    """
    ev = _attempt_dir_evidence(Path(out_dir))
    has = bool(ev.get("turns") or ev.get("calls"))
    return PlayerResult(
        stop_condition="error",
        error=error,
        env_crash=env_crash_evidence(out_dir),
        calls=int(ev.get("calls") or 0),
        max_dlvl=int(ev.get("max_dlvl") or 0),
        max_xl=int(ev.get("max_xl") or 1),
        # NEVER a measurement: no trace means no token usage, and token usage
        # is the only thing this tree ever costs from.
        spend_usd=0.0,
        spend_known=False,
        evidence_source="turns" if has else "none",
        rollouts_paid=int(billed["rollouts_paid"]),
        retry_errors=list(billed["retry_errors"]),
        raw={"attempt_dir_evidence": ev, "rollouts_billed": billed},
    )


def read_trace_result(out_dir) -> PlayerResult:
    """Read one rollout's `traces.jsonl` into a :class:`PlayerResult`.

    Every number comes from the trace the harness wrote (`metrics.*`) or from
    the per-call token `usage` (through aggregate.rollout_cost, the same
    costing the rest of this tree uses -- prompt and cached-input tokens are
    disjoint and additive, which a hand-rolled sum gets wrong by ~50%).
    The model's own prose is read from the final assistant message and kept in
    `summary`, where nothing numeric can reach it.
    """
    out_dir = Path(out_dir)
    path = out_dir / "traces.jsonl"
    billed = rollouts_billed(out_dir)
    if not path.is_file():
        return _unreported_result(out_dir, f"no traces.jsonl in {out_dir}", billed)
    traces = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not traces:
        return _unreported_result(out_dir, f"empty traces.jsonl in {out_dir}", billed)
    trace = traces[-1]
    metrics = trace.get("metrics") or {}
    # DEPTH IS CROSS-CHECKED, NOT TRUSTED. `metrics.max_dlvl_reached` is what
    # the env published when it finalized, and on a harness-aborted attempt it
    # has been observed reporting a depth the rollout's own turn files never
    # reach (6 against a run that never left Dlvl 3). The turn files are the
    # per-turn record of what the engine actually showed, so take the larger of
    # "what the metrics say" and "what the turns show" only when they AGREE,
    # and record the disagreement otherwise rather than silently picking one.
    turns_dlvl = _max_dlvl_from_turns(out_dir)
    metrics_dlvl = int(metrics.get("max_dlvl_reached") or 0)
    depth_disagreement = None
    if turns_dlvl is not None and turns_dlvl != metrics_dlvl:
        depth_disagreement = {"metrics": metrics_dlvl, "turns": turns_dlvl}
    # The turn files win: they are per-turn engine observations, while the
    # metric is one number written once at the end.
    max_dlvl = turns_dlvl if turns_dlvl is not None else metrics_dlvl
    costed = True
    try:
        from aggregate import price_table_for, rollout_cost  # same dir
        spend = rollout_cost(trace, price_table_for(trace)) or 0.0
    except Exception:
        try:
            from aggregate import PRICE_TABLE_GLM_5_2, rollout_cost
            spend = rollout_cost(trace, PRICE_TABLE_GLM_5_2) or 0.0
        except Exception:
            # A COSTING FAILURE, not a free rollout. Both price tables refused
            # a trace that exists and has token usage in it, so the number
            # below is the absence of an answer -- `spend_known: false` keeps
            # it out of the measured total exactly as an absent trace does.
            spend, costed = 0.0, False
    return PlayerResult(
        stop_condition=str(trace.get("stop_condition") or ""),
        died=bool(metrics.get("died")),
        ascended=bool(metrics.get("ascended")),
        error=str(trace.get("error") or ""),
        calls=int(metrics.get("skill_calls") or 0),
        spend_usd=float(spend),
        spend_known=costed,
        max_dlvl=max_dlvl,
        max_xl=int(metrics.get("max_xp_level") or 1),
        summary=_final_text(trace),
        rollouts_paid=int(billed["rollouts_paid"]),
        retry_errors=list(billed["retry_errors"]),
        raw={"metrics": metrics, "depth_disagreement": depth_disagreement,
             "rollouts_billed": billed},
    )


def _max_dlvl_from_turns(out_dir) -> Optional[int]:
    """The deepest Dlvl the per-turn NDJSON actually recorded, or None.

    `helpers._write_trace_entry` stamps `dlvl` and `max_dlvl_reached` on every
    turn record straight off the shaped observation, so this is the engine's
    own account of where the hero went -- independent of whatever the
    end-of-rollout metric computed.
    """
    best = None
    for path in all_turn_files(out_dir):
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("max_dlvl_reached", "dlvl"):
                v = rec.get(key)
                if isinstance(v, (int, float)) and v:
                    best = int(v) if best is None else max(best, int(v))
    return best


def _final_text(trace: dict, limit: int = 4000) -> str:
    """The player's last assistant message -- its own account of the game."""
    for key in ("completion", "messages", "nodes"):
        node = trace.get(key)
        if isinstance(node, list) and node:
            for item in reversed(node):
                if isinstance(item, dict) and item.get("role") == "assistant":
                    content = item.get("content")
                    if isinstance(content, str) and content.strip():
                        return content[:limit]
    return ""


# --------------------------------------------------------------------------- #
# the orchestrator's wiki access
# --------------------------------------------------------------------------- #

_WIKI_TOOL = '''\
#!/usr/bin/env python3
"""The E16 orchestrator's wiki tool -- the SAME pages the players are served.

Written into the run directory by the orchestrator's `prepare()`. It reads
`<run>/wiki/`, the run's own copy, through `nethack_harness.wiki_kb.WikiKB` --
the exact class behind the players' `wiki` skill, with the same ~2000-char cap.
Deliberately not a second knowledge base: an orchestrator planning against
different bytes than the players read is a confound wearing a disguise.

    python wiki_tool.py --list
    python wiki_tool.py --query wraith
    python wiki_tool.py --page why_do_i_keep_dying --section Praying
"""
import argparse
import sys

sys.path.insert(0, {harness_path!r})
from nethack_harness.wiki_kb import WikiKB

WIKI_DIR = {wiki_dir!r}

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
ap.add_argument("--page", default="")
ap.add_argument("--section", default="")
ap.add_argument("--query", default="")
a = ap.parse_args()
kb = WikiKB(WIKI_DIR)
if a.query:
    print(kb.search(a.query))
elif a.page:
    print(kb.read(a.page, a.section or None))
else:
    print(kb.toc())
'''


def write_orchestrator_wiki_tool(cfg: OrchestratorConfig) -> Path:
    """Drop the orchestrator's wiki CLI into its workspace."""
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.orchestrator_dir / "wiki_tool.py"
    atomic_write(path, _WIKI_TOOL.format(
        harness_path=str(REPO / "environments" / "nethack"),
        wiki_dir=str(cfg.wiki_dir)))
    os.chmod(path, 0o755)
    return path


# --------------------------------------------------------------------------- #
# seeding the archive
# --------------------------------------------------------------------------- #

def seed_archive(cfg: OrchestratorConfig, character: str = "Val-hum-neu-fem") -> Path:
    """Write checkpoint c1: a fresh game at the dungeon entrance.

    Go-Explore needs somewhere to start, and "no checkpoint" is not a state the
    selector can express. This is a real engine reset on the run's seed, saved
    through the same code path every other checkpoint uses -- so c1's numbers
    are harness-computed exactly like the rest.
    """
    from nethack_core.env import NetHackCoreEnv
    from nethack_harness.checkpoints import checkpoint_save

    cfg.archive_dir.mkdir(parents=True, exist_ok=True)
    env = NetHackCoreEnv(task_name="NetHackChallenge-v0", max_episode_steps=100_000)
    env.seed(core=cfg.game_seed, disp=cfg.game_seed)
    env.reset(character=character)
    # A freshly reset game is parked on the welcome --More--, and
    # `checkpoint_save` refuses to checkpoint a pending prompt (blstats and the
    # live heap can disagree there). Clear it, deliberately, before saving.
    env.step(13)
    target = cfg.archive_dir / "c1"
    checkpoint_save(env, target, name="entrance",
                    note=f"fresh game, seed {cfg.game_seed}, dungeon entrance",
                    created_by="orchestrator")
    return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="E16 Go-Explore orchestrator")
    ap.add_argument("run_dir")
    ap.add_argument("--wiki-src", default="/root/nld/e15-wiki/configs/continual/wiki")
    ap.add_argument("--tier", default="e16_gewiki")
    ap.add_argument("--arm", choices=ARMS, default=ARM_GO_EXPLORE,
                    help="THE EXPERIMENT ARM. go_explore = the method. "
                         "matched_restart = THE NULL: N independent attempts "
                         "from one fixed start state, no archive, no "
                         "selection, no directives, no lessons -- the "
                         "multi-restart baseline the method must beat before "
                         "any of its numbers mean anything.")
    ap.add_argument("--player-arm", default="prime_agent",
                    help="The LAUNCHER's arm (which scaffold launch_cell.sh "
                         "runs). Distinct from --arm, which names the "
                         "experiment condition.")
    ap.add_argument("--start-checkpoint", default="",
                    help="matched_restart only: the ONE checkpoint every "
                         "attempt restarts from. Default: the archive's seed "
                         "checkpoint.")
    ap.add_argument("--game-seed", type=int, default=1)
    ap.add_argument("--rng-seed", type=int, default=20260828)
    ap.add_argument("--w-balrog", type=float, default=1.0)
    ap.add_argument("--w-novelty", type=float, default=0.5)
    ap.add_argument("--w-attempts", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=385.0,
                    help="USD ceiling for this run (hard).")
    ap.add_argument("--min-headroom", type=float, default=5.0)
    ap.add_argument("--no-inflight-budget", action="store_true",
                    help="Do NOT enforce the ceiling during an attempt, only "
                         "before launching one. The pre-launch gate alone "
                         "cannot stop a single runaway attempt, because "
                         "players are uncapped -- a $9 smoke reached an "
                         "estimated $22 that way.")
    ap.add_argument("--inflight-rate", type=float,
                    default=DEFAULT_SPEND_RATE_USD_PER_HOUR,
                    help="USD/hour assumed for an attempt in flight until this "
                         "run has a completed attempt to calibrate against "
                         "(default %(default)s, just above the p90 measured "
                         "over 349 real rollouts).")
    ap.add_argument("--inflight-safety-factor", type=float, default=2.0,
                    help="Multiplier applied to the estimated in-flight spend "
                         "for the ENFORCEMENT decision only. There is no "
                         "in-flight usage source, so this is a guard on an "
                         "estimate; 2.0 covers the median within-cell rate "
                         "spread (2.4x) and not the p90 (4.1x).")
    ap.add_argument("--max-attempts", type=int, default=200)
    ap.add_argument("--stall-attempts", type=int, default=8)
    ap.add_argument("--selector", choices=("llm", "scripted"), default="llm",
                    help="llm = a persistent Prime Agent session decides "
                         "(primary); scripted = the softmax selector (ablation).")
    ap.add_argument("--no-directive", action="store_true",
                    help="RUN-WIDE CONTROL: launch every player with no "
                         "directive. Same archive, same selection, no "
                         "instructions. See --paired-control for the per-kind "
                         "version, which is the one the sims showed is needed.")
    ap.add_argument("--paired-control", action="store_true",
                    help="PER-DIRECTIVE-KIND CONTROL: run every directive "
                         "TWICE from the same checkpoint, with and without it. "
                         "Doubles the attempts per directive and is the only "
                         "way 'this KIND of directive changed behaviour' is "
                         "answerable -- a run-wide control cannot tell a "
                         "directive that steered from one that prohibited "
                         "something the player was not going to do anyway.")
    ap.add_argument("--reseed", action="store_true",
                    help="Reseed the engine's RNG after every restore, so two "
                         "attempts from one checkpoint get different dice. OFF "
                         "by default and deliberately so: with the dice fixed, "
                         "the difference between two attempts comes from the "
                         "model and the directive, which is what makes a "
                         "directive's effect attributable. Recorded in "
                         "provenance either way.")
    ap.add_argument("--orch-model", default="")
    ap.add_argument("--orch-agent-dir", default="")
    ap.add_argument("--orch-tmpdir", default="",
                    help="PRIVATE TMPDIR for the unsandboxed orchestrator. "
                         "Default: <agent-dir>/tmp. Never exported to players.")
    ap.add_argument("--orch-json-mode", default="discovery_only",
                    choices=("discovery_only", "always", "never"),
                    help="When --mode json is passed. Default discovery_only: "
                         "json mode + --resume was measured hanging where the "
                         "identical text --print call succeeded.")
    ap.add_argument("--orch-timeout", type=float, default=900.0,
                    help="HARD per-round deadline for one orchestrator call. "
                         "Its process group is killed at the deadline and the "
                         "round is recorded as an error, so a hang can never "
                         "consume the run's wall clock.")
    ap.add_argument("--resume", action="store_true",
                    help="CONTINUE an existing run directory instead of "
                         "starting one. Reconciles the archive against the "
                         "attempt rows, finalizes whatever attempt was in "
                         "flight as censored:interrupted, re-attaches the "
                         "orchestrator's recorded session so the conversation "
                         "continues, and carries spend forward so nothing is "
                         "charged twice.")
    ap.add_argument("--reconcile", action="store_true",
                    help="Run the reconciliation and exit. NO model calls, no "
                         "spend, no launches: it attributes orphaned archive "
                         "checkpoints, finalizes interrupted attempts, and "
                         "reports what it could not attribute.")
    ap.add_argument("--force", action="store_true",
                    help="--reconcile only: proceed even though a live "
                         "orchestrator still holds the run directory. Almost "
                         "always wrong -- it writes a censored row for an "
                         "attempt that is still playing.")
    ap.add_argument("--progress-interval", type=float, default=15.0,
                    help="Seconds between progress.jsonl samples while an "
                         "attempt plays. `tail -f` that file to watch a run.")
    ap.add_argument("--decide-only", default="",
                    help="PROBE THE DECISION LOOP WITHOUT PAYING FOR PLAYERS. "
                         "Path to a JSON list of synthetic attempt outcomes "
                         "(or {\"outcomes\": [...]}), one per round: the "
                         "orchestrator plays its real opening round and one "
                         "real decision round per entry, against the real "
                         "archive in <run_dir>/archive, and each round is fed "
                         "back the outcome you wrote instead of a player's. "
                         "No player is launched and no player budget is spent. "
                         "Every record it writes is marked synthetic:true.")
    ap.add_argument("--tree", action="store_true",
                    help="Print the archive as an indented lineage tree and "
                         "exit. NO model calls, no spend, no writes -- it "
                         "reads meta.json's `parent`. Says out loud when what "
                         "it found is not a tree.")
    ap.add_argument("--seed-archive", action="store_true",
                    help="Write c1 (a fresh game at the entrance) and exit.")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Create the run tree + provenance.json and exit.")
    ap.add_argument("--probe-session", action="store_true",
                    help="Run the two-call --resume continuity probe, write the "
                         "result into provenance.json, and exit. THE ONLY "
                         "command here that calls a model.")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg = OrchestratorConfig(
        run_dir=run_dir, wiki_src=Path(args.wiki_src),
        tier=args.tier, arm=args.arm, player_arm=args.player_arm,
        start_checkpoint_id=args.start_checkpoint, selector=args.selector,
        game_seed=args.game_seed,
        rng_seed=args.rng_seed, w_balrog=args.w_balrog,
        w_novelty=args.w_novelty, w_attempts=args.w_attempts,
        temperature=args.temperature, budget_ceiling_usd=args.budget,
        min_headroom_usd=args.min_headroom, max_attempts=args.max_attempts,
        enforce_inflight_budget=not args.no_inflight_budget,
        inflight_spend_prior_usd_per_hour=args.inflight_rate,
        inflight_spend_safety_factor=args.inflight_safety_factor,
        stall_attempts=args.stall_attempts, no_directive=args.no_directive,
        paired_control=args.paired_control,
        reseed_on_restore=args.reseed,
        orchestrator_model=args.orch_model,
        orchestrator_timeout_s=args.orch_timeout,
        orchestrator_json_mode=args.orch_json_mode,
        orchestrator_agent_dir=(Path(args.orch_agent_dir) if args.orch_agent_dir
                                else run_dir / "orchestrator" / "agent"),
        orchestrator_tmpdir=(Path(args.orch_tmpdir) if args.orch_tmpdir
                             else None),
        progress_interval_s=args.progress_interval,
    )
    # RECONCILE-ONLY runs against the directory as it stands. No wiki copy, no
    # provenance rewrite, no session -- so it is safe to point at a run someone
    # else is still holding, and safe to run twice.
    # --tree, like --reconcile, runs against the directory as it stands: no
    # wiki copy, no provenance rewrite, no session, no writes at all.
    if args.tree:
        if not cfg.archive_dir.is_dir():
            print(f"e16: {cfg.archive_dir} does not exist", file=sys.stderr)
            return 2
        rows = ledger_rows(cfg.archive_dir)
        print(render_archive_tree(rows))
        return 0 if not archive_tree(rows)["problems"] else 1
    if args.reconcile:
        if not cfg.archive_dir.is_dir():
            print(f"e16: {cfg.archive_dir} does not exist", file=sys.stderr)
            return 2
        try:
            report = reconcile_run(cfg, force=args.force)
        except RunStillLive as exc:
            print(f"[e16] {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if not report["unattributed_checkpoints"] else 1
    # THE NULL ARM HAS NO ORCHESTRATOR. Not "an orchestrator that is told to do
    # nothing" -- no session at all, no selection, no directive. An LM in the
    # loop that merely refrained from steering would still be a treatment.
    if cfg.arm == ARM_MATCHED_RESTART:
        if cfg.selector == "llm":
            print("[e16] --arm matched_restart: forcing --selector scripted "
                  "and disabling directives (the null arm has no orchestrator)",
                  file=sys.stderr)
        cfg.selector = "scripted"
        cfg.no_directive = True
        cfg.paired_control = False
    # A RESUME MUST NOT LOSE WHAT THE FIRST START PROVED. `prepare` rewrites
    # provenance.json from the config, so the resume-continuity verdict (which
    # `run_e16.sh run` gates on, and which cost real inference) has to be read
    # back off disk first or the next resume would demand the probe again.
    if args.resume:
        try:
            _prev = json.loads(cfg.provenance_path.read_text())
            cfg.session_resume_verified = (
                _prev.get("orchestrator_session") or {}).get(
                    "session_resume_verified")
        except Exception:
            pass
    session = build_session(cfg) if cfg.selector == "llm" else None

    def _no_player(ctx):
        raise RuntimeError("--decide-only launches no players; reaching the "
                           "launcher means the mode is not doing what it says")

    orch = Orchestrator(cfg,
                        _no_player if args.decide_only else SubprocessPlayer(),
                        session=session)
    prov = orch.prepare()
    print(json.dumps(prov, indent=2, sort_keys=True))
    if args.prepare_only:
        return 0
    if args.probe_session:
        if session is None:
            print("--probe-session needs --selector llm", file=sys.stderr)
            return 2
        result = session.probe_resume_carries_history()
        cfg.session_resume_verified = bool(result.get("verified"))
        orch.finish()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("verified") else 1
    if args.decide_only:
        # NO PLAYER EXISTS IN THIS MODE, and the launcher says so rather than
        # being a stub that could quietly run one.
        spec = json.loads(Path(args.decide_only).read_text())
        outcomes = spec["outcomes"] if isinstance(spec, dict) else spec
        if not isinstance(outcomes, list) or not outcomes:
            print(f"e16: --decide-only {args.decide_only} holds no outcomes",
                  file=sys.stderr)
            return 2
        if not checkpoint_list(cfg.archive_dir):
            print(f"e16: --decide-only needs an archive in {cfg.archive_dir}; "
                  f"copy one in (never point this at an archive you care "
                  f"about -- the mode appends lessons and bumps "
                  f"attempts_from)", file=sys.stderr)
            return 2
        try:
            summary = orch.run_decide_only(outcomes)
        except OrchestratorRoundFailed as exc:
            print(f"[e16] ORCHESTRATOR FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            print(json.dumps(orch.summary(), indent=2, sort_keys=True,
                             default=str))
            return 3
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    if args.seed_archive or not checkpoint_list(cfg.archive_dir):
        target = seed_archive(cfg)
        print(f"[e16] seeded archive: {target}")
        if args.seed_archive:
            return 0
    if args.resume:
        try:
            state = orch.resume()
        except RunStillLive as exc:
            print(f"[e16] {exc}", file=sys.stderr)
            return 2
        print("[e16] RESUMED " + json.dumps(
            {k: v for k, v in state.items() if k != "recovery"},
            sort_keys=True, default=str), file=sys.stderr)
        rep = state["recovery"]
        print(f"[e16] reconciled: {len(rep['attempts_finalized_now'])} attempt(s) "
              f"finalized as censored:interrupted, "
              f"{len(rep['checkpoints_attributed'])} checkpoint(s) attributed, "
              f"{len(rep['unattributed_checkpoints'])} unattributed",
              file=sys.stderr)
        for item in rep["unattributed_checkpoints"]:
            print(f"[e16]   UNATTRIBUTED {item['checkpoint']}: {item['why']}",
                  file=sys.stderr)
    try:
        summary = orch.run()
    except RunInterrupted as exc:
        # The in-flight attempt has already been finalized and the summary
        # flushed by `run`'s finally. Exit code 130 is the conventional
        # "killed by a signal", so a supervising script can tell an interrupted
        # run apart from a failed one.
        print(f"[e16] INTERRUPTED: {exc}. The in-flight attempt was finalized "
              f"as censored:interrupted; resume with --resume.", file=sys.stderr)
        print(json.dumps(orch.summary(), indent=2, sort_keys=True, default=str))
        return 130
    except OrchestratorRoundFailed as exc:
        # LOUD AND NON-ZERO. `run` has already flushed the summary with
        # stop_reason=orchestrator_failed and the error on it, and the raw
        # replies are under orchestrator/raw/. What must not happen is the
        # driver treating "the orchestrator produced nothing usable" as a
        # completed run: the E15 lesson is that a substitution nobody exits
        # non-zero on is a substitution nobody notices.
        print(f"[e16] ORCHESTRATOR FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        print(json.dumps(orch.summary(), indent=2, sort_keys=True, default=str))
        return 3
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def build_session(cfg: OrchestratorConfig):
    """The orchestrator's conversation, with a PRIVATE agent-state directory.

    Seeded from the operator's ``~/.prime/agent`` (settings/auth/models) so
    ``--provider prime-inference`` resolves -- credentials themselves live in
    ``~/.prime/config.json``, outside the agent dir, which is why a private
    agent dir does not lose them. It never writes into the shared directory,
    whose ``daemon-workers/`` and ``session-leases/`` a booting experiment was
    measured reaping from another experiment's live rollouts.
    """
    from e16_session import (DEFAULT_MODEL, PrimeAgentSession,
                             configure_native_compaction, seed_agent_dir,
                             usable_tmpdir)

    agent_dir = cfg.orchestrator_agent_dir or (cfg.orchestrator_dir / "agent")
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    seed_agent_dir(agent_dir)
    model = cfg.orchestrator_model or DEFAULT_MODEL
    # THE CONTEXT BOUND, HANDED TO THE CLI. After `seed_agent_dir`, because it
    # rewrites the settings.json that was just copied in -- and into the
    # PRIVATE agent dir, so the operator's own prime-agent keeps its defaults.
    cfg.orchestrator_compaction = configure_native_compaction(
        agent_dir, model=model,
        limit_tokens=cfg.orchestrator_context_limit_tokens)
    if not cfg.orchestrator_compaction.get("enforced"):
        print(f"[e16] WARNING: the "
              f"{cfg.orchestrator_context_limit_tokens}-token orchestrator "
              f"context bound is NOT enforced: "
              f"{cfg.orchestrator_compaction.get('why')}", file=sys.stderr)
    # THE SOCKET PATH, CHECKED BEFORE IT IS USED. See `usable_tmpdir`: a
    # too-long TMPDIR does not fail as a path error, it fails as a 30s daemon
    # timeout on every round, which is indistinguishable from a wedge.
    tmpdir, note = usable_tmpdir(cfg.orchestrator_tmpdir
                                 or (Path(agent_dir) / "tmp"))
    cfg.orchestrator_tmpdir = Path(tmpdir)
    cfg.orchestrator_tmpdir_note = note
    if note:
        print(f"[e16] {note}", file=sys.stderr)
    return PrimeAgentSession(
        work_dir=cfg.orchestrator_dir, agent_dir=agent_dir,
        log_path=cfg.orchestrator_log,
        model=model,
        timeout_s=cfg.orchestrator_timeout_s,
        summary_cap=cfg.orchestrator_summary_cap,
        tmpdir=tmpdir,
        json_mode_policy=cfg.orchestrator_json_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
