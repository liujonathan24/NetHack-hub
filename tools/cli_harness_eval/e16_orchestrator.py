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

CENSOR_REASONS = frozenset({
    CENSOR_WALL_CLOCK, CENSOR_HARNESS_ERROR, CENSOR_EMPTY_COMPLETION,
    CENSOR_BUDGET_STOP, CENSOR_INTEGRITY, CENSOR_UNKNOWN,
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

    # Stop conditions.
    max_attempts: int = 200
    stall_attempts: int = 8
    milestone_dlvl: int = 20
    milestone_dungeon: int = SOKOBAN_DUNGEON_NUMBER

    #: Rows the rendered ledger shows (frontier + named saves, ~20 per design).
    ledger_max_rows: int = 20
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
    #: Compaction trigger: cumulative characters the orchestrator has been sent
    #: across the session. Past this, the run asks the session to write a
    #: hand-off summary and continues in a FRESH session seeded with it. The
    #: session chain is recorded, so the conversation stays auditable even
    #: though it is no longer one file.
    orchestrator_context_chars: int = 400_000
    #: Chars the compaction hand-off may be.
    orchestrator_compaction_cap: int = 6000
    #: Agent-state dir for the orchestrator session. MUST NOT be shared with a
    #: player sandbox: /root/.prime/agent holds daemon-workers/ and
    #: session-leases/, and a booting experiment scanning it has been measured
    #: reaping another experiment's live sessions.
    orchestrator_agent_dir: Optional[Path] = None
    orchestrator_model: str = ""
    #: Per-round wall-clock cap on one orchestrator turn.
    orchestrator_timeout_s: float = 900.0
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

    @property
    def attempts_path(self) -> Path:
        return self.run_dir / "attempts.jsonl"

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


def render_ledger(rows: list, cfg: Optional[OrchestratorConfig] = None,
                  current_id: Optional[str] = None,
                  attempts: Optional[list] = None) -> str:
    """The archive as text for a player's first observation.

    Frontier rows plus every NAMED save (the design's "~20 rows max"): a
    dominated auto-checkpoint stays on disk and out of the prompt. Handles an
    empty archive, one row, and many.
    """
    cfg = cfg or OrchestratorConfig(run_dir=Path("."))
    if not rows:
        base = ("CHECKPOINT ARCHIVE: empty. Nothing has been saved yet -- "
                "this is the first attempt. Use save(label, note) at states "
                "worth returning to.")
        return base + ("\n\n" + render_attempt_history(attempts) if attempts else "")
    front_ids = {r.id for r in pareto_frontier(rows)}
    shown = [r for r in rows
             if r.id in front_ids or (r.created_by == "save" and r.name)]
    shown.sort(key=lambda r: (-r.dlvl, -r.xl, -r.score))
    shown = shown[:cfg.ledger_max_rows]

    lines = [f"CHECKPOINT ARCHIVE ({len(rows)} saved state(s); "
             f"{len(front_ids)} on the Dlvl/XL/score frontier). "
             f"All numbers below are measured by the harness from the game "
             f"engine, not written by anyone."]
    lines.append("  id  Dlvl  XL   HP     turn   score  attempts  name / why it was saved")
    for r in shown:
        mark = "->" if r.id == current_id else "  "
        lines.append(
            f"{mark}{r.id:>4}  {r.dlvl:>4}  {r.xl:>2}  "
            f"{r.hp:>3}/{r.max_hp:<3} {r.gameturn:>6} {r.score:>6}  "
            f"{r.attempts_from:>8}  {(r.name or '-')[:28]}"
            + (f" | {r.note[:60]}" if r.note else "")
        )
    if len(rows) > len(shown):
        lines.append(f"  ({len(rows) - len(shown)} dominated auto-checkpoint(s) "
                     f"not listed)")
    if attempts:
        lines.append("")
        lines.append(render_attempt_history(attempts))
    return "\n".join(lines)


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

    def add_player(self, usd: float) -> None:
        self.player_usd += float(usd or 0.0)

    def add_orchestrator(self, usd: float) -> None:
        self.orchestrator_usd += float(usd or 0.0)

    def lines(self) -> dict:
        return {
            "player_usd": round(self.player_usd, 4),
            "orchestrator_usd": round(self.orchestrator_usd, 4),
            "total_usd": round(self.spent_usd, 4),
            "ceiling_usd": self.ceiling_usd,
            "remaining_usd": round(self.remaining, 4),
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
    #: Model-written text. Recorded, appended to lessons.md, never measured.
    summary: str = ""
    lesson: str = ""
    exit_code: int = 0
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
    turns = Path(out_dir) / "turns"
    if not turns.is_dir():
        return out
    files = sorted(turns.glob("*.ndjson"), key=lambda p: p.stat().st_mtime)
    for path in files:
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
                "compaction_trigger_chars": cfg.orchestrator_context_chars,
                "handoff_cap_chars": cfg.orchestrator_compaction_cap,
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
        #: Phrasing warnings raised on the last directive, fed back into the
        #: next orchestrator round. A lint nobody reads changes nothing.
        self._last_lint: list = []

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
  seed is deliberately overfit to: it is the program's proven deep seed (D13 in
  the base control; D15 -- the program's deepest run -- via preemptive
  rollback). Its known hazard ladder: a Grey-elf / owlbear wall around D11, a
  wraith around D15.
  Players have: the standard action surface, rollback(n), save(label, note),
  and wiki(page|query|section) over the same two curated pages you have.

YOUR KNOWLEDGE BASE (read it now, before deciding anything)
  {wiki_dir}
    why_do_i_keep_dying.md   standard_strategy.md   MANIFEST.json
  Use your IPython kernel to read them. Also available:
    python {wiki_tool} --list
    python {wiki_tool} --query wraith
    python {wiki_tool} --page why_do_i_keep_dying --section Praying

ROUND 1 IS DISCUSSION, NOT A LAUNCH. Reply with:
  1. What you learned from the wiki that bears on THIS seed's hazard ladder --
     specifically: the wraith, prayer timing, floating eyes, and when to flee.
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
one or two sentences of concrete instruction for THIS attempt ("from c12, go
back up to D10 and clear the east rooms before descending"; "engrave Elbereth
before engaging anything on this floor"). The directive is served to the player
verbatim at the top of its first observation, and whether it followed you is
measured -- so make it checkable, not encouraging.

HOW TO PHRASE IT SO IT CAN BE SCORED. Two rules, both learned from directives
that could not be measured:
  * PROHIBIT OUTCOMES, NOT TOOLS. "Do not clear the level" is checkable against
    what the game shows. "Do not explore" is not: the player may have to
    explore in order to reach the staircase you told it to take, and nothing
    that stays out of its reasoning can tell that from disobedience.
  * ALWAYS PAIR A PROHIBITION WITH SOMETHING TO DO. A directive that only
    forbids is satisfied by dying on turn two, so it cannot distinguish your
    steering from an attempt that never started.

Reply with a short rationale and then EXACTLY this JSON object on its own line:
{{"checkpoint": "<id from the table above>", "directive": "<your instruction>", "rationale": "<one line>"}}

Only ids that appear in the table are accepted; anything else falls back to the
scripted selector and is recorded as your round having failed to decide.
"""

    COMPACTION_PROMPT = """\
Your context is getting long, so this conversation will be continued in a new
one seeded only with what you write now. Write the hand-off: at most {cap}
characters covering what you have learned about this seed, which checkpoints
have been productive and which are dead ends, what directives worked, and what
you intend to do next. Write it for yourself. No JSON this round.
"""

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
        res = self.session.ask(prompt, kind="opening")
        self.budget.add_orchestrator(res.spend_usd)
        self.opened = True
        atomic_write(self.cfg.orchestrator_dir / "opening_plan.txt", res.text or "")
        return {"text": res.text, "error": res.error, "session_id": res.session_id}

    def decide(self, rows: list) -> dict:
        """Who goes next and what they are told. Validated, then recorded.

        Returns ``{"checkpoint_id", "directive", "source", "selection",
        "decision"}``. ``source`` is one of ``llm``, ``scripted``, or
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
               "decision": None}
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
            + f", max Dlvl {last['max_dlvl']}, {last['calls']} calls.\n"
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
        prompt = self.ROUND_PROMPT.format(
            round=n, player_usd=self.budget.player_usd,
            orch_usd=self.budget.orchestrator_usd,
            ceiling=self.cfg.budget_ceiling_usd,
            last_block=last_block,
            ledger=render_ledger(rows, self.cfg, attempts=self.attempts),
        )
        res = self.session.ask(prompt, kind=f"round{n}")
        self.budget.add_orchestrator(res.spend_usd)

        from e16_session import parse_decision
        dec = parse_decision(res.text, [r.id for r in rows])
        out["decision"] = {
            "parsed": dec.parsed, "valid": dec.valid,
            "fallback_reason": dec.fallback_reason,
            "rationale": dec.rationale, "raw": (dec.raw or "")[:4000],
            "session_id": res.session_id,
            "continuity_broken": res.continuity_broken,
            "error": res.error,
        }
        if dec.valid:
            out["checkpoint_id"] = dec.checkpoint_id
            out["directive"] = dec.directive
            out["source"] = "llm"
        else:
            self.llm_fallbacks += 1
            out["source"] = "scripted_fallback"
        if self.cfg.no_directive:
            out["directive"] = ""  # the explicit control mode
        return out

    def _last_treatment(self) -> Optional[dict]:
        """The last attempt that actually carried a directive, if any."""
        for a in reversed(self.attempts):
            if a.get("pair_role") != ROLE_CONTROL:
                return a
        return None

    def maybe_compact(self) -> Optional[str]:
        """Hand the conversation off to a fresh one when context gets long.

        The orchestrator's context is the only thing here that grows without
        bound, and Prime Agent ENDS the agent loop when context exceeds
        `contextWindow - 16384` -- under `--print` nothing resumes it and the
        CLI exits 0 with no output, which reaches the harness as a silent
        success. That failure truncated three of five rollouts in an earlier
        run. So compaction is pre-emptive, and the session chain is recorded so
        the conversation stays auditable even though it is no longer one file.
        """
        if self.session is None or not self.session.needs_compaction():
            return None
        res = self.session.ask(
            self.COMPACTION_PROMPT.format(cap=self.cfg.orchestrator_compaction_cap),
            kind="compaction")
        self.budget.add_orchestrator(res.spend_usd)
        summary = (res.text or "")[:self.cfg.orchestrator_compaction_cap]
        if hasattr(self.session, "compact"):
            self.session.compact(summary)
        else:
            # A PrimeAgentSession compacts by starting a new session seeded
            # with the hand-off: dropping the id makes the next round create
            # one, and the chain is already in `session_ids`.
            self.session.session_id = ""
            self.session.sent_chars = 0
            self.session.compactions += 1
            self.session.ask(
                "You are continuing an E16 Go-Explore run. This is the hand-off "
                "from your previous conversation; treat it as your own memory:\n\n"
                + summary, kind="compaction_seed")
        atomic_write(self.cfg.orchestrator_dir / f"handoff_{self.session.compactions}.txt",
                     summary)
        return summary

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
            best = max((a["max_dlvl"] for a in self.attempts), default=0)
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
        self.maybe_compact()
        choice = self.decide(rows)
        chosen_id = choice["checkpoint_id"]
        directive = choice["directive"]
        _append_jsonl(cfg.selection_path,
                      {"attempt": n, "source": choice["source"],
                       "chosen_id": chosen_id, "directive": directive,
                       "directive_kind": classify_directive_kind(directive),
                       "directive_lint": lint_directive(directive),
                       "llm": choice["decision"], **choice["selection"]})

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
        if not directive and not cfg.no_directive and pair_role != ROLE_CONTROL:
            # The contract says every launch carries a directive. An empty one
            # outside --no-directive means the orchestrator failed to produce
            # one, and that must be visible rather than looking like the
            # control arm.
            ctx.directive = ("(orchestrator produced no directive for this "
                             "attempt; play as you judge best)")
        atomic_write(out_dir / "ledger_served.txt", ledger_text)
        atomic_write(out_dir / "directive_served.txt", ctx.directive)
        self._decision_of_attempt = choice

        before = {p.name for p in checkpoint_list(cfg.archive_dir)}
        before_front = self.frontier_ids(rows)
        t0 = time.time()
        try:
            result = self.launcher(ctx)
        except Exception as exc:  # a launcher blow-up is a CENSORED attempt
            result = PlayerResult(stop_condition="error", error=f"{type(exc).__name__}: {exc}",
                                  wall_s=time.time() - t0)
        if not result.wall_s:
            result.wall_s = time.time() - t0
        return self.ingest(ctx, result, before, before_front)

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

        selection_record = {
            "attempt": n, "source": "matched_restart_fixed_start",
            "chosen_id": start.id, "directive": "", "directive_kind": "none",
            "directive_lint": [], "llm": None,
            "reason": ("no selection: matched_restart restarts from one fixed "
                       "state by construction"),
            "candidates": [], "n_rows": len(rows),
        }
        _append_jsonl(cfg.selection_path, selection_record)

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
            result = PlayerResult(stop_condition="error",
                                  error=f"{type(exc).__name__}: {exc}",
                                  wall_s=time.time() - t0)
        if not result.wall_s:
            result.wall_s = time.time() - t0
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

        self.budget.add_player(result.spend_usd)
        self.cum_calls += int(result.calls or 0)
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
        for p in new_dirs:
            self._stamp_new_checkpoint(p, ctx, result)

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
            "calls": int(result.calls or 0),
            "spend_usd": round(float(result.spend_usd or 0.0), 6),
            "wall_s": round(float(result.wall_s or 0.0), 2),
            "max_dlvl": int(result.max_dlvl or 0),
            # Recorded, never smoothed over: a metric and the turn files
            # disagreeing about depth is a fact about the harness, and the one
            # place it can be noticed is here.
            "depth_disagreement": (result.raw or {}).get("depth_disagreement"),
            "max_xl": int(result.max_xl or 1),
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
        self.write_summary()
        return record

    def _stamp_new_checkpoint(self, path: Path, ctx: PlayerContext,
                              result: PlayerResult) -> None:
        """Record which attempt produced a checkpoint, and its lineage.

        The `save` skill knows nothing about attempts or parents -- it writes
        from inside the game. Stamping here is what makes the archive a tree
        rather than a pile. Numeric fields are NOT touched.
        """
        try:
            meta = checkpoint_meta(path)
        except Exception:
            return
        meta.setdefault("parent", None)
        if meta.get("parent") is None and ctx.checkpoint_id:
            meta["parent"] = ctx.checkpoint_id
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
                slot["delta"] = {
                    "max_dlvl": slot["treatment"]["max_dlvl"] - slot["control"]["max_dlvl"],
                    "max_xl": slot["treatment"]["max_xl"] - slot["control"]["max_xl"],
                    "calls": slot["treatment"]["calls"] - slot["control"]["calls"],
                }
                k = by_kind.setdefault(kind, {"pairs": 0, "sum_dlvl_delta": 0,
                                              "sum_xl_delta": 0,
                                              "treatment_compliance": {}})
                k["pairs"] += 1
                k["sum_dlvl_delta"] += slot["delta"]["max_dlvl"]
                k["sum_xl_delta"] += slot["delta"]["max_xl"]
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

    def summary(self) -> dict:
        rows = self.rows()
        best = max(rows, key=lambda r: r.key) if rows else None
        died = sum(1 for a in self.attempts if a["outcome"] == OUTCOME_DIED)
        censored = sum(1 for a in self.attempts if a["outcome"] == OUTCOME_CENSORED)
        out = {
            "run_dir": str(self.cfg.run_dir),
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
            "wall_clock_s": round(time.time() - self.started_at, 1),
            "stop_reason": self.stop_reason,
            "frontier_advances": self.frontier_advances,
            "luck": self.luck(),
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

    def run(self, max_attempts: Optional[int] = None) -> dict:
        limit = max_attempts if max_attempts is not None else self.cfg.max_attempts
        # ROUND 1 IS DISCUSSION. The plan goes on the record before any player
        # is launched, so "did it follow its own strategy?" stays answerable.
        self.open_discussion()
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
        else:
            self.stop_reason = self.stop_reason or STOP_MAX_ATTEMPTS
        self.finish()
        return self.write_summary()


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
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=self.timeout_s)
        (ctx.out_dir / "launch.log").write_text(proc.stdout + "\n" + proc.stderr)
        res = read_trace_result(ctx.out_dir)
        res.exit_code = proc.returncode
        res.wall_s = time.time() - t0
        if proc.returncode != 0 and not res.error:
            res.error = f"launch_cell exited {proc.returncode}"
            res.stop_condition = res.stop_condition or "error"
        return res


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
    if not path.is_file():
        return PlayerResult(stop_condition="error",
                            error=f"no traces.jsonl in {out_dir}")
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
        return PlayerResult(stop_condition="error",
                            error=f"empty traces.jsonl in {out_dir}")
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
    try:
        from aggregate import price_table_for, rollout_cost  # same dir
        spend = rollout_cost(trace, price_table_for(trace)) or 0.0
    except Exception:
        try:
            from aggregate import PRICE_TABLE_GLM_5_2, rollout_cost
            spend = rollout_cost(trace, PRICE_TABLE_GLM_5_2) or 0.0
        except Exception:
            spend = 0.0
    return PlayerResult(
        stop_condition=str(trace.get("stop_condition") or ""),
        died=bool(metrics.get("died")),
        ascended=bool(metrics.get("ascended")),
        error=str(trace.get("error") or ""),
        calls=int(metrics.get("skill_calls") or 0),
        spend_usd=float(spend),
        max_dlvl=max_dlvl,
        max_xl=int(metrics.get("max_xp_level") or 1),
        summary=_final_text(trace),
        raw={"metrics": metrics, "depth_disagreement": depth_disagreement},
    )


def _max_dlvl_from_turns(out_dir) -> Optional[int]:
    """The deepest Dlvl the per-turn NDJSON actually recorded, or None.

    `helpers._write_trace_entry` stamps `dlvl` and `max_dlvl_reached` on every
    turn record straight off the shaped observation, so this is the engine's
    own account of where the hero went -- independent of whatever the
    end-of-rollout metric computed.
    """
    turns = Path(out_dir) / "turns"
    if not turns.is_dir():
        return None
    best = None
    for path in sorted(turns.glob("*.ndjson")):
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
    )
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
    session = build_session(cfg) if cfg.selector == "llm" else None
    orch = Orchestrator(cfg, SubprocessPlayer(), session=session)
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
    if args.seed_archive or not checkpoint_list(cfg.archive_dir):
        target = seed_archive(cfg)
        print(f"[e16] seeded archive: {target}")
        if args.seed_archive:
            return 0
    summary = orch.run()
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
    from e16_session import DEFAULT_MODEL, PrimeAgentSession, seed_agent_dir

    agent_dir = cfg.orchestrator_agent_dir or (cfg.orchestrator_dir / "agent")
    cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
    seed_agent_dir(agent_dir)
    tmpdir = cfg.orchestrator_tmpdir or (Path(agent_dir) / "tmp")
    cfg.orchestrator_tmpdir = Path(tmpdir)
    return PrimeAgentSession(
        work_dir=cfg.orchestrator_dir, agent_dir=agent_dir,
        log_path=cfg.orchestrator_log,
        model=cfg.orchestrator_model or DEFAULT_MODEL,
        timeout_s=cfg.orchestrator_timeout_s,
        summary_cap=cfg.orchestrator_summary_cap,
        context_chars=cfg.orchestrator_context_chars,
        tmpdir=tmpdir,
        json_mode_policy=cfg.orchestrator_json_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
