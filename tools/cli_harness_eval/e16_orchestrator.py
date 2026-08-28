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
# config
# --------------------------------------------------------------------------- #

@dataclass
class OrchestratorConfig:
    """Everything the loop is allowed to depend on, in one auditable object."""

    run_dir: Path
    #: Source of the curated wiki pages; copied into ``run_dir/wiki`` at start.
    wiki_src: Path = Path("/root/nld/e15-wiki/configs/continual/wiki")
    tier: str = "e16_gewiki"
    arm: str = "prime_agent"
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
    #: CONTROL MODE. Launch players with no directive at all -- same archive,
    #: same selection, no instructions. This is an ablation, not a convenience:
    #: if it matches the directive condition, the orchestrator's strategy adds
    #: nothing, and the run should be able to establish that.
    no_directive: bool = False
    #: Whether `--resume` was PROVEN to carry history (the two-call probe). Set
    #: by the run, written to provenance; never assumed.
    session_resume_verified: Optional[bool] = None

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

    ``prefix.jsonl`` is currently written by nothing, so this returns "" on
    every checkpoint the current harness produces.
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
    arm: str
    directive: str = ""


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


def _tokens(text: str) -> list:
    return [w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS]


def classify_directive_compliance(directive: str, calls, outcome: str,
                                  window: int = DIRECTIVE_WINDOW) -> dict:
    """Did the player do what the orchestrator told it to?

    WHY THIS EXISTS, and it is the research point rather than a nicety: without
    compliance data there is no way to tell "the orchestrator steered the
    search" from "the orchestrator narrated while random restarts did the
    work". Those two produce identical depth curves and opposite conclusions,
    and the second is the null hypothesis this experiment has to be able to
    fail against. So compliance is measured per attempt, not assumed.

    WHAT THIS IS. A KEYWORD RUBRIC over the attempt's first ``window`` tool
    calls -- deterministic, cheap, auditable, and deliberately not an LM judge
    (an LM grading whether another LM followed a third LM's instruction is
    three models deep and not a measurement). Its limits are real and are
    reported alongside it: it matches surface tokens, so "take the south door"
    scores against a call whose args mention ``south``, and a player that
    complies by a route the directive did not name scores low. Read it as a
    signal over many attempts, never as a verdict on one.

    ``calls`` is a sequence of ``{"name": str, "args": dict|str}`` -- the
    player's tool calls in order.
    """
    directive = (directive or "").strip()
    if not directive:
        return {"class": COMPLY_NONE, "matched": [], "missed": [],
                "violated": [], "window": window, "n_calls_examined": 0,
                "rubric": "keyword-over-first-N-calls"}

    words = _tokens(directive)
    # Split at the first avoidance word: everything after it is a prohibition.
    lowered = re.findall(r"[a-z0-9_']+", directive.lower())
    avoid_at = next((i for i, w in enumerate(lowered) if w in _AVOID_WORDS), None)
    if avoid_at is None:
        want, avoid = words, []
    else:
        boundary = set(_tokens(" ".join(lowered[avoid_at:avoid_at + 6])))
        avoid = [w for w in words if w in boundary]
        want = [w for w in words if w not in boundary]

    calls = list(calls or [])[:window]
    if not calls:
        cls = COMPLY_PREVENTED if outcome in (OUTCOME_DIED, OUTCOME_CENSORED) \
            else COMPLY_UNKNOWN
        return {"class": cls, "matched": [], "missed": want, "violated": [],
                "window": window, "n_calls_examined": 0,
                "rubric": "keyword-over-first-N-calls"}

    haystack = " ".join(
        f"{c.get('name', '')} {c.get('args') if isinstance(c.get('args'), str) else json.dumps(c.get('args') or {})}"
        for c in calls if isinstance(c, dict)
    ).lower()

    matched = sorted({w for w in want if w in haystack})
    missed = sorted(set(want) - set(matched))
    violated = sorted({w for w in avoid if w in haystack})

    if violated:
        cls = COMPLY_VIOLATED
    elif not want:
        # Pure prohibition, and it was respected within the window.
        cls = COMPLY_FOLLOWED
    elif len(matched) >= max(1, (len(want) + 1) // 2):
        cls = COMPLY_FOLLOWED
    elif matched:
        cls = COMPLY_PARTIAL
    elif len(calls) < window and outcome in (OUTCOME_DIED, OUTCOME_CENSORED):
        # It never got enough turns to act on the instruction.
        cls = COMPLY_PREVENTED
    else:
        cls = COMPLY_IGNORED
    return {"class": cls, "matched": matched, "missed": missed,
            "violated": violated, "window": window,
            "n_calls_examined": len(calls),
            "rubric": "keyword-over-first-N-calls"}


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
        "arm": cfg.arm,
        "game_seed": cfg.game_seed,
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
            "mode": "none (control)" if cfg.no_directive else "per-attempt",
            "served_as": "first observation, "
                         "[ORCHESTRATOR DIRECTIVE for this attempt: ...]",
            "compliance_rubric": "keyword over the first "
                                 f"{DIRECTIVE_WINDOW} tool calls; "
                                 "surface match, not semantic judgement",
        },
        "orchestrator_session": {
            "kind": "prime_agent --print --mode json --resume <id>, "
                    "dedicated PRIME_AGENT_CODING_AGENT_DIR, fixed cwd",
            "agent_dir": str(cfg.orchestrator_agent_dir or
                             (cfg.orchestrator_dir / "agent")),
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

        last = self.attempts[-1] if self.attempts else None
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
        if self.milestone_row(rows) is not None:
            return STOP_MILESTONE
        if self._since_advance >= self.cfg.stall_attempts:
            return STOP_STALL
        if not rows:
            return STOP_EMPTY_ARCHIVE
        return None

    # -- one attempt ------------------------------------------------------- #

    def run_attempt(self, rows: list) -> dict:
        """Select, launch, ingest. Returns the attempt record."""
        cfg = self.cfg
        n = len(self.attempts) + 1
        self.maybe_compact()
        choice = self.decide(rows)
        chosen_id = choice["checkpoint_id"]
        directive = choice["directive"]
        _append_jsonl(cfg.selection_path,
                      {"attempt": n, "source": choice["source"],
                       "chosen_id": chosen_id, "directive": directive,
                       "llm": choice["decision"], **choice["selection"]})

        ck_dir = next((r.path for r in rows if r.id == chosen_id), None)

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
            tier=cfg.tier, arm=cfg.arm, directive=directive,
        )
        if not directive and not cfg.no_directive:
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
        compliance = classify_directive_compliance(ctx.directive, calls, outcome)

        # New checkpoints the player wrote. Discovered by diffing the archive,
        # not by trusting a count the player reported.
        after_paths = checkpoint_list(cfg.archive_dir)
        new_dirs = [p for p in after_paths if p.name not in before]
        for p in new_dirs:
            self._stamp_new_checkpoint(p, ctx, result)

        # attempts_from++ on the source checkpoint, atomically.
        if ctx.checkpoint_dir is not None:
            self._bump_attempts_from(ctx.checkpoint_dir)
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

    def summary(self) -> dict:
        rows = self.rows()
        best = max(rows, key=lambda r: r.key) if rows else None
        died = sum(1 for a in self.attempts if a["outcome"] == OUTCOME_DIED)
        censored = sum(1 for a in self.attempts if a["outcome"] == OUTCOME_CENSORED)
        out = {
            "run_dir": str(self.cfg.run_dir),
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
    ap.add_argument("--arm", default="prime_agent")
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
                    help="CONTROL MODE: launch players with no directive. Same "
                         "archive, same selection, no instructions.")
    ap.add_argument("--orch-model", default="")
    ap.add_argument("--orch-agent-dir", default="")
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
        tier=args.tier, arm=args.arm, selector=args.selector,
        game_seed=args.game_seed,
        rng_seed=args.rng_seed, w_balrog=args.w_balrog,
        w_novelty=args.w_novelty, w_attempts=args.w_attempts,
        temperature=args.temperature, budget_ceiling_usd=args.budget,
        min_headroom_usd=args.min_headroom, max_attempts=args.max_attempts,
        stall_attempts=args.stall_attempts, no_directive=args.no_directive,
        orchestrator_model=args.orch_model,
        orchestrator_agent_dir=(Path(args.orch_agent_dir) if args.orch_agent_dir
                                else run_dir / "orchestrator" / "agent"),
    )
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
    return PrimeAgentSession(
        work_dir=cfg.orchestrator_dir, agent_dir=agent_dir,
        log_path=cfg.orchestrator_log,
        model=cfg.orchestrator_model or DEFAULT_MODEL,
        timeout_s=cfg.orchestrator_timeout_s,
        summary_cap=cfg.orchestrator_summary_cap,
        context_chars=cfg.orchestrator_context_chars,
    )


if __name__ == "__main__":
    raise SystemExit(main())
