"""
nethack
=======

The Prime Intellect Environments Hub wrapper for the NetHack training env.

Consumes `nethack_core` and presents it as a verifiers MultiTurnEnv with 
chat-shaped tool calling and a composable rubric.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

import verifiers as vf
from datasets import Dataset

from nethack_core.env import NetHackCoreEnv
from nethack_harness.memory.journal import Journal
from nethack_core.observations import shape as shape_observation
from nethack_harness.tools.skills import registry as skill_registry, list_skills

# Load the harness overlay from the file that sits next to *this* module, by
# absolute path. A plain `from environments.nethack import harness_overlay` (or
# `import harness_overlay`) can silently resolve to a *different* checkout's copy
# when several `environments/` trees share sys.path (e.g. the engine checkout on
# PYTHONPATH), which would bypass this repo's overlay loader and make the
# NETHACK_HARNESS sweep a no-op. Loading by __file__-relative path guarantees the
# hub's own overlay is always used.
import importlib.util as _ilu
import os as _os
import sys as _sys_boot

_ho_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "harness_overlay.py")
_ho_spec = _ilu.spec_from_file_location("nethack_harness_overlay", _ho_path)
_harness_overlay = _ilu.module_from_spec(_ho_spec)
# Register before exec so dataclasses can resolve the module during class
# creation (sys.modules[cls.__module__] must exist under deferred annotations).
_sys_boot.modules[_ho_spec.name] = _harness_overlay
_ho_spec.loader.exec_module(_harness_overlay)


# ---------- game spec ----------
#
# The env runs the standard full NetHack ascension game. The former 13-tier
# named curriculum ladder (curriculum/curriculum.py + milestones.py +
# subgoals.py) has been removed; this minimal spec is the single default
# ("full_nle"). `success_milestone` is retained as an optional seam (always
# None for the full game) so env_response's milestone-check stays a harmless
# no-op and future task specs can opt in without reshaping the state.
from dataclasses import dataclass


@dataclass(frozen=True)
class GameSpec:
    name: str
    nle_task: str               # native NetHack gym id (e.g. NetHackScore-v0)
    des_file: Optional[str]     # always None (MiniHack des-file path removed)
    max_episode_steps: int
    description: str
    success_criterion: str      # human-readable, codified in rubric
    success_milestone: Any = None
    # Which env class setup_state instantiates: "full" -> NetHackCoreEnv (the
    # standard ascension game); "primitives" -> the six-floor honest-navigation
    # CurriculumPrimitivesEnv (lazily imported so the default game never depends
    # on the engine's invocation hooks).
    task_kind: str = "full"


FULL_GAME_SPEC = GameSpec(
    name="full_nle",
    nle_task="NetHackScore-v0",
    des_file=None,
    max_episode_steps=100_000,
    # Rendered every turn as `Objective:` in the JOURNAL block. It used to read
    # "The full game. Ascend." — NetHack jargon for "win", but to a language
    # model sitting next to a glyph key that says "`<` stairs UP (NOT down)" it
    # reads as an instruction to go UP, which is the opposite of the task.
    description=(
        "Descend as deep into the dungeon as you can and survive; ultimately "
        "win the game (retrieve the Amulet of Yendor and escape)."
    ),
    success_criterion="ascended",
)

# The six-floor "primitives" curriculum: an honest-navigation task (no
# descend/ascend mega-skill). The agent navigates onto real stairs and presses
# '>'/'<' through a compressed 6-floor down / 6-floor up tour
# (DoD 1-2-3 <-> Gehennom 48-49-50). At the DoD3->Gehennom boundary it is handed
# the invocation ritual kit; on Gehennom's Invocation level it is staged beside
# the vibrating square and completes the ritual itself. Backed by
# nethack_harness.curriculum.CurriculumPrimitivesEnv (subclass of NetHackCoreEnv),
# so it is a drop-in for the harness. `task_kind="primitives"` routes setup_state
# to that env class. See docs/CURRICULUM_PRIMITIVES_RESULTS.md.
#
# NOTE: depends on the engine's public invocation hooks (grant_invocation_kit /
# invocation_pos / seat_on_invocation_square), which land with engine PR #37
# (feat/4b-six-floor-hooks). Until that reaches nethack-core main, selecting this
# spec requires a nethack-core built from that branch.
PRIMITIVES_GAME_SPEC = GameSpec(
    name="six_floor_primitives",
    nle_task="engine",
    des_file=None,
    max_episode_steps=100_000,
    description=(
        "Six-floor primitives curriculum: navigate DoD 1->2->3, cross into "
        "Gehennom, descend to the Invocation level, and complete the invocation "
        "ritual — using only primitive stair navigation."
    ),
    success_criterion="reached_invocation_level",
    task_kind="primitives",
)

# Registry of selectable task specs (by GameSpec.name), resolved in setup_state
# from the env's `task_spec` kwarg. Default is the full ascension game.
GAME_SPECS: dict[str, "GameSpec"] = {
    FULL_GAME_SPEC.name: FULL_GAME_SPEC,
    PRIMITIVES_GAME_SPEC.name: PRIMITIVES_GAME_SPEC,
}

#: [ynq] confirms a skill may open ABOUT ITS OWN INTENT, keyed by the skill.
#: Deliberately tiny: an entry means "if the agent just called this skill and
#: NetHack asked this question, the answer is yes". Nothing mentioning
#: attack/really is ever auto-answered -- "Really attack the watch captain?"
#: must stay declined (a `y` there is run-ending in Minetown).
_SKILL_CONFIRMS = {
    "np_loot": ("loot it",),
    "np_tip": ("tip it",),
    "np_apply": ("force its lock", "unlock it"),
}


def _confirm_yes_for(skill_name, messages) -> bool:
    """Should this [ynq] prompt be answered `y` because `skill_name` opened it?

    True only when the newest [yn...]-bearing message matches the calling
    skill's own whitelist and mentions neither attack nor really. Pure
    function so the policy is unit-testable apart from the dismissal loop.
    """
    oks = _SKILL_CONFIRMS.get(skill_name or "", ())
    if not oks:
        return False
    for m in reversed(messages or []):
        if "[yn" in m:
            q = m.lower()
            return (any(k in q for k in oks)
                    and "attack" not in q and "really" not in q)
    return False


#: The NetPlay skills that open a "What do you want to <verb>?" item prompt and
#: can answer it themselves via `item_letter` -- the vendored
#: `create_inventory_command` set (drop, read, put_on, remove, takeoff, wield,
#: wear, apply, eat, drink, tip, dip) plus `zap`, whose signature also takes
#: item_letter. Used by the post-skill dismissal notice to tell the agent HOW
#: to retry instead of just that its prompt was closed. Published np_ names;
#: see netplay_true.NETPLAY_TRUE_TOOL_NAMES.
_DISMISSAL_ITEM_SKILLS = frozenset({
    "np_wear", "np_wield", "np_takeoff", "np_put_on", "np_remove",
    "np_eat", "np_drink", "np_read", "np_drop", "np_apply",
    "np_zap", "np_dip", "np_tip",
})


# ---------- verifiers 0.1.14 compat shim ----------
#
# verifiers 0.1.14's v1.utils.sandbox_program_utils.message_from_response
# assumes every tool_call has the OpenAI SDK nested shape (.function.name /
# .function.arguments). Some endpoints (e.g. api.pinference.ai for Qwen) and
# verifiers' own ToolCall dataclass use the flat shape (.name / .arguments).
# Without this shim, vf-eval raises:
#   AttributeError("'ToolCall' object has no attribute 'function'")
# Remove once the upstream PR lands.
def _patch_verifiers_message_from_response() -> None:
    try:
        from verifiers.v1.utils import sandbox_program_utils as _spu
    except ImportError:
        return
    if not hasattr(_spu, "message_from_response"):
        return  # different verifiers build — nothing to patch

    def _safe(response):  # type: ignore[no-redef]
        choice = response.choices[0]
        message = choice.message
        data = {"role": getattr(message, "role", "assistant")}
        content = getattr(message, "content", None)
        if content is not None:
            data["content"] = content
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            packed = []
            for call in tool_calls:
                fn = getattr(call, "function", None)
                name = getattr(fn, "name", None) if fn is not None else getattr(call, "name", None)
                args = getattr(fn, "arguments", None) if fn is not None else getattr(call, "arguments", None)
                packed.append({
                    "id": getattr(call, "id", None),
                    "type": getattr(call, "type", "function"),
                    "function": {"name": name, "arguments": args},
                })
            data["tool_calls"] = packed
        return data

    _spu.message_from_response = _safe


_patch_verifiers_message_from_response()


# ---------- system prompt ----------


# ---------- extracted modules (re-exported for back-compat) ----------
from nethack_harness.prompt.rendering import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERBOSE,
    render_system_prompt as _render_system_prompt,
    PUBLISHED_TOOLS_STATE_KEY as _PUBLISHED_TOOLS_STATE_KEY,
    _strip_blank_rows,
    _glyph_run_encode,
    _inventory_fingerprint,
    _run_length_encode_messages,
    _glyph_to_words,
    _format_obs_balrog,
    _format_obs_glyphbox,
    _format_obs_summarize_reset,
    _descent_status_block,
    _E1_BEARINGS,
    _e1_bearing,
    _e1_classify_frontier,
    _e1_frontiers_block,
    _e1_exploration_block,
    _e1_spatial_belief_block,
    _VARIANT_FORMATTERS,
    _paint_frontiers_on_map,
    format_observation_as_chat,
)
from nethack_harness.prompt.interactive_state import detect_blocking_ui
from nethack_core import trace_schema as TS
from nethack_harness.helpers import (
    _continual_reset,
    _write_trace_entry,
    TurnRecorder,
    build_tool_result,
    _drop_before_last_belief,
    _refinement_directive,
    _ch_build_window,
    _ch_inject_system,
    _ch_save_bootstrap,
    _compact_chat_history,
    _sanitize_assistant_content,
    _STATUS_SIG_RE,
    _compacted_status_signature,
    _dedupe_compacted_runs,
    _msg_role,
    _msg_content,
    _replace_content,
    _one_line_summary,
    _check_halt_condition,
    BELIEF_STATE_INTERVAL,
    _maybe_belief_state_summary,
    _maybe_distill,
    _to_action_indices,
    _cr_would_be_unknown_command,
    CARRIAGE_RETURN,
    scout_reward,
    descent_reward,
    success_reward,
    ascension_reward,
    _ASCENSION_MARKERS,
    _DEATH_MARKERS,
    _decode_tty,
    _detect_terminal_outcome,
    _code_tool_adapter,
    _build_skill_adapter_callables,
    _make_run_macro_adapter,
    _make_fixed_direction_adapter,
    _TYPE_MAP,
    _make_skill_adapter,
)
from nethack_harness.prompt.prompt_spec import (
    ObsSpec,
    ToolSpec,
    PromptSpec,
    build_prompt,
    VARIANT_REGISTRY,
    resolve_spec,
    attach_refiner,
)
from nethack_harness.prompt.content import compose_user_content, content_to_text


# Sub-experiment 1b: the per-cell SPATIAL/EXPLORATION attributes the JSON map
# can be enriched with. Each maps to one compact RLE 0/1 mask layer.
_VALID_CELL_ATTRS = frozenset({"seen", "visited", "reach"})


def _normalize_cell_schema(cell_schema) -> set:
    """Coerce a cell_schema arg to a validated set of attribute names.

    Accepts None, a set/list/tuple, or a string (comma / whitespace / '+'
    separated, so the vf-eval `-a` JSON args can pass e.g. "seen+visited").
    Unknown tokens are dropped. Empty result = current JSON, unchanged.
    """
    if not cell_schema:
        return set()
    if isinstance(cell_schema, str):
        import re
        tokens = [t for t in re.split(r"[,\s+]+", cell_schema.strip()) if t]
    else:
        tokens = list(cell_schema)
    return {t for t in (str(x).strip().lower() for x in tokens) if t in _VALID_CELL_ATTRS}


# Sentinel key `_parse_tool_call` uses to hand "the model emitted no tool call
# at all" across to `_apply_tool_call` through the `skill_args` dict, since
# that case has no real skill_name/skill_args to dispatch. Shared as a
# constant (not a literal duplicated in both methods) so the two ends of the
# handoff can't silently drift apart.
_NO_TOOL_CALL_SENTINEL = "__no_tool_call__"


class NetHackVerifiersEnv(vf.StatefulToolEnv):
    """
    Per-rollout state: a live NetHackCoreEnv plus character + cumulative scout count.

    We subclass StatefulToolEnv because each rollout owns a long-lived NLE
    instance that must be cleanly initialized in setup_state and torn down on
    completion.

    interface: "skill" (default) or "code". In code mode, `env_response` routes
    the model's `code(source=...)` tool call through `code_mode.run_user_code`,
    which executes against an `nh` namespace and produces a list of NLE actions
    that we then step.
    """

    def __init__(
        self,
        *args,
        interface: str = "skill",
        # Which task spec to run, keyed by GameSpec.name in GAME_SPECS.
        # "full_nle" (default) is the standard ascension game;
        # "six_floor_primitives" is the honest-navigation curriculum.
        task_spec: str = "full_nle",
        sub_lm=None,
        subgoal_proposer=None,
        # Compaction knobs (survey rec). Set via load_environment kwargs.
        compact_obs: bool = False,   # exp1 ran uncompacted; see load_environment
        history_keep_full: int = 5,
        history_drop_after: int = 100,
        belief_state_interval: int = 25,
        journal_render_max_chars: int = 2000,
        # Obs/skill-structure variant for wave-1 experiments. "B1" (default) is
        # the current shipping behavior. "P" is the Continual Harness adaptation:
        # periodic self-refinement turns that prompt the agent to revise its
        # objective and record a lesson note (no NLE step consumed when the
        # agent calls pin_objective/add_note). See docs/PROMPTING_SURVEY.md.
        variant: str = "B0",         # uncompressed ASCII; see load_environment
        # Detail level for the structured-map variants (JSON/TOON): "full"
        # emits rich entity attrs + RLE grid; "minimal" trims to kind/coord/desc.
        # Threaded onto state["map_detail"] for the per-turn template to read.
        map_detail: str = "full",
        # Sub-experiment 1b (JSON cell-content ablation): a collection of
        # per-cell SPATIAL/EXPLORATION attributes to enrich the JSON map with.
        # Drawn from {"seen","visited","reach"}; each enabled attr emits a
        # compact RLE 0/1 mask layer (seen_grid / visited_grid / reach_grid).
        # Accepts a set/list or a comma/space/+-separated string (so the
        # vf-eval `-a` JSON args can pass it). Empty/None = current JSON,
        # unchanged. Only consumed by variant="JSON"; ignored by TOON.
        cell_schema=None,
        refine_interval: int = 20,
        # Variant R (CPP/GPP summarize-and-reset): when True, get_prompt_messages
        # hard-drops every user/assistant turn that landed before the most-recent
        # belief_state:tN journal note. Combined with belief_state_interval > 0
        # this implements "the belief state IS the memory; chat is disposable."
        summarize_and_reset: bool = False,
        # Per-turn NDJSON trace (raw_grid + rendered_user_message +
        # assistant_message + tool_calls). One file per rollout. Off by default.
        trace_dir: Optional[str] = None,
        # Continual-harness mode: on death, auto-reset NLE and keep playing in
        # the same chat session, preserving journal + belief state across
        # episodes. The rollout terminates when continual_lives is exhausted
        # or the agent ascends.
        continual: bool = False,
        continual_lives: int = 5,
        # Variant CH (Continual Harness, arXiv:2605.09998) full Refiner.
        # `refiner` is a pluggable object satisfying nethack_harness.refiner.Refiner;
        # when None and variant=="CH", we build a TeacherLLMRefiner from
        # refiner_model (or fall back to OfflineRefiner). bootstrap_dir, if set,
        # is used to persist/load the four CH components (prompt addendum,
        # sub-agents, skills, journal) across rollouts.
        refiner: Any = None,
        refiner_model: Optional[str] = None,
        bootstrap_dir: Optional[str] = None,
        # Decouple the teacher Refiner from the obs format: when True, attach the
        # full CH refiner machinery (refiner + sub-agent hooks, system inject,
        # run_macro tool) onto whatever `variant` (obs form) is selected — e.g.
        # variant="JSON", refine=True gives JSON observations PLUS the teacher
        # refiner. variant="CH" implies refine=True (its canonical ASCII obs).
        refine: bool = False,
        # Resolved PromptSpec describing how this rollout builds its prompt
        # (obs form + processors, system prompt, per-turn template, tools). When
        # None we resolve it from `variant` for back-compat; load_environment
        # passes one built post-overlay so the NETHACK_HARNESS seam is honoured.
        spec: Optional[PromptSpec] = None,
        # Game-setup overrides applied to every episode's engine reset (the
        # difficulty/generation knobs from our interface). All None = vanilla
        # NetHack. See load_environment for the shapes.
        setup_tune: Optional[dict] = None,
        setup_modify: Optional[dict] = None,
        setup_level_blob: Optional[str] = None,
        # Pin the character (role-race-alignment-gender, e.g. "Val-hum-neu-fem")
        # for every episode's engine reset. None = engine default (random roll).
        # Threaded to NetHackCoreEnv.reset(character=...). Curriculum tiers own
        # their own character and ignore this.
        setup_character: Optional[str] = None,
        # Names of the skills actually exposed to the model this rollout (the
        # tool schema it was given). Tool calls for anything NOT in this set are
        # rejected in env_response instead of being dispatched against the full
        # registry — otherwise a hallucinated `move(direction=…)` executes even
        # under the netplay set (which withholds it), leaking the low-level
        # primitive into the effective action surface and confounding
        # cross-encoding comparisons. None/empty disables the gate (back-compat).
        allowed_skill_names: Optional[set] = None,
        no_progress_timeout: int = 10_000,
        # Memory-ablation knob (sub-experiment 1c). When False, the tier
        # description is NOT pre-pinned as the journal objective at setup, so a
        # rollout with journal tools excluded and belief_state_interval=0 keeps
        # the Journal empty → the "=== JOURNAL ===" block never renders. This
        # is the only way to reach a truly no-memory arm: the objective pin at
        # setup_state otherwise makes the journal non-empty on turn 1. Default
        # True preserves the always-pinned-objective behavior.
        pin_objective_on_setup: bool = True,
        # Task 18 Step 2: True for the CLI-agent arms (nethack_v1's MCP toolset
        # always sets this — self_dispatch=False has no v1 path at all), False
        # for the control arm (the v0 legacy bridge never passes it). Threaded
        # into state["_self_dispatch"] in setup_state, where
        # rendering.format_observation_as_chat reads it to drop the JOURNAL
        # block and the HINT ladder for the CLI arms while leaving the control
        # arm's rendering byte-identical to pre-Task-18 behavior.
        self_dispatch: bool = False,
        # e7 raw-prompt surfaces: when False, the post-call auto-dismiss loop
        # stops answering menus / item prompts / [yn] questions and only
        # acknowledges --More-- (the exact contract the BALROG-80 surface
        # already runs under, see _balrog_raw_prompts below). The agent answers
        # its own prompts -- np_press_key reaches every key incl. esc/space/
        # enter -- and the UI-freeze observation block names the legal answers
        # as np_press_key calls when that tool is published.
        auto_dismiss: bool = True,
        # Store the tty screen at EVERY engine step (not just the final frame
        # per LM turn), so a game is replayable move-by-move. Off by default:
        # existing arms stay byte-identical and pay no size cost. Enable per
        # cell via load_environment(record_step_frames=True) / env_args.
        record_step_frames: bool = False,
        # Append a plain-language "Args:" clause to each published tool's
        # description so the model doesn't PROBE for arguments at session start
        # (the MCP inputSchema does not survive transport to Prime Agent's
        # client). Default off = pinned arms byte-identical; launcher turns it
        # on for new runs. See helpers._args_clause.
        describe_args: bool = False,
        # E8a planning injection: soft-gate np_down. "off" (default) | "norm"
        # (human-anchored) | "directive" (imperative). First np_down per dungeon
        # level returns the gate line at zero engine cost; the second proceeds
        # unconditionally. docs/EXPERIMENT_E8.md.
        descent_gate: str = "off",
        # E8b mechanic guidance: comma list of system-prompt blocks ("prayer",
        # "descend_pacing"). Prompt-only; published tool schemas untouched.
        mechanic_hints: str = "",
        # Resume-from-trace: a prior cell dir (or its `turns/` dir). At
        # setup_state the turn file for this rollout's seed is replayed
        # byte-for-byte through the freshly seeded engine, so the agent starts
        # exactly where the recorded session stopped -- with the UNSPENT part
        # of the budget (set max_turns/MAX_CALLS to the remainder when
        # launching). The conversation is NOT restored: both CLI scaffolds run
        # without session persistence (claude_code passes
        # --no-session-persistence; prime_agent's own auto-resume is a fresh
        # chat by design), so the first observation instead carries a RESUMED
        # note with the old session's tail. Replay integrity is verified
        # against the recorded end state and fails loudly on divergence.
        resume_from: Optional[str] = None,
        **kwargs,
    ):
        self.interface = interface
        self.task_spec = task_spec
        self._resume_from = resume_from or None
        self.pin_objective_on_setup = pin_objective_on_setup
        self.self_dispatch = self_dispatch
        # env_args flow through the eval CLI as dotted-scalar STRINGS, so
        # auto_dismiss can arrive as "false" -- and bool("false") is True.
        # (Caught in the e7 smoke: the dumped config showed the string form.)
        if isinstance(auto_dismiss, str):
            auto_dismiss = auto_dismiss.strip().lower() not in (
                "false", "0", "no", "off", "")
        self.auto_dismiss = bool(auto_dismiss)
        if isinstance(record_step_frames, str):
            record_step_frames = record_step_frames.strip().lower() not in (
                "false", "0", "no", "off", "")
        self.record_step_frames = bool(record_step_frames)
        if isinstance(describe_args, str):
            describe_args = describe_args.strip().lower() not in ("false","0","no","off","")
        self.describe_args = bool(describe_args)
        self.descent_gate = str(descent_gate or "off").strip().lower()
        self.mechanic_hints = str(mechanic_hints or "").strip().lower()
        self._setup_tune = setup_tune
        self._setup_modify = setup_modify
        self._setup_level_blob = setup_level_blob
        self._setup_character = setup_character
        self._no_progress_timeout = int(no_progress_timeout)
        self._allowed_skill_names = set(allowed_skill_names or ())
        # BALROG's 80-command surface (tools/balrog_actions.py) makes NetHack's
        # own prompts part of the agent's job: `bal_eat` opens "What do you want
        # to eat?" and the agent answers it next turn with `bal_d`-style keys.
        # The auto-dismiss loop below would ESC that prompt shut before the
        # agent ever saw it, so `bal_eat` could never actually eat and the arm
        # would score badly for a reason that has nothing to do with encoding.
        # --More-- acknowledgement is NOT disabled: BALROG runs `skip_more:
        # True`, so both harnesses skip those automatically.
        self._balrog_raw_prompts = any(
            n.startswith("bal_") for n in self._allowed_skill_names
        )
        # Pluggable LM backends. Both default to None → the rollout-time code
        # falls back to the deterministic Offline* implementations. Swap in
        # prime-rl-backed clients by passing them here from load_environment.
        self.sub_lm = sub_lm
        self.subgoal_proposer = subgoal_proposer
        # Compaction knobs. compact_obs=False reverts to the v0.0.15-era
        # raw rendering (good for replay / debugging / A/B). The history /
        # belief-state / journal knobs let you trade off LM context size
        # against semantic fidelity per run.
        self.compact_obs = compact_obs
        self.history_keep_full = history_keep_full
        self.history_drop_after = history_drop_after
        self.belief_state_interval = belief_state_interval
        self.journal_render_max_chars = journal_render_max_chars
        self.variant = variant
        self.map_detail = map_detail
        # Sub-experiment 1b: normalize the requested per-cell attribute set once.
        self.cell_schema = _normalize_cell_schema(cell_schema)
        # Single predicate gating ALL refiner machinery (teacher construction,
        # separation guard, bootstrap I/O, edit capture, spec hooks). variant=="CH"
        # always implies the refiner; `refine=True` opts ANY other variant in.
        self.refine_enabled = (variant == "CH") or bool(refine)
        # The prompt recipe. Holds the four parts (obs/system/template/tools)
        # plus the per-turn and history hooks; the rollout loop dispatches
        # through it instead of branching on `variant`. When refine is enabled on
        # a non-CH variant, attach the CH refiner bundle onto the resolved spec
        # (CH already carries it, so don't double-attach).
        _spec = spec if spec is not None else resolve_spec(variant, SYSTEM_PROMPT)
        if self.refine_enabled and variant != "CH":
            _spec = attach_refiner(_spec)
        self.spec = _spec
        self.refine_interval = refine_interval
        self.summarize_and_reset = summarize_and_reset
        self.trace_dir = trace_dir
        self.continual = continual
        self.continual_lives = continual_lives
        self.bootstrap_dir = bootstrap_dir
        # Lazy: only build a refiner when refine is enabled (variant=="CH" or
        # refine=True), so other variants don't pull in API clients.
        self.refiner = refiner
        self.refiner_model = refiner_model
        # Explicit escape hatch: run CH with a no-op OfflineRefiner (no teacher
        # required). Tagged via self._ch_real=False so traces / callers can tell
        # this apart from a real teacher-driven CH run. Popped BEFORE
        # super().__init__ so the verifiers base class doesn't choke on it.
        allow_offline_refiner = kwargs.pop("allow_offline_refiner", False)
        # Same-teacher separation override (read in setup_state where the policy
        # model id is observable). Popped here for the same reason.
        self.allow_same_teacher = bool(kwargs.pop("allow_same_teacher", False))
        self._ch_real = False
        if self.refine_enabled and self.refiner is None:
            from nethack_harness.refiner import (
                OfflineRefiner,
                TeacherLLMRefiner,
                resolve_teacher,
            )
            if allow_offline_refiner:
                self.refiner = OfflineRefiner()
            else:
                # Fail loud (CHMisconfigured) if no teacher can be resolved.
                # Thread the resolved base_url + key into the refiner so a key
                # resolved from OPENAI_API_KEY / PI_API_KEY (e.g. GLM via Prime
                # Inference) is actually used, not silently dropped.
                cfg = resolve_teacher(refiner_model)
                self.refiner = TeacherLLMRefiner(
                    model=cfg["model"],
                    base_url=cfg["base_url"],
                    api_key=cfg["api_key"],
                )
                self._ch_real = True
        super().__init__(*args, **kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        # Teacher/policy separation guard (whenever the refiner is enabled). The
        # policy model id is injected by verifiers at rollout time (init_state
        # sets state["model"]), so this is the first place we can compare it
        # against the teacher model.
        if self.refine_enabled:
            from nethack_harness.refiner import CHMisconfigured
            policy_model = state.get("model")
            teacher_model = self.refiner_model
            if not policy_model:
                # Policy id genuinely not observable: trust the operator.
                state["ch_separation"] = "operator-asserted"
            elif teacher_model and policy_model == teacher_model:
                state["ch_separation"] = "refused-same-model"
                if not self.allow_same_teacher:
                    raise CHMisconfigured(
                        f"refiner enabled (variant={self.variant!r}) but policy model "
                        f"{policy_model!r} equals the teacher model {teacher_model!r}; "
                        "teacher and policy must differ. "
                        "Pass allow_same_teacher=True to override."
                    )
            else:
                state["ch_separation"] = "separate"

        task: dict = state["task"]
        info: dict = state.get("info") or {}
        # Resolve the task spec (default: the full NetHack ascension game).
        seed: int = task.get("seed", info.get("seed", random.randint(0, 2**31 - 1)))
        spec = GAME_SPECS.get(self.task_spec, FULL_GAME_SPEC)

        # Game-setup overrides (difficulty/generation knobs, state pokes, custom
        # level). None = vanilla NetHack generation; these are the interface
        # flags that turn the standard game into a customized scenario.
        if spec.task_kind == "primitives":
            # Six-floor honest-navigation curriculum. Lazily imported so the
            # default game never requires the engine's invocation hooks (which
            # ship with engine PR #37 / feat/4b-six-floor-hooks).
            from nethack_harness.curriculum import CurriculumPrimitivesEnv
            env = CurriculumPrimitivesEnv(
                task_name=spec.nle_task,
                max_episode_steps=spec.max_episode_steps,
                tune=self._setup_tune,
                modify=self._setup_modify,
                level_blob=self._setup_level_blob,
            )
        else:
            env = NetHackCoreEnv(
                task_name=spec.nle_task,
                max_episode_steps=spec.max_episode_steps,
                des_file=spec.des_file,
                tune=self._setup_tune,
                modify=self._setup_modify,
                level_blob=self._setup_level_blob,
                # BALROG's `no_progress_timeout: 150` -- 150 consecutive env
                # steps without the in-game turn counter advancing aborts the
                # episode. It is the ONLY early stop in their protocol (their
                # step cap is 100k), so matching it is what makes an
                # uncapped-ish run affordable: a stalled agent is cut off
                # instead of grinding to the cap. Default 10_000 = effectively
                # off, preserving every existing arm byte-for-byte.
                no_progress_timeout=self._no_progress_timeout,
            )
        env.seed(core=seed, disp=seed)
        # NB: bootstrap_character() is currently a stub; once wired up it
        # auto-invokes #attributes and stores role/race/alignment in state.
        # setup_character pins the role for standard tiers (e.g. full_nle);
        # None keeps the engine default. Curriculum envs own their character.
        obs, meta = env.reset(character=self._setup_character)
        from nethack_harness.tools.skills import bootstrap_character
        character = bootstrap_character(env)

        state["env"] = env
        state["character"] = character
        # The tool names this rollout actually publishes, carried in state so the
        # observation renderer can gate HINT vocabulary on them WITHOUT a process
        # global (docs/HARNESS_DEFECTS.md 3.7: `render_system_prompt` used to
        # stash the set module-level and nothing ever put it back, so booting an
        # env changed every later render in the process). See
        # rendering.PUBLISHED_TOOLS_STATE_KEY.
        state[_PUBLISHED_TOOLS_STATE_KEY] = set(self._allowed_skill_names)
        # Continual-harness bookkeeping (no-op when self.continual=False).
        state["_orig_seed"] = int(seed)
        state["_continual_life"] = 1
        state["_continual_lives_left"] = self.continual_lives if self.continual else 0
        state["spec"] = spec
        state["meta"] = meta
        # Standard NLE tiers (full_nle etc.) paint the intro/copyright banner
        # over the top tty rows on every step; the engine-driven CurriculumEnv
        # does not, and owns its own map — so per-turn banner scrubbing is gated
        # to the standard tiers only (see the render path in env_response).
        state["_standard_tier"] = spec.nle_task != "engine"
        state["scout_tiles_seen"] = set()
        state["scout_delta"] = 0
        state["scout_reward_total"] = 0.0
        state["max_dlvl_reached"] = 1
        state["descent_count"] = 0
        state["raw_obs"] = obs
        state["structured_obs"] = shape_observation(obs, character)
        # Drain the intro/startup screen (copyright + version banner) and any
        # message-paging --More-- that NetHack gates behind --More-- on reset.
        # Without this, turn 1's observation carries the banner bleeding into
        # the rendered MAP plus a dangling --More--, garbling the agent's view.
        # We only press MORE/CR (byte 13) while a --More-- is ACTUALLY present
        # (never blindly, so a real [yn]/getlin prompt is left for the agent),
        # and wrap defensively: a drain failure must never break the rollout.
        try:
            more_idx_list = _to_action_indices(env, [13])
            if more_idx_list:
                for _ in range(10):
                    so0 = state["structured_obs"]
                    has_more = (
                        any("--More--" in m for m in (getattr(so0, "messages", None) or []))
                        or _obs_tty_has_more(state["raw_obs"])
                    )
                    if not has_more:
                        break
                    obs, _r0, term0, trunc0, _info0 = env.step(more_idx_list[0])
                    state["raw_obs"] = obs
                    if term0 or trunc0:
                        state["structured_obs"] = shape_observation(obs, character)
                        break
                    state["structured_obs"] = shape_observation(obs, character)
            # The intro/copyright banner is painted over the top tty rows and,
            # in right-offset-map tiers, is never repainted by gameplay — so it
            # bleeds into the rendered MAP. Scrub it from the raw obs before the
            # first observation is shaped/shown. Mutates raw_obs.tty_chars, then
            # re-shape so map_view reflects the cleaned tty.
            _scrub_intro_banner(state["raw_obs"])
            state["structured_obs"] = shape_observation(state["raw_obs"], character)
        except Exception:
            pass
        state["map_detail"] = self.map_detail
        # Sub-experiment 1b: the requested per-cell attribute set (consumed by
        # the JSON template) and the per-level visited-tile tracker. The tracker
        # is depth -> set[(x, y)]; seed it with the hero's start tile so the
        # visited_grid is non-empty on turn 1. We key by the hero's ACTUAL
        # current depth (blstats[12]) — NOT max_dlvl_reached, which is updated
        # only later in env_response (line ~940), so on a descent turn the tiles
        # would file under the old level while the template reads the new one.
        state["cell_schema"] = set(self.cell_schema)
        state["_visited_tiles"] = {}
        try:
            _bl = state["raw_obs"].blstats
            _hx, _hy = int(_bl[0]), int(_bl[1])
            state["_visited_tiles"].setdefault(int(_bl[12]), set()).add((_hx, _hy))
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        # Track every (depth, x, y) at which `>` was seen on the visible map.
        # Needed because once the player steps ONTO `>`, the @ overlay hides it
        # and the feature extractor stops finding the tile — without memory, the
        # agent oscillates on/off the stairs without realizing to descend. Keyed
        # by depth: this set is never cleared, and (x,y) means a different tile
        # on every floor (see rendering._remember_stairs_down).
        state["_seen_stairs_down"] = set()
        # Wave-2 Track B: visited-frontier memory. Tracks (level_key, (x,y)) →
        # consecutive turns the agent has been within 1 step of this frontier
        # without scout_delta > 0. When the count hits FRONTIER_STUCK_TURNS,
        # the frontier is blacklisted on its level. Blacklist resets on level
        # change (we key by max_dlvl_reached at sighting time). Cleared by
        # `_update_frontier_blacklist` each turn.
        state["_frontier_approach_count"] = {}  # (dlvl, x, y) -> int
        state["_frontier_blacklist"] = {}       # dlvl -> set[(x, y)]
        state["_frontier_prev_dlvl"] = 1
        # Deadlock-breaker flag (Track C reads this; we only set it). Becomes
        # True when all reachable frontiers on the current level are
        # blacklisted AND scout_delta has been 0 for >= NEEDS_HIDDEN_TURNS.
        state["_needs_hidden_passage"] = False
        state["_zero_scout_streak"] = 0
        # Per-turn obs-block flags, declared by the spec's ObsSpec.setup_flags
        # and read by the rendering functions:
        #   _descent_salient -> _descent_status_block (ND/FD)
        #   _e1_obs          -> E1 frontier/coverage/spatial-belief blocks
        #   _e2_obs          -> paint frontier-adjacent unseen tiles on the map
        # Every key is written for all variants (defaulting False) so the
        # rendering reads stay bit-identical to the legacy variant checks.
        _obs_flags = self.spec.obs.setup_flags
        state["_descent_salient"] = _obs_flags.get("_descent_salient", False)
        state["_e1_obs"] = _obs_flags.get("_e1_obs", False)
        state["_e2_obs"] = _obs_flags.get("_e2_obs", False)
        # Task 18 Step 2: gates the JOURNAL block + HINT ladder off for the
        # CLI-agent arms (see the constructor's self_dispatch docstring).
        state["_self_dispatch"] = self.self_dispatch
        state["_auto_dismiss"] = self.auto_dismiss
        state["last_reward"] = 0.0
        state["terminated"] = False
        state["journal"] = Journal()
        # Variant CH (Continual Harness) component slots. Always initialized so
        # downstream code can read them unconditionally; only populated when
        # variant=="CH" and the Refiner runs (or bootstrap_dir loads them).
        state["_ch_prompt_addendum"] = ""
        state["_ch_subagents"] = {}
        state["_ch_skills"] = {}
        # Pre-pin the spec's description as the agent's objective so the
        # goal stays in every obs (without forcing the model to call
        # pin_objective). Gated by pin_objective_on_setup so the 1c no-memory
        # arm (journal tools excluded + belief_state_interval=0) keeps the
        # Journal empty and its block un-rendered.
        if (
            self.pin_objective_on_setup
            and spec is not None
            and getattr(spec, "description", None)
        ):
            state["journal"].pin_objective(spec.description)
        if self.sub_lm is not None:
            state["sub_lm"] = self.sub_lm  # used by belief-state distillation
        if self.subgoal_proposer is not None:
            state["subgoal_proposer"] = self.subgoal_proposer

        # Bootstrap I/O for the refiner: if bootstrap_dir is set and a prior
        # snapshot exists for this seed, load the four components in.
        if self.refine_enabled and self.bootstrap_dir:
            try:
                import os, json
                path = os.path.join(self.bootstrap_dir, f"seed{seed}.json")
                if os.path.exists(path):
                    from nethack_harness.refiner import load_components
                    with open(path) as f:
                        load_components(state, json.load(f))
            except Exception:
                pass  # bootstrap failures must never break a rollout

        # --- Resume-from-trace (see the constructor's resume_from note) -----
        if getattr(self, "_resume_from", None):
            import glob as _glob, json as _json, os as _os
            _rroot = self._resume_from
            _cand = (_glob.glob(_os.path.join(_rroot, "turns", f"{seed}_*.ndjson"))
                     or _glob.glob(_os.path.join(_rroot, f"{seed}_*.ndjson")))
            if not _cand:
                raise RuntimeError(
                    f"resume_from: no turn file for seed {seed} under {_rroot}")
            _tf = max(_cand, key=_os.path.getmtime)
            _recs = [_json.loads(_l) for _l in open(_tf) if _l.strip()]
            if not _recs:
                raise RuntimeError(f"resume_from: {_tf} holds no turn records")
            _bad = [r["turn"] for r in _recs
                    if not (r.get("actions") or {}).get("replayable", False)]
            if _bad:
                raise RuntimeError(
                    f"resume_from: turn(s) {_bad[:5]} of {_tf} are not replayable")
            _obs = state["raw_obs"]
            for _r in _recs:
                for _b in ((_r.get("actions") or {}).get("bytes") or []):
                    _obs, _r2, _t2, _tr2, _i2 = env.step(int(_b))
                    if _t2 or _tr2:
                        raise RuntimeError(
                            f"resume_from: episode ended mid-replay of {_tf} "
                            f"(turn {_r['turn']}) -- the recorded session did "
                            "not end here, so this is replay divergence")
            state["raw_obs"] = _obs
            _scrub_intro_banner(state["raw_obs"])
            state["structured_obs"] = shape_observation(state["raw_obs"], character)
            # Verify the reconstruction against the recorded end state. dlvl
            # and hp together are a strong fingerprint; matching every turn
            # boundary was verified offline for all candidate seeds.
            _bl = state["raw_obs"].blstats
            _last = _recs[-1]
            for _name, _got, _want in (("dlvl", int(_bl[12]), _last.get("dlvl")),
                                       ("hp", int(_bl[10]), _last.get("hp"))):
                if _want is not None and int(_want) != _got:
                    raise RuntimeError(
                        f"resume_from: replay divergence on {_name} "
                        f"(replayed {_got}, recorded {_want}) for {_tf}")
            state["max_dlvl_reached"] = max(
                [int(r.get("max_dlvl_reached") or 1) for r in _recs] + [int(_bl[12])])
            state["_frontier_prev_dlvl"] = int(_bl[12])
            state["_visited_tiles"].setdefault(int(_bl[12]), set()).add(
                (int(_bl[0]), int(_bl[1])))
            # The NetPlay tracker never saw the replayed steps; force re-init.
            try:
                from nethack_harness.tools.netplay_true import reset_agent_cache
                reset_agent_cache()
            except Exception:
                pass
            _tail = []
            for _r in _recs[-3:]:
                _am = _r.get("assistant_message") or ""
                if isinstance(_am, dict):
                    _am = _am.get("content") or ""
                _am = " ".join(str(_am).split())
                if _am:
                    _tail.append(f"(their turn {_r['turn']}) {_am[:220]}")
            state["_resume_notice"] = (
                f"RESUMED SESSION: this game continues a previous session that "
                f"used {len(_recs)} calls. You are NOT starting fresh -- the "
                f"dungeon, your position, HP and inventory are exactly as that "
                f"session left them (check STATUS; use reveal to reorient). "
                + ("Final notes from that session: " + " | ".join(_tail)
                   if _tail else ""))

        # Materialize this rollout's turn file NOW, empty, so the stall
        # watchdog can see the rollout from second zero. The watchdog watches
        # `turns/*.ndjson` mtimes; a rollout that hangs BEFORE its first trace
        # write has no file and is therefore invisible to it -- measured
        # 2026-08-02: a retry wedged pre-first-turn sat undetected for the full
        # 2h rollout timeout while the 300s stall timeout stood idle. The
        # run_id is stamped into state exactly as `_write_trace_entry` builds
        # it, so the lazy path reuses this file rather than creating a second.
        # An empty file is harmless downstream: `select_turn_files` drops
        # "no turn rows" files at aggregation.
        if getattr(self, "trace_dir", None):
            try:
                import os as _os, time as _time
                from pathlib import Path as _Path
                out_dir = _Path(self.trace_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                seeds = state["env"].current_seeds if state.get("env") else (0, 0)
                run_id = f"{seeds[0]}_{_os.getpid()}_{int(_time.time())}"
                state["_trace_run_id"] = run_id
                (out_dir / f"{run_id}.ndjson").touch()
            except Exception:
                pass  # tracing must never break a rollout

        return state

    def _parse_tool_call(self, messages: vf.Messages, state: vf.State) -> tuple[str, dict]:
        # Parse the assistant's tool call from messages[-1].
        # In v0 we expect native function calling (OpenAI tool format).
        assistant_msg = messages[-1]
        # assistant_msg can be a dict (legacy) or a vf.AssistantMessage pydantic object (current).
        if isinstance(assistant_msg, dict):
            tool_calls = assistant_msg.get("tool_calls") or []
        else:
            tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        # Stashed for the trace write in `_apply_tool_call`, which no longer
        # shares this method's local scope now that parsing and applying are
        # split. MCP-driven callers that invoke `_apply_tool_call` directly
        # (bypassing this method) simply leave these unset; the trace write
        # degrades gracefully (see `_write_trace_entry`'s None handling).
        state["_last_assistant_msg"] = assistant_msg
        state["_last_tool_calls"] = tool_calls
        if not tool_calls:
            # Filter harness-owned skills from the suggestion list — they
            # don't appear in the actual tool schema sent to the model.
            agent_tools = [s for s in list_skills() if s not in ("menu_option", "inventory_item")]
            return "", {_NO_TOOL_CALL_SENTINEL: "You must call a tool. Available tools: " + ", ".join(agent_tools)}

        # Apply the first tool call (NetHack is turn-based; we ignore multi-call this turn).
        # Verifiers passes tool calls in two shapes depending on version:
        #   old: dict {"function": {"name": ..., "arguments": "..."}}
        #   new: ToolCall pydantic model with flat .name / .arguments
        tc = tool_calls[0]
        # Surface that we dropped the extras so the agent knows only the
        # first call ran — otherwise it might assume all N actions were
        # applied and plan around a stale game state.
        state["_dropped_extra_tool_calls"] = max(0, len(tool_calls) - 1)
        import json
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            skill_name = fn.get("name", tc.get("name", "")) or ""
            raw_args = fn.get("arguments", tc.get("arguments", "{}"))
        else:
            fn = getattr(tc, "function", None)
            skill_name = (getattr(fn, "name", None) if fn is not None else getattr(tc, "name", "")) or ""
            raw_args = getattr(fn, "arguments", None) if fn is not None else getattr(tc, "arguments", "{}")

        # Defensive parsing: small models emit malformed args. Coerce to dict
        # so we never crash on dispatch.
        if raw_args is None or raw_args == "":
            skill_args = {}
        elif isinstance(raw_args, dict):
            skill_args = raw_args
        else:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    skill_args = parsed
                else:
                    # Model emitted a non-dict (list, scalar, ...). Treat as empty
                    # and let the skill registry surface a friendly error.
                    skill_args = {}
            except (ValueError, TypeError):
                # Malformed JSON — same recovery path.
                skill_args = {}
        return skill_name, skill_args

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        skill_name, skill_args = self._parse_tool_call(messages, state)
        content = await self._apply_tool_call(state, skill_name, skill_args)
        return [vf.UserMessage(role="user", content=content)]

    def _render_obs_text(self, state: vf.State, journal=None) -> str:
        """`spec.turn_template`, prefixed with a blocking-UI warning when one applies.

        Every per-turn render goes through here. The `=== MAP ===` block is
        built from the glyph plane (`prompt/ascii_map.py`), which has no
        representation for an open menu or prompt — so without this prefix an
        agent that opens the inventory sees an unchanged, normal-looking map
        while the game clock is frozen, and repeats the identical observation
        forever. Measured: 3 of 5 `b80_b0` seeds burned 2,499 calls each at
        game time 1 this way. See `prompt/interactive_state.py`.
        """
        obs_text = self.spec.turn_template(
            state["structured_obs"],
            state["journal"] if journal is None else journal,
            state,
            compact=self.compact_obs,
            journal_max_chars=self.journal_render_max_chars,
        )
        warning = detect_blocking_ui(
            state.get("raw_obs"), published_tools=state.get("_published_tools"))
        return f"{warning}\n{obs_text}" if warning else obs_text

    async def _apply_tool_call(self, state: vf.State, skill_name: str, skill_args: dict):
        """Instrument one LM turn, run it, and write exactly one trace record.

        The trace write used to live at the bottom of the turn body, which meant
        every early return (hallucinated tool, journal-only call, no tool call at
        all) produced an LM turn with NO trace record -- so `len(records)` could
        not be reconciled with `metrics.num_turns` even before the max_turns
        off-by-one. Hoisting it here makes the invariant structural: one call to
        this method == one record, whatever the body does.

        The `TurnRecorder` context wraps the WHOLE body so it captures engine
        steps taken anywhere inside it, including inside closed-loop skills.

        TWO ROUTES REACH HERE, and they differ only in what they can see of the
        MODEL, never in what happens to the game. `env_response` gets here via
        `_parse_tool_call`, which stashes the assistant message and its parsed
        calls in `state`. An MCP-driven CLI agent (`NetHackToolset._executing`)
        calls this method directly, in a different process from the one the
        model talks to, so those breadcrumbs are never set. Measured before this
        split was made explicit: `tool_calls` was empty on 100% of CLI-arm turns
        even though the dispatched name and arguments are RIGHT HERE as
        arguments to this method -- `tool_results[i]["name"]` had already been
        populated from them. The record now carries the same call on both
        routes, plus `dispatch_route` saying which one it was.
        """
        # `_last_tool_calls` is set on EVERY harness turn (including turns with
        # no tool call, where it is `[]`), so `is None` distinguishes the routes
        # without a flag the caller could forget to pass.
        parsed_calls = state.get("_last_tool_calls")
        route = "harness" if parsed_calls is not None else "mcp"
        state["_trace_lm_turn"] = int(state.get("_trace_lm_turn", 0)) + 1
        tt = state["_turn_trace"] = {
            "status": None, "action_indices": [], "reward": 0.0, "feedback": "",
        }
        clock_before = _game_clock(state)
        rec = TurnRecorder(state.get("env"), capture_frames=self.record_step_frames)
        with rec:
            content = await self._apply_tool_call_inner(state, skill_name, skill_args)
        clock_after = _game_clock(state)
        obs_text = content_to_text(content)
        # Stash what the model was last SHOWN, so the end-of-rollout flush (the
        # final, never-applied tool call) can record the observation that call
        # was actually made against.
        state["_last_rendered_user_message"] = obs_text
        result = build_tool_result(
            name=skill_name or None,
            arguments=skill_args if isinstance(skill_args, dict) else None,
            feedback=tt.get("feedback") or "",
            status=tt.get("status"),
            clock_before=clock_before, clock_after=clock_after,
            engine_steps=len(rec.actions), reward=tt.get("reward") or 0.0,
            game_message=rec.messages[-1] if rec.messages else "",
        )
        # On the MCP route there is no parsed assistant message to read the call
        # out of, but the call itself was handed to us -- so synthesize the same
        # `{"name", "arguments"}` shape the harness route produces. `arguments`
        # is the already-parsed dict here (the MCP client parsed the JSON), which
        # matches `tool_results[i]["arguments"]` on every other path; on the
        # harness route it stays the raw JSON string the model emitted, exactly
        # as version-0/1 readers expect.
        if parsed_calls is not None:
            trace_calls = parsed_calls
        elif skill_name:
            trace_calls = [{"name": skill_name,
                            "arguments": skill_args if isinstance(skill_args, dict) else {}}]
        else:
            trace_calls = []
        _write_trace_entry(
            self, state, state.get("_last_assistant_msg"),
            trace_calls,
            tt.get("action_indices") or [], tt.get("reward") or 0.0,
            obs_text, obs_content=content,
            actions=rec.action_record(), tool_results=[result],
            all_messages=rec.messages, step_frames=rec.frames,
            # `applied` = "this LM turn was dispatched by the harness". It is
            # False only for the end-of-rollout flush, whose tool call the
            # rollout loop never handed to us at all. A dispatched call that
            # took zero engine steps (no path found, hallucinated tool) is still
            # applied=True: `actions.n == 0` is what says nothing ran, and that
            # is a different fact.
            applied=True,
            dispatch_route=route,
        )
        return content

    async def _apply_tool_call_inner(self, state: vf.State, skill_name: str, skill_args: dict):
        """Execute one skill against the engine and return the rendered observation.

        This is the whole `env_response` body minus tool-call parsing: the gate,
        journal short-circuit, registry dispatch, engine stepping, menu drain,
        terminal detection, reward bookkeeping, banner re-scrub and render. Split
        out so MCP-driven CLI harnesses (which must return the observation from
        the tool call itself) share one code path with the native harness.

        Trace bookkeeping is the caller's job; this body only leaves breadcrumbs
        in `state["_turn_trace"]` at each of its several return points.
        """
        tt = state.setdefault("_turn_trace", {})
        # The skill this turn's observation is a response TO. Stamped before
        # any render so a turn template can condition on it — variant BBOX_MIN
        # withholds the entity/message blocks except on `reveal` turns, and
        # this is how its template knows which turn it is rendering.
        state["_last_skill_name"] = skill_name or ""
        # One yes-answer attempt per TURN (see the skill-initiated confirm
        # branch in the dismissal loop); reset here so the next turn's skill
        # gets its own attempt.
        state.pop("_confirm_yes_tried", None)
        # `_parse_tool_call` encodes "the model emitted no tool call at all" as
        # this sentinel (there is no skill to apply, so nothing below — the
        # gate, dispatch, engine stepping — applies). Surface the same
        # "must call a tool" text the pre-split env_response returned verbatim.
        if skill_name == "" and _NO_TOOL_CALL_SENTINEL in skill_args:
            tt["status"] = "no_tool_call"
            tt["feedback"] = "model emitted no tool call this turn"
            return skill_args[_NO_TOOL_CALL_SENTINEL]

        # Dead character: refuse to step the engine. The native verifiers
        # rollout loop already stops calling in via `is_completed` once
        # `state["terminated"]` is set, but the MCP-exposed toolset (Claude
        # Code, Prime Agent) calls `_apply_tool_call` directly, per tool
        # call, with no such gate in between — that is exactly how the
        # committed arm-2 acceptance rollout burned 7 of its 12 calls on a
        # tombstone `--More--` screen after dying at call 6. Once
        # `state["died"]` is set (by the zero-HP check or either terminal
        # detector below), every further call is a no-op against the engine:
        # no `env.step`, no turn-counter advance, no reward.
        # EXCEPT `rollback`. The system prompt's headline affordance is "if you
        # are about to die, roll back and choose differently" -- and this gate
        # sits BEFORE skill dispatch, so death was the one moment rollback could
        # not be used. Observed live: post-death `rollback(3)` returned "no
        # further actions are possible" without touching the snapshot stack.
        # Undoing into a live state clears `died` and the run genuinely resumes.
        if state.get("died") and skill_name == "rollback":
            _env = state["env"]          # `env` is not bound this early in the fn
            _res = skill_registry.call("rollback", _env, state["structured_obs"], **skill_args)
            for _a in (_res.actions or []):
                try:
                    _o, _r, _t, _tr, _i = _env.step(_a)
                    state["raw_obs"] = _o
                except Exception:
                    break
            state["structured_obs"] = shape_observation(state["raw_obs"], state["character"])
            if (state["structured_obs"].status or {}).get("hitpoints", 0) > 0:
                state["died"] = False
                state["terminated"] = False
            state["_stuck_n"] = 0
            tt["feedback"] = _res.feedback or ""
            content = self.spec.turn_template(
                state["structured_obs"], state["journal"], state,
                compact=self.compact_obs,
                journal_max_chars=self.journal_render_max_chars,
            )
            return compose_user_content(content, [f"[{_res.feedback}]"])

        if state.get("died"):
            content = self.spec.turn_template(
                state["structured_obs"], state["journal"], state,
                compact=self.compact_obs,
                journal_max_chars=self.journal_render_max_chars,
            )
            content = compose_user_content(
                content,
                ["[Your character is dead. The game is over; no further actions are possible.]"],
            )
            tt["status"] = "dead"
            tt["feedback"] = "character is dead; call refused without touching the engine"
            return content

        env: NetHackCoreEnv = state["env"]

        # Gate hallucinated tool calls: reject any skill the model was NOT given
        # in its tool schema this rollout, instead of dispatching it against the
        # full registry. Without this a model under the `netplay` set (which
        # withholds low-level `move`) can still emit `move(direction=…)` and have
        # it EXECUTE — leaking the primitive into the effective action surface and
        # confounding "hold actions fixed, vary the observation" comparisons.
        # Checked on the ORIGINAL name, before the dir8 rebind below (dir8's
        # compass tools ARE in the exposed set). No NLE step is consumed.
        if self._allowed_skill_names and skill_name not in self._allowed_skill_names:
            avail = ", ".join(sorted(self._allowed_skill_names))
            tt["status"] = "rejected"
            tt["feedback"] = f"tool {skill_name!r} is not in this rollout's exposed set"
            obs_text = self._render_obs_text(state)
            content = compose_user_content(
                obs_text,
                [f"[Tool {skill_name!r} is not available. Call one of: {avail}]"],
            )
            return content

        # dir8 baseline: rewrite north/northeast/.../northwest calls to
        # move(direction=...) so the existing dispatcher handles them.
        _DIR_BIND = {
            "north": "N", "northeast": "NE", "east": "E", "southeast": "SE",
            "south": "S", "southwest": "SW", "west": "W", "northwest": "NW",
        }
        if skill_name in _DIR_BIND:
            skill_args = {"direction": _DIR_BIND[skill_name]}
            skill_name = "move"

        # Variant CH: `run_macro(name=...)` expands a Refiner-registered
        # macro (an ordered list of existing skill calls) into a concatenated
        # SkillResult. Resolution happens here, BEFORE the registry dispatch
        # below, so the rest of env_response sees a normal SkillResult.
        if skill_name == "run_macro":
            from nethack_harness.tools.skills import SkillResult as _SR
            macro_name = (skill_args or {}).get("name", "")
            macro = (state.get("_ch_skills") or {}).get(macro_name)
            if not macro:
                result = _SR(actions=[], feedback=f"Unknown macro: {macro_name!r}", interrupted=True)
            else:
                actions_acc: list = []
                fb_parts: list[str] = []
                interrupted = False
                for step_def in macro:
                    sub_name = step_def.get("skill")
                    sub_args = dict(step_def.get("args") or {})
                    if not sub_name:
                        continue
                    sub_res = skill_registry.call(sub_name, env, state["structured_obs"], **sub_args)
                    if sub_res.journal_op is not None:
                        try:
                            sub_res.journal_op(state["journal"])
                        except Exception:
                            pass
                    actions_acc.extend(sub_res.actions)
                    if sub_res.feedback:
                        fb_parts.append(f"{sub_name}: {sub_res.feedback}")
                    if sub_res.interrupted:
                        interrupted = True
                        break
                result = _SR(
                    actions=actions_acc,
                    feedback=f"[macro:{macro_name}] " + " | ".join(fb_parts),
                    interrupted=interrupted,
                )
        # Code-mode dispatch: if the model called the `code` tool, run the
        # source against the nh namespace and convert its action queue into
        # a SkillResult shape so the rest of env_response can stay unchanged.
        elif self.interface == "code" and skill_name == "code":
            from nethack_harness.tools.code_mode import run_user_code
            from nethack_harness.tools.skills import SkillResult
            source = skill_args.get("source", "")
            cm_result = run_user_code(
                source, env, state["structured_obs"], journal=state.get("journal"),
                raw_obs=state.get("raw_obs"),
            )
            stdout = cm_result.stdout or ""
            err = f"\n[code error: {cm_result.error}]" if cm_result.error else ""
            feedback = (stdout + err).strip() or "(code executed; no stdout)"
            result = SkillResult(actions=cm_result.actions_taken, feedback=feedback)
        else:
            result = skill_registry.call(
                skill_name, env, state["structured_obs"], **skill_args
            )

        # ---- stuck-call breaker: RECORD ONLY --------------------------
        # The decision cannot be made here. At this point this turn's actions
        # have not run, so the only clock available is last turn's -- comparing
        # it against a value also written last turn is trivially equal, which
        # silently degrades the check back to pure call identity. That is
        # exactly the bug that made this fire on the kick that smashed a door
        # open (dtime 96) and on the call that found the stairs. So capture the
        # signature and the pre-action clock now, and judge after the engine has
        # actually stepped (see the post-step block below).
        # E8a descent gate: the FIRST np_down on a given dungeon level returns
        # the norm line at zero engine cost; the second proceeds. Soft gate --
        # agency preserved, the game is never blocked. docs/EXPERIMENT_E8.md.
        # Gate the descent ACTION, not one tool name: np_core has no np_down —
        # agents descend via np_press_key('>'). (Caught live in E8a attempt 1:
        # seeds reached Dlvl 8-11 with zero np_down calls and zero gate lines.)
        _is_descent = (
            skill_name == "np_down"
            or (skill_name == "np_press_key"
                and str((skill_args or {}).get("key", "")).strip() == ">")
        )
        if self.descent_gate in ("norm", "directive") and _is_descent:
            try:
                st_now = (state["structured_obs"].status or {})
                dlvl = int(st_now.get("depth") or 1)
                xl = int(st_now.get("experience_level") or 1)
            except Exception:
                dlvl, xl = 1, 1
            ack = state.setdefault("_descent_gate_ack", set())
            if dlvl not in ack:
                ack.add(dlvl)
                from nethack_harness.prompt.human_norms import norm_xl_for_leaving
                norm = norm_xl_for_leaving(dlvl)
                if self.descent_gate == "norm":
                    gate = (f"[descent check: you are XL {xl} on Dlvl {dlvl}. "
                            f"Typical successful human runs reach XL {norm} "
                            f"before leaving this depth. Repeat the call to "
                            f"descend anyway.]")
                else:
                    gate = (f"[descent check: level to XL {norm} before moving "
                            f"on. Repeat the call to descend anyway.]")
                tt = state["_turn_trace"]
                tt["status"] = "interrupted"
                tt["feedback"] = gate
                state["scout_delta"] = 0
                obs_text = self._render_obs_text(state)
                return compose_user_content(obs_text, [gate])

        state["_sig_now"] = (skill_name, repr(sorted(skill_args.items())))
        try:
            state["_gt_before"] = (state["structured_obs"].status or {}).get("time")
        except Exception:
            state["_gt_before"] = None

        # Sub-experiment 1d (delayed-map / DM variants): request_map and reveal
        # are info-only skills (empty actions) that force the FULL map back into
        # this turn's rendered observation. Skills can't reach `state`, so the
        # DM template's force flag is set here in env_response (which owns
        # `state`); _delayed_map_template pops it. No NLE step is consumed.
        if skill_name in ("request_map", "reveal"):
            state["_force_map"] = True

        # Journal skills: apply the journal op and short-circuit the env step.
        # No NLE turn is consumed; the agent's next prompt reflects the change.
        if result.journal_op is not None:
            journal: Journal = state["journal"]
            feedback = result.journal_op(journal)
            tt["status"] = "no_op"
            tt["feedback"] = feedback or "journal op; no engine step by design"
            state["scout_delta"] = 0  # no exploration happened
            obs_text = self._render_obs_text(state, journal)
            content = compose_user_content(obs_text, [f"[{feedback}]"] if feedback else [])
            return content

        # Capture pre-step scout set size so scout_reward can return a per-step delta
        # rather than a cumulative count. See onboarding/scout_reward.md.
        scout_before = len(state["scout_tiles_seen"])
        # Capture pre-step player (x, y) so we can detect blocked moves.
        pre_pos = None
        try:
            pre_blstats = state["raw_obs"].blstats if state.get("raw_obs") is not None else None
            if pre_blstats is not None:
                pre_pos = (int(pre_blstats[0]), int(pre_blstats[1]))
        except (KeyError, IndexError, TypeError, AttributeError):
            pre_pos = None

        # Skills can return either NLE action enum values (107 == N) or task
        # action-set indices (1 == N for NetHackScore). The underlying gym
        # step expects indices, so convert at the boundary.
        action_indices = _to_action_indices(env, result.actions)

        # Step the underlying env through the action sequence the skill produced.
        # Multi-action skills (autoexplore, move_to) expand into many env.step
        # calls; we halt early on three conditions to give the model a chance
        # to react before walking into a dragon: HP-drop, hostile-in-sight,
        # explicit terminal. This is the "halt on hostile/HP-drop/hunger" item
        # from the project plan.
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        last_obs = state["raw_obs"]
        hp_before = state["structured_obs"].status.get("hitpoints", 0) if state.get("structured_obs") else 0
        halt_reason: Optional[str] = None
        if getattr(result, "pre_executed", False):
            # Closed-loop skill (e.g. explore_and_descend) already stepped the env
            # in its own re-observe loop. Adopt its outcome and skip our step loop.
            total_reward = result.pre_reward
            last_obs = result.final_obs if result.final_obs is not None else state["raw_obs"]
            terminated = bool(result.pre_terminated)
            truncated = bool(result.pre_truncated)
            action_indices = []
            # `action_indices == []` means the loop below never runs, so
            # scout_tiles_seen / _visited_tiles would otherwise never see this
            # call at all (scout_reward is structurally zero for every
            # pre_executed skill). `pre_visible_obs` is opt-in (defaults to
            # None): only netplay_true's `run_netplay_skill` sets it, so the
            # hand-written `netplay` set's own pre_executed skill
            # (explore_and_descend) does not go through this branch and its
            # behaviour is unchanged.
            for step_obs in (getattr(result, "pre_visible_obs", None) or []):
                _record_scout_and_visited(state, step_obs)
        cr_swallowed = 0
        for step_i, action in enumerate(action_indices):
            # Swallow a stray carriage return. A CR that lands in command
            # context is never anything but `Unknown command '^M'.` on the top
            # line, which then persists in the observation for many turns and
            # has demonstrably sent agents chasing an imaginary "stuck input
            # buffer" (see `_cr_would_be_unknown_command`). Dropping it here --
            # the single funnel every skill's keystrokes pass through -- makes
            # it a no-op instead. A CR that a prompt is waiting for is untouched.
            if action == CARRIAGE_RETURN and _cr_would_be_unknown_command(last_obs):
                cr_swallowed += 1
                continue
            last_obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
            _record_scout_and_visited(state, last_obs)
            if terminated or truncated:
                break
            # Status-aware halt: check after each step (cheap — just blstats).
            # Only enabled for multi-step skills (>=4 actions in a single tool
            # call) so single-key skills aren't penalized by the overhead.
            if len(action_indices) >= 4 and step_i + 1 < len(action_indices):
                halt_reason = _check_halt_condition(last_obs, hp_before)
                if halt_reason:
                    break
                # Also halt if a y/n / menu prompt opened mid-sequence — the
                # remaining action indices would be consumed as keystroke
                # answers to the prompt rather than continuing the intended
                # action sequence (e.g. autoexplore step 16 would answer
                # "Really attack?" as 'n' instead of moving NE).
                msg_bytes = last_obs.get("message") if isinstance(last_obs, dict) else None
                if msg_bytes is not None:
                    msg = bytes(msg_bytes).split(b"\x00", 1)[0].decode("ascii", errors="replace")
                    if "[yn" in msg or "--More--" in msg:
                        halt_reason = "prompt opened mid-sequence"
                        break

        # Observability: a non-zero count means a skill tried to feed the engine
        # a CR in command context. It is harmless now, but it still points at a
        # skill that is emitting keystrokes it does not need.
        state["cr_swallowed_total"] = int(state.get("cr_swallowed_total", 0)) + cr_swallowed

        scout_after = len(state["scout_tiles_seen"])
        state["scout_delta"] = scout_after - scout_before
        # Accumulate cumulative scout reward: the rubric scores once at end of
        # rollout, so a per-step `scout_delta` alone would only reflect the
        # final step. Sum here so scout_reward can report total exploration.
        state["scout_reward_total"] += state["scout_delta"] / 1000.0

        state["raw_obs"] = last_obs
        state["structured_obs"] = shape_observation(last_obs, state["character"])
        # Auto-dismiss any menu/inventory_prompt that's still open. Menus are
        # mechanical (--More--, level-up choice picker, multi-page item lists)
        # and were a huge time-sink for the LM agent: Qwen3.5-9B spent 42% of
        # turns on menu_option / inventory_item calls (often nonsensical) before
        # this hook. By auto-pressing ESC, the harness owns the menu-navigation
        # responsibility and the agent sees a clean post-menu observation on the
        # next turn. The `eat`/`quaff`/`read` skills now bundle item selection
        # in-skill, so intentional inventory prompts also resolve here.
        dismissed = 0
        esc_idx_list = _to_action_indices(env, [27])
        more_idx_list = _to_action_indices(env, [13])
        y_idx_list = _to_action_indices(env, [ord('y')])
        n_idx_list = _to_action_indices(env, [ord('n')])
        esc_action = esc_idx_list[0] if esc_idx_list else (more_idx_list[0] if more_idx_list else None)
        y_action = y_idx_list[0] if y_idx_list else esc_action
        n_action = n_idx_list[0] if n_idx_list else esc_action
        # What KINDS of thing the loop closes, for the notice below. A one-line
        # item prompt, a --More-- acknowledgement and a real menu are three
        # different situations to the agent and must not share one label.
        saw_prompt, saw_more, saw_menu = False, False, False
        prompt_txt = None
        for _ in range(8):
            so = state["structured_obs"]
            yn = getattr(so, "yn_prompt", None)
            # Detect --More-- prompts in the message buffer too — they consume
            # the next keystroke, which would otherwise eat the model's
            # intended action. MORE/CR (13) acknowledges them.
            has_more = any("--More--" in m for m in (so.messages or [])) or _obs_tty_has_more(last_obs)
            if so.menu is None and so.inventory_prompt is None and yn is None and not has_more:
                break
            if so.inventory_prompt is not None or yn is not None:
                saw_prompt = True
                if prompt_txt is None:
                    for _m in reversed(so.messages or []):
                        if "?" in _m:
                            prompt_txt = _m.strip()
                            break
            elif has_more:
                saw_more = True
            else:
                saw_menu = True
            # Under the BALROG raw-command surface, answering menus / item
            # prompts / y-n questions is the AGENT's job (see
            # `_balrog_raw_prompts` in __init__). Only --More-- is still
            # acknowledged for it, matching BALROG's own `skip_more: True`.
            if (self._balrog_raw_prompts or not self.auto_dismiss) and not has_more:
                break
            if yn is not None:
                ans = yn["answer"]
                # Skill-initiated confirmation (exp4 fix 4, RELOCATED): the
                # first version of this lived in the `else` branch below and
                # was UNREACHABLE -- `extract_yn_prompt` parses any "[ynq]"
                # message into `yn_prompt`, so a [ynq] confirm always lands
                # HERE, where the parsed default for "loot it? [ynq] (q)" is
                # ESC and the chest never opened (the measured 10-call
                # brute-force spiral). When the skill the agent JUST called
                # opened a matching confirm, answer y ONCE; if the prompt
                # survives, the next iteration falls through to the default
                # so this can never loop. Attack/really prompts are excluded.
                if ans != "y" and not state.get("_confirm_yes_tried"):
                    if _confirm_yes_for(state.get("_last_skill_name"),
                                        so.messages or []):
                        ans = "y"
                        state["_confirm_yes_tried"] = True
                action = y_action if ans == "y" else (n_action if ans == "n" else esc_action)
            elif has_more:
                # MORE prompts want CR/space, not ESC.
                action = more_idx_list[0] if more_idx_list else esc_action
            else:
                # A real menu / inventory prompt with no yn question: ESC. The
                # skill-initiated [ynq] whitelist lives in the `yn is not None`
                # branch above -- its first home here was unreachable, because
                # any message containing "[ynq]" is parsed into `yn_prompt`.
                action = esc_action
            if action is None:
                break
            last_obs, _r, t2, tr2, _info = env.step(action)
            terminated = terminated or t2
            truncated = truncated or tr2
            state["raw_obs"] = last_obs
            state["structured_obs"] = shape_observation(last_obs, state["character"])
            dismissed += 1
            if terminated or truncated:
                break
        if dismissed:
            # Those ESC/CR/y/n presses went through `env.step` directly, not
            # through `agent.step`, so the NetPlay tracker never saw them and is
            # now stale -- the measured cause of map / VISIBLE MONSTERS /
            # tracker three-way disagreement and spurious "no monster at (x,y)".
            try:
                from nethack_harness.tools.netplay_true import reset_agent_cache
                reset_agent_cache()
            except Exception:
                pass
            # Report the SITUATION, not the mechanism. "menu auto-dismissed xN"
            # described the harness's plumbing and misnamed the common case (a
            # one-line item prompt is not a menu); the agent was left with three
            # open questions -- did the action happen? is something pending?
            # what now? -- and answered them by retrying: exp3 seed 4 called
            # `wear` (no item_letter) 11 times in a row. The notice now answers
            # all three inline, quoting the game's own prompt where one was
            # captured. Resume semantics are stated flatly because they are
            # simple: ESC returns the game to the command prompt, nothing is
            # ever pending after this loop.
            notices = []
            if saw_prompt:
                _last = state.get("_last_skill_name") or ""
                asked = f'NetHack asked "{prompt_txt}"' if prompt_txt else "NetHack asked which item to use"
                if _last in _DISMISSAL_ITEM_SKILLS:
                    # Suggest a letter the prompt actually offered ("[bcde or
                    # ?*]" -> 'b'), so the example is directly usable.
                    _m = re.search(r"\[\$?([A-Za-z])", prompt_txt or "")
                    _eg = _m.group(1) if _m else "a"
                    notices.append(
                        f"{asked} -- '{_last}' ended without choosing, so the "
                        f"harness pressed ESC. Nothing happened and no game time "
                        f"passed. Nothing is pending. To do it, call "
                        f"{_last} again with the item's inventory letter from "
                        f"that prompt, e.g. {_last}(item_letter='{_eg}')"
                    )
                else:
                    notices.append(
                        f"{asked} -- the harness pressed ESC, so nothing "
                        f"happened and nothing is pending"
                    )
            if saw_menu:
                notices.append("harness closed a leftover menu; the game is back at the command prompt")
            if saw_more:
                notices.append("harness acknowledged --More--")
            # NOT merged into halt_reason: "autohalt" is a skill being
            # interrupted mid-run, and this is post-skill cleanup. Each gets
            # its own bracketed prefix so the agent never reads a closed
            # prompt as an interrupted plan.
            state["_dismiss_notice"] = "; ".join(notices) or f"harness cleared a blocking prompt x{dismissed}"
        # Snapshot the post-turn state so `rollback(n)` has somewhere to go.
        # Only when the tool is actually published, because each snapshot is a
        # full engine heap image -- every other arm must not pay for a feature
        # it cannot use. Taken AFTER the menu auto-dismiss loop so a restored
        # frame is a clean, actionable state rather than a half-open menu.
        if "rollback" in self._allowed_skill_names and not (terminated or truncated):
            from nethack_harness.tools.skills import push_rollback_snapshot
            try:
                push_rollback_snapshot(env, state.get("turn_count", 0))
            except Exception:
                pass  # never let snapshotting break a rollout
        # ---- stuck-call breaker: JUDGE ---------------------------------
        # Now the engine HAS stepped, so `gt_after` vs `gt_before` is a real
        # measurement of whether this turn moved the game at all. Repeating
        # `explore_level` is correct NetHack play and must never be flagged on
        # its own; what is worth flagging is a call that repeats AND freezes the
        # clock. Warning goes on `halt_reason`, which reaches the next
        # observation.
        try:
            _gt_after = (state["structured_obs"].status or {}).get("time")
            _same_call = state.get("_sig_prev") == state.get("_sig_now")
            _frozen = (_gt_after is not None
                       and _gt_after == state.get("_gt_before")
                       and _same_call)
            state["_stuck_n"] = int(state.get("_stuck_n", 0)) + 1 if _frozen else 0
            state["_sig_prev"] = state.get("_sig_now")
            if int(state.get("_stuck_n", 0)) >= 2:
                _n = int(state["_stuck_n"]) + 1
                halt_reason = ((halt_reason or "") +
                    f" [STUCK: this exact call has run {_n} times and the game "
                    f"clock has not advanced once. Do something DIFFERENT.]").strip()
        except Exception:
            pass
        state["last_reward"] = total_reward
        state["terminated"] = terminated or truncated
        # Refiner: on terminal, persist the refined components for the
        # next rollout (if bootstrap_dir is configured).
        if state["terminated"] and self.refine_enabled:
            _ch_save_bootstrap(self, state)
        # Continual harness mode: if the agent died (not ascended), and lives
        # remain, auto-reseed and reset the NLE env so the chat session
        # continues into a new game. Journal + belief state survive across
        # lives — this is the "memory persists; episodes don't" pattern.
        if (
            self.continual
            and (terminated or truncated)
            and not state.get("ascended", False)
            and state.get("_continual_lives_left", self.continual_lives) > 0
        ):
            try:
                _continual_reset(state, env, self)
                terminated = truncated = False
                state["terminated"] = False
            except Exception as e:
                # Best-effort: if reset fails, end the rollout normally.
                state["_continual_error"] = repr(e)
        # BALROG-style progression score (informational; not in rubric).
        # Tracks deepest (DL, XL) achieved as an empirical-ish P(ascend).
        from nethack_harness.prompt.balrog import progression_score
        s = state["structured_obs"].status
        state["balrog_progression"] = progression_score(
            state["max_dlvl_reached"], s.get("experience_level", 1)
        )
        # Death/ascension detection from the game state, not raw NLE termination flag.
        _detect_terminal_outcome(last_obs, state)
        # Robust death fallback: the text-marker scan above misses most deaths
        # because the death / "Do you want your possessions identified?" screen is
        # auto-dismissed inside closed-loop skills (explore_and_descend) before
        # env_response ever sees it — so `died` was only catching ~1 in 7 deaths.
        # NLE's terminated flag is authoritative: a game that NLE ended and that
        # we did NOT detect as an ascension is, at these depths, a death. (Milestone
        # success sets state["terminated"] separately, AFTER this block, so it can't
        # be confused for a death here.)
        if terminated and not state["ascended"] and not state["died"]:
            state["died"] = True
            state.setdefault("death_dlvl", state.get("max_dlvl_reached", 1))
        # Zero-HP fallback: neither detector above catches a death whose
        # message screen never reaches the marker scan and whose NLE
        # `terminated` flag never fires — e.g. NetHack's death sequence parks
        # on a prompt chain (Final Attributes -> possessions -> tombstone)
        # that the env's own auto-dismiss loop cannot outlast, or a raw state
        # poke (`modify={"hp": 0}`) that bypasses the engine's death codepath
        # entirely. `hitpoints == 0` in the shaped status is authoritative
        # for death regardless of how the game got there: this is an
        # ADDITIONAL path, not a replacement for the two detectors above, so
        # a real ascension or a message-detected death still short-circuits
        # first via the `state["died"]`/`state["ascended"]` guards.
        if not state["ascended"] and not state["died"] and s.get("hitpoints", 1) == 0:
            state["died"] = True
            state["terminated"] = True
            state.setdefault("death_dlvl", state.get("max_dlvl_reached", 1))
        # Milestone-driven success: if the tier's success_milestone fires, we
        # treat the rollout as won and let success_reward pay out.
        spec = state.get("spec")
        if spec is not None and getattr(spec, "success_milestone", None) is not None:
            if spec.success_milestone.check(last_obs, state):
                state["succeeded"] = True
                state["terminated"] = True

        # Belief-state distillation (Track B v0.3): two trigger conditions.
        # 1) Level transition: summarize the prior level into the journal.
        # 2) Periodic (every BELIEF_STATE_INTERVAL turns): summarize the
        #    recent journal into a compact "belief_state" note so history-
        #    compaction can drop turns >100 without losing the LM's mental
        #    model. Survey rec #3.
        new_dlvl = state["structured_obs"].status.get("depth", 1)
        if new_dlvl > state["max_dlvl_reached"]:
            _maybe_distill(state, prior_dlvl=state["max_dlvl_reached"])
            # Count the descent here so descent_reward can read a cumulative
            # tally at end-of-rollout. (The rubric only fires score_rollout
            # once, so a per-step compare would lose every transition except
            # the last.)
            state["descent_count"] = state.get("descent_count", 0) + (new_dlvl - state["max_dlvl_reached"])
            state["max_dlvl_reached"] = new_dlvl  # update AFTER computing the level delta

        # Wave-2 Track B: update visited-frontier memory + deadlock flag.
        try:
            _update_frontier_blacklist(state)
        except Exception:
            pass

        state["turn_count"] = state.get("turn_count", 0) + 1
        if self.belief_state_interval > 0 and state["turn_count"] > 0 and state["turn_count"] % self.belief_state_interval == 0:
            _maybe_belief_state_summary(state)

        # Path-failure diagnosis (exp4 fix 3): "No valid path found to reach
        # position (X,Y)" was the single largest waste bucket in BOTH arms of
        # exp3b (~25% of all move_to calls), because it names no reason -- so
        # both agents re-issued the identical coordinates within a few calls.
        # Append WHY: the blocking door/monster on the route, or the nearest
        # reachable tile when no explored route exists at all.
        if (skill_name in ("np_move_to", "np_go_to", "move_to")
                and result.feedback
                and ("No valid path" in result.feedback or "No path found" in result.feedback)):
            try:
                from nethack_harness.navigation.path_explain import explain_path_failure
                extra = explain_path_failure(
                    state.get("raw_obs"), skill_args,
                    published_tools=state.get("_published_tools"))
                if extra:
                    from nethack_harness.tools.skills import SkillResult as _SR
                    result = _SR(actions=result.actions,
                                 feedback=f"{result.feedback} {extra}",
                                 interrupted=result.interrupted)
            except Exception:
                pass  # a diagnosis must never break the turn

        # Move-blocked detection: `move(direction=...)` always reports "Moved
        # S." even when the action bumped a wall. The model can't tell from
        # feedback whether the step succeeded. Compare pre/post player (x, y)
        # from blstats; if a single-step move kept us in place, override the
        # feedback so the model knows to pick a different direction.
        if skill_name == "move" and len(action_indices) == 1 and pre_pos is not None and not terminated and not truncated:
            try:
                from nethack_harness.tools.skills import SkillResult as _SR
                post_blstats = last_obs.blstats if hasattr(last_obs, "blstats") else last_obs.get("blstats")
                if post_blstats is not None:
                    post_pos = (int(post_blstats[0]), int(post_blstats[1]))
                    if post_pos == pre_pos:
                        result = _SR(
                            actions=result.actions,
                            feedback=f"Move blocked at {pre_pos}: wall or obstacle in {skill_args.get('direction', '?')}. Pick a different direction or `search` if you suspect a hidden door.",
                            interrupted=result.interrupted,
                        )
            except (KeyError, IndexError, TypeError, AttributeError):
                pass

        # Attack-outcome detection: replace the generic "Moved W." feedback
        # with hit/miss/kill info pulled from the NLE message buffer. The
        # model doesn't otherwise know whether its swing landed.
        if skill_name == "attack":
            try:
                from nethack_harness.tools.skills import SkillResult as _SR
                msg_bytes = last_obs.message if hasattr(last_obs, "message") else last_obs.get("message")
                if msg_bytes is not None:
                    msg = bytes(msg_bytes).split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
                    if msg:
                        outcome = None
                        msg_l = msg.lower()
                        if "you kill" in msg_l or "is killed" in msg_l:
                            outcome = f"Killed: {msg}"
                        elif "you hit" in msg_l or "you destroy" in msg_l:
                            outcome = f"Hit: {msg}"
                        elif "you miss" in msg_l:
                            outcome = f"Missed: {msg}"
                        elif "nothing here" in msg_l or "no monster" in msg_l:
                            outcome = f"No target: {msg}"
                        if outcome:
                            result = _SR(actions=result.actions, feedback=outcome, interrupted=result.interrupted)
            except (KeyError, IndexError, TypeError, AttributeError):
                pass

        # ---- Wave-2: position-stuck deadlock breaker -------------------
        # Diagnosis (experiment_log.md Wave-2): exploration skills can wedge.
        # A scripted `find_and_descend` loop on seed 22 froze at pos (63,6)
        # with the in-game clock stuck at T:51 for 90+ turns: A* kept choosing
        # a 1-step "far frontier" whose first step was a no-op (an adjacent
        # closed/locked door `+` that walking-into does nothing). Real LM
        # rollouts hit the same wedge and oscillate until they starve.
        #
        # Fix: when a movement/exploration skill leaves the player at the SAME
        # (x,y) AND the in-game turn counter did not advance, count it. After
        # 2 such no-progress calls, auto-KICK an adjacent closed door `+` to
        # break the wedge (or, if none, surface a strong search-or-redirect
        # hint). This runs for every variant — it's a pure correctness fix.
        _EXPLORE_SKILLS = {"autoexplore", "find_and_descend", "move_to", "move"}
        stuck_hint: Optional[str] = None
        if skill_name in _EXPLORE_SKILLS and pre_pos is not None and not (terminated or truncated):
            try:
                post_blstats = last_obs.blstats if hasattr(last_obs, "blstats") else last_obs.get("blstats")
                post_pos = (int(post_blstats[0]), int(post_blstats[1])) if post_blstats is not None else None
            except (KeyError, IndexError, TypeError, AttributeError):
                post_pos = None
            # No-progress = the in-game clock did not advance. This catches both
            # "bumped a wall" (pos same) AND the false-frontier oscillation
            # (pos toggles between two adjacent corridor tiles whose only
            # "unexplored" neighbor is actually solid rock, so the game clock
            # never moves and no new tiles are revealed). Clock-frozen is the
            # reliable signal: a real exploration step always advances T.
            cur_time = state["structured_obs"].status.get("time")
            prev_time = state.get("_last_stuck_time")
            clock_frozen = (prev_time is not None and cur_time == prev_time)
            # Also track revealed-tile count: if the scout set didn't grow,
            # the call revealed nothing new.
            no_new_tiles = state.get("scout_delta", 0) == 0
            no_progress = clock_frozen and no_new_tiles
            if no_progress:
                state["_stuck_count"] = state.get("_stuck_count", 0) + 1
            else:
                state["_stuck_count"] = 0
            state["_last_stuck_time"] = cur_time
            if state.get("_stuck_count", 0) >= 2:
                # Find an adjacent closed door `+` to kick.
                adj = getattr(state["structured_obs"], "adjacent", None) or {}
                door_dir = None
                for d, tile in adj.items():
                    if tile and (tile == "+" or tile.startswith("+")):
                        door_dir = d
                        break
                if door_dir is not None:
                    # Auto-kick: KICK command then the direction key. Step it.
                    try:
                        from nethack_harness.tools.skills import _DIRECTION_KEYS  # type: ignore
                    except Exception:
                        _DIRECTION_KEYS = None
                    _DKEY = {"N": ord("k"), "S": ord("j"), "E": ord("l"), "W": ord("h"),
                             "NE": ord("u"), "NW": ord("y"), "SE": ord("n"), "SW": ord("b")}
                    from nethack_core import actions as _nh
                    kick_cmd = int(_nh.Command.KICK)
                    kick_seq = _to_action_indices(env, [kick_cmd]) + _to_action_indices(env, [_DKEY.get(door_dir, ord("."))])
                    for ka in kick_seq:
                        last_obs, _kr, kt, ktr, _ki = env.step(ka)
                        terminated = terminated or kt
                        truncated = truncated or ktr
                        if terminated or truncated:
                            break
                    state["raw_obs"] = last_obs
                    state["structured_obs"] = shape_observation(last_obs, state["character"])
                    state["_stuck_count"] = 0
                    stuck_hint = (
                        f"[deadlock-breaker: stuck at {pre_pos}; auto-kicked the "
                        f"closed door to {door_dir}. If it didn't open, kick again "
                        f"or pick a different exploration target.]"
                    )
                else:
                    # No door to kick and the level's visible frontiers are
                    # all false (adjacent only to solid rock) — the genuine
                    # exit is a HIDDEN passage. Auto-search in place to reveal
                    # it, escalating count the longer we're wedged. NLE caps a
                    # search run, so this is safe. This is what unwedges the
                    # seed-22 corridor pocket (all frontiers border rock).
                    from nethack_core import actions as _nh
                    search_idx = _to_action_indices(env, [int(_nh.Command.SEARCH)])
                    n_search = min(10 * state.get("_stuck_count", 2), 30)
                    if search_idx:
                        for _ in range(n_search):
                            last_obs, _sr, stt, str_, _si = env.step(search_idx[0])
                            terminated = terminated or stt
                            truncated = truncated or str_
                            if terminated or truncated:
                                break
                        state["raw_obs"] = last_obs
                        state["structured_obs"] = shape_observation(last_obs, state["character"])
                    state["_stuck_count"] = 0
                    stuck_hint = (
                        f"[deadlock-breaker: exploration wedged at {pre_pos} (all "
                        f"reachable frontiers border solid rock). Auto-searched "
                        f"{n_search}x for a hidden passage. If still no new exit, "
                        f"`move_to` a DIFFERENT visible tile or `search` more — the "
                        f"way down is likely behind a hidden wall.]"
                    )
            # If the auto-kick/search advanced the game state to a terminal
            # outcome (e.g. starved mid-search), re-run detection so death is
            # attributed and the rollout ends cleanly.
            if terminated or truncated:
                state["terminated"] = True
                _detect_terminal_outcome(last_obs, state)
        # ----------------------------------------------------------------

        # Autoexplore-loop detection: when autoexplore returns "short" feedback
        # repeatedly (frontier shrunk to 1-2 step paths near level edges), the
        # model often spam-calls it ignoring the tail hint. After N consecutive
        # short trips, emit a stronger interrupt hint at the TOP of the obs.
        # Trace 9071d001 showed 66 autoexplore calls with 7-long runs ignoring
        # in-skill tail tips.
        loop_hint: Optional[str] = None
        if skill_name == "autoexplore" and result.feedback and "short" in result.feedback:
            state["consecutive_short_autoexplore"] = state.get("consecutive_short_autoexplore", 0) + 1
            n = state["consecutive_short_autoexplore"]
            if n >= 3:
                loop_hint = (
                    f"[autoexplore-loop: {n} short trips in a row. "
                    "Switch tactic: `search` adjacent walls, or `move_to(x,y)` "
                    "a specific feature, or pick a direction with `move`.]"
                )
        else:
            state["consecutive_short_autoexplore"] = 0

        # Re-scrub the intro/copyright banner before rendering. Each env.step
        # re-carries the banner in the still-unexplored top tty rows, and in the
        # right-offset-map standard tiers it is never repainted by gameplay — so
        # the one-time scrub in setup_state is not enough: it must be reapplied
        # to the freshly-stepped obs every turn or it bleeds into the rendered
        # MAP (which reads the raw tty). Gated to standard tiers; defensive
        # (never raises). Mutates raw_obs.tty_chars, then re-shape so the map
        # view reflects the cleaned tty.
        if state.get("_standard_tier", True):
            _scrub_intro_banner(state["raw_obs"])
            state["structured_obs"] = shape_observation(state["raw_obs"], state["character"])

        # Build the per-turn user message from the spec's turn template.
        obs_text = self._render_obs_text(state)
        prefix_parts = []
        # Per-turn hooks declared by the spec (P self-refinement directive; CH
        # refiner + sub-agent triggers). Each mutates prefix_parts/state in
        # place. Variants with no hooks (the default) skip this entirely.
        for hook in self.spec.turn_hooks:
            hook(self, state, prefix_parts)
        if stuck_hint:
            prefix_parts.append(stuck_hint)
        if loop_hint:
            prefix_parts.append(loop_hint)
        if halt_reason:
            prefix_parts.append(f"[autohalt: {halt_reason}]")
        # One-shot resume banner (setup_state's resume_from hook). Rendered
        # ahead of everything else on the first post-resume turn only.
        _rnote = state.pop("_resume_notice", None)
        if _rnote:
            prefix_parts.insert(0, f"[{_rnote}]")
        # Post-skill prompt/menu cleanup, deliberately NOT under the autohalt
        # label -- a closed prompt is not an interrupted plan.
        _dn = state.pop("_dismiss_notice", None)
        if _dn:
            prefix_parts.append(f"[{_dn}]")
        dropped = state.get("_dropped_extra_tool_calls", 0)
        if dropped:
            prefix_parts.append(
                f"[multi-tool warning: only the first of {dropped+1} tool "
                "calls was applied. NetHack is turn-based; emit ONE tool "
                "call per turn.]"
            )
            state["_dropped_extra_tool_calls"] = 0
        if result.feedback:
            prefix_parts.append(f"[{result.feedback}]")
        content = compose_user_content(obs_text, prefix_parts)
        # Trace breadcrumbs for the caller (`_apply_tool_call`), which owns the
        # single per-turn NDJSON write. `action_indices` is kept for the legacy
        # field only; the authoritative command stream comes from the caller's
        # TurnRecorder, which also saw the steps this loop did not take (menu
        # drain, deadlock-breaker, and every step inside a pre_executed skill).
        tt["action_indices"] = list(action_indices or [])
        tt["reward"] = float(total_reward)
        tt["feedback"] = result.feedback or ""
        return content

    async def is_completed(self, state: vf.State) -> bool:
        # Game-over (death/ascension/NLE truncation) ends the rollout.
        if bool(state.get("terminated")):
            self._flush_final_trace_entry(state, "terminated")
            return True
        # Also honor the verifiers per-rollout LM-turn cap (`max_turns`). Without
        # this, the override silently bypassed the base class's
        # `max_turns_reached`, so `max_turns` was a no-op and rollout length was
        # governed solely by the tier's `max_episode_steps` (in-game NLE steps).
        # OR-ing it in makes `max_turns` an effective per-rollout LM-call cap.
        if getattr(self, "max_turns", -1) and self.max_turns > 0:
            if await self.max_turns_reached(state):
                state["is_truncated"] = True
                self._flush_final_trace_entry(state, "max_turns_reached")
                return True
        return False

    def _flush_final_trace_entry(self, state: vf.State, stop_reason: str) -> None:
        """Write the trace record for the LM turn the rollout never applied.

        THE OFF-BY-ONE. `verifiers`' rollout loop is
        `while not is_completed(state): get_prompt_messages() -> env_response()
        -> get_model_response() -> trajectory.append()`. So the model's LAST
        assistant message is generated and counted (`num_turns`,
        `total_tool_calls` both read the trajectory) and only THEN does
        `is_completed` fire -- `env_response` never runs for it, so the writer
        never saw it. Measured: `num_turns=100 / total_tool_calls=100` against
        99 NDJSON records, on every rollout, always short by exactly one.

        We do NOT execute that final call to make the numbers line up: stepping
        the engine past the cap would change the game the metrics describe.
        Instead we record it as what it is -- `applied: false`, tool result
        `status: "not_applied"`, action record `recorded: false` with a reason --
        against the observation it was made from (the last one we rendered).
        `len(records) == num_turns == total_tool_calls` after this, and no
        record claims an action that never ran.
        """
        if state.get("_trace_final_written"):
            return
        state["_trace_final_written"] = True
        if not getattr(self, "trace_dir", None):
            return
        try:
            traj = state.get("trajectory") or []
            if not traj:
                return
            completion = traj[-1].get("completion") if isinstance(traj[-1], dict) \
                else getattr(traj[-1], "completion", None)
            msgs = completion if isinstance(completion, list) else []
            assistant, calls = None, []
            for m in msgs:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role != "assistant":
                    continue
                assistant = m
                calls = (m.get("tool_calls") if isinstance(m, dict)
                         else getattr(m, "tool_calls", None)) or []
            if assistant is None:
                return
            name, args = None, None
            if calls:
                # Same two shapes `_write_trace_entry` normalizes: OpenAI-style
                # `{"function": {...}}` and verifiers' flat `ToolCall`.
                first = calls[0]
                fn = (first.get("function") if isinstance(first, dict)
                      else getattr(first, "function", None)) or first
                name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None))
                args = (fn.get("arguments") if isinstance(fn, dict)
                        else getattr(fn, "arguments", None))
                # `tool_results[i]["arguments"]` is a dict on every other path
                # (the harness has already parsed them); keep it one type.
                if isinstance(args, str):
                    import json as _json
                    try:
                        parsed = _json.loads(args)
                        args = parsed if isinstance(parsed, dict) else {"_raw": args}
                    except (ValueError, TypeError):
                        args = {"_raw": args}
            result = build_tool_result(
                name=name, arguments=args,
                feedback=f"rollout ended ({stop_reason}) before this call was applied",
                status="not_applied" if calls else "no_tool_call",
                clock_before=_game_clock(state), clock_after=_game_clock(state),
                engine_steps=0, reward=0.0, game_message="",
            )
            _write_trace_entry(
                self, state, assistant, calls, [], 0.0,
                state.get("_last_rendered_user_message", ""),
                actions=TS.empty_action_record(
                    f"tool call was never applied: rollout ended ({stop_reason})"),
                tool_results=[result], all_messages=[], applied=False,
                lm_turn=int(state.get("_trace_lm_turn", 0)) + 1,
                turn=int(state.get("turn_count", 0)) + 1,
            )
        except Exception:
            pass

    async def get_prompt_messages(self, state: vf.State):
        """Override the verifiers default to compact older user-message content
        (i.e. our prior turn observations) before sending to the LM. This is
        the biggest token-bill win: chat history grew linearly in turns,
        re-sending the full tty grid (~25k tok/turn) every single time. After
        compaction:
          * last K=5 turns: full fidelity
          * turns K..100: replaced with a one-line "[turn N: <summary>]"
          * turns >100: dropped entirely
        Mirrors SWE-agent's "elide all but last 5" and Glyphbox's 10/100
        thresholds (see docs/PROMPTING_SURVEY.md).
        """
        messages = await super().get_prompt_messages(state)
        # Compaction is universal (the biggest token-bill win).
        messages = _compact_chat_history(messages, keep_full=self.history_keep_full, drop_after=self.history_drop_after)
        # Summarize-and-reset is an orthogonal knob (variant R sets it): drop
        # everything prior to the most-recent belief checkpoint.
        if self.summarize_and_reset:
            messages = _drop_before_last_belief(messages, state)
        # Spec-declared history transforms (CH appends the Refiner addendum +
        # macro list to the system message). Empty for most variants.
        for transform in self.spec.history_transforms:
            messages = transform(self, messages, state)
        # Strict OpenAI-compatible endpoints (e.g. Prime Inference / Qwen3.5)
        # reject a request whose history contains an assistant message with
        # content=None and no tool_calls -- which is exactly what a "thinking"
        # model emits when a turn is pure reasoning_content (HTTP 422:
        # "content is required unless an assistant message includes tool_calls").
        # Coerce such messages to a non-null content so the rollout can proceed.
        messages = _sanitize_assistant_content(messages)
        return messages

    def update_tool_args(self, tool_args: dict, messages, state) -> dict:
        """
        Required by StatefulToolEnv. We dispatch tool calls manually inside
        `env_response` (because each skill has a custom signature involving
        the env handle + structured observation), so this hook is a no-op:
        we never let the base class's `call_tool()` route get used.
        """
        return tool_args



# ---------- frontier blacklist (kept here: tests monkeypatch these on the nethack module) ----------

def _game_clock(state) -> Optional[int]:
    """The in-game turn counter (`status["time"]`), or None if unreadable.

    Sampled either side of a turn, this is the only honest test of "did
    anything actually happen": the vendored NetPlay skills report `completed`
    without inspecting the result, so a `move_to` that stopped dead against a
    wall and a `move_to` that crossed the level are indistinguishable by status
    alone. See `helpers.classify_tool_result`.
    """
    try:
        so = state.get("structured_obs")
        if so is None:
            return None
        val = (so.status or {}).get("time")
        return None if val is None else int(val)
    except Exception:
        return None


def _obs_tty_has_more(obs) -> bool:
    """True if a --More-- prompt is visible on the top tty rows.

    Works whether obs is a dict (``obs["tty_chars"]``) or a CoreObservation
    (attribute access ``obs.tty_chars``). Mirrors the tty-row detector used in
    nethack_harness/tools/skills.py. Never raises — a detector failure must
    never break a rollout.
    """
    try:
        tty = obs.get("tty_chars") if isinstance(obs, dict) else getattr(obs, "tty_chars", None)
        if tty is None:
            return False
        return any(b"--More--" in bytes(int(c) for c in row) for row in tty[:3])
    except Exception:
        return False


# Substrings that identify the NetHack startup/intro banner. NLE paints this
# copyright/version art over the top tty rows on reset, and — in tiers whose
# map is offset to the right so the player never walks over those cells — the
# banner is never repainted by gameplay. It then bleeds into the rendered MAP
# (render_map_view reads the raw tty), garbling the agent's first observations.
_INTRO_BANNER_MARKERS = (b"Copyright", b"Stichting", b"Version 3.6", b"See license")


def _scrub_intro_banner(obs) -> None:
    """Strip the stale intro/copyright banner from a CoreObservation's tty.

    The authoritative dungeon map lives in ``obs.chars`` (one row above the
    matching tty row, same columns). Wherever a top tty cell holds banner art
    that the clean ``chars`` plane reports as blank, we overwrite it with a
    space. Real map glyphs (which match ``chars`` exactly) are left untouched.
    Mutates ``obs.tty_chars`` in place. Never raises — a scrub failure must
    never break a rollout.
    """
    try:
        tty = getattr(obs, "tty_chars", None)
        chars = getattr(obs, "chars", None)
        if tty is None or chars is None:
            return
        # Only act when a banner signature is actually present on the top rows,
        # so we never disturb a normal observation.
        top = bytes(int(c) for r in range(min(6, tty.shape[0])) for c in tty[r])
        if not any(m in top for m in _INTRO_BANNER_MARKERS):
            return
        space = ord(" ")
        n_cols = min(tty.shape[1], chars.shape[1])
        # tty row r (1..21) corresponds to chars row r-1, same columns.
        for r in range(1, min(tty.shape[0], chars.shape[0] + 1)):
            for x in range(n_cols):
                if tty[r, x] != chars[r - 1, x] and chars[r - 1, x] == space:
                    tty[r, x] = space
    except Exception:
        pass


def _iterate_visible_tiles(obs):
    """Yield ((x, y), char) for currently-visible map tiles."""
    chars = obs.chars  # (21, 79)
    for y in range(chars.shape[0]):
        for x in range(chars.shape[1]):
            yield (x, y), bytes([int(chars[y, x])])


def _record_scout_and_visited(state: dict, obs) -> None:
    """Fold one observation's visible tiles + hero position into `state`.

    Factored out of the env_response step loop so the SAME bookkeeping can be
    replayed over a closed-loop skill's `pre_visible_obs` (see
    SkillResult.pre_visible_obs / run_netplay_skill), which never goes through
    that loop because `pre_executed=True` skills report `action_indices=[]`.
    """
    # Scout reward: count newly-revealed dungeon tiles.
    for (x, y), ch in _iterate_visible_tiles(obs):
        if ch not in (b" ", b"\x00"):
            state["scout_tiles_seen"].add((state["max_dlvl_reached"], x, y))
    # Sub-experiment 1b: record the hero's current tile into the per-level
    # visited set (drives visited_grid). Keyed by the hero's ACTUAL depth
    # (blstats[12]) so descent turns file under the level the template
    # will read (max_dlvl_reached lags until later in env_response).
    try:
        _vb = obs.blstats
        state["_visited_tiles"].setdefault(int(_vb[12]), set()).add(
            (int(_vb[0]), int(_vb[1]))
        )
    except (AttributeError, IndexError, TypeError, KeyError):
        pass


# ----- Wave-2 Track B: visited-frontier memory + deadlock-breaker -----
#
# Knobs (kept module-level so tests can monkeypatch):
FRONTIER_STUCK_TURNS = 3      # adjacency turns w/o new tiles before blacklist
FRONTIER_APPROACH_RADIUS = 1  # Chebyshev distance counting as "approached"
NEEDS_HIDDEN_TURNS = 5        # zero-scout streak that triggers needs-hidden


def _update_frontier_blacklist(state: dict) -> None:
    """Per-turn maintenance of the visited-frontier memory.

    Rules:
      * Reset blacklist + counters when max_dlvl_reached changes (per-level
        memory is what we want — a "stuck" frontier on L1 isn't stuck on L2).
      * Walk all current-level frontiers. For each frontier within
        FRONTIER_APPROACH_RADIUS of the player, increment its consecutive
        no-progress count iff `scout_delta == 0` this turn. Any turn that
        revealed new tiles resets all counts (we're making progress somehow).
      * When a frontier's count hits FRONTIER_STUCK_TURNS, add it to the
        per-level blacklist.
      * Set `_needs_hidden_passage` when every reachable frontier is
        blacklisted AND we've had `_zero_scout_streak >= NEEDS_HIDDEN_TURNS`.
        Cleared when scout_delta > 0 (the search/kick worked).

    Track C reads `state["_needs_hidden_passage"]`; we only set it here.
    """
    from nethack_harness.navigation.pathfinding import find_frontiers
    raw = state.get("raw_obs")
    if raw is None or not hasattr(raw, "chars") or not hasattr(raw, "blstats"):
        return
    chars = raw.chars
    blstats = raw.blstats
    px, py = int(blstats[0]), int(blstats[1])
    cur_dlvl = int(state.get("max_dlvl_reached", 1))
    prev_dlvl = state.get("_frontier_prev_dlvl", cur_dlvl)
    if cur_dlvl != prev_dlvl:
        # Level changed — clear per-level state and exit.
        state["_frontier_approach_count"] = {}
        state["_frontier_blacklist"].pop(prev_dlvl, None)
        state["_frontier_prev_dlvl"] = cur_dlvl
        state["_needs_hidden_passage"] = False
        state["_zero_scout_streak"] = 0
        return
    scout_delta = int(state.get("scout_delta", 0) or 0)
    if scout_delta > 0:
        # Real progress — wipe counters AND clear hidden-passage flag.
        state["_frontier_approach_count"] = {}
        state["_needs_hidden_passage"] = False
        state["_zero_scout_streak"] = 0
        return
    state["_zero_scout_streak"] = int(state.get("_zero_scout_streak", 0)) + 1
    blacklist_for_lvl = state["_frontier_blacklist"].setdefault(cur_dlvl, set())
    # Use legacy (loose) predicate for blacklist accounting so we don't miss
    # nominal frontiers — strict predicate culls them at pick time anyway.
    frontiers = find_frontiers(chars, blacklist=None, strict=False)
    approach = state["_frontier_approach_count"]
    for fx, fy in frontiers:
        if (fx, fy) in blacklist_for_lvl:
            continue
        cheb = max(abs(fx - px), abs(fy - py))
        if cheb <= FRONTIER_APPROACH_RADIUS:
            key = (cur_dlvl, fx, fy)
            approach[key] = approach.get(key, 0) + 1
            if approach[key] >= FRONTIER_STUCK_TURNS:
                blacklist_for_lvl.add((fx, fy))
    # Recompute reachable-frontier set (strict + blacklisted).
    open_frontiers = find_frontiers(chars, blacklist=blacklist_for_lvl, strict=True)
    if not open_frontiers and state["_zero_scout_streak"] >= NEEDS_HIDDEN_TURNS:
        state["_needs_hidden_passage"] = True
    # Expose the current-level blacklist on the env so the in-skill autoexplore
    # picker can consume it without needing the full verifiers state dict (which
    # it doesn't have access to).
    try:
        env_obj = state.get("env")
        if env_obj is not None:
            env_obj.frontier_blacklist_current = set(blacklist_for_lvl)
    except Exception:
        pass




def _build_task_dataset(n_examples: int, seed_base: int, explicit_seeds: Optional[list] = None, system_prompt: Optional[str] = None) -> Dataset:
    """Each row is one starting condition of the full ascension game.

    If `explicit_seeds` is provided (list of ints), each row uses one of
    those NLE seeds (cycling through), and n_examples is overridden by the
    list length. Use this to pin known-easy seeds for evaluation.

    `system_prompt` defaults to this module's SYSTEM_PROMPT (honoring any
    NETHACK_HARNESS overlay); load_environment passes the spec's prompt.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    rng = random.Random(seed_base)
    spec = FULL_GAME_SPEC
    rows = []
    if explicit_seeds is not None:
        n_examples = len(explicit_seeds)
    for i in range(n_examples):
        seed_val = (int(explicit_seeds[i]) if explicit_seeds is not None
                    else rng.randint(0, 2**31 - 1))
        rows.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {spec.description}\nSuccess: {spec.success_criterion}\n\nBegin."},
            ],
            # NB: no `task` column. `task` is a RESERVED rollout-input field in
            # verifiers (>=0.1.14): `flatten_task_input` (verifiers/types.py)
            # treats `input["task"]` — or `input["info"]["task"]` — as THE
            # canonical rollout payload and REPLACES the whole input with it, so
            # a row carrying `task={"tier":..,"seed":..}` loses its `prompt` and
            # `init_state` then raises KeyError('prompt'). The per-rollout seed
            # therefore rides in `info`, which `setup_state` already reads as its
            # fallback (`task.get("seed", info.get("seed", ...))`). Callers that
            # build a state by hand with an explicit `task` dict still work.
            "info": {
                "tier": spec.name,
                "seed": seed_val,
                "spec_description": spec.description,
            },
        })
    return Dataset.from_list(rows)



def load_environment(
    n_examples: int = 256,
    seed: int = 0,
    max_turns: int = 200,
    interface: str = "skill",
    # Which task spec to run (GameSpec.name in GAME_SPECS). "full_nle" (default)
    # is the standard ascension game; "six_floor_primitives" is the honest-
    # navigation six-floor curriculum (CurriculumPrimitivesEnv).
    task_spec: str = "full_nle",
    # --- Game-setup overrides: the difficulty / generation knobs from our
    # engine interface. All default to None = vanilla NetHack. Pass these to
    # customize the game instead of the standard ascension run; e.g.
    #   tune={"vision_radius": 5, "mob_spawn": 2, "room_density": 0.3}  # difficulty/generation
    #   modify={"hp": 200, "max_hp": 200, "gold": 1000}                 # starting stat pokes
    #   level_blob="path/to/level.blob"                                  # custom starting level
    # `tune` keys come from EngineEnv.tune.catalog(); `modify` keys are the
    # whitelisted state setters (hp/max_hp/gold/xp_level/hunger).
    tune: Optional[dict] = None,
    modify: Optional[dict] = None,
    level_blob: Optional[str] = None,
    # Pin the character for standard tiers, e.g. "Val-hum-neu-fem". None =
    # engine default (random roll). Ignored by curriculum tiers (own character).
    character: Optional[str] = None,
    sub_lm=None,
    # Sub-experiment 1c (memory ablation): build a real Prime-backed Sub-LM
    # from a model id so belief-state distillation calls the model instead of
    # the deterministic OfflineSubLM status-snapshot stub. Settable via the
    # vf-eval `-a` string args (sub_lm is a Python object and cannot be). When
    # set AND `sub_lm` is None, a PrimeSubLM(sub_lm_model) is constructed and
    # passed as the env's sub_lm. None = unchanged (Offline/status-snapshot).
    sub_lm_model: Optional[str] = None,
    # When False, skip pre-pinning the tier description as the journal
    # objective at setup → enables the no-memory arm (empty journal). See the
    # env constructor for the full rationale.
    pin_objective_on_setup: bool = True,
    # Task 18: A/B the BALROG-minimal prompt (default) against the pre-Task-18
    # strategy-heavy one via one config flag instead of a git revert.
    verbose_prompt: bool = False,
    # Task 18 Step 2: True for the CLI-agent arms (nethack_v1 sets this
    # unconditionally — see NetHackVerifiersEnv's self_dispatch docstring).
    # Gates the per-turn JOURNAL block and HINT ladder off for them; the
    # control arm's default (False) keeps both, unchanged from pre-Task-18.
    self_dispatch: bool = False,
    subgoal_proposer=None,
    compact_obs: bool = False,
    history_keep_full: int = 5,
    history_drop_after: int = 100,
    belief_state_interval: int = 25,
    journal_render_max_chars: int = 2000,
    variant: str = "B0",
    # Sub-experiment 1b (JSON cell-content ablation): per-cell SPATIAL/
    # EXPLORATION attributes to enrich the JSON map with, drawn from
    # {"seen","visited","reach"}. Pass as a JSON list (["seen","visited"]) or a
    # "seen+visited"-style string via the vf-eval `-a` args. Empty/None = base
    # JSON, unchanged. Only affects variant="JSON".
    cell_schema=None,
    refine_interval: int = 20,
    summarize_and_reset: bool = False,
    trace_dir: Optional[str] = None,
    continual: bool = False,
    continual_lives: int = 5,
    refiner: Any = None,
    refiner_model: Optional[str] = None,
    bootstrap_dir: Optional[str] = None,
    refine: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """
    Entrypoint used by `vf-eval` and prime-rl.

    Args:
        n_examples: dataset size for training rollouts / evaluation.
        seed: RNG seed for which per-episode seeds get sampled.
        max_turns: per-rollout turn cap (LM turns, not in-game turns).
        interface: "skill" (default — one tool per skill, OpenAI function-calling)
            or "code" (a single `code` tool that runs sandboxed Python against
            an `nh` namespace; the Track B / RLM-research path).
        task_spec: which task to run, keyed by GameSpec.name in GAME_SPECS.
            "full_nle" (default) is the standard ascension game;
            "six_floor_primitives" runs the honest-navigation six-floor
            curriculum (CurriculumPrimitivesEnv). The latter needs the engine's
            invocation hooks (engine PR #37 / feat/4b-six-floor-hooks).
        compact_obs: enable per-turn observation compaction (strip blank tty rows,
            glyph-run encoding, inventory diff). Default True; set False for raw v0.0.15
            rendering (replay/debugging).
        history_keep_full: number of most-recent turns kept at full fidelity in the
            LM prompt (older turns get a one-line summary or are dropped).
        history_drop_after: turns older than this distance are dropped behind a
            single elision marker.
        belief_state_interval: every N turns, SubLM.summarize is invoked and the
            result added to the journal as belief_state:tN. Set to 0 to disable.
        journal_render_max_chars: soft cap on per-turn journal block size; older
            non-belief-state notes get elided when over the cap.
        variant: obs/skill-structure variant for wave-1 experiments.
            "B1" (default) = current shipping behavior, no override.
            "P" = Continual Harness adaptation (arXiv:2605.09998): every
            `refine_interval` turns, inject a self-refinement directive
            asking the agent to revise its pinned objective and/or record
            a lesson note. Journal ops short-circuit the NLE step, so
            refinement is free game-turn-wise.
        refine_interval: cadence for variant=P self-refinement turns
            (default 20). Set to 0 to disable even when variant="P".
        summarize_and_reset: variant=R toggle. When True, get_prompt_messages
            drops every chat turn older than the most recent belief_state
            checkpoint. Pair with belief_state_interval > 0.
        trace_dir: if set, env_response writes per-turn NDJSON capturing
            raw_grid, structured_obs, rendered_user_message, assistant_message,
            tool_calls, action, reward, dlvl, hp. One file per rollout under
            <trace_dir>/<run_id>.ndjson. Off by default.
        continual: when True, the env auto-resets the underlying NLE on death
            and continues the same chat session, preserving journal + belief
            state. Implements the continual-harness mode (separate from
            variant=P's mid-rollout refinement).
        continual_lives: cap on auto-resets within a single rollout
            (default 5). Ignored unless continual=True.
        refine: when True, attach the full Continual-Harness teacher Refiner
            machinery (periodic teacher-LLM edits to the agent's prompt /
            sub-agents / skill-macros / journal, plus the run_macro tool) onto
            whatever `variant` (obs format) is selected — decoupling the refiner
            from the canonical-ASCII CH variant. e.g. variant="JSON",
            refine=True gives JSON observations PLUS the teacher refiner.
            variant="CH" always implies refine=True.
    """
    explicit_seeds = kwargs.pop("explicit_seeds", None)
    # The CLI arms' ENV_ARGS override path flattens every leaf to a dotted
    # scalar, so a list like [3] arrives here as the STRING "[3]" (measured:
    # int('[') ValueError in _build_task_dataset). Accept the JSON form.
    if isinstance(explicit_seeds, str):
        import json as _json
        explicit_seeds = _json.loads(explicit_seeds)
    # NETHACK_HARNESS overlay: mutates SYSTEM_PROMPT (consumed by _build_task_dataset
    # below) plus returns a HarnessConfig used to filter tools / re-weight rewards.
    # No-op when the env var is unset → bit-identical default behavior.
    import sys as _sys
    _overlay_cfg = _harness_overlay.apply_overlay(_sys.modules[__name__])
    # Resolve the prompt recipe AFTER the overlay so the spec carries the
    # (possibly-overlaid) system prompt. SYSTEM_PROMPT here is this module's
    # global, which apply_overlay just mutated in place. verbose_prompt swaps
    # in the pre-Task-18 SYSTEM_PROMPT_VERBOSE instead (see load_environment's
    # docstring for the A/B).
    _base_system_prompt = SYSTEM_PROMPT_VERBOSE if verbose_prompt else SYSTEM_PROMPT
    spec = resolve_spec(variant, _base_system_prompt)
    # Decouple the teacher refiner from the obs format: when refine=True on a
    # non-CH variant, attach the CH refiner bundle (hooks + system inject +
    # run_macro tool) onto the resolved spec so the tool gets exposed below and
    # the env's spec carries the refiner hooks. (CH already carries it.)
    if bool(refine) and variant != "CH":
        spec = attach_refiner(spec)
    _reward_funcs = _harness_overlay.apply_reward_weights(
        [scout_reward, descent_reward, success_reward, ascension_reward], _overlay_cfg,
    )
    # vf.Rubric scores from its `weights=` list (default [1.0]*n), not fn.weight,
    # so the overlay's reward re-weighting only takes effect if we pass the
    # resolved weights here. Returns None when no overlay/rewards → weights=None
    # → the shipped default, keeping baseline behavior bit-identical.
    _reward_weights = _harness_overlay.resolve_reward_weights(_reward_funcs, _overlay_cfg)
    rubric = vf.Rubric(funcs=_reward_funcs, weights=_reward_weights)

    _describe_args = kwargs.get("describe_args", False)
    if isinstance(_describe_args, str):
        _describe_args = _describe_args.strip().lower() not in ("false","0","no","off","")
    if interface == "skill":
        tool_callables = _build_skill_adapter_callables(
            skill_set=spec.tools.skill_set or kwargs.pop("skill_set", "full"),
            describe_args=bool(_describe_args),
        )
        # Spec-declared extra tools (e.g. CH's run_macro adapter).
        for make_tool in spec.tools.extra_tools:
            tool_callables.append(make_tool())
        tool_callables = _harness_overlay.filter_tool_callables(tool_callables, _overlay_cfg)
    elif interface == "code":
        tool_callables = [_code_tool_adapter()]
    else:
        raise ValueError(f"Unknown interface={interface!r}; expected 'skill' or 'code'.")
    # The exact set of tool names offered to the model — used to gate
    # hallucinated tool calls in env_response (see allowed_skill_names).
    _allowed_skill_names = {getattr(t, "__name__", "") for t in tool_callables} - {""}

    # Gate the system prompt on the tools we just resolved, so the advertised
    # surface cannot exceed the published one. Under `skill_set="netplay"` no
    # `move` adapter exists, yet the hand-written prompt told the agent to call
    # it — exp1 measured 232 rejected `move` attempts. The prompt is now
    # assembled from this set (see rendering.render_system_prompt), the way
    # NetPlay and BALROG both build theirs. Overlays that REPLACE the prompt
    # wholesale are left alone: they are the author's own text, not ours.
    from nethack_harness.prompt import rendering as _rendering
    if interface == "skill" and spec.system_prompt == _base_system_prompt:
        import dataclasses as _dc
        # BALROG's raw-command surface gets BALROG's OWN instruction prompt.
        # `_render_system_prompt` builds its cheat-sheet from `_SKILL_BLURBS`,
        # which has no `bal_*` entries, so this arm was silently shipping the
        # macro-skill prompt -- which never says descent needs `down` while
        # standing on the stairs, nor that `travel` takes a `>`/`<` follow-up.
        # The agent called `bal_down` once in 12,446 turns and scored a flat
        # 0.00%. Comparing against BALROG's 3.96 on this surface only means
        # something if the agent is briefed the way theirs is.
        if any(n.startswith("bal_") for n in _allowed_skill_names):
            from nethack_harness.tools.balrog_actions import balrog_instruction_prompt
            spec = _dc.replace(spec, system_prompt=balrog_instruction_prompt())
        else:
            spec = _dc.replace(
                spec,
                system_prompt=_render_system_prompt(_allowed_skill_names, verbose=verbose_prompt),
            )
    # E8b mechanic guidance: append system-prompt blocks only (published tool
    # schemas untouched). Default "" leaves every existing arm byte-identical.
    _hints = str(kwargs.get("mechanic_hints") or "").strip().lower()
    if _hints:
        from nethack_harness.prompt.human_norms import MECHANIC_HINT_BLOCKS
        _blocks = [MECHANIC_HINT_BLOCKS[h.strip()] for h in _hints.split(",")
                   if h.strip() in MECHANIC_HINT_BLOCKS]
        if _blocks:
            spec = _dc.replace(
                spec, system_prompt=spec.system_prompt + "\n\n" + "\n\n".join(_blocks))

    dataset = _build_task_dataset(
        n_examples, seed, explicit_seeds=explicit_seeds,
        system_prompt=spec.system_prompt,
    )

    # Sub-experiment 1c: when a sub_lm_model id is given (and no explicit
    # sub_lm object was passed), build a real Prime-backed Sub-LM so
    # belief-state distillation calls the model. `_maybe_belief_state_summary`
    # uses it iff it is NOT an OfflineSubLM.
    if sub_lm is None and sub_lm_model:
        from nethack_harness.tools.code_mode import PrimeSubLM
        sub_lm = PrimeSubLM(sub_lm_model)

    return NetHackVerifiersEnv(
        dataset=dataset,
        rubric=rubric,
        tools=tool_callables,
        max_turns=max_turns,
        interface=interface,
        task_spec=task_spec,
        sub_lm=sub_lm,
        subgoal_proposer=subgoal_proposer,
        compact_obs=compact_obs,
        history_keep_full=history_keep_full,
        history_drop_after=history_drop_after,
        belief_state_interval=belief_state_interval,
        journal_render_max_chars=journal_render_max_chars,
        variant=variant,
        cell_schema=cell_schema,
        spec=spec,
        refine_interval=refine_interval,
        summarize_and_reset=summarize_and_reset,
        trace_dir=trace_dir,
        continual=continual,
        continual_lives=continual_lives,
        refiner=refiner,
        refiner_model=refiner_model,
        bootstrap_dir=bootstrap_dir,
        refine=refine,
        setup_tune=tune,
        setup_modify=modify,
        setup_level_blob=level_blob,
        setup_character=character,
        allowed_skill_names=_allowed_skill_names,
        no_progress_timeout=int(kwargs.pop('no_progress_timeout', 10_000)),
        pin_objective_on_setup=pin_objective_on_setup,
        self_dispatch=self_dispatch,
        **kwargs,
    )



__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_VERBOSE",
    "GameSpec",
    "FULL_GAME_SPEC",
    "PRIMITIVES_GAME_SPEC",
    "GAME_SPECS",
    "_strip_blank_rows",
    "_glyph_run_encode",
    "_inventory_fingerprint",
    "_run_length_encode_messages",
    "_glyph_to_words",
    "_format_obs_balrog",
    "_format_obs_glyphbox",
    "_format_obs_summarize_reset",
    "_descent_status_block",
    "_E1_BEARINGS",
    "_e1_bearing",
    "_e1_classify_frontier",
    "_e1_frontiers_block",
    "_e1_exploration_block",
    "_e1_spatial_belief_block",
    "_VARIANT_FORMATTERS",
    "_paint_frontiers_on_map",
    "format_observation_as_chat",
    "_continual_reset",
    "_write_trace_entry",
    "TurnRecorder",
    "build_tool_result",
    "_game_clock",
    "_drop_before_last_belief",
    "_refinement_directive",
    "_ch_build_window",
    "_ch_inject_system",
    "_ch_save_bootstrap",
    "_compact_chat_history",
    "_STATUS_SIG_RE",
    "_compacted_status_signature",
    "_dedupe_compacted_runs",
    "_msg_role",
    "_msg_content",
    "_replace_content",
    "_one_line_summary",
    "_check_halt_condition",
    "BELIEF_STATE_INTERVAL",
    "_maybe_belief_state_summary",
    "_maybe_distill",
    "_to_action_indices",
    "scout_reward",
    "descent_reward",
    "success_reward",
    "ascension_reward",
    "_ASCENSION_MARKERS",
    "_DEATH_MARKERS",
    "_decode_tty",
    "_detect_terminal_outcome",
    "_code_tool_adapter",
    "_build_skill_adapter_callables",
    "_make_run_macro_adapter",
    "_make_fixed_direction_adapter",
    "_TYPE_MAP",
    "_make_skill_adapter",
    "_iterate_visible_tiles",
    "FRONTIER_STUCK_TURNS",
    "FRONTIER_APPROACH_RADIUS",
    "NEEDS_HIDDEN_TURNS",
    "_update_frontier_blacklist",
    "NetHackVerifiersEnv",
    "load_environment",
    "_build_task_dataset",
    # Prompt factory.
    "ObsSpec",
    "ToolSpec",
    "PromptSpec",
    "build_prompt",
    "VARIANT_REGISTRY",
    "resolve_spec",
    "attach_refiner",
]

