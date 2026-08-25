"""Harness runtime helpers extracted verbatim from nethack.py (no logic change):
chat-history compaction, continual-harness reset, refiner/CH glue, belief-state
distillation, terminal-outcome detection, reward functions, and the verifiers
code/skill tool adapters.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Optional

import verifiers as vf

from nethack_core.env import NetHackCoreEnv
from nethack_core import trace_schema as TS
from nethack_harness.memory.journal import Journal
from nethack_core.observations import shape as shape_observation
from nethack_harness.tools.skills import registry as skill_registry, list_skills


# ---------------------------------------------------------------------------
# Per-turn engine instrumentation (trace items A + D)
# ---------------------------------------------------------------------------


def _obs_message(obs) -> str:
    """Decode the top-line message out of a raw observation, cheaply.

    `shape()` does this too, but shaping a whole observation costs a map render
    + inventory parse + menu scrape; a 100-step macro would pay that 100 times
    just to read one string. This reads only the `message` plane.
    """
    try:
        msg = obs.get("message") if isinstance(obs, dict) else getattr(obs, "message", None)
        if msg is None:
            return ""
        return bytes(int(c) for c in msg).split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    except Exception:
        return ""


class TurnRecorder:
    """Intercept every engine step for the duration of ONE agent turn.

    WHY A HOOK AND NOT CALL-SITE BOOKKEEPING. Actions reach the engine from at
    least five places in a single turn: `_apply_tool_call`'s own step loop, the
    menu auto-dismiss drain, the deadlock-breaker's kick/search, the post-death
    `rollback` path, and -- for every `netplay_true` skill -- the skill's own
    internal loop inside `run_netplay_skill` (which is exactly why
    `action_indices` was empty on 100% of netplay_true turns: those skills are
    `pre_executed`, so the harness step loop never runs for them). All five
    funnel through `NetHackCoreEnv.step`, so patching that one method for the
    turn captures the complete, ordered command stream with no call site left to
    forget.

    Also harvests the per-step message, which is the other thing a macro loses:
    `shape()` keeps only the message of the FINAL observation, so `You kill the
    little dog!` from step 12 of 100 never reached the agent OR the trace.

    The patch is an instance attribute over the bound method and is removed in
    `__exit__`, so nothing outside the `with` block is affected.
    """

    __slots__ = ("_env", "_orig", "_had_own", "_eng", "_orig_restore",
                 "actions", "messages", "restores", "_installed",
                 "_capture_frames", "frames")

    def __init__(self, env, capture_frames: bool = False) -> None:
        self._env = env
        self._orig = None
        self._had_own = False
        self._eng = None
        self._orig_restore = None
        self._installed = False
        self.actions: list[int] = []
        self.messages: list[str] = []
        #: When capture_frames is True, one 24-row tty snapshot per engine step
        #: -- the missing "screen per move" that shape() collapses to the final
        #: frame only. Off by default: the existing arms stay byte-identical and
        #: pay no size/latency cost.
        self._capture_frames = bool(capture_frames)
        self.frames: list[dict] = []
        #: Snapshot restores this turn. `rollback` rewinds the engine heap
        #: directly (`engine.restore(handle)`) instead of stepping, so a turn
        #: with restores > 0 is NOT replayable from its byte stream and the
        #: record must say so rather than imply otherwise.
        self.restores: int = 0

    def note_message(self, msg) -> None:
        """Append `msg` unless it is empty or repeats the previous one."""
        if not msg:
            return
        m = str(msg).strip()
        if m and (not self.messages or self.messages[-1] != m):
            self.messages.append(m)

    def __enter__(self) -> "TurnRecorder":
        env = self._env
        if env is None:
            return self
        try:
            orig = env.step
            # If something else already patched `step` on the instance (a test
            # double, a nested recorder), put THAT back rather than falling
            # through to the class method.
            self._had_own = "step" in getattr(env, "__dict__", {})

            def _recording_step(action, *a, **kw):
                out = orig(action, *a, **kw)
                try:
                    ai = int(action)
                    self.actions.append(ai)
                    obs0 = out[0] if isinstance(out, tuple) else out
                    self.note_message(_obs_message(obs0))
                    if self._capture_frames:
                        self.frames.append({
                            "b": ai,
                            "k": chr(ai) if 32 <= ai < 127 else "",
                            "g": _tty_rows(obs0),
                            "gid": _glyph_rle(obs0),
                            "m": _obs_message(obs0) or "",
                        })
                except Exception:
                    pass
                return out

            env.step = _recording_step
            self._orig = orig
            self._installed = True
            eng = getattr(env, "_engine", None)
            restore = getattr(eng, "restore", None)
            if restore is not None:
                def _recording_restore(*a, **kw):
                    self.restores += 1
                    return restore(*a, **kw)
                eng.restore = _recording_restore
                self._eng, self._orig_restore = eng, restore
        except Exception:
            # Instrumentation must never break a rollout; the trace will just
            # say recorded=False for this turn.
            self._installed = False
        return self

    def __exit__(self, *exc) -> bool:
        if self._installed:
            try:
                if self._had_own:
                    self._env.step = self._orig
                else:
                    del self._env.step      # restore the bound class method
            except Exception:
                pass
        if self._eng is not None:
            try:
                del self._eng.restore
            except Exception:
                try:
                    self._eng.restore = self._orig_restore
                except Exception:
                    pass
        return False

    def action_record(self) -> dict:
        if not self._installed:
            return TS.empty_action_record(
                "engine step hook could not be installed for this turn")
        if self.restores:
            return TS.action_record(
                self.actions, replayable=False,
                reason=(f"engine state was rewound {self.restores}x from a heap "
                        "snapshot (`rollback`); the byte stream alone does not "
                        "reproduce this turn"))
        return TS.action_record(self.actions)

def _continual_reset(state: dict, env, env_self) -> None:
    """Continual harness: reseed and reset the underlying NLE so the chat
    session continues into a new life. Preserves journal + belief notes;
    bumps a `_continual_life` counter; records the death into the journal."""
    lives_left = state.get("_continual_lives_left", env_self.continual_lives)
    life_no = state.get("_continual_life", 1)
    # Snapshot death context into the journal so the agent can remember it.
    s = state.get("structured_obs")
    death_note = "died"
    if s is not None:
        death_note = (
            f"life {life_no}: died at Dlvl {s.status.get('depth','?')} "
            f"on turn {s.status.get('time','?')} "
            f"(max Dlvl reached {state.get('max_dlvl_reached','?')})"
        )
    journal = state.get("journal")
    if journal is not None:
        try:
            journal.add_note(f"death:life{life_no}", death_note)
        except Exception:
            pass
    # Reseed deterministically from the original seed + life number.
    orig_seed = state.get("_orig_seed")
    if orig_seed is None:
        # Recover from env metadata if not stored yet.
        orig_seed = (env.current_seeds or (0, 0))[0]
        state["_orig_seed"] = orig_seed
    new_seed = (int(orig_seed) * 1_000_003 + life_no) & 0x7FFFFFFF
    env.seed(core=new_seed, disp=new_seed)
    obs, _meta = env.reset()
    from nethack_harness.tools.skills import bootstrap_character
    character = bootstrap_character(env)
    state["character"] = character
    state["raw_obs"] = obs
    state["structured_obs"] = shape_observation(obs, character)
    state["max_dlvl_reached"] = max(state.get("max_dlvl_reached", 1), 1)
    state["died"] = False
    state["ascended"] = False
    state["_continual_life"] = life_no + 1
    state["_continual_lives_left"] = lives_left - 1
    # Reset per-life ephemera but keep cross-life memory (journal, belief).
    state["_seen_stairs_down"] = set()


def _capture_user_content(content, out_dir, *, run_id: str, turn: int):
    """Return a trace-safe copy of the per-turn user content.

    Strings pass through. For a multimodal list, each image_url data URI is
    decoded and written to ``<out_dir>/images/<run_id>_<turn>_<idx>.png`` and the
    entry is rewritten to reference the relative path instead of the inline
    base64, so the exact image the model saw is replayable without bloating the
    NDJSON.
    """
    if isinstance(content, str):
        return content
    import base64 as _b64
    images_dir = Path(out_dir) / "images"
    out = []
    idx = 0
    for entry in content:
        if entry.get("type") == "image_url":
            url = (entry.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and "base64," in url:
                images_dir.mkdir(parents=True, exist_ok=True)
                b64 = url.split("base64,", 1)[1]
                fname = f"{run_id}_{turn}_{idx}.png"
                (images_dir / fname).write_bytes(_b64.b64decode(b64))
                out.append({"type": "image_url", "image_url": {"path": f"images/{fname}"}})
                idx += 1
            else:
                out.append(entry)
        else:
            out.append(entry)
    return out


#: Markers the netplay_true adapter and the harness put in skill feedback.
#: Ordered: the first match wins, so an explicit failure beats the generic
#: "completed" that the vendored skills emit unconditionally.
_STATUS_MARKERS: tuple[tuple[str, str], ...] = (
    ("REFUSED:", "refused"),
    ("Skill raised ", "error"),
    ("' failed", "failed"),
    ("failed:", "failed"),
    ("No valid path", "failed"),
    ("Interrupting skill", "interrupted"),
    ("[INTERRUPTED:", "interrupted"),
    # NetPlay's own per-skill budget cap (100 in-game turns) -- the skill stopped
    # itself, which is an interruption, not a completion.
    ("timesteps without interruption", "interrupted"),
    ("' completed", "completed"),
    ("completed:", "completed"),
    # Harness-owned skills that do not go through the NetPlay adapter and so
    # have no `Skill 'x' <verb>` envelope of their own.
    ("rollback unavailable", "failed"),
    ("rollback failed", "failed"),
    ("cannot roll back", "failed"),
    ("rolled back ", "completed"),
    ("reveal: ", "failed"),          # "reveal: map unavailable this turn."
    ("reveal (x", "completed"),      # "reveal (x1-x20, y3-y9):\n..."
    # request_map takes no NLE step, so no skill envelope; without a marker it
    # tallied as "unknown" in status distributions (e7 smoke, 3 of 20 calls).
    ("Refreshing the full map", "completed"),
)


def classify_tool_result(feedback: str, *, default: str = "unknown") -> str:
    """Map a skill's prose feedback onto the `TS.TOOL_STATUSES` vocabulary.

    This exists so callers stop regex-scraping `rendered_user_message`. It is
    deliberately a *classification of what the skill claimed*, not a judgement
    of whether anything happened -- `clock_advanced` on the same record is what
    tells you that, and the two disagree often: measured on the probe, 6 of 10
    zero-clock turns still report `completed` (e.g. `completed: Tile (36,10) is
    blocked, stopping adjacent to it`). Treating "completed" as success alone
    overstates it by ~21 points.
    """
    fb = feedback or ""
    for marker, status in _STATUS_MARKERS:
        if marker in fb:
            return status
    if "No effect." in fb:
        return "no_op"
    return default


# ---------------------------------------------------------------------------
# Tool-call correlation id -- the barrier that aligns the two log streams
# ---------------------------------------------------------------------------
#: The LLM-side transcript (``traces.jsonl``) and the game-side turn NDJSON
#: never referenced each other; name-based joins fail exactly where they
#: matter (one ipython assistant turn -> many game calls). The only channel
#: the two streams share is the tool RESULT itself: whatever the server
#: returns flows back through the model transcript verbatim in every scaffold
#: (an MCP tool message, or printed output inside an ipython block). So the
#: server assigns a per-rollout monotonic call number at `_apply_tool_call`
#: (the single execution path every route shares), stamps it on the turn
#: record (``tool_results[0]["call_id"]``), and appends this marker to the
#: result payload. Alignment is then: turn record <-> the transcript message
#: carrying the same marker <-> the assistant message that issued the call --
#: exact by construction, no name-walk, no ordinal guessing.
#:
#: The marker is a trailing line so it never disturbs `startswith`-style
#: feedback heuristics (`[Moved ...]`, `[turn -N] ...`), and it deliberately
#: rides the PAYLOAD, never the published tool schema: the model-visible
#: function definitions stay byte-identical to an uninstrumented run.
#: Consumed by ``tools/trace_align.py`` (which owns the parsing regex).
CALL_ID_MARKER_FORMAT = "[call#{}]"


def call_id_marker(call_id) -> str:
    """The marker text for one call id (kept in one place; see the regex in
    ``tools/trace_align.py``, which must stay in sync)."""
    return CALL_ID_MARKER_FORMAT.format(int(call_id))


def append_call_marker(content, call_id):
    """Append the correlation marker to a result payload.

    Handles both payload shapes `_apply_tool_call_inner` produces: a plain
    string observation, and a multimodal content list (the marker becomes a
    trailing ``{"type": "text"}`` block, which ``content_to_text`` folds back
    into the trace's text view). Unknown shapes are returned untouched --
    losing a marker is recoverable (the join reports the gap); corrupting a
    payload is not.
    """
    if isinstance(content, str):
        marker = call_id_marker(call_id)
        return f"{content}\n{marker}" if content else marker
    if isinstance(content, list):
        return [*content, {"type": "text", "text": call_id_marker(call_id)}]
    return content


def build_tool_result(*, name, arguments, feedback, status=None,
                      clock_before=None, clock_after=None, engine_steps=0,
                      reward=0.0, game_message=None, call_id=None,
                      native_call_id=None, call_id_echoed=False) -> dict:
    """One machine-readable outcome record for one tool call.

    ``call_id`` is the per-rollout monotonic correlation id assigned at
    dispatch (see :data:`CALL_ID_MARKER_FORMAT`); explicitly ``None`` for a
    record no dispatched call produced (the end-of-rollout flush), so absence
    of a call is a stated fact rather than a missing key. ``native_call_id``
    is the transport's own id when the writer can see one (the OpenAI-style
    ``tool_calls[].id`` on the harness route) -- corroboration only, since the
    MCP and code-mode transports don't surface one to the server. ``call_id_echoed``
    records whether the marker was actually appended to the result the model
    saw (the config knob can disable the echo for token-matched cells), so a
    post-hoc join knows whether to expect markers in the transcript.
    """
    if status is None:
        status = classify_tool_result(feedback)
    advanced = None
    if clock_before is not None and clock_after is not None:
        try:
            advanced = int(clock_after) > int(clock_before)
        except (TypeError, ValueError):
            advanced = None
    return {
        "name": name,
        "arguments": arguments,
        "status": status,
        "game_message": game_message or "",
        "clock_before": clock_before,
        "clock_after": clock_after,
        "clock_advanced": advanced,
        "engine_steps": int(engine_steps),
        "reward": float(reward),
        "feedback": feedback or "",
        "call_id": int(call_id) if call_id is not None else None,
        "native_call_id": str(native_call_id) if native_call_id else None,
        "call_id_echoed": bool(call_id_echoed),
    }


#: Why the MCP route cannot see the model's message at write time, stated once
#: so every affected record carries the same auditable sentence. Consumed by
#: `tools/trace_reasoning.py`, which is the thing that fixes it after the fact.
MCP_REASONING_UNAVAILABLE = (
    "dispatched over MCP: the tool server is a separate process from the one "
    "the model talks to, so no assistant message exists at write time. Backfill "
    "it from the rollout trace with `python -m tools.trace_reasoning <run_dir>` "
    "(or let NetHackTask.finalize do it live)."
)

#: The model did produce a message this turn; it simply had no visible text
#: (a bare tool call). Distinct from "this route cannot see it" above.
NO_TEXT_REASONING_UNAVAILABLE = (
    "the model emitted a tool call with no assistant text this turn"
)


def _reasoning_block(assistant_msg, dispatch_route: str) -> dict:
    """The `reasoning` block for a record written live by the harness.

    Three outcomes, and keeping them apart is the whole point:
      * text in hand              -> available, source=assistant_message;
      * message in hand, no text  -> unavailable, "the model said nothing";
      * no message at all (MCP)   -> unavailable, and says why + how to fix it.
    """
    if assistant_msg is None:
        if dispatch_route == "mcp":
            return TS.unavailable_reasoning(MCP_REASONING_UNAVAILABLE)
        return TS.unavailable_reasoning(
            "no assistant message was associated with this turn")
    if isinstance(assistant_msg, dict):
        text = assistant_msg.get("content") or ""
        extra = assistant_msg.get("reasoning_content") or ""
    else:
        text = getattr(assistant_msg, "content", "") or ""
        extra = getattr(assistant_msg, "reasoning_content", "") or ""
    if not (text or extra):
        return TS.unavailable_reasoning(NO_TEXT_REASONING_UNAVAILABLE)
    return TS.reasoning_record(text, "assistant_message", "inline",
                               reasoning_text=extra)


def _write_trace_entry(env_self, state: dict, assistant_msg, tool_calls,
                       action_indices, total_reward: float, obs_text: str,
                       obs_content=None, *, actions=None, tool_results=None,
                       all_messages=None, step_frames=None, applied=True, lm_turn=None,
                       turn=None, gt_obs=True, dispatch_route="harness") -> None:
    """Write one NDJSON line per LM turn. Best-effort; never raises.

    Captures everything needed to render the game as the model saw it (raw
    24x80 tty grid, structured obs, the literal user message, the assistant
    message, the parsed tool calls, reward, dlvl, hp) AND everything needed to
    replay it independently of the rendering: the full ordered engine command
    stream (`actions`), the engine's own ground-truth observation (`gt_obs`),
    machine-readable call outcomes (`tool_results`) and every intermediate
    in-game message (`all_messages`).

    Exactly one record is written per LM turn -- including turns that never
    reached the engine (hallucinated tool, journal-only call, no tool call at
    all, post-death refusal) -- so `len(records)` matches `metrics.num_turns`
    and `metrics.total_tool_calls`. `lm_turn` is that authoritative counter;
    `turn` keeps its version-0/1 meaning (engine-advancing turns only).
    """
    if not env_self.trace_dir:
        return
    try:
        import os as _os
        import time as _time
        out_dir = Path(env_self.trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = state.get("_trace_run_id")
        if run_id is None:
            seeds = state.get("env").current_seeds if state.get("env") else (0, 0)
            run_id = f"{seeds[0]}_{_os.getpid()}_{int(_time.time())}"
            state["_trace_run_id"] = run_id
        path = out_dir / f"{run_id}.ndjson"
        raw = state.get("raw_obs")
        grid = []
        if raw is not None:
            try:
                grid = [
                    "".join(chr(int(c)) for c in row).rstrip()
                    for row in raw.tty_chars
                ]
            except Exception:
                pass
        s = state.get("structured_obs")
        status = dict(s.status) if s is not None else {}
        assist_content = ""
        if assistant_msg is not None:
            if isinstance(assistant_msg, dict):
                assist_content = assistant_msg.get("content", "") or ""
            else:
                assist_content = getattr(assistant_msg, "content", "") or ""
        tc_serial = []
        for tc in (tool_calls or []):
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                tc_serial.append({
                    "name": fn.get("name") or tc.get("name"),
                    "arguments": fn.get("arguments") or tc.get("arguments"),
                })
            else:
                fn = getattr(tc, "function", None)
                tc_serial.append({
                    "name": getattr(fn, "name", None) if fn is not None
                            else getattr(tc, "name", None),
                    "arguments": getattr(fn, "arguments", None) if fn is not None
                                 else getattr(tc, "arguments", None),
                })
        # `messages` (version 0/1) is last-message-only because it comes from
        # `shape()`, which only ever sees the FINAL observation of the turn.
        # `all_messages` is the full ordered stream the TurnRecorder harvested
        # from every engine step, with the final shaped message appended if the
        # post-turn re-shape produced something the hook did not see.
        shaped_msgs = list(s.messages) if s and s.messages else []
        full_msgs = list(all_messages) if all_messages else []
        for m in shaped_msgs:
            if m and (not full_msgs or full_msgs[-1] != m):
                full_msgs.append(m)
        turn_no = state.get("turn_count", 0) if turn is None else turn
        entry = {
            "turn": turn_no,
            "lm_turn": state.get("_trace_lm_turn", turn_no) if lm_turn is None else lm_turn,
            "applied": bool(applied),
            "t_wall": _time.time(),
            # Strictly monotonic clock (unaffected by NTP/wall-clock
            # adjustments), alongside `t_wall`. Diffing consecutive `t_mono`
            # values is the reliable way to measure seconds/call and detect
            # superlinear latency growth as context accumulates over a long
            # rollout (`t_wall` deltas can be corrupted by a clock step
            # mid-rollout; `t_mono` cannot).
            "t_mono": _time.monotonic(),
            "variant": env_self.variant,
            "raw_grid": grid,
            "status": status,
            "dlvl": status.get("depth"),
            "hp": status.get("hitpoints"),
            "max_hp": status.get("max_hitpoints"),
            "max_dlvl_reached": state.get("max_dlvl_reached"),
            "continual_life": state.get("_continual_life", 1),
            "rendered_user_message": obs_text,
            "rendered_user_content": _capture_user_content(
                obs_content if obs_content is not None else obs_text,
                out_dir, run_id=run_id, turn=turn_no),
            "assistant_message": assist_content,
            "tool_calls": tc_serial,
            # Which route delivered this call, and -- explicitly, never by
            # inference from an empty string -- whether the agent's own words
            # for this turn are available and why not (schema version 3).
            "dispatch_route": dispatch_route,
            "reasoning": _reasoning_block(assistant_msg, dispatch_route),
            # Kept verbatim for version-0/1 readers. It is the harness step
            # loop's PLANNED index list and is structurally empty for every
            # pre_executed skill; `actions` below is the authoritative record.
            "action_indices": list(action_indices) if action_indices else [],
            "reward": float(total_reward),
            "messages": shaped_msgs,
            "all_messages": full_msgs,
            # One tty screen per engine step this turn (present only when the
            # env was built with record_step_frames=True). This is the
            # move-by-move replay data shape() otherwise collapses.
            **({"step_frames": list(step_frames)} if step_frames else {}),
            "actions": actions if actions is not None else TS.empty_action_record(
                "no engine step hook was active for this turn"),
            "tool_results": list(tool_results) if tool_results else [],
            # Ground truth straight off the engine, independent of what was
            # rendered. See `TS.GT_OBS_PLANES` for the size/fidelity tradeoff.
            "gt_obs": TS.encode_obs_blob(raw) if (gt_obs and raw is not None) else None,
        }
        # Variant CH: capture the refiner's per-interval edits (set by
        # _ch_refiner_hook on refinement turns) so the trace records exactly
        # what the teacher changed this turn.
        if state.get("_ch_last_edits"):
            entry["ch_edits"] = state["_ch_last_edits"]
        # Route through the schema helper (NOT a bare json.dumps) so every
        # record carries `schema_version`. The bare dumps is why
        # `TS.record_version()` read 0 on freshly written traces.
        with path.open("a") as f:
            f.write(TS.to_json_line(entry))
    except Exception:
        # Tracing must never break a rollout.
        pass


def _drop_before_last_belief(messages, state) -> list:
    """Variant R: hard-drop every message before the chat-position corresponding
    to the most-recent belief_state:tN checkpoint.

    Heuristic: we don't track which chat turn produced each belief note, so
    instead we drop everything older than (refine_distance_from_end) where
    refine_distance is set to the belief_state_interval. The journal block
    inside the *current* user message already carries the belief notes, so
    no semantic info is lost.

    Leaves the system message and the most recent user/assistant pair fully
    intact. Inserts a single elision marker so the model knows context was
    dropped.
    """
    if not state:
        return messages
    journal = state.get("journal")
    if journal is None:
        return messages
    # If no belief checkpoint has fired yet, do nothing.
    has_belief = any(k.startswith("belief_state:") for k in (journal.notes or {}).keys())
    if not has_belief:
        return messages
    # Find indices of user messages and keep only the last K (here K=2 — the
    # last user obs and its preceding assistant exchange suffice once the
    # belief state carries the rest).
    keep_window = 2
    user_idx = [i for i, m in enumerate(messages) if _msg_role(m) == "user"]
    if len(user_idx) <= keep_window:
        return messages
    cut_at = user_idx[-keep_window]
    out = []
    for i, m in enumerate(messages):
        if i == 0 and _msg_role(m) == "system":
            out.append(m)
            continue
        if i >= cut_at:
            out.append(m)
    # Insert elision marker right after the system message.
    insert_at = 1 if out and _msg_role(out[0]) == "system" else 0
    n_dropped = len(messages) - len(out)
    if n_dropped > 0:
        out.insert(insert_at, vf.UserMessage(
            role="user",
            content=f"[variant=R: {n_dropped} prior turns dropped; see JOURNAL belief_state notes for context]",
        ))
    return out


def _refinement_directive(state: dict) -> str:
    """Variant P (Continual Harness, arXiv:2605.09998) periodic self-refinement
    prompt. Injected every refine_interval turns; asks the agent to reflect on
    the last window of play and update its objective and/or write a lesson
    note. Because `pin_objective` and `add_note` are journal ops that don't
    consume an NLE step, the agent can spend this turn editing its own
    persistent memory without losing a game action."""
    turn = state.get("turn_count", 0)
    max_dlvl = state.get("max_dlvl_reached", 1)
    cur_dlvl = state["structured_obs"].status.get("depth", 1) if state.get("structured_obs") else 1
    return (
        f"[self-refinement turn (variant=P, t={turn})] "
        f"You are at Dlvl {cur_dlvl} (max reached {max_dlvl}). "
        f"Before your next action, reflect: is your current objective still "
        f"the right one? What pattern from the last {state.get('_refine_window', 20)} "
        f"turns should you remember? Call `pin_objective(text=...)` to update "
        f"the goal if it has shifted, or `add_note(key='lesson:t{turn}', "
        f"text=...)` to record a short lesson. These calls do NOT consume a "
        f"game turn. If nothing needs updating, take your normal action."
    )


def _ch_build_window(trajectory: list, n_turns: int) -> list[dict]:
    """Slice the last `n_turns` of chat history into {role, content} dicts
    for the Refiner. Handles both dict-shape and verifiers pydantic msgs."""
    out: list[dict] = []
    for msg in trajectory[-(2 * n_turns):]:
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "?")
            content = getattr(msg, "content", "")
        out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return out


def _ch_inject_system(messages, state: dict):
    """Append CH prompt addendum + macro list onto the system message."""
    addendum = (state.get("_ch_prompt_addendum") or "").strip()
    macros = state.get("_ch_skills") or {}
    if not addendum and not macros:
        return messages
    extra_lines: list[str] = []
    if addendum:
        extra_lines.append("\n[continual-harness addendum]\n" + addendum)
    if macros:
        # Surface available macros so the agent can call run_macro(name=...).
        macro_names = ", ".join(sorted(macros.keys()))
        extra_lines.append(
            f"\n[continual-harness macros] You may also call "
            f"`run_macro(name=...)` with one of: {macro_names}."
        )
    extra = "".join(extra_lines)
    out = list(messages)
    for i, m in enumerate(out):
        if _msg_role(m) == "system":
            out[i] = _replace_content(m, _msg_content(m) + extra)
            return out
    # No system message found (shouldn't happen, dataset prepends one) —
    # prepend a fresh system block.
    out.insert(0, vf.SystemMessage(role="system", content=extra.lstrip()))
    return out


def _ch_save_bootstrap(env_self, state: dict) -> None:
    """Persist the four CH components to <bootstrap_dir>/seed<N>.json on
    terminal. Best-effort; never raises."""
    try:
        if not getattr(env_self, "bootstrap_dir", None):
            return
        import os, json
        from nethack_harness.refiner import snapshot_components
        os.makedirs(env_self.bootstrap_dir, exist_ok=True)
        seed = state.get("_orig_seed", 0)
        path = os.path.join(env_self.bootstrap_dir, f"seed{seed}.json")
        with open(path, "w") as f:
            json.dump(snapshot_components(state), f, indent=2)
    except Exception:
        pass


def _compact_chat_history(messages, keep_full: int = 5, drop_after: int = 100):
    """Compact older user (env-response) messages in-place so the chat doesn't
    grow without bound. Assistant messages are kept verbatim — their tool_calls
    matter for downstream replay. Only the *content* of user messages older
    than `keep_full` turns is rewritten.

    Compaction tiers:
      - turn distance ≤ keep_full: full fidelity (unchanged)
      - keep_full < distance ≤ drop_after: one-line summary
      - distance > drop_after: message dropped entirely

    Returns a NEW list so we never mutate state["trajectory"].
    """
    # Count from the end. "Turn distance" = how many user messages back this is.
    out = list(messages)
    # Walk backwards over user messages to compute distance.
    user_indices = [i for i, m in enumerate(out) if _msg_role(m) == "user"]
    if len(user_indices) <= keep_full:
        return out
    # Older user messages we want to compact (oldest first up to threshold).
    to_compact = user_indices[: -keep_full]
    to_drop_indices = set()
    for distance_from_end, idx in enumerate(reversed(to_compact), start=keep_full + 1):
        if distance_from_end > drop_after:
            to_drop_indices.add(idx)
            continue
        # One-line summary: keep prefix tags ([autohalt: ...], [feedback: ...])
        # and HP/Turn/Dlvl status line; drop everything else.
        summary = _one_line_summary(_msg_content(out[idx]), distance_from_end)
        out[idx] = _replace_content(out[idx], summary)
    if to_drop_indices:
        # Replace dropped chunks with a single elision marker if there's anything to drop.
        n_dropped = len(to_drop_indices)
        out = [m for i, m in enumerate(out) if i not in to_drop_indices]
        # Insert a single elision marker at the start (after system prompt if present).
        from typing import cast
        insert_at = 1 if out and _msg_role(out[0]) == "system" else 0
        out.insert(insert_at, vf.UserMessage(
            role="user",
            content=f"[elided {n_dropped} older turns; see journal for context]",
        ))
    # Second pass: collapse consecutive compacted user messages with the same
    # status signature. In the 9071d001 trace, 90 user msgs were literally
    # "[turn -X] HP: 14/14 AC: 4 Dlvl: 1 ..." with the same HP/AC/Dlvl — pure
    # token noise. We shrink runs to "[turn -Y] (status unchanged)".
    out = _dedupe_compacted_runs(out)
    return out


_STATUS_SIG_RE = re.compile(r"HP:\s*(\d+/\d+)\s+AC:\s*(-?\d+)\s+Dlvl:\s*(\d+)")


def _compacted_status_signature(content: str) -> Optional[tuple]:
    """Return (hp_ratio, ac, dlvl, has_feedback) for a compacted user message,
    or None if the message isn't recognized as compacted-only (i.e., still has
    a MAP block, or is an elision marker, or has unique feedback)."""
    if "=== MAP ===" in content or "elided" in content[:30]:
        return None
    if not content.lstrip().startswith("[turn -"):
        return None
    # Any non-trivial bracketed feedback (e.g. [Moved S.], [Attack hit]) makes
    # this turn unique — don't collapse.
    head = content.split("\n", 1)[0]
    # Strip leading [turn -N]
    rest = re.sub(r"^\[turn -\d+\]\s*", "", head)
    fb_match = re.match(r"\[([^\[\]]{1,80})\]", rest)
    has_feedback = bool(fb_match and "turn -" not in fb_match.group(1))
    sig = _STATUS_SIG_RE.search(content)
    if not sig:
        return None
    return (sig.group(1), sig.group(2), sig.group(3), has_feedback)


def _dedupe_compacted_runs(messages):
    """Collapse consecutive compacted user messages with identical
    HP/AC/Dlvl signatures and no per-turn feedback into terse `[turn -Y]
    (unchanged)` placeholders. Keeps the first message in each run intact
    so the model can read the actual status; later messages just mark
    that nothing changed.

    Does not change message count or assistant messages — purely shrinks
    redundant content.
    """
    out = list(messages)
    last_sig: Optional[tuple] = None
    for i, m in enumerate(out):
        if _msg_role(m) != "user":
            continue
        content = _msg_content(m)
        sig = _compacted_status_signature(content)
        if sig is None:
            last_sig = None
            continue
        # If feedback present, keep full and reset run.
        _, _, _, has_feedback = sig
        if has_feedback:
            last_sig = sig
            continue
        if last_sig is not None and sig[:3] == last_sig[:3]:
            # Same status as previous compacted msg: shrink.
            turn_match = re.match(r"\[turn -(\d+)\]", content)
            label = turn_match.group(0) if turn_match else "[turn -?]"
            out[i] = _replace_content(m, f"{label} (unchanged)")
        last_sig = sig
    return out


def _msg_role(m) -> str:
    """Pull role from either a dict-shaped message or a pydantic Message."""
    if isinstance(m, dict):
        return str(m.get("role", ""))
    return str(getattr(m, "role", ""))


def _msg_content(m) -> str:
    if isinstance(m, dict):
        c = m.get("content", "")
    else:
        c = getattr(m, "content", "")
    return c if isinstance(c, str) else ""


def _replace_content(m, new_content: str):
    """Return a copy of `m` with content swapped, preserving role + pydantic class."""
    if isinstance(m, dict):
        out = dict(m)
        out["content"] = new_content
        return out
    # pydantic: use model_copy(update=...).
    try:
        return m.model_copy(update={"content": new_content})
    except Exception:
        return vf.UserMessage(role=_msg_role(m), content=new_content)


def _msg_get(m, key, default=None):
    if isinstance(m, dict):
        return m.get(key, default)
    return getattr(m, key, default)


def _sanitize_assistant_content(messages: list) -> list:
    """Coerce any assistant message with null/empty content and no tool_calls
    to a non-null string so strict OpenAI-compatible endpoints accept the
    history.

    A "thinking" model (e.g. Qwen3.5) can emit a turn that is pure
    ``reasoning_content`` with ``content=None`` and ``tool_calls=None``. When
    that message is re-sent as history, Prime Inference returns HTTP 422:
    "content is required unless an assistant message includes tool_calls or
    function_call". We replace the null content with the message's
    ``reasoning_content`` (so no signal is lost) or a single space placeholder.
    Messages that already carry content or tool_calls are returned untouched.
    """
    out = []
    for m in messages:
        if _msg_role(m) == "assistant":
            content = _msg_get(m, "content", None)
            tool_calls = _msg_get(m, "tool_calls", None)
            func_call = _msg_get(m, "function_call", None)
            empty = content is None or (isinstance(content, str) and content.strip() == "")
            if empty and not tool_calls and not func_call:
                reasoning = _msg_get(m, "reasoning_content", None)
                replacement = reasoning if isinstance(reasoning, str) and reasoning.strip() else " "
                out.append(_replace_content(m, replacement))
                continue
        out.append(m)
    return out


def _one_line_summary(content: str, turn_distance: int) -> str:
    """Squash a full obs_text into one line. Heuristics:
       - Keep the STATUS line ("HP: x/y AC: z Dlvl: d Turn: t ...") if present.
       - Keep any [autohalt: ...] / [...] feedback prefix.
       - Otherwise just emit a placeholder.

    IMPORTANT (bug-fix 2026-05-16): `get_prompt_messages` walks the FULL
    chat history every turn, so already-compacted messages get re-fed into
    this function each turn. Previously, the loop would re-pick the
    `[turn -N]` label as `feedback` and prepend a new `[turn -K]` to it
    each round — after many turns, the message becomes a useless chain
    like "[turn -92] [turn -91] [turn -90] ... [turn -7]" with no content.
    Now we detect already-compacted messages and emit a single fresh
    label, dropping the chain. Idempotent.
    """
    stripped_content = content.strip()
    looks_compacted = (
        stripped_content.startswith("[turn -")
        and "=== " not in stripped_content
        and "MAP" not in stripped_content
    )
    if looks_compacted:
        # Already compacted: extract whatever feedback/status we saved earlier
        # and re-emit with the fresh distance label. Without this, every
        # subsequent compaction round drops the [Moved S.] / [Picked up]
        # marker, erasing the agent's action audit-log.
        feedback_part = ""
        hp_part = ""
        # Drop the leading "[turn -N] " then scan the remainder.
        remainder = re.sub(r"^\[turn -\d+\]\s*", "", stripped_content)
        # Feedback is a short bracketed token like "[Moved S.]" or "[Picked up]"
        fb_match = re.match(r"(\[[^\[\]]{1,80}\])\s*", remainder)
        if fb_match and "[turn -" not in fb_match.group(1):
            feedback_part = fb_match.group(1)
            remainder = remainder[fb_match.end():]
        # Status line: "HP: x/y AC: z ..."
        hp_match = re.search(r"HP:\s*\d+/\d+[^\n]*", remainder)
        if hp_match:
            hp_part = hp_match.group(0).strip()
        parts = [f"[turn -{turn_distance}]"]
        if feedback_part: parts.append(feedback_part)
        if hp_part: parts.append(hp_part)
        return " ".join(parts)

    status_line = ""
    feedback = ""
    for line in content.splitlines()[:25]:  # cap scan; obs is short prefix
        line = line.strip()
        # Only treat short bracketed lines as feedback — not chained turn labels.
        if line.startswith("[") and line.endswith("]") and len(line) < 200 and "[turn -" not in line:
            feedback = line
        if line.startswith("HP: "):
            status_line = line
            break
    parts = [f"[turn -{turn_distance}]"]
    if feedback:
        parts.append(feedback)
    if status_line:
        parts.append(status_line)
    return " ".join(parts)


def _check_halt_condition(raw_obs, hp_before: int) -> Optional[str]:
    """Per-step halt for multi-action skills. Returns a short reason string
    if the model should regain control NOW, or None to continue.

    Conditions:
      - HP dropped by ≥25% of the pre-skill value (we're being hit).
      - Hunger blstat indicates Weak/Fainting (need to eat).
      - HP/maxHP < 0.3 (precarious situation).
    """
    try:
        # NLE blstats indices: 10=HP, 11=maxHP, 21=hunger (0=Satiated/1=Normal/...)
        # See nle/nethack/nethack.py:BLStats.
        blstats = raw_obs.get("blstats") if isinstance(raw_obs, dict) else None
        if blstats is None:
            return None
        hp = int(blstats[10])
        max_hp = max(int(blstats[11]), 1)
        hunger = int(blstats[21]) if len(blstats) > 21 else 1
    except (KeyError, IndexError, TypeError):
        return None

    if hp_before > 0 and hp <= hp_before * 0.75:
        return f"HP dropped {hp_before}→{hp}"
    if hp <= max_hp * 0.3:
        return f"HP critical ({hp}/{max_hp})"
    if hunger >= 4:  # Weak (4) or Fainting (5) or Starving (6)
        return f"hunger level {hunger}"
    return None


BELIEF_STATE_INTERVAL = 25
"""Every N turns, call SubLM.summarize on the journal+status and store the
result as a `belief_state:<turn>` note. Allows history-compaction to drop
older turns without losing semantic context. Survey recommendation #3."""


def _maybe_belief_state_summary(state: dict) -> None:
    """Periodic belief-state distillation. Best-effort — silently skips on
    SubLM error so it never breaks a rollout.

    When the configured sub_lm is the OfflineSubLM stub (default), we skip
    the stub call and record a concrete status snapshot instead. The stub's
    "[offline-summary] ..." output isn't useful to the agent; a status
    snapshot at least surfaces HP/dlvl/turn at a known prior moment.
    """
    journal = state.get("journal")
    if journal is None:
        return
    try:
        s = state.get("structured_obs")
        turn = state.get("turn_count", 0)
        # Concrete status snapshot — useful regardless of SubLM backend.
        if s is not None:
            status_snap = (
                f"HP {s.status.get('hitpoints','?')}/{s.status.get('max_hitpoints','?')} "
                f"AC {s.status.get('armor_class','?')} "
                f"Dlvl {s.status.get('depth','?')} "
                f"Turn {s.status.get('time','?')} "
                f"max_dlvl={state.get('max_dlvl_reached','?')} "
                f"descents={state.get('descent_count',0)}"
            )
        else:
            status_snap = "(no obs)"

        # If a real (non-Offline) SubLM is wired, use its richer summary.
        sub_lm = state.get("sub_lm")
        from nethack_harness.tools.code_mode import OfflineSubLM
        if sub_lm is not None and not isinstance(sub_lm, OfflineSubLM):
            ctx_lines = [status_snap]
            for k, v in journal.notes.items():
                ctx_lines.append(f"- {k}: {v}")
            ctx = "\n".join(ctx_lines)
            try:
                summary = sub_lm.summarize(ctx, query=f"belief state at turn {turn}")
                journal.add_note(f"belief_state:t{turn}", summary)
                return
            except Exception:
                pass  # fall through to status snapshot

        journal.add_note(f"belief_state:t{turn}", status_snap)
    except Exception:
        pass


def _maybe_distill(state: dict, prior_dlvl: int) -> None:
    """Belief-state distillation hook. Calls the SubLM (default: Offline)
    to summarize what happened on `prior_dlvl` and adds it to the journal
    as `dlvl_<n>_summary`. Cheap when the SubLM is offline; nontrivial
    when wired to a real inference server.
    """
    journal = state.get("journal")
    if journal is None:
        return
    try:
        from nethack_harness.tools.code_mode import _default_sub_lm
        sub_lm = state.get("sub_lm") or _default_sub_lm()
        ctx_lines = []
        for k, v in journal.notes.items():
            ctx_lines.append(f"- {k}: {v}")
        ctx = "\n".join(ctx_lines) if ctx_lines else "(no notes recorded on this level)"
        summary = sub_lm.summarize(ctx, query=f"key events on dlvl {prior_dlvl}")
        journal.add_note(f"dlvl_{prior_dlvl}_summary", summary)
    except Exception:
        # Distillation is best-effort; don't break the rollout.
        pass


def _to_action_indices(env: NetHackCoreEnv, actions: list[int]) -> list[int]:
    """Normalize skill action values to the keystroke bytes the engine consumes.

    The semantic action enums ARE keystrokes (CompassDirection.N == 107 ==
    ord('k'), MiscAction.MORE == 13, ...) and ``EngineEnv.step`` takes those
    bytes directly -- so there is no longer an action-index translation layer.
    This is an identity pass-through (kept as a named seam, and so callers don't
    have to change). ``env`` is unused but retained for signature stability.
    """
    return [int(a) for a in actions]


# Carriage return. NetHack binds it in prompt contexts (getlin submit, --More--
# acknowledge, menu confirm) but NOT in command context, where `rhack()` falls
# through to its bad-command branch and prints `Unknown command '^M'.`
# (third_party/NetHack/src/src/cmd.c, `visctrl()` renders 13 as "^M").
CARRIAGE_RETURN = 13


def _cr_would_be_unknown_command(obs) -> bool:
    """True when feeding a CR to the engine *right now* would be a stray CR.

    A stray CR is one that reaches NetHack's command dispatcher rather than a
    prompt, and its only effect is the `Unknown command '^M'.` top-line message.
    That message then persists on the tty until something repaints it, so a
    single stray CR pollutes the agent's observation for many turns afterwards.

    The discriminator is the engine's own `misc` observation, which is exactly
    the three "am I waiting for input, and what kind" flags:

        misc == (in_yn_function, in_getlin, xwaitingforspace)

    Measured against a live engine (Val/Monk, seed 19):

        idle, awaiting a command  -> (0, 0, 0)   <- a CR here is stray
        "[- or ?*]" item prompt   -> (1, 0, 0)
        --More--                  -> (0, 1, 1)
        getlin ("write what?")    -> (0, 1, 0)
        inventory menu, "(end)"   -> (0, 0, 1)

    So a CR is stray iff every flag is clear. Anything else -- including a state
    we cannot read -- is treated as "a prompt might be open", and the CR is sent
    unchanged. Never raises: a detector failure must not break a rollout, and
    failing open only restores the previous behaviour.
    """
    try:
        misc = obs.get("misc") if isinstance(obs, dict) else getattr(obs, "misc", None)
        if misc is None:
            return False
        return all(int(v) == 0 for v in misc)
    except Exception:
        return False




# ---------- rewards ----------

@vf.reward(weight=1.0)
async def scout_reward(state: vf.State) -> float:
    """
    Total normalized tiles scouted across the rollout.

    Verifiers' `Rubric.score_rollout` runs once at end of rollout. Returning
    the last step's `scout_delta` alone effectively reports only the very
    last action's exploration — every prior tile discovery is invisible to
    the eval harness. So env_response accumulates `scout_reward_total` each
    step (delta/1000) and we return that running sum. Old call sites that
    only set `scout_delta` (unit tests) still work as a fallback.
    """
    if "scout_reward_total" in state:
        return float(state["scout_reward_total"])
    return float(state.get("scout_delta", 0)) / 1000.0


@vf.reward(weight=10.0)
async def descent_reward(state: vf.State) -> float:
    """+1 per new dungeon level reached this episode.

    env_response increments `descent_count` whenever max_dlvl advances; we
    return the running tally. (A per-step comparison here would always read
    `depth == max_dlvl_reached` because env_response updates max_dlvl first.)
    """
    if "descent_count" in state:
        return float(state["descent_count"])
    # Back-compat for unit tests that set max_dlvl_reached + structured_obs.
    s = state.get("structured_obs")
    if s is None:
        return 0.0
    dlvl = s.status.get("depth", 1)
    if dlvl > state["max_dlvl_reached"]:
        state["max_dlvl_reached"] = dlvl
        return 1.0
    return 0.0


@vf.reward(weight=100.0)
async def success_reward(state: vf.State) -> float:
    """
    +1 if the tier's success_milestone fired this episode. Distinct from
    `ascension_reward` (which weight=1000) because milestone success is a
    rung on the curriculum, not the endgame.
    """
    return 1.0 if state.get("succeeded") else 0.0


@vf.reward(weight=1000.0)
async def ascension_reward(state: vf.State) -> float:
    """
    The big one. +1 when the player ascends (escapes the dungeon with the
    Amulet of Yendor), 0 otherwise. _detect_terminal_outcome in env_response
    sets state["ascended"]=True when the standard ascension messages appear.
    """
    return 1.0 if state.get("ascended") else 0.0


# ---------- terminal outcome detection ----------

# NetHack-3.6 prints these phrases on the final screen. We treat any of them
# as proof of ascension; the death case is symmetric (a player who didn't
# ascend but did die has state["died"]=True so we can attribute the outcome).
_ASCENSION_MARKERS = (
    "ascended to demigod",
    "ascended to demigoddess",
    "with the Amulet",
    "offered the Amulet",
)
_DEATH_MARKERS = (
    "killed by",
    "starved to death",
    "petrified by",
    "drowned",
    "quit the game",
    "Do you want your possessions identified",
)


def _decode_tty(obs) -> str:
    """Render the full tty into a single string for marker-scanning."""
    return "\n".join(
        "".join(chr(c) for c in row) for row in obs.tty_chars
    )

def _glyph_rle(obs):
    """Compact RLE of the 21x79 glyph-id plane ("id xN,id xN,..." row-major).

    The tty chars show WHAT is drawn; glyph ids carry IDENTITY (which monster
    species, which object class) -- analysis-grade ground truth per step.
    ~0.2-0.8 KB/step. [] / "" on any failure -- frame capture must never break
    a rollout.
    """
    g = obs.get("glyphs") if isinstance(obs, dict) else getattr(obs, "glyphs", None)
    if g is None:
        return ""
    try:
        out = []
        prev = None; run = 0
        for row in g:
            for v in row:
                v = int(v)
                if v == prev:
                    run += 1
                else:
                    if prev is not None:
                        out.append(f"{prev}x{run}" if run > 1 else f"{prev}")
                    prev, run = v, 1
        if prev is not None:
            out.append(f"{prev}x{run}" if run > 1 else f"{prev}")
        return ",".join(out)
    except Exception:
        return ""


def _tty_rows(obs):
    """The 24 tty rows as a list of strings (one screen). Right-trimmed.

    Tolerant of dict-shaped or attribute-shaped observations, and of a missing
    tty plane (returns [] rather than raising -- frame capture must never break
    a rollout). This is the per-step analogue of the trace's final raw_grid.
    """
    tty = obs.get("tty_chars") if isinstance(obs, dict) else getattr(obs, "tty_chars", None)
    if tty is None:
        return []
    try:
        return ["".join(chr(int(c)) for c in row).rstrip() for row in tty]
    except Exception:
        return []


def _detect_terminal_outcome(obs, state: dict) -> None:
    """
    Inspect the last observation to determine death vs ascension. Mutates
    state in place. Called every step (cheap) so the reward functions can
    read precomputed booleans.
    """
    state.setdefault("ascended", False)
    state.setdefault("died", False)
    if state["ascended"] or state["died"]:
        return  # terminal outcomes are absorbing

    screen = _decode_tty(obs)
    if any(m in screen for m in _ASCENSION_MARKERS):
        state["ascended"] = True
        state["terminated"] = True
        return
    if any(m in screen for m in _DEATH_MARKERS):
        state["died"] = True
        state["terminated"] = True




def _code_tool_adapter():
    """The single 'code' tool exposed in interface='code' mode."""
    def code(source: str) -> str:
        """Execute Python against the `nh` namespace.

        Available: nh.move/attack/descend/search/pickup/move_to/autoexplore,
        nh.add_note/recall, nh.wiki_lookup/wiki_search, nh.status/inventory/
        map_view/character. Constants: Direction.{N,NE,E,SE,S,SW,W,NW,WAIT},
        Position(x, y). Imports and dunder access are blocked. Stdout returns
        as the tool result. 5s wallclock cap.
        """
        return ""  # never called directly; env_response routes the source.

    return code


def _build_skill_adapter_callables(skill_set: str = "full", describe_args: bool = False) -> list:
    """
    Build one callable per registered skill with the right __name__, doc, and
    annotations so verifiers' tool-schema introspection works.

    The callables are stubs: calling them raises (we want a loud error if
    something ever bypasses `env_response` and tries to invoke them).
    """
    import inspect
    from typing import Optional as _Opt

    # Skills the harness owns (never exposed as agent tools). Menu/inventory
    # selection is auto-dismissed in env_response; eat/quaff/read take an
    # `item` arg and bundle the selection in-skill. Exposing these as agent
    # tools caused Qwen3.5-9B to spend 42% of turns on spurious menu calls.
    _HARNESS_OWNED = {"inventory_item", "menu_option"}

    # Namespace prefix of the vendored NetPlay skills (see tools/netplay_true.py).
    _NETPLAY_TRUE_PREFIX = "np_"

    # Namespace prefix of BALROG's 80 raw commands (see tools/balrog_actions.py).
    _BALROG_PREFIX = "bal_"

    # skill_set: 'full' (default), 'move' (only move + survival), 'dir8'
    # (8 single-direction tools + survival, no `move` aggregator), or a
    # comma-separated whitelist e.g. 'move,descend,search'. The ladder
    # exists to measure how much "free reasoning" each helper-skill
    # offloads from the agent. dir8 is the most-faithful NLE baseline.
    if skill_set == "dir8":
        # Single-direction tools (N/NE/.../NW) + descend + search +
        # pickup + attack + survival. NO move/move_to/autoexplore/
        # find_and_descend/kick aggregators. Strips all "free" pathfinding.
        keep = {"descend", "search", "pickup", "attack",
                "engrave_elbereth", "pray", "eat", "quaff", "read",
                "add_note", "recall", "pin_objective",
                "wiki_lookup", "wiki_search"}
        out = []
        # Generate 8 direction skill-adapters by binding `move(direction=...)`
        # to a fixed direction. Naming: `north`, `northeast`, etc.
        _DIR_NAMES = [("north","N"),("northeast","NE"),("east","E"),
                      ("southeast","SE"),("south","S"),("southwest","SW"),
                      ("west","W"),("northwest","NW")]
        for tname, dir_canon in _DIR_NAMES:
            out.append(_make_fixed_direction_adapter(tname, dir_canon))
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
        return out
    elif skill_set == "netplay":
        # NetPlay (Jeurissen, CoG 2024): a skill-only action surface with NO
        # low-level `move(direction=...)` primitive — the agent acts through
        # high-level pathfinding (move_to/autoexplore/find_and_descend) plus
        # interactions. The standardized action set for cross-encoding
        # benchmarks (hold actions fixed, vary the observation).
        # explore_and_descend supersedes the weaker open-loop autoexplore /
        # find_and_descend (which bump on doors/corridors and don't loop), so we
        # drop those — otherwise the LLM defaults to the familiar weak tools and
        # never calls the robust one. move_to stays for precise single-target moves.
        keep = {"move_to", "explore_and_descend",
                "attack", "throw", "descend", "search", "pickup", "engrave_elbereth", "pray",
                "eat", "quaff", "read", "kick", "add_note", "recall",
                "pin_objective", "wiki_lookup", "wiki_search"}
        # NB: the 1d observation-delivery tools (reveal / request_map) are
        # deliberately NOT in the default netplay set — adding them would change
        # the fixed action surface for every netplay experiment (Exp 1, 1b, 1c)
        # and confound the comparison. The 1d arms expose them via an explicit
        # comma-`skill_set` (e.g. "<netplay tools>,reveal,request_map").
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
        return out
    elif skill_set == "move":
        # `move(direction=...)` + survival, but NO move_to, NO autoexplore,
        # NO find_and_descend. Single-step movement only, agent reasons
        # about which direction. Slightly above dir8 since the LM picks
        # a direction string instead of a fixed tool.
        keep = {"move", "descend", "search", "pickup", "attack",
                "engrave_elbereth", "pray", "eat", "quaff", "read",
                "add_note", "recall", "pin_objective",
                "wiki_lookup", "wiki_search"}
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
        return out
    elif skill_set == "netplay_true":
        # NetPlay's ACTUAL published action surface, vendored from
        # github.com/CommanderCero/NetPlay @ 6acb90d and bound to our engine
        # (see environments/nethack/vendor/PROVENANCE.md and
        # nethack_harness/tools/netplay_true.py).
        #
        # Deliberately a SEPARATE set from `netplay` above. That set is our own
        # hand-written approximation: 18 tools, of which `attack` is a
        # directional bump rather than NetPlay's pursue-until-dead
        # `melee_attack(x,y)`, and `explore_and_descend` caps its search where
        # NetPlay's `explore_level` runs until exploration is provably
        # exhausted. Three experiment arms are already proven against it, so it
        # stays untouched; use `netplay_true` to A/B the action surface itself.
        #
        # All 31 skills of upstream's exposed repository (netplay/__init__.py
        # lines 9-18) are registered under an `np_` prefix so they cannot
        # collide with our same-named skills.
        #
        # No `move` tool is published here either, so the literal gate holds --
        # but this set is NOT gate-equivalent to `netplay`. `np_press_key` /
        # `np_type_text` ARE published (faithful to upstream), and both pass a
        # raw keystroke straight to the engine; NetHack reads vi-style letters
        # (h/j/k/l/y/u/b/n) as compass steps, so `np_type_text(text="hhhh")` is
        # an unrestricted 4-step walk -- a strict superset of a `move` tool.
        # This is correct fidelity to upstream, not a bug to "fix" by
        # withholding those two tools. See vendor/PROVENANCE.md ("The move
        # gate does not carry over to netplay_true") and
        # test_raw_keystroke_surface_present_in_netplay_true_absent_in_netplay.
        from nethack_harness.tools import netplay_true as _npt
        keep = set(_npt.NETPLAY_TRUE_TOOL_NAMES)
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
        return out
    elif skill_set == "np_core":
        # e7 seeding experiment (2026-08-18): the NARROW, individually-debugged
        # core of netplay_true. Selection rationale:
        #   np_move_to over np_go_to     -- coordinates subsume room ids
        #   np_press_key over np_type_text -- press_key reaches esc/space/enter,
        #       which type_text cannot; type_text is repeated press_key minus
        #       the specials, so press_key is the generator
        #   np_pray / np_apply / np_rest -- survival essentials
        # Everything else (eat/quaff/wield/kick/zap/menu answers) is reachable
        # through np_press_key answering NetHack's OWN prompts -- which is the
        # point: this surface is designed to run with auto_dismiss=False, so
        # the model answers prompts itself instead of the harness ESCing them.
        from nethack_harness.tools import netplay_true as _npt  # registers np_*
        from nethack_harness import tool_flags as _flags
        keep = {"np_explore_level", "np_melee_attack", "np_move_to",
                "np_press_key", "np_pray", "np_apply", "np_rest", "np_kick"}
        # skills-as-code arm: retire the four COMPOSITES, keeping the primitive
        # floor (np_press_key / np_pray / np_apply / np_rest, plus request_map
        # and search from the tier's skill_set). The agent re-implements the
        # composites client-side in its own editable `netplay` package. Leaving
        # the server-side versions reachable would let it no-op the whole arm
        # by calling the frozen implementation instead of its own.
        if not _flags.enabled("netplay_composites"):
            keep -= {"np_move_to", "np_melee_attack", "np_explore_level", "np_kick"}
        missing = keep - set(skill_registry.all_schemas())
        assert not missing, f"np_core tools not registered: {missing}"
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
        return out
    elif skill_set == "balrog80":
        # BALROG's published NLE action surface: the 80 text commands in their
        # `balrog/environments/nle/__init__.py` ACTIONS dict, and nothing else.
        #
        # This is the matched-action-space baseline for reading our encoding
        # results against BALROG's leaderboard numbers. It is deliberately
        # HARSHER than `dir8`: no pathfinding of any kind, no closed loops, and
        # no in-skill item selection -- `bal_eat` opens NetHack's own "What do
        # you want to eat?" prompt and the agent answers it on the next turn.
        # It is also, in one respect, more capable than `dir8`: `bal_travel`
        # and `bal_far_*` are stock NetHack commands that cover ground, so a
        # BALROG agent is not the unaided single-stepper it first appears.
        #
        # No journal/wiki tools here. BALROG gives its agent none, and adding
        # them would confound exactly the memory axis 1c measures.
        from nethack_harness.tools import balrog_actions as _bal
        # The 80 documented commands PLUS `bal_a`..`bal_z`. BALROG documents 80
        # but its valid action space is 248 strings including the bare letters,
        # which is how their agent answers item prompts ("What do you want to
        # eat? [dgh...]" -> "d"). Without them a tool-calling agent cannot
        # answer such a prompt at all, so eat/quaff/read/wield are dead.
        keep = set(_bal.BALROG_TOOL_NAMES) | set(_bal.BALROG_MENU_LETTERS)
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
        return out
    elif "," in skill_set:
        # Tokens are tool names, EXCEPT a preset name, which expands to that
        # preset's tools. The 1d arms are documented above as
        # "<netplay tools>,reveal,request_map" — but a token like "netplay_true"
        # matches no *registered tool*, so the arm silently collapsed to just
        # `request_map` (1 tool). Expanding presets by recursion keeps this
        # correct as the presets themselves change.
        tokens = [s.strip() for s in skill_set.split(",") if s.strip()]
        presets = {"netplay", "netplay_true", "np_core", "dir8", "move", "full", "balrog80"}
        out = []
        seen: set = set()
        for tok in tokens:
            if tok in presets:
                for adapter in _build_skill_adapter_callables(skill_set=tok, describe_args=describe_args):
                    nm = getattr(adapter, "__name__", "")
                    if nm and nm not in seen:
                        seen.add(nm)
                        out.append(adapter)
        keep = {t for t in tokens if t not in presets}
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep or name in seen: continue
            params = schema.get("parameters", {}) or {}
            seen.add(name)
            out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
        return out
    # default 'full'
    out = []
    for name, schema in skill_registry.all_schemas().items():
        if name in _HARNESS_OWNED:
            continue
        # The vendored NetPlay skills register themselves globally the moment
        # nethack_harness.tools.netplay_true is imported (by the
        # `netplay_true` branch above, or by a test). They are an alternative
        # ACTION SURFACE, not extra tools, so they must never leak into 'full'
        # -- that would silently add 31 tools to every existing arm.
        if name.startswith(_NETPLAY_TRUE_PREFIX):
            continue
        # Same reasoning for BALROG's 80 raw commands: importing
        # tools.balrog_actions registers them globally, and they are an
        # alternative ACTION SURFACE, not extra tools.
        if name.startswith(_BALROG_PREFIX):
            continue
        # `rollback` is an opt-in CAPABILITY (engine snapshot/restore), not a
        # default tool. It is registered globally the moment tools.skills is
        # imported, so without this it would silently appear in every 'full'
        # arm already run and change their action surface.
        if name == "rollback":
            continue
        params = schema.get("parameters", {}) or {}
        out.append(_make_skill_adapter(name, schema.get("description", ""), params, describe_args))
    return out


def _make_run_macro_adapter():
    """Tool stub for variant=CH `run_macro(name=...)`. The actual dispatch
    lives in NetHackVerifiersEnv.env_response — this exists only to surface
    the tool to the verifiers schema-introspection layer."""
    def run_macro(name: str):
        """Run a Continual-Harness macro (a Refiner-registered sequence of
        skill calls). The macro must already exist; ask `recall` or check
        the system message for available macro names."""
        return None
    return run_macro


def _make_fixed_direction_adapter(tool_name: str, direction: str):
    """Bind `move(direction=...)` to a specific direction → 1-arg tool.

    Returns a callable named `tool_name` (north/northeast/.../northwest) with
    no parameters; calling it dispatches `move(direction=direction)` through
    the registry. Used by `skill_set='dir8'` baseline.
    """
    def _fn():
        """One step in this direction (NLE primitive)."""
        return None
    _fn.__name__ = tool_name
    _fn.__doc__ = f"Take one NLE step {direction}. No A*, no aggregation — single primitive action."
    return _fn


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


# Human-useful parameter hints for the common typed tools. The vendored
# netplay_true schemas describe each param as just its own name ("x" -> "x"),
# and the inputSchema does not survive MCP transport to Prime Agent's client
# (rlm.mcp_base renders every tool as **kwargs with an empty JSON Schema), so
# the model has to PROBE for arguments at session start -- ~10 wasted calls per
# game, and an asymmetry vs the native-MCP Claude Code arm. When describe_args
# is on, we append a plain-language "Args:" clause to the DESCRIPTION string,
# which reaches the model on every arm (proven in-trace). Coordinate frame
# matches the system prompt: x = column 0-78, y = row 0-20.
_ARG_HINTS = {
    "x": "column 0-78", "y": "row 0-20",
    "key": "one key; '>' descend, '<' ascend, 's' search, ESC/SPACE/ENTER by name",
    "item_letter": "inventory letter (a-z, A-Z), e.g. 'f'",
    "direction": "n/e/s/w or ne/se/sw/nw (or 'self' for np_zap)",
    "count": "number of turns",
    "times": "number of searches (1-20)",
    "x1": "left column 0-78", "y1": "top row 0-20",
    "x2": "right column 0-78", "y2": "bottom row 0-20",
    "n": "number of turns to undo (1-15)",
    "room_id": "room/corridor id from the map",
    "text": "characters to type in order",
}


def _args_clause(name: str, params: dict) -> str:
    """A one-line 'Args: a (hint), b (hint)' clause, or '' for no-arg tools."""
    if not params:
        return "\nArgs: none."
    parts = []
    for pname, pschema in params.items():
        hint = _ARG_HINTS.get(pname) or (pschema.get("description") or "").strip()
        opt = " optional" if "default" in pschema else ""
        parts.append(f"{pname} ({hint}{opt})" if hint and hint != pname else f"{pname}{opt}")
    return "\nArgs: " + ", ".join(parts) + f". Call: {name}(" + \
        ", ".join(f"{k}=..." for k in params) + ")."


def _make_skill_adapter(name: str, description: str, params: dict, describe_args: bool = False):
    """Create a callable that exposes the schema verifiers expects.

    describe_args=True appends a plain-language Args clause to the docstring so
    the model does not have to probe for arguments (see _ARG_HINTS)."""
    import inspect
    if describe_args:
        description = (description or "").rstrip() + _args_clause(name, params)

    # Build a signature with parameters in declared order.
    sig_params = []
    annotations: dict = {}
    for pname, pschema in params.items():
        ptype = _TYPE_MAP.get(pschema.get("type", "string"), str)
        annotations[pname] = ptype
        default = pschema.get("default", inspect.Parameter.empty)
        sig_params.append(inspect.Parameter(
            pname,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=ptype,
        ))

    def _adapter(**kwargs):
        raise RuntimeError(
            f"Skill adapter {name!r} was invoked directly; this should not "
            "happen — env_response dispatches via skill_registry.call()."
        )

    _adapter.__name__ = name
    _adapter.__doc__ = description
    _adapter.__signature__ = inspect.Signature(parameters=sig_params)
    _adapter.__annotations__ = annotations
    return _adapter
