"""The prompt factory.

A :class:`PromptSpec` bundles the four things that define how a NetHack rollout
talks to the model:

  1. **observation form + processors** — :class:`ObsSpec` (``ascii`` / ``img``
     plus the per-turn obs blocks toggled via ``setup_flags``).
  2. **system prompt** — the instruction block.
  3. **per-turn template** — ``turn_template(structured, journal, state, *,
     compact, journal_max_chars) -> str | list`` renders the user message each
     turn (f-string-style assembly of the observation).
  4. **tools** — :class:`ToolSpec` (which skill set + any extra tool factories).

:func:`build_prompt` is the single constructor; :data:`VARIANT_REGISTRY` defines
every shipped variant as a composition of the rendering functions in
``nethack_harness.prompt.rendering``. ``NetHackVerifiersEnv`` holds one resolved
``PromptSpec`` and dispatches through it instead of scattering ``self.variant ==``
checks across the rollout loop.

This module reproduces the *exact* behaviour of the legacy ``self.variant``
branches — it changes control flow (registry dispatch), not rendered bytes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nethack_harness.prompt.rendering import (
    SYSTEM_PROMPT,
    format_observation_as_chat,
    _format_obs_balrog,
    _format_obs_glyphbox,
    _format_obs_netplay,
    _format_obs_glyphbox_native,
    _format_obs_summarize_reset,
    _glyph_run_encode,
    _render_ascii_map,
    _strip_blank_rows,
)
from nethack_harness.helpers import (
    _refinement_directive,
    _ch_build_window,
    _drop_before_last_belief,  # noqa: F401  (kept for symmetry / external use)
    _ch_inject_system,
    _make_run_macro_adapter,
)

# ---------- the four-part spec ----------


@dataclass(frozen=True)
class ObsSpec:
    """Observation form + per-turn obs-block processors.

    ``mode`` selects how the observation is presented (``ascii`` text grid or a
    rendered ``img``). ``setup_flags`` are the per-rollout state flags that gate
    the optional obs blocks (descent-salience, E1 frontier surface, E2 paint);
    ``setup_state`` writes them and the rendering functions read them.
    """

    mode: str = "ascii"
    setup_flags: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """Which tools the agent gets.

    ``skill_set`` is ``None`` to honour the caller's ``skill_set`` kwarg (the
    default), or a concrete set name to force it. ``extra_tools`` is a tuple of
    zero-arg factories that each return one extra tool callable (e.g. the CH
    ``run_macro`` adapter).
    """

    skill_set: Optional[str] = None
    extra_tools: tuple = ()


@dataclass(frozen=True)
class PromptSpec:
    """A fully-specified harness prompt: the four parts + cross-cutting hooks."""

    name: str
    system_prompt: str
    obs: ObsSpec
    turn_template: Callable[..., Any]
    tools: ToolSpec
    # Cross-cutting hooks discovered in the variant inventory. Each is a tuple
    # so a variant can compose several. turn_hooks mutate the per-turn prefix
    # (P refinement directive, CH refiner + sub-agents); history_transforms
    # rewrite the chat message list in get_prompt_messages (CH system inject).
    turn_hooks: tuple = ()
    history_transforms: tuple = ()


def build_prompt(
    *,
    name: str,
    system_prompt: str,
    obs: ObsSpec,
    turn_template: Callable[..., Any],
    tools: ToolSpec,
    turn_hooks: tuple = (),
    history_transforms: tuple = (),
) -> PromptSpec:
    """Construct a :class:`PromptSpec` from the four parts (+ optional hooks).

    This is the one place a harness author assembles a prompt: pick an
    observation form, a system prompt, a per-turn template, and a tool set.
    """
    return PromptSpec(
        name=name,
        system_prompt=system_prompt,
        obs=obs,
        turn_template=turn_template,
        tools=tools,
        turn_hooks=turn_hooks,
        history_transforms=history_transforms,
    )


# ---------- per-turn templates (ascii) ----------


def _canonical_template(structured, journal, state, *, compact, journal_max_chars):
    """The default per-turn renderer (B1 and most variants)."""
    return format_observation_as_chat(
        structured, journal, state=state,
        compact=compact, journal_max_chars=journal_max_chars,
    )


def _formatter_template(formatter):
    """Adapt a legacy ``_format_obs_*`` formatter to the turn_template signature.

    The legacy formatters take ``(structured, journal, state, journal_max_chars)``
    and apply their own compaction policy, so ``compact`` is ignored here — this
    matches the pre-refactor call sites exactly.
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        return formatter(structured, journal, state, journal_max_chars)

    return _render


def _image_template(render_name):
    """Per-turn template that returns a multimodal [image_url, text] content list.

    ``render_name`` selects the strict render path: "glyph" → GlyphMapper tiles,
    "tty" → tty-text raster. The text block is journal + status + inventory only
    (the image is the sole spatial channel).
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        from nethack_harness.prompt.image_render import (
            glyphs_to_png_b64, tty_to_png_b64, to_data_uri,
        )

        raw = state["raw_obs"]
        b64 = glyphs_to_png_b64(raw) if render_name == "glyph" else tty_to_png_b64(raw)
        text = format_observation_as_chat(
            structured, journal, state,
            compact=compact, journal_max_chars=journal_max_chars,
            include_map=False, include_local=False,
        )
        return [
            {"type": "image_url", "image_url": {"url": to_data_uri(b64)}},
            {"type": "text", "text": text},
        ]

    return _render


def _balrog_plus_map_template(fmt):
    """BALROG's natural-language scene description, followed by the map.

    Reproduces the observation shape of BALROG's published NetHack runs, whose
    CSVs carry a language description ("horizontal wall near north") *and* the
    raw grid in the same observation. Variant ``B`` gives the description with
    no grid; ``B0``/``JSON`` give the map with no description. Neither alone
    matches the leaderboard's input, which is what these cells are for.

    ``fmt`` is "ascii" (append the rendered map view) or "json" (append the
    readable per-tile JSON body).
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        desc = _format_obs_balrog(structured, journal, state, journal_max_chars)
        if fmt == "json":
            from nethack_core.map_model import build_map_model
            from nethack_harness.prompt.map_encoders import json_encode

            raw = state["raw_obs"]
            body = json_encode(
                build_map_model(raw),
                detail=state.get("map_detail", "full"),
                chars=raw.chars,
            )
            return f"{desc}\n=== MAP (JSON) ===\n{body}\n"
        map_view = _render_ascii_map(structured, state)
        if compact:
            map_view = _glyph_run_encode(_strip_blank_rows(map_view))
        return f"{desc}\n=== MAP ===\n{map_view}\n"

    return _render


def _nle_language_template():
    """BALROG's observation as produced by the package BALROG actually uses.

    `_balrog_plus_map_template` above is OUR reconstruction of their scene text,
    written from their published run CSVs. This one calls ngoodger's
    `nle-language-wrapper` C++ converter directly -- the same code path their
    agent saw -- so an `NLE_LANG` cell can claim observation parity literally
    instead of by approximation. Verified against a live fork game: the
    converter's spatial claims land on the right tiles (gold `$` adjacent
    southeast, pet `d` adjacent northwest, grid bug `x` westnorthwest), so its
    stock-NLE glyph tables agree with the fork's glyph numbering.

    Deliberately NO map block and no JOURNAL: the wrapper's five text channels
    *are* the observation in BALROG's setup, and appending our grid would
    recreate B_ASCII rather than reproduce theirs.

    Raises rather than falling back if the converter is unavailable -- a cell
    that silently rendered a different observation would be an invalid
    experiment, not a degraded one (see `prompt/nle_language.py`).
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        from nethack_harness.prompt.nle_language import render_language_view

        return render_language_view(state["raw_obs"]) + "\n"

    return _render


def _structured_map_template(fmt):
    """Per-turn template emitting a structured-text map (JSON or TOON).

    ``fmt`` is "json" or "toon". The map_detail level is read from
    ``state["map_detail"]`` (default "full"). The status/inventory block is
    appended via the canonical formatter with the ASCII map gated off.

    Sub-experiment 1b (JSON only): when ``state["cell_schema"]`` names any of
    {"seen","visited","reach"}, the JSON base is enriched with a COMPACT 0/1
    mask layer per enabled attribute (``seen_grid`` / ``visited_grid`` /
    ``reach_grid``), computed here from ``state`` and RLE-encoded like the
    terrain grid. A disabled attribute is ABSENT from the JSON. TOON is
    unaffected (empty cell_schema leaves JSON bit-identical to before).
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        from nethack_core.map_model import build_map_model
        from nethack_harness.prompt.map_encoders import (
            json_encode, toon_encode, build_cell_masks,
        )

        detail = state.get("map_detail", "full")
        model = build_map_model(state["raw_obs"])
        if fmt == "json":
            import numpy as np
            raw = state["raw_obs"]
            cell_schema = state.get("cell_schema") or set()
            cell_masks = None
            if cell_schema:
                bl = np.asarray(raw.blstats)
                player = (int(bl[0]), int(bl[1]))
                # Key visited tiles by the hero's ACTUAL depth (blstats[12]) so
                # this read matches how the tracker files them (max_dlvl_reached
                # lags a descent turn — see setup_state/env_response).
                dlvl_key = int(bl[12])
                visited_all = state.get("_visited_tiles") or {}
                visited_xy = visited_all.get(dlvl_key, set())
                cell_masks = build_cell_masks(
                    raw.chars, player, visited_xy, cell_schema,
                )
            # Pass `chars` so the body carries readable per-tile records rather
            # than the raw-glyph-id RLE the model cannot interpret.
            map_text = json_encode(
                model, detail=detail, chars=raw.chars, cell_masks=cell_masks,
            )
        else:
            map_text = toon_encode(model, detail=detail, compact=compact)
        status = format_observation_as_chat(
            structured, journal, state, compact=compact,
            journal_max_chars=journal_max_chars,
            include_map=False, include_local=False,
        )
        return f"=== MAP ({fmt.upper()}) ===\n{map_text}\n\n{status}"

    return _render


# ---------- sub-experiment 1d: observation-DELIVERY variants ----------

# Placeholder that stands in for the map block on a delayed-map turn where the
# map is elided (no material change and no forced refresh).
_DELAYED_MAP_PLACEHOLDER = "=== MAP (unchanged; call request_map to refresh) ==="
# Placeholder shown by the bounding-box variant, where the map is hidden and
# only reachable through the reveal() tool.
_BBOX_MAP_PLACEHOLDER = "=== MAP (hidden; call reveal(x1,y1,x2,y2) to view a region) ==="


def _delayed_map_fingerprint(structured, state):
    """Floor-level material-change signal for the delayed-map variants.

    Decision (2026-07-24, 'new floor + on request only'): the FULL map is
    auto-re-sent only when the **current dungeon level changes** (a new floor).
    Within a floor the agent navigates on per-action feedback and calls
    ``request_map`` to refresh on demand (``state['_force_map']``). We key on the
    CURRENT dlvl (``blstats[12]`` — the same source the visited tracker uses),
    not ``max_dlvl_reached`` (which never decreases, so it would miss ascents).
    """
    dlvl = 1
    try:
        if state is not None:
            bl = getattr(state.get("raw_obs"), "blstats", None)
            if bl is not None:
                dlvl = int(bl[12])
            else:
                dlvl = int(state.get("max_dlvl_reached", 1))
    except Exception:
        dlvl = 1
    return (dlvl,)


def _delayed_map_template(base_fmt):
    """Delayed / on-demand map delivery (variants DM, DM_JSON).

    The FULL map is re-sent only on a NEW FLOOR (current-dlvl change) OR when the
    agent forced a refresh via request_map/reveal (``state['_force_map']``, set in
    env_response). Same-floor moves do NOT re-send it. Otherwise a one-line
    placeholder stands in for the map block. Action feedback + status render
    unchanged every turn (they ride the caller's prefix_parts / the status
    block here), independent of whether the map is shown.

    ``base_fmt`` selects the encoding the FULL map uses so delayed delivery
    composes with either baseline: "ascii" (B0-style grid via
    format_observation_as_chat) or "json" (structured JSON map via the same
    builder the JSON variant uses, including any 1b cell layers).
    """
    _json_full = _structured_map_template("json")

    def _render(structured, journal, state, *, compact, journal_max_chars):
        force = bool(state.pop("_force_map", False)) if state is not None else False
        cur_fp = _delayed_map_fingerprint(structured, state)
        prev_fp = state.get("_delayed_map_fp") if state is not None else None
        show_map = force or prev_fp is None or cur_fp != prev_fp
        if state is not None:
            state["_delayed_map_fp"] = cur_fp

        if base_fmt == "json":
            if show_map:
                return _json_full(
                    structured, journal, state,
                    compact=compact, journal_max_chars=journal_max_chars,
                )
            status = format_observation_as_chat(
                structured, journal, state, compact=compact,
                journal_max_chars=journal_max_chars,
                include_map=False, include_local=False,
            )
            return f"{_DELAYED_MAP_PLACEHOLDER}\n\n{status}"

        # ascii
        if show_map:
            return format_observation_as_chat(
                structured, journal, state,
                compact=compact, journal_max_chars=journal_max_chars,
            )
        text = format_observation_as_chat(
            structured, journal, state, compact=compact,
            journal_max_chars=journal_max_chars, include_map=False,
        )
        return _splice_placeholder(text, _DELAYED_MAP_PLACEHOLDER)

    return _render


def _balrog_delayed_map_template():
    """BALROG-style language scene, with the ASCII map on the delayed schedule.

    Combines the two treatments that scored best independently in the sweep:
    the natural-language description (variant ``B_ASCII``, which pairs it with a
    map every turn) and delayed map delivery (variant ``DM``, which re-sends the
    full grid only on a new floor or on ``request_map``).

    The pairing is the point. Under plain ``DM`` the agent has *nothing* spatial
    between refreshes; here the language description carries orientation every
    turn while the expensive grid is withheld, so withholding it should cost
    less. Pair with ``belief_state_interval=0`` for the journal-only memory arm.
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        force = bool(state.pop("_force_map", False)) if state is not None else False
        cur_fp = _delayed_map_fingerprint(structured, state)
        prev_fp = state.get("_delayed_map_fp") if state is not None else None
        show_map = force or prev_fp is None or cur_fp != prev_fp
        if state is not None:
            state["_delayed_map_fp"] = cur_fp

        desc = _format_obs_balrog(structured, journal, state, journal_max_chars)
        if not show_map:
            return f"{desc}\n{_DELAYED_MAP_PLACEHOLDER}\n"
        map_view = _render_ascii_map(structured, state)
        if compact:
            map_view = _glyph_run_encode(_strip_blank_rows(map_view))
        return f"{desc}\n=== MAP ===\n{map_view}\n"

    return _render


def _bbox_json_template(structured, journal, state, *, compact, journal_max_chars):
    """JSON body with the per-tile `cells` withheld; map regions via reveal().

    `json_encode(..., detail="minimal")` emits player + entities and no map body
    at all, which is exactly the half we want kept: entities stay structured and
    coordinate-addressable, while the 21x79 tile array — the part that costs
    thousands of tokens per turn — is served only when the agent asks for it.
    Everything non-map (journal, status, inventory, under-player) renders as
    usual, matching the plain BBOX cell.
    """
    from nethack_core.map_model import build_map_model
    from nethack_harness.prompt.map_encoders import json_encode

    body = json_encode(build_map_model(state["raw_obs"]), detail="minimal")
    rest = format_observation_as_chat(
        structured, journal, state, compact=compact,
        journal_max_chars=journal_max_chars,
        include_map=False, include_local=False,
    )
    return (
        f"=== MAP (JSON, entities only) ===\n{body}\n"
        f"{_BBOX_MAP_PLACEHOLDER}\n\n{rest}"
    )


def _bbox_template(structured, journal, state, *, compact, journal_max_chars):
    """Bounding-box on-demand map (variant BBOX).

    The map is never rendered inline; the agent sees it only by calling
    ``reveal(x1,y1,x2,y2)``, which returns the requested sub-rectangle as tool
    feedback (no NLE step). Everything else (journal, status, inventory,
    under-player) renders normally.
    """
    text = format_observation_as_chat(
        structured, journal, state, compact=compact,
        journal_max_chars=journal_max_chars, include_map=False,
    )
    return _splice_placeholder(text, _BBOX_MAP_PLACEHOLDER)


#: BBOX_MIN's placeholder names everything the variant withholds, not just the
#: map — the line is the agent's only standing notice of what `reveal` buys.
_BBOX_MIN_PLACEHOLDER = (
    "=== SURROUNDINGS (hidden; call reveal(x1,y1,x2,y2) for the map plus "
    "ADJACENT / VISIBLE FEATURES / VISIBLE MONSTERS / MESSAGES) ==="
)

#: e7 raw-prompt cells publish `request_map` instead of `reveal` -- the
#: standing notice must name the tool that is actually callable, or we
#: recreate the dead-vocabulary failure class (3,058 dead hints, rendering.py).
_BBOX_MIN_PLACEHOLDER_REQUEST_MAP = (
    "=== SURROUNDINGS (hidden; call request_map for the full map plus "
    "ADJACENT / VISIBLE FEATURES / VISIBLE MONSTERS / MESSAGES) ==="
)


def _bbox_min_template(structured, journal, state, *, compact, journal_max_chars):
    """BBOX_MIN: the quiet turn is feedback + STATUS, nothing else.

    BBOX withholds the ASCII grid but still pushes ADJACENT, UNDER PLAYER,
    VISIBLE FEATURES, VISIBLE MONSTERS and MESSAGES on every turn — so the
    coordinate feed `reveal` competes with is free, and `reveal` fires on ~2%
    of turns. This variant withholds all of it. A quiet turn is the action
    feedback line (prepended by the harness, and already carrying the last
    GAME message), the placeholder above, STATUS + Character, and the MENU /
    inventory-prompt safety notices. The turn a `reveal` executes, the FULL
    section set renders alongside the crop — everything else is saved for
    when reveal is called, exactly once per purchase.

    The game-over block always renders (death must never arrive silently),
    and `reveal` still consumes no game turn, so buying the sections is free
    on the in-game clock and costs exactly one unit of the call budget.
    """
    # env_response sets _force_map for both reveal and request_map. Pop it
    # unconditionally (it must never linger across turns), but only a
    # request_map turn pushes the inline MAP -- reveal turns carry the grid in
    # the tool result and must render exactly as they always have.
    force = bool(state.pop("_force_map", False)) if state else False
    reveal_turn = bool(state) and state.get("_last_skill_name") == "reveal"
    request_turn = (force and bool(state)
                    and state.get("_last_skill_name") == "request_map")
    text = format_observation_as_chat(
        structured, journal, state, compact=compact,
        journal_max_chars=journal_max_chars, include_map=request_turn,
        minimal=not (reveal_turn or request_turn),
    )
    if request_turn:
        # The full section set incl. MAP is inline this turn; the "hidden"
        # placeholder would contradict it.
        return text
    pub = {str(t) for t in ((state or {}).get("_published_tools") or ())}
    placeholder = _BBOX_MIN_PLACEHOLDER
    if pub and "reveal" not in pub and "request_map" in pub:
        placeholder = _BBOX_MIN_PLACEHOLDER_REQUEST_MAP
    return _splice_placeholder(text, placeholder)


def _bbox_min_odjson_template(structured, journal, state, *, compact,
                              journal_max_chars):
    """BBOX_BJSON_OD: BBOX_MIN quiet turns; request_map returns map PLUS JSON.

    E15 V-arm follow-up (on-demand x encoding). Always-on encodings (B1,
    B_JSON, JSON) all scored below the hidden-map control, and the V1 autopsy
    localized the losses to (a) instant full-level stair targeting on arrival
    and (b) the loss of the deliberate stop-and-look beat that `request_map`
    turns provided; separately, the ADJACENT feed's multi-`@` anchoring
    ambiguity generated phantom-target attacks. This variant keeps delivery
    identical to BBOX_MIN — quiet turns are byte-identical, the map is only
    ever bought — and changes ONLY what the purchase returns: the full section
    set plus a `=== MAP (JSON) ===` block (the same structured, coordinate-
    addressable body the B_JSON/JSON variants render), spliced ahead of
    STATUS. One treatment, one axis: the encoding of the bought look.
    """
    force = bool(state.pop("_force_map", False)) if state else False
    reveal_turn = bool(state) and state.get("_last_skill_name") == "reveal"
    request_turn = (force and bool(state)
                    and state.get("_last_skill_name") == "request_map")
    text = format_observation_as_chat(
        structured, journal, state, compact=compact,
        journal_max_chars=journal_max_chars, include_map=request_turn,
        minimal=not (reveal_turn or request_turn),
    )
    if request_turn:
        # Splice the JSON body between the ASCII map and STATUS. Built from
        # the same raw obs the ASCII render used, so the two views can never
        # disagree; encoder failure degrades to the plain BBOX_MIN render
        # rather than breaking the turn.
        try:
            from nethack_core.map_model import build_map_model
            from nethack_harness.prompt.map_encoders import json_encode

            raw = state["raw_obs"]
            body = json_encode(
                build_map_model(raw),
                detail=state.get("map_detail", "full"),
                chars=raw.chars,
            )
            block = f"=== MAP (JSON) ===\n{body}\n"
            marker = "=== STATUS ==="
            if marker in text:
                text = text.replace(marker, f"{block}\n{marker}", 1)
            else:
                text = f"{text}\n{block}"
        except Exception:
            pass
        return text
    pub = {str(t) for t in ((state or {}).get("_published_tools") or ())}
    placeholder = _BBOX_MIN_PLACEHOLDER
    if pub and "reveal" not in pub and "request_map" in pub:
        placeholder = _BBOX_MIN_PLACEHOLDER_REQUEST_MAP
    return _splice_placeholder(text, placeholder)


def _sparse_template(structured, journal, state, *, compact, journal_max_chars):
    """SPARSE: entity-only map, always shown. No terrain, no duplicate sections."""
    return format_observation_as_chat(
        structured, journal, state, compact=compact,
        journal_max_chars=journal_max_chars, include_map=True, sparse_entities=True,
    )


def _sparse_ondemand_template(structured, journal, state, *, compact, journal_max_chars):
    """SPARSE_ONDEMAND: the entity list is WITHHELD until `reveal` is called.

    This is the first configuration in which the delivery-timing axis is real.
    `BBOX` withheld the ASCII grid but kept publishing stair/door/item
    coordinates through VISIBLE FEATURES on ~100% of turns, so `reveal` was
    competing with a free feed and fired on 2.1% of turns. Here every entity
    lives under MAP, so withholding MAP withholds all of it.
    """
    text = format_observation_as_chat(
        structured, journal, state, compact=compact,
        journal_max_chars=journal_max_chars, include_map=False, sparse_entities=True,
    )
    return _splice_placeholder(text, _BBOX_MAP_PLACEHOLDER)


def _guided(base_template, mode="lead"):
    """Wrap a turn template so each observation opens with the objective hint.

    The hint is derived from BALROG's own achievements table (see
    objective_hint.py): it names which of depth / experience is currently
    BINDING the max() score, what the next checkpoint on each axis is worth, and
    which to pursue. Everything else in the observation is byte-identical to the
    wrapped variant, so `X` vs `X_GUIDE` isolates the hint alone.
    """

    def _t(structured, journal, state, *, compact, journal_max_chars):
        from nethack_harness.prompt.objective_hint import objective_hint
        body = base_template(structured, journal, state,
                             compact=compact, journal_max_chars=journal_max_chars)
        try:
            return f"{objective_hint(structured, state, mode=mode)}\n\n{body}"
        except Exception:
            # A hint must never be able to break a rollout.
            return body

    return _t


def _splice_placeholder(text: str, placeholder: str) -> str:
    """Insert a map placeholder where the (omitted) map block would have been.

    The canonical renderer puts the map immediately before ``=== STATUS ===``;
    with include_map=False that block is absent, so we splice the placeholder in
    just ahead of STATUS to keep the layout familiar. Falls back to prepending.
    """
    marker = "=== STATUS ==="
    if marker in text:
        return text.replace(marker, f"{placeholder}\n\n{marker}", 1)
    return f"{placeholder}\n\n{text}"


# ---------- cross-cutting hooks (verbatim from the legacy env_response) ----------


def _p_refinement_hook(env_self, state, prefix_parts):
    """Variant P: inject a self-refinement directive every ``refine_interval``."""
    if (
        env_self.refine_interval > 0
        and state.get("turn_count", 0) > 0
        and state["turn_count"] % env_self.refine_interval == 0
        and not state.get("_refine_emitted_this_turn")
    ):
        prefix_parts.append(_refinement_directive(state))
        state["_refine_emitted_this_turn"] = True
    else:
        state["_refine_emitted_this_turn"] = False


def _ch_refiner_hook(env_self, state, prefix_parts):
    """Variant CH: run the configured Refiner and apply its CRUD edits."""
    if (
        env_self.refiner is not None
        and env_self.refine_interval > 0
        and state.get("turn_count", 0) > 0
        and state["turn_count"] % env_self.refine_interval == 0
        and not state.get("_ch_refined_this_turn")
    ):
        try:
            from nethack_harness.refiner import snapshot_components, apply_edits
            window = _ch_build_window(state.get("trajectory") or [], n_turns=env_self.refine_interval)
            edits = env_self.refiner.refine(
                window=window,
                components=snapshot_components(state),
            )
            applied = apply_edits(state, edits)
            state["_ch_last_edits"] = edits.to_trace_dict()
            state["_ch_last_applied"] = applied
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("CH refiner failed: %s", e)
        state["_ch_refined_this_turn"] = True
    else:
        state["_ch_refined_this_turn"] = False
        state.pop("_ch_last_edits", None)


def _ch_subagents_hook(env_self, state, prefix_parts):
    """Variant CH: prepend any sub-agent directive whose trigger fires (cap 3)."""
    if not state.get("_ch_subagents"):
        return
    from nethack_harness.refiner import trigger_fires
    fired = []
    for name, spec in state["_ch_subagents"].items():
        if trigger_fires(spec.get("trigger", ""), state.get("structured_obs")):
            fired.append(f"[subagent:{name}] {spec.get('text', '')}")
            if len(fired) >= 3:
                break
    if fired:
        prefix_parts.extend(fired)


def _ch_history_inject(env_self, messages, state):
    """Variant CH: append the Refiner addendum + macro list to the system msg."""
    return _ch_inject_system(messages, state)


# The refiner-machinery bundle that the CH variant wires onto its spec: the
# per-turn refiner + sub-agent hooks, the system-inject history transform, and
# the run_macro tool adapter. Factored out so it can be attached to ANY obs
# variant (e.g. JSON) when refine=True, decoupling the teacher-refiner from the
# CH observation format.
_CH_REFINER_TURN_HOOKS = (_ch_refiner_hook, _ch_subagents_hook)
_CH_REFINER_HISTORY_TRANSFORMS = (_ch_history_inject,)
_CH_REFINER_EXTRA_TOOLS = (_make_run_macro_adapter,)


def attach_refiner(spec: "PromptSpec") -> "PromptSpec":
    """Return ``spec`` with the CH teacher-refiner machinery attached.

    Adds the refiner + sub-agent turn-hooks, the ``_ch_history_inject`` system
    addendum, and the ``run_macro`` tool — the exact bundle the CH variant uses
    — onto whatever obs/template ``spec`` already carries. Idempotent: hooks /
    tools already present (e.g. the CH spec built them in directly) are not
    duplicated.
    """
    turn_hooks = tuple(spec.turn_hooks) + tuple(
        h for h in _CH_REFINER_TURN_HOOKS if h not in spec.turn_hooks
    )
    history_transforms = tuple(spec.history_transforms) + tuple(
        t for t in _CH_REFINER_HISTORY_TRANSFORMS if t not in spec.history_transforms
    )
    existing_tools = tuple(spec.tools.extra_tools)
    extra_tools = existing_tools + tuple(
        mk for mk in _CH_REFINER_EXTRA_TOOLS if mk not in existing_tools
    )
    new_tools = dataclasses.replace(spec.tools, extra_tools=extra_tools)
    return dataclasses.replace(
        spec,
        turn_hooks=turn_hooks,
        history_transforms=history_transforms,
        tools=new_tools,
    )


# ---------- the registry ----------

_CANONICAL_OBS = ObsSpec(mode="ascii")
_FULL_TOOLS = ToolSpec()


def _spec(name, *, system_prompt, obs=_CANONICAL_OBS, turn_template=_canonical_template,
          tools=_FULL_TOOLS, turn_hooks=(), history_transforms=()):
    return build_prompt(
        name=name, system_prompt=system_prompt, obs=obs,
        turn_template=turn_template, tools=tools,
        turn_hooks=turn_hooks, history_transforms=history_transforms,
    )


def _build_registry(system_prompt: str) -> dict:
    """Build the variant→PromptSpec table against a given system prompt."""
    canonical = lambda name, **kw: _spec(name, system_prompt=system_prompt, **kw)  # noqa: E731
    return {
        # Default + close cousins: canonical render, full tools.
        "B1": canonical("B1"),
        "B0": canonical("B0"),
        "N": canonical("N"),
        # BALROG: natural-language scene, no ASCII grid.
        "B": canonical("B", turn_template=_formatter_template(_format_obs_balrog)),
        # BALROG's *actual* leaderboard observation carries BOTH a
        # natural-language scene description AND the raw map — its run CSVs show
        # "horizontal wall near north" alongside the 80x24 grid. Variant B drops
        # the grid on purpose (testing whether the map earns its tokens), so
        # neither B nor B0 alone reproduces what their Gemini-3-Flash run saw.
        # These two cells do: description + map, over ASCII and over JSON.
        "B_ASCII": canonical("B_ASCII", turn_template=_balrog_plus_map_template("ascii")),
        "B_JSON": canonical("B_JSON", turn_template=_balrog_plus_map_template("json")),
        # The real `nle-language-wrapper` text — the observation BALROG's own
        # agent consumed, not our reconstruction of it. Pair with
        # skill_set="balrog80" for a literal action+observation replication.
        "NLE_LANG": canonical("NLE_LANG", turn_template=_nle_language_template()),
        # Glyphbox: canonical render, paired with interface=code by the caller.
        "G": canonical("G", turn_template=_formatter_template(_format_obs_glyphbox)),
        # Experiment 1 baseline encodings — FAITHFUL ports of the prior
        # frameworks' *native* observation text (NOT the G/N approximations
        # above). The turn_template raises NotImplementedError until the port
        # lands (see rendering._format_obs_netplay / _format_obs_glyphbox_native),
        # so these variants register and route apples-to-apples through
        # prime_runner but fail loudly rather than silently substituting our own
        # render. See docs/experiments/exp1_encoding_ablations.md §(e).
        "NETPLAY": canonical("NETPLAY", turn_template=_formatter_template(_format_obs_netplay)),
        "GLYPHBOX": canonical("GLYPHBOX", turn_template=_formatter_template(_format_obs_glyphbox_native)),
        # Summarize-and-reset: canonical-equivalent render; the drop is driven by
        # the orthogonal summarize_and_reset kwarg, not the variant.
        "R": canonical("R", turn_template=_formatter_template(_format_obs_summarize_reset)),
        # Descent-salience block (ND, FD).
        "ND": canonical("ND", obs=ObsSpec(setup_flags={"_descent_salient": True})),
        "FD": canonical("FD", obs=ObsSpec(setup_flags={"_descent_salient": True})),
        # E1 frontier-surface blocks.
        "E1": canonical("E1", obs=ObsSpec(setup_flags={"_e1_obs": True})),
        # E2 paint frontiers onto the map.
        "E2": canonical("E2", obs=ObsSpec(setup_flags={"_e2_obs": True})),
        # Image observation: rendered tiles (IMG) or tty raster (IMG_TTY).
        "IMG": canonical("IMG", obs=ObsSpec(mode="img"),
                         turn_template=_image_template("glyph")),
        "IMG_TTY": canonical("IMG_TTY", obs=ObsSpec(mode="img"),
                             turn_template=_image_template("tty")),
        # Structured-text map: JSON or in-repo TOON, built from the canonical
        # map model; map_detail (full/minimal) rides on state["map_detail"].
        "JSON": canonical("JSON", turn_template=_structured_map_template("json")),
        "TOON": canonical("TOON", turn_template=_structured_map_template("toon")),
        # Sub-experiment 1d (observation DELIVERY): delayed / on-demand map.
        # The FULL map is re-sent only on a material change or an explicit
        # request_map/reveal; otherwise a placeholder stands in. Action feedback
        # is unchanged every turn. Composes with the ASCII (DM) and JSON
        # (DM_JSON) encodings.
        "DM": canonical("DM", turn_template=_delayed_map_template("ascii"),
                        obs=ObsSpec(setup_flags={"_delayed_map": True})),
        "DM_JSON": canonical("DM_JSON", turn_template=_delayed_map_template("json"),
                             obs=ObsSpec(setup_flags={"_delayed_map": True})),
        # Combined cell: language description (B_ASCII) + delayed map (DM).
        # Run it with belief_state_interval=0 to add the third winner,
        # journal-only memory. `_delayed_map` must stay set — it is what exposes
        # the `request_map` tool the agent needs to refresh on demand.
        "DM_B_ASCII": canonical("DM_B_ASCII",
                                turn_template=_balrog_delayed_map_template(),
                                obs=ObsSpec(setup_flags={"_delayed_map": True})),
        # Bounding-box on-demand: the map is hidden; the agent views regions via
        # reveal(x1,y1,x2,y2), which returns an ASCII crop as tool feedback.
        "BBOX": canonical("BBOX", turn_template=_bbox_template,
                          obs=ObsSpec(setup_flags={"_bbox_map": True})),
        # BBOX with the quiet-turn push stripped to feedback + STATUS; the
        # entity/message sections arrive only on `reveal` turns. See
        # _bbox_min_template.
        "BBOX_MIN": canonical("BBOX_MIN", turn_template=_bbox_min_template,
                              obs=ObsSpec(setup_flags={"_bbox_map": True})),
        # BBOX_MIN with the bought look upgraded: request_map turns render the
        # full section set PLUS the structured JSON map body. Quiet turns are
        # byte-identical to BBOX_MIN. See _bbox_min_odjson_template.
        "BBOX_BJSON_OD": canonical("BBOX_BJSON_OD",
                                   turn_template=_bbox_min_odjson_template,
                                   obs=ObsSpec(setup_flags={"_bbox_map": True})),
        # BBOX_MIN + the adaptive objective hint in LAG (explore) mode: every
        # observation opens by naming the BALROG axis that is BEHIND. Chosen
        # for exp4 as the metric-side answer to the exp3b death pattern -- 10
        # of 15 deaths at XL 1, diving with a level-1 character -- because
        # under LAG the hint keeps pointing at experience until XL catches up
        # with depth, which is exactly the survival discipline a hand-written
        # prompt would have tried to impose, minus the hand-writing.
        "BBOX_MIN_GUIDE_LAG": canonical(
            "BBOX_MIN_GUIDE_LAG",
            turn_template=_guided(_bbox_min_template, "lag"),
            obs=ObsSpec(setup_flags={"_bbox_map": True})),
        # Entity-only map: terrain dropped, and ADJACENT / VISIBLE FEATURES /
        # VISIBLE MONSTERS folded into it so there is exactly one place state
        # lives. SPARSE always shows it; SPARSE_ONDEMAND withholds it behind
        # `reveal`, which is the first honest test of delivery timing.
        "SPARSE": canonical("SPARSE", turn_template=_sparse_template),
        # seen-vs-remembered axis: identical to their partners except that the
        # `_remember_monsters` flag adds lapsed sightings (species, last
        # position, age in game turns, dropped on kill and on descent).
        "SPARSE_MEM": canonical("SPARSE_MEM", turn_template=_sparse_template,
                                obs=ObsSpec(setup_flags={"_remember_monsters": True})),
        "BBOX_MEM": canonical("BBOX_MEM", turn_template=_bbox_template,
                              obs=ObsSpec(setup_flags={"_bbox_map": True,
                                                       "_remember_monsters": True})),
        "SPARSE_ONDEMAND": canonical("SPARSE_ONDEMAND",
                                     turn_template=_sparse_ondemand_template,
                                     obs=ObsSpec(setup_flags={"_bbox_map": True})),
        # Adaptive objective hint (see objective_hint.py). BALROG progression is
        # max(depth_value, xp_value), so effort on the trailing axis scores
        # nothing until it overtakes -- these arms tell the agent which axis is
        # actually binding and what the next checkpoint is worth. Paired 1:1
        # with the un-hinted variants for a clean ablation.
        # LEAD = exploit: push the axis already binding the max().
        # LAG  = explore: push the trailing axis. Scores nothing now, but
        # experience is survival currency and death ends accumulation, so it may
        # raise the eventual max. 2x2 against the two best encodings.
        "BBOX_GUIDE_LEAD": canonical("BBOX_GUIDE_LEAD",
                                     turn_template=_guided(_bbox_template, "lead"),
                                     obs=ObsSpec(setup_flags={"_bbox_map": True})),
        "BBOX_GUIDE_LAG": canonical("BBOX_GUIDE_LAG",
                                    turn_template=_guided(_bbox_template, "lag"),
                                    obs=ObsSpec(setup_flags={"_bbox_map": True})),
        "B0_GUIDE_LEAD": canonical("B0_GUIDE_LEAD",
                                   turn_template=_guided(_canonical_template, "lead")),
        "B0_GUIDE_LAG": canonical("B0_GUIDE_LAG",
                                  turn_template=_guided(_canonical_template, "lag")),
        # JSON body (player + entities, structured and addressable) with the
        # per-tile `cells` array WITHHELD; the agent pulls map regions via
        # reveal(x1,y1,x2,y2). Measured motivation: JSON inline costs ~3,400
        # tok/turn (~4,300 with all 1b layers) against B0's ~690, while BBOX
        # delivery runs ~480 because `reveal` fires on only 1-2% of turns. This
        # cell asks whether JSON's structure is worth having once you stop
        # paying for it every turn — and re-tests 1b's null result, which may
        # have been an attention problem at 12-15k chars rather than the layers
        # carrying no information.
        "BBOX_JSON": canonical("BBOX_JSON", turn_template=_bbox_json_template,
                               obs=ObsSpec(setup_flags={"_bbox_map": True})),
        # Continual-harness adaptation: periodic self-refinement directive.
        "P": canonical("P", turn_hooks=(_p_refinement_hook,)),
        # Full Continual Harness: refiner + sub-agents + system inject + run_macro.
        # Built by attaching the refiner bundle to a canonical (ASCII) spec, so
        # CH and JSON+refine share one wiring path.
        "CH": attach_refiner(canonical("CH")),
    }


VARIANT_REGISTRY = _build_registry(SYSTEM_PROMPT)


def resolve_spec(variant: str, system_prompt: str) -> PromptSpec:
    """Return the PromptSpec for ``variant`` with ``system_prompt`` injected.

    ``system_prompt`` is passed explicitly (rather than read from a module
    global) so callers can hand in the value *after* the NETHACK_HARNESS overlay
    has mutated it — keeping the overlay seam intact.
    """
    base = VARIANT_REGISTRY.get(variant) or VARIANT_REGISTRY["B1"]
    return dataclasses.replace(base, system_prompt=system_prompt)
