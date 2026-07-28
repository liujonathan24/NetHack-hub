"""Interpreter-wide compatibility patches, applied before any user code runs.

Python's `site` module imports `sitecustomize` at startup for every interpreter
that has it on `sys.path`. Putting this directory first on PYTHONPATH therefore
patches the main process AND every worker subprocess vf-eval spawns — which is
why this lives here rather than in `nethack.py`: patching at env-import time
only covers processes that import the env, and response parsing does not always
happen in one of those.

Currently one patch:

`service_tier` — Prime Inference returns ``service_tier: "provisioned"`` on some
responses. The OpenAI SDK types the field as
``Literal["auto","default","flex","scale","priority"]``, so those responses fail
pydantic validation and the rollout dies with::

    ModelError -> 1 validation error for ChatCompletion
    service_tier  Input should be 'auto', ... [input_value='provisioned']

No released SDK accepts it (checked 2.38.0 and 2.48.0 — identical literal), and
it is intermittent: it depends on which capacity Prime routes the call to, which
is why only some seeds died. Widening the field to `str` changes nothing for
conforming values and stops the client discarding otherwise-valid completions.
"""
from __future__ import annotations


def _widen_service_tier() -> None:
    from typing import Optional

    import openai.types.chat as chat_types

    targets = []
    for name in dir(chat_types):
        obj = getattr(chat_types, name, None)
        fields = getattr(obj, "model_fields", None)
        if isinstance(fields, dict) and "service_tier" in fields:
            targets.append(obj)

    for model in targets:
        field = model.model_fields["service_tier"]
        try:
            import typing

            args = typing.get_args(field.annotation)
            inner = typing.get_args(args[0]) if args else ()
            if "provisioned" in inner:
                continue  # a future SDK learned the value
        except Exception:
            pass
        field.annotation = Optional[str]
        try:
            model.model_rebuild(force=True)
        except Exception:
            pass


try:
    _widen_service_tier()
except Exception:
    # A compatibility shim must never prevent the interpreter from starting.
    pass


def _recover_gemini_text_tool_calls():
    """Gemini sometimes emits a tool call as TEXT; recover it into a real one.

    Measured failure (p1, gemini-3-flash-preview, all three observations). The
    final assistant message of every Claude Code rollout carried no structured
    tool call and this content instead::

        call:default_api:mcp__nethack__bal_south{}
        call:default_api:mcp__nethack__bal_southwest{}
        call:default_api:mcp__nethack__bal_far_east{}

    `default_api:` is Gemini's own function-calling text convention leaking
    through the OpenAI/Anthropic-compatible endpoint. The intended action was
    valid every time -- it just arrived in the content field, so `stop_reason`
    was `end_turn` rather than `tool_use`.

    Why that is fatal for exactly one arm: Claude Code's `--print` mode exits
    the instant a turn arrives with no tool call, the process returns 0, and
    verifiers records `stop_condition="agent_completed"` (v1/harness.py:118).
    Prime Agent does not treat a tool-call-free turn as terminal -- measured:
    `b_ascii_flash/prime_agent` hit the SAME quirk once and still ran 1,440
    calls to a natural death at dlvl 4, while `b_ascii_flash/claude_code` died
    at 200. So this is not a model/harness incompatibility, it is one narrow
    interaction with a well-defined trigger.

    Correlation across every claude_code rollout on disk: 12/12 Gemini rollouts
    that ended `agent_completed` contain this string; 0 of 25 GLM rollouts do.

    Recovering in place rather than re-requesting: the model's intent is
    unambiguous and already present, and the malformed shape is a systematic
    serialization habit rather than a random glitch, so a re-request would
    likely reproduce it while costing another full-context call. If the text
    does NOT parse we leave the response untouched, so the failure mode is the
    status quo, never a fabricated action.
    """
    import json
    import re

    from verifiers.v1.dialects import anthropic as _anth
    from verifiers.v1.types import ToolCall

    pattern = re.compile(
        r"call:\s*default_api\s*:\s*([A-Za-z0-9_]+)\s*(\{.*?\})\s*$", re.S
    )
    original = _anth.response_from_wire

    def patched(message):
        response = original(message)
        if response.message.tool_calls:
            return response
        text = response.message.content
        if not isinstance(text, str) or "default_api" not in text:
            return response
        match = pattern.search(text.strip())
        if not match:
            return response
        try:
            arguments = json.dumps(json.loads(match.group(2)))
        except ValueError:
            return response
        response.message.tool_calls = [
            ToolCall(id="recovered_default_api", name=match.group(1), arguments=arguments)
        ]
        response.message.content = None
        response.finish_reason = "tool_calls"
        return response

    _anth.response_from_wire = patched


try:
    _recover_gemini_text_tool_calls()
except Exception:
    # A compatibility shim must never prevent the interpreter from starting.
    pass
