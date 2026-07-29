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


def _normalize_gemini_wire_tool_calls():
    """Fix Gemini's `default_api:` leakage in the bytes actually served to the CLI.

    SUPERSEDES the trace-level patch above, which was applied at the wrong layer.
    `_recover_gemini_text_tool_calls` rewrites the parsed `Response`, but the
    interception server hands the program `response.raw` -- the native provider
    body (`server.py:378`, `serve()`). So that patch made the TRACE look correct
    while Claude Code still received the malformed original and exited. Measured:
    every p2 rollout recorded exactly one `recovered_default_api` tool call as its
    final assistant message and died immediately after, at dlvl 1.

    `_completion_response` is the single chokepoint every served turn passes
    through, so normalizing the wire dict here reaches the program.

    Two distinct malformations, both observed:

      1. TEXT form -- a text block containing
         `call:default_api:mcp__nethack__bal_south{}`, with `stop_reason:
         "end_turn"`. Claude Code's --print mode exits on a tool-call-free turn,
         so one of these ends the rollout.
      2. PREFIXED NAME -- a real tool_use block whose `name` is
         `default_api:mcp__nethack__bal_search` (it prefixes Claude Code's own
         built-ins too: `default_api:Edit`). Claude Code answers
         `<tool_use_error>No such tool available`.

    Both are Gemini's internal function-calling convention leaking through the
    Anthropic-compatible endpoint. GLM never emits either (0 of 25 rollouts).

    Anything that does not match is passed through untouched, so the failure mode
    is the status quo rather than a fabricated action.
    """
    import json
    import re

    from verifiers.v1.interception import server as _server

    text_form = re.compile(
        r"call:\s*default_api\s*:\s*([A-Za-z0-9_]+)\s*(\{.*?\})\s*$", re.S
    )
    counter = {"n": 0}

    def _fix_anthropic(body: dict) -> dict:
        blocks = body.get("content")
        if not isinstance(blocks, list):
            return body
        changed = False
        out = []
        for block in blocks:
            if not isinstance(block, dict):
                out.append(block)
                continue
            # (2) strip the namespace prefix off a real tool_use block.
            if block.get("type") == "tool_use":
                name = block.get("name") or ""
                if name.startswith("default_api:"):
                    block = {**block, "name": name.split("default_api:", 1)[1]}
                    changed = True
                out.append(block)
                continue
            # (1) a tool call emitted as text.
            if block.get("type") == "text":
                match = text_form.search((block.get("text") or "").strip())
                if match:
                    try:
                        args = json.loads(match.group(2))
                    except ValueError:
                        out.append(block)
                        continue
                    counter["n"] += 1
                    out.append({
                        "type": "tool_use",
                        "id": f"vf_recovered_{counter['n']}",
                        "name": match.group(1),
                        "input": args,
                    })
                    changed = True
                    continue
            out.append(block)
        if not changed:
            return body
        body = {**body, "content": out}
        if any(b.get("type") == "tool_use" for b in out if isinstance(b, dict)):
            body["stop_reason"] = "tool_use"
        return body

    original = _server._completion_response

    def patched(completion):
        if isinstance(completion, dict):
            try:
                if "content" in completion:
                    completion = _fix_anthropic(completion)
            except Exception:
                pass  # never let normalization break a served turn
        return original(completion)

    _server._completion_response = patched


try:
    _normalize_gemini_wire_tool_calls()
except Exception:
    pass
