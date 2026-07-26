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
