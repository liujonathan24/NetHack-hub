"""Minimal stand-in for ``gradio``.

Vendored ``netplay/core/agent_base.py`` is kept byte-verbatim, and it does
``import gradio`` and annotates ``AgentRenderer.init`` with
``List[gradio.components.Component]`` -- evaluated at class-definition time, so
the attribute must exist at import. Upstream's web UI is not part of the skill
layer we run, so this provides the name and nothing else.
"""

from . import components  # noqa: F401
