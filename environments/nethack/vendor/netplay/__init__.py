"""Vendored NetPlay skill layer (github.com/CommanderCero/NetPlay, MIT).

ADAPTED. Upstream's ``netplay/__init__.py`` is a single ``create_llm_agent()``
factory that wires the skill repository together with upstream's LLM plumbing
(langchain + openai), their gradio renderer, and their NLE/MiniHack env. We
reuse only the skill layer, so that factory is not reproduced here -- it would
drag in four heavy dependencies we do not run.

The one part of it that defines the *action surface*, upstream lines 9-18, is
preserved verbatim in ``nethack_harness/tools/netplay_true.py`` as
``NETPLAY_SKILL_REPOSITORY``: the same eight entries, in the same order.

Importing this package requires this directory's parent (``vendor/``) to be on
sys.path so that ``netplay.*``, ``nle.*``, ``nle_language_wrapper`` and
``gradio`` all resolve. ``nethack_harness.tools.netplay_true`` does that.
"""
