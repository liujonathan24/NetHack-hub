"""Minimal stand-in for the ``nle-language-wrapper`` package.

Vendored ``netplay/nethack_agent/tracking.py`` uses exactly one thing from it:
``NLELanguageObsv().text_message(tty_chars)``, to turn the terminal's top line
into the current game message for GameMessageEvent. Upstream's implementation is
a C extension; ours reproduces that single function in Python.
"""

from .nle_language_obsv import NLELanguageObsv  # noqa: F401
