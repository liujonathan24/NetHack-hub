"""Shim for ``nle.env``.

NetPlay's ``skills.py`` does ``from nle.env import NLE`` but never references
``NLE`` in any code path -- it is a leftover import. We expose a placeholder so
the verbatim import line resolves without pulling in the NLE package.
"""


class NLE:  # pragma: no cover - never instantiated; annotation placeholder only
    """Placeholder for ``nle.env.NLE``; our engine seam is NetHackCoreEnv."""
