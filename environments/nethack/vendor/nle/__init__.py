"""Compatibility shim package standing in for the real ``nle`` distribution.

Only what the vendored NetPlay skill layer imports is provided:
``nle.nethack`` (constants + glyph arithmetic) and ``nle.env.NLE`` (a type name
used in an annotation). See vendor/PROVENANCE.md.

This directory is appended to ``sys.path`` (never prepended), so a real ``nle``
installation would take precedence if one were ever added.
"""
