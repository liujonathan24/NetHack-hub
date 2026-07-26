"""Shim for ``netplay.logging``.

Upstream's ``netplay/logging/`` writes JSONL step logs and renders an mp4 via
moviepy/PIL. Our harness does its own tracing, and the video renderer pulls in
heavy optional deps, so these are inert stand-ins. ``NethackBaseAgent`` calls
them on every step, so they must exist -- but they carry no skill behaviour.
"""
