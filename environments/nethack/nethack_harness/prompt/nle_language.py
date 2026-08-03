"""BALROG's *actual* observation: `nle_language_wrapper`'s natural-language text.

Variant ``B`` is our own reimplementation of a BALROG-style scene description,
written from their published CSVs. This module is the real thing -- ngoodger's
`nle-language-wrapper`, the package BALROG itself wraps NLE with -- so an
``NLE_LANG`` cell can claim "same observation as the paper" literally rather
than by approximation.

Why a subprocess: the converter is a pybind11 extension linking stock NLE's
`libnethack.so` for its glyph->name tables, built for Python 3.10. Our engine
venv is 3.12 with the NetHack fork on PYTHONPATH. Installing stock NLE beside
the fork would put two NetHack implementations in one interpreter, so the
converter stays behind a pipe (`nle_language_worker.py`). Cost is a JSON
round-trip per turn -- sub-millisecond against a ~5s LLM call.

Set ``NLE_LANG_PYTHON`` to the interpreter that has the wrapper installed. The
default points at the build made for this experiment; when the wrapper is
packaged for 3.12 this whole module collapses into a direct import.

Glyph-numbering caveat, load-bearing: the converter maps *stock NLE* glyph ids
to names. If the fork ever renumbers glyphs the text silently becomes wrong
rather than failing, so `selftest.py --check-language-parity` asserts a known
scene end-to-end. Run it after any engine change that touches glyph ids.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

_DEFAULT_PYTHON = "/home/jl0796/.claude/jobs/6389f3db/tmp/nlw-venv/bin/python"
_WORKER = str(Path(__file__).with_name("nle_language_worker.py"))

# One worker per process, guarded: a rollout renders from a single thread, but
# the v1 toolset dispatches tool calls concurrently across rollouts sharing an
# interpreter, and two writers on one stdin pipe interleave into garbage.
_lock = threading.Lock()
_proc: subprocess.Popen | None = None


class LanguageWrapperUnavailable(RuntimeError):
    """The converter could not be started. Raised rather than silently degrading:
    an NLE_LANG cell that quietly fell back to a different observation would be
    an invalid experiment, not a degraded one."""


def _start() -> subprocess.Popen:
    python = os.environ.get("NLE_LANG_PYTHON", _DEFAULT_PYTHON)
    if not os.path.exists(python):
        raise LanguageWrapperUnavailable(
            f"NLE_LANG_PYTHON={python!r} does not exist. Point it at an interpreter "
            "with `nle-language-wrapper` installed."
        )
    proc = subprocess.Popen(
        [python, _WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    hello = proc.stdout.readline()
    if not hello or not json.loads(hello).get("ready"):
        raise LanguageWrapperUnavailable(f"worker failed to start under {python!r}")
    return proc


def _worker() -> subprocess.Popen:
    global _proc
    if _proc is None or _proc.poll() is not None:
        _proc = _start()
    return _proc


def language_obs(raw_obs) -> dict:
    """Return the wrapper's five text channels for one observation.

    Raises `LanguageWrapperUnavailable` if the converter cannot run -- never
    substitutes a different rendering.
    """
    import numpy as np

    def arr(name):
        v = getattr(raw_obs, name, None)
        if v is None:
            raise LanguageWrapperUnavailable(f"observation has no {name!r}")
        return np.asarray(v).tolist()

    req = {
        "glyphs": arr("glyphs"),
        "blstats": arr("blstats"),
        "tty_chars": arr("tty_chars"),
        "tty_cursor": arr("tty_cursor"),
        "inv_strs": arr("inv_strs"),
        "inv_letters": arr("inv_letters"),
    }
    with _lock:
        proc = _worker()
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
    if not line:
        raise LanguageWrapperUnavailable("converter died mid-rollout")
    out = json.loads(line)
    if not out.get("ok"):
        raise LanguageWrapperUnavailable(out.get("error", "unknown converter error"))
    return out


def render_language_view(raw_obs) -> str:
    """The observation block for an `NLE_LANG` turn, in the wrapper's own order.

    Channel order and headings mirror `NLELanguageWrapper`'s own dict so the
    text a model sees matches what BALROG's agent saw, not a re-layout of it.
    """
    o = language_obs(raw_obs)
    parts = []
    for key, heading in (
        ("text_message", "message"),
        ("text_blstats", "stats"),
        ("text_inventory", "inventory"),
        ("text_cursor", "cursor"),
        ("text_glyphs", "surroundings"),
    ):
        body = (o.get(key) or "").strip()
        if body:
            parts.append(f"{heading}:\n{body}")
    return "\n\n".join(parts)


def shutdown() -> None:
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            try:
                _proc.stdin.close()
                _proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - best-effort teardown
                _proc.kill()
        _proc = None
