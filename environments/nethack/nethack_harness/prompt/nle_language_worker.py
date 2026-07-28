"""Out-of-process worker exposing `nle_language_wrapper`'s C++ converter.

Runs under the wrapper's OWN interpreter (Python 3.10 + stock NLE 0.9.0), not
ours. `nle_language_obsv` links `libnethack.so` from stock NLE for its
glyph->name tables; our engine venv is Python 3.12 with the NetHack fork on
PYTHONPATH, and installing stock NLE beside the fork would put two NetHack
implementations in one interpreter. Keeping the converter behind a pipe is the
cheap way to avoid that entirely.

Protocol, one request per line on stdin, one response per line on stdout:

    -> {"glyphs": [[...]], "blstats": [...], "tty_chars": [[...]],
        "tty_cursor": [...], "inv_strs": [[...]], "inv_letters": [...]}
    <- {"ok": true, "text_glyphs": "...", "text_blstats": "...", ...}
    <- {"ok": false, "error": "..."}

Arrays arrive as nested lists (json), which costs a few hundred microseconds
per turn against a ~5s LLM call -- not worth a binary protocol.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv


def main() -> None:
    conv = NLELanguageObsv()
    # Signal readiness so the parent can fail fast on a broken venv instead of
    # blocking on the first real observation.
    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            glyphs = np.asarray(req["glyphs"], dtype=np.int16)
            blstats = np.asarray(req["blstats"], dtype=np.int64)
            tty_chars = np.asarray(req["tty_chars"], dtype=np.uint8)
            tty_cursor = np.asarray(req["tty_cursor"], dtype=np.int64)
            inv_strs = np.asarray(req["inv_strs"], dtype=np.uint8)
            inv_letters = np.asarray(req["inv_letters"], dtype=np.uint8)

            def _s(b) -> str:
                return b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b)

            out = {
                "ok": True,
                "text_glyphs": _s(conv.text_glyphs(glyphs, blstats)),
                "text_blstats": _s(conv.text_blstats(blstats)),
                "text_message": _s(conv.text_message(tty_chars)),
                "text_inventory": _s(conv.text_inventory(inv_strs, inv_letters)),
                "text_cursor": _s(conv.text_cursor(glyphs, blstats, tty_cursor)),
            }
        except Exception as exc:  # noqa: BLE001 - the parent decides what is fatal
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
