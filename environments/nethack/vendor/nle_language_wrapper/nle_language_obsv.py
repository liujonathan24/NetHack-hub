"""Python stand-in for ``nle_language_wrapper.nle_language_obsv``.

Only ``text_message`` is used by the vendored NetPlay code. Upstream's C
implementation reads the NetHack message from the top line(s) of the terminal
and returns it as bytes; NetPlay then does ``.decode("latin-1")``.

We reproduce that: read tty row 0, strip the "--More--" marker, and return the
trimmed text as bytes. Reading the tty (rather than the engine's ``message``
buffer) is deliberate -- it matches the data source upstream used, so event text
matches what NetPlay's agent would have seen.
"""

import numpy as np


class NLELanguageObsv:
    def text_message(self, tty_chars) -> bytes:
        rows = np.asarray(tty_chars)
        if rows.ndim == 1:
            rows = rows[None, :]
        line = "".join(chr(int(c)) for c in rows[0])
        line = line.replace("--More--", " ")
        return line.strip().encode("latin-1")
