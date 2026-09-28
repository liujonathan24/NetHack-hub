# Vendored from NetPlay @ 6acb90d865411f28d440e042372d9034f971e54a
#   netplay/nethack_utils/nle_wrapper.py
#
# ADAPTED (see vendor/PROVENANCE.md). Upstream this file holds two things:
#   1. RawKeyPress            -- reproduced below BYTE-FOR-BYTE (upstream 12-134)
#   2. render_ascii_map + NethackGymnasiumWrapper
#                             -- REMOVED (upstream 136-273)
#
# (2) is precisely the NLE/MiniHack engine seam this port replaces: it
# constructs `nle.env.base.NLE`/`minihack.MiniHack`, sets NLE's option string,
# and adapts NLE's 4-tuple step to gymnasium's 5-tuple. We drive our own fork
# through nethack_core instead (see nethack_harness/tools/netplay_true.py),
# so keeping it would only add unusable code and hard deps on nle, minihack and
# PIL. Nothing in the skill layer imports it -- skills.py imports only
# RawKeyPress from this module.
#
# Only the now-unused sys/numpy/PIL/gymnasium/nle/minihack imports were
# dropped from upstream. RawKeyPress itself was then EXTENDED (marked
# "ADDED (fix/netcore-input)" below); every upstream member and every input
# upstream's parse() accepted still maps to the same key code:
#   * members for the printable ASCII keys upstream omitted (- | { } ] ~ '),
#     TAB, BACKSPACE, LINEFEED and Ctrl+A..Ctrl+Z. Upstream's enum made
#     `np_press_key --key -` fail, so the fingers answer to "What do you want
#     to write with? [- ...]" (and #force's `|`, the `{`/`}`/`]` commands,
#     ...) was unreachable.
#   * parse() also accepts "tab" / "backspace" by name.
#   * parse_sequence(): a string of keys with <NAME> specials, used by the
#     multi-key press_key skill.

import enum

# ADDED (fix/netcore-input): names usable inside angle brackets in
# RawKeyPress.parse_sequence, e.g. "E-Elbereth<ENTER>". Module-level because
# a dict inside the IntEnum body would become an enum member.
_SEQUENCE_NAMES = {
    "esc": 27,
    "enter": 13,
    "space": 32,
    "tab": 9,
    "backspace": 8,
    "lt": ord("<"),
    "gt": ord(">"),
}

class RawKeyPress(enum.IntEnum):
    """
    Enumeration class (hopefully) representing all key presses used in NetHack.
    Includes uppercase letters, many symbols, numbers, and keys like Space and Escape.

    Usage:
        keypress = RawKeyPress.parse("ESC")
        keypress = RawKeyPress.parse("A")
    """
    # Uppercase Letters
    KEYPRESS_A = ord("A")
    KEYPRESS_B = ord("B")
    KEYPRESS_C = ord("C")
    KEYPRESS_D = ord("D")
    KEYPRESS_E = ord("E")
    KEYPRESS_F = ord("F")
    KEYPRESS_G = ord("G")
    KEYPRESS_H = ord("H")
    KEYPRESS_I = ord("I")
    KEYPRESS_J = ord("J")
    KEYPRESS_K = ord("K")
    KEYPRESS_L = ord("L")
    KEYPRESS_M = ord("M")
    KEYPRESS_N = ord("N")
    KEYPRESS_O = ord("O")
    KEYPRESS_P = ord("P")
    KEYPRESS_Q = ord("Q")
    KEYPRESS_R = ord("R")
    KEYPRESS_S = ord("S")
    KEYPRESS_T = ord("T")
    KEYPRESS_U = ord("U")
    KEYPRESS_V = ord("V")
    KEYPRESS_W = ord("W")
    KEYPRESS_X = ord("X")
    KEYPRESS_Y = ord("Y")
    KEYPRESS_Z = ord("Z")

    # Lowercase Letters
    KEYPRESS_a = ord("a")
    KEYPRESS_b = ord("b")
    KEYPRESS_c = ord("c")
    KEYPRESS_d = ord("d")
    KEYPRESS_e = ord("e")
    KEYPRESS_f = ord("f")
    KEYPRESS_g = ord("g")
    KEYPRESS_h = ord("h")
    KEYPRESS_i = ord("i")
    KEYPRESS_j = ord("j")
    KEYPRESS_k = ord("k")
    KEYPRESS_l = ord("l")
    KEYPRESS_m = ord("m")
    KEYPRESS_n = ord("n")
    KEYPRESS_o = ord("o")
    KEYPRESS_p = ord("p")
    KEYPRESS_q = ord("q")
    KEYPRESS_r = ord("r")
    KEYPRESS_s = ord("s")
    KEYPRESS_t = ord("t")
    KEYPRESS_u = ord("u")
    KEYPRESS_v = ord("v")
    KEYPRESS_w = ord("w")
    KEYPRESS_x = ord("x")
    KEYPRESS_y = ord("y")
    KEYPRESS_z = ord("z")

    # Numbers Letters
    KEYPRESS_0 = ord("0")
    KEYPRESS_1 = ord("1")
    KEYPRESS_2 = ord("2")
    KEYPRESS_3 = ord("3")
    KEYPRESS_4 = ord("4")
    KEYPRESS_5 = ord("5")
    KEYPRESS_6 = ord("6")
    KEYPRESS_7 = ord("7")
    KEYPRESS_8 = ord("8")
    KEYPRESS_9 = ord("9")

    # Symbols
    KEYPRESS_NUMBER_SIGN = ord("#")
    KEYPRESS_SEMICOLON = ord(";")
    KEYPRESS_DOUBLECOLON = ord(":")
    KEYPRESS_DOT = ord(".")
    KEYPRESS_COMMA = ord(",")
    KEYPRESS_SMALLER = ord("<")
    KEYPRESS_GREATER = ord(">")
    KEYPRESS_FORWARD_SLASH = ord("/")
    KEYPRESS_BACKWARD_SLASH = ord("\\")
    KEYPRESS_CARET = ord("^")
    KEYPRESS_OPEN_BRACKET = ord("(")
    KEYPRESS_CLOSE_BRACKET = ord(")")
    KEYPRESS_OPEN_SQUARE_BRACKET = ord("[")
    KEYPRESS_EQUALS = ord("=")
    KEYPRESS_STAR = ord("*")
    KEYPRESS_DOLLAR = ord("$")
    KEYPRESS_PLUS = ord("+")
    KEYPRESS_AT_SIGN = ord("@")
    KEYPRESS_QUESTION_MARK = ord("?")
    KEYPRESS_AMPERSAND = ord("&")
    KEYPRESS_EXCLAMATION_MARK = ord("!")
    KEYPRESS_UNDERSCORE = ord("_")
    KEYPRESS_DOUBLE_QUOTATION_MARK = ord("\"")
    KEYPRESS_BACKTICK = ord("`")
    KEYPRESS_PERCENT = ord("%")

    # Special Keys
    KEYPRESS_ENTER = 13
    KEYPRESS_ESC = 27
    KEYPRESS_SPACE = 32

    # ADDED (fix/netcore-input): the printable ASCII keys upstream left out,
    # so that every printable key can be pressed.
    KEYPRESS_MINUS = ord("-")
    KEYPRESS_PIPE = ord("|")
    KEYPRESS_OPEN_CURLY_BRACKET = ord("{")
    KEYPRESS_CLOSE_CURLY_BRACKET = ord("}")
    KEYPRESS_CLOSE_SQUARE_BRACKET = ord("]")
    KEYPRESS_TILDE = ord("~")
    KEYPRESS_APOSTROPHE = ord("'")

    # ADDED (fix/netcore-input): control keys. Ctrl+I, Ctrl+J, Ctrl+M are the
    # same codes as TAB, LINEFEED and ENTER, so IntEnum makes them aliases.
    KEYPRESS_BACKSPACE = 8
    KEYPRESS_TAB = 9
    KEYPRESS_LINEFEED = 10
    KEYPRESS_CTRL_A = 1
    KEYPRESS_CTRL_B = 2
    KEYPRESS_CTRL_C = 3
    KEYPRESS_CTRL_D = 4
    KEYPRESS_CTRL_E = 5
    KEYPRESS_CTRL_F = 6
    KEYPRESS_CTRL_G = 7
    KEYPRESS_CTRL_H = 8
    KEYPRESS_CTRL_I = 9
    KEYPRESS_CTRL_J = 10
    KEYPRESS_CTRL_K = 11
    KEYPRESS_CTRL_L = 12
    KEYPRESS_CTRL_M = 13
    KEYPRESS_CTRL_N = 14
    KEYPRESS_CTRL_O = 15
    KEYPRESS_CTRL_P = 16
    KEYPRESS_CTRL_Q = 17
    KEYPRESS_CTRL_R = 18
    KEYPRESS_CTRL_S = 19
    KEYPRESS_CTRL_T = 20
    KEYPRESS_CTRL_U = 21
    KEYPRESS_CTRL_V = 22
    KEYPRESS_CTRL_W = 23
    KEYPRESS_CTRL_X = 24
    KEYPRESS_CTRL_Y = 25
    KEYPRESS_CTRL_Z = 26

    @staticmethod
    def parse(key: str) -> "RawKeyPress":
        special_keys = {
            "enter": RawKeyPress.KEYPRESS_ENTER,
            "space": RawKeyPress.KEYPRESS_SPACE,
            "esc": RawKeyPress.KEYPRESS_ESC,
            # ADDED (fix/netcore-input)
            "tab": RawKeyPress.KEYPRESS_TAB,
            "backspace": RawKeyPress.KEYPRESS_BACKSPACE,
        }

        if len(key) == 1:
            return RawKeyPress(ord(key))
        elif key.lower() in special_keys:
            return special_keys[key.lower()]
        
        raise ValueError(f"Cannot parse the given key {key}.")

    # ADDED (fix/netcore-input): multi-key parsing for the press_key skill.
    @staticmethod
    def parse_sequence(key: str) -> "list[RawKeyPress]":
        """Parse a press_key argument into the keys to send, in order.

        * A single character is that key (so "<", ">" and "," alone work).
        * ESC, ENTER, SPACE, TAB or BACKSPACE alone (any case) is that key.
        * Anything else is a sequence: each character is one key, and
          <ESC> <ENTER> <SPACE> <TAB> <BACKSPACE> <LT> <GT> <C-x> (Ctrl+x)
          in angle brackets, any case, are special keys. A "<" that does not
          start one of those is the literal "<" key; <NAME> with an unknown
          NAME is an error.

        Raises ValueError (nothing is sent) on an empty string, an unknown
        <NAME>, or a character that is not an ASCII key.
        """
        import re
        if not isinstance(key, str) or key == "":
            raise ValueError("No key given.")
        if len(key) == 1:
            return [RawKeyPress._from_char(key)]
        if key.strip().lower() in ("esc", "enter", "space", "tab", "backspace"):
            return [RawKeyPress.parse(key.strip().lower())]

        keys = []
        i = 0
        while i < len(key):
            ch = key[i]
            if ch == "<":
                m = re.match(r"<([A-Za-z][A-Za-z0-9_-]*|[Cc]-.)>", key[i:])
                if m:
                    name = m.group(1)
                    low = name.lower()
                    if low.startswith("c-") and len(name) == 3:
                        letter = name[2].lower()
                        if not ("a" <= letter <= "z"):
                            raise ValueError(f"Unknown special key <{name}>: Ctrl works with letters, e.g. <C-d>.")
                        keys.append(RawKeyPress(ord(letter) & 0x1F))
                    elif low in _SEQUENCE_NAMES:
                        keys.append(RawKeyPress(_SEQUENCE_NAMES[low]))
                    else:
                        raise ValueError(
                            f"Unknown special key <{name}>. Known: <ESC> <ENTER> <SPACE> "
                            f"<TAB> <BACKSPACE> <LT> <GT> <C-x>.")
                    i += m.end()
                    continue
            keys.append(RawKeyPress._from_char(ch))
            i += 1
        return keys

    @staticmethod
    def _from_char(ch: str) -> "RawKeyPress":
        code = ord(ch)
        try:
            return RawKeyPress(code)
        except ValueError:
            raise ValueError(f"Cannot press {ch!r}: not a key on the NetHack keyboard.") from None
