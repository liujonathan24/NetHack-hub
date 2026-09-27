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
# No behavioural line of RawKeyPress was changed; only the now-unused
# sys/numpy/PIL/gymnasium/nle/minihack imports were dropped.

import enum

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
    # Added 2026-08-21 (E10/E11 audit): keys NetHack's own prompts require that
    # the enum lacked. `-` is the critical one -- "select nothing / bare hands"
    # in wield/wear prompts and "your fingers" in the engrave prompt, i.e. the
    # only path to dust-engraving Elbereth; its absence made the model's `E`
    # then `-` attempt error out ("Unable to press the given key -"). The rest
    # complete text-entry prompts (naming, wishes).
    KEYPRESS_MINUS = ord("-")
    KEYPRESS_APOSTROPHE = ord("'")
    KEYPRESS_CLOSE_SQUARE_BRACKET = ord("]")
    KEYPRESS_OPEN_CURLY_BRACKET = ord("{")
    KEYPRESS_CLOSE_CURLY_BRACKET = ord("}")
    KEYPRESS_PIPE = ord("|")
    KEYPRESS_TILDE = ord("~")

    # Special Keys
    KEYPRESS_ENTER = 13
    KEYPRESS_ESC = 27
    KEYPRESS_SPACE = 32

    @staticmethod
    def parse(key: str) -> "RawKeyPress":
        special_keys = {
            "enter": RawKeyPress.KEYPRESS_ENTER,
            "space": RawKeyPress.KEYPRESS_SPACE,
            "esc": RawKeyPress.KEYPRESS_ESC
        }

        if len(key) == 1:
            member = RawKeyPress(ord(key))
            # netplay_telemetry gates the keys c1a0bec ADDED to this enum.
            # Enum members cannot be removed at runtime, so the gate lives at
            # the parse boundary: with the flag off, these raise exactly the
            # ValueError the baseline raised ("Unable to press the given key
            # -"), which is what made dust-engraving Elbereth impossible then.
            from nethack_harness import tool_flags as _flags
            if member in _POST_BASELINE_KEYS and not _flags.enabled("netplay_telemetry"):
                raise ValueError(f"Cannot parse the given key {key}.")
            return member
        elif key.lower() in special_keys:
            return special_keys[key.lower()]
        
        raise ValueError(f"Cannot parse the given key {key}.")


# The keys c1a0bec added to RawKeyPress. Named here rather than inline so the
# gate and the additions cannot drift apart: anything listed is unavailable
# unless `netplay_telemetry` is on.
_POST_BASELINE_KEYS = frozenset({
    RawKeyPress.KEYPRESS_MINUS,
    RawKeyPress.KEYPRESS_APOSTROPHE,
    RawKeyPress.KEYPRESS_CLOSE_SQUARE_BRACKET,
    RawKeyPress.KEYPRESS_OPEN_CURLY_BRACKET,
    RawKeyPress.KEYPRESS_CLOSE_CURLY_BRACKET,
    RawKeyPress.KEYPRESS_PIPE,
    RawKeyPress.KEYPRESS_TILDE,
})
