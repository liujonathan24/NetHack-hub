"""Generate the NetHack object table (otyp -> oc_class, oc_name, oc_descr).

NetPlay's vendored code calls ``nle.nethack.objclass(i).oc_class`` and
``nle.nethack.objdescr.from_idx(i)``. Our engine (``nethack_core``) exposes glyph
offsets and monster names but no object table, so we generate one directly from
the NetHack fork's ``objects.c`` -- the same source ``makedefs`` reads.

Each object is declared by exactly one class-implying macro, in ``otyp`` order.
The macro name determines ``oc_class`` unambiguously; raw ``OBJECT(...)`` entries
carry the class symbol as their 4th argument.

Run:
    python gen_object_table.py <path/to/objects.c> <path/to/onames.h> <out.py>
"""

import re
import sys

# Macro -> object class name, read off the macro definitions in objects.c.
# NOTE: the ROCK() macro expands to GEM_CLASS (rocks/flint are gems); genuine
# ROCK_CLASS objects (boulder, statue) are declared with raw OBJECT(...).
MACRO_CLASS = {
    "WEAPON": "WEAPON_CLASS",
    "PROJECTILE": "WEAPON_CLASS",
    "BOW": "WEAPON_CLASS",
    "WEPTOOL": "TOOL_CLASS",
    "TOOL": "TOOL_CLASS",
    "CONTAINER": "TOOL_CLASS",
    "ARMOR": "ARMOR_CLASS",
    "CLOAK": "ARMOR_CLASS",
    "HELM": "ARMOR_CLASS",
    "BOOTS": "ARMOR_CLASS",
    "GLOVES": "ARMOR_CLASS",
    "SHIELD": "ARMOR_CLASS",
    "DRGN_ARMR": "ARMOR_CLASS",
    "RING": "RING_CLASS",
    "AMULET": "AMULET_CLASS",
    "FOOD": "FOOD_CLASS",
    "POTION": "POTION_CLASS",
    "SCROLL": "SCROLL_CLASS",
    "SPELL": "SPBOOK_CLASS",
    "WAND": "WAND_CLASS",
    "COIN": "COIN_CLASS",
    "GEM": "GEM_CLASS",
    "ROCK": "GEM_CLASS",
}

# Must match nethack_core.glyphs.NUM_OBJECTS for the glyph arithmetic to line up.
EXPECTED_NUM_OBJECTS = 453

CLASS_VALUES = {
    "ILLOBJ_CLASS": 1,
    "WEAPON_CLASS": 2,
    "ARMOR_CLASS": 3,
    "RING_CLASS": 4,
    "AMULET_CLASS": 5,
    "TOOL_CLASS": 6,
    "FOOD_CLASS": 7,
    "POTION_CLASS": 8,
    "SCROLL_CLASS": 9,
    "SPBOOK_CLASS": 10,
    "WAND_CLASS": 11,
    "COIN_CLASS": 12,
    "GEM_CLASS": 13,
    "ROCK_CLASS": 14,
    "BALL_CLASS": 15,
    "CHAIN_CLASS": 16,
    "VENOM_CLASS": 17,
}

ENTRY_RE = re.compile(
    r"(?m)^(%s)\(" % "|".join(sorted(MACRO_CLASS, key=len, reverse=True) + ["OBJECT"])
)


def split_args(text):
    """Split a macro argument list on top-level commas."""
    args, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    args.append(cur.strip())
    return args


def parse_string(tok):
    """Parse a C string literal / None into a Python str or None."""
    tok = tok.strip()
    if tok == "None" or tok == "0":
        return None
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', tok)
    if not parts:
        return None
    return "".join(parts).replace('\\"', '"').replace("\\\\", "\\")


def main(objects_c, onames_h, out_path):
    src = open(objects_c, encoding="latin-1").read()

    # Strip comments first -- the file's header comment quotes the
    # "#define OBJECTS_PASS_2_" line, which would otherwise truncate everything.
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)

    # Only the first pass (before the recursive re-include) declares entries.
    cut = src.find("#define OBJECTS_PASS_2_")
    if cut != -1:
        src = src[:cut]

    # Join line continuations.
    src = re.sub(r"\\\n", " ", src)

    # Minimal preprocessor: objects.c guards a handful of entries behind
    # "#if 0 /* DEFERRED */" and "#ifdef MAIL". Neither is compiled into our
    # engine build, so those objects are absent from objects[] and must not be
    # counted -- otherwise every later otyp is shifted. All other directives
    # are simply dropped.
    EXCLUDED = (re.compile(r"^#\s*if\s+0\b"), re.compile(r"^#\s*ifdef\s+MAIL\b"))
    kept, skip_at = [], None
    depth = 0
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if re.match(r"^#\s*(if|ifdef|ifndef)\b", stripped):
                depth += 1
                if skip_at is None and any(p.match(stripped) for p in EXCLUDED):
                    skip_at = depth
            elif re.match(r"^#\s*endif\b", stripped):
                if skip_at is not None and depth == skip_at:
                    skip_at = None
                depth -= 1
            continue
        if skip_at is None:
            kept.append(line)
    src = "\n".join(kept)

    entries = []
    i = 0
    while i < len(src):
        m = ENTRY_RE.search(src, i)
        if m is None:
            break
        macro = m.group(1)
        start = m.end()
        depth = 1
        j = start
        while j < len(src) and depth:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
            j += 1
        body = src[start : j - 1]
        args = split_args(body)

        if macro == "OBJECT":
            # OBJECT(obj, bits, prp, sym, ...) -- obj is OBJ(name, desc)
            objarg = args[0]
            inner = objarg[objarg.find("(") + 1 : objarg.rfind(")")]
            nd = split_args(inner)
            name = parse_string(nd[0])
            descr = parse_string(nd[1]) if len(nd) > 1 else None
            cls = args[3].strip()
        else:
            name = parse_string(args[0])
            descr = parse_string(args[1]) if len(args) > 1 else None
            if macro in ("DRGN_ARMR", "FOOD", "COIN"):
                descr = None  # these macros pass None for desc
            cls = MACRO_CLASS[macro]

        if cls not in CLASS_VALUES:
            raise SystemExit(f"unknown class {cls!r} for macro {macro}")
        entries.append((name, descr, CLASS_VALUES[cls]))
        i = j

    # --- validation against onames.h -------------------------------------
    onames = {}
    for m in re.finditer(r"^#define\s+(\w+)\s+(\d+)\s*$", open(onames_h).read(), re.M):
        onames[m.group(1)] = int(m.group(2))

    # Drop the terminating OBJECT(NULL, NULL, ...) sentinel.
    if entries and entries[-1][0] is None:
        entries.pop()

    problems = []
    if len(entries) != EXPECTED_NUM_OBJECTS:
        problems.append(
            f"parsed {len(entries)} objects, expected NUM_OBJECTS="
            f"{EXPECTED_NUM_OBJECTS}"
        )
    if "BOULDER" in onames:
        idx = onames["BOULDER"]
        if idx >= len(entries) or entries[idx][2] != CLASS_VALUES["ROCK_CLASS"]:
            problems.append(
                f"BOULDER at otyp {idx} is not ROCK_CLASS "
                f"(got {entries[idx] if idx < len(entries) else 'OOB'})"
            )
    # Classes must be non-decreasing in otyp order (NetHack relies on this).
    for k in range(1, len(entries)):
        if entries[k][2] < entries[k - 1][2]:
            problems.append(f"class not monotonic at otyp {k}: {entries[k-1]} -> {entries[k]}")
            break
    if problems:
        raise SystemExit("VALIDATION FAILED:\n" + "\n".join(problems))

    with open(out_path, "w") as f:
        f.write('"""GENERATED by vendor/tools/gen_object_table.py -- do not edit.\n\n')
        f.write(f"Source: {objects_c}\n")
        f.write('Fields: (oc_name, oc_descr, oc_class) indexed by otyp.\n"""\n\n')
        f.write("OBJECT_TABLE = [\n")
        for name, descr, cls in entries:
            f.write(f"    ({name!r}, {descr!r}, {cls}),\n")
        f.write("]\n")

    print(f"wrote {len(entries)} objects -> {out_path}")
    if "BOULDER" in onames:
        print("BOULDER otyp", onames["BOULDER"], "->", entries[onames["BOULDER"]])


if __name__ == "__main__":
    main(*sys.argv[1:4])
