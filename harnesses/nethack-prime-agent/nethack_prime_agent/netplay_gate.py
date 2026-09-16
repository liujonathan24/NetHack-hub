#!/usr/bin/env python3
"""Validate an agent-authored `netplay` tree before it is allowed to persist.

    python netplay_gate.py <tree-dir> [--json]

Exit 0 when the tree is safe to keep, 1 when it is not. The full report goes to
stdout as JSON with `--json`, and as readable lines otherwise.

WHY THIS EXISTS. `{install_dir}/skills` is ONE tree shared by every rollout of
every round. Prime Agent does not lint skill code, and an import failure binds a
placeholder that raises only when called -- so a `SyntaxError` an agent writes in
round 1 surfaces as a mysterious runtime error in round 4. This gate is what
keeps a broken or hostile tree from leaving the rollout that produced it.

It runs in three places, and it is the SAME code in all three (SPEC §6.1):

  1. in-episode, as `netplay.check()` -- advisory, agents skip advice;
  2. at rollout teardown, unconditionally -- the gate that actually holds;
  3. after each three-way merge at the round boundary, because two individually
     valid trees can merge into an invalid one.

WHAT IT CANNOT DO. It is a static check. It cannot tell a good explore policy
from a bad one, and it is not the anti-cheat rail on its own -- reconciling the
client call log against the server's record is (SPEC §6.2 rail 2). What it does
do is make `_base` the only reachable door, which is what makes that
reconciliation meaningful.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# Modules the agent-editable tree may import. Short on purpose: `_base` is the
# only sanctioned way to touch the game, so everything else is bookkeeping.
ALLOWED_ROOTS = frozenset({
    "netplay",
    "__future__", "collections", "dataclasses", "enum", "functools",
    "heapq", "itertools", "json", "math", "random", "re", "string",
    "textwrap", "time", "typing",
})

# Removed from the allowlist after an adversarial pass found each of them was a
# one-line escape from every other rule here:
#   importlib -- `importlib.import_module("socket")` imports anything the
#                denylist forbids, and the import never appears as an AST import
#   os        -- `os.system` / `os.popen` are process escapes
#   ast       -- only `_base.check()` needs it, and that file is frozen
#   pathlib   -- filesystem reach the policies do not need
#   warnings  -- only the frozen `__init__` needs it
# The frozen files legitimately use several of these and are policy-exempt, so
# nothing we ship is affected. If a composite genuinely needs one, add it back
# deliberately -- do not widen this set to make a failing gate pass.

# `nethack` is the MCP shim itself. Only the FROZEN files may import it: a
# composite that reaches it directly gets the game without the correlation
# beacons, and its calls then have no client-side record to reconcile against
# the server's -- which is exactly the signal SPEC §6.2 rail 2 depends on.
FROZEN_ONLY_ROOTS = frozenset({"nethack"})

# Never importable from agent code, even where stdlib: each is a way around the
# MCP boundary, out of the sandbox, or into the engine the kernel must not have.
DENIED_ROOTS = frozenset({
    "socket", "subprocess", "ctypes", "urllib", "http", "httpx", "requests",
    "nle", "gym", "gymnasium", "verifiers", "multiprocessing", "shutil",
    "pickle", "marshal", "tempfile", "glob", "sysconfig", "site",
})

# Dynamic-execution escapes. `compile`/`exec`/`eval`/`__import__` would let a
# tree import a denied module at runtime, which every static check above would
# miss. `open` is NOT denied -- reading its own source is legitimate.
# Checked as any REFERENCE, not only as a call. `e = eval` followed by `e(...)`
# is a call whose func is `Name('e')`, which no denylist of call names can catch
# -- so the alias is rejected where it is created.
DENIED_NAMES = frozenset({
    "exec", "eval", "compile", "__import__", "__builtins__",
    # Attribute and namespace reflection: each one turns a static rule below
    # into a suggestion. `vars(_base)["press"] = None` and
    # `setattr(b, "press", None)` both rebind the floor.
    "getattr", "setattr", "delattr", "vars", "globals", "locals",
})
DENIED_CALLS = DENIED_NAMES   # back-compat for callers importing the old name

# Names whose ATTRIBUTES must never be assigned, however the expression is
# spelled. Resolved by walking to the root of the attribute/subscript chain, so
# `netplay._base.press = x` and `b.press, q = ...` are caught alongside the
# simple form.
BOUNDARY_ROOTS = frozenset({"nethack", "_base", "netplay", "_nethack"})

# Files we ship. An agent may add modules, but not replace these.
FROZEN_FILES = frozenset({"__init__.py", "_base.py"})


def _finding(path: Path, node: ast.AST | None, kind: str, detail: str) -> dict:
    return {
        "file": path.name,
        "line": getattr(node, "lineno", 0),
        "kind": kind,
        "detail": detail,
    }


def _root_name(node: ast.AST) -> str | None:
    """The base Name of an attribute/subscript chain, or None.

    `netplay._base.press` and `vars(_base)["press"]` both have to resolve to
    something the boundary rules can test; matching the literal spelling
    `_base.press` only catches the shortest of the several ways to write it.
    """
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    if isinstance(node, ast.Call):                 # vars(_base)[...] / getattr(...)
        return _root_name(node.func) if not node.args else _root_name(node.args[0])
    return node.id if isinstance(node, ast.Name) else None


def _targets(node: ast.AST) -> list[ast.expr]:
    """Every expression a statement ASSIGNS TO.

    Covers the forms the first version of this walker missed entirely: tuple and
    starred packing, annotated assignment, `for` targets, `with ... as`, and
    comprehension targets. Each of them can rebind an attribute of the floor.
    """
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
        return [node.target]
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.target]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [i.optional_vars for i in node.items if i.optional_vars]
    if isinstance(node, ast.comprehension):
        return [node.target]
    if isinstance(node, ast.Delete):
        return list(node.targets)
    return []


def check_source(path: Path, source: str, *, policy: bool = True) -> list[dict]:
    """Every violation in one file. Empty list means the file is acceptable.

    `policy=False` runs the syntax check ONLY. Used for the frozen files, which
    are ours: `__init__.py` legitimately uses `__import__` to bind the
    composites, and `_base.check()` legitimately calls `compile` -- they are the
    implementation of the rules, so they cannot also be subject to them. Their
    integrity is established by byte-comparison against the repository copy
    instead, which is a strictly stronger check than any AST rule.
    """
    findings: list[dict] = []
    try:
        tree = ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return [_finding(path, None, "syntax", f"{type(exc).__name__}: {exc}")]
    if not policy:
        return findings

    # Local names that end up bound to the floor or the shim. `from netplay
    # import _base as b` makes `b.press = None` a rebinding of the floor under
    # a name no fixed list could contain, so the names are collected from the
    # file's own imports before any rule is applied.
    boundary = set(BOUNDARY_ROOTS)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in BOUNDARY_ROOTS:
                    boundary.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            mod = "netplay" if n.level else (n.module or "")
            if mod.split(".")[0] in BOUNDARY_ROOTS:
                for a in n.names:
                    boundary.add(a.asname or a.name)

    # Propagate through plain `x = y` rebinding, to a fixpoint. A module bound
    # to a local (`m = _base`) is ordinary style, so the assignment itself is
    # not a finding -- but every attribute write through `m` has to be. Two
    # passes are not enough in a chain (`m = _base; n = m`), so this loops.
    for _ in range(len(tree.body) + 1):
        grew = False
        for n in ast.walk(tree):
            if not isinstance(n, ast.Assign) or not isinstance(n.value, ast.Name):
                continue
            if n.value.id not in boundary:
                continue
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id not in boundary:
                    boundary.add(t.id)
                    grew = True
        if not grew:
            break

    for node in ast.walk(tree):
        # --- imports ---
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = ["netplay"] if node.level else (
                [node.module.split(".")[0]] if node.module else []
            )
        for root in roots:
            if root in DENIED_ROOTS:
                findings.append(_finding(path, node, "denied-import", root))
            elif root in FROZEN_ONLY_ROOTS:
                findings.append(_finding(
                    path, node, "bypasses-floor",
                    f"{root} -- import netplay._base instead",
                ))
            elif root not in ALLOWED_ROOTS:
                findings.append(_finding(path, node, "unlisted-import", root))

        # --- dynamic execution and reflection, by REFERENCE not by call ---
        # Catching the call site is not enough: `e = eval` then `e(...)` is a
        # call on a name the denylist has never heard of. Rejecting the name
        # wherever it is mentioned closes the alias, the subscript form
        # (`__builtins__["eval"]`) and the getattr form in one rule.
        if isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            findings.append(_finding(path, node, "dynamic-exec", node.id))

        # --- reaching the boundary through an object instead of an import ---
        # `_base` imports the shim; before it was bound privately, `_base.nethack`
        # handed any composite the game with no beacon attached. Both spellings
        # are rejected, as is `__dict__`, which is the same escape one level down.
        if isinstance(node, ast.Attribute):
            if node.attr in ("nethack", "_nethack"):
                findings.append(_finding(
                    path, node, "bypasses-floor",
                    f".{node.attr} -- the shim is reachable only from _base",
                ))
            elif node.attr == "__dict__":
                findings.append(_finding(path, node, "patches-boundary", "__dict__"))
            elif node.attr.startswith("_") and _root_name(node) in boundary:
                findings.append(_finding(
                    path, node, "bypasses-floor",
                    f"{_root_name(node)}.{node.attr} -- private access",
                ))

        # --- rebinding the floor, in every assignment form ---
        for tgt in _targets(node):
            for sub in ast.walk(tgt):
                if not isinstance(sub, (ast.Attribute, ast.Subscript)):
                    continue
                root = _root_name(sub)
                if root in boundary:
                    label = getattr(sub, "attr", "[...]")
                    findings.append(_finding(
                        path, node, "patches-boundary", f"{root}.{label}"))
    return findings


def check_tree(root: Path, *, frozen_reference: Path | None = None) -> dict:
    """Validate every module in a `netplay` tree.

    `frozen_reference` is the repo's own copy. When given, the frozen files are
    compared byte-for-byte against it -- an agent that rewrote `_base.py` in a
    way that still parses would otherwise pass every other check here.
    """
    root = Path(root)
    report: dict = {"tree": str(root), "files": [], "findings": [], "ok": False}
    if not root.is_dir():
        report["findings"].append(
            {"file": str(root), "line": 0, "kind": "missing", "detail": "not a directory"}
        )
        return report

    # RECURSIVE. A non-recursive glob was a total bypass: a subpackage
    # (`netplay/evil/__init__.py`) importing socket and subprocess was never
    # opened, so every rule above -- allowlist, boundary, dynamic exec --
    # applied only to files the agent chose to put at the top level. `.git` is
    # skipped because the round merge keeps history in the tree itself.
    for path in sorted(root.rglob("*.py")):
        if any(part == ".git" or part == "__pycache__" for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        report["files"].append(rel)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            report["findings"].append(
                {"file": rel, "line": 0, "kind": "unreadable", "detail": str(exc)})
            continue
        # Only the top-level frozen files are policy-exempt. A nested file
        # NAMED `_base.py` is agent code and gets the full treatment.
        for f in check_source(path, source, policy=rel not in FROZEN_FILES):
            f["file"] = rel
            report["findings"].append(f)

    missing = FROZEN_FILES - set(report["files"])
    for name in sorted(missing):
        report["findings"].append(
            {"file": name, "line": 0, "kind": "missing",
             "detail": "frozen file deleted from the tree"}
        )

    if frozen_reference is not None:
        for name in sorted(FROZEN_FILES):
            ours, theirs = Path(frozen_reference) / name, root / name
            if not ours.is_file() or not theirs.is_file():
                continue
            if ours.read_bytes() != theirs.read_bytes():
                report["findings"].append(
                    {"file": name, "line": 0, "kind": "frozen-modified",
                     "detail": "differs from the repository copy"}
                )

    report["ok"] = not report["findings"]
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", type=Path, help="the netplay/ directory to validate")
    ap.add_argument("--frozen-reference", type=Path, default=None,
                    help="repo copy of netplay/, to detect edits to frozen files")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = ap.parse_args(argv)

    report = check_tree(args.tree, frozen_reference=args.frozen_reference)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"netplay gate: {args.tree}")
        print(f"  files:    {len(report['files'])}")
        if report["ok"]:
            print("  verdict:  PASS")
        else:
            print(f"  verdict:  FAIL ({len(report['findings'])} finding(s))")
            for f in report["findings"]:
                print(f"    {f['file']}:{f['line']}  {f['kind']}: {f['detail']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
