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
    "__future__", "ast", "collections", "dataclasses", "enum", "functools",
    "heapq", "importlib", "itertools", "json", "math", "os", "pathlib",
    "random", "re", "string", "textwrap", "time", "typing", "warnings",
})

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
DENIED_CALLS = frozenset({"exec", "eval", "compile", "__import__"})

# Files we ship. An agent may add modules, but not replace these.
FROZEN_FILES = frozenset({"__init__.py", "_base.py"})


def _finding(path: Path, node: ast.AST | None, kind: str, detail: str) -> dict:
    return {
        "file": path.name,
        "line": getattr(node, "lineno", 0),
        "kind": kind,
        "detail": detail,
    }


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

        # --- dynamic execution ---
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DENIED_CALLS:
                findings.append(_finding(path, node, "dynamic-exec", node.func.id))

        # --- monkey-patching the frozen shim ---
        # `nethack.np_press_key = my_fake` would redirect the boundary itself,
        # which no amount of import policing would catch.
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for tgt in targets:
            if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id in ("nethack", "_base")):
                findings.append(_finding(
                    path, node, "patches-boundary",
                    f"{tgt.value.id}.{tgt.attr}",
                ))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "setattr" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id in ("nethack", "_base"):
                    findings.append(_finding(
                        path, node, "patches-boundary", f"setattr({first.id}, ...)"
                    ))
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

    for path in sorted(root.glob("*.py")):
        report["files"].append(path.name)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            report["findings"].append(_finding(path, None, "unreadable", str(exc)))
            continue
        report["findings"].extend(
            check_source(path, source, policy=path.name not in FROZEN_FILES)
        )

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
