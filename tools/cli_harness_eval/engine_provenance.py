"""Fingerprint the NetHack ENGINE that a run is about to link, and refuse to
launch when that engine is provably stale.

Why this exists (two measured incidents, both silent):

1. SUBMODULE POINTER DRIFT. A session checked `third_party/NetHack` out to
   66c84e6 while the engine repo's gitlink recorded fefd557. About 30
   engine-dependent tests then failed with errors that read like engine LOGIC
   bugs -- `test_engine_env::test_snapshot_restore_branching_via_env` failed
   `assert not np.array_equal(glyphs_a, glyphs_b)`, i.e. two *branched* games
   returning byte-identical maps. Nothing in any traceback mentioned the
   submodule. The diagnosis took hours because the recorded provenance
   (`results/run_provenance.json`) carried only the HARNESS commit.

2. STALE COMPILED ARTIFACT. Even with the pointer restored, the .so on disk was
   built from the *other* source line:
       src/src/nle.c    2026-07-31 18:52   (restored source)
       libnethack.so    2026-07-22 04:26   (built 9 days earlier)
   Every rollout `ctypes.CDLL`s that .so. A source checkout does not rebuild it
   and nothing warns. On a high-parallelism box this corrupts an entire sweep
   with no error at all -- the numbers come out, they are just not the numbers
   for the code in the tree.

So: capture what engine is actually loaded, write it next to the results, and
block the launch when the artifact cannot correspond to the source.

Everything here DEGRADES rather than raises. A provenance helper that can abort
a launch by throwing is worse than no helper -- unknown fields come back as
``None`` with a reason appended to ``errors``, and an unknown staleness verdict
never blocks (only a *proven* one does).

CLI:
    python3 tools/cli_harness_eval/engine_provenance.py            # one-line fingerprint
    python3 tools/cli_harness_eval/engine_provenance.py --check    # + exit 3 if stale/dirty
    python3 tools/cli_harness_eval/engine_provenance.py --json P   # dump the full dict to P
    python3 tools/cli_harness_eval/engine_provenance.py --run-record results/run_provenance.json --run p1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

# Extensions that, when newer than the .so, mean the .so cannot be the product
# of the tree on disk. Deliberately NOT "any tracked file": a README or a .md
# touched by a checkout must not block a launch, because this check REFUSES to
# run and a false positive costs a sweep. Kept to things the engine build
# actually consumes -- C/C++ sources and headers, the lex/yacc grammars, cmake
# inputs, and the .des level descriptions (build_engine.sh recompiles dat/ from
# them, and the level layout is an experimental factor).
SOURCE_SUFFIXES = frozenset(
    {".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".y", ".l", ".in", ".cmake", ".des"}
)
SOURCE_NAMES = frozenset({"CMakeLists.txt", "Makefile", "makefile"})

SUBMODULE_REL = Path("third_party") / "NetHack"

# Where the fingerprint lands inside a cell's output dir, so a future reader who
# only has `outputs/<run>/<arm>/` can still say which engine produced it.
CELL_FILENAME = "engine_provenance.json"


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return _dt.datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def _git(repo: Path, *args: str) -> str | None:
    """Run git in ``repo``; return stripped stdout, or None on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _resolve_library_path(engine_root: Path | None, errors: list) -> Path | None:
    """Find the .so the way the ENGINE finds it, not by guessing a path.

    There are at least two libnethack.so on disk in a normal checkout
    (`third_party/NetHack/lib/libnethack.so`, a stale June artifact, and
    `third_party/NetHack/src/build/libnethack.so`, the one actually loaded), plus
    one per sibling worktree. Fingerprinting the wrong one is worse than
    fingerprinting none, so ask `nethack_core._engine.library_path()` -- the same
    function `load_library()` calls -- which honours NLE_LIB_PATH, then the
    build dir, then the bundled-wheel copy.

    ``engine_root``, when given, overrides that resolution entirely. It exists so
    tests can point at a synthetic engine tree, and so a human can fingerprint a
    checkout other than the importable one.
    """
    if engine_root is not None:
        env_lib = os.environ.get("NLE_LIB_PATH")
        if env_lib and Path(env_lib).exists():
            return Path(env_lib)
        for cand in (
            engine_root / SUBMODULE_REL / "src" / "build" / "libnethack.so",
            engine_root / "nethack_core" / "libnethack.so",
        ):
            if cand.exists():
                return cand
        errors.append(f"no libnethack.so under engine_root={engine_root}")
        return None

    try:
        from nethack_core import _engine  # noqa: PLC0415  (deliberately lazy)

        return Path(_engine.library_path())
    except Exception as exc:  # noqa: BLE001 -- must never break a launch
        errors.append(f"nethack_core library_path() unavailable: {exc.__class__.__name__}: {exc}")
        return None


def _infer_engine_root(so_path: Path | None) -> Path | None:
    """Walk up from the loaded .so for the dir that owns `third_party/NetHack`."""
    starts = []
    if so_path is not None:
        starts.append(so_path.resolve())
    try:
        import nethack_core

        starts.append(Path(nethack_core.__file__).resolve())
    except Exception:  # noqa: BLE001
        pass
    for start in starts:
        for parent in start.parents:
            if (parent / SUBMODULE_REL).is_dir():
                return parent
    return None


def _newest_source(submodule: Path, errors: list) -> tuple[str | None, float | None]:
    """Newest mtime among TRACKED build inputs in the submodule.

    Tracked only: an untracked scratch .c dropped in the tree is not part of the
    engine and must not block a launch. `git ls-files` over the fork's ~1.2k
    files plus a stat each measures at ~25 ms, so this is cheap enough to run on
    every launch.
    """
    listing = _git(submodule, "ls-files", "-z")
    if listing is None:
        errors.append(f"git ls-files failed in {submodule}")
        return None, None
    newest_path: str | None = None
    newest_mtime: float | None = None
    for rel in listing.split("\0"):
        if not rel:
            continue
        name = rel.rsplit("/", 1)[-1]
        suffix = name[name.rfind(".") :] if "." in name else ""
        if suffix not in SOURCE_SUFFIXES and name not in SOURCE_NAMES:
            continue
        try:
            mtime = (submodule / rel).stat().st_mtime
        except OSError:
            continue
        if newest_mtime is None or mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = rel
    if newest_mtime is None:
        errors.append(f"no tracked build inputs found under {submodule}")
    return newest_path, newest_mtime


def engine_fingerprint(engine_root: Path | str | None = None) -> dict:
    """Describe the engine this process would link. Never raises.

    Keys whose value could not be determined are ``None`` and the reason is in
    ``errors``; ``so_stale``/``submodule_dirty`` are tri-state (True / False /
    None-for-unknown) precisely so an unknown never masquerades as "fine" nor as
    a reason to refuse.
    """
    errors: list = []
    root = Path(engine_root) if engine_root is not None else None

    so_path = _resolve_library_path(root, errors)
    if root is None:
        root = _infer_engine_root(so_path)

    fp: dict = {
        "captured_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "engine_root": str(root) if root else None,
        "engine_head": None,
        "engine_dirty": None,
        "submodule_sha": None,
        "submodule_pinned_sha": None,
        "submodule_matches_pin": None,
        "submodule_dirty": None,
        "so_path": str(so_path) if so_path else None,
        "so_mtime": None,
        "so_mtime_iso": None,
        "so_size": None,
        "newest_source_path": None,
        "newest_source_mtime": None,
        "newest_source_mtime_iso": None,
        "so_stale": None,
        "so_stale_by_seconds": None,
        "errors": errors,
    }

    if so_path is not None:
        try:
            st = so_path.stat()
            fp["so_mtime"] = st.st_mtime
            fp["so_mtime_iso"] = _iso(st.st_mtime)
            fp["so_size"] = st.st_size
        except OSError as exc:
            errors.append(f"stat({so_path}) failed: {exc}")

    if root is None:
        errors.append("could not locate an engine repo root (no third_party/NetHack above the .so)")
        return fp

    fp["engine_head"] = _git(root, "rev-parse", "HEAD")
    porcelain = _git(root, "status", "--porcelain")
    if porcelain is not None:
        fp["engine_dirty"] = bool(porcelain.strip())

    submodule = root / SUBMODULE_REL
    if not submodule.is_dir():
        errors.append(f"submodule dir missing: {submodule}")
        return fp

    fp["submodule_sha"] = _git(submodule, "rev-parse", "HEAD")
    # The gitlink the SUPERPROJECT records, i.e. the commit this tree is
    # *supposed* to be at. Incident 1 was exactly a mismatch between these two,
    # so both are recorded and compared -- one alone cannot detect it.
    ls_tree = _git(root, "ls-tree", "HEAD", str(SUBMODULE_REL))
    if ls_tree:
        parts = ls_tree.split()
        if len(parts) >= 3 and parts[1] == "commit":
            fp["submodule_pinned_sha"] = parts[2]
    if fp["submodule_sha"] and fp["submodule_pinned_sha"]:
        fp["submodule_matches_pin"] = fp["submodule_sha"] == fp["submodule_pinned_sha"]

    sub_porcelain = _git(submodule, "status", "--porcelain")
    if sub_porcelain is None:
        errors.append(f"git status failed in {submodule}")
    else:
        fp["submodule_dirty"] = bool(sub_porcelain.strip())

    src_path, src_mtime = _newest_source(submodule, errors)
    fp["newest_source_path"] = src_path
    fp["newest_source_mtime"] = src_mtime
    fp["newest_source_mtime_iso"] = _iso(src_mtime)
    if src_mtime is not None and fp["so_mtime"] is not None:
        fp["so_stale"] = fp["so_mtime"] < src_mtime
        fp["so_stale_by_seconds"] = max(0.0, src_mtime - fp["so_mtime"])

    return fp


def fingerprint_line(fp: dict) -> str:
    """One line for the launch log. Short enough to grep, complete enough to act on."""

    def short(sha):
        return sha[:7] if sha else "unknown"

    so = fp.get("so_path") or "unknown"
    flags = []
    if fp.get("so_stale"):
        flags.append("STALE_SO")
    if fp.get("submodule_dirty"):
        flags.append("DIRTY_SUBMODULE")
    if fp.get("submodule_matches_pin") is False:
        flags.append("SUBMODULE_OFF_PIN")
    if fp.get("engine_dirty"):
        flags.append("dirty_engine_repo")
    if fp.get("so_stale") is None:
        flags.append("staleness_unknown")
    return (
        f"[engine] head={short(fp.get('engine_head'))} "
        f"submodule={short(fp.get('submodule_sha'))} "
        f"pinned={short(fp.get('submodule_pinned_sha'))} "
        f"so={so} built={fp.get('so_mtime_iso') or 'unknown'} "
        f"newest_src={fp.get('newest_source_path') or 'unknown'}@{fp.get('newest_source_mtime_iso') or 'unknown'} "
        f"flags={','.join(flags) if flags else 'ok'}"
    )


def staleness_reasons(fp: dict) -> list:
    """The conditions that MUST block a launch, with the numbers that prove them.

    Only PROVEN problems are listed. `so_stale is None` (couldn't tell) is not a
    reason -- refusing on an unknown would make an unbuildable dev box unusable
    and teach everyone to export the bypass permanently.
    """
    reasons = []
    if fp.get("so_stale"):
        gap = fp.get("so_stale_by_seconds") or 0.0
        reasons.append(
            f"libnethack.so is STALE: {fp.get('so_path')} built {fp.get('so_mtime_iso')} but "
            f"{fp.get('newest_source_path')} was modified {fp.get('newest_source_mtime_iso')} "
            f"({gap / 3600.0:.1f}h newer). Rebuild with nethack_core/build_engine.sh."
        )
    if fp.get("submodule_dirty"):
        reasons.append(
            f"third_party/NetHack has uncommitted changes ({fp.get('submodule_sha')}), so no commit "
            "identifies the engine that produced this run."
        )
    if fp.get("submodule_matches_pin") is False:
        reasons.append(
            f"third_party/NetHack is at {fp.get('submodule_sha')} but the engine repo pins "
            f"{fp.get('submodule_pinned_sha')} -- the exact drift that made ~30 engine tests fail "
            "with what looked like logic bugs."
        )
    return reasons


BYPASS_ENV = "ALLOW_STALE_ENGINE"


def check(fp: dict, allow_stale: bool | None = None, stream=None) -> int:
    """Print the fingerprint and return the process exit code (0 ok, 3 refuse)."""
    stream = stream or sys.stderr
    if allow_stale is None:
        allow_stale = bool(os.environ.get(BYPASS_ENV))
    reasons = staleness_reasons(fp)
    if not reasons:
        return 0
    if allow_stale:
        print(
            f"!!! {BYPASS_ENV}=1 -- launching against an engine that does NOT match the source tree.",
            file=stream,
        )
        for r in reasons:
            print(f"!!!   {r}", file=stream)
        print(
            "!!! Results from this run are NOT reproducible from any commit. "
            "Record why in the run's notes.",
            file=stream,
        )
        return 0
    print("launch refused: the engine on disk cannot have come from the source tree.", file=stream)
    for r in reasons:
        print(f"  - {r}", file=stream)
    print(f"  Set {BYPASS_ENV}=1 to launch anyway (loudly).", file=stream)
    return 3


# ---------------------------------------------------------------------------
# run_provenance.json
# ---------------------------------------------------------------------------


def record_run(path: Path | str, run: str, driver: str = "", extra: str = "", fp: dict | None = None) -> dict:
    """Append one run entry (harness HEAD + ENGINE fingerprint) to run_provenance.json.

    The historical file recorded `run -> harness HEAD` only, which is why no past
    result can be attributed to an engine. New entries carry `engine`.
    """
    path = Path(path)
    doc = {"runs": [], "cells": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                doc = loaded
                doc.setdefault("runs", [])
                doc.setdefault("cells", [])
        except (OSError, ValueError):
            pass
    if fp is None:
        fp = engine_fingerprint()
    repo = path.resolve().parent.parent
    entry = {
        "driver": driver,
        "run": run,
        "started": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "head": (_git(repo, "rev-parse", "--short", "HEAD") or "unknown"),
        "extra": extra,
        "engine": fp,
    }
    doc["runs"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-root", default=os.environ.get("ENGINE_PROVENANCE_ROOT") or None,
                    help="fingerprint this checkout instead of the importable nethack_core "
                         "(also settable via ENGINE_PROVENANCE_ROOT; used by the tests)")
    ap.add_argument("--check", action="store_true",
                    help=f"exit 3 when the engine is provably stale, unless {BYPASS_ENV} is set")
    ap.add_argument("--json", dest="json_path", default=None, help="write the full fingerprint dict here")
    ap.add_argument("--quiet", action="store_true", help="do not print the one-line fingerprint")
    ap.add_argument("--run-record", default=None, help="append a run entry to this run_provenance.json")
    ap.add_argument("--run", default="", help="run name for --run-record")
    ap.add_argument("--driver", default="", help="driver log name for --run-record")
    ap.add_argument("--extra", default="", help="free-form note for --run-record")
    args = ap.parse_args(argv)

    fp = engine_fingerprint(args.engine_root)
    if not args.quiet:
        # Flushed explicitly: stdout is block-buffered when the launcher pipes it
        # to a log, so without this the refusal on stderr lands ABOVE the
        # fingerprint that explains it.
        print(fingerprint_line(fp), flush=True)
    if args.json_path:
        try:
            p = Path(args.json_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(fp, indent=2) + "\n")
        except OSError as exc:
            print(f"[engine] could not write {args.json_path}: {exc}", file=sys.stderr)
    if args.run_record:
        try:
            record_run(args.run_record, run=args.run, driver=args.driver, extra=args.extra, fp=fp)
        except OSError as exc:
            print(f"[engine] could not update {args.run_record}: {exc}", file=sys.stderr)
    if args.check:
        return check(fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
