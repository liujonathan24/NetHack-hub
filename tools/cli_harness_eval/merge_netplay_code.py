#!/usr/bin/env python3
"""Merge per-rollout `netplay` code trees into the canonical one, once per round.

    merge_netplay_code.py --canonical <dir> [--branch agent-X ...] [--report R]
    merge_netplay_code.py --canonical <dir> --rollback round-3

WHY NOT `merge_harness_stores.py`. That merger is keyed by `(kind, id)` and
resolves collisions by newest `updated_at`. That is right for independent
180-character text entries and wrong for code, four ways:

  1. newest-wins is a SILENT OVERWRITE -- two rollouts both improving
     `explore.py` produce one winner and one discarded body;
  2. the unit is wrong -- a code change is a diff against a known base, not a
     whole-value replacement, so it cannot express "seed 7 fixed the corridor
     bug and seed 9 added door-kicking" even though those compose trivially;
  3. files are COUPLED -- `explore.py` may start calling a helper that only
     exists in seed 9's tree, and per-file newest-wins yields a tree that
     imports a module that is not there;
  4. it has no notion of "this contribution is invalid", which a code tree
     needs because a merge result can fail the gate even when both parents
     passed.

So: three-way merge, in a deterministic order, gated after every step.

POLICY, stated so the report can be read without the source:
  - branches merge in ascending name order, so the merge is replayable;
  - a CLEAN merge that PASSES the gate is committed;
  - a clean merge that FAILS the gate is reverted and the branch `rejected`;
  - a CONFLICT is never auto-resolved. The later branch is dropped and the
    conflicting paths recorded. An unresolved conflict between two rollouts of
    the same arm is a research finding about whether contributions compose --
    silently picking one hides exactly the thing worth knowing.
  - every round ends tagged `round-<n>`, so rollback is one command.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

try:
    from nethack_prime_agent import netplay_gate
except ImportError:  # pragma: no cover - the harness package is always present
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                           / "harnesses" / "nethack-prime-agent"))
    from nethack_prime_agent import netplay_gate


def _git(repo: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=check,
    )


def _natural(name: str) -> tuple:
    """Sort key that orders `agent-2` before `agent-10`.

    Plain lexicographic order put `agent-10` first, which mattered: the first
    branch merged wins every conflict, so string order silently privileged
    trace ids that sort low as text. SPEC 5.2 says ascending trace id, so this
    makes the code match the claim rather than the other way round.
    """
    return tuple(int(p) if p.isdigit() else p
                 for p in re.split(r"(\d+)", name))


def _branches(repo: pathlib.Path) -> list[str]:
    out = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").stdout
    return sorted((b.strip() for b in out.splitlines()
                   if b.strip().startswith("agent-")), key=_natural)


def _is_dirty(repo: pathlib.Path) -> str:
    """Non-empty when the tree is dirty or mid-merge -- both make a merge unsafe."""
    if (repo / ".git" / "MERGE_HEAD").exists():
        return "a previous merge was left unconcluded (.git/MERGE_HEAD present)"
    porcelain = _git(repo, "status", "--porcelain", check=False).stdout.strip()
    return f"working tree not clean:\n{porcelain}" if porcelain else ""


def _gate(repo: pathlib.Path, frozen_reference: pathlib.Path | None) -> dict:
    return netplay_gate.check_tree(repo, frozen_reference=frozen_reference)


def merge(canonical: pathlib.Path, branches: list[str], *, round_tag: str | None,
          frozen_reference: pathlib.Path | None = None,
          base_ref: str = "HEAD") -> dict:
    """Three-way merge each branch onto canonical, gating after every step.

    Refuses to start on a dirty or mid-merge tree, and refuses to start when
    the BASE already fails the gate -- both were found to poison a whole round,
    rejecting every branch and attributing the base's own findings to agents
    that never touched the offending file.
    """
    report: dict = {
        "canonical": str(canonical),
        "base": _git(canonical, "rev-parse", "--short", base_ref).stdout.strip(),
        "branches": branches,
        "merged": [], "rejected": [], "conflicts": [], "unchanged": [],
        "failed": [], "round_tag": round_tag, "head": None, "aborted": None,
    }

    dirty = _is_dirty(canonical)
    if dirty:
        report["aborted"] = f"refusing to merge: {dirty}"
        report["head"] = report["base"]
        return report

    # The base must itself be valid, or every rejection below is meaningless:
    # a merge result inherits the base's violations, so one bad file in
    # canonical rejects 100% of the round and blames the wrong rollouts.
    base_report = _gate(canonical, frozen_reference)
    report["base_gate_ok"] = base_report["ok"]
    if not base_report["ok"]:
        report["base_gate_findings"] = base_report["findings"]
        report["aborted"] = (
            "refusing to merge: the base tree itself fails the gate. Every "
            "branch would be rejected for findings it did not cause. Fix "
            "canonical (or roll back) first."
        )
        report["head"] = report["base"]
        return report

    for branch in branches:
        if not _git(canonical, "rev-parse", "--verify", "-q", f"{branch}^{{commit}}",
                    check=False).stdout.strip():
            # Distinct from "produced nothing": a missing branch is a harness
            # fault, and reporting it as an unproductive agent is a false
            # research finding.
            report["failed"].append({"branch": branch, "reason": "no such branch"})
            continue

        ahead = _git(canonical, "rev-list", "--count", f"{base_ref}..{branch}",
                     check=False).stdout.strip()
        if ahead in ("", "0"):
            report["unchanged"].append({"branch": branch, "reason": "no commits over base"})
            continue

        proc = _git(canonical, "merge", "--no-commit", "--no-ff", branch, check=False)
        if proc.returncode != 0:
            paths = _git(canonical, "diff", "--name-only", "--diff-filter=U",
                         check=False).stdout.split()
            _git(canonical, "merge", "--abort", check=False)
            if paths:
                report["conflicts"].append({
                    "branch": branch, "paths": paths,
                    "resolution": "dropped -- conflicts are not auto-resolved",
                })
            else:
                # git refused to START the merge (unrelated histories, local
                # changes in the way). Calling that a content conflict with an
                # empty path list is an untrue statement about what happened.
                report["failed"].append({
                    "branch": branch,
                    "reason": "merge could not start",
                    "git": (proc.stderr or proc.stdout).strip()[-300:],
                })
            continue

        gate = _gate(canonical, frozen_reference)
        if not gate["ok"]:
            # Two individually valid trees can merge into an invalid one.
            # `merge --abort` alone restores the pre-merge state; the extra
            # `reset --hard` that used to follow also destroyed uncommitted
            # work in canonical that no branch had touched.
            _git(canonical, "merge", "--abort", check=False)
            report["rejected"].append({
                "branch": branch,
                "reason": "merge result failed the gate",
                "findings": gate["findings"],
            })
            continue

        commit = _git(canonical, "commit", "--no-edit",
                      "-m", f"merge {branch} (gate ok)", check=False)
        if commit.returncode != 0:
            # Unchecked, this was the worst failure here: the branch was
            # recorded as merged, `head` read back the BASE commit, and
            # canonical was left mid-merge with a staged index -- which then
            # made the NEXT branch fail to start and abort the staged work too.
            # A hook, a missing identity, or a stale index.lock all trigger it.
            _git(canonical, "merge", "--abort", check=False)
            report["failed"].append({
                "branch": branch,
                "reason": "commit failed after a clean, gate-passing merge",
                "git": (commit.stderr or commit.stdout).strip()[-300:],
            })
            continue

        base_ref = "HEAD"
        report["merged"].append({
            "branch": branch,
            "commit": _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip(),
        })

    report["head"] = _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip()
    if round_tag:
        _git(canonical, "tag", "-f", round_tag, check=False)
    report["clean_after"] = not _is_dirty(canonical)
    return report


def rollback(canonical: pathlib.Path, tag: str, reason: str) -> dict:
    """Reset canonical to a round tag. The counterpart of every merge above.

    Three things this does beyond `git reset --hard`, each because the bare
    form was found not to restore what it claimed:

    - it REFUSES a ref that is not a tag. `rollback(repo, "agent-1")` used to
      succeed and move canonical onto a rollout branch while the report still
      called it a tag;
    - it removes untracked files. In the shared-tree mode agents write freely,
      so a bad generation can leave a gate-failing `.py` that no reset touches
      -- the tree hash matches the tag and the gate still fails;
    - it refuses when the tree is dirty, rather than destroying uncommitted
      work silently.
    """
    if not _git(canonical, "rev-parse", "--verify", "-q", f"refs/tags/{tag}",
                check=False).stdout.strip():
        raise SystemExit(
            f"rollback: {tag!r} is not a tag in {canonical}. Rolling back to a "
            "branch or a raw sha would move canonical somewhere no round ever "
            f"was. Known tags: {_git(canonical, 'tag', check=False).stdout.split() or '(none)'}"
        )
    dirty = _is_dirty(canonical)
    if dirty:
        raise SystemExit(
            f"rollback: refusing to discard uncommitted work -- {dirty}"
        )

    before = _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip()
    proc = _git(canonical, "reset", "--hard", tag, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"rollback to {tag} failed: {proc.stderr.strip()}")
    # `-x` so ignored files (__pycache__) go too; a stale .pyc of a module the
    # rollback removed is still importable.
    removed = _git(canonical, "clean", "-fdx", check=False).stdout.split()
    after = _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip()
    return {"rollback": {"tag": tag, "from": before, "to": after,
                         "reason": reason, "removed_untracked": removed}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--canonical", required=True, type=pathlib.Path,
                    help="the canonical netplay git repo")
    ap.add_argument("--branch", action="append", default=[],
                    help="rollout branch to merge; defaults to every agent-* branch")
    ap.add_argument("--frozen-reference", type=pathlib.Path, default=None,
                    help="repo copy of netplay/, to detect edits to frozen files")
    ap.add_argument("--round", dest="round_tag", default=None,
                    help="tag the resulting commit, e.g. round-3")
    ap.add_argument("--rollback", default=None,
                    help="reset canonical to this tag instead of merging")
    ap.add_argument("--reason", default="unspecified", help="why, for --rollback")
    ap.add_argument("--report", type=pathlib.Path)
    args = ap.parse_args(argv)

    if not (args.canonical / ".git").is_dir():
        print(f"merge: {args.canonical} is not a git repo -- nothing to merge.",
              file=sys.stderr)
        return 0

    if args.rollback:
        report = rollback(args.canonical, args.rollback, args.reason)
        print(f"rolled back to {args.rollback}: "
              f"{report['rollback']['from']} -> {report['rollback']['to']}")
    else:
        branches = args.branch or _branches(args.canonical)
        if not branches:
            # Still tag and still report. "Every round ends tagged" has to hold
            # even for a round where every rollout crashed, or a later
            # `--rollback round-N` fails on a tag that was never written.
            print("merge: no agent-* branches found -- nothing to merge.",
                  file=sys.stderr)
            report = {"canonical": str(args.canonical), "branches": [],
                      "merged": [], "rejected": [], "conflicts": [],
                      "unchanged": [], "failed": [], "round_tag": args.round_tag,
                      "head": _git(args.canonical, "rev-parse", "--short",
                                   "HEAD").stdout.strip()}
            if args.round_tag:
                _git(args.canonical, "tag", "-f", args.round_tag, check=False)
            if args.report:
                args.report.write_text(json.dumps(report, indent=2))
            return 0
        report = merge(args.canonical, branches, round_tag=args.round_tag,
                       frozen_reference=args.frozen_reference)
        print(f"merged {len(report['merged'])} of {len(branches)} branch(es): "
              f"{len(report['rejected'])} rejected, "
              f"{len(report['conflicts'])} conflicted, "
              f"{len(report['unchanged'])} unchanged -> {report['head']}")
        for r in report["rejected"]:
            print(f"  rejected {r['branch']}: {r['reason']}")
        for c in report["conflicts"]:
            print(f"  conflict {c['branch']}: {', '.join(c['paths'])}")
        for f in report.get("failed", []):
            print(f"  FAILED {f['branch']}: {f['reason']}", file=sys.stderr)
        if report.get("aborted"):
            print(f"  ABORTED: {report['aborted']}", file=sys.stderr)

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    # A round that aborted, or lost a branch to a harness fault, must not exit
    # 0 -- run_e13.sh has no other way to notice. A rejection or a conflict is
    # a legitimate research outcome and does NOT fail the round.
    if report.get("aborted") or report.get("failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
