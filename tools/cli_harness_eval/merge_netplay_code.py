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


def _branches(repo: pathlib.Path) -> list[str]:
    out = _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/").stdout
    return sorted(b.strip() for b in out.splitlines()
                  if b.strip().startswith("agent-"))


def _gate(repo: pathlib.Path, frozen_reference: pathlib.Path | None) -> dict:
    return netplay_gate.check_tree(repo, frozen_reference=frozen_reference)


def merge(canonical: pathlib.Path, branches: list[str], *, round_tag: str | None,
          frozen_reference: pathlib.Path | None = None,
          base_ref: str = "HEAD") -> dict:
    """Three-way merge each branch onto canonical, gating after every step."""
    report: dict = {
        "canonical": str(canonical),
        "base": _git(canonical, "rev-parse", "--short", base_ref).stdout.strip(),
        "branches": branches,
        "merged": [], "rejected": [], "conflicts": [], "unchanged": [],
        "round_tag": round_tag, "head": None,
    }

    # The base must itself be valid, or every rejection below is meaningless.
    base_report = _gate(canonical, frozen_reference)
    report["base_gate_ok"] = base_report["ok"]
    if not base_report["ok"]:
        report["base_gate_findings"] = base_report["findings"]

    for branch in branches:
        ahead = _git(canonical, "rev-list", "--count", f"{base_ref}..{branch}",
                     check=False).stdout.strip()
        if ahead in ("", "0"):
            report["unchanged"].append({"branch": branch, "reason": "no commits over base"})
            continue

        proc = _git(canonical, "merge", "--no-commit", "--no-ff", branch, check=False)
        if proc.returncode != 0:
            # Record WHICH paths disagreed; "it conflicted" is not a finding.
            paths = _git(canonical, "diff", "--name-only", "--diff-filter=U",
                         check=False).stdout.split()
            _git(canonical, "merge", "--abort", check=False)
            report["conflicts"].append({
                "branch": branch,
                "paths": paths,
                "resolution": "dropped -- conflicts are not auto-resolved",
            })
            continue

        gate = _gate(canonical, frozen_reference)
        if not gate["ok"]:
            # Two individually valid trees can merge into an invalid one; this
            # is the case merge_harness_stores.py has no way to express.
            _git(canonical, "merge", "--abort", check=False)
            _git(canonical, "reset", "--hard", base_ref, check=False)
            report["rejected"].append({
                "branch": branch,
                "reason": "merge result failed the gate",
                "findings": gate["findings"],
            })
            continue

        _git(canonical, "commit", "-q", "--no-edit",
             "-m", f"merge {branch} (gate ok)", check=False)
        base_ref = "HEAD"
        report["merged"].append({
            "branch": branch,
            "commit": _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip(),
        })

    report["head"] = _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip()
    if round_tag:
        _git(canonical, "tag", "-f", round_tag, check=False)
    return report


def rollback(canonical: pathlib.Path, tag: str, reason: str) -> dict:
    """Reset canonical to a round tag. The counterpart of every merge above."""
    before = _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip()
    proc = _git(canonical, "reset", "--hard", tag, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"rollback to {tag} failed: {proc.stderr.strip()}")
    after = _git(canonical, "rev-parse", "--short", "HEAD").stdout.strip()
    return {"rollback": {"tag": tag, "from": before, "to": after, "reason": reason}}


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
            print("merge: no agent-* branches found -- nothing to merge.",
                  file=sys.stderr)
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
            print(f"  conflict {c['branch']}: {', '.join(c['paths']) or '(paths unknown)'}")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
