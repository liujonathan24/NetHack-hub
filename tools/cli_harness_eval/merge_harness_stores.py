#!/usr/bin/env python3
"""Merge per-rollout continual-harness stores back into the canonical one.

Why this exists, measured: `rlm.harness` persists with a non-atomic whole-file
rewrite (`open("w")` + `json.dump`, no lock, no tmp+rename), and its loader
treats an unreadable state file as EMPTY rather than erroring. Five concurrent
writers each adding six entries to one shared store kept 12 of 30; one of the
five contributed nothing, and nothing reported it. So in the players-write arm
each rollout gets a private copy (`continual_harness_mode = copy-merge`) and
their edits are reconciled here, once, single-threaded.

    merge_harness_stores.py --canonical <dir> --run-dir <cell-out> [--install-dir D]
    merge_harness_stores.py --canonical <dir> --store <file> [--store <file> ...]

`--run-dir` reads the cell's traces.jsonl for its rollout ids and finds each
private store at `<install-dir>/agent-<id>/harness/harness_state.json`, so the
caller does not have to know the layout.

Policy: entries are keyed by (kind, id); the newest `updated_at` wins. Every
same-id collision where the content actually differs is recorded in the merge
report whether or not it changed the outcome -- a silent merge is how five
seeds' disagreement turns into one seed's opinion.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

KINDS = ("prompt", "memory", "skill", "subagent")


def _load(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"schema": 1, "entries": {k: {} for k in KINDS}, "refinements": []}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        # Do NOT fall back to empty the way the runtime loader does: here that
        # would silently discard a whole rollout's contribution.
        raise SystemExit(f"merge: {path} is unreadable ({exc}). Refusing to merge.")
    data.setdefault("entries", {})
    for k in KINDS:
        data["entries"].setdefault(k, {})
    data.setdefault("refinements", [])
    return data


def _stores_from_run_dir(run_dir: pathlib.Path, install_dir: pathlib.Path) -> list[pathlib.Path]:
    traces = run_dir / "traces.jsonl"
    if not traces.exists():
        raise SystemExit(f"merge: no traces.jsonl in {run_dir}; cannot find rollout stores.")
    out = []
    for line in traces.read_text().splitlines():
        if not line.strip():
            continue
        tid = json.loads(line).get("id")
        if not tid:
            continue
        p = install_dir / f"agent-{tid}" / "harness" / "harness_state.json"
        if p.exists():
            out.append(p)
    return out


def merge(canonical: pathlib.Path, stores: list[pathlib.Path]) -> dict:
    base = _load(canonical / "harness_state.json")
    report = {
        "canonical": str(canonical),
        "stores": [str(s) for s in stores],
        "added": [], "updated": [], "conflicts": [], "unchanged": 0,
    }
    for store in stores:
        incoming = _load(store)
        origin = store.parent.parent.name          # agent-<trace id>
        for kind in KINDS:
            for eid, entry in incoming["entries"].get(kind, {}).items():
                cur = base["entries"][kind].get(eid)
                if cur is None:
                    base["entries"][kind][eid] = entry
                    report["added"].append({"kind": kind, "id": eid, "from": origin})
                    continue
                if cur == entry:
                    report["unchanged"] += 1
                    continue
                # Same id, different content: a real disagreement between two
                # rollouts (or against what was already known). Newest wins, and
                # the loser is recorded either way.
                newer = (entry.get("updated_at") or "") > (cur.get("updated_at") or "")
                report["conflicts"].append({
                    "kind": kind, "id": eid, "from": origin,
                    "kept": "incoming" if newer else "canonical",
                    "incoming_updated_at": entry.get("updated_at"),
                    "canonical_updated_at": cur.get("updated_at"),
                    "incoming_content": (entry.get("content") or "")[:200],
                    "canonical_content": (cur.get("content") or "")[:200],
                })
                if newer:
                    base["entries"][kind][eid] = entry
                    report["updated"].append({"kind": kind, "id": eid, "from": origin})
    canonical.mkdir(parents=True, exist_ok=True)
    # Atomic, unlike the runtime writer: a reader must never see a half file.
    tmp = canonical / "harness_state.json.tmp"
    tmp.write_text(json.dumps(base, indent=2, ensure_ascii=False))
    tmp.replace(canonical / "harness_state.json")
    report["totals"] = {k: len(base["entries"][k]) for k in KINDS}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, type=pathlib.Path)
    ap.add_argument("--run-dir", type=pathlib.Path)
    ap.add_argument("--install-dir", type=pathlib.Path,
                    default=pathlib.Path("/tmp/vf-prime-agent"))
    ap.add_argument("--store", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--report", type=pathlib.Path)
    args = ap.parse_args()

    stores = list(args.store)
    if args.run_dir:
        stores += _stores_from_run_dir(args.run_dir, args.install_dir)
    if not stores:
        print("merge: no per-rollout stores found -- nothing to merge.", file=sys.stderr)
        return 0

    report = merge(args.canonical, stores)
    out = args.report or (args.run_dir / "merge_report.json" if args.run_dir else None)
    if out:
        out.write_text(json.dumps(report, indent=2))
    print(f"merged {len(stores)} store(s): +{len(report['added'])} new, "
          f"{len(report['updated'])} updated, {len(report['conflicts'])} conflicts, "
          f"totals {report['totals']}")
    for c in report["conflicts"]:
        print(f"  conflict {c['kind']}:{c['id']} from {c['from']} -> kept {c['kept']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
