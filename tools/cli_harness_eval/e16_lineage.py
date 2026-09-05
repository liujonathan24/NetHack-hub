#!/usr/bin/env python3
"""Reconstruct Go-Explore lineages from an E16 archive, cut at N attempts.

WHAT A LINEAGE IS. A Go-Explore attempt resumes a checkpoint, so it starts
partway along the game clock and cannot be plotted against game turns as if it
were a game. The archive is a tree. Take the checkpoint with the best
BALROG-min among those created at or before the cut, walk `parent` links back
to the seed, and that chain is one continuous history from turn 1 in which every
edge was really played. It is a SURVIVING path by selection -- the branches that
died are not on it -- so it is a best-of-search trajectory, not a typical run.

WHY THE CUT USES attempts.jsonl AND NOT meta.created_in_attempt. Six checkpoints
across the finished runs have no `created_in_attempt` key, and filtering on a
missing key silently admits them: an earlier build of this data leaked a
checkpoint created in attempt 69 into a set that was supposed to stop at 30.
`attempts.jsonl` records `new_checkpoints` per attempt as ids with a "c" prefix,
which is authoritative and complete. The seed checkpoint c1 predates every
attempt and is always admitted.

    python3 e16_lineage.py --cut 30 -o out.json RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import os


def load_archive(run):
    """id -> meta, for every checkpoint the archive holds."""
    out = {}
    adir = os.path.join(run, "archive")
    if not os.path.isdir(adir):
        return out
    for name in os.listdir(adir):
        p = os.path.join(adir, name, "meta.json")
        if not os.path.exists(p):
            continue
        try:
            m = json.load(open(p))
        except Exception:
            continue
        if "id" in m:
            out[int(m["id"])] = m
    return out


def admitted(run, cut):
    """Checkpoint ids created at or before attempt `cut`, plus the seed.

    Returns None when the run has no attempts.jsonl, meaning "no cut applied".
    """
    p = os.path.join(run, "attempts.jsonl")
    if not os.path.exists(p):
        return None
    keep = {1}
    for ln in open(p):
        try:
            a = json.loads(ln)
        except Exception:
            continue
        if cut is not None and (a.get("attempt") or 0) > cut:
            continue
        for c in a.get("new_checkpoints") or []:
            s = str(c)
            keep.add(int(s[1:] if s.startswith("c") else s))
    return keep


def lineage(run, cut):
    arc = load_archive(run)
    if not arc:
        return None
    keep = admitted(run, cut)
    pool = {k: v for k, v in arc.items() if keep is None or k in keep}
    if not pool:
        return None

    # Best by BALROG-min, ties broken by BALROG-max then depth, so a tie does
    # not depend on dict ordering.
    best = max(pool.values(), key=lambda m: (m.get("balrog_min") or 0.0,
                                             m.get("balrog") or 0.0,
                                             m.get("dlvl") or 0))
    chain, seen = [], set()
    cur = best
    while cur is not None:
        cid = int(cur["id"])
        if cid in seen:                      # defensive: a cycle would hang
            break
        seen.add(cid)
        chain.append(cur)
        par = cur.get("parent")
        cur = arc.get(int(par)) if par is not None else None
    chain.reverse()

    pts = [[m.get("gameturn") or 0,
            100.0 * (m.get("balrog_min") or 0.0),
            100.0 * (m.get("balrog") or 0.0),
            m.get("dlvl") or 0,
            m.get("xl") or 0] for m in chain]
    return {
        "run": os.path.basename(run.rstrip("/")),
        "seed": (best.get("seed") or [None])[0],
        "n_checkpoints_pool": len(pool),
        "chain_len": len(chain),
        "best_id": int(best["id"]),
        "best_balrog_min": 100.0 * (best.get("balrog_min") or 0.0),
        "best_balrog_max": 100.0 * (best.get("balrog") or 0.0),
        "best_dlvl": best.get("dlvl") or 0,
        "best_xl": best.get("xl") or 0,
        "end_turn": pts[-1][0] if pts else 0,
        "pts": pts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cut", type=int, default=None,
                    help="include only checkpoints created at or before this "
                         "attempt number; omit for the whole run")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    res = [x for x in (lineage(r, a.cut) for r in a.runs) if x]
    doc = {"cut": a.cut, "lineages": res}
    s = json.dumps(doc, separators=(",", ":"))
    if a.out:
        open(a.out, "w").write(s)
        print(f"wrote {a.out}: {len(res)} lineages, cut={a.cut}")
        for x in res:
            print(f"  {x['run']:<18} seed={x['seed']} chain={x['chain_len']:<5}"
                  f" min={x['best_balrog_min']:6.2f} max={x['best_balrog_max']:6.2f}"
                  f" dlvl={x['best_dlvl']:<3} xl={x['best_xl']}")
    else:
        print(s)


if __name__ == "__main__":
    main()
