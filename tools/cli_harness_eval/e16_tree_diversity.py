#!/usr/bin/env python3
"""Is the E16 archive a branching TREE or a dressed-up chain?

A chain has one leaf, no branch points, and its best lineage contains every
node. Anything that departs from those three is search that spread out. The
last two columns are the ones that answer "not just tracing one path": how much
of the archive sits OFF the winning lineage, and how many separate attempts
contributed the states that lineage is stitched from.

    python3 e16_tree_diversity.py RUN_DIR [RUN_DIR ...]
"""
import json, math, os, sys, glob, collections


def load(run):
    arc = {}
    for p in glob.glob(os.path.join(run, "archive", "c*", "meta.json")):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        if "id" in m:
            arc[int(m["id"])] = m
    return arc


def stats(run):
    arc = load(run)
    if not arc:
        return None
    kids = collections.defaultdict(list)
    for i, m in arc.items():
        p = m.get("parent")
        if p is not None and int(p) in arc:
            kids[int(p)].append(i)
    n = len(arc)
    leaves = [i for i in arc if not kids[i]]
    branch = [i for i in arc if len(kids[i]) > 1]
    roots = [i for i in arc if m_parent(arc, i) is None]

    # Best lineage: highest BALROG-min, walk to root.
    best = max(arc.values(), key=lambda m: (m.get("balrog_min") or 0.0,
                                            m.get("balrog") or 0.0))
    chain, cur, seen = [], best, set()
    while cur is not None and int(cur["id"]) not in seen:
        seen.add(int(cur["id"]))
        chain.append(cur)
        p = cur.get("parent")
        cur = arc.get(int(p)) if p is not None else None
    on_path = {int(c["id"]) for c in chain}
    atts = {c.get("created_in_attempt") for c in chain
            if c.get("created_in_attempt") is not None}

    # Effective number of independent lines: exp(Shannon entropy) over the
    # sizes of the subtrees hanging off the root's descendants at each branch
    # point -- a chain gives 1, k equal branches give k.
    sizes = []
    for b in branch:
        for k in kids[b]:
            sizes.append(subtree_size(kids, k))
    tot = sum(sizes) or 1
    H = -sum((s / tot) * math.log(s / tot) for s in sizes if s > 0)
    eff = math.exp(H) if sizes else 1.0

    # Distinct resume points actually used by the orchestrator.
    resumed = set()
    ap = os.path.join(run, "attempts.jsonl")
    if os.path.exists(ap):
        for l in open(ap):
            try:
                resumed.add(json.loads(l).get("from_checkpoint"))
            except Exception:
                pass
    resumed.discard(None)

    return dict(run=os.path.basename(run.rstrip("/")), n=n,
                leaves=len(leaves), branch=len(branch), roots=len(roots),
                maxkids=max((len(v) for v in kids.values()), default=0),
                depth=max(depth_of(arc, i) for i in arc),
                chain=len(chain), off=n - len(chain),
                atts=len(atts), eff=eff, resumed=len(resumed))


def m_parent(arc, i):
    p = arc[i].get("parent")
    return int(p) if p is not None and int(p) in arc else None


def subtree_size(kids, node):
    n, stack = 0, [node]
    while stack:
        x = stack.pop(); n += 1; stack.extend(kids[x])
    return n


def depth_of(arc, i, _memo={}):
    d, seen = 0, set()
    while True:
        p = arc[i].get("parent")
        if p is None or int(p) not in arc or int(p) in seen:
            return d
        seen.add(int(p)); i = int(p); d += 1


def main(runs):
    print(f"{'run':<18}{'nodes':>7}{'leaves':>8}{'branch pts':>12}{'max kids':>10}"
          f"{'tree depth':>12}{'best path':>11}{'OFF path':>10}{'attempts on path':>18}"
          f"{'eff. lines':>12}{'resume pts':>12}")
    agg = collections.Counter()
    for r in runs:
        s = stats(r)
        if not s:
            print(f"{os.path.basename(r):<18}  (no archive)"); continue
        print(f"{s['run']:<18}{s['n']:>7}{s['leaves']:>8}{s['branch']:>12}"
              f"{s['maxkids']:>10}{s['depth']:>12}{s['chain']:>11}"
              f"{s['off']:>7} {100*s['off']/s['n']:>2.0f}%{s['atts']:>18}"
              f"{s['eff']:>12.1f}{s['resumed']:>12}")
        for k in ("n", "leaves", "branch", "off", "chain"):
            agg[k] += s[k]
    if agg["n"]:
        print(f"\n  TOTAL {agg['n']:,} checkpoints, {agg['leaves']:,} leaves, "
              f"{agg['branch']:,} branch points, "
              f"{100*agg['off']/agg['n']:.0f}% of all states lie OFF their "
              f"run's winning lineage.")


if __name__ == "__main__":
    main(sys.argv[1:])
