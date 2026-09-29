#!/usr/bin/env python3
"""Build the round-over-round report the reflection pass never had.

Motivating failure (E13-v2, run v2-corpus6to10-r1, 2026-08-25). The orchestrator
is handed exactly one directory -- the round it just ran -- and told: "delete any
existing entry this round's traces CONTRADICT ... An entry that made things worse
is the most valuable thing you can find."  It is asked to find harmful entries
while being given no way to observe harm. It cannot see the previous round's
scores, it cannot see which entry it added when, and it never sees its own prior
rationale. Round 3 duly wrote `don't descend to dlvl > XL x 2, gain levels first`,
a rule that reads as caution and is a treadmill: melee share rose on 4/4 seeds
(+16.9 points), experience level went DOWN (-0.25), and depth fell 2.75 levels.
Nothing in that round's own traces contradicts the rule -- only the comparison
with the round before does.

Emits JSON on stdout:
  rounds[]        per-round aggregates, seed-matched deltas against round 1
  per_seed[]      every seed's (dlvl, XL, BALROG, calls, stop) across all rounds
  entries[]       each store entry: when it appeared, which rounds ran under it,
                  and how the seeds moved while it was in force
  regressions[]   entries in force across a round that got materially worse

Read-only. Never touches the store.
"""
from __future__ import annotations
import json, pathlib, re, statistics, sys, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "environments" / "nethack"))
try:
    from nethack_harness.prompt.balrog import balrog_both
except Exception:                                     # pragma: no cover
    balrog_both = None

CELL = "corpus__prime_agent"


def _seed_rows(cell: pathlib.Path) -> dict[int, dict]:
    """Per-seed outcome for one round, keyed by the REAL seed.

    Only seeds with a completed trace row are returned: turn files are written
    during play, so scoring one mid-flight reports a half-finished game as a
    result.
    """
    tr = cell / "traces.jsonl"
    if not tr.exists() or not tr.stat().st_size:
        return {}
    out: dict[int, dict] = {}
    for line in tr.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        data = (r.get("task") or {}).get("data") or {}
        seed = data.get("seed")
        if seed is None:
            continue
        m = r.get("metrics") or {}
        out[int(seed)] = {
            "seed": int(seed),
            "dlvl": int(m.get("max_dlvl_reached") or 1),
            # max_xp_level is published as of the BALROG-metrics change; fall
            # back to the turn files for runs recorded before it.
            "xl": int(m.get("max_xp_level") or 0) or None,
            "balrog_pct": m.get("balrog_pct"),
            "balrog_min_pct": m.get("balrog_min_pct"),
            "calls": int(m.get("skill_calls") or 0),
            "died": bool(m.get("died")),
            "stop": r.get("stop_condition"),
        }
    for seed, row in out.items():
        if row["xl"] is None:
            row["xl"] = _xl_from_turns(cell, seed)
        if row["balrog_pct"] is None and balrog_both is not None:
            hi, lo = balrog_both(row["dlvl"], row["xl"] or 1)
            row["balrog_pct"], row["balrog_min_pct"] = 100.0 * hi, 100.0 * lo
        row["xp_carried"] = bool(
            (row["balrog_pct"] or 0) > 0 and (row["balrog_min_pct"] or 0) == 0
        )
    return out


def _xl_from_turns(cell: pathlib.Path, seed: int) -> int:
    """Deepest experience level, recovered from the turn stream."""
    xl = 1
    for f in (cell / "turns").glob(f"{seed}_*.ndjson"):
        for line in f.open():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            st = d.get("status") or {}
            if isinstance(st, dict):
                xl = max(xl, int(st.get("experience_level") or 1))
    return xl


def _agg(rows: dict[int, dict]) -> dict:
    if not rows:
        return {}
    v = list(rows.values())
    return {
        "n": len(v),
        "mean_dlvl": statistics.mean(x["dlvl"] for x in v),
        "mean_xl": statistics.mean(x["xl"] for x in v),
        "mean_balrog": statistics.mean(x["balrog_pct"] or 0 for x in v),
        "mean_balrog_min": statistics.mean(x["balrog_min_pct"] or 0 for x in v),
        "mean_calls": statistics.mean(x["calls"] for x in v),
        "died": sum(1 for x in v if x["died"]),
        "cap_hits": sum(1 for x in v if x["stop"] == "call_budget_exhausted"),
        "xp_carried": sum(1 for x in v if x["xp_carried"]),
    }


def _delta(cur: dict[int, dict], base: dict[int, dict]) -> dict:
    """Seed-matched delta. Comparing a partial round's mean against a full base
    mean reads arrival order as a regression -- that false signal halted a run."""
    shared = sorted(set(cur) & set(base))
    if not shared:
        return {"shared_seeds": 0}
    f = lambda d, k: statistics.mean(d[s][k] or 0 for s in shared)
    return {
        "shared_seeds": len(shared),
        "seeds": shared,
        "d_dlvl": f(cur, "dlvl") - f(base, "dlvl"),
        "d_xl": f(cur, "xl") - f(base, "xl"),
        "d_balrog": f(cur, "balrog_pct") - f(base, "balrog_pct"),
        "d_balrog_min": f(cur, "balrog_min_pct") - f(base, "balrog_min_pct"),
        "d_calls": f(cur, "calls") - f(base, "calls"),
        "per_seed": {s: {"d_dlvl": cur[s]["dlvl"] - base[s]["dlvl"],
                         "d_xl": cur[s]["xl"] - base[s]["xl"]} for s in shared},
    }


def build(run_dir: pathlib.Path) -> dict:
    # Only directories named exactly round<N>. A sibling like
    # "round7.wedged-attempt" (an archived failed attempt) used to reach
    # int("7.wedged-attempt") and raise, which made the orchestrator fall back
    # to "reflecting on this round alone" -- silently dropping the scoreboard
    # that is the whole point of this file.
    rounds = sorted(
        (d for d in run_dir.glob("round*")
         if d.is_dir() and d.name[5:].isdigit()),
        key=lambda p: int(p.name[5:]))
    per_round: list[dict] = []
    rows_by_round: dict[str, dict[int, dict]] = {}
    for rd in rounds:
        rows = _seed_rows(rd / CELL)
        if not rows:
            continue
        rows_by_round[rd.name] = rows
        per_round.append({"round": rd.name, **_agg(rows)})

    names = list(rows_by_round)
    base = rows_by_round[names[0]] if names else {}
    for entry in per_round:
        rows = rows_by_round[entry["round"]]
        entry["vs_round1"] = _delta(rows, base) if entry["round"] != names[0] else None
        i = names.index(entry["round"])
        entry["vs_prev"] = _delta(rows, rows_by_round[names[i - 1]]) if i else None

    # --- entry provenance: track CONTENT, not just presence -----------------
    # Two traps here, both hit on the first real run:
    #  * `update_memory` rewrites content but keeps the original id AND title,
    #    so an entry titled "Rest to full HP before descending stairs" can carry
    #    the text "Don't descend to dlvl > XL x 2". Keying on the title reports
    #    the wrong rule. Hash the content.
    #  * every entry is "in force" during a bad round, so flagging all of them
    #    correlates nothing. Only an entry that was ADDED or CHANGED going into
    #    a round is a causal candidate for that round's move; entries carried
    #    unchanged from a round that went fine are not.
    import hashlib

    def _h(txt: str) -> str:
        return hashlib.sha256((txt or "").encode()).hexdigest()[:12]

    entries: dict[tuple[str, str], dict] = {}
    prev_seen: dict[tuple[str, str], str] = {}
    for rd in rounds:
        before = rd / CELL / "harness_state.before.json"
        if not before.exists():
            continue
        try:
            st = json.loads(before.read_text())
        except Exception:
            continue
        this_seen: dict[tuple[str, str], str] = {}
        for kind, items in (st.get("entries") or {}).items():
            for eid, e in (items or {}).items():
                key = (kind, eid)
                digest = _h(e.get("content"))
                this_seen[key] = digest
                rec = entries.setdefault(key, {
                    "id": eid, "kind": kind, "title": e.get("title"),
                    "content": e.get("content"), "version": e.get("version"),
                    "created_at": e.get("created_at"),
                    "rounds_in_force": [], "changed_before": [],
                })
                rec["version"], rec["content"] = e.get("version"), e.get("content")
                rec["title"] = e.get("title")
                rec["rounds_in_force"].append(rd.name)
                if prev_seen.get(key) != digest:      # new, or rewritten
                    rec["changed_before"].append(rd.name)
        for key in set(prev_seen) - set(this_seen):
            entries.setdefault(key, {"id": key[1], "kind": key[0], "title": None,
                                     "content": None, "version": None,
                                     "created_at": None, "rounds_in_force": [],
                                     "changed_before": []})
            entries[key].setdefault("deleted_before", []).append(rd.name)
        prev_seen = this_seen

    entries = list(entries.values())
    by_round = {r["round"]: r for r in per_round}
    for rec in entries:
        # Attribute only the rounds this entry actually changed going into.
        rec["outcome_when_changed"] = [
            {"round": rn,
             "d_dlvl": by_round[rn]["vs_prev"].get("d_dlvl"),
             "d_xl": by_round[rn]["vs_prev"].get("d_xl"),
             "d_balrog": by_round[rn]["vs_prev"].get("d_balrog")}
            for rn in rec["changed_before"]
            if rn in by_round and by_round[rn].get("vs_prev")
        ]

    # Group suspicion BY ROUND. When k entries all changed at the same boundary
    # and that round regressed, the round cannot tell them apart -- listing each
    # as an independent suspect implies a discrimination the data does not
    # support. Say how many changed together, and mark the case where exactly
    # one did (the only situation where attribution is clean).
    regressions = []
    for rn, row in by_round.items():
        d = row.get("vs_prev")
        if not d:
            continue
        bad = (d.get("d_dlvl") or 0) <= -1.0 or (d.get("d_xl") or 0) <= -0.5
        if not bad:
            continue
        changed = [r for r in entries if rn in r["changed_before"]]
        regressions.append({
            "round": rn,
            "d_dlvl": d.get("d_dlvl"), "d_xl": d.get("d_xl"),
            "d_balrog": d.get("d_balrog"), "shared_seeds": d.get("shared_seeds"),
            "n_changed": len(changed),
            "discriminating": len(changed) == 1,
            "candidates": [{"id": r["id"], "kind": r["kind"], "title": r["title"],
                            "content": r["content"]} for r in changed],
        })

    # The reflection pass's OWN past reasoning. Without this it re-derives its
    # rules from scratch every round and cannot notice that it already tried
    # something -- it has no memory of its own arguments, only of their output.
    rationales = []
    for rd in rounds:
        f = rd / "orchestrator_rationale.json"
        if not f.exists():
            continue
        try:
            edits = json.loads(f.read_text())
        except Exception:
            continue
        rationales.append({"round": rd.name, "edits": edits})

    return {"run": run_dir.name, "rounds": per_round, "rationales": rationales,
            "per_seed": {rn: list(rows.values()) for rn, rows in rows_by_round.items()},
            "entries": entries, "regressions": regressions}


def render(rep: dict) -> str:
    """Compact text for a model prompt -- JSON is for machines, this is read."""
    L = [f"ROUND-OVER-ROUND REPORT for {rep['run']}", ""]
    L.append(f"{'round':<8}{'n':>3}{'dlvl':>7}{'XL':>6}{'BALmax':>9}{'BALmin':>9}"
             f"{'calls':>7}{'died':>6}{'cap':>5}   vs round 1 (seed-matched)")
    for r in rep["rounds"]:
        d = r.get("vs_round1")
        tail = "base" if d is None else (
            f"dlvl {d['d_dlvl']:+.2f}  XL {d['d_xl']:+.2f}  BALmax {d['d_balrog']:+.2f}"
            f"  ({d['shared_seeds']} seeds)")
        L.append(f"{r['round']:<8}{r['n']:>3}{r['mean_dlvl']:>7.2f}{r['mean_xl']:>6.2f}"
                 f"{r['mean_balrog']:>9.2f}{r['mean_balrog_min']:>9.2f}{r['mean_calls']:>7.0f}"
                 f"{r['died']:>4}/{r['n']}{r['cap_hits']:>5}   {tail}")
    L += ["", "PER SEED (dlvl / XL):"]
    for rn, rows in rep["per_seed"].items():
        L.append(f"  {rn:<8} " + "  ".join(
            f"s{x['seed']}:{x['dlvl']}/{x['xl']}" + ("*cap" if x["stop"] == "call_budget_exhausted" else "")
            for x in sorted(rows, key=lambda y: y["seed"])))
    L += ["", "STORE ENTRIES. `changed` marks the rounds an entry was added or "
              "rewritten; only those rounds are attributed to it. NOTE: update_memory "
              "keeps the original title, so an entry's title may describe an older "
              "version of its text -- trust the content, not the title."]
    for e in rep["entries"]:
        if not e.get("rounds_in_force"):
            continue
        L.append(f"  [{e['kind']}] {e['title']!r} (v{e['version']}, in force: "
                 f"{', '.join(e['rounds_in_force'])}; changed before: "
                 f"{', '.join(e['changed_before']) or 'never'})")
        L.append(f"      text: {(e['content'] or '')[:180]}")
        for o in e["outcome_when_changed"]:
            L.append(f"      -> {o['round']} moved: dlvl {o['d_dlvl']:+.2f}  "
                     f"XL {o['d_xl']:+.2f}  BALmax {o['d_balrog']:+.2f} (vs prev round)")
    if rep.get("rationales"):
        L += ["", "YOUR OWN REASONING IN PREVIOUS ROUNDS -- what you changed, and why "
                  "you thought it would help. Read it against the scoreboard above: a "
                  "rule you argued for that was followed by a drop is the strongest "
                  "evidence you have."]
        for r in rep["rationales"]:
            L.append(f"  {r['round']}:")
            for e in r["edits"]:
                ev = e.get("evidence")
                ev = ev if isinstance(ev, str) else json.dumps(ev) if ev else ""
                L.append(f"    [{e.get('action','edit')}] {e.get('title') or e.get('id','')}")
                if ev:
                    L.append(f"        evidence you cited: {ev[:220]}")
                if e.get("expected_effect"):
                    L.append(f"        effect you expected: {str(e['expected_effect'])[:220]}")
    if rep["regressions"]:
        L += ["", "ROUNDS THAT GOT MATERIALLY WORSE, AND WHAT HAD JUST CHANGED:"]
        for r in rep["regressions"]:
            L.append(f"  {r['round']}: dlvl {r['d_dlvl']:+.2f}  XL {r['d_xl']:+.2f}  "
                     f"BALmax {r['d_balrog']:+.2f}  over {r['shared_seeds']} seeds")
            if r["discriminating"]:
                L.append("     ONE entry changed going into this round, so the "
                         "regression is attributable to it:")
            else:
                L.append(f"     {r['n_changed']} entries changed together going into "
                         "this round, so this round CANNOT tell them apart. Treat the "
                         "list as the suspect set, not as individual verdicts:")
            for c in r["candidates"]:
                L.append(f"       - {c['title']!r}: {(c['content'] or '')[:180]}")
        L += ["", "To attribute a regression to a single entry, change one thing at a "
                  "time: revise one entry per round and leave the rest alone."]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--text", action="store_true", help="human/model-readable instead of JSON")
    a = ap.parse_args()
    rep = build(a.run_dir)
    print(render(rep) if a.text else json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
