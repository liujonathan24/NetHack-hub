"""Per-seed progress for a CLI-harness cell, safe against the aggregation traps.

Usage:  progress.py <cell-dir> [<cell-dir> ...]

Every trap below has cost a real measurement in this project at least once, so
this script exists to stop re-deriving the same fixes in ad-hoc one-liners:

1. ONE ROW PER SEED. A rollout that retried writes a NEW turn file per attempt.
   Globbing `turns/*.ndjson` double-counts: v3_cli/claude_code seed 1 had a
   2-turn stub beside the real 247-turn attempt and was reported as n=6. We key
   on the seed prefix and keep the LONGEST attempt.

2. TRUST TURN FILES OVER `traces.jsonl` METRICS FOR GAME STATE. A retried
   rollout records only its failed final attempt, so `skill_calls` can read 0
   for a rollout that played thousands of turns (b0_flash/prime_agent: metrics
   said 0, turn files held 2,956 calls to dlvl 2). `traces.jsonl` is still the
   authority for `stop_condition`.

3. `tool_calls` IS EMPTY IN CLI-ARM TURN FILES. Those arms dispatch over MCP and
   the env-side record never populates it. Skill usage must come from the trace
   `nodes`.

4. PRIME AGENT NESTS SKILLS INSIDE `ipython`. Counting the outer tool name
   reports 100% `ipython` / 0% everything else. We parse the code payload.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "environments", "nethack"))
from nethack_harness.prompt.balrog import balrog_progress  # noqa: E402

SKILL_RE = re.compile(r"\b(reveal|rollback|np_[a-z_]+|explore_and_descend|move_to|descend)\s*\(")


def _se(v):
    return (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0


def _is_degenerate(stop, calls):
    return stop == "error" or calls < 40 or (stop == "agent_completed" and calls < 80)


def _trace_index(cell):
    """seed -> (stop_condition, skill_calls, died); plus the skill histogram."""
    by_seed, skills = {}, collections.Counter()
    path = os.path.join(cell, "traces.jsonl")
    if not os.path.exists(path):
        return by_seed, skills
    for line in open(path):
        if not line.strip():
            continue
        t = json.loads(line)
        m = t.get("metrics") or {}
        seed = ((t.get("task") or {}).get("data") or {}).get("idx")
        by_seed[seed] = (t.get("stop_condition"), int(m.get("skill_calls") or 0), bool(m.get("died")))
        for node in t.get("nodes") or []:
            msg = node.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            for call in msg.get("tool_calls") or []:
                name = (call.get("name") or "").replace("mcp__nethack__", "")
                if name == "ipython":  # trap 4
                    args = call.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args).get("code", "")
                        except ValueError:
                            pass
                    for s in SKILL_RE.findall(str(args)):
                        skills[s] += 1
                elif name:
                    skills[name] += 1
    return by_seed, skills


def _longest_attempt_per_seed(cell):
    """trap 1: one row per seed, the longest attempt wins."""
    best = {}
    for f in sorted(glob.glob(os.path.join(cell, "turns", "*.ndjson"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        seed = int(os.path.basename(f).split("_")[0])
        if seed not in best or len(rows) > len(best[seed]):
            best[seed] = rows
    return best


def report(cell):
    traces, skills = _trace_index(cell)
    attempts = _longest_attempt_per_seed(cell)
    total_skills = sum(skills.values())
    reveal = skills.get("reveal", 0)

    rows, no_monster = [], 0
    for seed in sorted(attempts):
        rr = attempts[seed]
        dlvl = max(r.get("max_dlvl_reached") or 1 for r in rr)
        xp = max((r.get("status") or {}).get("experience_level") or 1 for r in rr)
        last = rr[-1].get("status") or {}
        no_monster += sum(
            1 for r in rr if "no monster at" in (r.get("rendered_user_message") or "").lower()
        )
        stop, calls, died = traces.get(seed, ("RUNNING", len(rr), last.get("hitpoints") == 0))
        calls = max(calls, len(rr))  # trap 2
        rows.append(dict(seed=seed, turns=len(rr), dlvl=dlvl, xp=xp,
                         balrog=100 * balrog_progress(dlvl, xp),
                         hp=last.get("hitpoints"), maxhp=last.get("max_hitpoints"),
                         stop=stop, calls=calls, died=died))

    name = "/".join(cell.rstrip("/").split("/")[-2:])
    pct = (100 * reveal / total_skills) if total_skills else 0.0
    print(f"== {name}: {len(traces)}/5 finished | skills={total_skills} "
          f"reveal={reveal} ({pct:.1f}%) | no-monster={no_monster}")
    for r in rows:
        print(f"   seed {r['seed']}: {r['turns']:4d}t dlvl={r['dlvl']} XL={r['xp']} "
              f"BALROG={r['balrog']:5.2f}% hp={r['hp']}/{r['maxhp']} {r['stop']}")

    deg = [r for r in rows if _is_degenerate(r["stop"], r["calls"])]
    real = [r for r in rows if r not in deg]
    if real:
        d = [r["dlvl"] for r in real]
        b = [r["balrog"] for r in real]
        c = [r["calls"] for r in real]
        print(f"   -> n={len(real)}  dlvl {st.mean(d):.2f}±{_se(d):.2f}  "
              f"BALROG {st.mean(b):.2f}±{_se(b):.2f}%  "
              f"died {100 * sum(r['died'] for r in real) / len(real):.0f}%  calls {st.mean(c):.0f}")
    print(f"   -> degenerate: {[(r['seed'], r['stop']) for r in deg] if deg else 'none'}")


if __name__ == "__main__":
    for cell in sys.argv[1:]:
        report(cell)
