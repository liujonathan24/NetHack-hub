#!/usr/bin/env python3
"""SIM 4: post-hoc `directive_compliance` classification for one attempt.

The design says each attempt records followed / partially / ignored /
prevented, "scored post-hoc from the trace", and that without it the central
claim ("the orchestrator steered the search") is unfalsifiable. This is the
smallest classifier that can produce those four labels, written down
explicitly so its failure cases are arguable rather than hidden.

RUBRIC
------
A directive is reduced to a small set of CHECKS. Each check reads only
harness-measured trace data (metrics, and the skill calls the model actually
emitted) -- never the model's own prose, which is the thing under test.
Each check returns satisfied / violated / no-evidence.

  followed   every check satisfied, and at least one check had evidence
  ignored    a majority of checks with evidence are violated
  partially  some checks satisfied, some violated
  prevented  the attempt ended (death, budget, error) before any check could
             acquire evidence -- the directive was never really put to the
             test. `prevented` OUTRANKS `ignored`: an attempt that died in
             three calls did not ignore anything.

`prevented` is decided FIRST because it is a statement about the attempt, not
about the model. The order matters: without it, every fast death would be
scored as compliance data and the compliance rate would measure survival.

WHERE THIS IS AMBIGUOUS -- stated rather than hidden, because these are the
cases that will decide real labels:

 1. INTENT VS OUTCOME. An attempt that spends every call trying to reach the
    staircase and never arrives has followed a "descend fast" directive in
    behaviour and failed it in outcome. This classifier scores the ATTEMPT
    (was the goal achieved) and separately reports `pursued`, the behavioural
    signal, so the two are never silently merged. Which one the run reports
    as `directive_compliance` is a decision the experiment must make once and
    write down; the honest label for that case is `prevented`, and the
    behavioural evidence is what makes it defensible.
 2. VACUOUS SATISFACTION. "Do not descend" is satisfied by dying on turn 2.
    The `prevented`-first rule handles the extreme case, but a short attempt
    that survives its budget without ever nearing a staircase still scores
    `followed` on no real evidence. Hence `evidence` counts are reported with
    every label and a label backed by zero positive evidence is flagged.
 3. UNDERSPECIFIED CLAUSES. "fight only weak monsters" needs a monster-strength
    model this harness does not have; it is scored as no-evidence rather than
    guessed. A directive whose every clause is unscoreable should be an
    orchestrator-side warning, not a silent `followed`.
 4. TWO CLAUSES, OPPOSITE VERDICTS. "Descend fast AND do not fight" with a
    descent achieved after three fights is genuinely `partially`; the label is
    right but hides which half failed, so per-check results are always kept.
 5. INSTRUMENTAL VIOLATION -- the one this classifier gets WRONG on real data.
    The SIM 3 `descend_fast` arm was told "do not explore" and called
    `np_explore_level` seven times, which `no_exploring` scores `violated`.
    Its own reasoning says why: "No valid path found. I need to explore more
    first" and "I need to explore more to find the path". It explored IN ORDER
    TO descend, in service of the directive's own goal. A syntactic check
    cannot separate "explored instead of obeying" from "explored in order to
    obey", and scoring the second as a violation understates compliance.
    There is no clean fix that keeps the classifier off the model's prose --
    which is the whole point of the classifier -- so the honest options are
    (a) write directives that prohibit OUTCOMES rather than TOOLS ("do not
    clear the level" rather than "do not explore"), or (b) treat a prohibition
    violated only while the goal check was unmet and being pursued as
    `partially` with an `instrumental` flag. (a) is cheaper and is the
    recommendation: the orchestrator can be told to phrase prohibitions that
    way, and the ledger already shows it which directives scored well.

This classifier is deliberately hand-written per directive, not parsed from
free text. Parsing arbitrary imperative English into checks is a research
project; a run that issues ~20 directives can afford ~20 check definitions,
and each one is auditable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SAT, VIO, NOE = "satisfied", "violated", "no-evidence"

DESCEND_SKILLS = ("np_down", "np_descend", "np_stairs_down")
DESCEND_KEYS = (">",)


def load(rollout: Path) -> dict:
    return json.loads((rollout / "traces.jsonl").read_text().splitlines()[0])


def skill_calls(trace: dict) -> list:
    out = []
    for i, n in enumerate(trace["nodes"]):
        msg = n.get("message") or {}
        if msg.get("role") != "assistant" or not n.get("sampled"):
            continue
        for tc in msg.get("tool_calls") or []:
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
            code = (args or {}).get("code") or ""
            for m in re.finditer(r"nethack\.(\w+)\s*\(([^)]*)\)", code):
                out.append({"node": i, "skill": m.group(1),
                            "args": m.group(2).strip()})
    return out


def tried_to_descend(calls: list) -> bool:
    for c in calls:
        if c["skill"] in DESCEND_SKILLS:
            return True
        if c["skill"] == "np_press_key" and any(
                f'"{k}"' in c["args"] or f"'{k}'" in c["args"]
                for k in DESCEND_KEYS):
            return True
    return False


# ---- check builders ------------------------------------------------------ #

def check_no_descent(trace, calls) -> dict:
    """'Do NOT descend' -- violated by any descent, or any attempt at one."""
    n = int(trace["metrics"].get("descent_count") or 0)
    if n > 0:
        return {"name": "no_descent", "result": VIO,
                "why": f"descent_count={n} (harness metric)"}
    if tried_to_descend(calls):
        return {"name": "no_descent", "result": VIO,
                "why": "a descend skill/'>' key was called even though no "
                       "descent registered"}
    return {"name": "no_descent", "result": SAT,
            "why": "descent_count=0 and no descend skill or '>' key in any "
                   f"of the {len(calls)} skill calls"}


def check_reach_xl(trace, calls, target=3) -> dict:
    xl = int(trace["metrics"].get("max_xp_level") or 0)
    if xl >= target:
        return {"name": f"reach_xl{target}", "result": SAT,
                "why": f"max_xp_level={xl}"}
    return {"name": f"reach_xl{target}", "result": VIO,
            "why": f"max_xp_level={xl} < {target}"}


def check_did_descend(trace, calls) -> dict:
    n = int(trace["metrics"].get("descent_count") or 0)
    if n > 0:
        return {"name": "did_descend", "result": SAT,
                "why": f"descent_count={n}"}
    return {"name": "did_descend", "result": VIO,
            "why": "descent_count=0"}


def check_no_fighting(trace, calls) -> dict:
    fights = [c for c in calls if "attack" in c["skill"]]
    if fights:
        return {"name": "no_fighting", "result": VIO,
                "why": f"{len(fights)} attack skill call(s): "
                       + ", ".join(f"{c['skill']}({c['args']})" for c in fights[:3])}
    return {"name": "no_fighting", "result": SAT,
            "why": "no attack skill calls"}


def check_no_exploring(trace, calls) -> dict:
    exp = [c for c in calls if c["skill"] == "np_explore_level"]
    if exp:
        return {"name": "no_exploring", "result": VIO,
                "why": f"{len(exp)} np_explore_level call(s)"}
    return {"name": "no_exploring", "result": SAT,
            "why": "no np_explore_level calls"}


def check_weak_monsters_only(trace, calls) -> dict:
    return {"name": "weak_monsters_only", "result": NOE,
            "why": "no monster-strength model in this harness; deliberately "
                   "unscored rather than guessed (ambiguity 3)"}


#: directive tag -> the checks it reduces to, and the behavioural "did it even
#: try" signal used to separate `prevented` from `ignored`.
RUBRICS = {
    "no_descend": {
        "checks": [check_no_descent, check_reach_xl, check_weak_monsters_only],
        "pursued": lambda t, c: any("attack" in x["skill"] for x in c),
        "pursued_desc": "fought at least one monster (the XP-seeking half of "
                        "the directive)",
    },
    "descend_fast": {
        "checks": [check_did_descend, check_no_fighting, check_no_exploring],
        "pursued": lambda t, c: any(
            x["skill"] in ("np_move_to",) or x["skill"] in DESCEND_SKILLS
            for x in c),
        "pursued_desc": "moved toward or tried to take the down staircase",
    },
}


def classify(rollout: Path, tag: str) -> dict:
    trace = load(rollout)
    calls = skill_calls(trace)
    rub = RUBRICS[tag]
    results = [f(trace, calls) for f in rub["checks"]]
    sat = [r for r in results if r["result"] == SAT]
    vio = [r for r in results if r["result"] == VIO]
    noe = [r for r in results if r["result"] == NOE]
    pursued = bool(rub["pursued"](trace, calls))

    budget_exhausted = bool(trace["metrics"].get("budget_exhausted"))
    died = bool(trace["metrics"].get("died"))
    goal_checks = [r for r in results
                   if r["name"] in ("did_descend", "reach_xl3")]
    goal_missed = any(r["result"] == VIO for r in goal_checks)

    # ---- per-clause labels (added after the first run of this classifier) --
    # The first run scored BOTH SIM 3 arms `prevented`, because both hit the
    # 10-call budget with a goal clause unmet -- and that erased the fact that
    # arm A's PROHIBITION ("do NOT descend") was fully and checkably obeyed.
    # A prohibition is scoreable at any budget: not descending in ten calls is
    # real evidence, where reaching XL 3 in ten calls was never possible. So
    # clauses are split by kind and labelled separately, and the single
    # headline label is kept only because the design asks for one field.
    PROHIBITIONS = {"no_descent", "no_fighting", "no_exploring"}
    by_kind = {}
    for kind in ("prohibition", "goal"):
        sel = [r for r in results
               if (r["name"] in PROHIBITIONS) == (kind == "prohibition")]
        s = [r for r in sel if r["result"] == SAT]
        v = [r for r in sel if r["result"] == VIO]
        if not sel:
            lab = "n/a"
        elif v and not s:
            lab = ("prevented" if kind == "goal" and pursued
                   and (budget_exhausted or died) else "ignored")
        elif v:
            lab = "partially"
        elif s:
            lab = "followed"
        else:
            lab = "prevented"
        by_kind[kind] = {"label": lab,
                         "checks": [r["name"] for r in sel],
                         "n_satisfied": len(s), "n_violated": len(v)}

    # prevented FIRST: the attempt ran out before the directive could be
    # tested, but the behaviour shows it was being pursued.
    if goal_missed and pursued and (budget_exhausted or died):
        label = "prevented"
        why = (f"the attempt ended on "
               f"{'death' if died else 'call budget'} with the directive's "
               f"goal check still unmet, and the behaviour shows it was being "
               f"pursued ({rub['pursued_desc']}). Scored `prevented`, not "
               f"`ignored`.")
    elif vio and not sat:
        label, why = "ignored", f"{len(vio)} check(s) violated, none satisfied"
    elif vio:
        label, why = "partially", (f"{len(sat)} satisfied, {len(vio)} violated")
    elif sat:
        label, why = "followed", f"all {len(sat)} scoreable check(s) satisfied"
    else:
        label, why = "prevented", "no check acquired any evidence"

    out = {
        "rollout": str(rollout), "rubric": tag,
        "directive": (rollout / "directive.txt").read_text(),
        "label": label, "why": why,
        "by_clause_kind": by_kind,
        "pursued": pursued, "pursued_desc": rub["pursued_desc"],
        "checks": results,
        "n_satisfied": len(sat), "n_violated": len(vio),
        "n_no_evidence": len(noe),
        "flag_no_positive_evidence": (label == "followed" and not sat),
        "metrics": {k: trace["metrics"].get(k) for k in
                    ("descent_count", "max_dlvl_reached", "max_xp_level",
                     "skill_calls", "died", "budget_exhausted")},
        "skill_calls": [f"{c['skill']}({c['args']})" for c in calls],
    }
    (rollout / "compliance.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv) -> int:
    pairs = [tuple(a.split("=", 1)) for a in argv]
    for tag, path in pairs:
        r = classify(Path(path), tag)
        print("=" * 78)
        print(f"{Path(path).name}  rubric={tag}")
        print(f"  directive: {r['directive']}")
        print(f"  LABEL: {r['label'].upper()}   ({r['why']})")
        for k, v in r["by_clause_kind"].items():
            print(f"    clause-kind {k:<12} -> {v['label'].upper():<10} "
                  f"({v['n_satisfied']} sat / {v['n_violated']} vio: "
                  f"{', '.join(v['checks']) or 'none'})")
        print(f"  pursued={r['pursued']}  ({r['pursued_desc']})")
        for c in r["checks"]:
            print(f"    [{c['result']:<11}] {c['name']:<20} {c['why']}")
        print(f"  metrics: {json.dumps(r['metrics'])}")
        if r["flag_no_positive_evidence"]:
            print("  !! FLAG: label rests on zero positive evidence "
                  "(ambiguity 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
