#!/usr/bin/env python3
"""Audit that an E16 run's SERVED BYTES match its ICLR arm (A1/A2/A3/full).

Reads what the players and the orchestrator were actually sent -- the trace
files, the served directive and ledger files, the selection and round logs --
and checks each against the arm's contract. A check that cannot be evaluated
is reported as such, never as a pass.

    python e16_audit_condition.py RUN_DIR --arm A2 [--json OUT]

Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BANNER = "[RESUMED FROM CHECKPOINT"
DIRECTIVE = "[ORCHESTRATOR DIRECTIVE for this attempt:"
BLIND_LABEL = "TRANSCRIPT SO FAR (your own earlier turns in this game"
FULL_PREFIX_LABEL = "THE PLAN THAT WAS LIVE WHEN THIS STATE WAS SAVED"
LEDGER_MARKERS = ("BRANCH ", "EQUALLY CHOOSABLE", "previous attempt(s) started",
                  "lesson:", "LESSON", "ATTEMPT HISTORY", "attempt ")
FIXED_NOTICE = "WILL resume checkpoint c"
BLIND_RESUME_PROMPT = "Continue playing."   # == e16_orchestrator.BLIND_RESUME_PROMPT
CHOICE_TEXT = "Choose the checkpoint the next player resumes from"

ARMS = {
    "A1": dict(selector="llm", directive=False, blind=True, rounds=True),
    "A2": dict(selector="pre_death", directive=False, blind=True, rounds=False),
    "A3": dict(selector="pre_death", directive=True, blind=False, rounds=True),
    "full": dict(selector="llm", directive=True, blind=False, rounds=True),
}
SOURCE_FOR = {"A1": "llm", "A2": "pre_death", "A3": "pre_death+llm_directive",
              "full": "llm"}


def _texts(msg) -> list:
    """Every text fragment in one trace message, whatever its shape."""
    out = []
    if isinstance(msg, str):
        return [msg]
    if isinstance(msg, dict):
        for k in ("content", "reasoning_content", "text"):
            v = msg.get(k)
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, list):
                for part in v:
                    out.extend(_texts(part))
        for tc in msg.get("tool_calls") or []:
            out.extend(_texts(tc))
        if "function" in msg and isinstance(msg["function"], dict):
            out.append(json.dumps(msg["function"]))
    return out


def served_texts(traces_path: Path) -> list:
    """All (role, text) the player was SENT (system/user/tool roles)."""
    out = []
    if not traces_path.is_file():
        return out
    for line in traces_path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for node in rec.get("nodes") or []:
            m = node.get("message")
            if isinstance(m, str):
                try:
                    m = json.loads(m)
                except Exception:
                    m = {"role": "?", "content": m}
            role = (m or {}).get("role", "?") if isinstance(m, dict) else "?"
            if role in ("system", "user", "tool"):
                for t in _texts(m):
                    out.append((role, t))
    return out


def session_records(path: Path) -> list:
    """Parsed records of a Prime Agent session file (unparseable lines kept
    as ``{"_raw": line}`` so counts stay honest)."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"_raw": line})
    return out


def _message_text(rec) -> str:
    m = rec.get("message") if isinstance(rec, dict) else None
    if not isinstance(m, dict):
        return ""
    return "\n".join(_texts(m))


def _strip_header(recs: list) -> list:
    """Records minus the session header (whose id the seal rewrites)."""
    return [r for r in recs if r.get("type") != "session"]


def audit_session_attempt(add, spec, adir: Path, from_ck_dir: Path, scope: str,
                          resume_prompt: str, k: int) -> None:
    """Session-resume checks for one attempt whose prefix_continuity is
    'session': the exported conversation begins with the seed byte-for-byte,
    the next user turn is exactly the recorded resume prompt, and the
    blind/served contract holds on the NEW part of the conversation."""
    exp_dir = adir / "session" / "sessions"
    files = sorted(exp_dir.glob("*.jsonl")) if exp_dir.is_dir() else []
    add("session.exported", len(files) == 1, f"{len(files)} exported session file(s)", scope)
    if len(files) != 1:
        return
    got = session_records(files[0])
    seed = session_records(from_ck_dir / "session" / "session.jsonl")
    add("session.seed_present", bool(seed), f"{len(seed)} seed records from c{from_ck_dir.name.lstrip('c')}", scope)
    if not seed:
        return
    # Header ids must match (the harness copies the seed under its header id
    # and Prime Agent appends to that file); the bodies must be identical.
    hdr_ok = got and seed and got[0].get("type") == "session" and got[0].get("id") == seed[0].get("id")
    add("session.header_id_kept", bool(hdr_ok), f"seed {seed[0].get('id')!r} got {got[0].get('id') if got else None!r}", scope)
    n = len(seed)
    same = len(got) >= n and _strip_header(got[:n]) == _strip_header(seed)
    add("session.prefix_is_the_seed_verbatim", same,
        f"first {n} records identical: {same}; got {len(got)} records", scope)
    # The last seed record must be the tool result at the checkpoint's call,
    # and the FIRST message after the seed must be the resume user turn.
    last_seed = seed[-1]
    add("session.seed_ends_at_a_tool_result",
        (last_seed.get("message") or {}).get("role") == "toolResult",
        f"last seed role {(last_seed.get('message') or {}).get('role')!r}", scope)
    new = [r for r in got[n:] if r.get("type") == "message"]
    first_new = new[0] if new else {}
    role = (first_new.get("message") or {}).get("role")
    text = _message_text(first_new)
    add("session.first_new_turn_is_user", role == "user", f"role {role!r}", scope)
    add("session.first_new_turn_is_the_resume_prompt", text == resume_prompt,
        f"len {len(text)} vs served {len(resume_prompt)}; match {text == resume_prompt}", scope)
    new_text = "\n".join(_message_text(r) for r in new)
    # What the arm may and may not say in the resume turn.
    if spec["blind"]:
        add("session.blind_prompt_exact", resume_prompt == BLIND_RESUME_PROMPT,
            f"{resume_prompt[:60]!r}", scope)
        leaks = {m: new_text.count(m) for m in (BANNER, DIRECTIVE, BLIND_LABEL, FULL_PREFIX_LABEL,
                                                 "EQUALLY CHOOSABLE", "previous attempt(s) started",
                                                 "CHECKPOINT ARCHIVE", "ATTEMPT HISTORY", "LESSON")
                 if m in new_text}
        add("session.blind_no_leaks_in_new_turns", not leaks, f"leaks={leaks}", scope)
    else:
        add("session.banner_in_resume_turn", text.count(BANNER) == 1, f"banner count {text.count(BANNER)}", scope)
        add("session.ledger_in_resume_turn", "CHECKPOINT ARCHIVE" in text or "BRANCH" in text, "", scope)
        add("session.no_quoted_tail", FULL_PREFIX_LABEL not in text and BLIND_LABEL not in text, "", scope)
        if spec["directive"]:
            add("session.directive_in_resume_turn_once", text.count(DIRECTIVE) == 1 and text.startswith(DIRECTIVE),
                f"directive block count {text.count(DIRECTIVE)}", scope)
        else:
            add("session.no_directive_in_resume_turn", DIRECTIVE not in text, "", scope)
    # Continuity of the call counter: the seed's last marker and the first
    # new tool result's marker must be consecutive.
    import re as _re
    seed_marks = [int(x) for x in _re.findall(r"\[call#(\d+)\]", _message_text(last_seed))]
    first_res = next((r for r in new if (r.get("message") or {}).get("role") == "toolResult"), None)
    new_marks = [int(x) for x in _re.findall(r"\[call#(\d+)\]", _message_text(first_res or {}))]
    if seed_marks and new_marks:
        add("session.call_counter_continues", new_marks[0] == max(seed_marks) + 1,
            f"seed ends at [call#{max(seed_marks)}], resumed starts at [call#{new_marks[0]}]", scope)
    else:
        add("session.call_counter_continues", False,
            f"markers not found (seed {seed_marks}, new {new_marks})", scope)


def pre_death_expected(attempts: list, k: int) -> tuple:
    """What the fixed rule should have picked for attempt k (1-based)."""
    if k == 1:
        return None, "first"
    prev = attempts[k - 2]
    new = [str(c).lstrip("c") for c in (prev.get("new_checkpoints") or [])]
    if new:
        return max(new, key=lambda i: (0, int(i)) if i.isdigit() else (1, i)), "latest_of_prev"
    return str(prev.get("from_checkpoint")), "prev_start"


def audit(run_dir: Path, arm: str) -> dict:
    spec = ARMS[arm]
    checks = []

    def add(name, ok, detail="", scope="run"):
        checks.append({"check": name, "ok": ok, "scope": scope, "detail": detail})

    prov = json.loads((run_dir / "provenance.json").read_text()) if (run_dir / "provenance.json").is_file() else {}
    add("provenance.iclr_arm", prov.get("iclr_arm") == arm,
        f"provenance says {prov.get('iclr_arm')!r}")
    add("provenance.selector", (prov.get("selector") or {}).get("mode") == spec["selector"],
        f"selector mode {(prov.get('selector') or {}).get('mode')!r}")

    att_path = run_dir / "attempts.jsonl"
    attempts = [json.loads(l) for l in att_path.read_text().splitlines()] if att_path.is_file() else []
    add("attempts.present", bool(attempts), f"{len(attempts)} attempt rows")
    sel_path = run_dir / "selection.jsonl"
    sels = {}
    if sel_path.is_file():
        for l in sel_path.read_text().splitlines():
            r = json.loads(l)
            sels[int(r["attempt"])] = r

    for a in attempts:
        k = int(a["attempt"])
        scope = f"a{k:03d}"
        adir = Path(a.get("out_dir") or (run_dir / "attempts" / scope))
        if not adir.is_dir():
            adir = run_dir / "attempts" / scope
        dsv = (adir / "directive_served.txt").read_text() if (adir / "directive_served.txt").is_file() else None
        lsv = (adir / "ledger_served.txt").read_text() if (adir / "ledger_served.txt").is_file() else None
        resumed_from_seed = str(a.get("from_checkpoint")) in ("1", "None", "")
        mode = ((adir / "prefix_continuity.txt").read_text().strip()
                if (adir / "prefix_continuity.txt").is_file() else "text_only")
        rp = ((adir / "resume_prompt_served.txt").read_text()
              if (adir / "resume_prompt_served.txt").is_file() else "")
        if prov.get("session_resume"):
            add("session.mode_recorded", mode == a.get("prefix_continuity"),
                f"file {mode!r} record {a.get('prefix_continuity')!r}", scope)
            add("session.no_unsealed_fallback", mode != "fresh_unsealed", mode, scope)
            if mode == "session":
                audit_session_attempt(add, spec, adir,
                                      run_dir / "archive" / f"c{str(a.get('from_checkpoint')).lstrip('c')}",
                                      scope, rp, k)
        # -- what the launcher wrote down as served ----------------------------
        if spec["directive"]:
            add("directive_served.nonempty", bool(dsv and dsv.strip()), f"{(dsv or '')[:60]!r}", scope)
        else:
            add("directive_served.empty", not (dsv or "").strip(), f"{(dsv or '')[:60]!r}", scope)
        if spec["blind"]:
            body = (lsv or "")
            ok = body == "" or body.startswith(BLIND_LABEL)
            leaks = [m for m in (BANNER, FULL_PREFIX_LABEL) + LEDGER_MARKERS
                     if m in body.replace(BLIND_LABEL, "")]
            add("ledger_served.blind", ok and not leaks,
                f"len={len(body)} leaks={leaks}", scope)
        else:
            if k >= 2 and not resumed_from_seed:
                add("ledger_served.has_ledger", bool(lsv) and ("BRANCH" in lsv or "checkpoint" in lsv),
                    f"len={len(lsv or '')}", scope)
            if prov.get("session_resume"):
                add("ledger_served.no_quoted_tail", FULL_PREFIX_LABEL not in (lsv or ""), "", scope)
        # -- what the model was actually sent ----------------------------------
        texts = served_texts(adir / "traces.jsonl")
        if not texts:
            add("traces.readable", False, "no served messages found (empty traces.jsonl?)", scope)
            continue
        # Blocks are counted in what the ENV served (tool-role results). The
        # user-role nodes of a resumed trace carry the resume turn itself and,
        # if Prime Agent's auto-refine were on, its review prompts -- both
        # would double-count. Auto-refine traffic is a failure on its own.
        joined = "\n".join(t for role, t in texts if role == "tool")
        user_joined = "\n".join(t for role, t in texts if role == "user")
        refine = sum(user_joined.count(m) for m in ("<trigger>", "<current_harness_state>", "auto-refine review"))
        add("traces.no_auto_refine_traffic", refine == 0, f"{refine} auto-refine marker(s) in user-role nodes", scope)
        n_banner = joined.count(BANNER)
        n_dir = joined.count(DIRECTIVE)
        n_blind = joined.count(BLIND_LABEL)
        n_fullpre = joined.count(FULL_PREFIX_LABEL)
        if mode == "session":
            # The env served none of the blocks in THIS attempt's own results.
            # The resumed trace also replays the ancestor's results (which may
            # legitimately carry attempt 1's first-observation blocks), so
            # only tool results after the resume turn count.
            tool_after = []
            seen_resume = False
            for role, t in texts:
                if role == "user" and t.strip() == rp.strip() and rp.strip():
                    seen_resume = True
                    continue
                if seen_resume and role == "tool":
                    tool_after.append(t)
            after = "\n".join(tool_after)
            add("traces.resume_turn_found", seen_resume, "", scope)
            leaks = {m: after.count(m) for m in (BANNER, DIRECTIVE, BLIND_LABEL, FULL_PREFIX_LABEL)
                     if m in after}
            add("traces.env_served_no_blocks_in_session_mode", not leaks and seen_resume,
                f"leaks={leaks} in {len(tool_after)} post-resume tool results", scope)
        elif spec["blind"]:
            leaks = {m: joined.count(m) for m in (BANNER, DIRECTIVE, FULL_PREFIX_LABEL, "EQUALLY CHOOSABLE",
                                                   "previous attempt(s) started", "lesson:", "LESSON:")
                     if m in joined}
            add("traces.no_banner_no_directive_no_ledger", not leaks, f"leaks={leaks}", scope)
            if lsv:
                add("traces.transcript_label_once", n_blind == 1, f"label count {n_blind}", scope)
            else:
                add("traces.no_transcript_when_none_served", n_blind == 0, f"label count {n_blind}", scope)
        else:
            if k >= 2 or not resumed_from_seed:
                add("traces.banner_once", n_banner == 1, f"banner count {n_banner}", scope)
            if spec["directive"]:
                add("traces.directive_once_and_matches", n_dir == 1 and (dsv or "").strip() in joined,
                    f"directive block count {n_dir}", scope)
            else:
                add("traces.no_directive_block", n_dir == 0, f"directive block count {n_dir}", scope)
        # -- selection source and the fixed rule -------------------------------
        s = sels.get(k)
        if s is None:
            add("selection.record", False, "no selection record", scope)
        else:
            add("selection.source", s.get("source") == SOURCE_FOR[arm],
                f"source {s.get('source')!r}", scope)
            if spec["selector"] == "pre_death":
                exp, why = pre_death_expected(attempts, k)
                if exp is not None:
                    add("selection.pre_death_rule", str(s.get("chosen_id")) == exp,
                        f"chosen {s.get('chosen_id')} expected {exp} ({why})", scope)
                if arm == "A3":
                    add("selection.fixed_notice_served", bool((s.get("pre_death") or {}).get("fixed_pick_notice_served")),
                        "", scope)
            if arm == "A1":
                add("selection.directive_empty", not (s.get("directive") or "").strip(), "", scope)

    # -- orchestrator side ------------------------------------------------------
    rlog = run_dir / "orchestrator_rounds.jsonl"
    rounds = [json.loads(l) for l in rlog.read_text().splitlines()] if rlog.is_file() else []
    import re as _re
    # first-try decision rounds only; retries carry the retry prompt by design
    decide = [r for r in rounds if _re.fullmatch(r"round\d+", str(r.get("kind", "")))]
    raw = sorted((run_dir / "orchestrator" / "raw").glob("decide_*")) if (run_dir / "orchestrator" / "raw").is_dir() else []
    if not spec["rounds"]:
        add("orchestrator.no_rounds", not decide and not raw,
            f"{len(decide)} decide rounds, {len(raw)} raw decide files")
        add("orchestrator.no_spend", float((prov.get("budget") or {}).get("orchestrator_usd", 0) or 0) == 0
            and not (run_dir / "orchestrator" / "opening_plan.txt").is_file(), "")
    else:
        add("orchestrator.rounds_present", len(decide) >= len(attempts),
            f"{len(decide)} decide rounds for {len(attempts)} attempts")
        for r in decide:
            p = r.get("prompt") or ""
            scope = f"round{r.get('round')}"
            if arm == "A3":
                add("orchestrator.prompt_fixed_notice", FIXED_NOTICE in p and CHOICE_TEXT not in p, "", scope)
            elif arm == "A1":
                add("orchestrator.prompt_selection_only",
                    "NOTHING YOU WRITE REACHES THE PLAYER" in p
                    and "served to the player verbatim" not in p
                    and "Choose the checkpoint the next player resumes from" in p, "", scope)
            else:
                add("orchestrator.prompt_offers_choice", CHOICE_TEXT in p, "", scope)
            add("orchestrator.prompt_has_ledger", "BRANCH" in p or "checkpoint" in p.lower(), "", scope)

    ok = all(c["ok"] for c in checks)
    return {"run_dir": str(run_dir), "arm": arm, "ok": ok, "n_checks": len(checks),
            "n_failed": sum(1 for c in checks if not c["ok"]), "checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rep = audit(Path(args.run_dir), args.arm)
    for c in rep["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['scope']:>9s}  {c['check']:<45s} {c['detail']}")
    print(f"\n{'OK' if rep['ok'] else 'FAILED'}: {rep['n_failed']} of {rep['n_checks']} checks failed  arm={rep['arm']}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
