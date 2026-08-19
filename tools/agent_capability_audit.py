#!/usr/bin/env python3
"""Guardrail: per-rollout audit of which Prime Agent capabilities actually fired.

    python tools/agent_capability_audit.py <cell_dir> [<cell_dir> ...] [--json out.json]

For every rollout in a cell (traces.jsonl + the per-rollout agent dir under
/tmp/vf-prime-agent/agent-<trace.id>), report MODEL-initiated scaffold use
(parsed from the ipython code the model wrote) and SCAFFOLD-initiated activity
(parsed from the agent's own logs/stdout):

  model-initiated                     scaffold-initiated
  ---------------                     ------------------
  rlm_spawns        await rlm("...")  compactions_threshold  auto-compaction fired
  memory_writes     rlm.harness.*     auto_refine            (expected 0: --no-session)
  compact_requests  compact.run()     rlm_child_events       child session updates
  refine_requests   refine.run()
  websearch_calls   websearch(...)    context: peak prompt tokens vs the
  rlm_curiosity     imports/reads rlm    compaction threshold (window-16384)

Memory writes also capture the WRITTEN TEXT (the string args), so "what did it
add to memory" is answerable per game, not just countable. Never raises on
missing artifacts -- absent evidence is reported as absent, not skipped.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys

AGENT_ROOT = "/tmp/vf-prime-agent"

RX = {
    # rlm("task"...) / await rlm(  -- NOT rlm.harness.* / rlm.list_* / from rlm import
    "rlm_spawns": re.compile(r"(?:await\s+)?\brlm\s*\(\s*[\"'f]"),
    "memory_writes": re.compile(r"rlm\.harness\.(create|update|delete)_(memory|skill|subagent|prompt_note)\s*\("),
    "compact_requests": re.compile(r"\bcompact\.run\s*\("),
    "refine_requests": re.compile(r"\brefine\.run\s*\("),
    "websearch_calls": re.compile(r"\bwebsearch\s*\("),
    "rlm_curiosity": re.compile(r"(?:from\s+rlm\s+import|import\s+rlm\b|inspect\.getsource\s*\(\s*rlm|rlm\.mcp_base)"),
}
# capture the text being memorized: first long string literal in the call
MEM_TEXT = re.compile(
    r"rlm\.harness\.(?:create|update)_\w+\s*\(([^)]*)\)", re.S)
STRLIT = re.compile(r"[\"']([^\"']{10,400})[\"']")


def audit_rollout(trace: dict) -> dict:
    tid = trace.get("id") or ""
    counts = {k: 0 for k in RX}
    mem_texts: list[str] = []
    ipython_calls = 0
    for n in trace.get("nodes") or []:
        m = n.get("message") or {}
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if tc.get("name") == "ipython":
                ipython_calls += 1
            code = tc.get("arguments") or ""
            if isinstance(code, dict):
                code = json.dumps(code)
            for k, rx in RX.items():
                counts[k] += len(rx.findall(code))
            for call in MEM_TEXT.findall(code):
                lit = STRLIT.search(call)
                if lit:
                    mem_texts.append(lit.group(1))
    # context pressure vs compaction threshold
    pts = [c.get("usage", {}).get("prompt_tokens") or 0 for c in trace.get("calls") or []]
    ctx_window = 1048576
    try:
        h = ((trace.get("agent") or {}).get("harness") or {})
        ctx_window = int(h.get("context_window") or ctx_window)
    except Exception:
        pass
    threshold = ctx_window - 16384

    # scaffold-side evidence from the per-rollout agent dir
    adir = os.path.join(AGENT_ROOT, f"agent-{tid}")
    scaffold = {"compactions_threshold": 0, "compactions_manual": 0,
                "auto_refine": 0, "rlm_child_events": 0,
                "harness_state_files": [], "agent_dir_present": os.path.isdir(adir)}
    if scaffold["agent_dir_present"]:
        for hs in glob.glob(os.path.join(adir, "**", "harness_state.json"), recursive=True):
            try:
                scaffold["harness_state_files"].append(
                    {"path": hs, "content": json.load(open(hs))})
            except Exception:
                scaffold["harness_state_files"].append({"path": hs, "content": "unreadable"})
        for log in (glob.glob(os.path.join(adir, "logs", "*.jsonl"))
                    + glob.glob(os.path.join(adir, "program.stdout.txt"))):
            try:
                text = open(log, errors="replace").read()
            except Exception:
                continue
            scaffold["compactions_threshold"] += len(re.findall(r'"reason"\s*:\s*"threshold"', text)) \
                + text.count("compaction (threshold")
            scaffold["compactions_manual"] += len(re.findall(r'"reason"\s*:\s*"manual"', text))
            scaffold["auto_refine"] += len(re.findall(r"auto.?refine", text, re.I))
            scaffold["rlm_child_events"] += text.count("rlm_child_update")

    return {
        "trace_id": tid[:12],
        "seed": ((trace.get("task") or {}).get("data") or {}).get("idx"),
        "ipython_calls": ipython_calls,
        **counts,
        "memory_texts": mem_texts,
        "peak_prompt_tokens": max(pts) if pts else 0,
        "compaction_threshold": threshold,
        "compaction_reachable": bool(pts) and max(pts) >= threshold,
        **scaffold,
    }


def audit_cell(cell_dir: str) -> list[dict]:
    tf = os.path.join(cell_dir, "traces.jsonl")
    out = []
    if not os.path.exists(tf):
        return out
    for line in open(tf):
        try:
            out.append(audit_rollout(json.loads(line)))
        except Exception as e:
            out.append({"error": f"{type(e).__name__}: {e}"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dirs", nargs="+")
    ap.add_argument("--json", help="also write full JSON here")
    args = ap.parse_args(argv)
    everything = {}
    for cd in args.cell_dirs:
        rows = audit_cell(cd)
        everything[cd] = rows
        print(f"\n=== {cd}  ({len(rows)} rollouts) ===")
        hdr = ("seed ipython rlm_spawn mem_write compact_req refine_req websearch "
               "rlm_curiosity peak_tok cmp_thresh auto_cmp child_ev")
        print(hdr)
        for r in rows:
            if "error" in r:
                print("  ERROR:", r["error"]); continue
            print(f"{str(r['seed']):>4} {r['ipython_calls']:>7} {r['rlm_spawns']:>9} "
                  f"{r['memory_writes']:>9} {r['compact_requests']:>11} {r['refine_requests']:>10} "
                  f"{r['websearch_calls']:>9} {r['rlm_curiosity']:>13} {r['peak_prompt_tokens']:>8} "
                  f"{'YES' if r['compaction_reachable'] else 'no':>10} "
                  f"{r['compactions_threshold']:>8} {r['rlm_child_events']:>8}")
            for t in r["memory_texts"]:
                print(f"       MEMORY WRITTEN: {t[:120]!r}")
    if args.json:
        json.dump(everything, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def prayer_hp_timing(turn_ndjson_path):
    """Pre-prayer HP at each np_pray (the record's own hp is POST-heal -- reading
    it reports full HP and inverts the finding; use the PRIOR turn's hp)."""
    import json
    recs = [json.loads(l) for l in open(turn_ndjson_path)]
    out = []
    for i, r in enumerate(recs):
        if any(tc.get("name") == "np_pray" for tc in (r.get("tool_calls") or [])):
            hp = recs[i-1].get("hp") if i > 0 else None
            mx = recs[i-1].get("max_hp") if i > 0 else None
            out.append({"pre_hp": hp, "max_hp": mx,
                        "in_heal_band": bool(hp is not None and mx and (hp < mx/7 or hp < 6))})
    return out
