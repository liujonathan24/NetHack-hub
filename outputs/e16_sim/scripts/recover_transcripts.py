#!/usr/bin/env python3
"""Rebuild the SIM 1/2 transcript records from the Prime Agent session files.

Needed because a concurrent agent removed `outputs/e16_sim/` from the repo
worktree at 10:31 (a `git clean`-shaped event: the tree is shared and the
directory was untracked). The SESSIONS are outside the repo, under
`/root/nld/.prime-agent-e16-sim/sessions/`, and they are the primary record --
they carry the prompt, the thinking, every tool call and result, and the
per-message token usage and USD cost. Everything the lost transcript JSONs
held except argv/stdout is recoverable from them, and argv is fixed and
recorded in the report.

Rounds are split on user messages: each user message starts a round.
"""
from __future__ import annotations

import json
from pathlib import Path

SESSIONS = Path("/root/nld/.prime-agent-e16-sim/sessions")
OUT = Path(__file__).resolve().parent.parent / "transcripts"

# Which session file holds which simulation round, established from the run
# log at launch time (header id -> file, printed by run_pa.py).
MAP = {
    "01a047da-c6d0-73b7-8e1f-0b72c8ad50ce.jsonl": {
        "header_id": "01a047da-c7a0-757e-851c-de1bf8c7f9d5",
        "rounds": ["sim1_roundA", "sim2_resume", "sim2_probe_codename",
                   "sim2_resume_r3"],
    },
    "01a047e1-5a4a-71b0-bd32-28d65e532f02.jsonl": {
        "header_id": "01a047e1-5b19-7129-a852-5123da9be9fc",
        "rounds": ["sim1_roundB"],
    },
    "01a047e2-f992-72ef-851a-d2d1f299441a.jsonl": {
        "header_id": "01a047e2-fa7f-74dc-a79d-9f1475fc1da6",
        "rounds": ["sim2_control_noresume"],
    },
}


def load(path: Path) -> list:
    recs = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def split_rounds(recs: list) -> list:
    rounds, cur = [], []
    for r in recs:
        msg = r.get("message") or {}
        if r.get("type") == "message" and msg.get("role") == "user":
            if cur:
                rounds.append(cur)
            cur = [r]
        elif cur:
            cur.append(r)
    if cur:
        rounds.append(cur)
    return rounds


def summarize(round_recs: list) -> dict:
    prompt, thinking, tool_calls, texts = "", [], [], []
    usage = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
             "cost_usd": 0.0, "assistant_messages": 0}
    for r in round_recs:
        msg = r.get("message") or {}
        role = msg.get("role")
        for b in msg.get("content") or []:
            t = b.get("type")
            if role == "user" and t == "text" and not prompt:
                prompt = b.get("text") or ""
            elif t == "thinking":
                thinking.append(b.get("thinking") or "")
            elif t == "toolCall":
                tool_calls.append({"name": b.get("name"),
                                   "arguments": b.get("arguments")})
            elif t == "text" and role == "assistant":
                texts.append(b.get("text") or "")
        u = msg.get("usage")
        if u:
            usage["assistant_messages"] += 1
            for k in ("input", "output", "cacheRead", "cacheWrite"):
                usage[k] += int(u.get(k) or 0)
            usage["cost_usd"] += float((u.get("cost") or {}).get("total") or 0.0)
    return {"prompt": prompt, "thinking": thinking, "tool_calls": tool_calls,
            "assistant_texts": texts, "final_text": texts[-1] if texts else "",
            "usage": usage, "records": round_recs}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for fname, info in MAP.items():
        path = SESSIONS / fname
        if not path.is_file():
            print(f"MISSING {path}")
            continue
        rounds = split_rounds(load(path))
        for tag, recs in zip(info["rounds"], rounds):
            rec = summarize(recs)
            rec.update({"tag": tag, "session_file": fname,
                        "session_header_id": info["header_id"],
                        "session_file_bytes": path.stat().st_size,
                        "model": "z-ai/glm-5.2",
                        "provider": "prime-inference"})
            (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2))
            u = rec["usage"]
            total += u["cost_usd"]
            print(f"{tag:24s} cost=${u['cost_usd']:.4f} in={u['input']:6d} "
                  f"out={u['output']:5d} cacheRead={u['cacheRead']:6d} "
                  f"tools={len(rec['tool_calls'])}")
        if len(rounds) != len(info["rounds"]):
            print(f"  note: {fname} holds {len(rounds)} round(s), "
                  f"named {len(info['rounds'])}")
    print(f"TOTAL ${total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
