#!/usr/bin/env python3
"""Turn a NetHack eval run directory into a self-contained HTML gameplay viewer.

    python -m tools.game_viewer.export <cell_dir> [<cell_dir> ...] -o out.html

A cell dir holds `traces.jsonl` (LLM-side: reasoning) and `turns/*.ndjson`
(game-side: per-LM-turn grid/obs/call/status, and, when the run enabled
`record_step_frames`, per-engine-step screens under `step_frames`).

The emitted HTML embeds the data and needs no server or network -- open it
locally or push it to a blog branch and edit around it. Reasoning is aligned to
game turns by ordered `nethack.*` call matching (best-effort until the call-id
barrier lands on this branch). One <title> and one favicon are the caller's job.

Design intent: this is the versioned, reusable form of the ad-hoc blog export --
the widget layout (wide 80-col engine screen, dual LLM/game-turn scrubbers,
per-move keystrokes + step frames, collapsible full reasoning, function-call
log) lives in template.html next to this file so the blog author edits one place.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys

CALL_RE = re.compile(r"nethack\.([a-z_0-9]+)\s*\(")
MAP_TOOLS = {"reveal", "request_map"}
HERE = os.path.dirname(os.path.abspath(__file__))
OBS_CAP = 5000  # obs text per turn; keep it generous but bounded

# balrog table is vendored in the env package; import lazily so the exporter
# runs even from a bare checkout (falls back to a depth-only estimate).
def _balrog(dlvl, xp):
    try:
        from nethack_harness.prompt.balrog import balrog_progress, balrog_progress_min
        return round(balrog_progress(dlvl, xp, [], False) * 100, 2), \
               round(balrog_progress_min(dlvl, xp, [], False) * 100, 2)
    except Exception:
        return round(min(dlvl, 50) / 50 * 80.68, 2), 0.0


def _select_turn_files(cell_dir):
    """One turn file per seed: the attempt with the most rows (PID join)."""
    best = {}
    pats = [os.path.join(cell_dir, "turns", "*.ndjson"),
            os.path.join(cell_dir, "turns.stalled", "*", "*.ndjson")]
    for pat in pats:
        for f in glob.glob(pat):
            base = os.path.basename(f)
            if not base[:1].isdigit():
                continue
            seed = int(base.split("_")[0])
            n = sum(1 for _ in open(f))
            if seed not in best or n > best[seed][0]:
                best[seed] = (n, f)
    return {s: p for s, (n, p) in best.items()}


def _load_turns(path):
    turns, recs = [], []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        recs.append(r)
        st = r.get("status") or {}
        tc = r.get("tool_calls") or []
        tr = (r.get("tool_results") or [{}])[0]
        obs = r.get("rendered_user_message") or ""
        if len(obs) > OBS_CAP:
            obs = obs[:OBS_CAP] + "\n[...]"
        fb = tr.get("feedback") or ""
        if len(fb) > 400:
            fb = fb[:400] + " ..."
        turns.append({
            "lm": r.get("lm_turn"), "T": st.get("time"),
            "hp": r.get("hp"), "mhp": r.get("max_hp"),
            "dl": r.get("dlvl"), "mdl": r.get("max_dlvl_reached"),
            "grid": r.get("raw_grid") or [], "obs": obs,
            "call": (tc[-1].get("name") if tc else None),
            "args": json.dumps(tc[-1].get("arguments")) if tc else "",
            "status": tr.get("status"), "fb": fb,
            "clock": [tr.get("clock_before"), tr.get("clock_after")],
            "keys": (r.get("actions") or {}).get("keys") or "",
            "steps": [m for m in (r.get("all_messages") or []) if m][:12],
            # per-move screens when the run captured them (record_step_frames)
            "frames": r.get("step_frames") or [],
        })
    return turns, recs


def _load_reasoning(cell_dir):
    out = {}
    tf = os.path.join(cell_dir, "traces.jsonl")
    if not os.path.exists(tf):
        return out
    for line in open(tf):
        try:
            t = json.loads(line)
        except Exception:
            continue
        idx = ((t.get("task") or {}).get("data") or {}).get("idx")
        items = []
        for n in (t.get("nodes") or []):
            m = n.get("message") or {}
            if m.get("role") != "assistant":
                continue
            txt = (m.get("reasoning_content") or "").strip()
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            body = (txt + ("\n---\n" + content.strip() if content.strip() else "")).strip()
            calls = []
            for tc in (m.get("tool_calls") or []):
                a = tc.get("arguments") or ""
                if isinstance(a, dict):
                    a = json.dumps(a)
                calls += CALL_RE.findall(a)
            if body or calls:
                items.append({"text": body or "(tool call only)", "calls": calls})
        if idx is not None:
            out[idx] = items
    return out


def _align(turns, ritems):
    ptr = 0
    seq = [t["call"] for t in turns]
    for it in ritems:
        it["turn"] = min(ptr, max(0, len(turns) - 1))
        for k, name in enumerate(it["calls"]):
            j = ptr
            while j < len(seq) and seq[j] != name:
                j += 1
            if j < len(seq):
                if k == 0:
                    it["turn"] = j
                ptr = j + 1
        it.pop("calls", None)
    return ritems


def build_games(cell_dirs, labels=None):
    games = []
    for ci, cdir in enumerate(cell_dirs):
        label = (labels or {}).get(cdir) or os.path.basename(cdir.rstrip("/"))
        chosen = _select_turn_files(cdir)
        rz = _load_reasoning(cdir)
        for seed, path in sorted(chosen.items()):
            turns, recs = _load_turns(path)
            if not turns:
                continue
            ritems = _align(turns, list(rz.get(seed, [])))
            last = turns[-1]
            died = last.get("hp") == 0
            maxdl = max((t.get("mdl") or 0) for t in turns)
            maxxp = max(((r.get("status") or {}).get("experience_level") or 0) for r in recs)
            maxT = max(((r.get("status") or {}).get("time") or 0) for r in recs)
            bal_max, bal_min = _balrog(maxdl, maxxp)
            frames_n = sum(len(t["frames"]) for t in turns)
            games.append({
                "id": f"{label} · seed {seed}", "cell": label, "seed": seed,
                "outcome": "died" if died else ("budget" if len(turns) >= 195 else "ended"),
                "died": died, "dlvl": maxdl, "xp": maxxp, "gameT": maxT,
                "calls": len(turns), "maptool": sum(1 for t in turns if t["call"] in MAP_TOOLS),
                "freeze": sum(1 for t in turns if "GAME IS WAITING" in t["obs"]),
                "bal_max": bal_max, "bal_min": bal_min, "has_frames": frames_n > 0,
                "cause": "—", "tag": "—",
                "turns": turns, "reasoning": ritems,
            })
    return games


def render(games, template=None):
    tpl_path = template or os.path.join(HERE, "template.html")
    tpl = open(tpl_path, encoding="utf-8").read()
    data = json.dumps(games, separators=(",", ":")).replace("</", "<\\/")
    out = tpl.replace("/*__DATA__*/[]", data)
    assert out.count("</script>") == 1, "embedded data broke the <script> tag"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a NetHack run dir to an HTML viewer.")
    ap.add_argument("cell_dirs", nargs="+", help="run directories (each with traces.jsonl + turns/)")
    ap.add_argument("-o", "--out", default="game_viewer.html")
    ap.add_argument("--label", action="append", default=[], help="dir=Label overrides, repeatable")
    ap.add_argument("--template", help="override template.html path")
    ap.add_argument("--no-frames", action="store_true",
                    help="strip per-move step_frames (light overview across many cells)")
    args = ap.parse_args(argv)
    labels = dict(x.split("=", 1) for x in args.label if "=" in x)
    games = build_games(args.cell_dirs, labels)
    if args.no_frames:
        for g in games:
            for t in g["turns"]:
                t["frames"] = []
    html = render(games, args.template)
    open(args.out, "w", encoding="utf-8").write(html)
    fr = sum(g["has_frames"] for g in games)
    print(f"wrote {args.out}: {len(games)} games "
          f"({fr} with per-move frames), {len(html)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
