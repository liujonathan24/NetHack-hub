#!/usr/bin/env python3
"""Turn a NetHack eval run directory into a self-contained HTML gameplay viewer.

    python -m tools.game_viewer.export <cell_dir> [<cell_dir> ...] -o out.html

A cell dir holds `traces.jsonl` (LLM-side: reasoning) and `turns/*.ndjson`
(game-side: per-LM-turn grid/obs/call/status, and, when the run enabled
`record_step_frames`, per-engine-step screens under `step_frames`).

The emitted HTML embeds the data and needs no server or network -- open it
locally or push it to a blog branch and edit around it. Reasoning is read
straight from each record's `reasoning` block (populated by
`tools.trace_reasoning`, the single alignment authority) -- the exporter no
longer re-derives the move<->reasoning mapping from prose. One <title> and one
favicon are the caller's job.

Design intent: this is the versioned, reusable form of the ad-hoc blog export --
the widget layout (wide 80-col engine screen, dual LLM/game-turn scrubbers,
per-move keystrokes + step frames, collapsible full reasoning, function-call
log) lives in template.html next to this file so the blog author edits one place.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

MAP_TOOLS = {"reveal", "request_map"}
HERE = os.path.dirname(os.path.abspath(__file__))
OBS_CAP = 500000  # obs text per turn; keep it generous but bounded

# balrog table is vendored in the env package; import lazily so the exporter
# runs even from a bare checkout (falls back to a depth-only estimate).
def _balrog(dlvl, xp):
    try:
        # reached_planes/ascended are keyword-only; passing them positionally
        # raised TypeError and silently dropped every viewer onto the
        # depth-only fallback below.
        from nethack_harness.prompt.balrog import balrog_progress, balrog_progress_min
        return round(balrog_progress(dlvl, xp) * 100, 2), \
               round(balrog_progress_min(dlvl, xp) * 100, 2)
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


def _reasoning_items(recs):
    """Reasoning entries for the panel, read straight from the records.

    Each record already carries its own `reasoning` block, aligned to the move
    by `tools.trace_reasoning`'s backfill -- the single alignment authority,
    which joins the game-side records to the model-side trace nodes by call
    name/order (control arm) or by wall clock (Prime Agent, whose only tool is
    `ipython` so names can't be walked). We do NOT re-derive the mapping here;
    re-deriving from prose is exactly what produced the old "same reasoning on
    consecutive turns" artifact.

    Consecutive records that share a narration collapse into one entry labeled
    with the FIRST turn it governs -- because the model narrates a plan once and
    then executes several silent moves under it. `show()` highlights the last
    entry whose turn <= the current move, so the governing plan stays lit across
    those moves. That shared narration is the honest picture, not a bug.
    """
    items, last = [], None
    for i, r in enumerate(recs):
        rz = r.get("reasoning") or {}
        content = (rz.get("text") or "").strip()
        thinking = (rz.get("reasoning_text") or "").strip()
        # Prefer the richer channel: GLM's `reasoning_content` is usually a
        # superset of `content`; the control arm fills `content` and leaves
        # `reasoning_text` empty. Longer wins; ties keep `content`.
        txt = thinking if len(thinking) > len(content) else content
        if not txt or txt == last:
            continue
        items.append({"text": txt, "turn": i})
        last = txt
    return items


def build_games(cell_dirs, labels=None):
    games = []
    for ci, cdir in enumerate(cell_dirs):
        label = (labels or {}).get(cdir) or os.path.basename(cdir.rstrip("/"))
        chosen = _select_turn_files(cdir)
        for seed, path in sorted(chosen.items()):
            turns, recs = _load_turns(path)
            if not turns:
                continue
            ritems = _reasoning_items(recs)
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
