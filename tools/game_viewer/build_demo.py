#!/usr/bin/env python3
"""Build the small in-article gameplay embed for blog/index.md.

    python -m tools.game_viewer.build_demo --data-root /root/nld/outputs-copy

Unlike `export.py`/`build_viewers.py` -- which emit whole standalone viewer
pages -- this writes a *fragment*: a compact, self-contained widget (scoped CSS
+ JS, no <html>/<body>) that `build_index.py` splices into the article where the
Markdown carries

    <div class="game-embed" data-demo="opening"></div>

The fragment is committed under `blog/embeds/`, so `index.html` stays
reproducible from checked-in files alone -- the raw run outputs it is built from
are gitignored/ephemeral. Re-run this only when the underlying run changes.

The default embed is the opening of the control run's seed 0: the first few LLM
actions, one engine frame per keystroke, so a reader can watch the skill calls
expand into raw NetHack keys instead of reading a description of it.
"""
from __future__ import annotations
import argparse, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# name -> (run dir relative to --data-root, seed, how many LLM turns to include)
DEMOS = {
    "opening": ("e7_seed/NPCORE_v3__prime_agent", 0, 5),
}


def _turn_file(cell_dir, seed):
    """The seed's turn file: the attempt with the most rows (same rule as export)."""
    best = None
    for f in glob.glob(os.path.join(cell_dir, "turns", f"{seed}_*.ndjson")):
        n = sum(1 for _ in open(f))
        if best is None or n > best[0]:
            best = (n, f)
    if best is None:
        raise SystemExit(f"no turn file for seed {seed} in {cell_dir}")
    return best[1]


def collect(cell_dir, seed, n_turns):
    """The first `n_turns` LLM actions as {name, args, status, keys, frames}."""
    calls = []
    with open(_turn_file(cell_dir, seed)) as fh:
        for i, line in enumerate(fh):
            if i >= n_turns:
                break
            r = json.loads(line)
            tc = (r.get("tool_calls") or [{}])[-1]
            tr = (r.get("tool_results") or [{}])[0]
            args = tc.get("arguments") or {}
            frames = [{"g": f["g"], "k": f.get("k") or ""} for f in (r.get("step_frames") or [])]
            if not frames:  # no keystrokes (map request, failed route): the turn's own screen
                frames = [{"g": r.get("raw_grid") or [], "k": ""}]
            calls.append({
                "name": tc.get("name") or "—",
                "args": ", ".join(f"{k}={v}" for k, v in args.items()),
                "status": tr.get("status") or "",
                "keys": (r.get("actions") or {}).get("keys") or "",
                "frames": frames,
            })
    return calls


CSS = """
.gamedemo{--gd-term-bg:#0b100e;--gd-term-ink:#c7e6c7;--gd-line:var(--line,#e7e2df);
  --gd-ink2:var(--ink2,#544c5e);--gd-muted:var(--muted,#8a8194);--gd-accent:var(--accent,#7a3b8f);
  margin:1.6em 0;border:1px solid var(--gd-line);border-radius:10px;overflow:hidden;
  background:var(--surface,#fff);}
.gamedemo .gd-calls{display:flex;flex-wrap:wrap;gap:.35rem;padding:.6rem .7rem;
  border-bottom:1px solid var(--gd-line);}
.gamedemo .gd-call{font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color:var(--gd-ink2);background:transparent;border:1px solid var(--gd-line);border-radius:999px;
  padding:.18em .62em;cursor:pointer;white-space:nowrap;}
.gamedemo .gd-call[aria-current="true"]{border-color:var(--gd-accent);color:var(--gd-accent);
  background:color-mix(in oklab,var(--gd-accent) 10%,transparent);}
.gamedemo .gd-call.gd-failed{text-decoration:line-through;opacity:.72;}
.gamedemo .gd-screen{background:var(--gd-term-bg);padding:.7rem .8rem;overflow-x:auto;}
.gamedemo .gd-screen pre{margin:0;background:none;padding:0;color:var(--gd-term-ink);
  font:12.5px/1.28 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  white-space:pre;min-width:max-content;}
.gamedemo .gd-bar{display:flex;align-items:center;gap:.5rem;padding:.55rem .7rem;
  border-top:1px solid var(--gd-line);}
.gamedemo .gd-bar button{font:13px/1 inherit;color:var(--gd-ink2);background:transparent;
  border:1px solid var(--gd-line);border-radius:6px;padding:.32em .6em;cursor:pointer;}
.gamedemo .gd-bar button:hover{border-color:var(--gd-accent);color:var(--gd-accent);}
.gamedemo .gd-bar input[type=range]{flex:1;min-width:4rem;accent-color:var(--gd-accent);}
.gamedemo .gd-status{padding:0 .7rem .6rem;font-size:.8rem;color:var(--gd-muted);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.gamedemo .gd-status b{color:var(--gd-ink2);font-weight:600;}
.gamedemo .gd-keys{display:inline-flex;flex-wrap:wrap;gap:.18rem;vertical-align:middle;}
.gamedemo .gd-key{border:1px solid var(--gd-line);border-radius:4px;padding:0 .3em;min-width:1.2em;
  text-align:center;color:var(--gd-muted);}
.gamedemo .gd-key.gd-now{border-color:var(--gd-accent);color:var(--gd-accent);font-weight:700;}
"""

FRAGMENT = """<div class="gamedemo" id="{eid}">
<style>{css}</style>
<div class="gd-calls" data-role="calls"></div>
<div class="gd-screen"><pre data-role="screen"></pre></div>
<div class="gd-bar">
  <button type="button" data-role="prev" aria-label="previous frame">&#8592;</button>
  <button type="button" data-role="play" aria-label="play">&#9654;</button>
  <button type="button" data-role="next" aria-label="next frame">&#8594;</button>
  <input type="range" data-role="scrub" min="0" value="0" aria-label="frame">
</div>
<div class="gd-status" data-role="status"></div>
<script type="application/json" data-role="data">{data}</script>
<script>{js}</script>
</div>
"""

JS = """
(function(){
  var root = document.getElementById("__EID__");
  var calls = JSON.parse(root.querySelector('[data-role=data]').textContent);
  // flatten every keystroke of every call into one timeline of engine frames
  var seq = [];
  calls.forEach(function(c, ci){
    c.frames.forEach(function(f, fi){ seq.push({ci:ci, fi:fi, g:f.g, k:f.k}); });
  });
  var q = function(r){ return root.querySelector('[data-role=' + r + ']'); };
  var screen = q("screen"), scrub = q("scrub"), status = q("status"), chips = q("calls");
  var timer = null, i = 0;
  scrub.max = seq.length - 1;

  calls.forEach(function(c, ci){
    var b = document.createElement("button");
    b.type = "button";
    b.className = "gd-call" + (c.status === "failed" ? " gd-failed" : "");
    b.textContent = c.name + "(" + c.args + ")";
    b.onclick = function(){ stop(); show(seq.findIndex(function(s){ return s.ci === ci; })); };
    chips.appendChild(b);
  });

  function show(n){
    i = Math.max(0, Math.min(seq.length - 1, n));
    var s = seq[i], c = calls[s.ci];
    screen.textContent = s.g.join("\\n");
    scrub.value = i;
    Array.prototype.forEach.call(chips.children, function(b, ci){
      b.setAttribute("aria-current", ci === s.ci ? "true" : "false");
    });
    var keys = "";
    if (c.keys) {
      keys = ' <span class="gd-keys">' + c.keys.split("").map(function(k, ki){
        return '<span class="gd-key' + (ki === s.fi ? " gd-now" : "") + '">' + k + "</span>";
      }).join("") + "</span> ";
    }
    status.innerHTML =
      "action <b>" + (s.ci + 1) + "/" + calls.length + "</b> \\u00b7 <b>" + c.name +
      "(" + c.args + ")</b> \\u2192 " +
      (c.status === "failed" ? "no route found, 0 keys"
        : c.keys ? keys + "<b>" + c.keys.length + "</b> raw key" + (c.keys.length > 1 ? "s" : "") +
                   " (" + (s.fi + 1) + "/" + c.keys.length + ")"
                 : "no keys (screen only)");
  }
  function stop(){ if (timer) { clearInterval(timer); timer = null; q("play").innerHTML = "&#9654;"; } }
  q("prev").onclick = function(){ stop(); show(i - 1); };
  q("next").onclick = function(){ stop(); show(i + 1); };
  scrub.oninput = function(e){ stop(); show(+e.target.value); };
  q("play").onclick = function(){
    if (timer) { stop(); return; }
    q("play").innerHTML = "&#9208;";
    if (i >= seq.length - 1) show(0);
    timer = setInterval(function(){ if (i >= seq.length - 1) stop(); else show(i + 1); }, 420);
  };
  show(0);
})();
"""


def render(name, calls):
    eid = "gamedemo-" + name
    data = json.dumps(calls, separators=(",", ":")).replace("</", "<\\/")
    js = JS.replace("__EID__", eid)
    return FRAGMENT.format(eid=eid, css=CSS, data=data, js=js)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the in-article gameplay embeds.")
    ap.add_argument("--data-root", default="outputs", help="root holding the run dirs")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "blog", "embeds"))
    ap.add_argument("--only", nargs="*", help="subset of demo names (default: all)")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    for name, (rel, seed, n) in DEMOS.items():
        if args.only and name not in args.only:
            continue
        calls = collect(os.path.join(args.data_root, rel), seed, n)
        out = os.path.join(args.outdir, name + ".html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render(name, calls))
        frames = sum(len(c["frames"]) for c in calls)
        print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB): "
              f"{len(calls)} actions, {frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
