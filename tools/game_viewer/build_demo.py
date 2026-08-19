#!/usr/bin/env python3
"""Build the in-article gameplay embeds for blog/index.md.

    python -m tools.game_viewer.build_demo --data-root /root/nld/outputs-copy

Unlike `export.py`/`build_viewers.py` -- which emit whole standalone viewer
pages -- this writes *fragments*: compact, self-contained widgets (scoped CSS +
JS, no <html>/<body>) that `build_index.py` splices into the article wherever
the Markdown carries

    <div class="game-embed" data-demo="opening"></div>

The fragments are committed under `blog/embeds/`, so `index.html` rebuilds from
checked-in files alone -- the raw run outputs they are built from are
gitignored/ephemeral. Re-run this only when the underlying runs change.

Two kinds of embed:

- `keystrokes` -- the opening of one game, one engine frame per keystroke, so a
  reader can watch skill calls expand into raw NetHack keys.
- `games` -- the whole set of games behind a results table: a seed/run picker
  and one screen per LLM turn. Screens are stored as row diffs against the
  previous turn (a turn usually redraws a handful of rows), which is what keeps
  a 429-turn archive around a quarter of a megabyte instead of eight.

The `games` embed reads every `turns/<seed>_*.ndjson` in the run dir, so when a
cell grows from one rollout per seed to three, this picks up the extra runs and
offers them as `run 1/2/3` under each seed with no code change.
"""
from __future__ import annotations
import argparse, glob, json, os, re

from .export import _balrog

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEMOS = {
    # the opening of the control run, keystroke by keystroke
    "opening": {"kind": "keystrokes",
                "run": "e7_seed/NPCORE_v3__prime_agent", "seed": 0, "turns": 5},
    # every game behind the initial-harness results table
    "harness": {"kind": "games",
                "run": "e7_seed/NPCORE_v3__prime_agent", "seeds": [0, 1, 2, 3, 4]},
}


def _turn_files(cell_dir, seed):
    """Every rollout recorded for `seed`, oldest first (run 1, run 2, ...)."""
    pat = os.path.join(cell_dir, "turns", f"{seed}_*.ndjson")
    return sorted(glob.glob(pat), key=lambda p: os.path.basename(p))


def _read(path):
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------- keystrokes

def collect_keystrokes(cell_dir, seed, n_turns):
    """The first `n_turns` LLM actions as {name, args, status, keys, frames}."""
    files = _turn_files(cell_dir, seed)
    if not files:
        raise SystemExit(f"no turn file for seed {seed} in {cell_dir}")
    calls = []
    for r in _read(files[0])[:n_turns]:
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


# --------------------------------------------------------------------- games

def _diff(prev, rows):
    """Rows of `rows` that differ from `prev`, as [index, text] pairs."""
    out = []
    for i, row in enumerate(rows):
        if i >= len(prev) or prev[i] != row:
            out.append([i, row])
    return out


def collect_games(cell_dir, seeds):
    """One entry per rollout: headline stats + per-turn screens (row diffs)."""
    games = []
    for seed in seeds:
        files = _turn_files(cell_dir, seed)
        for run, path in enumerate(files, 1):
            recs = _read(path)
            if not recs:
                continue
            turns, prev = [], []
            for r in recs:
                tc = (r.get("tool_calls") or [{}])[-1]
                tr = (r.get("tool_results") or [{}])[0]
                st = r.get("status") or {}
                rows = r.get("raw_grid") or []
                turns.append({
                    "c": tc.get("name") or "—",
                    "a": ", ".join(f"{k}={v}" for k, v in (tc.get("arguments") or {}).items()),
                    "s": tr.get("status") or "",
                    "k": (r.get("actions") or {}).get("keys") or "",
                    "hp": r.get("hp"), "mhp": r.get("max_hp"),
                    "dl": r.get("dlvl"), "T": st.get("time"),
                    "d": _diff(prev, rows),
                })
                prev = rows
            dlvl = max((r.get("max_dlvl_reached") or 0) for r in recs)
            xl = max(((r.get("status") or {}).get("experience_level") or 0) for r in recs)
            bal, _ = _balrog(dlvl, xl)
            games.append({
                "seed": seed, "run": run, "runs": len(files),
                "bal": bal, "dlvl": dlvl, "xl": xl,
                "died": recs[-1].get("hp") == 0, "turns": turns,
            })
    return games


# ------------------------------------------------------------------ fragment

CSS_COMMON = """
.gamedemo{--gd-term-bg:#0b100e;--gd-term-ink:#c7e6c7;--gd-line:var(--line,#e7e2df);
  --gd-ink2:var(--ink2,#544c5e);--gd-muted:var(--muted,#8a8194);--gd-accent:var(--accent,#7a3b8f);
  margin:1.6em 0;border:1px solid var(--gd-line);border-radius:10px;overflow:hidden;
  background:var(--surface,#fff);}
.gamedemo .gd-picks{display:flex;flex-wrap:wrap;gap:.35rem;padding:.6rem .7rem;
  border-bottom:1px solid var(--gd-line);align-items:center;}
.gamedemo .gd-pick{font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color:var(--gd-ink2);background:transparent;border:1px solid var(--gd-line);border-radius:999px;
  padding:.18em .62em;cursor:pointer;white-space:nowrap;}
.gamedemo .gd-pick[aria-current="true"]{border-color:var(--gd-accent);color:var(--gd-accent);
  background:color-mix(in oklab,var(--gd-accent) 10%,transparent);}
.gamedemo .gd-pick.gd-failed{text-decoration:line-through;opacity:.72;}
.gamedemo .gd-label{font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;
  color:var(--gd-muted);margin-right:.15rem;}
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
{picks}<div class="gd-screen"><pre data-role="screen"></pre></div>
<div class="gd-bar">
  <button type="button" data-role="prev" aria-label="previous">&#8592;</button>
  <button type="button" data-role="play" aria-label="play">&#9654;</button>
  <button type="button" data-role="next" aria-label="next">&#8594;</button>
  <input type="range" data-role="scrub" min="0" value="0" aria-label="position">
</div>
<div class="gd-status" data-role="status"></div>
<script type="application/json" data-role="data">{data}</script>
<script>{js}</script>
</div>
"""

PICKS_ONE = '<div class="gd-picks" data-role="calls"></div>\n'
PICKS_TWO = ('<div class="gd-picks" data-role="seeds"></div>\n'
             '<div class="gd-picks" data-role="runs" hidden></div>\n')

# shared player plumbing: q(), stop(), the transport buttons, the scrubber
JS_PLAYER = """
  var q = function(r){ return root.querySelector('[data-role=' + r + ']'); };
  var screen = q("screen"), scrub = q("scrub"), status = q("status");
  var timer = null, i = 0;
  function stop(){ if (timer) { clearInterval(timer); timer = null; q("play").innerHTML = "&#9654;"; } }
  q("prev").onclick = function(){ stop(); show(i - 1); };
  q("next").onclick = function(){ stop(); show(i + 1); };
  scrub.oninput = function(e){ stop(); show(+e.target.value); };
  q("play").onclick = function(){
    if (timer) { stop(); return; }
    q("play").innerHTML = "&#9208;";
    if (i >= LAST()) show(0);
    timer = setInterval(function(){ if (i >= LAST()) stop(); else show(i + 1); }, SPEED);
  };
  function chip(host, text, failed, onclick){
    var b = document.createElement("button");
    b.type = "button";
    b.className = "gd-pick" + (failed ? " gd-failed" : "");
    b.textContent = text;
    b.onclick = onclick;
    host.appendChild(b);
    return b;
  }
  function mark(host, active){
    Array.prototype.forEach.call(host.children, function(b, k){
      if (b.tagName === "BUTTON") b.setAttribute("aria-current", k === active ? "true" : "false");
    });
  }
"""

JS_KEYSTROKES = """
(function(){
  var root = document.getElementById("__EID__");
  var calls = JSON.parse(root.querySelector('[data-role=data]').textContent);
  // flatten every keystroke of every call into one timeline of engine frames
  var seq = [];
  calls.forEach(function(c, ci){
    c.frames.forEach(function(f, fi){ seq.push({ci:ci, fi:fi, g:f.g}); });
  });
  var SPEED = 420;
  function LAST(){ return seq.length - 1; }
__PLAYER__
  var chips = q("calls");
  calls.forEach(function(c, ci){
    chip(chips, c.name + "(" + c.args + ")", c.status === "failed", function(){
      stop(); show(seq.findIndex(function(s){ return s.ci === ci; }));
    });
  });
  scrub.max = LAST();

  function show(n){
    i = Math.max(0, Math.min(LAST(), n));
    var s = seq[i], c = calls[s.ci];
    screen.textContent = s.g.join("\\n");
    scrub.value = i;
    mark(chips, s.ci);
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
  show(0);
})();
"""

JS_GAMES = """
(function(){
  var root = document.getElementById("__EID__");
  var games = JSON.parse(root.querySelector('[data-role=data]').textContent);
  var SPEED = 260;
  var G = 0, rows = [], built = -1;
  function LAST(){ return games[G].turns.length - 1; }
__PLAYER__
  var seedHost = q("seeds"), runHost = q("runs");
  var seeds = [];
  games.forEach(function(g){ if (seeds.indexOf(g.seed) < 0) seeds.push(g.seed); });

  function runsOf(sd){ return games.filter(function(g){ return g.seed === sd; }); }
  function outcome(g){ return g.died ? "died" : "survived"; }
  // one rollout per seed: the chip carries that game's result. Several: it
  // carries the spread, and the per-run results move to the run chips.
  function seedLabel(sd){
    var mine = runsOf(sd);
    if (mine.length < 2) {
      return "seed " + sd + " \\u00b7 BALROG " + mine[0].bal.toFixed(2) +
             " \\u00b7 Dlvl " + mine[0].dlvl + " \\u00b7 " + outcome(mine[0]);
    }
    var bals = mine.map(function(g){ return g.bal; });
    var mean = bals.reduce(function(a, b){ return a + b; }, 0) / bals.length;
    return "seed " + sd + " \\u00b7 " + mine.length + " runs \\u00b7 BALROG mean " +
           mean.toFixed(2) + " (" + Math.min.apply(null, bals).toFixed(2) + "\\u2013" +
           Math.max.apply(null, bals).toFixed(2) + ")";
  }
  seedHost.appendChild(Object.assign(document.createElement("span"),
    {className: "gd-label", textContent: "game"}));
  seeds.forEach(function(sd){
    chip(seedHost, seedLabel(sd), false, function(){ stop(); pick(sd, 1); });
  });
  runHost.appendChild(Object.assign(document.createElement("span"),
    {className: "gd-label", textContent: "run"}));

  // screens are stored as row diffs against the previous turn: replay to rebuild
  function screenAt(n){
    var t = games[G].turns;
    if (built > n) { rows = []; built = -1; }
    for (var j = built + 1; j <= n; j++)
      t[j].d.forEach(function(p){ rows[p[0]] = p[1]; });
    built = n;
    return rows.join("\\n");
  }

  function pick(seed, run){
    var n = games.findIndex(function(g){ return g.seed === seed && g.run === run; });
    if (n < 0) return;
    G = n; rows = []; built = -1;
    scrub.max = LAST();
    mark(seedHost, seeds.indexOf(seed) + 1);           // +1: the "game" label
    // a second row of chips appears only once a seed has more than one rollout
    var mine = runsOf(seed);
    runHost.hidden = mine.length < 2;
    while (runHost.children.length > 1) runHost.removeChild(runHost.lastChild);
    if (!runHost.hidden)
      mine.forEach(function(g){
        chip(runHost, "run " + g.run + " \\u00b7 " + g.bal.toFixed(2) + " \\u00b7 " + outcome(g),
             false, function(){ stop(); pick(seed, g.run); });
      });
    mark(runHost, run);
    show(0);
  }

  function show(n){
    i = Math.max(0, Math.min(LAST(), n));
    var g = games[G], t = g.turns[i];
    screen.textContent = screenAt(i);
    scrub.value = i;
    status.innerHTML =
      "turn <b>" + (i + 1) + "/" + g.turns.length + "</b> \\u00b7 <b>" + t.c +
      (t.a ? "(" + t.a + ")" : "()") + "</b>" +
      (t.s === "failed" ? " \\u2192 failed" : "") +
      (t.k ? " \\u00b7 keys <b>" + t.k + "</b>" : "") +
      " \\u00b7 HP " + t.hp + "/" + t.mhp + " \\u00b7 Dlvl " + t.dl + " \\u00b7 T " + t.T;
  }
  pick(seeds[0], 1);
})();
"""


def render(name, kind, payload):
    eid = "gamedemo-" + name
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    js = (JS_KEYSTROKES if kind == "keystrokes" else JS_GAMES)
    js = js.replace("__PLAYER__", JS_PLAYER).replace("__EID__", eid)
    picks = PICKS_ONE if kind == "keystrokes" else PICKS_TWO
    return FRAGMENT.format(eid=eid, css=CSS_COMMON, picks=picks, data=data, js=js)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the in-article gameplay embeds.")
    ap.add_argument("--data-root", default="outputs", help="root holding the run dirs")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "blog", "embeds"))
    ap.add_argument("--only", nargs="*", help="subset of demo names (default: all)")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    for name, spec in DEMOS.items():
        if args.only and name not in args.only:
            continue
        cell = os.path.join(args.data_root, spec["run"])
        if spec["kind"] == "keystrokes":
            payload = collect_keystrokes(cell, spec["seed"], spec["turns"])
            note = (f"{len(payload)} actions, "
                    f"{sum(len(c['frames']) for c in payload)} frames")
        else:
            payload = collect_games(cell, spec["seeds"])
            note = (f"{len(payload)} games, "
                    f"{sum(len(g['turns']) for g in payload)} turns; "
                    + ", ".join(f"s{g['seed']}={g['bal']}" for g in payload))
        out = os.path.join(args.outdir, name + ".html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render(name, spec["kind"], payload))
        print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB): {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
