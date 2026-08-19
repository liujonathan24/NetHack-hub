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

Three kinds of embed:

- `keystrokes` -- the opening of one game, one engine frame per keystroke, so a
  reader can watch skill calls expand into raw NetHack keys.
- `games` -- the games behind a results table, with up to three picker rows:
  arm (control vs treatment), seed, and run. Screens are stored as row diffs
  against the previous turn (a turn usually redraws a handful of rows), which
  keeps a 400-turn archive around a fifth of a megabyte instead of eight.
- `gates` -- every descent-gate panel the E8a run showed the model, next to what
  the model did with it. Static HTML, no JS.

An arm lists one run directory per repetition: `dirs[0]` is run 1, `dirs[1]` is
run 2, and so on. Within a directory a seed can still have several turn files
(a stalled rollout that was restarted); those are attempts at the *same* run, so
the one with the most rows wins -- the same rule `export.py` uses. A rollout that
neither died nor hit the call budget is still in flight and is labelled as such.
"""
from __future__ import annotations
import argparse, glob, html, json, os

from .export import _balrog

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

CONTROL = "e7_seed/NPCORE_v3__prime_agent"
CONTROL_R2 = "e9_control_rep/NPCORE_v3_r2__prime_agent"
SEEDS = [0, 1, 2, 3, 4]
BUDGET = 195  # calls: at or above this a rollout ended on budget, not by dying

DEMOS = {
    # the opening of the control run, keystroke by keystroke
    "opening": {"kind": "keystrokes", "run": CONTROL, "seed": 0, "turns": 5},
    # every game behind the initial-harness results table
    "harness": {"kind": "games", "seeds": SEEDS,
                "arms": [{"label": "initial harness", "dirs": [CONTROL, CONTROL_R2]}]},
    # the descent-gate panels, and what the model did with them
    "gates": {"kind": "gates", "run": "e8_plan/NORM__prime_agent", "seeds": SEEDS},
    # descent gate against its own control, seed by seed
    "e8a_games": {"kind": "games", "seeds": SEEDS, "arms": [
        {"label": "control", "dirs": [CONTROL]},
        {"label": "descent gate", "dirs": ["e8_plan/NORM__prime_agent"]},
        # byte-identical eval config to the gated arm, but no panel ever fired:
        # in practice a second control replicate, and the noise yardstick
        {"label": "v1 cell (no gate fired)", "dirs": ["e8_plan/NORM_v1_ungated__prime_agent"]},
    ]},
    # the two navigation simplifications against the same control
    "e8cd_games": {"kind": "games", "seeds": SEEDS, "arms": [
        {"label": "control", "dirs": [CONTROL]},
        {"label": "doors unlocked", "dirs": ["e8c_doors/UNLOCKED__prime_agent"]},
        {"label": "density 0.25", "dirs": ["e8d_density/D025__prime_agent"]},
        {"label": "density 0.10", "dirs": ["e8d_density/D010__prime_agent"]},
        {"label": "density 0.025", "dirs": ["e8d_density/D0025__prime_agent"]},
    ]},
}


def _attempt(cell_dir, seed):
    """The seed's turn file in this run dir: the attempt with the most rows."""
    best = None
    for f in glob.glob(os.path.join(cell_dir, "turns", f"{seed}_*.ndjson")):
        n = sum(1 for _ in open(f))
        if best is None or n > best[0]:
            best = (n, f)
    return best[1] if best else None


def _read(path):
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:  # a run still being written can end mid-line
                continue
    return out


def _call(r):
    tc = (r.get("tool_calls") or [{}])[-1]
    return (tc.get("name") or "—",
            ", ".join(f"{k}={v}" for k, v in (tc.get("arguments") or {}).items()))


# ---------------------------------------------------------------- keystrokes

def collect_keystrokes(cell_dir, seed, n_turns):
    """The first `n_turns` LLM actions as {name, args, status, keys, frames}."""
    path = _attempt(cell_dir, seed)
    if not path:
        raise SystemExit(f"no turn file for seed {seed} in {cell_dir}")
    calls = []
    for r in _read(path)[:n_turns]:
        tr = (r.get("tool_results") or [{}])[0]
        name, args = _call(r)
        frames = [{"g": f["g"], "k": f.get("k") or ""} for f in (r.get("step_frames") or [])]
        if not frames:  # no keystrokes (map request, failed route): the turn's own screen
            frames = [{"g": r.get("raw_grid") or [], "k": ""}]
        calls.append({"name": name, "args": args, "status": tr.get("status") or "",
                      "keys": (r.get("actions") or {}).get("keys") or "", "frames": frames})
    return calls


# --------------------------------------------------------------------- games

def _diff(prev, rows):
    """Rows of `rows` that differ from `prev`, as [index, text] pairs."""
    return [[i, row] for i, row in enumerate(rows) if i >= len(prev) or prev[i] != row]


def collect_games(data_root, arms, seeds):
    """One entry per rollout: headline stats + per-turn screens (row diffs)."""
    games = []
    for ai, arm in enumerate(arms):
        for seed in seeds:
            for run, rel in enumerate(arm["dirs"], 1):
                path = _attempt(os.path.join(data_root, rel), seed)
                if not path:
                    continue
                recs = _read(path)
                if not recs:
                    continue
                turns, prev = [], []
                for r in recs:
                    tr = (r.get("tool_results") or [{}])[0]
                    stt = r.get("status") or {}
                    rows = r.get("raw_grid") or []
                    name, args = _call(r)
                    turns.append({
                        "c": name, "a": args, "s": tr.get("status") or "",
                        "k": (r.get("actions") or {}).get("keys") or "",
                        "hp": r.get("hp"), "mhp": r.get("max_hp"),
                        "dl": r.get("dlvl"), "T": stt.get("time"),
                        "d": _diff(prev, rows),
                    })
                    prev = rows
                dlvl = max((r.get("max_dlvl_reached") or 0) for r in recs)
                xl = max(((r.get("status") or {}).get("experience_level") or 0) for r in recs)
                bal, _ = _balrog(dlvl, xl)
                died = recs[-1].get("hp") == 0
                games.append({
                    "arm": ai, "armLabel": arm["label"], "seed": seed, "run": run,
                    "bal": bal, "dlvl": dlvl, "xl": xl, "died": died,
                    "live": not died and len(recs) < BUDGET,   # still being played
                    "turns": turns,
                })
    return games


# --------------------------------------------------------------------- gates

GATE_RE = ("[descent check: you are XL ", "Typical successful human runs reach XL ")


def collect_gates(data_root, cell, seeds):
    """Each descent-gate panel, the call that tripped it, and the next call."""
    import re
    pat = re.compile(r"\[descent check: you are XL (\d+) on Dlvl (\d+)\. "
                     r"Typical successful human runs reach XL (\d+) before leaving this depth\.[^\]]*\]")
    out = []
    for seed in seeds:
        path = _attempt(os.path.join(data_root, cell), seed)
        if not path:
            continue
        recs = _read(path)
        for i, r in enumerate(recs):
            m = pat.search(r.get("rendered_user_message") or "")
            if not m:
                continue
            xl, dlvl, want = (int(x) for x in m.groups())
            this, nxt = _call(r), (_call(recs[i + 1]) if i + 1 < len(recs) else ("—", ""))
            out.append({
                "seed": seed, "turn": i, "xl": xl, "dlvl": dlvl, "want": want,
                "panel": m.group(0),
                "call": f"{this[0]}({this[1]})", "next": f"{nxt[0]}({nxt[1]})",
                "dove": this == nxt,
            })
    return out


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

CSS_GATES = """
.gamedemo .gd-gate{border-bottom:1px solid var(--gd-line);padding:.7rem .8rem;
  display:flex;flex-wrap:wrap;gap:.5rem .9rem;align-items:baseline;}
.gamedemo .gd-gate:last-child{border-bottom:0;}
.gamedemo .gd-panel{flex:1 1 22rem;background:var(--gd-term-bg);color:var(--gd-term-ink);
  border-radius:6px;padding:.45rem .6rem;
  font:11.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.gamedemo .gd-who{font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color:var(--gd-muted);flex:0 0 6.5rem;}
.gamedemo .gd-did{font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color:var(--gd-ink2);flex:1 1 16rem;}
.gamedemo .gd-tag{border-radius:999px;padding:.05em .55em;font-size:10.5px;
  border:1px solid var(--gd-line);white-space:nowrap;}
.gamedemo .gd-tag.dove{color:#b23a3a;border-color:color-mix(in oklab,#b23a3a 45%,transparent);}
.gamedemo .gd-tag.held{color:#2f7d4f;border-color:color-mix(in oklab,#2f7d4f 45%,transparent);}
.gamedemo .gd-sum{padding:.6rem .8rem;border-bottom:1px solid var(--gd-line);
  font-size:.85rem;color:var(--gd-ink2);}
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
PICKS_GAMES = ('<div class="gd-picks" data-role="arms" hidden></div>\n'
               '<div class="gd-picks" data-role="seeds"></div>\n'
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
  function label(host, text){
    host.appendChild(Object.assign(document.createElement("span"),
      {className: "gd-label", textContent: text}));
  }
  function mark(host, active){
    Array.prototype.forEach.call(host.children, function(b, k){
      if (b.tagName === "BUTTON") b.setAttribute("aria-current", k === active ? "true" : "false");
    });
  }
  function clear(host){ while (host.children.length > 1) host.removeChild(host.lastChild); }
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
  var armHost = q("arms"), seedHost = q("seeds"), runHost = q("runs");
  var arms = [], seeds = [];
  games.forEach(function(g){
    if (arms.indexOf(g.armLabel) < 0) arms.push(g.armLabel);
    if (seeds.indexOf(g.seed) < 0) seeds.push(g.seed);
  });
  var A = 0;                                            // selected arm index

  function pool(arm, seed){
    return games.filter(function(g){
      return g.arm === arm && (seed === undefined || g.seed === seed); });
  }
  function outcome(g){ return g.live ? "in progress" : (g.died ? "died" : "survived"); }
  function seedLabel(arm, sd){
    var mine = pool(arm, sd);
    if (!mine.length) return "seed " + sd + " \\u00b7 no data";
    if (mine.length < 2)
      return "seed " + sd + " \\u00b7 BALROG " + mine[0].bal.toFixed(2) +
             " \\u00b7 Dlvl " + mine[0].dlvl + " \\u00b7 " + outcome(mine[0]);
    var b = mine.map(function(g){ return g.bal; });
    return "seed " + sd + " \\u00b7 " + mine.length + " runs \\u00b7 BALROG " +
           Math.min.apply(null, b).toFixed(2) + "\\u2013" + Math.max.apply(null, b).toFixed(2);
  }

  armHost.hidden = arms.length < 2;
  if (!armHost.hidden){
    label(armHost, "arm");
    arms.forEach(function(nm, ai){ chip(armHost, nm, false, function(){ stop(); pickArm(ai); }); });
  }
  label(seedHost, "game");
  label(runHost, "run");

  // screens are stored as row diffs against the previous turn: replay to rebuild
  function screenAt(n){
    var t = games[G].turns;
    if (built > n) { rows = []; built = -1; }
    for (var j = built + 1; j <= n; j++)
      t[j].d.forEach(function(p){ rows[p[0]] = p[1]; });
    built = n;
    return rows.join("\\n");
  }

  function pickArm(ai){
    A = ai;
    mark(armHost, ai + 1);
    clear(seedHost);
    seeds.forEach(function(sd){
      var b = chip(seedHost, seedLabel(ai, sd), false, function(){ stop(); pick(sd, 1); });
      if (!pool(ai, sd).length) b.disabled = true;
    });
    var first = seeds.filter(function(sd){ return pool(ai, sd).length; })[0];
    pick(first, 1);
  }

  function pick(seed, run){
    var n = games.findIndex(function(g){
      return g.arm === A && g.seed === seed && g.run === run; });
    if (n < 0) return;
    G = n; rows = []; built = -1;
    scrub.max = LAST();
    mark(seedHost, seeds.indexOf(seed) + 1);
    // a run row appears only once this seed has more than one repetition
    var mine = pool(A, seed);
    runHost.hidden = mine.length < 2;
    clear(runHost);
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
      "(" + t.a + ")</b>" + (t.s === "failed" ? " \\u2192 failed" : "") +
      (t.k ? " \\u00b7 keys <b>" + t.k + "</b>" : "") +
      " \\u00b7 HP " + t.hp + "/" + t.mhp + " \\u00b7 Dlvl " + t.dl + " \\u00b7 T " + t.T +
      (g.live ? " \\u00b7 <b>run still in progress</b>" : "");
  }
  pickArm(0);
})();
"""


def render_gates(events):
    dove = sum(e["dove"] for e in events)
    rows = [f'<div class="gd-sum">The run showed the model <b>{len(events)}</b> descent '
            f'panels across {len(set(e["seed"] for e in events))} seeds. On <b>{dove}</b> of '
            f'them the very next call repeated the descend keystroke unchanged; on '
            f'<b>{len(events) - dove}</b> the model did something else next.</div>']
    for e in events:
        tag = ("dove", "descended anyway") if e["dove"] else ("held", "did something else")
        rows.append(
            f'<div class="gd-gate">'
            f'<div class="gd-who">seed {e["seed"]} · call {e["turn"] + 1}</div>'
            f'<div class="gd-panel">{html.escape(e["panel"])}</div>'
            f'<div class="gd-did">tripped by <b>{html.escape(e["call"])}</b><br>'
            f'next call <b>{html.escape(e["next"])}</b></div>'
            f'<span class="gd-tag {tag[0]}">{tag[1]}</span></div>')
    return ('<div class="gamedemo" id="gamedemo-gates">\n<style>'
            + CSS_COMMON + CSS_GATES + "</style>\n" + "\n".join(rows) + "\n</div>\n")


def render(name, kind, payload):
    eid = "gamedemo-" + name
    if kind == "gates":
        return render_gates(payload)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    js = (JS_KEYSTROKES if kind == "keystrokes" else JS_GAMES)
    js = js.replace("__PLAYER__", JS_PLAYER).replace("__EID__", eid)
    picks = PICKS_ONE if kind == "keystrokes" else PICKS_GAMES
    return FRAGMENT.format(eid=eid, css=CSS_COMMON, picks=picks, data=data, js=js)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the in-article gameplay embeds.")
    ap.add_argument("--data-root", default="outputs", help="root holding the run dirs")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="additional root searched when a run dir is absent from --data-root")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "blog", "embeds"))
    ap.add_argument("--only", nargs="*", help="subset of demo names (default: all)")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    roots = [args.data_root] + args.extra_root

    def resolve(rel):
        for r in roots:
            if os.path.isdir(os.path.join(r, rel)):
                return r
        return roots[0]

    for name, spec in DEMOS.items():
        if args.only and name not in args.only:
            continue
        if spec["kind"] == "keystrokes":
            payload = collect_keystrokes(os.path.join(resolve(spec["run"]), spec["run"]),
                                         spec["seed"], spec["turns"])
            note = f"{len(payload)} actions, {sum(len(c['frames']) for c in payload)} frames"
        elif spec["kind"] == "gates":
            payload = collect_gates(resolve(spec["run"]), spec["run"], spec["seeds"])
            note = (f"{len(payload)} gate panels, "
                    f"{sum(e['dove'] for e in payload)} answered by descending anyway")
        else:
            arms = [{"label": a["label"],
                     "dirs": [os.path.join(resolve(d), d) for d in a["dirs"]]}
                    for a in spec["arms"]]
            payload = collect_games("", arms, spec["seeds"])
            live = sum(g["live"] for g in payload)
            note = (f"{len(payload)} games, {sum(len(g['turns']) for g in payload)} turns"
                    + (f", {live} still in progress" if live else ""))
        out = os.path.join(args.outdir, name + ".html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render(name, spec["kind"], payload))
        print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB): {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
