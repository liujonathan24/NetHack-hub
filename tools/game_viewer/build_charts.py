#!/usr/bin/env python3
"""Build the in-article charts for blog/index.md.

    python -m tools.game_viewer.build_charts --compare /root/nld/compare_data.json \
        --data-root /root/nld/outputs-copy

Like `build_demo.py`, this writes committed fragments under `blog/embeds/` that
`build_index.py` splices in at

    <div class="game-embed" data-demo="pace"></div>

`pace` puts our reduced-skill seeds on the same axes as the two human
populations decoded from NLD-NAO: BALROG max against game turns, log x. The
geometry is computed here and shipped as static SVG -- the chart is fully
readable with no JavaScript; JS only adds the crosshair readout.

Colors are the validated three-slot categorical palette (blue / orange / aqua),
which clears the CVD and normal-vision floors in both modes; aqua sits under 3:1
on the light surface, so every series is also directly labelled and a table view
ships beneath the figure.
"""
from __future__ import annotations
import argparse, glob, html, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "environments", "nethack"))

W, H = 760, 400
PAD = {"l": 52, "r": 116, "t": 18, "b": 46}
CELL = "e7_seed/NPCORE_v3__prime_agent"


def agent_curves(data_root):
    """Per seed: [(game_turn, balrog_max)] on running maxima of Dlvl and XL."""
    from nethack_harness.prompt.balrog import balrog_progress
    out = {}
    for f in sorted(glob.glob(os.path.join(data_root, CELL, "turns", "*.ndjson"))):
        seed = int(os.path.basename(f).split("_")[0])
        pts, mdl, mxl = [], 0, 0
        for line in open(f):
            r = json.loads(line)
            t = (r.get("status") or {}).get("time")
            if t is None:
                continue
            mdl = max(mdl, r.get("max_dlvl_reached") or 0, r.get("dlvl") or 0)
            mxl = max(mxl, (r.get("status") or {}).get("experience_level") or 0)
            pts.append((t, round(balrog_progress(mdl, mxl) * 100, 3)))
        if pts:
            out[seed] = pts
    return out


def value_at(curve, t):
    """Carry-forward: a finished game holds its final score."""
    v = 0.0
    for pt in curve:
        if pt[0] > t:
            break
        v = pt[1]
    return v


def build_series(compare, data_root):
    cmp_ = json.load(open(compare))
    grid = cmp_["grid"]
    ag = agent_curves(data_root)
    ours_end = max(c[-1][0] for c in ag.values())
    # our seeds only exist out to ~850 turns; past that a "median" would just be
    # five dead games carried forward, so the agent series stops where play does
    ogrid = [t for t in grid if t <= ours_end]
    ours = {s: [value_at(c, t) for t in ogrid] for s, c in ag.items()}
    med = [st.median([ours[s][i] for s in ours]) for i in range(len(ogrid))]
    alive = [sum(1 for c in ag.values() if c[-1][0] >= t) for t in ogrid]
    return {
        "grid": grid, "ogrid": ogrid, "ours_end": ours_end,
        "nao": cmp_["pop"]["nao"]["series"]["max"],
        "top10": cmp_["pop"]["top10"]["series"]["max"],
        "nao_stats": cmp_["pop"]["nao"]["stats"],
        "top10_stats": cmp_["pop"]["top10"]["stats"],
        "ours": ours, "ours_med": med, "alive": alive,
    }


# ------------------------------------------------------------------ geometry

def scales(ymax, xmin, xmax):
    lx0, lx1 = math.log10(xmin), math.log10(xmax)
    def X(t):
        t = max(t, xmin)
        return PAD["l"] + (math.log10(t) - lx0) / (lx1 - lx0) * (W - PAD["l"] - PAD["r"])
    def Y(v):
        return H - PAD["b"] - (v / ymax) * (H - PAD["t"] - PAD["b"])
    return X, Y


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def band(xs, lo, hi, X, Y):
    up = " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, hi))
    dn = " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(reversed(xs), reversed(lo)))
    return f"M{up} L{dn} Z"


CSS = """
.viz{--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
  --viz-ink:var(--ink,#23202a);--viz-ink2:var(--ink2,#544c5e);--viz-mut:var(--muted,#8a8194);
  --viz-line:var(--line,#e7e2df);--viz-surface:var(--surface,#fff);
  margin:1.6em 0;border:1px solid var(--viz-line);border-radius:10px;
  background:var(--viz-surface);padding:1rem 1rem .6rem;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .viz{
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;}}
:root[data-theme="dark"] .viz{--s1:#3987e5;--s2:#d95926;--s3:#199e70;}
.viz figure{margin:0;position:relative;}
.viz svg{display:block;width:100%;height:auto;overflow:visible;}
.viz .ttl{font-size:.95rem;font-weight:650;color:var(--viz-ink);}
.viz .sub{font-size:.8rem;color:var(--viz-mut);margin-bottom:.5rem;}
.viz .lg{display:flex;flex-wrap:wrap;gap:.9rem;font-size:.78rem;color:var(--viz-ink2);
  margin-top:.5rem;}
.viz .lg i{display:inline-flex;align-items:center;gap:.35rem;font-style:normal;}
.viz .sw{width:15px;height:0;border-top:2px solid;display:inline-block;}
.viz .sw.bd{height:9px;border:0;border-radius:2px;}
.viz text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums;}
.viz .ax{fill:var(--viz-mut);font-size:10px;}
.viz .dl{font-size:10.5px;font-weight:600;}
.viz .gr{stroke:var(--viz-line);stroke-width:1;}
.viz .tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--viz-surface);border:1px solid var(--viz-line);border-radius:5px;
  padding:.4rem .55rem;font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color:var(--viz-ink);white-space:nowrap;font-variant-numeric:tabular-nums;
  box-shadow:0 4px 14px rgba(10,14,22,.14);z-index:3;}
.viz details{margin-top:.7rem;font-size:.82rem;color:var(--viz-ink2);}
.viz summary{cursor:pointer;color:var(--viz-mut);}
.viz table{width:100%;border-collapse:collapse;margin-top:.5rem;font-size:.8rem;
  font-variant-numeric:tabular-nums;}
.viz th,.viz td{text-align:right;padding:.3em .5em;border-bottom:1px solid var(--viz-line);}
.viz th:first-child,.viz td:first-child{text-align:left;}
"""

JS = """
(function(){
  var root=document.getElementById("__EID__"), D=JSON.parse(
    root.querySelector('[data-role=data]').textContent);
  var svg=root.querySelector("svg"), tip=root.querySelector(".tip"),
      cross=root.querySelector("[data-role=cross]"), fig=root.querySelector("figure");
  function nearest(t){var b=0;for(var i=0;i<D.grid.length;i++){
    if(Math.abs(Math.log(D.grid[i])-Math.log(t))<Math.abs(Math.log(D.grid[b])-Math.log(t)))b=i;}
    return b;}
  svg.addEventListener("pointermove",function(e){
    var r=svg.getBoundingClientRect(), sx=(e.clientX-r.left)/r.width*D.W;
    if(sx<D.padl||sx>D.W-D.padr){tip.style.opacity=0;cross.setAttribute("opacity",0);return;}
    var t=Math.pow(10,D.lx0+(sx-D.padl)/(D.W-D.padl-D.padr)*(D.lx1-D.lx0));
    var i=nearest(t), turn=D.grid[i];
    cross.setAttribute("x1",D.xs[i]);cross.setAttribute("x2",D.xs[i]);cross.setAttribute("opacity",1);
    var rows="turn <b>"+turn.toLocaleString()+"</b>";
    rows+="<br>NAO median "+D.nao[i].toFixed(2)+"%";
    rows+="<br>top-10 median "+D.top10[i].toFixed(2)+"%";
    rows+= i<D.ours.length ? "<br>ours median "+D.ours[i].toFixed(2)+"% ("+D.alive[i]+"/5 still playing)"
                           : "<br>ours \\u2014 all runs over";
    tip.innerHTML=rows;
    tip.style.opacity=1;
    var px=D.xs[i]/D.W*r.width;
    tip.style.left=Math.min(r.width-tip.offsetWidth-4,Math.max(0,px+10))+"px";
    tip.style.top="8px";
  });
  svg.addEventListener("pointerleave",function(){tip.style.opacity=0;cross.setAttribute("opacity",0);});
})();
"""


def render_pace(S):
    # Scale to the medians, not the bands: the top-10 p75 runs past 40% and
    # would squash every line worth reading into the bottom of the frame. The
    # bands are clipped to the plot area instead.
    ymax = math.ceil(max(max(S["nao"]["50"]), max(S["top10"]["50"]),
                         max(S["ours_med"])) * 1.12)
    X, Y = scales(ymax, 10, S["grid"][-1])
    g = S["grid"]
    parts = [f'<clipPath id="pace-clip"><rect x="{PAD["l"]}" y="{PAD["t"]}" '
             f'width="{W-PAD["l"]-PAD["r"]}" height="{H-PAD["t"]-PAD["b"]}"/></clipPath>']

    # recessive grid + axes
    for v in range(0, int(ymax) + 1, max(1, int(ymax // 6))):
        parts.append(f'<line class="gr" x1="{PAD["l"]}" y1="{Y(v):.1f}" '
                     f'x2="{W-PAD["r"]}" y2="{Y(v):.1f}"/>')
        parts.append(f'<text class="ax" x="{PAD["l"]-8}" y="{Y(v)+3.5:.1f}" '
                     f'text-anchor="end">{v}%</text>')
    for t in (10, 100, 1000, 10000, 50000):
        parts.append(f'<text class="ax" x="{X(t):.1f}" y="{H-PAD["b"]+16}" '
                     f'text-anchor="middle">{t:,}</text>')
    parts.append(f'<text class="ax" x="{(PAD["l"]+W-PAD["r"])/2:.0f}" y="{H-8}" '
                 f'text-anchor="middle">game turns (log)</text>')

    # human percentile bands, then medians (clipped: p75 leaves the frame)
    parts.append('<g clip-path="url(#pace-clip)">')
    for key, col in (("nao", "var(--s1)"), ("top10", "var(--s2)")):
        s = S[key]
        parts.append(f'<path d="{band(g, s["25"], s["75"], X, Y)}" fill="{col}" '
                     f'opacity=".13"/>')
    # our seeds: faint per-game lines under the median
    og = S["ogrid"]
    for seed, ys in S["ours"].items():
        parts.append(f'<path d="{path(og, ys, X, Y)}" fill="none" stroke="var(--s3)" '
                     f'stroke-width="1" opacity=".38"/>')
    for key, col in (("nao", "var(--s1)"), ("top10", "var(--s2)")):
        parts.append(f'<path d="{path(g, S[key]["50"], X, Y)}" fill="none" stroke="{col}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
    parts.append(f'<path d="{path(og, S["ours_med"], X, Y)}" fill="none" stroke="var(--s3)" '
                 f'stroke-width="2.5" stroke-linejoin="round"/>')
    parts.append("</g>")

    # where our runs stop: a budget/death boundary, not a plateau
    xe = X(S["ours_end"])
    parts.append(f'<line x1="{xe:.1f}" y1="{PAD["t"]}" x2="{xe:.1f}" y2="{H-PAD["b"]}" '
                 f'stroke="var(--s3)" stroke-width="1" stroke-dasharray="3 3" opacity=".55"/>')
    parts.append(f'<text class="ax" x="{xe+5:.1f}" y="{PAD["t"]+10}" fill="var(--s3)">'
                 f'last of our runs ends (turn {S["ours_end"]:,})</text>')

    # direct labels — required relief for the aqua slot on a light surface
    for lbl, col, x, y in (
        ("NAO population", "var(--s1)", g[-1], S["nao"]["50"][-1]),
        ("NAO top 10", "var(--s2)", g[-1], S["top10"]["50"][-1]),
    ):
        parts.append(f'<text class="dl" x="{X(x)+7:.1f}" y="{Y(y)+3.5:.1f}" fill="{col}">'
                     f'{html.escape(lbl)}</text>')
    parts.append(f'<text class="dl" x="{X(og[-1])+7:.1f}" y="{Y(S["ours_med"][-1])-6:.1f}" '
                 f'fill="var(--s3)">our seeds</text>')
    parts.append('<line data-role="cross" y1="%d" y2="%.1f" stroke="var(--viz-mut)" '
                 'stroke-width="1" stroke-dasharray="2 3" opacity="0" x1="0" x2="0"/>'
                 % (PAD["t"], H - PAD["b"]))

    rows = []
    for t in (100, 300, 500, 850, 2000, 10000):
        i = min(range(len(g)), key=lambda k: abs(g[k] - t))
        ours = (f'{S["ours_med"][i]:.2f}%' if i < len(og) else "—")
        alive = (f'{S["alive"][i]}/5' if i < len(og) else "0/5")
        rows.append(f"<tr><td>{g[i]:,}</td><td>{S['nao']['50'][i]:.2f}%</td>"
                    f"<td>{S['top10']['50'][i]:.2f}%</td><td>{ours}</td><td>{alive}</td></tr>")

    data = json.dumps({
        "W": W, "padl": PAD["l"], "padr": PAD["r"],
        "lx0": math.log10(10), "lx1": math.log10(g[-1]),
        "grid": g, "xs": [round(X(t), 1) for t in g],
        "nao": S["nao"]["50"], "top10": S["top10"]["50"],
        "ours": S["ours_med"], "alive": S["alive"],
    }, separators=(",", ":"))

    return f"""<div class="viz" id="gamechart-pace">
<style>{CSS}</style>
<div class="ttl">BALROG progression against game turns</div>
<div class="sub">Our five reduced-skill seeds against every decoded NLD-NAO game
({S['nao_stats']['n']:,}) and the ten strongest NAO players
({S['top10_stats']['n']:,} games). Medians, carried forward; shaded bands are p25–p75,
clipped where the top-10 band leaves the frame.</div>
<figure><svg viewBox="0 0 {W} {H}" role="img"
 aria-label="BALROG progression against game turns for three populations">
{chr(10).join(parts)}
</svg><div class="tip"></div></figure>
<div class="lg">
  <i><span class="sw" style="border-color:var(--s1)"></span>NAO population (median)</i>
  <i><span class="sw" style="border-color:var(--s2)"></span>NAO top 10 (median)</i>
  <i><span class="sw" style="border-color:var(--s3)"></span>our seeds (median; faint = each seed)</i>
  <i><span class="sw bd" style="background:var(--viz-mut);opacity:.2"></span>p25–p75</i>
</div>
<details><summary>the same numbers as a table</summary>
<table><thead><tr><th>game turn</th><th>NAO median</th><th>top-10 median</th>
<th>our median</th><th>our games still playing</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></details>
<script type="application/json" data-role="data">{data}</script>
<script>{JS.replace("__EID__", "gamechart-pace")}</script>
</div>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the in-article charts.")
    ap.add_argument("--compare", default="/root/nld/compare_data.json",
                    help="merged human/agent population curves (see prep_compare.py)")
    ap.add_argument("--data-root", default="outputs")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "blog", "embeds"))
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    S = build_series(args.compare, args.data_root)
    out = os.path.join(args.outdir, "pace.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_pace(S))
    print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB): "
          f"{len(S['ours'])} seeds to turn {S['ours_end']}, "
          f"human grid to {S['grid'][-1]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
