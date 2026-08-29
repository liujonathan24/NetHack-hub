#!/usr/bin/env python3
"""E16 run explorer: SVG node-and-edge tree left, detail pane right."""
import json, os

SP = os.environ.get("E16_ART_DIR") or os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, "e16_explorer.html")
DATA = json.load(open(f"{SP}/e16_data.json"))
CURVES = json.load(open(f"{SP}/curves.json"))

HTML = r"""<title>E16 Run Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f1f3f6; --surface:#fff; --surface-2:#f7f8fa;
  --ink:#14171d; --ink-2:#565d6c; --ink-3:#818897;
  --rule:#dde1e8; --rule-2:#eceff3;
  --accent:#b0501c; --death:#982f2f; --gain:#276b4e; --flat:#9aa1ad;
  --shadow:0 1px 2px rgba(20,23,29,.05),0 4px 14px rgba(20,23,29,.05);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0e1116; --surface:#161a21; --surface-2:#1b2029;
  --ink:#e7eaef; --ink-2:#9aa2b1; --ink-3:#727b8b;
  --rule:#262c37; --rule-2:#1f242d;
  --accent:#e0783c; --death:#e2736c; --gain:#54ba8c; --flat:#5d6674;
  --shadow:0 1px 2px rgba(0,0,0,.35),0 4px 16px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --ground:#0e1116; --surface:#161a21; --surface-2:#1b2029;
  --ink:#e7eaef; --ink-2:#9aa2b1; --ink-3:#727b8b;
  --rule:#262c37; --rule-2:#1f242d;
  --accent:#e0783c; --death:#e2736c; --gain:#54ba8c; --flat:#5d6674;
  --shadow:0 1px 2px rgba(0,0,0,.35),0 4px 16px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif;font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.shell{max-width:1340px;margin:0 auto;padding:0 20px 60px}
h1{font-family:Archivo,sans-serif;font-size:clamp(1.6rem,3.4vw,2.3rem);font-weight:700;letter-spacing:-.02em;margin:0 0 .25em}
.eyebrow{font-family:Archivo,sans-serif;font-size:.68rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 .9em}
header{padding:38px 0 20px;border-bottom:1px solid var(--rule)}
.sub{color:var(--ink-2);max-width:68ch;margin:0}
code{font-family:"IBM Plex Mono",monospace;font-size:.86em;background:var(--surface-2);
  border:1px solid var(--rule-2);border-radius:4px;padding:.06em .32em}
nav.runs{display:flex;gap:2px;overflow-x:auto;border-bottom:1px solid var(--rule);margin:0 0 16px;position:sticky;top:0;background:var(--ground);z-index:10}
nav.runs button{font-family:Archivo,sans-serif;font-size:.8rem;font-weight:600;background:none;border:0;
  border-bottom:2px solid transparent;color:var(--ink-3);padding:13px 14px 10px;cursor:pointer;white-space:nowrap}
nav.runs button:hover{color:var(--ink)}
nav.runs button[aria-current="true"]{color:var(--ink);border-bottom-color:var(--accent)}
nav.runs button:focus-visible{outline:2px solid var(--accent);outline-offset:-3px}
.panes{display:grid;grid-template-columns:minmax(320px,440px) 1fr;gap:20px;align-items:start}
@media (max-width:900px){.panes{grid-template-columns:1fr}}
.pane{background:var(--surface);border:1px solid var(--rule);border-radius:11px;box-shadow:var(--shadow)}
.pane h2{font-family:Archivo,sans-serif;font-size:.68rem;font-weight:600;letter-spacing:.11em;text-transform:uppercase;
  color:var(--ink-3);margin:0;padding:12px 16px;border-bottom:1px solid var(--rule-2)}
#treepane{position:sticky;top:56px;max-height:calc(100vh - 86px);display:flex;flex-direction:column}
#svgwrap{overflow:auto;flex:1;padding:6px}
.runmeta{padding:10px 16px;border-bottom:1px solid var(--rule-2);font-family:"IBM Plex Mono",monospace;
  font-size:.72rem;color:var(--ink-2);font-variant-numeric:tabular-nums;line-height:1.75}
.runmeta b{color:var(--ink);font-weight:600}
.warn{color:var(--death)}
/* svg tree */
svg.tree{display:block}
svg.tree .edge{fill:none;stroke:var(--rule);stroke-width:1.6}
svg.tree .edge.hot{stroke:var(--accent);stroke-width:2.2}
svg.tree g.node{cursor:pointer}
svg.tree g.node circle.hit{fill:transparent}
svg.tree g.node circle.dot{stroke-width:2.5;transition:r .12s}
svg.tree g.node:hover circle.dot{r:15}
svg.tree g.node .lab{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;
  fill:var(--ink);text-anchor:middle;dominant-baseline:central;pointer-events:none}
svg.tree g.node .sub{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--ink-3);
  dominant-baseline:central;pointer-events:none}
svg.tree g.node.sel circle.dot{stroke:var(--accent);stroke-width:3.5}
svg.tree g.node.sel .lab{fill:var(--accent)}
svg.tree g.node:focus{outline:none}
svg.tree g.node:focus-visible circle.dot{stroke:var(--accent);stroke-dasharray:3 2}
.legend{padding:9px 16px 12px;font-family:Archivo,sans-serif;font-size:.66rem;color:var(--ink-3);
  border-top:1px solid var(--rule-2);display:flex;flex-wrap:wrap;gap:14px}
.legend i{font-style:normal;display:inline-block;width:10px;height:10px;border-radius:50%;
  vertical-align:-1px;margin-right:5px;border:2px solid}
/* segmented toggle */
.segbar{display:flex;gap:0;padding:10px 16px 0}
.segbar button{font-family:Archivo,sans-serif;font-size:.7rem;font-weight:600;letter-spacing:.04em;
  background:var(--surface-2);border:1px solid var(--rule);color:var(--ink-3);
  padding:5px 12px;cursor:pointer}
.segbar button:first-child{border-radius:6px 0 0 6px}
.segbar button:last-child{border-radius:0 6px 6px 0}
.segbar button+button{border-left:0}
.segbar button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.segbar button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#xy{overflow-x:auto;padding:12px 8px 4px}
svg.xy .norm{fill:none;stroke:var(--ink-3);stroke-width:2;stroke-dasharray:6 4}
svg.xy .path{fill:none;stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}
svg.xy .end{stroke-width:2}
svg.xy .normlab{font-family:Archivo,sans-serif;font-size:9.5px;font-weight:600;fill:var(--ink-3)}
/* chart */
#chart{overflow-x:auto;padding:12px 8px 4px}
svg.chart{display:block}
svg.chart .ax{stroke:var(--rule);stroke-width:1}
svg.chart .grid{stroke:var(--rule-2);stroke-width:1}
svg.chart .tick{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--ink-3)}
svg.chart .axlab{font-family:Archivo,sans-serif;font-size:10px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;fill:var(--ink-3)}
svg.chart path.ln{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
svg.chart path.ln.min{stroke-dasharray:4 3;stroke-width:1.8}
svg.chart .amark{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 3}
svg.chart .alab{font-family:"IBM Plex Mono",monospace;font-size:9px;fill:var(--ink-3)}
svg.chart .dimline{opacity:.28}
.legend .sw{display:inline-block;width:16px;height:0;border-top:2.5px solid;vertical-align:3px;margin-right:5px}
.legend .sw.dash{border-top-style:dashed}
/* detail */
.dhead{padding:16px 20px 12px;border-bottom:1px solid var(--rule-2)}
.dtitle{font-family:Archivo,sans-serif;font-size:1.25rem;font-weight:700;letter-spacing:-.015em;margin:0 0 6px}
.pills{display:flex;gap:7px;flex-wrap:wrap}
.pill{font-family:Archivo,sans-serif;font-size:.62rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  padding:2px 8px;border-radius:99px;border:1px solid currentColor}
.pill.d{color:var(--death)} .pill.g{color:var(--gain)} .pill.n{color:var(--ink-3)} .pill.a{color:var(--accent)}
.sect{padding:14px 20px;border-bottom:1px solid var(--rule-2)}
.sect:last-child{border-bottom:0}
.sect h3{font-family:Archivo,sans-serif;font-size:.66rem;font-weight:600;letter-spacing:.11em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 8px}
.kv{display:grid;grid-template-columns:120px 1fr;gap:6px 12px;font-family:"IBM Plex Mono",monospace;
  font-size:.79rem;font-variant-numeric:tabular-nums}
.kv dt{color:var(--ink-3)} .kv dd{margin:0}
.q{font-family:"Source Serif 4",serif;font-size:.94rem;line-height:1.55;color:var(--ink-2);
  border-left:2px solid var(--rule);padding-left:13px;margin:0}
.q.said{border-left-color:var(--accent);color:var(--ink)}
.ck{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}
.ck span{font-family:"IBM Plex Mono",monospace;font-size:.7rem;background:var(--surface-2);
  border:1px solid var(--rule-2);border-radius:5px;padding:2px 7px;color:var(--ink-2)}
.empty{color:var(--ink-3);font-style:italic}
.hint{color:var(--ink-3);font-size:.9rem;padding:40px 20px;text-align:center}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="shell">
<header>
  <p class="eyebrow">E16 &middot; Go-Explore &middot; NetHack seed 1</p>
  <h1>Run Explorer</h1>
  <p class="sub"><b>Depth ratchets; experience does not.</b> Every circle is one attempt. An edge runs from an attempt to any attempt that resumed from a checkpoint it created &mdash; so a circle with two edges leaving it is a point the run branched. Click a circle for its full record.</p>
</header>
<nav class="runs" id="runs" aria-label="Runs"></nav>
<section class="pane" id="chartpane" style="margin-bottom:20px">
  <h2>Improvement over the run &mdash; cumulative LM turns vs best BALROG so far</h2>
  <p style="margin:0;padding:10px 16px 0;font-size:.86rem;color:var(--ink-2);max-width:78ch">Best-so-far, never resets. Solid is BALROG <b>max</b> (set by depth); dashed is <b>min</b> (set by XL). Dotted verticals mark where each attempt ended. Curves read the turn stream directly, so they include an attempt still in flight; the tree below shows closed attempts only.</p>
  <div class="segbar" role="group" aria-label="metric">
    <button data-m="both" aria-pressed="true">both</button>
    <button data-m="max" aria-pressed="false">BALROG max</button>
    <button data-m="min" aria-pressed="false">BALROG min</button>
  </div>
  <div id="chart"></div>
  <div class="legend" id="chartlegend"></div>
</section>

<section class="pane" id="xypane" style="margin-bottom:20px">
  <h2>Depth against experience &mdash; where every run actually went</h2>
  <p style="margin:0;padding:10px 16px 0;font-size:.86rem;color:var(--ink-2);max-width:78ch">Each path is one run's high-water state as it advanced: x is dungeon level, y is experience level. The dashed line is the pace <b>winning human games</b> keep &mdash; the median XL at which 433 ascensions first reached each depth. A run that hugs the bottom is diving without getting stronger.</p>
  <div id="xy"></div>
  <div class="legend" id="xylegend"></div>
</section>

<div class="panes">
  <section class="pane" id="treepane">
    <h2>Attempt tree</h2>
    <div class="runmeta" id="runmeta"></div>
    <div id="svgwrap"></div>
    <div class="legend">
      <span><i style="border-color:var(--gain);background:color-mix(in srgb,var(--gain) 25%,transparent)"></i>gained depth</span>
      <span><i style="border-color:var(--flat);background:color-mix(in srgb,var(--flat) 25%,transparent)"></i>gained nothing</span>
      <span><i style="border-color:var(--death);background:color-mix(in srgb,var(--death) 25%,transparent)"></i>started starving</span>
      <span>circle size = BALROG</span>
    </div>
  </section>
  <section class="pane"><h2>Attempt detail</h2><div id="detail"><p class="hint">Click a node.</p></div></section>
</div>
</div>

<script id="e16data" type="application/json">__DATA__</script>
<script id="e16curves" type="application/json">__CURVES__</script>
<script>
(function(){
var D=JSON.parse(document.getElementById('e16data').textContent);
// UNION, not just the attempt data. A run whose first attempt has not closed
// yet has no rows in e16_data.json but DOES have a progress curve, and it was
// being dropped from both plots without a word -- the failure mode this whole
// artifact keeps catching elsewhere. Curve-only runs get a tab and appear in
// the charts; their tree says why it is empty.
var CURVE_RUNS=JSON.parse(document.getElementById('e16curves').textContent);
var runs=Object.keys(D).slice();
Object.keys(CURVE_RUNS).forEach(function(r){ if(runs.indexOf(r)<0) runs.push(r); });
var cur=runs[0], sel=null;
var nav=document.getElementById('runs');
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function pct(a,b){return (b?Math.round(100*a/b):0)+'%';}
function nn(v,suf){return v==null?'—':(v+(suf||''));}

runs.forEach(function(r){
  var b=document.createElement('button');
  b.textContent=r+' · '+((D[r]&&D[r].arm==='matched_restart')?'null':'method')
    +((!D[r])?' · in flight':'');
  b.onclick=function(){cur=r;sel=null;draw();};
  nav.appendChild(b);
});

/* ---- layout: layered tree, depth = generation, y = leaf order ---- */
function layout(run){
  if(!D[run]) return {pos:{},kids:{},by:{},roots:[],rows:0};
  var A=D[run].attempts, by={}, kids={};
  A.forEach(function(a){by[a.n]=a;});
  A.forEach(function(a){ if(a.parent!=null&&by[a.parent]) (kids[a.parent]=kids[a.parent]||[]).push(a.n); });
  Object.keys(kids).forEach(function(k){kids[k].sort(function(x,y){return x-y;});});
  var roots=A.filter(function(a){return a.parent==null||!by[a.parent];})
             .map(function(a){return a.n;}).sort(function(x,y){return x-y;});
  var pos={}, row=0;
  function walk(n,d){
    var ks=kids[n]||[];
    if(!ks.length){ pos[n]={d:d,y:row++}; return pos[n].y; }
    var ys=ks.map(function(c){return walk(c,d+1);});
    var y=(Math.min.apply(null,ys)+Math.max.apply(null,ys))/2;
    /* keep the parent on its own row so no two circles collide */
    pos[n]={d:d,y:y};
    return y;
  }
  roots.forEach(function(r){walk(r,0);});
  return {pos:pos,kids:kids,by:by,roots:roots,rows:row};
}

function radius(a){
  var b=a.balrog==null?0:a.balrog;      /* 0..31 */
  return 9+Math.min(9,Math.sqrt(b)*1.9);
}
function colorOf(a){
  if(a.impaired) return 'var(--death)';
  if(a.gain>0) return 'var(--gain)';
  return 'var(--flat)';
}

function drawTree(){
  var host0=document.getElementById('svgwrap');
  if(!D[cur]){
    host0.innerHTML='<p class="hint">This run is still on its first attempt, so no '
      +'node exists yet \u2014 an attempt becomes a node when it closes. Its progress '
      +'is already in both plots above.</p>';
    return;
  }
  var L=layout(cur), A=D[cur].attempts;
  var COLW=118, ROWH=62, PADX=34, PADY=30;
  var maxd=0,maxy=0;
  Object.keys(L.pos).forEach(function(k){maxd=Math.max(maxd,L.pos[k].d);maxy=Math.max(maxy,L.pos[k].y);});
  var W=PADX*2+maxd*COLW+120, H=PADY*2+maxy*ROWH+30;
  function X(n){return PADX+L.pos[n].d*COLW;}
  function Y(n){return PADY+L.pos[n].y*ROWH;}
  var s='<svg class="tree" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'" role="tree">';
  /* edges first */
  A.forEach(function(a){
    (L.kids[a.n]||[]).forEach(function(c){
      var x1=X(a.n),y1=Y(a.n),x2=X(c),y2=Y(c), mx=(x1+x2)/2;
      var hot=(sel===a.n||sel===c)?' hot':'';
      s+='<path class="edge'+hot+'" d="M'+x1+','+y1+' C'+mx+','+y1+' '+mx+','+y2+' '+x2+','+y2+'"/>';
    });
  });
  /* nodes */
  A.forEach(function(a){
    var x=X(a.n),y=Y(a.n),r=radius(a),c=colorOf(a);
    s+='<g class="node'+(sel===a.n?' sel':'')+'" data-n="'+a.n+'" tabindex="0" role="treeitem"'
      +' aria-label="attempt '+a.n+', reached dungeon level '+a.max_dlvl+'">'
      +'<circle class="dot" cx="'+x+'" cy="'+y+'" r="'+r+'" fill="color-mix(in srgb,'+c+' 22%,transparent)" stroke="'+c+'"/>'
      +'<text class="lab" x="'+x+'" y="'+y+'">a'+a.n+'</text>'
      +'<text class="sub" x="'+(x+r+7)+'" y="'+(y-5)+'">D'+a.max_dlvl+' XL'+a.max_xl
      +(a.deficit>0?'  (-'+a.deficit+')':'')+'</text>'
      +'<text class="sub" x="'+(x+r+7)+'" y="'+(y+7)+'">'+(a.balrog==null?'—':a.balrog.toFixed(1)+'%')
      +' / '+(a.balrog_min==null?'—':a.balrog_min.toFixed(1)+'%')+'</text>'
      +'<circle class="hit" cx="'+x+'" cy="'+y+'" r="'+(r+9)+'"/></g>';
  });
  s+='</svg>';
  var wrap=document.getElementById('svgwrap'); wrap.innerHTML=s;
  wrap.querySelectorAll('g.node').forEach(function(g){
    function pick(){sel=+g.dataset.n;draw();}
    g.addEventListener('click',pick);
    g.addEventListener('keydown',function(e){
      if(e.key==='Enter'||e.key===' '){e.preventDefault();pick();}});
  });
}

function drawMeta(){
  var r=D[cur];
  if(!r){
    var c=CURVE_RUNS[cur]||{};
    document.getElementById('runmeta').innerHTML=
       '<b>'+esc(c.arm||'?')+'</b> · '+esc(c.tier||'?')
     +'<br>no attempt has closed yet — <b>'+(c.total_turns||0)+'</b> LM turns so far'
     +'<br>BALROG max <b>'+(c.final_best_max!=null?c.final_best_max+'%':'—')+'</b>'
     +' · <b>min '+(c.final_best_min!=null?c.final_best_min+'%':'—')+'</b>';
    return;
  }
  var gains=r.attempts.map(function(a){return a.gain==null?0:a.gain;});
  var mean=gains.length?gains.reduce(function(s,x){return s+x;},0)/gains.length:0;
  var f={}; r.attempts.forEach(function(a){if(a.from_ck)f[a.from_ck]=(f[a.from_ck]||0)+1;});
  var nf=Object.keys(f).filter(function(k){return f[k]>1;}).length;
  document.getElementById('runmeta').innerHTML=
     '<b>'+esc(r.arm)+'</b> · '+esc(r.tier)+' · seed '+r.seed
   +'<br>attempts <b>'+r.attempts.length+'</b>/'+r.max_attempts+' · checkpoints <b>'+r.n_ck+'</b>'
   +' · forks <b>'+nf+'</b> · mean gain <b>'+mean.toFixed(2)+'</b>'
   +'<br>BALROG max <b>'+nn(r.balrog_max,'%')+'</b> (mean '+nn(r.balrog_mean,'%')+')'
   +' · <b>min '+nn(r.balrog_min_max,'%')+'</b> (mean '+nn(r.balrog_min_mean,'%')+')'
   +(r.hunger_recorded?'':'<br><span class="warn">selector could not see nutrition in this run</span>');
}

function row(k,v){return '<dt>'+k+'</dt><dd>'+v+'</dd>';}

function drawDetail(){
  var host=document.getElementById('detail');
  if(!D[cur]){host.innerHTML='<p class="hint">No closed attempt to inspect yet.</p>';return;}
  if(sel==null){host.innerHTML='<p class="hint">Click a node.</p>';return;}
  var a=D[cur].attempts.filter(function(x){return x.n===sel;})[0];
  if(!a){host.innerHTML='<p class="hint">Click a node.</p>';return;}
  var s=a.start||{}, h='';
  h+='<div class="dhead"><p class="dtitle">Attempt '+a.n+'</p><div class="pills">'
    +'<span class="pill d">'+esc(a.outcome)+(a.censor?':'+esc(a.censor):'')+'</span>'
    +'<span class="pill '+(a.gain>0?'g':'n')+'">gain '+(a.gain==null?'?':(a.gain>0?'+'+a.gain:'0'))+'</span>'
    +(a.impaired?'<span class="pill d">started '+esc(a.hunger0)+'</span>'
                :'<span class="pill n">'+esc(a.hunger0||'hunger unknown')+'</span>')
    +(a.agreed==null?'':'<span class="pill a">selector '+(a.agreed?'=':'≠')+' scripted</span>')
    +'</div></div>';
  h+='<div class="sect"><h3>Score</h3><dl class="kv">'
    +row('BALROG','<b>'+nn(a.balrog,'%')+'</b> <span style="color:var(--ink-3)">max — set by Dlvl</span>')
    +row('BALROG-min','<b>'+nn(a.balrog_min,'%')+'</b> <span style="color:var(--ink-3)">set by XL '+a.max_xl+'</span>')
    +row('reached','D'+a.max_dlvl+' · XL'+a.max_xl)
    +row('winners\u2019 norm',a.winners_norm_xl==null?'—':('XL'+a.winners_norm_xl+' at this depth'))
    +row('deficit',a.deficit==null?'—':(a.deficit>0
        ?'<b style="color:var(--death)">'+a.deficit+' levels under the winning pace</b>'
        :'<b style="color:var(--gain)">at or above pace</b>'))
    +'</dl>'
    +(a.deficit>0?'<p class="q" style="margin-top:9px">Closing this would move the MIN half of the pair, which is the half nothing in this program has ever moved. The max half is already set by depth.</p>':'')
    +'</div>';
  h+='<div class="sect"><h3>Start — the state it was handed</h3><dl class="kv">'
    +row('checkpoint','c'+esc(a.from_ck)+(a.parent?' <span style="color:var(--ink-3)">(made by a'+a.parent+')</span>'
                                                  :' <span style="color:var(--ink-3)">(seed)</span>'))
    +row('position','D'+nn(s.dlvl)+' · XL'+nn(s.xl)+' · '+esc(s.br))
    +row('health',nn(s.hp)+'/'+nn(s.mx)+' · '+pct(s.hp,s.mx))
    +row('nutrition','<b'+(a.impaired?' style="color:var(--death)"':'')+'>'+esc(a.hunger0||'unknown')+'</b>')
    +row('BALROG here',nn(s.balrog,'%')+' / '+nn(s.balrog_min,'%')+' min')
    +'</dl></div>';
  h+='<div class="sect"><h3>End</h3><dl class="kv">'
    +row('outcome','<b>'+esc(a.outcome)+'</b>'+(a.killer?' — '+esc(a.killer):''))
    +row('cost',nn(a.calls)+' calls · '+a.wall_m+' min · $'+a.usd)
    +(a.faints?row('fainting','<b style="color:var(--death)">'+a.faints+' faint messages</b>'):'')
    +'</dl>'+(a.endmsg?'<p class="q" style="margin-top:9px">'+esc(a.endmsg)+'</p>':'')+'</div>';
  h+='<div class="sect"><h3>Selection — what it was offered</h3><dl class="kv">'
    +row('chose','c'+esc(a.chose))
    +row('scripted pick',a.scripted?('c'+esc(a.scripted)+(a.agreed?' (agreed)':' (overruled)')):'—')
    +row('on frontier',a.on_frontier==null?'—':(a.on_frontier?'yes':'<b>no — took a dominated state</b>'))
    +row('candidates',nn(a.n_cand))
    +row('impaired offered',a.n_impaired==null
        ?'<span style="color:var(--ink-3)">not recorded — predates the hunger fix</span>'
        :a.n_impaired+(a.impaired_ids.length?' · '+a.impaired_ids.map(function(i){return 'c'+i;}).join(' '):''))
    +'</dl></div>';
  h+='<div class="sect"><h3>Told — the orchestrator’s directive</h3>'
    +(a.directive?'<p class="q">'+esc(a.directive)+'</p>':'<p class="empty">none</p>')+'</div>';
  h+='<div class="sect"><h3>Said — the player’s own last words</h3>'
    +(a.account?'<p class="q said">'+esc(a.account)+'</p>'
      :'<p class="empty">Empty. This run predates the <code>_final_text</code> fix, so the orchestrator received a blank account every round.</p>')
    +'</div>';
  if(a.made&&a.made.length){
    h+='<div class="sect"><h3>Checkpoints it created ('+a.n_made+')</h3><div class="ck">'
      +a.made.map(function(c){return '<span>c'+c.id+' D'+c.dlvl+' XL'+c.xl+'</span>';}).join('')
      +'</div></div>';
  }
  host.innerHTML=h;
}

var C=CURVE_RUNS;
var METRIC='both';
// normXL(d): median XL at which winners first reach depth d. Step function as
// published: XL 1 to depth 3, parity to 9, flattening to 12 by 15, flat to 20.
function normXL(d){
  if(d<1) return null;
  if(d<=3) return 1;
  if(d<=9) return d;
  if(d<=15) return Math.min(12, 9+Math.round((d-9)*3/6));
  if(d<=20) return 12;
  return null;
}
var RUNCOLORS={};
runs.forEach(function(r,i){
  RUNCOLORS[r]=['var(--accent)','var(--gain)','var(--flat)','var(--death)'][i%4];
});
function drawChart(){
  var W=Math.max(660,Math.min(1180,(window.innerWidth||1000)-90)), H=300;
  var L=54,R=18,T=14,B=40;
  var maxT=0,maxY=0;
  runs.forEach(function(r){var c=C[r]; if(!c)return;
    maxT=Math.max(maxT,c.total_turns);
    maxY=Math.max(maxY, METRIC==='min'?c.final_best_min:c.final_best_max);});
  maxY=Math.max(5,Math.ceil(maxY/5)*5); maxT=Math.ceil(maxT/200)*200;
  function X(t){return L+(W-L-R)*(t/maxT);}
  function Y(v){return T+(H-T-B)*(1-v/maxY);}
  var s='<svg class="chart" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'">';
  for(var g=0;g<=maxY;g+=(maxY>20?5:1)){
    s+='<line class="grid" x1="'+L+'" y1="'+Y(g)+'" x2="'+(W-R)+'" y2="'+Y(g)+'"/>'
      +'<text class="tick" x="'+(L-7)+'" y="'+(Y(g)+3)+'" text-anchor="end">'+g+'%</text>';
  }
  for(var t=0;t<=maxT;t+=Math.max(200,Math.round(maxT/6/200)*200)){
    s+='<text class="tick" x="'+X(t)+'" y="'+(H-B+15)+'" text-anchor="middle">'+t+'</text>';
  }
  s+='<line class="ax" x1="'+L+'" y1="'+Y(0)+'" x2="'+(W-R)+'" y2="'+Y(0)+'"/>';
  s+='<text class="axlab" x="'+((L+W-R)/2)+'" y="'+(H-6)+'" text-anchor="middle">cumulative LM turns</text>';
  s+='<text class="axlab" transform="translate(13,'+((T+H-B)/2)+') rotate(-90)" text-anchor="middle">best BALROG so far</text>';
  runs.forEach(function(r){
    var c=C[r]; if(!c||!c.points.length)return;
    var col=RUNCOLORS[r], dim=(r===cur?'':' dimline');
    function path(key){
      var d='';
      c.points.forEach(function(p,i){
        var x=X(p.t),y=Y(p[key]);
        if(i===0){d='M'+x+','+y;}
        else{d+=' L'+x+','+Y(c.points[i-1][key])+' L'+x+','+y;}
      });
      return d;
    }
    if(METRIC!=='min') s+='<path class="ln'+dim+'" d="'+path('best_max')+'" stroke="'+col+'"/>';
    if(METRIC!=='max') s+='<path class="ln min'+dim+'" d="'+path('best_min')+'" stroke="'+col+'"/>';
    if(r===cur){
      (c.attempt_marks||[]).forEach(function(m){
        s+='<line class="amark" x1="'+X(m.t_end)+'" y1="'+T+'" x2="'+X(m.t_end)+'" y2="'+Y(0)+'"/>'
          +'<text class="alab" x="'+(X(m.t_end)-3)+'" y="'+(T+9)+'" text-anchor="end">a'+m.attempt+'</text>';
      });
    }
  });
  s+='</svg>';
  document.getElementById('chart').innerHTML=s;
  document.getElementById('chartlegend').innerHTML=
    runs.map(function(r){
      var c=C[r]; if(!c)return '';
      return '<span style="'+(r===cur?'font-weight:600;color:var(--ink)':'')+'">'
        +'<i class="sw" style="border-color:'+RUNCOLORS[r]+'"></i>'+r
        +' <span style="color:var(--ink-3)">'+(c.final_best_max)+'% / '+(c.final_best_min)+'% min</span></span>';
    }).join('')
    +'<span><i class="sw" style="border-color:var(--ink-3)"></i>solid = BALROG max</span>'
    +'<span><i class="sw dash" style="border-color:var(--ink-3)"></i>dashed = BALROG min</span>';
}

function drawXY(){
  var W=Math.max(620,Math.min(1180,(window.innerWidth||1000)-90)), H=330;
  var L=48,R=118,T=14,B=42;
  var maxD=1,maxX=1;
  runs.forEach(function(r){var c=C[r]; if(!c)return;
    c.points.forEach(function(p){maxD=Math.max(maxD,p.dlvl);maxX=Math.max(maxX,p.xl);});});
  maxD=Math.max(12,Math.ceil(maxD)+1); maxX=Math.max(12,Math.ceil(maxX)+1);
  function X(d){return L+(W-L-R)*((d-1)/(maxD-1));}
  function Y(x){return T+(H-T-B)*(1-(x-1)/(maxX-1));}
  var s='<svg class="xy" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'">';
  for(var g=1;g<=maxX;g+=(maxX>14?2:1)){
    s+='<line class="grid" x1="'+L+'" y1="'+Y(g)+'" x2="'+(W-R)+'" y2="'+Y(g)+'"/>'
      +'<text class="tick" x="'+(L-7)+'" y="'+(Y(g)+3)+'" text-anchor="end">'+g+'</text>';
  }
  for(var d=1;d<=maxD;d+=(maxD>16?3:2)){
    s+='<text class="tick" x="'+X(d)+'" y="'+(H-B+15)+'" text-anchor="middle">'+d+'</text>';
  }
  s+='<text class="axlab" x="'+((L+W-R)/2)+'" y="'+(H-6)+'" text-anchor="middle">dungeon level</text>';
  s+='<text class="axlab" transform="translate(12,'+((T+H-B)/2)+') rotate(-90)" text-anchor="middle">experience level</text>';
  // winners' norm
  var nd='';
  for(var d2=1;d2<=maxD;d2++){var n=normXL(d2); if(n==null)continue;
    nd+=(nd?' L':'M')+X(d2)+','+Y(Math.min(n,maxX));}
  s+='<path class="norm" d="'+nd+'"/>';
  var lastN=normXL(Math.min(maxD,20));
  s+='<text class="normlab" x="'+(X(Math.min(maxD,20))+6)+'" y="'+(Y(Math.min(lastN,maxX))+3)+'">winners\u2019 pace</text>';
  runs.forEach(function(r){
    var c=C[r]; if(!c||!c.points.length)return;
    var col=RUNCOLORS[r], dim=(r===cur?'':' dimline');
    var d='';
    c.points.forEach(function(p,i){
      var x=X(Math.min(p.dlvl,maxD)),y=Y(Math.min(p.xl,maxX));
      d+=(i===0?'M':' L')+x+','+y;
    });
    s+='<path class="path'+dim+'" d="'+d+'" stroke="'+col+'"/>';
    var last=c.points[c.points.length-1];
    var ex=X(Math.min(last.dlvl,maxD)), ey=Y(Math.min(last.xl,maxX));
    s+='<circle class="end'+dim+'" cx="'+ex+'" cy="'+ey+'" r="5" fill="'+col+'" stroke="var(--surface)"/>';
    s+='<text class="tick'+dim+'" x="'+(ex+9)+'" y="'+(ey+3)+'" fill="'+col+'">D'+last.dlvl+' XL'+last.xl+'</text>';
  });
  s+='</svg>';
  document.getElementById('xy').innerHTML=s;
  document.getElementById('xylegend').innerHTML=
    runs.map(function(r){var c=C[r]; if(!c)return '';
      var p=c.points[c.points.length-1]; if(!p) return '';
      var def=(function(){var n=normXL(p.dlvl); return n==null?null:n-p.xl;})();
      return '<span style="'+(r===cur?'font-weight:600;color:var(--ink)':'')+'">'
        +'<i class="sw" style="border-color:'+RUNCOLORS[r]+'"></i>'+r
        +' <span style="color:var(--ink-3)">D'+p.dlvl+' XL'+p.xl
        +(def!=null&&def>0?' &middot; '+def+' under pace':'')+'</span></span>';
    }).join('')
    +'<span><i class="sw dash" style="border-color:var(--ink-3)"></i>winners\u2019 pace (433 ascensions)</span>';
}

function draw(){
  [].forEach.call(nav.children,function(b,i){b.setAttribute('aria-current',runs[i]===cur);});
  drawMeta(); drawChart(); drawXY(); drawTree(); drawDetail();
}
[].forEach.call(document.querySelectorAll('.segbar button'),function(b){
  b.addEventListener('click',function(){
    METRIC=b.dataset.m;
    [].forEach.call(document.querySelectorAll('.segbar button'),function(x){
      x.setAttribute('aria-pressed',String(x===b));});
    drawChart();
  });
});
draw();
})();
</script>
"""

payload = json.dumps(DATA, separators=(",", ":")).replace("</", "<\\/")
cpayload = json.dumps(CURVES, separators=(",", ":")).replace("</", "<\\/")
open(OUT, "w").write(HTML.replace("__DATA__", payload).replace("__CURVES__", cpayload))
print(OUT, os.path.getsize(OUT), "bytes")
for k, v in DATA.items():
    print(f"  {k}: {len(v['attempts'])} attempts, balrog {v['balrog_max']} / min {v['balrog_min_max']}")
