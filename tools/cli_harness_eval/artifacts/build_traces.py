#!/usr/bin/env python3
"""E16 reasoning traces: every orchestrator reply in full, every player opening."""
import json, os

SP = os.environ.get("E16_ART_DIR") or os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, "e16_traces.html")
DATA = json.load(open(f"{SP}/traces_data.json"))

HTML = r"""<title>E16 Reasoning Traces</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#f4f5f3; --surface:#fff; --sunken:#eef0ec;
  --ink:#181b17; --ink-2:#555c53; --ink-3:#848b82;
  --rule:#dcdfd8; --rule-2:#e9ebe6;
  --orch:#a8531d;       /* orchestrator voice */
  --play:#1f6b70;       /* player voice */
  --serve:#6b6f68;      /* what was served in */
  --alert:#94302c;
  --shadow:0 1px 2px rgba(24,27,23,.05),0 3px 12px rgba(24,27,23,.04);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#101310; --surface:#171a16; --sunken:#1c201b;
  --ink:#e6e9e3; --ink-2:#9ba39a; --ink-3:#6f776e;
  --rule:#272c26; --rule-2:#20241f;
  --orch:#e08348; --play:#63bcc2; --serve:#8b938a; --alert:#e5776f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 3px 14px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --ground:#101310; --surface:#171a16; --sunken:#1c201b;
  --ink:#e6e9e3; --ink-2:#9ba39a; --ink-3:#6f776e;
  --rule:#272c26; --rule-2:#20241f;
  --orch:#e08348; --play:#63bcc2; --serve:#8b938a; --alert:#e5776f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 3px 14px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif;font-size:16px;line-height:1.6}
.wrap{max-width:60rem;margin:0 auto;padding:0 20px 80px}
header{padding:44px 0 22px;border-bottom:1px solid var(--rule)}
.eyebrow{font-family:Archivo,sans-serif;font-size:.68rem;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:var(--orch);margin:0 0 .8em}
h1{font-family:Archivo,sans-serif;font-size:clamp(1.7rem,3.6vw,2.4rem);font-weight:700;
  letter-spacing:-.02em;margin:0 0 .4em;line-height:1.12}
.sub{color:var(--ink-2);max-width:66ch;margin:0}
code{font-family:"IBM Plex Mono",monospace;font-size:.86em;background:var(--sunken);
  border:1px solid var(--rule-2);border-radius:4px;padding:.05em .32em}
nav{display:flex;gap:2px;overflow-x:auto;border-bottom:1px solid var(--rule);
  margin:0 0 6px;position:sticky;top:0;background:var(--ground);z-index:10}
nav button{font-family:Archivo,sans-serif;font-size:.8rem;font-weight:600;background:none;border:0;
  border-bottom:2px solid transparent;color:var(--ink-3);padding:13px 14px 10px;cursor:pointer;white-space:nowrap}
nav button:hover{color:var(--ink)}
nav button[aria-current="true"]{color:var(--ink);border-bottom-color:var(--orch)}
nav button:focus-visible{outline:2px solid var(--orch);outline-offset:-3px}
.runbar{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-2);
  padding:10px 0 18px;border-bottom:1px solid var(--rule);margin-bottom:22px}
.key{display:flex;gap:18px;flex-wrap:wrap;font-family:Archivo,sans-serif;font-size:.68rem;
  color:var(--ink-3);margin:14px 0 0}
.key i{font-style:normal;display:inline-block;width:3px;height:11px;vertical-align:-1px;margin-right:6px}
/* blocks */
.blk{margin:0 0 20px;border-radius:9px;overflow:hidden;box-shadow:var(--shadow);
  border:1px solid var(--rule);background:var(--surface)}
.blk > .hd{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
  padding:9px 15px;border-bottom:1px solid var(--rule-2);background:var(--sunken)}
.who{font-family:Archivo,sans-serif;font-size:.66rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.who.o{color:var(--orch)} .who.p{color:var(--play)} .who.s{color:var(--serve)}
.meta{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--ink-3);
  font-variant-numeric:tabular-nums;margin-left:auto}
.raw{font-family:"IBM Plex Mono",monospace;font-size:.775rem;line-height:1.62;
  white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;margin:0;padding:14px 16px}
.blk.orch{border-left:3px solid var(--orch)}
.blk.play{border-left:3px solid var(--play)}
.served{border-left:3px solid var(--serve);background:var(--sunken)}
.served .raw{color:var(--ink-2);font-size:.75rem;padding:11px 16px}
.served .lbl{font-family:Archivo,sans-serif;font-size:.6rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--serve);padding:9px 16px 0;display:block}
.standing{color:var(--alert)}
.empty{color:var(--ink-3);font-style:italic;font-family:"Source Serif 4",serif;padding:12px 16px;margin:0}
details.srv{border-bottom:1px solid var(--rule-2)}
details.srv summary{cursor:pointer;font-family:Archivo,sans-serif;font-size:.64rem;font-weight:600;
  letter-spacing:.09em;text-transform:uppercase;color:var(--serve);padding:9px 15px;list-style:none}
details.srv summary::-webkit-details-marker{display:none}
details.srv summary::before{content:"▸ ";}
details.srv[open] summary::before{content:"▾ ";}
.note{background:var(--sunken);border:1px solid var(--rule);border-left:3px solid var(--alert);
  border-radius:0 8px 8px 0;padding:13px 16px;margin:0 0 22px}
.note p{margin:0 0 .6em;font-size:.94rem} .note p:last-child{margin:0}
.note b{color:var(--alert)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">E16 · Go-Explore · NetHack seed 1</p>
  <h1>Reasoning Traces</h1>
  <p class="sub">Every orchestrator reply, complete and unedited. Every player's opening message, complete and unedited. Nothing here is summarised or truncated — the only processing is interleaving the two voices in the order they happened, and lifting out the three lines the orchestrator was served just before it answered.</p>
  <div class="key">
    <span><i style="background:var(--serve)"></i>served IN to the orchestrator</span>
    <span><i style="background:var(--orch)"></i>orchestrator, full reply</span>
    <span><i style="background:var(--play)"></i>player, first words of the attempt</span>
  </div>
</header>
<nav id="nav" aria-label="Runs"></nav>
<div class="runbar" id="runbar"></div>
<div id="note"></div>
<div id="feed"></div>
</div>

<script id="tdata" type="application/json">__DATA__</script>
<script>
(function(){
var D=JSON.parse(document.getElementById('tdata').textContent);
var runs=Object.keys(D), cur=runs[0];
var nav=document.getElementById('nav');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}

runs.forEach(function(r){
  var b=document.createElement('button');
  b.textContent=r; b.onclick=function(){cur=r;draw();}; nav.appendChild(b);
});

var NOTES={
  treesmoke8:'<p>Rounds 1–8 ran with <b>no</b> experience signal. From round 9 the orchestrator is served a computed line telling it exactly how far under the winners’ pace the character is, and by how much that raises its hazard.</p><p>Read rounds 9 onward against the replies: the signal arrives, is legible, and the directives that follow say <b>“your only goal is to reach new depths”</b>. Depth went D7→D18; experience has sat at XL 6 since attempt 2.</p>',
  treesmoke10:'<p>The clean arm: the experience signal is served from round 1, into an empty archive, so a deficit can be prevented rather than needing to be undone.</p>',
  treesmoke7:'<p>Ran blind on two channels at once — no nutrition in the ledger, and the player’s account never delivered (a role lookup one level too shallow returned <code>""</code> every round). Watch the orchestrator build a confident theory about a level-draining monster that does not exist.</p>',
  treesmoke6:'<p>Rollback published, which also arms forced revive. Attempts are not single-life here.</p>',
  'treesmoke10.aborted':'<p>Stopped and kept for the record. It lost 2 of 4 attempts to infrastructure rather than gameplay: one <code>watchdog_stall</code> that discarded 405 played turns and $29.21, and one real archive defect — restore fidelity on c28, score meta=169 vs engine=219, a <b>model-authored</b> save taken at a level boundary before the 50-point depth bonus settled (<a href="https://github.com/liujonathan24/NetHack-hub/issues/46">issue #46</a>).</p><p>It is also the run whose orchestrator read slow attempts as failures and answered by demanding faster descent — the reason the prompt now says slow is not failing.</p>',
  treesmoke11:'<p>Replaces treesmoke10. Winners’-norm signal from round 1 into an empty archive, plus the new rule that an attempt which comes back stronger has done something even when it gained no depth.</p>'
};

function block(cls,who,whoCls,meta,text,extra){
  return '<div class="blk '+cls+'">'
    +'<div class="hd"><span class="who '+whoCls+'">'+who+'</span>'
    +(meta?'<span class="meta">'+meta+'</span>':'')+'</div>'
    +(extra||'')
    +(text&&text.trim()?'<pre class="raw">'+esc(text)+'</pre>'
                       :'<p class="empty">(empty — nothing was produced here)</p>')
    +'</div>';
}

function servedPanel(s){
  if(!s) return '';
  var bits='';
  if(s.last_attempt) bits+='<span class="lbl">outcome it was given</span><pre class="raw">'+esc(s.last_attempt)+'</pre>';
  if(s.standing)     bits+='<span class="lbl">computed standing</span><pre class="raw standing">'+esc(s.standing)+'</pre>';
  if(s.account_served!==undefined)
    bits+='<span class="lbl">player’s account, as delivered</span>'
        +(s.account_served.trim()?'<pre class="raw">'+esc(s.account_served)+'</pre>'
                                 :'<p class="empty">(blank — the delivery bug)</p>');
  if(!bits) return '';
  return '<details class="srv"'+(s.standing?' open':'')+'><summary>what was served in</summary>'+bits+'</details>';
}

function draw(){
  [].forEach.call(nav.children,function(b,i){b.setAttribute('aria-current',runs[i]===cur);});
  var R=D[cur];
  document.getElementById('runbar').textContent=
    R.arm+' · '+R.tier+' · '+R.rounds.length+' orchestrator rounds · '
    +R.attempts.length+' attempts';
  document.getElementById('note').innerHTML=NOTES[cur]?'<div class="note">'+NOTES[cur]+'</div>':'';

  // interleave: round k, then the attempt that round launched
  var byN={}; R.attempts.forEach(function(a){byN[a.n]=a;});
  var html='', used={};
  R.rounds.forEach(function(rd){
    var isProbe=(rd.kind||'').indexOf('probe')===0;
    html+=block('orch','Orchestrator · '+(rd.kind||('round '+rd.round)),'o',
      (rd.wall_s!=null?rd.wall_s.toFixed(1)+'s':''), rd.reply, servedPanel(rd.served));
    if(isProbe) return;
    // round N launches attempt N (opening round launches attempt 1)
    var m=/round(\d+)/.exec(rd.kind||''); var n=m?(+m[1]+1):1;
    var a=byN[n];
    if(a&&!used[n]){
      used[n]=1;
      var meta='D'+a.dlvl+' XL'+a.xl+' · '+a.outcome+(a.censor?':'+a.censor:'');
      html+=block('play','Player · attempt '+a.n+' opening','p',meta,a.first_msg,'');
    }
  });
  // any attempts not paired to a round
  R.attempts.forEach(function(a){
    if(used[a.n]) return;
    html+=block('play','Player · attempt '+a.n+' opening','p',
      'D'+a.dlvl+' XL'+a.xl+' · '+a.outcome+(a.censor?':'+a.censor:''),a.first_msg,'');
  });
  document.getElementById('feed').innerHTML=html;
}
draw();
})();
</script>
"""

payload = json.dumps(DATA, separators=(",", ":")).replace("</", "<\\/")
open(OUT, "w").write(HTML.replace("__DATA__", payload))
print(OUT, os.path.getsize(OUT), "bytes")
for k, v in DATA.items():
    print(f"  {k}: {len(v['rounds'])} rounds, {len(v['attempts'])} attempts")
