import json,glob,sys,re
sys.path.insert(0,'/root/nld/zombie-fix/environments/nethack')
from nethack_harness.prompt.balrog import balrog_progress, balrog_progress_min
FI,CA,OU=1.54,0.154,4.84
def price(f,c,o): return (f*FI+c*CA+o*OU)/1e6
# ours base: CC lives, dedupe by trace id
seen={};
for p in sorted(glob.glob('/root/nld/cc_runs/base_r*__claude_code/traces.jsonl')):
  for ln in open(p):
    try:r=json.loads(ln)
    except:continue
    tid=r.get('id') or r.get('trace_id') or r.get('rollout_id')
    seen[tid]=r
f=c=o=0;mx=[];mn=[]
for r in seen.values():
  for x in r.get('calls',[]):
    u=x.get('usage') or {};f+=u.get('prompt_tokens',0) or 0;c+=u.get('cached_input_tokens',0) or 0;o+=u.get('completion_tokens',0) or 0
  m=r.get('metrics',{}); mx.append(m.get('balrog_pct'));mn.append(m.get('balrog_min_pct'))
print('base lives',len(seen),'cost',round(price(f,c,o),2),'list',round((f+c)*FI/1e6+o*OU/1e6,2),'BALmax',sum(mx)/len(mx),'BALmin',sum(mn)/len(mn))
base=dict(cost=price(f,c,o),mx=sum(mx)/len(mx),mn=sum(mn)/len(mn),n=len(seen))
# PAE fog
tot=0;pm=[];pn=[];per=[]
for s in range(5):
  d=f'/root/nld/e16_runs/e16_fog_s{s}_r1'; f=c=o=0
  for p in glob.glob(d+'/attempts/*/traces.jsonl'):
    for ln in open(p):
      try:r=json.loads(ln)
      except:continue
      for x in r.get('calls',[]):
        u=x.get('usage') or {};f+=u.get('prompt_tokens',0) or 0;c+=u.get('cached_input_tokens',0) or 0;o+=u.get('completion_tokens',0) or 0
  S=json.load(open(d+'/summary.json'));orch=S['orchestrator']['cost_usd_reported']
  cs=price(f,c,o)+orch;tot+=cs;pm.append(100*S['best_state']['balrog']);pn.append(100*S['best_state']['balrog_min']);per.append(round(cs,2))
print('PAE fog per seed',per,'total',round(tot,2),'BALmax',sum(pm)/5,'BALmin',sum(pn)/5)
pae=dict(cost=tot,mx=sum(pm)/5,mn=sum(pn)/5)
# leaderboard
d3=json.load(open('suites.json'));k=[x for x in d3 if 'NetHack' in x][0]
lb={}
for pt in d3[k]['pts']:
  if pt['kind']!='lb':continue
  alias={'gemini-3.1-pro-preview-thinking':'20260225_naive_gemini-3.1-pro-thinking','gemini-3.1-pro-preview':'20260221_naive_gemini-3.1-pro','gemini-3-flash-preview':'20260213_naive_gemini-3-flash'}
  sub=['/root/overleaf/balrog-experiments/submissions/LLM/'+alias[pt['label']]] if pt['label'] in alias else [s for s in glob.glob('/root/overleaf/balrog-experiments/submissions/LLM/*') if re.search(re.escape(pt['label'])+r'(-max|-high)?$',s) or s.endswith('_'+pt['label'])]
  eps=[]
  for s in sub:
    for j in glob.glob(s+'/nle/NetHackChallenge-v0/*.json'):
      e=json.load(open(j));dl=max([int(x.split(':')[1]) for x in e['dlvl_list']] or [1]);xl=max([int(x.split(':')[1]) for x in e['xplvl_list']] or [1])
      eps.append((100*balrog_progress(dl,xl),100*balrog_progress_min(dl,xl),e.get('progression')))
  if eps: lb[pt['label']]=dict(sub=[x.split('/')[-1] for x in sub],n=len(eps),mx=sum(e[0] for e in eps)/len(eps),mn=sum(e[1] for e in eps)/len(eps),pub=100*sum(e[2] for e in eps)/len(eps),cost=pt['cost'])
  print(pt['label'],lb.get(pt['label']))
json.dump(dict(base=base,pae=pae,lb=lb),open('nh_minmax.json','w'),indent=1)
