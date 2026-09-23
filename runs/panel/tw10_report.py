#!/usr/bin/env python3
"""TextWorld 10-seed readout + archived-leaderboard comparison."""
import json, glob, os, math, statistics as st
ROOT="/root/nld/gen_runs/panel"
LB="/root/overleaf/balrog-experiments/submissions/LLM/*/textworld/textworld_summary.json"
TASKS=["treasure_hunter","the_cooking_game","coin_collector"]
def loadl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
rows={t:[] for t in TASKS}
for t in TASKS:
    for s in range(10):
        d=f"{ROOT}/tw_{t}_s{s}"
        sp=f"{d}/summary.json"
        if not os.path.exists(sp): rows[t].append(None); continue
        S=json.load(open(sp))
        rf=loadl(f"{d}/restore_fidelity.jsonl"); rp=loadl(f"{d}/resume_prompt_check.jsonl")
        rows[t].append(dict(seed=s,a1=S["attempt1_progression"],best=S["pae_best_progression"],
            d=S["pae_best_progression"]-S["attempt1_progression"],att=S["attempts"],
            stop=S["stop_reason"],rx=S["client_retry_exhausted"],inv=S["invalid_actions"],
            billed=S["tokens"]["billed_by_provider_usd"],
            rf=(sum(1 for r in rf if r.get("identical")),len(rf)),
            rp=(sum(1 for r in rp if r.get("history_matches_checkpoint")),len(rp)),
            cap=S["committed_steps_max"]>=(S["step_cap"] or 10**9)))
def sign(ds):
    p=sum(1 for x in ds if x>0); n=sum(1 for x in ds if x<0); m=p+n
    return p,n,m,(sum(math.comb(m,k) for k in range(p,m+1))/2**m if m else 1.0)
def tsf(t,dof):
    x=dof/(dof+t*t)
    from math import lgamma,exp,log
    def bcf(a,b,x):
        qab,qap,qam=a+b,a+1.0,a-1.0; c,d=1.0,1.0-qab*x/qap
        d=1.0/(d if abs(d)>1e-300 else 1e-300); h=d
        for m in range(1,300):
            m2=2*m
            aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1.0+aa*d; c=1.0+aa/c
            d=1.0/(d if abs(d)>1e-300 else 1e-300); h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1.0+aa*d; c=1.0+aa/c
            d=1.0/(d if abs(d)>1e-300 else 1e-300); de=d*c; h*=de
            if abs(de-1.0)<3e-16: break
        return h
    def binc(a,b,x):
        if x<=0: return 0.0
        if x>=1: return 1.0
        lb=lgamma(a)+lgamma(b)-lgamma(a+b)
        if x<(a+1)/(a+b+2): return exp(log(x)*a+log(1-x)*b-lb)/a*bcf(a,b,x)
        return 1.0-exp(log(1-x)*b+log(x)*a-lb)/b*bcf(b,a,1-x)
    return 0.5*binc(dof/2.0,0.5,x) if t>0 else 1.0-0.5*binc(dof/2.0,0.5,dof/(dof+t*t))
lb={}
for p in sorted(glob.glob(LB)):
    name=p.split("/LLM/")[1].split("/")[0]
    J=json.load(open(p)); lb[name]={k:v["progression_percentage"] for k,v in J["tasks"].items()}
grand=0.0
for t in TASKS:
    rs=[r for r in rows[t] if r]
    miss=[i for i,r in enumerate(rows[t]) if r is None]
    print(f"\n===== textworld/{t}  n={len(rs)} {'MISSING '+str(miss) if miss else ''} =====")
    print(f"{'seed':>4} {'att1':>7} {'best':>7} {'delta':>8} {'att':>4} {'stop':>19} {'retryX':>7} {'billed':>8}")
    for r in rs:
        print(f"{r['seed']:>4} {r['a1']:>7.4f} {r['best']:>7.4f} {r['d']:>+8.4f} {r['att']:>4} {r['stop']:>19} {r['rx']:>7} {r['billed']:>8.4f}")
    ds=[r['d'] for r in rs]; a1=[r['a1'] for r in rs]; bs=[r['best'] for r in rs]
    sd=st.stdev(ds) if len(ds)>1 else 0.0; sem=sd/len(ds)**0.5 if sd else 0.0
    print(f"  BASE attempt-1 mean = {st.mean(a1):.4f} ({100*st.mean(a1):.1f}%)   PAE best mean = {st.mean(bs):.4f}")
    print(f"  PAIRED DELTA mean = {st.mean(ds):+.4f}  SD = {sd:.4f}  SEM = {sem:.4f}")
    p,n,m,pv=sign(ds)
    if t=="the_cooking_game":
        tv=st.mean(ds)/sem if sem else float('inf')
        pt=tsf(tv,len(ds)-1) if sem else 0.0
        print(f"  graded /17 -> PAIRED t-TEST one-sided: t({len(ds)-1})={tv:.3f} p={pt:.5f}  "
              f"{'separated from 0' if pt<0.05 else '*** NOT separated from 0 ***'}")
        print(f"  (sign test: {p}+/{n}- of {m} discordant, p={pv:.5f})")
    else:
        print(f"  binary -> SIGN TEST exact one-sided: {p}+/{n}- over {m} discordant of {len(ds)} pairs, "
              f"p={pv:.5f}  {'separated from 0' if pv<0.05 else '*** NOT separated from 0 ***'}")
    if sd>0:
        lo,hi=st.mean(ds)-1.96*sem, st.mean(ds)+1.96*sem
        print(f"  95% CI mean delta = [{lo:+.4f}, {hi:+.4f}]  {'excludes 0' if lo>0 else 'INCLUDES 0'}")
    print(f"  restore fidelity {sum(r['rf'][0] for r in rs)}/{sum(r['rf'][1] for r in rs)}   "
          f"resumed-prompt parity {sum(r['rp'][0] for r in rs)}/{sum(r['rp'][1] for r in rs)}")
    print(f"  client_retry_exhausted total {sum(r['rx'] for r in rs)}   invalid_actions {sum(r['inv'] for r in rs)}   "
          f"runs at step cap {sum(1 for r in rs if r['cap'])}")
    sub=sum(r['billed'] for r in rs); grand+=sub
    print(f"  billed this task ${sub:.4f}  (seeds 0-4 ${sum(r['billed'] for r in rs if r['seed']<5):.4f} / "
          f"seeds 5-9 ${sum(r['billed'] for r in rs if r['seed']>=5):.4f})")
    print(f"  archived leaderboard (10 episodes each): " +
          ", ".join(f"{k.split('_',1)[1]} {v[t]:.1f}%" for k,v in lb.items()))
print(f"\nTOTAL BILLED (all 30 TextWorld runs) = ${grand:.4f}")
print(f"seeds 5-9 only = ${sum(r['billed'] for t in TASKS for r in rows[t] if r and r['seed']>=5):.4f}")
