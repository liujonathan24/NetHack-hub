#!/usr/bin/env python3
"""Explorer dataset: one record per CLOSED attempt, per run."""
import json,glob,os,re,sys
sys.path.insert(0,'/root/nld/zombie-fix/environments/nethack')
sys.path.insert(0,'/root/nld/zombie-fix/tools/cli_harness_eval')
from nethack_harness.prompt.balrog import balrog_both
import e16_level_balance as LB
HUNGER={0:'Satiated',1:'Normal',2:'Hungry',3:'Weak',4:'Fainting',5:'Fainted',6:'Starved'}
BR={0:'main',2:'Mines',3:'Quest',4:'Soko'}
KILL=re.compile(r'\b(gnome lord|gnome|fire ant|iguana|rothe|white unicorn|unicorn|elf zombie|kobold zombie|zombie|jackal|newt|sewer rat|dwarf lord|dwarf|hill orc|bat|acid blob|homunculus|leprechaun|soldier ant|garter snake|snake|wererat|water moccasin|gnome king|dingo|giant beetle|hill giant|owlbear|quasit|chameleon|wraith|mumak)\b',re.I)

def turn_rows(d,n):
    fs=sorted(glob.glob(d+'/attempts/a%03d/turns*/**/*.ndjson'%n,recursive=True))+\
       sorted(glob.glob(d+'/attempts/a%03d/turns/*.ndjson'%n))
    rows=[]
    for f in dict.fromkeys(fs):
        for l in open(f,errors='replace'):
            l=l.strip()
            if not l: continue
            try: rows.append(json.loads(l))
            except Exception: pass
    return rows

def collect(run):
    d='/root/nld/e16_runs/'+run
    if not os.path.exists(d+'/attempts.jsonl') or not os.path.getsize(d+'/attempts.jsonl'):
        return None
    att=[json.loads(l) for l in open(d+'/attempts.jsonl') if l.strip()]
    sel={s['attempt']:s for s in (json.loads(l) for l in open(d+'/selection.jsonl') if l.strip())} \
        if os.path.exists(d+'/selection.jsonl') else {}
    nodes={}
    for m in glob.glob(d+'/archive/c*/meta.json'):
        j=json.load(open(m)); nodes[str(j['id'])]=j
    owner={}
    for a in att:
        for c in (a.get('new_checkpoints') or []): owner[str(c).lstrip('c')]=a['attempt']
    prov=json.load(open(d+'/provenance.json'))
    hrec=any('hunger' in c for s in sel.values() for c in (s.get('candidates_shown') or []))
    out=[]
    for a in att:
        n=a['attempt']; s=sel.get(n,{}); cid=str(a.get('from_checkpoint') or '')
        src=nodes.get(cid,{}); h0=killer=None; faints=0; acct=''; calls=None; endmsg=''
        end_d=end_x=None
        rows=turn_rows(d,n)
        if rows:
            calls=len(rows)
            h0=(rows[0].get('status') or {}).get('hunger_state')
            faints=sum(1 for r in rows if 'faint' in (r.get('rendered_user_message') or '').lower())
            for r in reversed(rows[-6:]):
                mm=KILL.findall(r.get('rendered_user_message') or '')
                if mm and not killer: killer=mm[0]
            for l in (rows[-1].get('rendered_user_message') or '').split('\n'):
                if 'You died' in l: endmsg=l.strip()[:200]; break
            for r in reversed(rows):
                st=r.get('status') or {}
                if (st.get('experience_level') or 0)>0:
                    end_d=r.get('dlvl'); end_x=st.get('experience_level'); break
        tf=d+'/attempts/a%03d/traces.jsonl'%n
        if os.path.exists(tf) and os.path.getsize(tf):
            try:
                t=json.loads(open(tf).read().strip().splitlines()[-1])
                for it in reversed(t.get('nodes') or []):
                    m=it.get('message') or {}
                    if m.get('role')=='assistant' and isinstance(m.get('content'),str) and m['content'].strip():
                        acct=m['content'].strip()[:1200]; break
            except Exception: pass
        cs=s.get('candidates_shown') or []
        imp=[c for c in cs if c.get('hunger_impaired')]
        ncs=[str(c).lstrip('c') for c in (a.get('new_checkpoints') or [])]
        made=[{'id':i,'dlvl':nodes[i].get('dlvl'),'xl':nodes[i].get('xl')} for i in ncs if i in nodes]
        gain=(a.get('max_dlvl')-(src.get('dlvl') or 0)) if src and a.get('max_dlvl') is not None else None
        bmx=bmn=None
        if a.get('max_dlvl') and a.get('max_xl'):
            x,y=balrog_both(a['max_dlvl'],a['max_xl']); bmx,bmn=round(100*x,2),round(100*y,2)
        norm=defi=None
        if end_d and end_x:
            norm=LB.norm_xl(end_d); defi=LB.deficit(end_d,end_x)
        st=None
        if src:
            st={'dlvl':src.get('dlvl'),'xl':src.get('xl'),'hp':src.get('hp'),'mx':src.get('max_hp'),
                'score':src.get('score'),'br':BR.get(src.get('dungeon_number'),'?')}
            if src.get('dlvl') and src.get('xl'):
                x,y=balrog_both(src['dlvl'],src['xl'])
                st['balrog'],st['balrog_min']=round(100*x,2),round(100*y,2)
                st['deficit']=LB.deficit(src['dlvl'],src['xl'])
        out.append(dict(n=n,parent=owner.get(cid),from_ck=cid,
          start_dlvl=src.get('dlvl'),start_xl=src.get('xl'),start=st,
          hunger0=HUNGER.get(h0),impaired=(h0 in (3,4,5,6)),
          max_dlvl=a.get('max_dlvl'),max_xl=a.get('max_xl'),gain=gain,
          end_dlvl=end_d,end_xl=end_x,winners_norm_xl=norm,deficit=defi,
          balrog=bmx,balrog_min=bmn,
          outcome=a.get('outcome'),censor=a.get('censor_reason') or '',
          killer=killer,endmsg=endmsg,faints=faints,calls=calls,
          wall_m=round((a.get('wall_s') or 0)/60,1),usd=round(a.get('spend_usd') or 0,2),
          directive=s.get('directive') or '',account=acct,
          chose=s.get('chosen_id'),scripted=s.get('scripted_would_pick'),
          agreed=s.get('scripted_agreed'),on_frontier=s.get('chosen_on_frontier'),
          n_cand=s.get('n_candidates_shown'),
          n_impaired=(len(imp) if hrec else None),
          impaired_ids=([c['id'] for c in imp][:10] if hrec else []),
          made=made[:60],n_made=len(made)))
    bs=[a['balrog'] for a in out if a['balrog'] is not None]
    bm=[a['balrog_min'] for a in out if a['balrog_min'] is not None]
    return dict(run=run,arm=prov.get('experiment_arm'),tier=prov.get('tier'),seed=prov.get('game_seed'),
                engine=(prov.get('commits') or {}).get('engine','')[:12],
                max_attempts=(prov.get('stop_conditions') or {}).get('max_attempts'),
                n_ck=len(nodes),hunger_recorded=hrec,
                balrog_max=max(bs) if bs else None,
                balrog_mean=round(sum(bs)/len(bs),2) if bs else None,
                balrog_min_max=max(bm) if bm else None,
                balrog_min_mean=round(sum(bm)/len(bm),2) if bm else None,
                attempts=out)

data={}
for r in sys.argv[1:]:
    try:
        c=collect(r)
        if c: data[r]=c
    except Exception as e:
        print('skip %s: %s'%(r,e),file=sys.stderr)
print(json.dumps(data,default=str,separators=(',',':')))
