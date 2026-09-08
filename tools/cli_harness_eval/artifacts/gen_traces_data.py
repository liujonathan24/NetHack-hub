#!/usr/bin/env python3
"""Traces dataset: every orchestrator reply verbatim + each player's opening."""
import json,glob,os,re,sys
def first_player_msg(d,n):
    tf=d+'/attempts/a%03d/traces.jsonl'%n
    if not os.path.exists(tf) or not os.path.getsize(tf): return None
    try: t=json.loads(open(tf).read().strip().splitlines()[-1])
    except Exception: return None
    for it in (t.get('nodes') or []):
        m=it.get('message') or {}
        if m.get('role')=='assistant' and isinstance(m.get('content'),str) and m['content'].strip():
            return m['content']
    return None
def served_bits(p):
    out={}
    m=re.search(r'LAST ATTEMPT \(([^)]*)\):([^\n]*)',p or '')
    if m: out['last_attempt']=m.group(0).strip()
    m=re.search(r'STANDING OF THE RUN \(computed, not opinion\):\s*\n\s*(.+?)(?:\n\n|\nCHECKPOINT)',p or '',re.S)
    if m: out['standing']=m.group(1).strip()
    m=re.search(r"The player's own account[^\n]*:\s*\n(.*?)(?=\nCHECKPOINT ARCHIVE)",p or '',re.S)
    if m: out['account_served']=m.group(1).strip()
    return out
data={}
for run in sys.argv[1:]:
    d='/root/nld/e16_runs/'+run
    if not os.path.exists(d+'/orchestrator_rounds.jsonl'): continue
    try: prov=json.load(open(d+'/provenance.json'))
    except Exception: prov={}
    rounds=[dict(round=r.get('round'),kind=r.get('kind'),wall_s=r.get('wall_s'),
                 reply=r.get('reply') or '',served=served_bits(r.get('prompt') or ''))
            for r in (json.loads(l) for l in open(d+'/orchestrator_rounds.jsonl') if l.strip())]
    att={}
    if os.path.exists(d+'/attempts.jsonl'):
        for a in (json.loads(l) for l in open(d+'/attempts.jsonl') if l.strip()):
            att[a['attempt']]=dict(n=a['attempt'],from_ck=a.get('from_checkpoint'),
              dlvl=a.get('max_dlvl'),xl=a.get('max_xl'),outcome=a.get('outcome'),
              censor=a.get('censor_reason') or '',directive=a.get('directive') or '',
              first_msg=first_player_msg(d,a['attempt']))
    data[run]=dict(run=run,arm=prov.get('experiment_arm'),tier=prov.get('tier'),
                   rounds=rounds,attempts=[att[k] for k in sorted(att)])
print(json.dumps(data,default=str,separators=(',',':')))
