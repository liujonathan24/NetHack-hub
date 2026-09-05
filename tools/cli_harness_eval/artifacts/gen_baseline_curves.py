#!/usr/bin/env python3
"""Progress curves for BASELINE rollouts, in the same shape as e16_progress_curve.

    python3 gen_baseline_curves.py SEED RUN_DIR [RUN_DIR ...]

Why a separate script. A baseline cell is one directory holding FIVE seeds'
rollouts side by side (`turns/<seed>_<pid>_<ts>.ndjson`), with no attempts/
subtree -- there is one life per seed and no orchestrator. e16_progress_curve
walks `attempts/a*/turns`, so it finds nothing here. Same output shape, so the
explorer can plot both on one axis.

These are REFERENCE lines, not arms: a single uninterrupted life each, no
archive, no resume, no directive.
"""
import json,glob,os,sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..','..','..','environments','nethack'))
from nethack_harness.prompt.balrog import balrog_both

def curve(run_dir, seed, every=10):
    fs=sorted(glob.glob(os.path.join(run_dir,'turns','%s_*.ndjson'%seed)))
    pts=[]; t=0; hi_d=hi_x=0; bmax=bmin=0.0
    for f in fs:
        for line in open(f,errors='replace'):
            line=line.strip()
            if not line: continue
            try: rec=json.loads(line)
            except json.JSONDecodeError: continue
            t+=1
            st=rec.get('status') or {}
            d=rec.get('max_dlvl_reached') or rec.get('dlvl') or 0
            x=st.get('experience_level') or 0
            if not (d and x): continue
            hi_d,hi_x=max(hi_d,int(d)),max(hi_x,int(x))
            mx,mn=balrog_both(hi_d,hi_x); mx,mn=100.0*mx,100.0*mn
            improved=(mx>bmax+1e-9) or (mn>bmin+1e-9)
            bmax,bmin=max(bmax,mx),max(bmin,mn)
            if improved or t%every==0:
                pts.append({'t':t,'attempt':1,'best_max':round(bmax,2),
                            'best_min':round(bmin,2),'dlvl':hi_d,'xl':hi_x})
    if pts and pts[-1]['t']!=t:
        pts.append({'t':t,'attempt':1,'best_max':round(bmax,2),
                    'best_min':round(bmin,2),'dlvl':hi_d,'xl':hi_x})
    return {'run':os.path.basename(run_dir),'arm':'baseline','tier':'base',
            'total_turns':t,'attempts':1,
            'final_best_max':round(bmax,2),'final_best_min':round(bmin,2),
            'after_attempt_1_max':round(bmax,2),'after_attempt_1_min':round(bmin,2),
            'improvement_after_attempt_1_max':0.0,'improvement_after_attempt_1_min':0.0,
            'points':pts,'attempt_marks':[],'is_baseline':True}

if __name__=='__main__':
    seed=sys.argv[1]; out={}
    for d in sys.argv[2:]:
        rep=os.path.basename(d).replace('__prime_agent','')
        c=curve(d,seed)
        if c['points']: out['base %s · seed %s'%(rep.replace('base_',''),seed)]=c
    print(json.dumps(out,separators=(',',':')))
