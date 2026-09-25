import json,os
L='/root/overleaf/balrog-experiments/submissions/LLM/'
P=json.load(open('/root/nld/zombie-fix/results/model_prices.json'))
P['_gemini-3-pro']={'input_per_million':2.0,'output_per_million':12.0}  # assumed = 3.1 Pro preview list price
SUBS={ # dir: (price key, short label)
 '20241030_naive-Llama-3.2-1B-Instruct':('meta-llama/Llama-3.2-1B-Instruct','Llama-3.2-1B'),
 '20241030_naive-Llama-3.2-3B-Instruct':('meta-llama/Llama-3.2-3B-Instruct','Llama-3.2-3B'),
 '20241209_naive-mistral-nemo-instruct':('mistralai/mistral-nemo','Mistral-Nemo'),
 '20241209_naive_Llama-3.3-70B-Instruct':('meta-llama/llama-3.3-70b-instruct','Llama-3.3-70B'),
 '20250425_naive_Gemini-2.5-Pro-Exp-03-25':('google/gemini-2.5-pro','Gem-2.5-Pro'),
 '20250719-naive_gemini-2.5-flash':('google/gemini-2.5-flash','Gem-2.5-Flash'),
 '20250808_naive_gpt-5-minimal':('openai/gpt-5','GPT-5 (minimal)'),
 '20260203_naive_gemini-3-pro':('_gemini-3-pro','Gem-3-Pro'),
 '20260213_naive_gemini-3-flash':('google/gemini-3-flash-preview','Gem-3-Flash'),
 '20260221_naive_gemini-3.1-pro':('google/gemini-3.1-pro-preview','Gem-3.1-Pro'),
 '20260223_naive_claude-haiku-4.5':('anthropic/claude-haiku-4.5','Haiku 4.5'),
 '20260224_naive_claude-opus-4.5':('anthropic/claude-opus-4.5','Opus 4.5'),
 '20260224_naive_claude-opus-4.5-thinking':('anthropic/claude-opus-4.5','Opus 4.5-T'),
 '20260225_naive_gemini-3.1-pro-thinking':('google/gemini-3.1-pro-preview','Gem-3.1-Pro-T'),
}
STD={'minihack':40,'textworld':30,'crafter':10}
ours=json.load(open('aggregate.json'))['heldout3']
pts=[]
for d,(pk,lab) in SUBS.items():
    pr=P[pk];prog=[];cost=0;note=[]
    for e in STD:
        s=json.load(open(f'{L}{d}/{e}/{e}_summary.json'))
        c=(s['input_tokens']*pr['input_per_million']+s['output_tokens']*pr['output_per_million'])/1e6
        n=s['episodes_played']
        if n!=STD[e]: note.append(f'{e} n={n}')
        cost+=c*STD[e]/n
        prog.append(s['progression_percentage']/100)
    pts.append(dict(label=lab,kind='lb',prog=sum(prog)/3,cost=cost,note=note,suites=[round(x,4) for x in prog]))
for p in ours:
    if p['kind']!='lb': pts.append(p)
for p in sorted(pts,key=lambda q:-q['prog']): print(f"{p['label']:18s} {p['prog']:.3f} ${p['cost']:8.2f} {p.get('suites','')} {p.get('note','')}")
json.dump(pts,open('aggregate_old_heldout.json','w'),indent=1)
