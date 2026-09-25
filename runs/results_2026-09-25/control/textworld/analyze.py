"""Paired comparison: PAE best-over-N vs matched fresh best-of-N control (TextWorld).

Exact tests, implemented directly (no scipy dependency):
  * sign test  - one-sided, H1: PAE > control, on discordant pairs only
  * Wilcoxon signed-rank - one-sided exact, on non-zero differences
"""
import json, os, glob, math, statistics as st
from itertools import product

PANEL = "/root/nld/gen_runs/panel"
CTL = "/root/nld/gen_runs/control_bestofn"
TASKS = ["treasure_hunter", "the_cooking_game", "coin_collector"]
SEEDS = range(10)


def sign_test(diffs):
    """One-sided exact sign test, H1: median(diff) > 0."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg                                  # discordant pairs
    if n == 0:
        return dict(pos=0, neg=0, discordant=0, p=1.0, p_min=1.0)
    p = sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n
    return dict(pos=pos, neg=neg, discordant=n, p=p, p_min=1 / 2 ** n)


def wilcoxon(diffs):
    """One-sided exact Wilcoxon signed-rank, H1: median(diff) > 0."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return dict(n_nonzero=0, W_plus=0.0, p=1.0, p_min=1.0)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                                    # midranks for ties
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if nz[i] > 0)
    cnt = 0
    for signs in product([0, 1], repeat=n):         # exact null (n<=10 here)
        if sum(ranks[i] for i in range(n) if signs[i]) >= wp:
            cnt += 1
    return dict(n_nonzero=n, W_plus=wp, p=cnt / 2 ** n, p_min=1 / 2 ** n)


report = {"tasks": {}, "totals": {}}
tot = dict(pae_billed=0.0, ctl_billed=0.0, pae_attempts=0, ctl_attempts=0,
           pae_ledger=0.0, ctl_ledger=0.0, pae_llm=0, ctl_llm=0)
for task in TASKS:
    rows = []
    for s in SEEDS:
        P = json.load(open(f"{PANEL}/tw_{task}_s{s}/summary.json"))
        C = json.load(open(f"{CTL}/ctl_{task}_s{s}/control_summary.json"))
        pb = float(P["pae_best_progression"])
        cb = float(C["control_best_progression"])
        pbill = float(P["tokens"].get("billed_by_provider_usd", 0.0))
        # budget-matched variant: the control's best over only the first
        # k = (number of attempts PAE actually used) episodes. Removes the
        # confound that PAE's plateau_4 rule stops it before 10 attempts on
        # some seeds while the control is allowed the full 10.
        per = [float(x) for x in C["per_episode_progression"]]
        k = int(P["attempts"])
        cb_m = max(per[:k], default=0.0)
        rows.append(dict(
            seed=s, pae_best=pb, control_best=cb, diff=pb - cb,
            control_best_at_pae_budget=cb_m, diff_matched=pb - cb_m,
            control_episodes_at_pae_budget=min(k, len(per)),
            pae_attempt1=float(P["attempt1_progression"]),
            control_episode1=float(C["episode1_progression"]),
            pae_attempts=P["attempts"], control_episodes=C["episodes_used"],
            pae_stop=P["stop_reason"], control_stop=C["stop_reason"],
            pae_billed=pbill, control_billed=float(C["billed_by_provider_usd"]),
            pae_ledger=float(P["tokens"]["total"]["cost_usd"]), control_ledger=float(C["ledger_cost_usd"]),
            pae_llm_steps=P["llm_steps"], control_llm_steps=C["llm_steps"],
        ))
    d = [r["diff"] for r in rows]
    dm = [r["diff_matched"] for r in rows]
    report["tasks"][task] = dict(
        rows=rows,
        pae_mean=st.mean(r["pae_best"] for r in rows),
        control_mean=st.mean(r["control_best"] for r in rows),
        diff_mean=st.mean(d), diff_sd=st.stdev(d),
        diff_se=st.stdev(d) / math.sqrt(len(d)),
        sign_test=sign_test(d), wilcoxon=wilcoxon(d),
        matched=dict(control_mean=st.mean(r["control_best_at_pae_budget"] for r in rows),
                     diff_mean=st.mean(dm), diff_sd=st.stdev(dm),
                     sign_test=sign_test(dm), wilcoxon=wilcoxon(dm)),
        first_episode=dict(pae_attempt1_mean=st.mean(r["pae_attempt1"] for r in rows),
                           control_episode1_mean=st.mean(r["control_episode1"] for r in rows),
                           pae_attempt1_solved=sum(1 for r in rows if r["pae_attempt1"] >= 1.0),
                           control_episode1_solved=sum(1 for r in rows if r["control_episode1"] >= 1.0)),
        pae_attempts_total=sum(r["pae_attempts"] for r in rows),
        control_episodes_total=sum(r["control_episodes"] for r in rows),
        pae_billed_usd=round(sum(r["pae_billed"] for r in rows), 4),
        control_billed_usd=round(sum(r["control_billed"] for r in rows), 4),
        pae_llm_steps=sum(r["pae_llm_steps"] for r in rows),
        control_llm_steps=sum(r["control_llm_steps"] for r in rows),
    )
    t = report["tasks"][task]
    tot["pae_billed"] += t["pae_billed_usd"]; tot["ctl_billed"] += t["control_billed_usd"]
    tot["pae_attempts"] += t["pae_attempts_total"]; tot["ctl_attempts"] += t["control_episodes_total"]
    tot["pae_llm"] += t["pae_llm_steps"]; tot["ctl_llm"] += t["control_llm_steps"]
report["totals"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tot.items()}

# pooled across all 30 seeds
alld = [r["diff"] for t in TASKS for r in report["tasks"][t]["rows"]]
alldm = [r["diff_matched"] for t in TASKS for r in report["tasks"][t]["rows"]]
report["pooled"] = dict(n=len(alld), diff_mean=st.mean(alld), diff_sd=st.stdev(alld),
                        sign_test=sign_test(alld), wilcoxon=wilcoxon(alld),
                        matched=dict(diff_mean=st.mean(alldm), diff_sd=st.stdev(alldm),
                                     sign_test=sign_test(alldm), wilcoxon=wilcoxon(alldm)))
json.dump(report, open(f"{CTL}/comparison.json", "w"), indent=2)

for task in TASKS:
    t = report["tasks"][task]
    print(f"\n=== {task} ===")
    print(f"{'seed':>4} {'PAE best':>9} {'ctl best':>9} {'diff':>7} | {'PAE att':>7} {'ctl eps':>7} "
          f"| {'PAE $':>7} {'ctl $':>7} | PAE stop / ctl stop")
    for r in t["rows"]:
        print(f"{r['seed']:>4} {r['pae_best']:>9.4f} {r['control_best']:>9.4f} {r['diff']:>7.4f} "
              f"| {r['pae_attempts']:>7} {r['control_episodes']:>7} "
              f"| {r['pae_billed']:>7.4f} {r['control_billed']:>7.4f} | {r['pae_stop']} / {r['control_stop']}")
    print(f"mean  PAE={t['pae_mean']:.4f}  control={t['control_mean']:.4f}  "
          f"diff(PAE-ctl)={t['diff_mean']:+.4f}  SD={t['diff_sd']:.4f}  SE={t['diff_se']:.4f}")
    s, w = t["sign_test"], t["wilcoxon"]
    print(f"  sign test (one-sided, PAE>ctl): +{s['pos']} / -{s['neg']} of {s['discordant']} discordant, "
          f"p={s['p']:.4f}  (min achievable p at this discordant count = {s['p_min']:.4f})")
    print(f"  wilcoxon  (one-sided, exact):   n_nonzero={w['n_nonzero']} W+={w['W_plus']:.1f} "
          f"p={w['p']:.4f}  (min achievable p = {w['p_min']:.4f})")
    f = t["first_episode"]
    print(f"  [validity check - these are the SAME distribution by construction] "
          f"first-episode mean: PAE attempt1={f['pae_attempt1_mean']:.4f} "
          f"vs control episode1={f['control_episode1_mean']:.4f}; "
          f"solved 1.0: {f['pae_attempt1_solved']}/10 vs {f['control_episode1_solved']}/10")
    m = t["matched"]
    print(f"  [budget-matched: control truncated to PAE's own attempt count] "
          f"control mean={m['control_mean']:.4f} diff={m['diff_mean']:+.4f} SD={m['diff_sd']:.4f} "
          f"sign p={m['sign_test']['p']:.4f} (+{m['sign_test']['pos']}/-{m['sign_test']['neg']} of "
          f"{m['sign_test']['discordant']}, min p={m['sign_test']['p_min']:.4f}) "
          f"wilcoxon p={m['wilcoxon']['p']:.4f}")
    print(f"  attempts: PAE {t['pae_attempts_total']} vs control {t['control_episodes_total']}   "
          f"llm steps: PAE {t['pae_llm_steps']} vs control {t['control_llm_steps']}   "
          f"billed: PAE ${t['pae_billed_usd']:.2f} vs control ${t['control_billed_usd']:.2f}")

p = report["pooled"]
print(f"\n=== pooled (n={p['n']}) ===")
print(f"mean diff {p['diff_mean']:+.4f}  SD {p['diff_sd']:.4f}")
print(f"  sign: +{p['sign_test']['pos']}/-{p['sign_test']['neg']} of {p['sign_test']['discordant']}, "
      f"p={p['sign_test']['p']:.4f} (min {p['sign_test']['p_min']:.6f})")
print(f"  wilcoxon p={p['wilcoxon']['p']:.4f} (min {p['wilcoxon']['p_min']:.6f})")
pm = p["matched"]
print(f"  [budget-matched] diff {pm['diff_mean']:+.4f} SD {pm['diff_sd']:.4f}  "
      f"sign +{pm['sign_test']['pos']}/-{pm['sign_test']['neg']} of {pm['sign_test']['discordant']} "
      f"p={pm['sign_test']['p']:.4f} (min {pm['sign_test']['p_min']:.6f})  "
      f"wilcoxon p={pm['wilcoxon']['p']:.4f}")
print(f"\nTOTALS  attempts PAE {report['totals']['pae_attempts']} vs control {report['totals']['ctl_attempts']}"
      f"   billed PAE ${report['totals']['pae_billed']:.2f} vs control ${report['totals']['ctl_billed']:.2f}")
