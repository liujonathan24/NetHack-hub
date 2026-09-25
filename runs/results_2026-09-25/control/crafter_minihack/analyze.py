"""Paired comparison: PAE best-over-N vs matched fresh best-of-N control.

Crafter (graded: achievements/22, 10 seeds) and MiniHack-CorridorBattle-Dark
(binary, 5 seeds). Exact tests implemented directly (no scipy):
  * sign test              - one-sided exact, H1: PAE > control
  * Wilcoxon signed-rank   - one-sided exact
  * sign-flip permutation  - one-sided exact randomization test on the mean
                             difference (appropriate for the graded metric)
Each reports the MINIMUM achievable p at the observed discordant/non-zero
count, so a null result is not read as evidence of absence.
"""
import json, math, os, statistics as st
from itertools import product

CTL = "/root/nld/gen_runs/control_bestofn_2"
def _mh(task):
    return lambda s: f"/root/nld/gen_runs/panel_minihack/{task}_s{s}_pae"

ARMS = [
    ("crafter", "default", range(10), lambda s: f"/root/nld/gen_runs/panel/crafter_s{s}"),
    ("minihack", "MiniHack-CorridorBattle-Dark-v0", range(5), _mh("MiniHack-CorridorBattle-Dark-v0")),
    ("minihack", "MiniHack-Corridor-R3-v0", range(5), _mh("MiniHack-Corridor-R3-v0")),
]


def sign_test(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0); n = pos + neg
    if n == 0:
        return dict(pos=0, neg=0, discordant=0, p=1.0, p_min=1.0)
    return dict(pos=pos, neg=neg, discordant=n,
                p=sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n, p_min=1 / 2 ** n)


def wilcoxon(d):
    nz = [x for x in d if x != 0]; n = len(nz)
    if n == 0:
        return dict(n_nonzero=0, W_plus=0.0, p=1.0, p_min=1.0)
    order = sorted(range(n), key=lambda i: abs(nz[i])); ranks = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if nz[i] > 0)
    cnt = sum(1 for sg in product([0, 1], repeat=n)
              if sum(ranks[i] for i in range(n) if sg[i]) >= wp)
    return dict(n_nonzero=n, W_plus=wp, p=cnt / 2 ** n, p_min=1 / 2 ** n)


def perm_test(d):
    """Exact one-sided sign-flip permutation test on the mean difference."""
    nz = [x for x in d if x != 0]; n = len(nz)
    if n == 0:
        return dict(n_nonzero=0, mean=0.0, p=1.0, p_min=1.0)
    obs = sum(nz)
    cnt = sum(1 for sg in product([1, -1], repeat=n)
              if sum(s * v for s, v in zip(sg, nz)) >= obs)
    return dict(n_nonzero=n, mean=st.mean(d), p=cnt / 2 ** n, p_min=1 / 2 ** n)


def sd(xs):
    xs = list(xs)
    return st.stdev(xs) if len(xs) > 1 else 0.0


def ci95(d):
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    se = sd(d) / math.sqrt(n)
    t = {4: 2.776, 5: 2.571, 9: 2.262, 10: 2.228}.get(n - 1, 2.26)
    m = st.mean(d)
    return (m - t * se, m + t * se)


report = {"arms": {}}
for game, task, seeds, paedir in ARMS:
    rows = []
    skipped = []
    for s in seeds:
        pae_f = f"{paedir(s)}/summary.json"
        ctl_f = f"{CTL}/ctl_{game}_{task}_s{s}/control_summary.json"
        if not os.path.exists(pae_f) or not os.path.exists(ctl_f):
            skipped.append(dict(seed=s, pae_closed=os.path.exists(pae_f),
                                control_closed=os.path.exists(ctl_f)))
            continue
        P = json.load(open(pae_f))
        C = json.load(open(ctl_f))
        per = [float(x) for x in C["per_episode_progression"]]
        k = int(P["attempts"])
        rows.append(dict(
            seed=s,
            pae_best=float(P["pae_best_progression"]),
            control_best=float(C["control_best_progression"]),
            diff=float(P["pae_best_progression"]) - float(C["control_best_progression"]),
            control_best_at_pae_budget=max(per[:k], default=0.0),
            diff_matched=float(P["pae_best_progression"]) - max(per[:k], default=0.0),
            control_episodes_at_pae_budget=min(k, len(per)),
            pae_aux=P.get("aux_max_measured"), control_aux=C.get("control_best_aux"),
            pae_attempt1=float(P["attempt1_progression"]),
            control_episode1=float(C["episode1_progression"]),
            pae_attempts=k, control_episodes=C["episodes_used"],
            pae_stop=P["stop_reason"], control_stop=C["stop_reason"],
            pae_billed=float(P["tokens"].get("billed_by_provider_usd", 0.0)),
            control_billed=float(C["billed_by_provider_usd"]),
            control_billed_at_pae_budget=round(sum(C["per_episode_billed_usd"][:k]), 6),
            pae_llm_steps=P["llm_steps"], control_llm_steps=C["llm_steps"],
            control_llm_steps_at_pae_budget=sum(C["per_episode_llm_steps"][:k]),
            digests_match=sum(1 for x in C["episode_start_digests"]
                              if x == json.load(open(f"{paedir(s)}/archive/c1/meta.json"))["obs_digest"]),
            n_control_eps=len(C["episode_start_digests"]),
        ))
    d = [r["diff"] for r in rows]; dm = [r["diff_matched"] for r in rows]
    if not rows:
        report["arms"][f"{game}/{task}"] = dict(task=task, n=0, rows=[], skipped_seeds=skipped)
        continue
    report["arms"][f"{game}/{task}"] = dict(
        task=task, n=len(rows), rows=rows, skipped_seeds=skipped,
        seeds_paired=[r["seed"] for r in rows],
        pae_mean=st.mean(r["pae_best"] for r in rows),
        pae_sd=sd(r["pae_best"] for r in rows),
        control_mean=st.mean(r["control_best"] for r in rows),
        control_sd=sd(r["control_best"] for r in rows),
        diff_mean=st.mean(d), diff_sd=sd(d), diff_ci95=ci95(d),
        sign_test=sign_test(d), wilcoxon=wilcoxon(d), permutation=perm_test(d),
        matched=dict(
            control_mean=st.mean(r["control_best_at_pae_budget"] for r in rows),
            control_sd=sd(r["control_best_at_pae_budget"] for r in rows),
            diff_mean=st.mean(dm), diff_sd=sd(dm), diff_ci95=ci95(dm),
            sign_test=sign_test(dm), wilcoxon=wilcoxon(dm), permutation=perm_test(dm),
            control_llm_steps=sum(r["control_llm_steps_at_pae_budget"] for r in rows),
            control_billed_usd=round(sum(r["control_billed_at_pae_budget"] for r in rows), 4),
        ),
        first_episode=dict(
            pae_attempt1_mean=st.mean(r["pae_attempt1"] for r in rows),
            control_episode1_mean=st.mean(r["control_episode1"] for r in rows),
            pae_attempt1_solved=sum(1 for r in rows if r["pae_attempt1"] >= 1.0),
            control_episode1_solved=sum(1 for r in rows if r["control_episode1"] >= 1.0),
            permutation=perm_test([r["pae_attempt1"] - r["control_episode1"] for r in rows]),
        ),
        pae_attempts_total=sum(r["pae_attempts"] for r in rows),
        control_episodes_total=sum(r["control_episodes"] for r in rows),
        pae_billed_usd=round(sum(r["pae_billed"] for r in rows), 4),
        control_billed_usd=round(sum(r["control_billed"] for r in rows), 4),
        pae_llm_steps=sum(r["pae_llm_steps"] for r in rows),
        control_llm_steps=sum(r["control_llm_steps"] for r in rows),
        digest_pairs_matched=sum(r["digests_match"] for r in rows),
        digest_pairs_total=sum(r["n_control_eps"] for r in rows),
    )
# Pooled across the two MiniHack tasks: both are binary solve/no-solve with the
# same attempt-1 signature, so pooling them is the second, larger test of
# whether resumption beats re-rolling on MiniHack at all.
mh = [r for k, a in report["arms"].items() if k.startswith("minihack/") and a["n"]
      for r in a["rows"]]
if mh:
    d = [r["diff"] for r in mh]; dm = [r["diff_matched"] for r in mh]
    report["minihack_pooled"] = dict(
        n=len(mh), seeds=[(r["seed"]) for r in mh],
        pae_mean=st.mean(r["pae_best"] for r in mh),
        control_mean=st.mean(r["control_best"] for r in mh),
        diff_mean=st.mean(d), diff_sd=sd(d),
        sign_test=sign_test(d), wilcoxon=wilcoxon(d), permutation=perm_test(d),
        matched=dict(control_mean=st.mean(r["control_best_at_pae_budget"] for r in mh),
                     diff_mean=st.mean(dm), diff_sd=sd(dm),
                     sign_test=sign_test(dm), wilcoxon=wilcoxon(dm), permutation=perm_test(dm)),
        pae_attempt1_solved=sum(1 for r in mh if r["pae_attempt1"] >= 1.0),
        control_episode1_solved=sum(1 for r in mh if r["control_episode1"] >= 1.0),
        pae_billed_usd=round(sum(r["pae_billed"] for r in mh), 4),
        control_billed_usd=round(sum(r["control_billed"] for r in mh), 4),
        control_billed_at_pae_budget_usd=round(sum(r["control_billed_at_pae_budget"] for r in mh), 4),
        pae_llm_steps=sum(r["pae_llm_steps"] for r in mh),
        control_llm_steps=sum(r["control_llm_steps"] for r in mh),
        control_llm_steps_at_pae_budget=sum(r["control_llm_steps_at_pae_budget"] for r in mh),
    )

json.dump(report, open(f"{CTL}/comparison.json", "w"), indent=2)

for tname, a in report["arms"].items():
    print(f"\n=== {tname} (n={a['n']} paired seeds) ===")
    if a["skipped_seeds"]:
        print(f"    EXCLUDED (not closed on both arms): {a['skipped_seeds']}")
    if a["n"] == 0:
        print("    no paired seeds yet"); continue
    print(f"    seeds paired: {a['seeds_paired']}")
    print(f"{'seed':>4} {'PAE best':>9} {'ctl best':>9} {'diff':>8} | {'ctlBest@k':>9} {'diff@k':>8} "
          f"| {'PAEatt':>6} {'ctlEps':>6} | {'PAE$':>7} {'ctl$':>7} | PAEstop/ctlstop")
    for r in a["rows"]:
        print(f"{r['seed']:>4} {r['pae_best']:>9.4f} {r['control_best']:>9.4f} {r['diff']:>+8.4f} "
              f"| {r['control_best_at_pae_budget']:>9.4f} {r['diff_matched']:>+8.4f} "
              f"| {r['pae_attempts']:>6} {r['control_episodes']:>6} "
              f"| {r['pae_billed']:>7.4f} {r['control_billed']:>7.4f} | {r['pae_stop']}/{r['control_stop']}")
    print(f"\n-- FULL BUDGET (PAE best-over-N vs control best-over-10) --")
    print(f"   PAE mean={a['pae_mean']:.4f} SD={a['pae_sd']:.4f} | control mean={a['control_mean']:.4f} "
          f"SD={a['control_sd']:.4f}")
    print(f"   diff(PAE-ctl) mean={a['diff_mean']:+.4f} SD={a['diff_sd']:.4f} "
          f"95%CI=[{a['diff_ci95'][0]:+.4f},{a['diff_ci95'][1]:+.4f}]")
    s, w, pm = a["sign_test"], a["wilcoxon"], a["permutation"]
    print(f"   sign: +{s['pos']}/-{s['neg']} of {s['discordant']} discordant p={s['p']:.4f} "
          f"(min achievable p={s['p_min']:.4f})")
    print(f"   wilcoxon: n_nz={w['n_nonzero']} W+={w['W_plus']:.1f} p={w['p']:.4f} (min p={w['p_min']:.4f})")
    print(f"   permutation: n_nz={pm['n_nonzero']} p={pm['p']:.4f} (min p={pm['p_min']:.4f})")
    m = a["matched"]
    print(f"\n-- MATCHED ATTEMPT COUNT (control truncated to PAE's own attempts) --")
    print(f"   control mean={m['control_mean']:.4f} SD={m['control_sd']:.4f} | "
          f"diff mean={m['diff_mean']:+.4f} SD={m['diff_sd']:.4f} "
          f"95%CI=[{m['diff_ci95'][0]:+.4f},{m['diff_ci95'][1]:+.4f}]")
    s, w, pm = m["sign_test"], m["wilcoxon"], m["permutation"]
    print(f"   sign: +{s['pos']}/-{s['neg']} of {s['discordant']} discordant p={s['p']:.4f} (min p={s['p_min']:.4f})")
    print(f"   wilcoxon: n_nz={w['n_nonzero']} p={w['p']:.4f} (min p={w['p_min']:.4f})")
    print(f"   permutation: n_nz={pm['n_nonzero']} p={pm['p']:.4f} (min p={pm['p_min']:.4f})")
    print(f"   attempts: PAE {a['pae_attempts_total']} vs control(truncated) {a['pae_attempts_total']}")
    print(f"   llm steps: PAE {a['pae_llm_steps']} vs control {m['control_llm_steps']}")
    print(f"   billed:    PAE ${a['pae_billed_usd']:.2f} vs control ${m['control_billed_usd']:.2f}")
    f = a["first_episode"]
    print(f"\n-- VALIDITY: first-episode distributions (should be the same by construction) --")
    print(f"   PAE attempt1 mean={f['pae_attempt1_mean']:.4f} vs control episode1 mean={f['control_episode1_mean']:.4f}"
          f"  (perm p={f['permutation']['p']:.4f})")
    print(f"   solved 1.0: PAE {f['pae_attempt1_solved']}/{a['n']} vs control {f['control_episode1_solved']}/{a['n']}")
    print(f"\n-- FULL-BUDGET totals: attempts PAE {a['pae_attempts_total']} vs control {a['control_episodes_total']};"
          f" llm steps PAE {a['pae_llm_steps']} vs control {a['control_llm_steps']};"
          f" billed PAE ${a['pae_billed_usd']:.2f} vs control ${a['control_billed_usd']:.2f}")
    print(f"-- binding: {a['digest_pairs_matched']}/{a['digest_pairs_total']} control episodes' reset digests "
          f"== the PAE run's episode-start digest")

mp = report.get("minihack_pooled")
if mp:
    print(f"\n=== MiniHack POOLED across both tasks (n={mp['n']} seeds) ===")
    print(f"   full budget:  PAE mean={mp['pae_mean']:.4f} vs control mean={mp['control_mean']:.4f}  "
          f"diff={mp['diff_mean']:+.4f} SD={mp['diff_sd']:.4f}")
    s_, w_, p_ = mp["sign_test"], mp["wilcoxon"], mp["permutation"]
    print(f"     sign +{s_['pos']}/-{s_['neg']} of {s_['discordant']} discordant p={s_['p']:.4f} "
          f"(min achievable p={s_['p_min']:.4f});  wilcoxon p={w_['p']:.4f};  perm p={p_['p']:.4f}")
    m_ = mp["matched"]
    s_, w_, p_ = m_["sign_test"], m_["wilcoxon"], m_["permutation"]
    print(f"   matched:      control mean={m_['control_mean']:.4f}  diff={m_['diff_mean']:+.4f} SD={m_['diff_sd']:.4f}")
    print(f"     sign +{s_['pos']}/-{s_['neg']} of {s_['discordant']} discordant p={s_['p']:.4f} "
          f"(min achievable p={s_['p_min']:.4f});  wilcoxon p={w_['p']:.4f};  perm p={p_['p']:.4f}")
    print(f"   first episode solved: PAE attempt1 {mp['pae_attempt1_solved']}/{mp['n']} vs "
          f"control episode1 {mp['control_episode1_solved']}/{mp['n']}")
    print(f"   llm steps: PAE {mp['pae_llm_steps']} vs control(full) {mp['control_llm_steps']} "
          f"vs control(matched) {mp['control_llm_steps_at_pae_budget']}")
    print(f"   billed: PAE ${mp['pae_billed_usd']:.2f} vs control(full) ${mp['control_billed_usd']:.2f} "
          f"vs control(matched) ${mp['control_billed_at_pae_budget_usd']:.2f}")
