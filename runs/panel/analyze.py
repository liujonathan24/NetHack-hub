#!/usr/bin/env python3
"""Panel readout: paired deltas, resume quality, fidelity/parity, pathology."""
import json, os, glob, math, statistics as st

def _t_sf(t, dof):
    """One-sided upper-tail p for Student-t, via the regularized incomplete beta."""
    if t <= 0:
        return 1.0 - _t_sf(-t, dof)
    x = dof / (dof + t * t)
    return 0.5 * _betainc(dof / 2.0, 0.5, x)

def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b) by continued fraction (Lentz)."""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta)
    if x < (a + 1) / (a + b + 2):
        return front / a * _betacf(a, b, x)
    return 1.0 - math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta) / b * _betacf(b, a, 1 - x)

def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300: d = 1e-300
    d = 1.0 / d; h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d;  c = 1.0 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d;  c = 1.0 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < eps: break
    return h
from collections import defaultdict

ROOT = "/root/nld/gen_runs/panel"

def loadl(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            l = l.strip()
            if l:
                try: out.append(json.loads(l))
                except Exception: pass
    return out

runs = []
for d in sorted(glob.glob(f"{ROOT}/*/")):
    name = os.path.basename(d.rstrip("/"))
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        runs.append(dict(name=name, incomplete=True)); continue
    s = json.load(open(sp))
    att = loadl(os.path.join(d, "attempts.jsonl"))
    sel = loadl(os.path.join(d, "selection.jsonl"))
    rf  = loadl(os.path.join(d, "restore_fidelity.jsonl"))
    rp  = loadl(os.path.join(d, "resume_prompt_check.jsonl"))
    resumed = att[1:]
    genuine = [a for a in resumed if (a.get("steps_played") or 0) > 2]
    trivial = [a for a in resumed if (a.get("steps_played") or 0) <= 2]
    # fatal-checkpoint pathology: a checkpoint from which an attempt already
    # ended in <=2 steps gets re-picked by a LATER attempt, i.e. the 'tried'
    # column reported the fatality and the orchestrator chose it anyway.
    fatal_at = {}   # ckpt -> attempt index at which it first proved fatal
    patho = []
    for a in resumed:
        c = a.get("from_checkpoint")
        if c is None: continue
        if c in fatal_at and fatal_at[c] < a["attempt"]:
            patho.append((a["attempt"], c, a.get("steps_played")))
        if (a.get("steps_played") or 0) <= 2:
            fatal_at.setdefault(c, a["attempt"])
    runs.append(dict(
        name=name, incomplete=False, game=s["game"], task=s["task"], seed=s["seed"],
        a1=s["attempt1_progression"], best=s["pae_best_progression"],
        delta=s["pae_best_progression"] - s["attempt1_progression"],
        attempts=s["attempts"], stop=s["stop_reason"],
        cap_hit=(s["committed_steps_max"] >= (s["step_cap"] or 10**9)),
        committed=s["committed_steps_max"], step_cap=s["step_cap"],
        retry_exh=s["client_retry_exhausted"], invalid=s["invalid_actions"],
        billed=s["tokens"]["billed_by_provider_usd"], ledger=s["tokens"]["total"]["cost_usd"],
        llm_steps=s["llm_steps"], wall=s["wall_s"],
        n_resumed=len(resumed), n_genuine=len(genuine), n_trivial=len(trivial),
        resume_steps=[a.get("steps_played") for a in resumed],
        rf_total=len(rf), rf_ok=sum(1 for r in rf if r.get("identical")),
        rp_total=len(rp), rp_ok=sum(1 for r in rp if r.get("history_matches_checkpoint")),
        patho=patho, root_picks=s["orchestrator"]["root_picks"],
        rejections=s["orchestrator"]["directive_rejections"], aux=s["aux_max_measured"],
    ))

ok = [r for r in runs if not r["incomplete"]]
inc = [r for r in runs if r["incomplete"]]

def key(r):
    return "crafter" if r["game"] == "crafter" else f"textworld/{r['task']}"

groups = defaultdict(list)
for r in ok: groups[key(r)].append(r)

print(f"runs complete: {len(ok)}   incomplete/missing: {len(inc)} {[r['name'] for r in inc]}\n")
for g in sorted(groups):
    rs = sorted(groups[g], key=lambda r: r["seed"])
    print(f"===== {g}  (n={len(rs)}) =====")
    print(f"{'seed':>4} {'attempt1':>9} {'best':>8} {'delta':>8} {'att':>4} {'stop':>18} {'resumed':>8} {'genuine':>8} {'retryX':>7} {'billed':>8}")
    for r in rs:
        print(f"{r['seed']:>4} {r['a1']:>9.4f} {r['best']:>8.4f} {r['delta']:>+8.4f} {r['attempts']:>4} "
              f"{r['stop']:>18} {r['n_resumed']:>8} {r['n_genuine']:>8} {r['retry_exh']:>7} {r['billed']:>8.4f}")
    ds = [r["delta"] for r in rs]; a1s = [r["a1"] for r in rs]; bs = [r["best"] for r in rs]
    sd = st.stdev(ds) if len(ds) > 1 else 0.0
    sem = sd / (len(ds) ** 0.5) if len(ds) > 1 else 0.0
    npos = sum(1 for x in ds if x > 0); nzero = sum(1 for x in ds if x == 0)
    print(f"  BASE (attempt-1 mean) = {st.mean(a1s):.4f}   PAE best mean = {st.mean(bs):.4f}")
    print(f"  PAIRED DELTA mean = {st.mean(ds):+.4f}  sd = {sd:.4f}  sem = {sem:.4f}  "
          f"mean/sd = {(st.mean(ds)/sd if sd else float('inf')):.2f}  positive {npos}/{len(ds)} zero {nzero}")
    if sd > 0:
        lo, hi = st.mean(ds) - 1.96 * sem, st.mean(ds) + 1.96 * sem
        print(f"  95% CI on the mean delta = [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SEPARATED from 0' if lo > 0 else '*** NOT separated from 0 ***'}")
    # Primary test.  treasure_hunter / coin_collector are binary (won/not), so the
    # honest test is an exact one-sided sign test over the discordant pairs; the
    # cooking game is graded out of 17, so a paired t-test on the deltas is valid.
    binary = g.endswith("treasure_hunter") or g.endswith("coin_collector")
    nneg = sum(1 for x in ds if x < 0)
    m = npos + nneg
    if binary:
        pv = sum(math.comb(m, k) for k in range(npos, m + 1)) / 2 ** m if m else 1.0
        print(f"  SIGN TEST (exact, one-sided): {npos}+ / {nneg}- over {m} discordant "
              f"of {len(ds)} pairs -> p = {pv:.5f}  "
              f"{'significant at 0.05' if pv < 0.05 else '*** not significant at 0.05 ***'}")
    else:
        if sd > 0 and len(ds) > 1:
            tstat = st.mean(ds) / sem
            dof = len(ds) - 1
            pv = _t_sf(tstat, dof)
            print(f"  PAIRED t-TEST (one-sided): t({dof}) = {tstat:.3f}  p = {pv:.5f}  "
                  f"{'significant at 0.05' if pv < 0.05 else '*** not significant at 0.05 ***'}")
        pvs = sum(math.comb(m, k) for k in range(npos, m + 1)) / 2 ** m if m else 1.0
        print(f"  (sign test for reference: {npos}+ / {nneg}- of {m} discordant -> p = {pvs:.5f})")
    print(f"  restore fidelity {sum(r['rf_ok'] for r in rs)}/{sum(r['rf_total'] for r in rs)}   "
          f"resumed-prompt parity {sum(r['rp_ok'] for r in rs)}/{sum(r['rp_total'] for r in rs)}")
    print(f"  resumes: genuine(>2 steps) {sum(r['n_genuine'] for r in rs)} / "
          f"{sum(r['n_resumed'] for r in rs)} total; trivial(<=2 steps) {sum(r['n_trivial'] for r in rs)}")
    print(f"  client_retry_exhausted total {sum(r['retry_exh'] for r in rs)}   "
          f"invalid_actions {sum(r['invalid'] for r in rs)}")
    caps = [r["name"] for r in rs if r["cap_hit"]]
    print(f"  runs reaching the step cap: {len(caps)} {caps}")
    pa = [(r["name"], r["patho"]) for r in rs if r["patho"]]
    print(f"  fatal-checkpoint pathology (re-pick after 'tried' showed <=2-step end): {len(pa)} runs {pa if pa else ''}")
    print(f"  billed subtotal ${sum(r['billed'] for r in rs):.4f}   wall max {max(r['wall'] for r in rs)/60:.0f} min\n")

print(f"ATTRIBUTABLE TOTAL (sum of traces usage.cost) = ${sum(r['billed'] for r in ok):.4f}")
print(f"ledger cost_usd total (overstated)            = ${sum(r['ledger'] for r in ok):.4f}")
print(f"retry_exhausted per run: " + ", ".join(f"{r['name']}={r['retry_exh']}" for r in ok))
cr = f"{ROOT}/crashes.jsonl"
print("\ncrashes: " + (open(cr).read().strip() if os.path.exists(cr) else "none"))
