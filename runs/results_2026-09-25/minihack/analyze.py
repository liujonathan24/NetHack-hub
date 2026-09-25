#!/usr/bin/env python3
"""MiniHack panel readout. Same definitions as /root/nld/gen_runs/panel/analyze.py
(paired within-run delta, genuine resume = >2 steps played, fidelity/parity
totals, billed_by_provider_usd as the attributable cost), plus an exact test
appropriate to a BINARY progression metric."""
import json, os, glob, statistics as st
from collections import defaultdict
from math import comb

ROOT = "/root/nld/gen_runs/panel_minihack"

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
for d in sorted(glob.glob(f"{ROOT}/*_pae/")):
    name = os.path.basename(d.rstrip("/"))
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        runs.append(dict(name=name, incomplete=True)); continue
    s = json.load(open(sp))
    att = loadl(os.path.join(d, "attempts.jsonl"))
    rf  = loadl(os.path.join(d, "restore_fidelity.jsonl"))
    rp  = loadl(os.path.join(d, "resume_prompt_check.jsonl"))
    resumed = att[1:]
    genuine = [a for a in resumed if (a.get("steps_played") or 0) > 2]
    trivial = [a for a in resumed if (a.get("steps_played") or 0) <= 2]
    runs.append(dict(
        name=name, incomplete=False, game=s["game"], task=s["task"], seed=s["seed"],
        a1=s["attempt1_progression"], best=s["pae_best_progression"],
        delta=s["pae_best_progression"] - s["attempt1_progression"],
        attempts=s["attempts"], stop=s["stop_reason"], ckpts=s["checkpoints"],
        committed=s["committed_steps_max"], step_cap=s["step_cap"],
        cap_hit=(s["committed_steps_max"] >= (s["step_cap"] or 10**9)),
        retry_exh=s["client_retry_exhausted"], invalid=s["invalid_actions"],
        billed=s["tokens"]["billed_by_provider_usd"], ledger=s["tokens"]["total"]["cost_usd"],
        llm_steps=s["llm_steps"], wall=s["wall_s"], aux=s["aux_max_measured"],
        n_resumed=len(resumed), n_genuine=len(genuine), n_trivial=len(trivial),
        resume_steps=[a.get("steps_played") for a in resumed],
        rf_total=len(rf), rf_ok=sum(1 for r in rf if r.get("identical")),
        rp_total=len(rp), rp_ok=sum(1 for r in rp if r.get("history_matches_checkpoint")),
        attempts_rows=att,
        root_picks=s["orchestrator"]["root_picks"], rejections=s["orchestrator"]["directive_rejections"],
    ))

ok  = [r for r in runs if not r["incomplete"]]
inc = [r for r in runs if r["incomplete"]]
groups = defaultdict(list)
for r in ok: groups[r["task"]].append(r)

print(f"runs complete: {len(ok)}   incomplete/missing: {len(inc)} {[r['name'] for r in inc]}\n")
for g in sorted(groups):
    rs = sorted(groups[g], key=lambda r: r["seed"])
    print(f"===== {g}  (n={len(rs)}) =====")
    # CLASSIFICATION IS A VISIBLE COLUMN, never a derived note: the three
    # classes carry OPPOSITE meanings and all print as +0.0 delta.
    #   zero-by-ceiling -> attempt 1 ALREADY SOLVED; run ended at attempt 1.
    #   zero-by-floor   -> nothing solved in N attempts.
    # Pooling them misreports both.
    def klass(r):
        if r["best"] > r["a1"]: return "NON-ZERO"
        return "zero-by-ceiling" if r["a1"] >= 1.0 else "zero-by-floor"
    print(f"{'seed':>4} {'attempt1':>9} {'best':>6} {'delta':>7} {'att':>4} {'ckpt':>5} "
          f"{'CLASS':>16} {'stop':>20} {'resumed':>8} {'genuine':>8} {'retryX':>7} {'billed':>8}")
    for r in rs:
        print(f"{r['seed']:>4} {r['a1']:>9.1f} {r['best']:>6.1f} {r['delta']:>+7.1f} {r['attempts']:>4} "
              f"{r['ckpts']:>5} {klass(r):>16} {r['stop']:>20} {r['n_resumed']:>8} {r['n_genuine']:>8} "
              f"{r['retry_exh']:>7} {r['billed']:>8.4f}")
    ds = [r["delta"] for r in rs]; a1s = [r["a1"] for r in rs]; bs = [r["best"] for r in rs]
    n = len(ds)
    npos = sum(1 for x in ds if x > 0)
    n_ceiling = sum(1 for r in rs if r["a1"] >= 1.0)
    n_floor   = sum(1 for r in rs if r["a1"] < 1.0 and r["best"] <= r["a1"])
    # attempt number at which the first scoring attempt occurred, per seed
    first_hit = []
    for r in rs:
        fh = None
        for a in r["attempts_rows"]:
            if (a.get("progression") or 0) >= 1.0:
                fh = a["attempt"]; break
        first_hit.append(fh)
    print(f"  ATTEMPT-1 SOLVE RATE (unmodified naive agent, leaderboard-comparable)"
          f" = {sum(1 for x in a1s if x>=1.0)}/{n} = {st.mean(a1s):.3f}")
    print(f"  seeds reaching a non-zero best : {npos}/{n}")
    print(f"  attempt # of first success     : {first_hit}")
    print(f"  classification                 : non-zero {npos}, zero-by-floor {n_floor}, "
          f"zero-by-ceiling {n_ceiling}")
    print(f"  PAE best mean = {st.mean(bs):.3f}   (paired delta mean = {st.mean(ds):+.3f})")
    # -- deliberately NO significance test. --------------------------------
    # `pae_best_progression` is the max over ALL attempts INCLUDING attempt 1,
    # and delta subtracts attempt 1 from it, so delta >= 0 BY CONSTRUCTION:
    # negatives are impossible. A sign test on direction therefore has a
    # guaranteed zero in its second column and tests nothing, and a p-value
    # against a 50/50 null is unjustified.
    #
    # Worse, the delta conflates the two things we need to separate. "Some
    # attempt after the first beat the first" would ALSO happen if attempts
    # 2..N were N fresh independent episodes with no checkpointing at all,
    # because a larger attempt budget helps regardless of mechanism. Only the
    # resumption part is PAE's claim.
    #
    # The comparison PAE needs is against a MATCHED CONTROL: N independent
    # fresh episodes at the same attempt budget. WE DO NOT HAVE THAT ARM YET.
    # Until we do, these are descriptive facts, not evidence of an effect.
    print(f"  restore fidelity {sum(r['rf_ok'] for r in rs)}/{sum(r['rf_total'] for r in rs)}   "
          f"resumed-prompt parity {sum(r['rp_ok'] for r in rs)}/{sum(r['rp_total'] for r in rs)}")
    print(f"  resumes: genuine(>2 steps) {sum(r['n_genuine'] for r in rs)} / "
          f"{sum(r['n_resumed'] for r in rs)} total; trivial(<=2 steps) {sum(r['n_trivial'] for r in rs)}")
    print(f"  client_retry_exhausted total {sum(r['retry_exh'] for r in rs)}   "
          f"invalid_actions {sum(r['invalid'] for r in rs)}")
    print(f"  aux max per seed: {[r['aux'] for r in rs]}   root_picks {sum(r['root_picks'] for r in rs)}"
          f"  directive_rejections {sum(r['rejections'] for r in rs)}")
    print(f"  billed subtotal ${sum(r['billed'] for r in rs):.4f}   "
          f"wall max {max(r['wall'] for r in rs)/60:.0f} min\n")

print(f"ATTRIBUTABLE TOTAL (sum of per-run billed_by_provider_usd) = ${sum(r['billed'] for r in ok):.4f}")
print(f"ledger cost_usd total (overstated)                        = ${sum(r['ledger'] for r in ok):.4f}")
print("retry_exhausted per run: " + ", ".join(f"{r['name'].replace('MiniHack-','').replace('-v0','')}={r['retry_exh']}" for r in ok))
