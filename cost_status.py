#!/usr/bin/env python3
"""E14 Diversity spend: what has been billed, what is left, what it will total.

Two independent numbers, on purpose:
  WALLET   `prime wallet` -- the only authority for what the account is billed.
           Day-aggregated and it LAGS, and this box is multi-tenant, so a day's
           total includes other experiments. Use it as the ceiling check.
  TRACES   per-call usage out of each finished rollout, priced per model with
           that model's OWN measured cache share. Exact where it applies -- it
           matched a wallet row to $0.002 on claude-fable-5 -- but it can only
           see rollouts that have finished writing.
"""
import json, glob, os, subprocess, sys, collections

PRICE = {'openai/gpt-5.6-luna': (0.20, 1.20), 'openai/gpt-5.6-sol': (5.00, 30.00),
         'deepseek/deepseek-v4-flash-0731': (0.44, 1.32),
         'google/gemini-3.7-flash': (1.35, 6.75), 'qwen/qwen3.8-max': (2.00, 6.00)}
# Measured per-model cache share (E14 Diversity seed-1 pilot). NOT a constant:
# Anthropic models measured 0.0% and cost ~10x what this rate would imply.
CACHE_RATE = 0.12
TARGET = 60          # 4 models x 5 seeds x 3 reps (deepseek dropped after rep 1)
PROJECTED = 96.70    # pilot estimate, 5 cells; deepseek contributed 1 rep only

def rollout_cost(t):
    f = c = o = 0
    for x in t.get('calls') or []:
        u = x.get('usage') or {}
        f += u.get('prompt_tokens') or 0
        c += u.get('cached_input_tokens') or 0
        o += u.get('completion_tokens') or 0
    pi, po = PRICE[t['agent']['model']]
    return (f * pi + c * pi * CACHE_RATE + o * po) / 1e6

def main():
    per = collections.defaultdict(lambda: [0.0, 0, 0])   # model -> [$, rollouts, calls]
    for p in sorted(glob.glob('outputs/e14_diversity_full/*/traces.jsonl')):
        if not os.path.getsize(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            t = json.loads(line)
            a = per[t['agent']['model']]
            a[0] += rollout_cost(t); a[1] += 1; a[2] += len(t.get('calls') or [])
    done = sum(a[1] for a in per.values())
    spent = sum(a[0] for a in per.values())
    print(f"{'model':<32}{'done':>6}{'calls':>8}{'$ so far':>10}{'$/rollout':>11}{'proj x15':>10}")
    proj_total = 0.0
    for m, a in sorted(per.items(), key=lambda kv: -kv[1][0]):
        per_roll = a[0] / a[1]
        proj_total += per_roll * 15
        print(f"{m:<32}{a[1]:>6}{a[2]:>8}{a[0]:>10.2f}{per_roll:>11.3f}{per_roll*15:>10.2f}")
    print(f"\n{done}/{TARGET} rollouts done -- ${spent:.2f} spent on them")
    if done:
        print(f"projection from THIS run's own rollouts: ${proj_total:.2f}"
              f"   (pilot said ${PROJECTED:.2f})")
        if done < 15:
            print("  ^ fewer than 15 rollouts in: treat as a direction, not a number.")
    try:
        w = json.loads(subprocess.run(['prime', 'wallet', '-n', '60', '-o', 'json', '--plain'],
                                      capture_output=True, text=True, timeout=60).stdout)
        day = collections.defaultdict(float)
        for b in w['recent_billings']:
            if b['resource_type'] == 'inference':
                day[b['created_at'][:10]] += float(b['amount_usd'])
        print(f"\nwallet balance ${w['balance_usd']:.2f}")
        for k in sorted(day)[-2:]:
            print(f"  {k} inference (ALL tenants on this box): ${day[k]:.2f}")
    except Exception as e:
        print(f"\nwallet unavailable: {e}")

if __name__ == '__main__':
    sys.exit(main())
