#!/usr/bin/env python
"""Cost table for the full MiniHack panel: 5 tasks x 5 seeds x {base, PAE N}.

Built ONLY from numbers measured on this branch (the calibration runs under
/root/nld/gen_runs/minihack), never from a price guess:

  base arm   per seed = S_base LLM calls at the base episode's own measured
             input/output tokens per call.
  PAE arm    per seed = attempt 1 (== a base episode) + (N-1) resumed attempts.
             A resumed attempt costs S_resume calls at the *resumed* attempt's
             own measured tokens per call - which is higher on input, because a
             resume starts with a full 16-observation window while a fresh
             episode spends its first 16 steps on short prompts.
             Plus (N-1) orchestrator rounds at the measured orchestrator cost
             per round.

Reported at list price and at the billed ratio measured here (Prime returns the
true billed cost per completion in usage.cost; see MEMORY: a ledger built from
a price table overstates the wallet).

  python -m tools.balrog_pae.games.minihack.cost_table --runs /root/nld/gen_runs/minihack -N 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRICE_IN, PRICE_OUT = 1.54, 4.84   # results/model_prices.json, z-ai/glm-5.2

TASKS = [
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
    "MiniHack-CorridorBattle-Dark-v0",
    "MiniHack-Boxoban-Medium-v0",
    "MiniHack-Boxoban-Hard-v0",
]
CAP = 100        # BALROG's MiniHack step cap (see tasks.md)


def load(run_dir: Path):
    s = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(x) for x in (run_dir / "attempts.jsonl").read_text().splitlines() if x.strip()]
    return s, rows


def attempt_from_trace(run_dir: Path, n: int = 1):
    """Rebuild one attempt's totals from its step trace.

    A run that crashed (Boxoban does, see games/minihack/tasks.md) never wrote
    summary.json or attempts.jsonl, but attempts/aNNN/trace.jsonl is complete up
    to the crash and carries per-step input/output tokens - which is all the
    projection needs. Rows built this way are marked ``partial``.
    """
    f = run_dir / "attempts" / f"a{n:03d}" / "trace.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if not rows:
        return None
    return {
        "calls": len(rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "partial": True,
    }


def per_task(runs: Path, task: str, N: int, resume_model: str = "fixed_rule"):
    base_dir, pae_dir = runs / f"{task}_s0_base", runs / f"{task}_s0_pae"
    partial = False
    if (base_dir / "summary.json").exists():
        bs, brows = load(base_dir)
        a1 = brows[0]
    else:
        bs, a1 = {}, attempt_from_trace(base_dir)
        if a1 is None:
            return None
        partial = True
    S_base = a1["calls"]
    in_base = a1["input_tokens"] / max(1, S_base)
    out_base = a1["output_tokens"] / max(1, S_base)

    src_resume = "measured"
    if (pae_dir / "summary.json").exists():
        ps, prows = load(pae_dir)
        res = [r for r in prows if r["from_checkpoint"]]
    else:
        ps, res = None, []
        a2 = attempt_from_trace(pae_dir, 2)
        if a2:
            res = [a2]
            ps = {"tokens": {}}
            partial = True
    if res:
        S_res = sum(r["calls"] for r in res) / len(res)
        in_res = sum(r["input_tokens"] for r in res) / max(1, sum(r["calls"] for r in res))
        out_res = sum(r["output_tokens"] for r in res) / max(1, sum(r["calls"] for r in res))
        orch = (ps.get("tokens") or {}).get("orchestrator", {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        rounds = max(1, len(res))
        orch_in, orch_out = orch["input_tokens"] / rounds, orch["output_tokens"] / rounds
    else:
        # fall back to the fixed rule's own geometry: resume from the last
        # periodic checkpoint of a base episode of length S_base.
        src_resume = "modelled"
        ckpt = (max(0, S_base - 1) // 10) * 10
        S_res = max(1, CAP - ckpt)
        in_res, out_res = in_base * 1.25, out_base
        orch_in = orch_out = 0.0
    # How long a resumed attempt runs in the REAL panel (the calibration PAE
    # smokes are deliberately short, so their attempt length is not the number
    # to project with; their tokens/call is).
    ckpt = (max(0, S_base - 1) // 10) * 10
    if resume_model == "fixed_rule":
        # the fixed rule resumes the latest resumable checkpoint of the
        # previous attempt, i.e. the last periodic one before it ended
        S_res = max(1, CAP - ckpt)
    elif resume_model == "worst":
        # an orchestrator that branches at the very start: a full episode
        S_res = CAP
    elif resume_model == "measured":
        pass          # whatever the calibration run actually played
    else:
        raise ValueError(resume_model)

    base = {"calls": S_base, "in": S_base * in_base, "out": S_base * out_base}
    pae_calls = S_base + (N - 1) * S_res
    pae = {
        "calls": pae_calls,
        "in": S_base * in_base + (N - 1) * (S_res * in_res + orch_in),
        "out": S_base * out_base + (N - 1) * (S_res * out_res + orch_out),
    }
    return {
        "task": task, "S_base": S_base, "in_per_step_base": in_base, "out_per_step_base": out_base,
        "S_resume": S_res, "in_per_step_resume": in_res, "out_per_step_resume": out_res,
        "orch_in_per_round": orch_in, "orch_out_per_round": orch_out,
        "resume_source": src_resume, "base": base, "pae": pae, "partial": partial,
        "base_progression": bs.get("attempt1_progression"),
    }


def billed_ratio(runs: Path) -> tuple[float, float, float]:
    lst = bil = 0.0
    for f in sorted(runs.rglob("summary.json")):
        s = json.loads(f.read_text())
        if s.get("game") != "minihack":
            continue
        lst += s["tokens"]["total"]["cost_usd"]
        bil += s["tokens"].get("billed_by_provider_usd", 0.0) or 0.0
    return lst, bil, (bil / lst if lst else float("nan"))


def usd(i, o):
    return i / 1e6 * PRICE_IN + o / 1e6 * PRICE_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/nld/gen_runs/minihack")
    ap.add_argument("-N", "--attempts", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=None, help="override the measured billed ratio")
    ap.add_argument("--resume-model", default="fixed_rule", choices=["fixed_rule", "worst", "measured"],
                    help="how many steps a resumed attempt plays in the real panel: "
                         "fixed_rule = cap - last periodic checkpoint of a base-length episode; "
                         "worst = the full cap (orchestrator branches at the start); "
                         "measured = whatever the short calibration run played")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    runs = Path(a.runs)
    lst, bil, ratio = billed_ratio(runs)
    if a.ratio:
        ratio = a.ratio

    rows = [per_task(runs, t, a.attempts, a.resume_model) for t in TASKS]
    rows = [r for r in rows if r is not None]

    print(f"Calibration source: {runs}   seeds/task: {a.seeds}   PAE attempts N={a.attempts}"
          f"   resume model: {a.resume_model}")
    print(f"List price: ${PRICE_IN}/M in, ${PRICE_OUT}/M out. "
          f"Measured billed ratio: {ratio:.3f}  (list ${lst:.2f} -> billed ${bil:.2f} over the calibration runs)\n")

    h = (f"{'task':<32}{'S_base':>7}{'in/st':>7}{'out/st':>7}{'S_res':>7}{'in/st*':>7}"
         f"{'base $':>9}{'PAE $':>9}{'both $':>9}{'billed $':>10}")
    print(h)
    print("-" * len(h))
    tb = tp = 0.0
    out_rows = []
    for r in rows:
        b = usd(r["base"]["in"], r["base"]["out"]) * a.seeds
        p = usd(r["pae"]["in"], r["pae"]["out"]) * a.seeds
        tb += b
        tp += p
        print(f"{r['task']:<32}{r['S_base']:>7}{r['in_per_step_base']:>7.0f}{r['out_per_step_base']:>7.0f}"
              f"{r['S_resume']:>7.1f}{r['in_per_step_resume']:>7.0f}"
              f"{b:>9.2f}{p:>9.2f}{b + p:>9.2f}{(b + p) * ratio:>10.2f}"
              f"{'  partial' if r['partial'] else ''}")
        out_rows.append(dict(r, base_usd_list=b, pae_usd_list=p, both_usd_list=b + p,
                             both_usd_billed=(b + p) * ratio, seeds=a.seeds, N=a.attempts))
    print("-" * len(h))
    print(f"{'TOTAL (list)':<32}{'':>7}{'':>7}{'':>7}{'':>7}{'':>7}{tb:>9.2f}{tp:>9.2f}{tb + tp:>9.2f}"
          f"{(tb + tp) * ratio:>10.2f}")
    print(f"{'TOTAL (at measured billed ratio)':<32}{'':>7}{'':>7}{'':>7}{'':>7}{'':>7}"
          f"{tb * ratio:>9.2f}{tp * ratio:>9.2f}{(tb + tp) * ratio:>9.2f}")
    print("\n* in/st for a RESUMED attempt: higher than base because a resume starts with a")
    print("  full 16-observation window. 'modelled' rows had no PAE calibration run.")
    mod = [r["task"] for r in rows if r["resume_source"] == "modelled"]
    if mod:
        print("  modelled:", ", ".join(mod))
    par = [r["task"] for r in rows if r["partial"]]
    if par:
        print("  partial (run crashed; tokens/step taken from the step trace):", ", ".join(par))
    missing = [t for t in TASKS if t not in {r["task"] for r in rows}]
    if missing:
        print("  NO CALIBRATION DATA (excluded from the totals):", ", ".join(missing))
        print("  -> for a panel estimate, assume each behaves like Boxoban-Medium.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"price_in": PRICE_IN, "price_out": PRICE_OUT, "billed_ratio": ratio,
             "calibration_list_usd": lst, "calibration_billed_usd": bil,
             "seeds": a.seeds, "attempts": a.attempts, "resume_model": a.resume_model, "tasks": out_rows,
             "total_base_usd_list": tb, "total_pae_usd_list": tp,
             "total_usd_list": tb + tp, "total_usd_billed": (tb + tp) * ratio},
            indent=2, default=str))


if __name__ == "__main__":
    main()
