#!/usr/bin/env python
"""Turn the two Crafter calibration runs into games/crafter/calibration.json.

  python -m tools.balrog_pae.games.crafter.calibration \
      --base /root/nld/gen_runs/crafter/cal_s0_base \
      --pae  /root/nld/gen_runs/crafter/cal_s0_pae3 \
      --out  tools/balrog_pae/games/crafter/calibration.json

Reports, per run: steps, LLM calls, input/output tokens per step (mean, median,
first/last), list cost, the provider's own summed ``usage.cost``, progression
and the achievements behind it. For the PAE run it additionally reports whether
PAE kept the agent alive longer than the base episode (the budget risk in
tasks.md section 7).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def trace_rows(run: Path) -> list[dict]:
    rows = []
    for f in sorted(run.glob("attempts/*/trace.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def per_attempt(run: Path) -> list[dict]:
    p = run / "attempts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def tok_stats(rows: list[dict], key: str) -> dict:
    xs = [r[key] for r in rows]
    if not xs:
        return {}
    return {"mean": round(st.fmean(xs)), "median": round(st.median(xs)),
            "min": min(xs), "max": max(xs), "first": xs[0], "last": xs[-1]}


def describe(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    rows = trace_rows(run)
    atts = per_attempt(run)
    ach = None
    # achievements behind the final progression, from the last attempt's summary line
    for a in reversed(atts):
        if a.get("aux_measured") is not None:
            ach = int(a["aux_measured"])
            break
    out = {
        "run_dir": str(run),
        "arm": s["arm"],
        "seed": s["seed"],
        "env_patches": s["env_patches"],
        "attempts": s["attempts"],
        "checkpoints": s["checkpoints"],
        "committed_steps_max": s["committed_steps_max"],
        "total_env_steps": s["total_env_steps"],
        "llm_steps": s["llm_steps"],
        "attempt1_progression": s["attempt1_progression"],
        "pae_best_progression": s["pae_best_progression"],
        "achievements_unlocked": ach,
        "achievements_of": 22,
        "stop_reason": s["stop_reason"],
        "input_tokens_total": s["tokens"]["total"]["input_tokens"],
        "output_tokens_total": s["tokens"]["total"]["output_tokens"],
        "list_usd": s["tokens"]["total"]["cost_usd"],
        "billed_usd": s["tokens"].get("billed_by_provider_usd"),
        "billed_over_list": (round(s["tokens"]["billed_by_provider_usd"] / s["tokens"]["total"]["cost_usd"], 4)
                             if s["tokens"]["total"]["cost_usd"] else None),
        "input_tokens_per_step": tok_stats(rows, "input_tokens"),
        "output_tokens_per_step": tok_stats(rows, "output_tokens"),
        "invalid_action_rate": (round(sum(1 for r in rows if not r["valid"]) / len(rows), 4) if rows else None),
        "wall_s": s["wall_s"],
        "wall_s_per_llm_step": round(s["wall_s"] / max(1, s["llm_steps"]), 2),
        "attempts_detail": [
            {k: a[k] for k in ("attempt", "from_checkpoint", "start_step", "end_step", "steps_played",
                               "committed_steps", "outcome", "progression", "aux_measured", "calls",
                               "input_tokens", "output_tokens", "directive_kind", "selection_source")}
            for a in atts
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pae", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base, pae = describe(Path(a.base)), describe(Path(a.pae))
    base_end = base["attempts_detail"][0]["end_step"] if base["attempts_detail"] else base["committed_steps_max"]
    pae_ends = [d["end_step"] for d in pae["attempts_detail"]]
    res = {
        "what": "Crafter calibration: 1 base episode + 1 three-attempt PAE run, seed 0, GLM-5.2 over Prime",
        "base": base,
        "pae": pae,
        "pae_vs_base": {
            "base_episode_end_step": base_end,
            "base_outcome": base["attempts_detail"][0]["outcome"] if base["attempts_detail"] else None,
            "pae_attempt_end_steps": pae_ends,
            "pae_attempt_outcomes": [d["outcome"] for d in pae["attempts_detail"]],
            "pae_kept_agent_alive_longer": bool(pae_ends and max(pae_ends) > base_end),
            "longest_committed_trajectory": max(pae_ends + [base_end]) if pae_ends else base_end,
            "base_progression": base["attempt1_progression"],
            "pae_attempt1_progression": pae["attempt1_progression"],
            "pae_best_progression": pae["pae_best_progression"],
            "progression_delta_pae_best_minus_base": pae["pae_best_progression"] - base["attempt1_progression"],
        },
        # the numbers games/crafter/cost_table.py consumes
        "measured": {
            "in_tok_per_step": (base["input_tokens_total"] + pae["input_tokens_total"])
                               // max(1, base["llm_steps"] + pae["llm_steps"]),
            "out_tok_per_step": (base["output_tokens_total"] + pae["output_tokens_total"])
                                // max(1, base["llm_steps"] + pae["llm_steps"]),
            "base_steps": base_end,
            "billed_ratio": round(((base["billed_usd"] or 0) + (pae["billed_usd"] or 0))
                                  / max(1e-9, base["list_usd"] + pae["list_usd"]), 4),
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("pae_vs_base", "measured")}, indent=2, default=str))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
