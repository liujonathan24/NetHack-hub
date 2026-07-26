"""Summarise a calibration/smoke run: turns, depth, BALROG %, tokens, cost.

Depth and XL come from the per-turn traces (authoritative game state); turn
counts and token usage come from `results.jsonl`. BALROG progression is the
real scorer (`nethack_harness.prompt.balrog.balrog_progress`), validated
against the published DL10/XL6 = 12.56% anchor — not the older smooth proxy.

Death is derived from `hitpoints == 0` in the trace, never from a `died`
metric: a prior rollout sat on a tombstone screen for 7 of its 12 calls while
reporting `died = 0`.

Usage:
  PYTHONPATH=<engine>:.:environments/nethack \
    .venv-cli-eval/bin/python tools/encoding_eval/aggregate_calib.py <RUN_DIR>
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "environments", "nethack")
)
from nethack_harness.prompt.balrog import balrog_progress  # noqa: E402


def _trace_stats(path: str) -> dict:
    """max dlvl, max XL, death, turn count and wall-clock span for one rollout."""
    md, mx, died, n = 1, 1, False, 0
    t_first = t_last = None
    for line in open(path):
        try:
            t = json.loads(line)
        except ValueError:
            continue
        n += 1
        md = max(md, int(t.get("max_dlvl_reached") or t.get("dlvl") or 1))
        st = t.get("status") or {}
        if isinstance(st, dict):
            mx = max(mx, int(st.get("experience_level") or 1))
            if st.get("hitpoints") == 0:
                died = True
        tw = t.get("t_wall")
        if tw:
            t_first = t_first if t_first is not None else tw
            t_last = tw
    span = (t_last - t_first) if (t_first and t_last) else None
    return {"turns": n, "max_dlvl": md, "max_xl": mx, "died": died, "span_s": span}


def _results_rows(run_dir: str) -> dict:
    """example_id -> {turns, tool_calls, in_tok, out_tok, truncated, error}."""
    out: dict = {}
    for f in glob.glob(os.path.join(run_dir, "**", "results.jsonl"), recursive=True):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("metrics") or {}
            tu = r.get("token_usage") or {}
            out[r.get("example_id")] = {
                "turns": int(m.get("num_turns") or 0),
                "tool_calls": int(m.get("total_tool_calls") or 0),
                "in_tok": float(tu.get("input_tokens") or 0),
                "out_tok": float(tu.get("output_tokens") or 0),
                "truncated": bool(r.get("is_truncated")),
                "error": bool(r.get("error")),
            }
    return out


def main(run_dir: str) -> None:
    traces = sorted(glob.glob(os.path.join(run_dir, "trace", "*.ndjson")))
    res = _results_rows(run_dir)

    print(f"{'seed':>4} {'turns':>6} {'dlvl':>5} {'XL':>3} {'BALROG%':>8} "
          f"{'died':>5} {'trunc':>6} {'in_tok':>10} {'tok/turn':>9} {'mins':>6}")
    tot_in = tot_out = 0.0
    rows = []
    for p in traces:
        seed = os.path.basename(p).split("_")[0]
        ts = _trace_stats(p)
        rr = res.get(int(seed), {}) if seed.isdigit() else {}
        if rr.get("error"):
            print(f"{seed:>4}  ERRORED — excluded")
            continue
        bal = 100.0 * balrog_progress(ts["max_dlvl"], ts["max_xl"])
        turns = rr.get("turns") or ts["turns"]
        itok = rr.get("in_tok", 0.0)
        tot_in += itok
        tot_out += rr.get("out_tok", 0.0)
        per = itok / turns if turns else 0.0
        mins = (ts["span_s"] / 60.0) if ts["span_s"] else 0.0
        rows.append((turns, ts["max_dlvl"], bal))
        print(f"{seed:>4} {turns:>6} {ts['max_dlvl']:>5} {ts['max_xl']:>3} "
              f"{bal:>8.2f} {str(ts['died']):>5} {str(rr.get('truncated')):>6} "
              f"{itok:>10,.0f} {per:>9,.0f} {mins:>6.1f}")

    if rows:
        n = len(rows)
        print()
        print(f"n={n}  mean turns {sum(r[0] for r in rows)/n:.1f}  "
              f"mean dlvl {sum(r[1] for r in rows)/n:.2f}  "
              f"mean BALROG {sum(r[2] for r in rows)/n:.2f}%  "
              f"max BALROG {max(r[2] for r in rows):.2f}%")
        print(f"tokens: {tot_in:,.0f} in / {tot_out:,.0f} out")
        print("cost: read the team wallet delta — "
              "GET /api/v1/billing/wallet?teamId=<id> (tokens alone cannot price "
              "cached vs uncached input)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/encoding_eval/calib_flash_b0")
