"""Best score so far against attempts, one line per model.

    python tools/explore/plot_scaling.py --out figs/explore_scaling \
        "DeepSeek V4.1 Flash=/root/nld/explore-runs/m5-dsv41" \
        "GLM 5.3=/root/nld/explore-runs/m5-glm53" ...

Reads each run's ledger. An attempt's score is the paper's BALROG
progression (``balrog_both``: the max and min variants) over the attempt's
best Dlvl and XL, which for a resumed attempt includes its checkpoint's
lineage. Attempt 1 is the opener: a plain first life with no checkpoint and
no directive. Censored attempts (wall clock, crash) still count as tried and
keep whatever progress they made; the JSON beside the figure lists them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "environments" / "nethack"))


def attempt_scores(run_dir: pathlib.Path) -> list[dict]:
    from nethack_harness.prompt.balrog import balrog_both

    outcomes: dict[int, dict] = {}
    for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "outcome":
            continue
        attempt = int(record["attempt"])
        score = record.get("final_score") or {}
        best = outcomes.get(attempt)
        dlvl, xl = int(score.get("max_dlvl_reached") or 0), int(score.get("max_xp_level") or 0)
        if best is None or (dlvl, xl) > (best["dlvl"], best["xl"]):
            outcomes[attempt] = {"attempt": attempt, "dlvl": dlvl, "xl": xl, "result": record.get("result"),
                                 "reason": record.get("reason") or ""}
    rows, best_max, best_min, best_dlvl = [], 0.0, 0.0, 0
    for attempt in sorted(outcomes):
        o = outcomes[attempt]
        bmax, bmin = balrog_both(o["dlvl"], o["xl"])
        best_max, best_min, best_dlvl = max(best_max, bmax), max(best_min, bmin), max(best_dlvl, o["dlvl"])
        rows.append({**o, "balrog_max": bmax, "balrog_min": bmin,
                     "best_balrog_max": best_max, "best_balrog_min": best_min, "best_dlvl": best_dlvl})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    parser.add_argument("--out", required=True, help="output path without extension (.pdf, .png, .json)")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        series[label] = attempt_scores(pathlib.Path(path))
    fig, (ax_score, ax_dlvl) = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for label, rows in series.items():
        if not rows:
            continue
        x = [r["attempt"] + 1 for r in rows]
        line, = ax_score.plot(x, [100 * r["best_balrog_max"] for r in rows], marker="o", label=label)
        ax_score.plot(x, [100 * r["best_balrog_min"] for r in rows], marker="o", linestyle="--",
                      color=line.get_color(), alpha=0.6)
        ax_dlvl.plot(x, [r["best_dlvl"] for r in rows], marker="o", color=line.get_color())
    ax_score.set_xlabel("attempt")
    ax_score.set_ylabel("best BALROG progression so far (%)\nsolid: max, dashed: min")
    ax_dlvl.set_xlabel("attempt")
    ax_dlvl.set_ylabel("best Dlvl so far")
    for ax in (ax_score, ax_dlvl):
        ax.grid(alpha=0.3)
        ax.xaxis.get_major_locator().set_params(integer=True)
    ax_score.legend(fontsize=8, frameon=False)
    fig.suptitle("Prime-Agent-Explore on NetHack, seed 1 (attempt 1 = a plain first life)", fontsize=10)
    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=160)
    out.with_suffix(".json").write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.with_suffix('.pdf')}, .png, .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
