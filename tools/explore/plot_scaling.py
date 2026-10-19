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
    rows, best_max, best_min, best_dlvl, best_xl = [], 0.0, 0.0, 0, 0
    for attempt in sorted(outcomes):
        o = outcomes[attempt]
        bmax, bmin = balrog_both(o["dlvl"], o["xl"])
        best_max, best_min = max(best_max, bmax), max(best_min, bmin)
        best_dlvl, best_xl = max(best_dlvl, o["dlvl"]), max(best_xl, o["xl"])
        rows.append({**o, "balrog_max": bmax, "balrog_min": bmin, "best_balrog_max": best_max,
                     "best_balrog_min": best_min, "best_dlvl": best_dlvl, "best_xl": best_xl})
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
    panels = [
        ("best_balrog_min", 100, "BALROG progression, min (%)"),
        ("best_balrog_max", 100, "BALROG progression, max (%)"),
        ("best_dlvl", 1, "deepest Dlvl"),
        ("best_xl", 1, "highest XL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    for ax, (key, scale, title) in zip(axes.flat, panels):
        for label, rows in series.items():
            if rows:
                ax.plot([r["attempt"] + 1 for r in rows], [scale * r[key] for r in rows], marker="o", label=label)
        ax.set_title(f"{title}, best so far", fontsize=10)
        ax.grid(alpha=0.3)
        ax.xaxis.get_major_locator().set_params(integer=True)
    for ax in axes[1]:
        ax.set_xlabel("attempt")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle("Prime-Agent-Explore on NetHack, seed 1 (attempt 1 = a plain first life)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=160)
    out.with_suffix(".json").write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.with_suffix('.pdf')}, .png, .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
