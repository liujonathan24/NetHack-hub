"""One Pareto plot for the BALROG transfer: suites weighted equally.

Progression: mean of the per-suite means, each suite weight 1 (NetHack BAL % / 100).
Cost: SUM of per-suite costs, each normalised to the leaderboard's episode count for
that suite (MiniHack 40 = 8 tasks x 5, TextWorld 30, Crafter 10, NetHack 5), so no
arm is charged for more or fewer episodes than the others. Mean cost would be the
sum / n_suites for every arm -- the same plot on a shifted log axis.
Control is omitted: it has no NetHack arm and ran only two MiniHack tasks.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from style import labelsize, ticksize, style_axes, save
from plots.fig_balrog_pareto import scatter_panel, SRC

OUTDIR = "/root/overleaf/final-drafting/v33-claude/images/proposed"


def build():
    D = json.load(open(f"{SRC}/fig3/suites.json"))
    P = {r["game"]: r for r in json.load(open(f"{SRC}/pergame.json"))}
    suites = {"mh": "MiniHack (8-task mean, Boxoban=0)", "tw": "TextWorld (3-game mean)",
              "cr": "Crafter (10 seeds)", "nmax": "NetHack (BALmax %)", "nmin": "NetHack (BALmin %)"}
    arms = {}
    for s, key in suites.items():
        for p in D[key]["pts"]:
            if p["kind"] == "ctl":
                continue
            name = p["label"] if p["kind"] == "lb" else p["kind"]
            a = arms.setdefault(name, {"label": p["label"] if p["kind"] == "lb" else name, "kind": p["kind"]})
            prog = p["prog"] / 100 if s.startswith("n") else p["prog"]
            cost = p["cost"]
            if s.startswith("n"):          # normalise to 5 NetHack episodes
                cost = cost * 5 / p["eps"]
            a[s] = (prog, cost)
    # our MiniHack: per-run cost x 5 seeds per task we ran, + Boxoban-Medium partial
    mh = [g for g in P if g.startswith("MiniHack") and P[g].get("n")]
    for kind, bx in (("base", 0.0), ("pae", 2.73)):
        c = sum(P[g][f"{kind}_cost_per_run"] * 5 for g in mh) + bx
        arms[kind]["mh"] = (arms[kind]["mh"][0], c)
    out = {}
    for nm, sel in (("all4_balmax", ["mh", "tw", "cr", "nmax"]), ("all4_balmin", ["mh", "tw", "cr", "nmin"]),
                    ("heldout3", ["mh", "tw", "cr"])):
        pts = []
        for a in arms.values():
            pts.append({"label": a["label"], "kind": a["kind"],
                        "prog": float(np.mean([a[s][0] for s in sel])),
                        "cost": float(sum(a[s][1] for s in sel))})
        out[nm] = pts
    json.dump(out, open(f"{SRC}/aggregate.json", "w"), indent=1)
    return out


def main():
    out = build()
    for nm, pts in out.items():
        print(nm)
        for p in sorted(pts, key=lambda q: -q["prog"]):
            print(f"   {p['label'][:32]:32s} prog {p['prog']:.3f}  cost ${p['cost']:8.2f}")
    titles = {"all4_balmax": r"All four (NetHack BAL$_{\max}$)",
              "all4_balmin": r"All four (NetHack BAL$_{\min}$)",
              "heldout3": "Held out: MiniHack, TextWorld, Crafter"}
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.2))
    for i, (ax, nm) in enumerate(zip(axs, titles)):
        ymax = max(p["prog"] for p in out[nm])
        scatter_panel(ax, out[nm], titles[nm], "Mean progression (suites equal weight)" if i == 0 else None,
                      fs=12, pct=True)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_xlabel("Total cost, summed over suites (USD, log)", fontsize=labelsize - 5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_balrog_aggregate", outdir=OUTDIR)


if __name__ == "__main__":
    main()
