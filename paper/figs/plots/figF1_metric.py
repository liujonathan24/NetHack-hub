"""Is an F1 (harmonic-mean) BALROG score linear in game turns for human wins?

BALROG scores a run as max(depth percentile, experience percentile), which in our
rollouts is decided by depth every time. The alternative tested here is the
harmonic mean of the same two percentiles.

Both metrics are rebuilt from the raw (dlvl, xp) curve columns plus the plane and
ascension turns: the stored balrog_max/balrog_min columns were written without
plane or ascension credit and so stall at Dlvl:50 = 80.68% even for a win. With
the credit restored (BALROG's own rule -- the Planes and ascension are terminal
achievements maxed into both axes) every ascension ends at 100.

Panels: median curve against a straight line, the residual from that line, the
log-log slope, and per-game R^2 for both metrics on both time axes.
"""
import numpy as np
import matplotlib.pyplot as plt
from style import (HUMAN, HUMAN_TOP, BAD, GREY, labelsize, legendsize, ticksize,
                   LW, ALPHA, style_axes, save, tex)
from common import ascension_curves, f1_score, on_fraction

NAME = "figF1_metric"

F1_COL = HUMAN_TOP     # the candidate metric
MAX_COL = HUMAN        # BALROG's own metric


def _curves():
    """F1 and BALROG-max for every ascension, resampled on fraction-elapsed."""
    frac = np.linspace(0, 1, 201)
    f1, bmx, turns = [], [], []
    for ev, total in ascension_curves():
        hi = on_fraction(ev, total, frac, 1)
        lo = on_fraction(ev, total, frac, 2)
        f1.append(f1_score(hi, lo))
        bmx.append(hi)
        turns.append(total)
    return frac, np.array(f1), np.array(bmx), np.array(turns, float)


def _r2(x, y):
    p = np.polyfit(x, y, 1)
    r = y - np.polyval(p, x)
    return 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum(), p


def render():
    frac, F1, BM, turns = _curves()
    x = 100 * frac
    n = len(turns)

    med1, med_m = np.median(F1, 0), np.median(BM, 0)
    q1, q3 = np.percentile(F1, [25, 75], axis=0)
    m1, m3 = np.percentile(BM, [25, 75], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    (axA, axB), (axC, axD) = axes

    # ---- A: median curve + IQR band against a straight line ----
    axA.fill_between(x, q1, q3, color=F1_COL, alpha=.22, lw=0)
    axA.fill_between(x, m1, m3, color=MAX_COL, alpha=.16, lw=0)
    axA.plot(x, med_m, color=MAX_COL, lw=LW, alpha=ALPHA, label="BALROG max")
    axA.plot(x, med1, color=F1_COL, lw=LW, alpha=ALPHA, label="F1 (harmonic mean)")
    axA.plot([0, 100], [0, 100], color=GREY, lw=2.0, ls=":", label="perfectly linear")
    axA.plot([100], [100], marker="*", ms=22, color=BAD, zorder=5, clip_on=False)
    z = 100 * float(np.median((F1 == 0).mean(axis=1)))
    jump = float(np.median([100 - r[r < 87][-1] for r in F1]))
    axA.annotate(tex("F1 pinned at 0 for the first %.1f%% of a game" % z),
                 xy=(z, 0.6), xytext=(30, 5), fontsize=ticksize - 2, color=BAD,
                 arrowprops=dict(arrowstyle="->", color=BAD, lw=1.5))
    axA.text(54, 92, tex("ascension: F1 %.0f %s 100" % (100 - jump, "->"))
             .replace(tex("->"), r"$\rightarrow$"),
             fontsize=ticksize - 2, color=BAD)
    axA.set_xlim(0, 100)
    axA.set_ylim(0, 105)
    style_axes(axA, tex("Game elapsed (%)"), tex("Progression (%)"),
               tex("Median over %d NAO ascensions (IQR band)" % n))
    axA.legend(fontsize=legendsize - 4, loc="upper left")

    # ---- B: residual from each metric's own best straight line ----
    for y, col, lab in ((med1, F1_COL, "F1"), (med_m, MAX_COL, "BALROG max")):
        _, p = _r2(x, y)
        axB.plot(x, y - np.polyval(p, x), color=col, lw=LW, alpha=ALPHA, label=lab)
    axB.axhline(0, color=GREY, lw=2.0, ls=":")
    axB.set_xlim(0, 100)
    axB.set_ylim(-13, 34)
    style_axes(axB, tex("Game elapsed (%)"), tex("Residual (pct. points)"),
               tex("Departure from the fitted straight line"))
    axB.legend(fontsize=legendsize - 4, loc="upper center")

    # ---- C: log-log, power-law exponent ----
    tmed = frac * float(np.median(turns))
    ks = []
    for row, tot in zip(F1, turns):
        tt = frac * tot
        s_ = (tt > 0) & (row > 0)
        axC.plot(tt[s_], row[s_], color=F1_COL, lw=1.0, alpha=.28)
        ks.append(np.polyfit(np.log10(tt[s_]), np.log10(row[s_]), 1)[0])
    ks = np.array(ks)
    axC.plot(tmed[med1 > 0], med1[med1 > 0], color=F1_COL, lw=LW, alpha=ALPHA,
             label="F1 (median)")
    good = (tmed > 0) & (med1 > 0)
    kk, cc = np.polyfit(np.log10(tmed[good]), np.log10(med1[good]), 1)
    tl = np.logspace(np.log10(tmed[good][0]), np.log10(tmed.max()), 50)
    axC.plot(tl, 10 ** (cc + kk * np.log10(tl)), color=BAD, lw=LW, ls="--",
             label=r"$\propto t^{k}$, $k = %.2f$" % kk)
    axC.plot(tl, 100 * tl / tmed.max(), color=GREY, lw=2.0, ls=":",
             label=r"$k = 1$ (linear)")
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlim(150, 6e4)
    axC.set_ylim(0.5, 220)
    style_axes(axC, tex("Game turns"), tex("F1 (%)"),
               tex("Power law: per-game k = %.2f (IQR %.2f-%.2f)"
                   % (np.median(ks), *np.percentile(ks, [25, 75]))))
    axC.legend(fontsize=legendsize - 4, loc="upper left")

    # ---- D: per-game R^2, both metrics, both axes ----
    stats = []
    for M in (F1, BM):
        lin, log = [], []
        for row, tot in zip(M, turns):
            tt = frac * tot
            lin.append(_r2(tt, row)[0])
            log.append(_r2(np.log10(tt[tt > 0]), row[tt > 0])[0])
        stats.append((np.array(lin), np.array(log)))
    labels = [tex("linear turns\n(= % elapsed)"), tex("log turns")]
    w, pos = 0.34, np.arange(2)
    for i, (col, lab) in enumerate(((F1_COL, "F1"), (MAX_COL, "BALROG max"))):
        vals = [np.median(v) for v in stats[i]]
        err = np.array([[np.median(v) - np.percentile(v, 25) for v in stats[i]],
                        [np.percentile(v, 75) - np.median(v) for v in stats[i]]])
        b = axD.bar(pos + (i - .5) * w, vals, w, color=col, alpha=ALPHA, label=lab)
        axD.errorbar(pos + (i - .5) * w, vals, yerr=err, fmt="none",
                     ecolor=GREY, capsize=5, lw=2)
        for r, v in zip(b, vals):
            axD.text(r.get_x() + r.get_width() / 2, v + .035, "%.3f" % v,
                     ha="center", fontsize=ticksize - 2)
    axD.set_xticks(pos)
    axD.set_xticklabels(labels)
    axD.set_ylim(0, 1.12)
    style_axes(axD, None, r"$R^{2}$ per game (median, IQR)",
               tex("Fit quality, %d games" % n))
    axD.legend(fontsize=legendsize - 4, loc="upper right")

    fig.tight_layout()
    save(fig, NAME)
