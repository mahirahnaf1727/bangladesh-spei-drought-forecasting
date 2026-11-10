"""
Fig 10  --  SHAP interpretation of the selected models (Stage 6 merge of the
former Fig 9 and Fig S10)

a-c  Mean |SHAP value| of each predictor for CNN-BiLSTM (SPEI-3), Bi-LSTM (SPEI-6)
     and CNN (SPEI-12): bar = mean over the 5 seeds, whisker = +/- 1 SD, dots = seeds.
     Colour = sign of the mean signed SHAP where all 5 seeds agree (the former
     Fig09_sign version; blue wetter, red drier, grey = sign differs).
d-f  Beeswarms of the median-skill seed of each model (CNN-BiLSTM 44, Bi-LSTM 42,
     CNN 44): one dot per test division-month, x = signed SHAP summed over the input
     window, colour = window mean of the predictor scaled between its 5th and 95th
     percentiles (viridis). Same row order in all six panels.

The drawing code is taken unchanged from fig09_shap.py (draw, handles, checks) and
figS10_shap_si.py (swarm_offsets, checks); only the layout is new. Both sets of
checks against Table S17 run before anything is drawn.

Sources: TableS17_shap_summary.csv, TableS17_shap_per_seed.csv,
_shap/shap_values_<model>_SPEI<ts>_seed<s>.npz.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js
import fig_data as D
import p1_fig_config as C
import shap_bars as F9
import figS09_shap_shares as F10s


def draw_swarms(axes, order):
    data, xmax = {}, 0.0
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        sd = D.median_seed(m, ts)
        sv, fv, labs = F10s.load_npz(m, ts, sd)
        data[ts] = (m, sd, sv, fv, labs)
        xmax = max(xmax, np.abs(sv).max())
    xmax = np.ceil(xmax * 1.05 * 10) / 10
    y0 = np.arange(len(order))[::-1]
    rng = np.random.default_rng(42)
    sc = None
    for ax, ts in zip(axes, D.TIMESCALES):
        m, sd, sv, fv, labs = data[ts]
        ax.axvline(0, color=js.GREY, lw=0.6, zorder=1)
        for yi, v in zip(y0, order):
            j = labs.index(v)
            s, f = sv[:, j], fv[:, j]
            lo, hi = np.nanpercentile(f, [5, 95])
            c = np.clip((f - lo) / (hi - lo + 1e-12), 0, 1)
            yy = yi + 0.38 * F10s.swarm_offsets(s, rng=rng)
            k = np.argsort(np.abs(s))
            sc = ax.scatter(s[k], yy[k], c=c[k], cmap=F10s.CMAP, vmin=0, vmax=1, s=2.2, lw=0,
                            alpha=0.9, zorder=2)
        ax.set_yticks(y0)
        ax.set_yticklabels([F9.LABEL[v] for v in order])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(f"{F9.TITLE[ts]} (seed {sd})", fontweight="bold", pad=4)
        ax.set_xlabel("SHAP value (signed)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    return sc


def make(outdir, log, stem="Fig10"):
    log.section("Fig 10 - SHAP importance (a-c) and beeswarms (d-f)")
    summ = D.table("TableS17_shap_summary.csv")
    per = D.table("TableS17_shap_per_seed.csv")
    F9.checks(log, summ, per)
    F10s.checks(log, summ, per)
    order = F9.row_order(summ)
    log.ok("row order (all six panels): " + ", ".join(F9.LABEL[v] for v in order))
    js.apply()
    fig = plt.figure(figsize=(js.WIDTH["double"], 158 * js.MM))
    top, bot = fig.subfigures(2, 1, height_ratios=[70, 88], hspace=0.0)
    ax_t = top.subplots(1, 3, sharey=True)
    F9.draw(ax_t, summ, per, order, "sign")
    top.legend(handles=F9.handles("sign"), loc="outside upper center", ncol=3, frameon=False)
    ax_b = bot.subplots(1, 3, sharey=True)
    sc = draw_swarms(ax_b, order)
    # an invisible twin of the colour bar keeps the bar panels exactly as wide as the swarms
    ghost = top.colorbar(sc, ax=ax_t, fraction=0.025, pad=0.01, aspect=30)
    ghost.set_ticks([])
    ghost.ax.set_visible(False)
    cb = bot.colorbar(sc, ax=ax_b, fraction=0.025, pad=0.01, aspect=30, ticks=[0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Predictor value", labelpad=-4)
    cb.outline.set_linewidth(0.5)
    js.panel_labels([ax_t[0]], dx=-96, dy=4)
    js.panel_labels(list(ax_t[1:]), start="b", dx=-14, dy=4)
    js.panel_labels([ax_b[0]], start="d", dx=-96, dy=4)
    js.panel_labels(list(ax_b[1:]), start="e", dx=-14, dy=4)
    return js.save(fig, stem, outdir)


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
