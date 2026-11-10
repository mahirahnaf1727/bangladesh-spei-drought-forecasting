"""
Fig S9   --  Meteorological SHAP shares (SI).
The beeswarm helpers here (load_npz, swarm_offsets, make_s10) are used by
fig10_shap.py for panels d-f of Fig. 10; make() builds Fig. S9 only.

S10: one beeswarm per selected model, median-skill seed (CNN-BiLSTM 44, Bi-LSTM 42,
     CNN 44). Each dot is one test division-month; x = signed SHAP value
     (summed over the input window); colour = the predictor's value (window mean),
     scaled per predictor between its 5th and 95th percentiles, viridis. Same row order as Fig. 10 and the same x-limits in all panels.
S11: share of each meteorological predictor in the meteorological mean |SHAP| of each
     selected model (lagged SPEI excluded), mean +/- SD over the 5 seeds, viridis.
     The thesis colour bar read "Mean SHAP Value"; the cells are shares (%), so the
     label is corrected here.

Sources: _shap/shap_values_<model>_SPEI<ts>_seed<s>.npz, TableS17_shap_summary.csv,
TableS17_shap_per_seed.csv.
"""

import numpy as np
import matplotlib.pyplot as plt

import journal_style as js
import fig_data as D
import p1_fig_config as C
from shap_bars import LABEL, TITLE, row_order

CMAP = "viridis"


def load_npz(model, ts, seed):
    p = C.OUT_ROOT / "_shap" / f"shap_values_{model}_SPEI{ts}_seed{seed}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    z = np.load(p, allow_pickle=True)
    return z["shap_2d"], z["feat_2d"], [str(x) for x in z["labels"]]


def swarm_offsets(v, nbins=100, rng=None):
    """Vertical offsets for a beeswarm row (same idea as the shap package)."""
    rng = rng or np.random.default_rng(0)
    n = len(v)
    q = np.round(nbins * (v - v.min()) / (v.max() - v.min() + 1e-12))
    order = np.lexsort((rng.random(n), q))
    ys = np.zeros(n)
    layer, last = 0, None
    for i in order:
        if q[i] != last:
            layer = 0
        ys[i] = np.ceil(layer / 2) * (1 if layer % 2 else -1)
        layer += 1
        last = q[i]
    return ys / (np.abs(ys).max() + 1)


def checks(log, summ, per):
    worst = 0.0
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        sd = D.median_seed(m, ts)
        sv, fv, labs = load_npz(m, ts, sd)
        if sv.shape != (576, 8):
            log.fail(f"{m} SPEI-{ts} seed {sd}: SHAP array shape {sv.shape}")
        q = per[(per.model == m) & (per.timescale == ts) & (per.seed == sd)].set_index("variable")
        for j, lab in enumerate(labs):
            worst = max(worst, abs(np.abs(sv[:, j]).mean() - q.loc[lab, "mean_abs_shap"]),
                        abs(sv[:, j].mean() - q.loc[lab, "mean_signed_shap"]))
    if worst > 1e-5:
        log.fail(f"beeswarm SHAP values differ from Table S17 per-seed (max {worst})")
    log.ok(f"beeswarms: mean |SHAP| and mean signed SHAP of the median-seed files equal Table S17 "
           f"(max |diff| {worst:.1e}); 576 test division-months x 8 predictors each")
    met = per[per.variable != "SPEI"]
    tot = met.groupby(["model", "timescale", "seed"]).met_share_pct.sum()
    if (tot - 100).abs().max() > 0.01:
        log.fail("meteorological shares do not sum to 100% per seed")
    log.ok(f"S11: meteorological shares sum to 100% in every seed (max deviation "
           f"{(tot - 100).abs().max():.1e}); mean/SD checked against Table S17")


def make_s10(outdir, log, order):
    js.apply()
    data = {}
    xmax = 0.0
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        sd = D.median_seed(m, ts)
        sv, fv, labs = load_npz(m, ts, sd)
        data[ts] = (m, sd, sv, fv, labs)
        xmax = max(xmax, np.abs(sv).max())
    xmax = np.ceil(xmax * 1.05 * 10) / 10
    fig, axes = plt.subplots(1, 3, figsize=(js.WIDTH["double"], 80 * js.MM), sharey=True)
    y0 = np.arange(len(order))[::-1]
    rng = np.random.default_rng(42)
    for ax, ts in zip(axes, D.TIMESCALES):
        m, sd, sv, fv, labs = data[ts]
        ax.axvline(0, color=js.GREY, lw=0.6, zorder=1)
        for yi, v in zip(y0, order):
            j = labs.index(v)
            s, f = sv[:, j], fv[:, j]
            lo, hi = np.nanpercentile(f, [5, 95])
            c = np.clip((f - lo) / (hi - lo + 1e-12), 0, 1)
            yy = yi + 0.38 * swarm_offsets(s, rng=rng)
            k = np.argsort(np.abs(s))                       # big effects drawn last
            sc = ax.scatter(s[k], yy[k], c=c[k], cmap=CMAP, vmin=0, vmax=1, s=2.2, lw=0,
                            alpha=0.9, zorder=2, rasterized=False)
        ax.set_yticks(y0)
        ax.set_yticklabels([LABEL[v] for v in order])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(f"{TITLE[ts]} (seed {sd})", fontweight="bold", pad=4)
        ax.set_xlabel("SHAP value (signed)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01, aspect=30, ticks=[0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Predictor value", labelpad=-4)
    cb.outline.set_linewidth(0.5)
    js.panel_labels([axes[0]], dx=-96, dy=4)
    js.panel_labels(list(axes[1:]), start="b", dx=-14, dy=4)
    return js.save(fig, "FigS10", outdir)


def make_s11(outdir, log, summ, order):
    js.apply()
    met = [v for v in order if v != "SPEI"]
    M = np.zeros((3, len(met)))
    S = np.zeros_like(M)
    for i, ts in enumerate(D.TIMESCALES):
        s = summ[(summ.model == D.SELECTED[ts]) & (summ.timescale == ts)].set_index("variable")
        M[i] = [s.loc[v, "met_share_pct_mean"] for v in met]
        S[i] = [s.loc[v, "met_share_pct_sd"] for v in met]
    fig, ax = plt.subplots(figsize=(js.WIDTH["double"], 58 * js.MM))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=np.ceil(M.max() / 5) * 5, aspect="auto")
    cmap = plt.get_cmap(CMAP)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r, g, b, _ = cmap(M[i, j] / im.norm.vmax)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{M[i, j]:.1f} ± {S[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="black" if lum > 0.5 else "white")
    ax.set_xticks(range(len(met)))
    two = {"Total cloud cover": "Total cloud\ncover"}
    ax.set_xticklabels([two.get(LABEL[v], LABEL[v].replace(" ", "\n", 1)) for v in met])
    ax.set_yticks(range(3))
    ax.set_yticklabels([TITLE[t] for t in D.TIMESCALES])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for x in np.arange(-0.5, len(met), 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(-0.5, 3, 1):
        ax.axhline(y, color="white", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.set_label("Share of meteorological\nmean |SHAP| (%)")
    cb.outline.set_linewidth(0.5)
    return js.save(fig, "FigS09", outdir)


def make(outdir, log):
    log.section("Fig S9 - meteorological SHAP shares (SI)")
    summ = D.table("TableS17_shap_summary.csv")
    per = D.table("TableS17_shap_per_seed.csv")
    checks(log, summ, per)
    order = row_order(summ)
    return make_s11(outdir, log, summ, order)


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
