"""
Fig 5  --  Lead-1 fidelity (thesis Fig 19 + Fig 20, §4.5.2 and §4.5.3)

a-c  Normalised Taylor diagrams, one per timescale: standard deviation
     divided by the observed SD (population SD, as in the Taylor table), correlation
     on the angle, centred RMSE (normalised) as grey arcs. Each model: 5-seed mean
     (large marker) and the 5 seeds (small dots). Persistence is added as a reference
     point (black X); its two numbers are computed from the prediction files and
     written to the check log.
d-r  Predicted vs observed scatter, rows SPEI-3/6/12, columns the five models in
     fixed order; each panel shows the model's median-skill seed (C8), with the 1:1
     line and the least-squares line (both black). The R2 printed in each panel is
     that seed's value from Table S14 (= squared Pearson r; checked against the file).

Writes Fig05 (combined) and the split pair Fig05a (Taylor) / Fig05b (scatter).

Sources: Taylor_stats_summary.csv, Taylor_stats_per_seed.csv, Median_skill_seed.csv,
pred_<model>_SPEI<ts>_lead1_seed<s>.csv, Table 2 (persistence R2 cross-check).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import journal_style as js
import fig_data as D

R_MIN = 0.6                      # lowest correlation shown on the Taylor diagrams
TH_MAX = np.arccos(R_MIN)
S_MAX = 1.2                      # largest normalised SD shown
CORR_TICKS = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
LIM = (-2.75, 2.75)              # common scatter limits (all data within -2.46..2.57)


# ---------------------------------------------------------------------------
def persistence_taylor(log):
    """SD ratio and r of persistence per timescale (lead 1), from the prediction files."""
    s11 = D.table("Table2_lead1_all_models.csv")
    tay = D.table("Fig05_taylor_stats_summary.csv")
    out = {}
    for ts in D.TIMESCALES:
        d = D.persistence_series(log, ts, 1)
        sd_o = d.observed.std(ddof=0)
        sd_p = d.persistence.std(ddof=0)
        r = np.corrcoef(d.observed, d.persistence)[0, 1]
        sd_o_tab = float(tay[tay.timescale == ts].SD_obs_mean.iloc[0])
        r2_tab = float(s11[(s11.timescale == ts) & (s11.model == "Persistence")].R2_mean.iloc[0])
        if abs(sd_o - sd_o_tab) > 1e-5:
            log.fail(f"SPEI-{ts}: observed SD {sd_o:.6f} vs Taylor table {sd_o_tab:.6f}")
        if abs(r ** 2 - r2_tab) > 1e-5:
            log.fail(f"SPEI-{ts}: persistence r^2 {r ** 2:.6f} vs Table 2 R2 {r2_tab:.6f}")
        out[ts] = (sd_p / sd_o, r)
        log.note(f"NEW NUMBER (Fig 5 persistence point) SPEI-{ts}: normalised SD = "
                 f"{sd_p / sd_o:.3f}, r = {r:.3f}  (r^2 = Table 2 persistence R2 {r2_tab:.3f}; "
                 f"observed SD = Taylor table {sd_o_tab:.3f})")
    return out


def checks(log):
    tay_s = D.table("Fig05_taylor_stats_summary.csv")
    tay_p = D.table("Fig05_taylor_stats_per_seed.csv")
    D.check_per_seed_against(log, tay_p, tay_s, ["model", "timescale"],
                             [("SD_ratio", "SD_ratio_mean", "SD_ratio_sd"),
                              ("r", "r_mean", "r_sd")], "Taylor_stats_summary")
    D.check_median_seed_files(log, D.MODELS, lead=1)
    s15 = D.table("TableS14_per_seed_runs.csv")
    worst = 0.0
    for ts in D.TIMESCALES:
        for m in D.MODELS:
            sd = D.median_seed(m, ts)
            d = D.pred(m, ts, 1, sd)
            r_file = np.corrcoef(d.observed, d.predicted)[0, 1]
            r2_tab = float(s15[(s15.model == m) & (s15.timescale == ts) & (s15.lead == 1)
                               & (s15.seed == sd)].R2.iloc[0])
            worst = max(worst, abs(r_file ** 2 - r2_tab))
            if abs(r_file ** 2 - r2_tab) > 1e-5:
                log.fail(f"{m} SPEI-{ts} seed {sd}: r^2 from file {r_file ** 2:.6f} vs S15 R2 {r2_tab:.6f}")
    log.ok(f"scatter R2 labels: the R2 of each median-seed run in Table S14 equals r^2 "
           f"from its prediction file (max |diff| {worst:.1e}), so R2 here is the squared "
           f"Pearson correlation")


# ---------------------------------------------------------------------------
def _taylor_frame(ax):
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(TH_MAX))
    ax.set_rlim(0, S_MAX)
    ax.set_xticks(np.arccos(CORR_TICKS))
    ax.set_xticklabels([f"{c:g}" for c in CORR_TICKS])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.tick_params(axis="x", pad=1.5)
    ax.tick_params(axis="y", pad=1.0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color(js.INK)
    # correlation rays
    for c in CORR_TICKS:
        ax.plot([np.arccos(c)] * 2, [0, S_MAX], color="#E3E3E3", lw=0.5, zorder=0)
    # SD arcs (observed SD = 1 dashed)
    th = np.linspace(0, TH_MAX, 200)
    for s in (0.25, 0.5, 0.75):
        ax.plot(th, np.full_like(th, s), color="#E3E3E3", lw=0.5, zorder=0)
    ax.plot(th, np.ones_like(th), color=js.GREY, lw=0.8, ls=(0, (3, 1.6)), zorder=1)
    # centred RMSE arcs around the reference point (1, 0)
    T, R = np.meshgrid(np.linspace(0, TH_MAX, 300), np.linspace(0, S_MAX, 300))
    E = np.sqrt(1 + R ** 2 - 2 * R * np.cos(T))
    ax.contour(T, R, E, levels=[0.25, 0.5, 0.75], colors=js.GREY,
               linewidths=0.6, linestyles="dotted", zorder=1)
    # label each centred-RMSE arc at 150 deg around the reference point, away from
    # the SD tick labels on the baseline and from the model clusters
    phi = np.radians(150)
    for e in (0.25, 0.5, 0.75):
        x, y = 1 + e * np.cos(phi), e * np.sin(phi)
        ax.text(np.arctan2(y, x), np.hypot(x, y), f"{e:g}", color=js.GREY, fontsize=7,
                ha="center", va="center", zorder=2,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
    # observed reference
    ax.plot(0, 1, marker="*", ms=6.5, mfc="white", mec="black", mew=0.7, ls="none",
            zorder=6, clip_on=False)
    mid = (np.arccos(0.8) + np.arccos(0.9)) / 2
    ax.text(mid, S_MAX * 1.24, "Correlation", rotation=np.degrees(mid) - 90,
            ha="center", va="center")
    ax.set_xlabel("Standard deviation (normalised)", labelpad=3)


def draw_taylor(axes, pers, log):
    tay_s = D.table("Fig05_taylor_stats_summary.csv")
    tay_p = D.table("Fig05_taylor_stats_per_seed.csv")
    for ax, ts in zip(axes, D.TIMESCALES):
        _taylor_frame(ax)
        for m in D.MODELS:
            name = D.DISPLAY[m]
            ps = tay_p[(tay_p.model == m) & (tay_p.timescale == ts)]
            ax.scatter(np.arccos(ps.r), ps.SD_ratio, s=4, color=js.MODEL_COLORS[name],
                       alpha=0.8, lw=0, zorder=3)
            r = tay_s[(tay_s.model == m) & (tay_s.timescale == ts)].iloc[0]
            ax.plot(np.arccos(r.r_mean), r.SD_ratio_mean, marker=js.MODEL_MARKERS[name],
                    ms=4.2, mfc=js.MODEL_COLORS[name], mec=js.INK, mew=0.5, ls="none", zorder=5)
        sdr, rp = pers[ts]
        ax.plot(np.arccos(rp), sdr, marker="X", ms=5, mfc="black", mec="white", mew=0.4,
                ls="none", zorder=5)
        ax.text(0.97, 0.97, f"SPEI-{ts}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, fontweight="bold")


def draw_scatter(axes_grid, log):
    s15 = D.table("TableS14_per_seed_runs.csv")
    for ri, ts in enumerate(D.TIMESCALES):
        for ci, m in enumerate(D.MODELS):
            ax = axes_grid[ri, ci]
            name = D.DISPLAY[m]
            sd = D.median_seed(m, ts)
            d = D.pred(m, ts, 1, sd)
            ax.scatter(d.observed, d.predicted, s=2.2, color=js.MODEL_COLORS[name],
                       alpha=0.45, lw=0, zorder=2)
            # reference lines in black (CEEMDAN-BiLSTM points are charcoal grey)
            ax.plot(LIM, LIM, color="black", lw=0.8, ls=(0, (3, 1.6)), zorder=3)
            b, a = np.polyfit(d.observed, d.predicted, 1)
            ax.plot(np.array(LIM), a + b * np.array(LIM), color="black", lw=1.0, zorder=4)
            r2 = float(s15[(s15.model == m) & (s15.timescale == ts) & (s15.lead == 1)
                           & (s15.seed == sd)].R2.iloc[0])
            ax.text(0.96, 0.04, f"R² = {r2:.3f}\nseed {sd}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, linespacing=1.15)
            ax.set_xlim(*LIM)
            ax.set_ylim(*LIM)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            if ri == 0:
                ax.set_title(name, fontsize=7, pad=3)
            if ri < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Observed SPEI")
            if ci > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(f"Predicted SPEI-{ts}")


def legend_handles(scatter=True, taylor=True):
    h = [Line2D([], [], marker=js.MODEL_MARKERS[D.DISPLAY[m]], ls="none", ms=4.6,
                mfc=js.MODEL_COLORS[D.DISPLAY[m]], mec=js.INK, mew=0.5, label=D.DISPLAY[m])
         for m in D.MODELS]
    if taylor:
        h += [Line2D([], [], marker="X", ls="none", ms=5, mfc="black", mec="white", mew=0.4,
                     label="Persistence"),
              Line2D([], [], marker="*", ls="none", ms=6.5, mfc="white", mec="black", mew=0.7,
                     label="Observed (reference)"),
              Line2D([], [], marker="o", ls="none", ms=2.2, mfc=js.GREY, mec="none",
                     label="Individual seeds")]
    if scatter:
        h += [Line2D([], [], color="black", lw=0.8, ls=(0, (3, 1.6)), label="1:1 line"),
              Line2D([], [], color="black", lw=1.0, label="Least-squares fit")]
    return h


# ---------------------------------------------------------------------------
def make(outdir, log):
    log.section("Fig 5 - Taylor diagrams and scatter")
    checks(log)
    pers = persistence_taylor(log)
    js.apply()
    problems = []

    # --- combined ----------------------------------------------------------
    fig = plt.figure(figsize=(js.WIDTH["double"], 194 * js.MM))
    top, bot = fig.subfigures(2, 1, height_ratios=[58, 128], hspace=0.0)
    tax = [top.add_subplot(1, 3, i + 1, projection="polar") for i in range(3)]
    draw_taylor(tax, pers, log)
    sax = bot.subplots(3, 5, sharex=True, sharey=True)
    draw_scatter(sax, log)
    js.panel_labels(tax, dx=-6, dy=-2)
    js.panel_labels(list(sax.ravel()), start="d", dx=-2, dy=2)
    fig.legend(handles=legend_handles(), loc="outside upper center", ncol=6, frameon=False)
    problems += js.save(fig, "Fig05", outdir)

    # --- split: a Taylor ----------------------------------------------------
    fig = plt.figure(figsize=(js.WIDTH["double"], 68 * js.MM))
    tax = [fig.add_subplot(1, 3, i + 1, projection="polar") for i in range(3)]
    draw_taylor(tax, pers, log)
    js.panel_labels(tax, dx=-6, dy=-2)
    fig.legend(handles=legend_handles(scatter=False), loc="outside upper center", ncol=8,
               frameon=False)
    problems += js.save(fig, "Fig05a", outdir)

    # --- split: b scatter ---------------------------------------------------
    fig = plt.figure(figsize=(js.WIDTH["double"], 132 * js.MM))
    sax = fig.subplots(3, 5, sharex=True, sharey=True)
    draw_scatter(sax, log)
    js.panel_labels(list(sax.ravel()), start="a", dx=-2, dy=2)
    fig.legend(handles=legend_handles(taylor=False), loc="outside upper center", ncol=7,
               frameon=False)
    problems += js.save(fig, "Fig05b", outdir)
    return problems


if __name__ == "__main__":
    import p1_fig_config as C
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
