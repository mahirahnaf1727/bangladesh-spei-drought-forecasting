"""
Fig S6  --  Test-period series, national mean (thesis Fig 21, §4.5.4)
Fig S7  --  Test-period series, per division (thesis Fig 22, §4.5.5)

S6: mean SPEI over the 8 divisions (unweighted), Jan 2019 - Dec 2024, one row per
    timescale. Observed (black) and the lead-1 forecast of each of the 5 models from
    its median-skill seed (C8); the selected model drawn bold. Shaded band below the
    drought threshold (SPEI = -1).
S7: 8 divisions (rows) x SPEI-3/6/12 (columns): observed and the selected model's
    median-skill seed (CNN-BiLSTM 44, Bi-LSTM 42, CNN 44). Selected model only
    (the thesis also drew a runner-up, which is a tie at SPEI-3).

Sources: pred_<model>_SPEI<ts>_lead1_seed<s>.csv, Median_skill_seed.csv;
check against Table S16 (per seed) (national means).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import journal_style as js
import fig_data as D

YLIM = (-3.0, 3.0)
DROUGHT_FILL = "#F7E1E1"


def national(model, ts):
    sd = D.median_seed(model, ts)
    d = D.pred(model, ts, 1, sd)
    return d.groupby("target_date")[["observed", "predicted"]].mean(), sd


def checks(log):
    D.check_median_seed_files(log, D.MODELS, lead=1)
    sp = D.table("TableS16_spatial_cases_per_seed.csv")
    worst = 0.0
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        nat, sd = national(m, ts)
        for _, r in sp[(sp.timescale == ts) & (sp.seed == sd)].iterrows():
            t = np.datetime64(r.date)
            o, p = nat.loc[t, "observed"], nat.loc[t, "predicted"]
            worst = max(worst, abs(o - r.national_obs), abs(p - r.national_pred))
    if worst > 1e-5:
        log.fail(f"national means differ from Table S16 (per seed) (max {worst})")
    log.ok(f"national-mean series reproduce the observed and predicted national means of the "
           f"spatial-case months in Table S16 (per seed) (max |diff| {worst:.1e})")


def _style_time_axis(ax, bottom):
    ax.set_xlim(np.datetime64("2018-12-01"), np.datetime64("2025-01-01"))
    # year ticks 2019-2024 only, so the end label of one column never crowds the next
    ax.set_xticks([np.datetime64(f"{y}-01-01") for y in range(2019, 2025)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.5)
    if not bottom:
        ax.tick_params(labelbottom=False)


def make_s6(outdir, log):
    js.apply()
    fig, axes = plt.subplots(3, 1, figsize=(js.WIDTH["double"], 150 * js.MM), sharex=True)
    for ax, ts in zip(axes, D.TIMESCALES):
        ax.axhspan(YLIM[0], -1, color=DROUGHT_FILL, lw=0, zorder=0)
        ax.axhline(-1, color=js.GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(0, color="#D0D0D0", lw=0.5, zorder=1)
        sel = D.SELECTED[ts]
        obs = None
        for m in D.MODELS:
            nat, sd = national(m, ts)
            obs = nat.observed
            name = D.DISPLAY[m]
            ax.plot(nat.index, nat.predicted, color=js.MODEL_COLORS[name],
                    lw=1.7 if m == sel else 1.0, alpha=1.0 if m == sel else 0.85,
                    zorder=4 if m == sel else 3)
        ax.plot(obs.index, obs.values, color="black", lw=1.4, zorder=5)
        ax.set_ylim(*YLIM)
        ax.set_ylabel(f"SPEI-{ts}\n(mean of 8 divisions)")
        ax.text(0.01, 0.95, f"SPEI-{ts}  ·  selected: {D.DISPLAY[sel]} (bold)",
                transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        _style_time_axis(ax, bottom=(ts == 12))
    js.panel_labels(list(axes), dx=-34, dy=2)
    h = [Line2D([], [], color="black", lw=1.4, label="Observed")]
    h += [Line2D([], [], color=js.MODEL_COLORS[D.DISPLAY[m]], lw=1.2, label=D.DISPLAY[m])
          for m in D.MODELS]
    h += [Patch(facecolor=DROUGHT_FILL, edgecolor="none", label="Drought (SPEI ≤ −1)")]
    fig.legend(handles=h, loc="outside upper center", ncol=7, frameon=False)
    return js.save(fig, "FigS06", outdir)


def make_s7(outdir, log):
    js.apply()
    divs = sorted(D.pred(D.SELECTED[3], 3, 1, D.median_seed(D.SELECTED[3], 3)).division.unique())
    fig, axes = plt.subplots(len(divs), 3, figsize=(js.WIDTH["double"], 222 * js.MM),
                             sharex=True, sharey=True)
    for ci, ts in enumerate(D.TIMESCALES):
        m = D.SELECTED[ts]
        sd = D.median_seed(m, ts)
        d = D.pred(m, ts, 1, sd)
        col = js.MODEL_COLORS[D.DISPLAY[m]]
        for ri, dv in enumerate(divs):
            ax = axes[ri, ci]
            g = d[d.division == dv]
            ax.axhspan(YLIM[0], -1, color=DROUGHT_FILL, lw=0, zorder=0)
            ax.axhline(-1, color=js.GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
            ax.plot(g.target_date, g.predicted, color=col, lw=1.1, zorder=3)
            ax.plot(g.target_date, g.observed, color="black", lw=1.1, zorder=4)
            ax.set_ylim(*YLIM)
            ax.set_yticks([-2, 0, 2])
            ax.text(0.015, 0.93, dv, transform=ax.transAxes, ha="left", va="top",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
            _style_time_axis(ax, bottom=(ri == len(divs) - 1))
            if ri == len(divs) - 1:
                ax.tick_params(axis="x", labelrotation=0)
                for lab in ax.get_xticklabels():
                    lab.set_fontsize(7)
            if ci == 0:
                ax.set_ylabel("SPEI")
            if ri == 0:
                ax.set_title(f"SPEI-{ts} · {D.DISPLAY[m]} (seed {sd})", fontweight="bold", pad=3)
    h = [Line2D([], [], color="black", lw=1.1, label="Observed")]
    h += [Line2D([], [], color=js.MODEL_COLORS[D.DISPLAY[D.SELECTED[t]]], lw=1.1,
                 label=f"{D.DISPLAY[D.SELECTED[t]]} (SPEI-{t})") for t in D.TIMESCALES]
    h += [Patch(facecolor=DROUGHT_FILL, edgecolor="none", label="Drought (SPEI ≤ −1)")]
    fig.legend(handles=h, loc="outside upper center", ncol=5, frameon=False)
    return js.save(fig, "FigS07", outdir)


def make(outdir, log):
    log.section("Figs S6-S7 - test-period series")
    checks(log)
    return make_s6(outdir, log) + make_s7(outdir, log)


if __name__ == "__main__":
    import p1_fig_config as C
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
