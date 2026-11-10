"""
Fig 6  --  Event detection at lead 1 (thesis Fig 24, §4.6)

a-c  Performance diagrams (Roebber 2009), one per timescale: success ratio
     (1 - FAR) on x, probability of detection on y, CSI isolines (solid grey,
     labelled at the right edge) and frequency-bias lines (dashed grey). Drought =
     SPEI <= -1. Each model: 5-seed mean of POD and of SR (large marker) and the
     5 seeds (small dots); persistence as a black X. Same axis limits in every panel
     (zoomed to SR 0.5-1, POD 0.2-1, which holds every point).

Note: the large marker is the mean of POD and the mean of SR over seeds (as in
Table S11); the CSI isoline under it can differ slightly from the mean CSI,
because CSI is not linear in POD and SR.

Sources: Table S11 (means, persistence), Table S14 (per-seed POD and SR).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import journal_style as js
import fig_data as D

XLIM, YLIM = (0.5, 1.0), (0.2, 1.0)
CSI_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BIAS = [0.5, 0.8, 1.0]           # every point has bias <= 1.04, so lines > 1 are not needed


def checks(log):
    s12 = D.table("TableS11_contingency.csv")
    s15 = D.table("TableS14_per_seed_runs.csv")
    m12 = s12[(s12.lead == 1) & (s12.model != "Persistence")]
    D.check_per_seed_against(log, s15[s15.lead == 1], m12, ["model", "timescale"],
                             [("POD", "POD_mean", "POD_sd"), ("SR", "SR_mean", "SR_sd"),
                              ("CSI", "CSI_mean", "CSI_sd")], "Table S11 (lead 1)")
    for ts in D.TIMESCALES:
        p = s12[(s12.lead == 1) & (s12.model == "Persistence") & (s12.timescale == ts)].iloc[0]
        if abs(p.SR_mean - (1 - p.FAR_mean)) > 1e-6:
            log.fail(f"persistence SPEI-{ts}: SR != 1 - FAR in Table S11")
        if int(round(p.hits_mean + p.misses_mean)) != D.N_DROUGHT[ts]:
            log.fail(f"persistence SPEI-{ts}: hits + misses != {D.N_DROUGHT[ts]}")
    log.ok("persistence rows: SR = 1 - FAR and hits + misses = 119 / 156 / 204 drought months")
    r1 = s15[s15.lead == 1]
    bias = ((r1.hits + r1.false_alarms) / (r1.hits + r1.misses)).max()
    p1 = s12[(s12.lead == 1) & (s12.model == "Persistence")]
    pbias = ((p1.hits_mean + p1.false_alarms_mean) / (p1.hits_mean + p1.misses_mean)).max()
    if max(bias, pbias) > 1.05:
        log.fail(f"a point has frequency bias {max(bias, pbias):.3f} > 1.05; add bias lines > 1")
    log.ok(f"largest frequency bias: models {bias:.3f}, persistence {pbias:.3f} "
           f"(so only bias lines 0.5, 0.8, 1 are drawn)")


def _frame(ax):
    sr, pod = np.meshgrid(np.linspace(0.3, 1, 400), np.linspace(0.05, 1, 400))
    csi = 1 / (1 / sr + 1 / pod - 1)
    ax.contour(sr, pod, csi, levels=CSI_LEVELS, colors="#C8C8C8", linewidths=0.6, zorder=1)
    # CSI labels sit on the curves inside the frame, along the left side (CSI <= 0.5)
    # and the top (CSI >= 0.6); bias labels sit outside the right edge, so the two
    # sets of numbers never share an edge.
    for c in CSI_LEVELS:
        if c <= 0.5:
            x0 = 0.53
            y0 = 1 / (1 / c - 1 / x0 + 1)
        else:
            y0 = 0.965
            x0 = 1 / (1 / c - 1 / y0 + 1)
        ax.text(x0, y0, f"{c:g}", color=js.GREY, fontsize=7, ha="center", va="center",
                zorder=2, bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none"))
    x = np.linspace(*XLIM, 50)
    for b in BIAS:
        ax.plot(x, b * x, color=js.GREY, lw=0.6, ls=(0, (3, 1.6)), zorder=1)
        ax.text(1.012, b, f"{b:g}", color=js.GREY, fontsize=7, ha="left", va="center",
                clip_on=False)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def make(outdir, log):
    log.section("Fig 6 - performance diagrams, lead 1")
    checks(log)
    s12 = D.table("TableS11_contingency.csv")
    s15 = D.table("TableS14_per_seed_runs.csv")
    js.apply()
    fig, axes = plt.subplots(1, 3, figsize=(js.WIDTH["double"], 74 * js.MM))
    for ax, ts in zip(axes, D.TIMESCALES):
        _frame(ax)
        for m in D.MODELS:
            name = D.DISPLAY[m]
            runs = s15[(s15.model == m) & (s15.timescale == ts) & (s15.lead == 1)]
            ax.scatter(runs.SR, runs.POD, s=4, color=js.MODEL_COLORS[name], alpha=0.8,
                       lw=0, zorder=3)
            r = s12[(s12.model == m) & (s12.timescale == ts) & (s12.lead == 1)].iloc[0]
            ax.plot(r.SR_mean, r.POD_mean, marker=js.MODEL_MARKERS[name], ms=4.4,
                    mfc=js.MODEL_COLORS[name], mec=js.INK, mew=0.5, ls="none", zorder=5)
        p = s12[(s12.model == "Persistence") & (s12.timescale == ts) & (s12.lead == 1)].iloc[0]
        ax.plot(p.SR_mean, p.POD_mean, marker="X", ms=5.2, mfc="black", mec="white",
                mew=0.4, ls="none", zorder=6)
        ax.set_title(f"SPEI-{ts}", fontweight="bold", pad=4)
        ax.set_xlabel("Success ratio (1 − FAR)")
    axes[0].set_ylabel("Probability of detection (POD)")
    js.panel_labels(axes, dx=-24, dy=6)

    handles = [Line2D([], [], marker=js.MODEL_MARKERS[D.DISPLAY[m]], ls="none", ms=4.4,
                      mfc=js.MODEL_COLORS[D.DISPLAY[m]], mec=js.INK, mew=0.5,
                      label=D.DISPLAY[m]) for m in D.MODELS]
    handles += [Line2D([], [], marker="X", ls="none", ms=5.2, mfc="black", mec="white",
                       mew=0.4, label="Persistence"),
                Line2D([], [], marker="o", ls="none", ms=2.2, mfc=js.GREY, mec="none",
                       label="Individual seeds"),
                Line2D([], [], color="#C8C8C8", lw=0.6, label="CSI"),
                Line2D([], [], color=js.GREY, lw=0.6, ls=(0, (3, 1.6)), label="Frequency bias")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
    return js.save(fig, "Fig06", outdir)


if __name__ == "__main__":
    import p1_fig_config as C
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
