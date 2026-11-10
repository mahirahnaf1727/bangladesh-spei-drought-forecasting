"""
Fig 4  --  Lead-1 model comparison (thesis Fig 18 + Fig A7, §4.5.1 and §4.5.6)

a RMSE, b NSE, c skill against persistence; SPEI-3/6/12 groups; five models.
Each model: 5-seed mean (marker) +/- 1 SD (whisker), the five seeds as faint dots
beside it. Persistence: black dashed tick (skill = 0 by definition). The selected
model of each timescale sits on a light grey band. No bootstrap CIs.

Sources: Table 2 (means, SDs, persistence), Table S14 (per-seed values).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import journal_style as js
import fig_data as D

METRICS = [("RMSE", "RMSE", (0.2, 0.9)),
           ("NSE", "NSE", (0.25, 0.95)),
           ("Skill", "Skill against persistence", (-0.2, 0.5))]
OFFSETS = np.linspace(-0.32, 0.32, 5)
SEED_DX = 0.065


def make(outdir, log):
    log.section("Fig 4 - lead-1 comparison")
    s11 = D.table("Table2_lead1_all_models.csv")
    s15 = D.table("TableS14_per_seed_runs.csv")
    s15_1 = s15[s15.lead == 1]
    models = s11[s11.model != "Persistence"]
    D.check_per_seed_against(log, s15_1, models, ["timescale", "model"],
                             [(m, f"{m}_mean", f"{m}_sd") for m, _, _ in METRICS], "Table 2")
    for ts in D.TIMESCALES:
        sel = s11[(s11.timescale == ts) & (s11.selected.astype(str) == "True")].model.tolist()
        if sel != [D.SELECTED[ts]]:
            log.fail(f"selected model at SPEI-{ts} in S11 is {sel}, expected {D.SELECTED[ts]}")
    log.ok("selected models in Table 2 = CNN-BiLSTM / Bi-LSTM / CNN")

    js.apply()
    fig, axes = plt.subplots(1, 3, figsize=(js.WIDTH["double"], 72 * js.MM))
    for ax, (col, ylabel, ylim) in zip(axes, METRICS):
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for gi, ts in enumerate(D.TIMESCALES):
            # selected-model band
            si = D.MODELS.index(D.SELECTED[ts])
            ax.axvspan(gi + OFFSETS[si] - 0.075, gi + OFFSETS[si] + 0.105,
                       color="#ECECEC", lw=0, zorder=0.5)
            # persistence tick
            pv = 0.0 if col == "Skill" else float(
                s11[(s11.timescale == ts) & (s11.model == "Persistence")][f"{col}_mean"].iloc[0])
            ax.plot([gi - 0.42, gi + 0.42], [pv, pv], ls=(0, (3, 1.6)), color="black",
                    lw=1.0, zorder=2)
            for mi, m in enumerate(D.MODELS):
                name = D.DISPLAY[m]
                x = gi + OFFSETS[mi]
                r = s11[(s11.timescale == ts) & (s11.model == m)].iloc[0]
                seeds = s15_1[(s15_1.timescale == ts) & (s15_1.model == m)][col].values
                ax.scatter(np.full(5, x + SEED_DX), seeds, s=3.5, color=js.MODEL_COLORS[name],
                           alpha=0.75, lw=0, zorder=3)
                ax.errorbar(x, r[f"{col}_mean"], yerr=r[f"{col}_sd"], fmt="none",
                            ecolor=js.INK, elinewidth=0.8, zorder=4)
                ax.plot(x, r[f"{col}_mean"], marker=js.MODEL_MARKERS[name], ms=4.2,
                        mfc=js.MODEL_COLORS[name], mec=js.INK, mew=0.5, ls="none", zorder=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"SPEI-{t}" for t in D.TIMESCALES])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
    js.panel_labels(axes, dx=-26)

    handles = [Line2D([], [], marker=js.MODEL_MARKERS[D.DISPLAY[m]], ls="none", ms=4.2,
                      mfc=js.MODEL_COLORS[D.DISPLAY[m]], mec=js.INK, mew=0.5,
                      label=D.DISPLAY[m]) for m in D.MODELS]
    handles += [Line2D([], [], color="black", ls=(0, (3, 1.6)), lw=1.0, label="Persistence"),
                Line2D([], [], marker="o", ls="none", ms=2.1, mfc=js.GREY, mec="none",
                       label="Individual seeds"),
                Patch(facecolor="#ECECEC", edgecolor="none", label="Selected model")]
    fig.legend(handles=handles, loc="outside upper center", ncol=8, frameon=False)
    return js.save(fig, "Fig04", outdir)


if __name__ == "__main__":
    import p1_fig_config as C
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
