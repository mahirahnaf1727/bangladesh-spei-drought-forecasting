"""
Fig 8  --  SPEI-12: how much the seed decides (thesis: uncaptioned figure in §4.11,
CNN SPEI-12 with 5 seeds; numbers replaced by the as-tuned Stage 2 runs, C1)

a  NSE vs lead, CNN at SPEI-12: 5 seeds (dots), 5-seed mean +/- 1 SD (marker and
   bar, means joined); persistence NSE (black dashed, X).
b  Skill against persistence vs lead, same runs; persistence = 0 (dashed line).
   The two degraded runs are labelled: seed 43 at lead 2, seed 46 at lead 3. They are
   different seeds, because each lead is a separately trained model, so seeds are not
   joined across leads.
c  Lead-1 skill of all five models at SPEI-12 (5 seeds + mean +/- 1 SD); grey band =
   selected model (CNN). b and c share the y axis.
Numbers above the points: seeds with skill > 0 (from Table S14; equal to Table 2
"Seeds > 0" and the number log, C1).

Sources: Table2_multilead.csv, TableS11_lead1_all_models.csv, TableS15_per_seed_runs.csv.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import journal_style as js
import fig_data as D

LEADS = [1, 2, 3]
SEED_DX, MEAN_DX = 0.13, -0.03
SKILL_LIM = (-0.8, 0.45)


def checks(log):
    t2 = D.table("Table3_multilead.csv")
    s11 = D.table("Table2_lead1_all_models.csv")
    s15 = D.table("TableS14_per_seed_runs.csv")
    cnn = t2[(t2.timescale == 12) & (t2.model == "CNN")]
    D.check_per_seed_against(log, s15, cnn, ["timescale", "lead", "model"],
                             [("NSE", "NSE_mean", "NSE_sd"), ("Skill", "Skill_mean", "Skill_sd")],
                             "Table 3 (CNN, SPEI-12)")
    l1 = s11[(s11.timescale == 12) & (s11.model != "Persistence")]
    D.check_per_seed_against(log, s15[s15.lead == 1], l1, ["timescale", "model"],
                             [("Skill", "Skill_mean", "Skill_sd")], "Table 2 (SPEI-12)")
    runs = s15[(s15.timescale == 12) & (s15.lead == 1)]
    for _, r in l1.iterrows():
        n = int((runs[runs.model == r.model].Skill > 0).sum())
        if n != int(r.Skill_n_pos):
            log.fail(f"{r.model}: {n} seeds > 0 vs Table 2 {r.Skill_n_pos}")
    c = s15[(s15.model == "CNN") & (s15.timescale == 12)]
    npos = [int((c[c.lead == L].Skill > 0).sum()) for L in LEADS]
    if npos != [5, 4, 3]:
        log.fail(f"CNN SPEI-12 seeds > 0 by lead {npos}, number log says 5/4/3")
    worst = {L: c[c.lead == L].sort_values("Skill").iloc[0] for L in (2, 3)}
    if int(worst[2].seed) != 43 or int(worst[3].seed) != 46 \
            or abs(worst[2].NSE - 0.4968) > 5e-4 or abs(worst[3].NSE - 0.0843) > 5e-4:
        log.fail("degraded seeds are not seed 43 (lead 2, NSE 0.497) and seed 46 (lead 3, NSE 0.084)")
    log.ok("seeds > 0 equal Table 2 (lead 1, all models) and the number log for the CNN "
           "(5/5, 4/5, 3/5); degraded runs = seed 43 at lead 2 (NSE 0.497, skill "
           f"{worst[2].Skill:.3f}) and seed 46 at lead 3 (NSE 0.084, skill {worst[3].Skill:.3f})")


def _seed_mean(ax, x, vals, mean, sd, color, marker):
    ax.scatter(np.full(len(vals), x + SEED_DX), vals, s=5, color=color, alpha=0.8, lw=0, zorder=3)
    ax.errorbar(x + MEAN_DX, mean, yerr=sd, fmt="none", ecolor=js.INK, elinewidth=0.8, zorder=4)
    ax.plot(x + MEAN_DX, mean, marker=marker, ms=4.6, mfc=color, mec=js.INK, mew=0.5,
            ls="none", zorder=5)


def make(outdir, log):
    log.section("Fig 8 - SPEI-12 seed spread")
    checks(log)
    t2 = D.table("Table3_multilead.csv")
    s11 = D.table("Table2_lead1_all_models.csv")
    s15 = D.table("TableS14_per_seed_runs.csv")
    cnn_runs = s15[(s15.model == "CNN") & (s15.timescale == 12)]
    cnn_tab = t2[(t2.model == "CNN") & (t2.timescale == 12)].sort_values("lead")
    per_tab = t2[(t2.model == "Persistence") & (t2.timescale == 12)].sort_values("lead")
    green, mk = js.MODEL_COLORS["CNN"], js.MODEL_MARKERS["CNN"]

    js.apply()
    fig = plt.figure(figsize=(js.WIDTH["double"], 70 * js.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharey=ax_b)

    for ax, met in ((ax_a, "NSE"), (ax_b, "Skill")):
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        if met == "NSE":
            ax.plot(np.array(LEADS), per_tab.NSE_mean, color="black", lw=1.0,
                    ls=(0, (3, 1.6)), zorder=2)
            ax.plot(np.array(LEADS), per_tab.NSE_mean, marker="X", ms=5.0, mfc="black",
                    mec="white", mew=0.4, ls="none", zorder=5)
        else:
            ax.axhline(0, color="black", lw=1.0, ls=(0, (3, 1.6)), zorder=2)
        ax.plot(np.array(LEADS) + MEAN_DX, cnn_tab[f"{met}_mean"], color=green, lw=1.0, zorder=2)
        for L in LEADS:
            r = cnn_tab[cnn_tab.lead == L].iloc[0]
            _seed_mean(ax, L, cnn_runs[cnn_runs.lead == L][met].values,
                       r[f"{met}_mean"], r[f"{met}_sd"], green, mk)
        for L, sd in ((2, 43), (3, 46)):       # label the degraded runs
            v = float(cnn_runs[(cnn_runs.lead == L) & (cnn_runs.seed == sd)][met].iloc[0])
            # in panel a the seed-43 label goes left of its dot, clear of the persistence line
            left = (met == "NSE" and sd == 43)
            ax.text(L + SEED_DX + (-0.07 if left else 0.07), v, f"seed {sd}",
                    ha="right" if left else "left", va="center", fontsize=7)
        ax.set_xticks(LEADS)
        ax.set_xlim(0.6, 3.6)
        ax.set_xlabel("Lead time (months)")
        ax.set_title("CNN, SPEI-12", pad=4)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("NSE")
    ax_b.set_ylim(*SKILL_LIM)
    ax_b.set_ylabel("Skill against persistence")
    for L in LEADS:
        n = int((cnn_runs[cnn_runs.lead == L].Skill > 0).sum())
        ax_b.text(L + 0.05, SKILL_LIM[1] - 0.03, f"{n}/5 > 0", ha="center", va="top", fontsize=7)

    # c: lead 1, all models
    ax = ax_c
    ax.yaxis.grid(True, color="#E8E8E8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", lw=1.0, ls=(0, (3, 1.6)), zorder=2)
    runs1 = s15[(s15.timescale == 12) & (s15.lead == 1)]
    for i, m in enumerate(D.MODELS):
        name = D.DISPLAY[m]
        r = s11[(s11.timescale == 12) & (s11.model == m)].iloc[0]
        if m == D.SELECTED[12]:
            ax.axvspan(i - 0.16, i + 0.26, color="#ECECEC", lw=0, zorder=0.5)
        _seed_mean(ax, i, runs1[runs1.model == m].Skill.values, r.Skill_mean, r.Skill_sd,
                   js.MODEL_COLORS[name], js.MODEL_MARKERS[name])
        ax.text(i + 0.05, SKILL_LIM[1] - 0.03, f"{int(r.Skill_n_pos)}/5 > 0", ha="center",
                va="top", fontsize=7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["XGBoost", "CNN", "Bi-LSTM", "CNN-\nBiLSTM", "CEEMDAN-\nBiLSTM"])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(-0.55, 4.6)
    ax.set_title("All models, SPEI-12, lead 1", pad=4)
    js.panel_labels([ax_a, ax_b, ax_c], dx=-24, dy=6)

    handles = [Line2D([], [], marker=js.MODEL_MARKERS[D.DISPLAY[m]], ls="none", ms=4.6,
                      mfc=js.MODEL_COLORS[D.DISPLAY[m]], mec=js.INK, mew=0.5,
                      label=D.DISPLAY[m]) for m in D.MODELS]
    handles += [Line2D([], [], marker="X", ls=(0, (3, 1.6)), lw=1.0, ms=5.0, color="black",
                       mfc="black", mec="white", mew=0.4, label="Persistence"),
                Line2D([], [], marker="o", ls="none", ms=2.4, mfc=js.GREY, mec="none",
                       label="Individual seeds"),
                Patch(facecolor="#ECECEC", edgecolor="none", label="Selected model")]
    fig.legend(handles=handles, loc="outside upper center", ncol=8, frameon=False)
    return js.save(fig, "Fig08", outdir)


if __name__ == "__main__":
    import p1_fig_config as C
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
