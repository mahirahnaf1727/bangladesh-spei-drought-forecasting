"""
Fig 7  --  Core: accuracy, skill and detection diverge with lead
(thesis: uncaptioned figures in §4.10.1 (skill vs lead) and §4.10.5 (NSE vs skill
bars; POD and CSI vs lead), merged)

3 x 3: rows NSE / skill against persistence / CSI; columns SPEI-3/6/12; leads 1-3.
Series: the selected model of each timescale, XGBoost, persistence.
  marker      = 5-seed mean (persistence: single value)
  thick bar   = +/- 1 SD across the 5 seeds
  thin bar    = 95% circular block bootstrap CI (12-month blocks, pooled over seeds)
  CSI row     : filled marker = the paired difference from persistence (Delta CSI)
                has a 95% CI that excludes 0; open marker = it does not.
                This shows SPEI-6 detection as level with persistence (C3).
  skill row   : persistence = 0 by definition (dashed line).
"0 of 119": XGBoost, SPEI-3, lead 3 makes zero hits in every seed (checked).

Sources: Table3_multilead.csv (means, SDs, CIs, Delta CSI CI), TableS13A/B (CI
cross-check), TableS15_per_seed_runs.csv (per-seed check).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import journal_style as js
import fig_data as D

ROWS = [("NSE", "NSE", (-1.5, 1.0)),
        ("Skill", "Skill against persistence", (-1.25, 0.65)),
        ("CSI", "CSI", (0.0, 0.85))]
DX = {"sel": -0.15, "Persistence": 0.0, "XGBoost": 0.15}
LEADS = [1, 2, 3]


def checks(log):
    t2 = D.table("Table3_multilead.csv")
    s15 = D.table("TableS14_per_seed_runs.csv")
    s14a = D.table("TableS13A_bootstrap_ci.csv")
    s14b = D.table("TableS13B_bootstrap_differences.csv")
    mods = t2[t2.model != "Persistence"]
    D.check_per_seed_against(log, s15, mods, ["timescale", "lead", "model"],
                             [("NSE", "NSE_mean", "NSE_sd"), ("Skill", "Skill_mean", "Skill_sd"),
                              ("CSI", "CSI_mean", "CSI_sd")], "Table 3")
    worst = 0.0
    for _, r in t2.iterrows():
        for met in ("NSE", "Skill", "CSI"):
            q = s14a[(s14a.model == r.model) & (s14a.timescale == r.timescale) &
                     (s14a.lead == r.lead) & (s14a.metric == met)]
            if len(q):
                worst = max(worst, abs(q.lo.iloc[0] - r[f"{met}_lo"]), abs(q.hi.iloc[0] - r[f"{met}_hi"]))
        if r.model != "Persistence":
            q = s14b[(s14b.model == r.model) & (s14b.vs == "Persistence") & (s14b.timescale == r.timescale)
                     & (s14b.lead == r.lead) & (s14b.metric == "CSI")]
            worst = max(worst, abs(q.lo.iloc[0] - r.dCSI_lo), abs(q.hi.iloc[0] - r.dCSI_hi))
    if worst > 1e-9:
        log.fail(f"Table 3 CIs differ from Tables S13A/S13B (max {worst})")
    log.ok("all CIs in Table 3 (NSE, skill, CSI, Delta CSI) equal Tables S13A/S13B")
    x = s15[(s15.model == "XGBoost") & (s15.timescale == 3) & (s15.lead == 3)]
    if len(x) != 5 or x.hits.sum() != 0 or set(x.n_observed_events) != {119}:
        log.fail("'0 of 119' does not hold for XGBoost SPEI-3 lead 3")
    log.ok(f"'0 of 119': XGBoost SPEI-3 lead 3 has 0 hits in all 5 seeds (119 observed drought "
           f"months; false alarms per seed {x.false_alarms.tolist()})")
    sig = []
    for _, r in mods.iterrows():
        s = "differs" if (r.dCSI_lo > 0 or r.dCSI_hi < 0) else "level"
        sig.append(f"{D.DISPLAY[r.model]} SPEI-{r.timescale} L{r.lead}: {s} "
                   f"[{r.dCSI_lo:+.3f}, {r.dCSI_hi:+.3f}]")
    log.note("CSI vs persistence (filled = differs, open = level): " + "; ".join(sig))


def _series(ax, x, mean, sd, lo, hi, color, marker, filled=True, is_pers=False):
    """Means joined by a line; thick SD bars and thin CI bars at each lead."""
    x = np.asarray(x, float)
    ls = (0, (3, 1.6)) if is_pers else "-"
    ax.plot(x, mean, color=color, lw=1.0, ls=ls, zorder=2)
    for i in range(len(x)):
        if lo is not None and np.isfinite(lo[i]):
            ax.plot([x[i], x[i]], [lo[i], hi[i]], color=color, lw=0.8, zorder=3,
                    solid_capstyle="butt")
        if sd is not None and np.isfinite(sd[i]) and sd[i] > 0:
            ax.plot([x[i], x[i]], [mean[i] - sd[i], mean[i] + sd[i]], color=color, lw=3.0,
                    alpha=0.45, zorder=3, solid_capstyle="butt")
        f = filled if np.isscalar(filled) else filled[i]
        if is_pers:
            ax.plot(x[i], mean[i], marker="X", ms=5.0, mfc="black", mec="white", mew=0.4,
                    ls="none", zorder=5)
        else:
            ax.plot(x[i], mean[i], marker=marker, ms=4.6, mfc=color if f else "white",
                    mec=js.INK if f else color, mew=0.5 if f else 1.0, ls="none", zorder=5)


def make(outdir, log):
    log.section("Fig 7 - multi-lead core figure")
    checks(log)
    t2 = D.table("Table3_multilead.csv")
    js.apply()
    fig, axes = plt.subplots(3, 3, figsize=(js.WIDTH["double"], 150 * js.MM),
                             sharex=True, sharey="row")
    for ci, ts in enumerate(D.TIMESCALES):
        sel = D.SELECTED[ts]
        for ri, (met, ylabel, ylim) in enumerate(ROWS):
            ax = axes[ri, ci]
            ax.yaxis.grid(True, color="#E8E8E8", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            if met == "NSE":
                ax.axhline(0, color="#BDBDBD", lw=0.5, zorder=1)
            for key, m in (("Persistence", "Persistence"), ("sel", sel), ("XGBoost", "XGBoost")):
                d = t2[(t2.timescale == ts) & (t2.model == m)].sort_values("lead")
                x = np.array(LEADS) + DX[key]
                if m == "Persistence":
                    if met == "Skill":
                        ax.axhline(0, color="black", lw=1.0, ls=(0, (3, 1.6)), zorder=2)
                        continue
                    _series(ax, x, d[f"{met}_mean"].values, None, d[f"{met}_lo"].values,
                            d[f"{met}_hi"].values, "black", "X", is_pers=True)
                    continue
                name = D.DISPLAY[m]
                filled = True
                if met == "CSI":
                    filled = ((d.dCSI_lo > 0) | (d.dCSI_hi < 0)).values
                _series(ax, x, d[f"{met}_mean"].values, d[f"{met}_sd"].values,
                        d[f"{met}_lo"].values, d[f"{met}_hi"].values,
                        js.MODEL_COLORS[name], js.MODEL_MARKERS[name], filled=filled)
            ax.set_ylim(*ylim)
            ax.set_xlim(0.6, 3.4)
            ax.set_xticks(LEADS)
            if ci == 0:
                ax.set_ylabel(ylabel)
            if ri == 0:
                ax.set_title(f"SPEI-{ts}", fontweight="bold", pad=4)
            if ri == 2:
                ax.set_xlabel("Lead time (months)")
    # "0 of 119" note: XGBoost, SPEI-3, lead 3, CSI = 0
    ax = axes[2, 0]
    # text sits above-right so the arrow drops almost vertically onto the point and
    # clears the persistence marker and CI at lead 3
    ax.annotate("XGBoost: 0 of 119\ndrought months detected",
                xy=(3 + DX["XGBoost"], 0.006), xytext=(3.27, 0.34), ha="right", va="bottom",
                fontsize=7, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=js.INK,
                                            shrinkA=1, shrinkB=2, mutation_scale=6,
                                            relpos=(0.92, 0.0)))
    js.panel_labels(list(axes.ravel()), dx=-20, dy=3)

    handles = [Line2D([], [], marker=js.MODEL_MARKERS[D.DISPLAY[m]], ls="-", lw=1.0, ms=4.6,
                      color=js.MODEL_COLORS[D.DISPLAY[m]], mfc=js.MODEL_COLORS[D.DISPLAY[m]],
                      mec=js.INK, mew=0.5, label=lab)
               for m, lab in (("CNN-BiLSTM", "CNN-BiLSTM (selected, SPEI-3)"),
                              ("BiLSTM", "Bi-LSTM (selected, SPEI-6)"),
                              ("CNN", "CNN (selected, SPEI-12)"),
                              ("XGBoost", "XGBoost"))]
    handles += [Line2D([], [], marker="X", ls=(0, (3, 1.6)), lw=1.0, ms=5.0, color="black",
                       mfc="black", mec="white", mew=0.4, label="Persistence"),
                Line2D([], [], color=js.GREY, lw=3.0, alpha=0.6, label="±1 SD across seeds"),
                Line2D([], [], color=js.GREY, lw=0.8, label="95% bootstrap CI"),
                Line2D([], [], marker="o", ls="none", ms=4.6, mfc="white", mec=js.INK, mew=1.0,
                       label="CSI level with persistence (open)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    return js.save(fig, "Fig07", outdir)


if __name__ == "__main__":
    import p1_fig_config as C
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
