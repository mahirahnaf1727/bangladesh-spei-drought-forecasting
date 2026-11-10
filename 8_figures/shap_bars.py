"""
shap_bars.py  --  drawing code of the SHAP importance bars, used by fig10_shap.py (panels a-c)
(formerly fig09_shap.py; make() still draws the two stand-alone colourings)

a-c  Mean |SHAP value| of each predictor for CNN-BiLSTM (SPEI-3), Bi-LSTM (SPEI-6)
     and CNN (SPEI-12). Bar = mean over the 5 seeds, whisker = +/- 1 SD, dots = the
     5 seeds. Same row order and x-limits in all panels: lagged SPEI first, then the
     meteorological predictors ordered by their mean rank across the three models.
Two colourings (Fig. 10 uses Fig09_sign):
  Fig09_role : lagged SPEI wine #882255, meteorological predictors sky blue #56B4E9.
  Fig09_sign : sign of the mean signed SHAP, shown only where all 5 seeds agree.
               Blue = pushes the forecast up (wetter), red = pushes it down (drier),
               matching the SPEI map colours (dry = red). Grey = the sign differs
               between seeds. (The thesis used the opposite red/blue and seed 42 only.)
Only the top three ranks are stable across seeds.

Sources: TableS17_shap_summary.csv, TableS17_shap_per_seed.csv,
_shap/shap_values_<model>_SPEI<ts>_seed<s>.npz (cross-check).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import journal_style as js
import fig_data as D
import p1_fig_config as C

LABEL = {"SPEI": "Lagged SPEI", "Total Precipitation": "Total precipitation",
         "Surface Solar Radiation Downwards": "Solar radiation",
         "Maximum Temperature": "Maximum temperature", "Minimum Temperature": "Minimum temperature",
         "Dewpoint Temperature": "Dew-point temperature", "Total Cloud Cover": "Total cloud cover",
         "Wind Speed": "Wind speed"}
ROLE_COL = {"lag": "#882255", "met": "#56B4E9"}
SIGN_COL = {"up": "#2166AC", "down": "#B2182B", "varies": "#C8C8C8"}
TITLE = {3: "CNN-BiLSTM, SPEI-3", 6: "Bi-LSTM, SPEI-6", 12: "CNN, SPEI-12"}


def row_order(summ):
    met = summ[summ.variable != "SPEI"].groupby("variable").rank_mean.mean().sort_values()
    return ["SPEI"] + met.index.tolist()


def checks(log, summ, per):
    D.check_per_seed_against(log, per, summ, ["model", "timescale", "variable"],
                             [("mean_abs_shap", "mean_abs_shap_mean", "mean_abs_shap_sd"),
                              ("mean_signed_shap", "mean_signed_shap_mean", "mean_signed_shap_sd")],
                             "Table S17")
    worst = 0.0
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        for sd in D.SEEDS:
            p = C.OUT_ROOT / "_shap" / f"shap_values_{m}_SPEI{ts}_seed{sd}.npz"
            if not p.exists():
                log.note(f"{p.name} not found; npz cross-check skipped for it")
                continue
            z = np.load(p, allow_pickle=True)
            labs = [str(x) for x in z["labels"]]
            mabs = np.abs(z["shap_2d"]).mean(axis=0)
            q = per[(per.model == m) & (per.timescale == ts) & (per.seed == sd)].set_index("variable")
            for lab, v in zip(labs, mabs):
                worst = max(worst, abs(v - q.loc[lab, "mean_abs_shap"]))
    if worst > 1e-5:
        log.fail(f"mean |SHAP| from the npz files differs from Table S17 per-seed (max {worst})")
    log.ok(f"mean |SHAP| recomputed from the 15 SHAP value files equals Table S17 (max |diff| {worst:.1e})")
    lines = []
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        g = per[(per.model == m) & (per.timescale == ts)].groupby("variable").mean_signed_shap
        npos = g.apply(lambda x: int((x > 0).sum()))
        lines.append(f"SPEI-{ts}: " + ", ".join(f"{LABEL[v]} {n}/5 positive" for v, n in npos.items()))
    log.note("sign agreement across seeds (Fig09_sign greys any predictor that is not 5/5 or 0/5): "
             + " | ".join(lines))


def draw(axes, summ, per, order, mode):
    xmax = 0.0
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        s = summ[(summ.model == m) & (summ.timescale == ts)].set_index("variable")
        xmax = max(xmax, (s.mean_abs_shap_mean + s.mean_abs_shap_sd).max(),
                   per[(per.model == m) & (per.timescale == ts)].mean_abs_shap.max())
    xmax = np.ceil(xmax * 1.08 * 100) / 100
    y = np.arange(len(order))[::-1]          # first item on top
    for ax, ts in zip(axes, D.TIMESCALES):
        m = D.SELECTED[ts]
        s = summ[(summ.model == m) & (summ.timescale == ts)].set_index("variable")
        p = per[(per.model == m) & (per.timescale == ts)]
        for yi, v in zip(y, order):
            if mode == "role":
                col = ROLE_COL["lag"] if v == "SPEI" else ROLE_COL["met"]
            else:
                sg = p[p.variable == v].mean_signed_shap
                col = SIGN_COL["up"] if (sg > 0).all() else SIGN_COL["down"] if (sg < 0).all() \
                    else SIGN_COL["varies"]
            ax.barh(yi, s.loc[v, "mean_abs_shap_mean"], height=0.68, color=col, edgecolor=js.INK,
                    linewidth=0.5, zorder=2)
            ax.errorbar(s.loc[v, "mean_abs_shap_mean"], yi, xerr=s.loc[v, "mean_abs_shap_sd"],
                        fmt="none", ecolor=js.INK, elinewidth=0.8, zorder=3)
            seeds = p[p.variable == v].mean_abs_shap.values
            ax.scatter(seeds, np.full(len(seeds), yi) + np.linspace(-0.2, 0.2, len(seeds)),
                       s=3.5, color=js.INK, lw=0, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[v] for v in order])
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.xaxis.grid(True, color="#E8E8E8", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(TITLE[ts], fontweight="bold", pad=4)
        ax.set_xlabel("Mean |SHAP value|")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def handles(mode):
    common = [Line2D([], [], color=js.INK, lw=0.8, label="±1 SD across seeds"),
              Line2D([], [], marker="o", ls="none", ms=2.1, mfc=js.INK, mec="none",
                     label="Individual seeds")]
    if mode == "role":
        h = [Patch(facecolor=ROLE_COL["lag"], edgecolor=js.INK, lw=0.5, label="Lagged SPEI"),
             Patch(facecolor=ROLE_COL["met"], edgecolor=js.INK, lw=0.5, label="Meteorological predictor")]
    else:
        h = [Patch(facecolor=SIGN_COL["up"], edgecolor=js.INK, lw=0.5,
                   label="Mean effect wetter (raises SPEI) in all 5 seeds"),
             Patch(facecolor=SIGN_COL["down"], edgecolor=js.INK, lw=0.5,
                   label="Mean effect drier (lowers SPEI) in all 5 seeds"),
             Patch(facecolor=SIGN_COL["varies"], edgecolor=js.INK, lw=0.5,
                   label="Sign of the mean effect differs between seeds")]
    return h + common


def make(outdir, log):
    log.section("SHAP importance bars")
    summ = D.table("TableS17_shap_summary.csv")
    per = D.table("TableS17_shap_per_seed.csv")
    checks(log, summ, per)
    order = row_order(summ)
    log.ok("row order (all panels): " + ", ".join(LABEL[v] for v in order))
    js.apply()
    problems = []
    for mode in ("role", "sign"):
        fig, axes = plt.subplots(1, 3, figsize=(js.WIDTH["double"], 70 * js.MM), sharey=True)
        draw(axes, summ, per, order, mode)
        fig.legend(handles=handles(mode), loc="outside upper center",
                   ncol=4 if mode == "role" else 3, frameon=False)
        js.panel_labels([axes[0]], dx=-96, dy=4)                 # 'a' left of the row labels
        js.panel_labels(list(axes[1:]), start="b", dx=-14, dy=4)
        problems += js.save(fig, f"Fig09_{mode}", outdir)
    return problems


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
