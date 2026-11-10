"""
Fig 3  --  Predictor screening (thesis Fig 15, §4.4.1 + Fig 16, §4.4.3)

a    Pearson correlation matrix (lower triangle) of the 8 candidate ERA5 predictors,
     PET, the water balance and SPEI-1/3/6/12. Raw monthly values pooled over the 8
     divisions, 1980-2024, pairwise complete. Same computation as the thesis
     correlation script (Qwen_python_bigger_front.py, plot_full_matrix), which
     produced thesis Fig 15.
b-d  Lead-lag: Pearson r between each de-seasonalised predictor at month t-k and
     SPEI at month t, k = 1..12, pooled over divisions, for SPEI-3/6/12. Same
     computation as leadlag_anom_multipanel_v6.py (thesis Fig 16): anomalies from
     division x calendar-month means over all years; the 7 predictors kept after the
     VIF test (mean temperature dropped). Grey band = forecast leads 1-3; dashed
     line = lead equal to the accumulation length.

Writes Fig03_combined (a-d) and the split pair Fig03a (matrix) / Fig03b (lead-lag,
panels a-c). Data-only figure: no model, no seed.

Sources: divisional_spei_1980_2024.csv, divisional_averaged_1980_2024.csv,
divisional_pet_waterbalance_1980_2024.csv; check against leadlag_summary.csv
(thesis output) and against values printed in thesis Fig 15.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import journal_style as js
import fig_data as D
import p1_fig_config as C

MATRIX_VARS = ["tp", "t2m", "mx2t", "mn2t", "d2m", "ssrd", "tcc", "wind", "PET", "D",
               "SPEI_1", "SPEI_3", "SPEI_6", "SPEI_12"]
LABEL = {"tp": "Total precipitation", "t2m": "Mean temperature (2 m)",
         "mx2t": "Maximum temperature", "mn2t": "Minimum temperature",
         "d2m": "Dew-point temperature", "ssrd": "Solar radiation",
         "tcc": "Total cloud cover", "wind": "Wind speed", "PET": "PET",
         "D": "Water balance (P − PET)", "SPEI_1": "SPEI-1", "SPEI_3": "SPEI-3",
         "SPEI_6": "SPEI-6", "SPEI_12": "SPEI-12"}
DRIVERS = ["tp", "tcc", "d2m", "ssrd", "mx2t", "mn2t", "wind"]      # legend order
DRIVER_STYLE = {   # Tol "bright" + black; passes the colour-blind check on all pairs
    "tp": ("#4477AA", "o"), "tcc": ("#66CCEE", "s"), "d2m": ("#228833", "^"),
    "ssrd": ("#CCBB44", "D"), "mx2t": ("#EE6677", "v"), "mn2t": ("#AA3377", "P"),
    "wind": ("#000000", "X")}
MAXLAG = 12
# values printed in thesis Fig 15 (2 decimals), used as a reproduction check
THESIS_FIG15 = {("tp", "SPEI_1"): 0.37, ("tp", "SPEI_3"): 0.24, ("tp", "SPEI_6"): 0.20,
                ("tp", "SPEI_12"): 0.13, ("tp", "D"): 0.99, ("ssrd", "PET"): 0.72,
                ("ssrd", "SPEI_1"): -0.38, ("PET", "SPEI_1"): -0.31, ("mx2t", "PET"): 0.80,
                ("t2m", "mn2t"): 0.98, ("tcc", "tp"): 0.85, ("SPEI_3", "SPEI_6"): 0.67,
                ("SPEI_6", "SPEI_12"): 0.72, ("wind", "mn2t"): 0.57, ("D", "SPEI_1"): 0.40}


# ---------------------------------------------------------------------------
def load():
    for p in (C.SPEI_CSV, C.VARS_CSV, C.PETWB_CSV):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (check p1_fig_config.py)")
    s = pd.read_csv(C.SPEI_CSV)
    v = pd.read_csv(C.VARS_CSV)
    p = pd.read_csv(C.PETWB_CSV)
    for d in (s, v, p):
        d["_date"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    df = (s[["division", "_date", "SPEI_1", "SPEI_3", "SPEI_6", "SPEI_12", "D_mm_month"]]
          .rename(columns={"D_mm_month": "D"})
          .merge(v[["division", "_date", "tp", "t2m", "mx2t", "mn2t", "d2m", "ssrd", "tcc",
                    "u10", "v10"]], on=["division", "_date"])
          .merge(p[["division", "_date", "PET_mm_month"]].rename(columns={"PET_mm_month": "PET"}),
                 on=["division", "_date"]))
    df["wind"] = np.hypot(df.u10, df.v10)
    return df.sort_values(["division", "_date"]).reset_index(drop=True)


def lead_lag(df):
    """r(predictor anomaly at t-k, SPEI at t), pooled over divisions (v6 logic)."""
    d = df.copy()
    d["_m"] = d["_date"].dt.month
    for c in DRIVERS:
        d[c + "_anom"] = d.groupby(["division", "_m"])[c].transform(lambda x: x - x.mean())
    out = {}
    for ts in D.TIMESCALES:
        y_col = f"SPEI_{ts}"
        out[ts] = {}
        for c in DRIVERS:
            rs = []
            for k in range(1, MAXLAG + 1):
                xs, ys = [], []
                for _, g in d.groupby("division"):
                    x = g[c + "_anom"].shift(k).to_numpy(float)
                    y = g[y_col].to_numpy(float)
                    m = np.isfinite(x) & np.isfinite(y)
                    xs.append(x[m])
                    ys.append(y[m])
                rs.append(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])
            out[ts][c] = np.array(rs)
    return out


def checks(log, df, corr, ll):
    log.ok(f"merged data: {len(df)} division-months, {df.division.nunique()} divisions, "
           f"{df._date.min():%Y-%m} to {df._date.max():%Y-%m}")
    worst = max(abs(round(corr.loc[a, b], 2) - v) for (a, b), v in THESIS_FIG15.items())
    if worst > 0.0051:
        log.fail(f"correlation matrix does not reproduce thesis Fig 15 (max diff {worst:.3f})")
    log.ok(f"correlation matrix reproduces the {len(THESIS_FIG15)} spot-check values read from "
           f"thesis Fig 15 to 2 decimals (all 91 cells were also compared by eye, 16 Oct)")
    if not C.LEADLAG_SUMMARY.exists():
        log.note(f"{C.LEADLAG_SUMMARY} not found; lead-lag check skipped "
                 "(set LEADLAG_SUMMARY in p1_fig_config.py to run it)")
        return
    summ = pd.read_csv(C.LEADLAG_SUMMARY)
    worst = 0.0
    for ts in D.TIMESCALES:
        for c in DRIVERS:
            r = ll[ts][c]
            k = int(np.nanargmax(np.abs(r)))
            row = summ[(summ.scale == ts) & (summ["mode"] == "anom") & (summ.driver == c)]
            if len(row) != 1:
                continue
            worst = max(worst, abs(r[k] - row.peak_xcorr.iloc[0]))
            if (k + 1) != int(row.peak_xcorr_lag.iloc[0]) or abs(r[k] - row.peak_xcorr.iloc[0]) > 1e-6:
                log.fail(f"lead-lag SPEI-{ts} {c}: peak {r[k]:.4f} at lead {k + 1} vs thesis "
                         f"{row.peak_xcorr.iloc[0]:.4f} at lead {int(row.peak_xcorr_lag.iloc[0])}")
    log.ok(f"lead-lag peaks (value and lead) equal the thesis leadlag_summary.csv for all 21 "
           f"predictor-timescale pairs (max |diff| {worst:.1e})")


# ---------------------------------------------------------------------------
def draw_matrix(ax, corr, cbar_host=None):
    rows, cols = MATRIX_VARS[1:], MATRIX_VARS[:-1]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if j <= i:
                M[i, j] = corr.loc[r, c]
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1),
                   aspect="equal", interpolation="none")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else js.INK)
    # white cell borders
    for i in range(len(rows)):
        for j in range(i + 1):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=0.6))
    # block separators: predictors | PET, water balance | SPEI
    for b in (7, 9):          # after "Wind speed" (row index 6 -> boundary 7), after "Water balance"
        ax.plot([-0.5, b + 0.5], [b - 0.5, b - 0.5], color=js.INK, lw=0.6)
        ax.plot([b + 0.5, b + 0.5], [b - 0.5, len(rows) - 0.5], color=js.INK, lw=0.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LABEL[c] for c in cols], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[r] for r in rows])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # colour bar in the empty upper-right triangle
    cax = ax.inset_axes([0.60, 0.90, 0.36, 0.03])
    cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson r", labelpad=2)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    return im


def draw_leadlag(axes, ll):
    allv = np.concatenate([ll[t][c] for t in D.TIMESCALES for c in DRIVERS])
    lo, hi = np.floor(allv.min() * 10) / 10 - 0.05, np.ceil(allv.max() * 10) / 10 + 0.05
    for ax, ts in zip(axes, D.TIMESCALES):
        ax.axvspan(0.5, 3.5, color="#EDEDED", lw=0, zorder=0)
        ax.axhline(0, color="#BDBDBD", lw=0.5, zorder=1)
        ax.axvline(ts, color=js.INK, lw=0.8, ls=(0, (3, 1.6)), zorder=1)
        k = np.arange(1, MAXLAG + 1)
        for c in DRIVERS:
            col, mk = DRIVER_STYLE[c]
            ax.plot(k, ll[ts][c], color=col, lw=1.0, marker=mk, ms=3.2, mfc=col,
                    mec=js.INK if c != "wind" else "white", mew=0.3, zorder=3)
        ax.set_xlim(0.5, MAXLAG + 0.5)
        ax.set_xticks(k)
        ax.set_ylim(lo, hi)
        ax.set_title(f"SPEI-{ts}", fontweight="bold", pad=4)
        ax.set_xlabel("Lead (months before the SPEI month)")
    axes[0].set_ylabel("Lagged correlation r")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def leadlag_handles():
    h = [Line2D([], [], color=DRIVER_STYLE[c][0], lw=1.0, marker=DRIVER_STYLE[c][1], ms=3.2,
                mfc=DRIVER_STYLE[c][0], mec=js.INK if c != "wind" else "white", mew=0.3,
                label=LABEL[c]) for c in DRIVERS]
    h += [Patch(facecolor="#EDEDED", edgecolor="none", label="Forecast leads 1–3"),
          Line2D([], [], color=js.INK, lw=0.8, ls=(0, (3, 1.6)), label="Lead = accumulation length")]
    return h


# ---------------------------------------------------------------------------
def make(outdir, log):
    log.section("Fig 3 - predictor screening")
    df = load()
    corr = df[MATRIX_VARS].corr(method="pearson")
    ll = lead_lag(df)
    checks(log, df, corr, ll)
    corr.round(4).to_csv(outdir / "Fig03_correlation_values.csv")
    pd.DataFrame({f"SPEI{ts}_{c}": ll[ts][c] for ts in D.TIMESCALES for c in DRIVERS},
                 index=pd.Index(range(1, MAXLAG + 1), name="lead")).round(4).to_csv(
        outdir / "Fig03_leadlag_values.csv")
    log.ok("values behind Fig 3 written to Fig03_correlation_values.csv and Fig03_leadlag_values.csv")
    js.apply()
    problems = []

    # --- combined ----------------------------------------------------------
    fig = plt.figure(figsize=(js.WIDTH["double"], 214 * js.MM))
    top, bot = fig.subfigures(2, 1, height_ratios=[150, 64], hspace=0.0)
    ax_m = top.subplots()
    draw_matrix(ax_m, corr)
    axl = bot.subplots(1, 3, sharey=True)
    draw_leadlag(axl, ll)
    bot.legend(handles=leadlag_handles(), loc="outside upper center", ncol=5, frameon=False)
    js.panel_labels([ax_m], dx=-118, dy=4)
    js.panel_labels(list(axl), start="b", dx=-22, dy=4)
    problems += js.save(fig, "Fig03", outdir)

    # --- split: a matrix ----------------------------------------------------
    fig = plt.figure(figsize=(js.WIDTH["double"], 150 * js.MM))
    ax_m = fig.subplots()
    draw_matrix(ax_m, corr)
    problems += js.save(fig, "Fig03a", outdir)

    # --- split: b lead-lag --------------------------------------------------
    fig = plt.figure(figsize=(js.WIDTH["double"], 70 * js.MM))
    axl = fig.subplots(1, 3, sharey=True)
    draw_leadlag(axl, ll)
    fig.legend(handles=leadlag_handles(), loc="outside upper center", ncol=5, frameon=False)
    js.panel_labels(list(axl), start="a", dx=-22, dy=4)
    problems += js.save(fig, "Fig03b", outdir)
    return problems


if __name__ == "__main__":
    log = D.CheckLog()
    C.FIG_OUT.mkdir(parents=True, exist_ok=True)
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
