"""
Fig S1  --  Quality screening of the BMD station data against ERA5 (SI)
Fig S2  --  SPEI-3 from BMD station vs ERA5 precipitation, Dhaka and Barisal
            (thesis Fig 10, §4.2.4)

All values come from the thesis Tier-1 calculations (tier1_logic.py, copied unchanged),
1980-2022, 35 BMD stations mapped to the 8 divisions.
S1  a  mean monthly precipitation: 35-station mean (BMD) vs 8-division mean (ERA5)
    b  monthly precipitation, ERA5 vs BMD, all division-months (1:1 line, LS fit)
    c  mean daily maximum and d minimum temperature climatologies; the BMD file's
       temperatures have little or no seasonal cycle (station phase errors), which is
       why only precipitation was used for validation (Table S4).
S2  SPEI-3 computed from BMD and from ERA5 precipitation (same ERA5 Thornthwaite PET
    for both), 2005-2022 window, full-period r in each panel.
Colours: BMD black, ERA5 sky blue (#56B4E9); neither is a model colour.

Checks: the recomputed precipitation table, month-shift table, SPEI agreement and SPEI
series equal the thesis Tier-1 outputs (table1, table5, table3, spei_series_bmd_vs_era5).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import journal_style as js
import fig_data as D
import p1_fig_config as C
import tier1_logic as T

C_BMD, C_ERA5 = "#000000", "#56B4E9"
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def checks(log, res):
    t1 = C.VALIDATION_DIR
    pairs = [("table1_precipitation_validation.csv", res["precip"], "division",
              ["n", "r", "r_anom", "pbias", "RMSE", "NSE", "KGE"]),
             ("table5_per_station_month_shifts.csv", res["shift_tab"], "Station",
              ["TMAX_shift", "TMIN_shift", "P_shift"]),
             ("table3_spei_agreement.csv", res["agree"], "scale", ["n", "r", "NSE", "KGE"])]
    for fname, df, key, cols in pairs:
        p = t1 / fname
        if not p.exists():
            log.note(f"{fname} not found; check skipped")
            continue
        ref = pd.read_csv(p).set_index(key)
        new = df.set_index(key)
        worst = max(float(np.nanmax(np.abs(new.loc[ref.index, c].astype(float).values
                                            - ref[c].astype(float).values))) for c in cols)
        if worst > 1e-6:
            log.fail(f"{fname}: recomputed values differ (max {worst})")
        log.ok(f"{fname}: recomputed values equal the Validation_1.py output (max |diff| {worst:.1e})")
    p = t1 / "spei_series_bmd_vs_era5.csv"
    if p.exists():
        ref = pd.read_csv(p)
        m = ref.merge(res["series"], on=["division", "year", "month"], suffixes=("_ref", ""))
        c = [f"SPEI{T.TS_SCALE}_b", f"SPEI{T.TS_SCALE}_e"]
        worst = np.nanmax(np.abs(m[[x for x in c]].values - m[[x + "_ref" for x in c]].values))
        if worst > 1e-6:
            log.fail(f"SPEI series differ from spei_series_bmd_vs_era5.csv (max {worst})")
        log.ok(f"SPEI-3 series equal spei_series_bmd_vs_era5.csv (max |diff| {worst:.1e})")


def make_s1(outdir, log, res):
    clim, mg = res["clim"], res["mg"]
    js.apply()
    fig, axes = plt.subplots(2, 2, figsize=(js.WIDTH["double"], 140 * js.MM))
    a, b, c, d = axes.ravel()
    x = clim.month.values
    # a: precipitation climatology
    a.plot(x, clim.P_bmd, "-o", color=C_BMD, lw=1.2, ms=3.5, zorder=3)
    a.plot(x, clim.P_era, "--s", color=C_ERA5, lw=1.2, ms=3.5, mec=js.INK, mew=0.4, zorder=4)
    a.set_ylabel("Precipitation (mm per month)")
    a.set_ylim(0, max(clim.P_bmd.max(), clim.P_era.max()) * 1.15)
    # b: scatter
    r = np.corrcoef(mg.P_b, mg.P_e)[0, 1]
    hi = np.ceil(max(mg.P_b.max(), mg.P_e.max()) / 100) * 100
    b.scatter(mg.P_b, mg.P_e, s=2.5, color=C_BMD, alpha=0.3, lw=0, zorder=2)
    b.plot([0, hi], [0, hi], color=js.INK, lw=0.8, ls=(0, (3, 1.6)), zorder=3)
    sl, ic = np.polyfit(mg.P_b, mg.P_e, 1)
    b.plot([0, hi], [ic, ic + sl * hi], color=C_ERA5, lw=1.4, zorder=4)
    b.set_xlim(0, hi)
    b.set_ylim(0, hi)
    b.set_aspect("equal", adjustable="box")
    b.set_xlabel("BMD monthly precipitation (mm)")
    b.set_ylabel("ERA5 monthly precipitation (mm)")
    b.text(0.04, 0.96, f"r = {r:.3f}\nn = {len(mg):,} division-months", transform=b.transAxes,
           ha="left", va="top")
    b.legend(handles=[Line2D([], [], color=js.INK, lw=0.8, ls=(0, (3, 1.6)), label="1:1 line"),
                      Line2D([], [], color=C_ERA5, lw=1.4, label="Least-squares fit")],
             loc="lower right")
    # c, d: temperature climatologies
    rng = {}
    for ax, v, lab in ((c, "TMAX", "Mean daily maximum temperature (°C)"),
                       (d, "TMIN", "Mean daily minimum temperature (°C)")):
        bv, ev = clim[f"{v}_bmd"], clim[f"{v}_era"]
        rng[v] = (bv.max() - bv.min(), ev.max() - ev.min())
        ax.plot(x, bv, "-o", color=C_BMD, lw=1.2, ms=3.5, zorder=3,
                label=f"BMD (annual range {rng[v][0]:.1f} °C)")
        ax.plot(x, ev, "--s", color=C_ERA5, lw=1.2, ms=3.5, mec=js.INK, mew=0.4, zorder=4,
                label=f"ERA5 (annual range {rng[v][1]:.1f} °C)")
        ax.set_ylabel(lab)
        lo, top = min(bv.min(), ev.min()), max(bv.max(), ev.max())
        ax.set_ylim(lo - 0.45 * (top - lo), top + 0.1 * (top - lo))
        ax.legend(loc="lower center")
    for ax in (a, c, d):
        ax.set_xticks(x)
        ax.set_xticklabels(MONTHS)
        ax.set_xlabel("Month")
        ax.yaxis.grid(True, color="#E8E8E8", lw=0.5)
        ax.set_axisbelow(True)
    js.panel_labels([a, b, c, d], dx=-26, dy=4)
    h = [Line2D([], [], color=C_BMD, lw=1.2, marker="o", ms=3.5, label="BMD stations (35-station mean)"),
         Line2D([], [], color=C_ERA5, lw=1.2, ls="--", marker="s", ms=3.5, mec=js.INK, mew=0.4,
                label="ERA5 (8-division mean)")]
    fig.legend(handles=h, loc="outside upper center", ncol=2, frameon=False)
    log.note(f"S1 legend numbers (computed from the data): annual range Tmax BMD {rng['TMAX'][0]:.1f} / "
             f"ERA5 {rng['TMAX'][1]:.1f} °C; Tmin BMD {rng['TMIN'][0]:.1f} / ERA5 {rng['TMIN'][1]:.1f} °C; "
             f"pooled precipitation r = {r:.3f} (= Table 1 / S3 'ALL'), n = {len(mg)}")
    return js.save(fig, "FigS01", outdir)


def make_s2(outdir, log, res):
    ser = res["series"]
    y0, y1 = T.TS_WINDOW
    js.apply()
    fig, axes = plt.subplots(2, 1, figsize=(js.WIDTH["double"], 105 * js.MM), sharex=True)
    for ax, dv in zip(axes, T.TS_DIVISIONS):
        full = ser[ser.division == dv]
        bcol, ecol = f"SPEI{T.TS_SCALE}_b", f"SPEI{T.TS_SCALE}_e"
        k = full[[bcol, ecol]].dropna()
        r = np.corrcoef(k[bcol], k[ecol])[0, 1]
        s = full[(full.year >= y0) & (full.year < y1)].sort_values(["year", "month"])
        t = s.year + (s.month - 1) / 12
        ax.axhspan(-3.9, T.DROUGHT_THRESHOLD, color="#F7E1E1", lw=0, zorder=0)
        ax.axhline(0, color="#C8C8C8", lw=0.5, zorder=1)
        ax.axhline(T.DROUGHT_THRESHOLD, color=js.GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.plot(t, s[bcol], color=C_BMD, lw=1.1, zorder=3)
        ax.plot(t, s[ecol], color=C_ERA5, lw=1.1, ls=(0, (3, 1.2)), zorder=4)
        ax.set_xlim(y0, s.year.max() + 1)
        ax.set_ylim(-3.6, 3.2)
        ax.set_ylabel("SPEI-3")
        ax.text(0.01, 0.95, f"{dv} Division · full-period r = {r:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        log.ok(f"S2 {dv}: full-period r(BMD SPEI-3, ERA5 SPEI-3) = {r:.3f} (n = {len(k)})")
    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(range(y0, int(s.year.max()) + 2, 2))
    js.panel_labels(list(axes), dx=-26, dy=2)
    h = [Line2D([], [], color=C_BMD, lw=1.1, label="SPEI-3 from BMD station precipitation"),
         Line2D([], [], color=C_ERA5, lw=1.1, ls=(0, (3, 1.2)), label="SPEI-3 from ERA5 precipitation"),
         Line2D([], [], color=js.GREY, lw=0.6, ls=(0, (1, 1.5)), label="Drought threshold (−1)")]
    fig.legend(handles=h, loc="outside upper center", ncol=3, frameon=False)
    return js.save(fig, "FigS02", outdir)


def make(outdir, log):
    log.section("Figs S1-S2 - ERA5 vs BMD")
    for f in (T.BMD_CSV, T.ERA5_CSV):
        log.ok(f"input: {T.find_source(f)}")
    res = T.run_analysis()
    checks(log, res)
    return make_s1(outdir, log, res) + make_s2(outdir, log, res)


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
