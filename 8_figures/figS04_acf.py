"""
Fig S4  --  ACF and PACF of SPEI (thesis Fig 17, §4.4.4)

Rows SPEI-3/6/12; columns ACF (total memory) and PACF (direct memory); lags 0-24.
Same computation as the thesis acf_pacf_test.py: for each division the series without
its leading gaps, statsmodels acf (fft) and pacf (method "ywmle"), then the mean over
the 8 divisions (bar) and their SD (whisker). Dashed lines: 95% white-noise bounds,
+/- 1.96 / sqrt(n), n = mean series length. Full record 1980-2024 (as in the thesis).

Source: divisional_spei_1980_2024.csv. Values written to FigS04_acf_pacf_values.csv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import journal_style as js
import fig_data as D
import p1_fig_config as C

MAXLAG = 24
BAR = "#6E6E6E"


def compute():
    from statsmodels.tsa.stattools import acf, pacf
    s = pd.read_csv(C.SPEI_CSV)
    s["_date"] = pd.to_datetime(s["date"]).dt.to_period("M").dt.to_timestamp()
    out = {}
    for ts in D.TIMESCALES:
        A, P, N = [], [], []
        for _, g in s.groupby("division"):
            x = g.sort_values("_date")[f"SPEI_{ts}"].dropna().to_numpy(float)
            A.append(acf(x, nlags=MAXLAG, fft=True))
            P.append(pacf(x, nlags=MAXLAG, method="ywmle"))
            N.append(len(x))
        A, P = np.vstack(A), np.vstack(P)
        out[ts] = dict(acf_mean=A.mean(0), acf_std=A.std(0), pacf_mean=P.mean(0),
                       pacf_std=P.std(0), n=int(np.mean(N)), ndiv=len(N))
    return out


def make(outdir, log):
    log.section("Fig S4 - ACF / PACF")
    res = compute()
    rows = []
    for ts, r in res.items():
        band = 1.96 / np.sqrt(r["n"])
        sig_acf = [k for k in range(1, MAXLAG + 1) if r["acf_mean"][k] > band]
        last_acf = max(k for k in range(1, MAXLAG + 1) if all(r["acf_mean"][1:k + 1] > band))
        sig_pacf = [k for k in range(1, MAXLAG + 1) if abs(r["pacf_mean"][k]) > band]
        log.ok(f"SPEI-{ts}: {r['ndiv']} divisions, mean n = {r['n']}; lag-1 ACF {r['acf_mean'][1]:.2f}, "
               f"lag-1 PACF {r['pacf_mean'][1]:.2f}; ACF above the 95% bound up to lag {last_acf}; "
               f"PACF significant at lags {sig_pacf}")
        for k in range(MAXLAG + 1):
            rows.append(dict(timescale=ts, lag=k, acf_mean=r["acf_mean"][k], acf_sd=r["acf_std"][k],
                             pacf_mean=r["pacf_mean"][k], pacf_sd=r["pacf_std"][k], bound=band))
    pd.DataFrame(rows).round(4).to_csv(outdir / "FigS04_acf_pacf_values.csv", index=False)
    log.note("ACF/PACF values written to FigS04_acf_pacf_values.csv. Besides lag 1, the mean PACF has a "
             "spike one month beyond the window (SPEI-6 lag 7 = 0.27; SPEI-12 lag 13 = 0.40), as in Section 3.2")
    js.apply()
    fig, axes = plt.subplots(3, 2, figsize=(js.WIDTH["double"], 150 * js.MM), sharex=True, sharey=True)
    lags = np.arange(MAXLAG + 1)
    for ri, ts in enumerate(D.TIMESCALES):
        r = res[ts]
        band = 1.96 / np.sqrt(r["n"])
        for ci, key in enumerate(("acf", "pacf")):
            ax = axes[ri, ci]
            ax.bar(lags, r[f"{key}_mean"], width=0.6, color=BAR, zorder=2)
            ax.errorbar(lags, r[f"{key}_mean"], yerr=r[f"{key}_std"], fmt="none", ecolor=js.INK,
                        elinewidth=0.6, zorder=3)
            ax.axhline(0, color=js.INK, lw=0.6, zorder=1)
            for b in (band, -band):
                ax.axhline(b, color=js.INK, lw=0.8, ls=(0, (3, 1.6)), zorder=1)
            ax.set_ylim(-0.4, 1.05)
            ax.set_xlim(-0.8, MAXLAG + 0.8)
            ax.set_xticks(range(0, MAXLAG + 1, 3))
            ax.yaxis.grid(True, color="#EDEDED", lw=0.5)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title("ACF (total memory)" if key == "acf" else "PACF (direct memory)",
                             fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(f"SPEI-{ts}\nCorrelation")
            if ri == 2:
                ax.set_xlabel("Lag (months)")
    js.panel_labels(list(axes.ravel()), dx=-26, dy=3)
    return js.save(fig, "FigS04", outdir)


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
