"""
ACF / PACF analysis of the four SPEI series (SPEI-1/3/6/12).

PURPOSE
  Decide the AUTOREGRESSIVE lag depth - how many months of a drought index's OWN past to feed
  the forecasting model. This is the one lag choice the lead-lag test did not cover (lead-lag
  measured climate -> future SPEI; this measures past SPEI -> future SPEI).

WHY BOTH ACF AND PACF
  ACF  (autocorrelation)          = TOTAL relationship at each lag, incl. indirect "echoes"
                                    (drought this month relates to 2 months ago partly THROUGH
                                     last month). Confirms the series is persistent.
  PACF (partial autocorrelation)  = DIRECT relationship only, echoes removed. The last lag where
                                     PACF is significant = how many lags carry INDEPENDENT info
                                     = the autoregressive lag depth to use.

METHOD / HONEST NOTES
  - Each SPEI series has LEADING NaNs by construction (SPEI-12 starts 11 months in). These are
    dropped per division before computing (ACF/PACF need gap-free segments).
  - Computed PER DIVISION, then AVERAGED (with a spread band). Divisions are NOT concatenated -
    concatenation would create false discontinuities at the seams between divisions.
  - Significance band = +/- 1.96/sqrt(n), the standard 95% white-noise bound.
  - Expected: SPEI-1 short memory (PACF cuts off early); SPEI-12 long memory (PACF significant deep).

Reads only; writes figures + a CSV of suggested lag depths.
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
SPEI_CSV = str(P.SPEI_CSV)
OUT_DIR  = str(P.WORK / "3_predictor_screening" / "acf_pacf")

MAXLAG = 24            # inspect up to 24 months of memory
SPEI_SCALES = [1, 3, 6, 12]

# ----------------------------------------------------------------------------
# Column detection (verified against the real file)
# ----------------------------------------------------------------------------
def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_division_col(df):
    for c in df.columns:
        if _norm(c) in {"division", "div", "divname", "name", "region", "zone"}:
            return c
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 20:
            return c
    raise ValueError("No division column: %s" % list(df.columns))

def build_datetime(df):
    cols = {_norm(c): c for c in df.columns}
    for k in ("date", "time", "datetime", "yearmonth", "ym"):
        if k in cols:
            return pd.to_datetime(df[cols[k]], errors="coerce")
    y = next((cols[k] for k in cols if k in ("year", "yr", "yyyy")), None)
    m = next((cols[k] for k in cols if k in ("month", "mon", "mm")), None)
    if y and m:
        return pd.to_datetime(dict(year=df[y].astype(int), month=df[m].astype(int), day=1), errors="coerce")
    for c in df.columns:
        p = pd.to_datetime(df[c], errors="coerce")
        if p.notna().mean() > 0.9:
            return p
    raise ValueError("No date column: %s" % list(df.columns))

def detect_spei_cols(df):
    out = {}
    for c in df.columns:
        n = _norm(c)
        if "class" in n or "categ" in n or "flag" in n:
            continue
        m = re.fullmatch(r"spei_?(\d{1,2})", n)
        if m and int(m.group(1)) in (1, 3, 6, 12) and df[c].dtype != object:
            out[int(m.group(1))] = c
    return out

# ----------------------------------------------------------------------------
# ACF / PACF averaged across divisions
# ----------------------------------------------------------------------------
def acf_pacf_per_division(df, div_col, date_col, ycol, maxlag):
    from statsmodels.tsa.stattools import acf, pacf
    acfs, pacfs, ns = [], [], []
    for _, g in df.groupby(div_col):
        s = g.sort_values(date_col)[ycol].dropna().to_numpy(dtype=float)  # drop leading NaNs
        if len(s) < maxlag + 20:
            continue
        try:
            a = acf(s, nlags=maxlag, fft=True)
            p = pacf(s, nlags=maxlag, method="ywmle")
            acfs.append(a); pacfs.append(p); ns.append(len(s))
        except Exception:
            continue
    if not acfs:
        return None
    A = np.vstack(acfs); P = np.vstack(pacfs)
    return dict(acf_mean=A.mean(0), acf_std=A.std(0),
                pacf_mean=P.mean(0), pacf_std=P.std(0),
                n_mean=int(np.mean(ns)), n_div=len(ns))

def significant_cutoff(pacf_mean, n, maxlag):
    """Return the last lag (>=1) where mean PACF exceeds the 95% white-noise band."""
    band = 1.96 / np.sqrt(n)
    last = 0
    for k in range(1, min(maxlag, len(pacf_mean) - 1) + 1):
        if abs(pacf_mean[k]) > band:
            last = k
    return last, band

# ----------------------------------------------------------------------------
# Plot ACF + PACF side by side for one timescale
# ----------------------------------------------------------------------------
def plot_acf_pacf(res, scale, out_png):
    lags = np.arange(0, len(res["acf_mean"]))
    band = 1.96 / np.sqrt(res["n_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "acf", "ACF (total memory)"),
                           (axes[1], "pacf", "PACF (direct memory)")]:
        mean = res[key + "_mean"]; std = res[key + "_std"]
        ax.bar(lags, mean, width=0.5, color="#1a4d2e", alpha=0.8)
        ax.errorbar(lags, mean, yerr=std, fmt="none", ecolor="gray", alpha=0.4, capsize=2)
        ax.axhline(band, color="red", ls="--", lw=0.9)
        ax.axhline(-band, color="red", ls="--", lw=0.9)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title("%s - SPEI-%d" % (title, scale), fontsize=11)
        ax.set_xlabel("Lag (months)"); ax.set_ylabel("correlation")
        ax.set_ylim(-0.4, 1.05)
    fig.tight_layout(); fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    try:
        import statsmodels  # noqa
    except Exception:
        sys.exit("Needs statsmodels. Run: pip install statsmodels")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SPEI_CSV):
        sys.exit("MISSING SPEI file: %s" % SPEI_CSV)

    df = pd.read_csv(SPEI_CSV)
    div_col = find_division_col(df)
    date_col = "_date"; df[date_col] = build_datetime(df)
    df[div_col] = df[div_col].astype(str).str.strip()
    spei_cols = detect_spei_cols(df)
    scales = [s for s in SPEI_SCALES if s in spei_cols]
    print("SPEI targets:", {s: spei_cols[s] for s in scales})
    print("Divisions:", df[div_col].nunique())

    rows = []
    for scale in scales:
        ycol = spei_cols[scale]
        res = acf_pacf_per_division(df, div_col, date_col, ycol, MAXLAG)
        if res is None:
            print("SPEI-%d: insufficient data, skipped." % scale); continue
        cutoff, band = significant_cutoff(res["pacf_mean"], res["n_mean"], MAXLAG)
        plot_acf_pacf(res, scale, os.path.join(OUT_DIR, "acf_pacf_SPEI%d.png" % scale))
        # first few PACF values for the console
        pacf_head = ", ".join("lag%d=%.2f" % (k, res["pacf_mean"][k]) for k in range(1, 7))
        print("\n=== SPEI-%d (n~%d/division, %d divisions) ===" % (scale, res["n_mean"], res["n_div"]))
        print("  PACF: %s" % pacf_head)
        print("  95%% band = +/-%.3f" % band)
        print("  -> PACF significant through lag %d  => suggested autoregressive depth: %d" % (cutoff, cutoff))
        rows.append(dict(scale=scale, n_per_division=res["n_mean"], n_divisions=res["n_div"],
                         pacf_band=round(band, 3), suggested_lag_depth=cutoff,
                         acf_lag1=round(res["acf_mean"][1], 3)))

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "acf_pacf_lag_depths.csv"), index=False)
    print("\nSaved lag depths:", os.path.join(OUT_DIR, "acf_pacf_lag_depths.csv"))
    print("Figures in:", OUT_DIR)
    print("\nREAD: the 'suggested_lag_depth' column is how many lags of each SPEI series to feed the model.")
    print("      Expect SPEI-1 short, SPEI-12 long. ACF near 1 at lag 1 confirms persistence.")

if __name__ == "__main__":
    main()
